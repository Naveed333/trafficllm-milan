import re
import json

from .llm import chat
from .metrics import mae, fit_sincos
from .prompts import (
    SYSTEM, PQUES, PVALIDATE, PCRITIQUE,
    p_input_timestamped, p_feed, p_refine, render_demo_chain,
)


def _trim_feedback(text, max_chars=600):
    """Truncate LLM critique to a safe length to avoid bloated demo chains."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars // 2:
        truncated = truncated[:last_newline]
    return truncated + "\n[truncated for conciseness]"


def _parse_24(text, prefer_last=False):
    """Extract a sequence of exactly 24 comma-separated floats from text.

    prefer_last=True returns the last match (for Phase B where the refined
    prediction appears at the end of any chain the LLM follows from demos).
    prefer_last=False returns the first match (for Phase A initial prediction).
    """
    result = None
    for line in text.splitlines():
        clean = re.sub(r'^[^0-9\-\.]*', '', line).strip()
        if not clean:
            continue
        parts = [p.strip() for p in clean.split(',')]
        try:
            nums = [float(p) for p in parts if p]
            if len(nums) == 24:
                if not prefer_last:
                    return nums
                result = nums  # keep scanning for a later match
        except ValueError:
            continue
    return result


def _initial_prediction(messages):
    """Step 1 — Eqn (3): predict from p_exam + p_input + p_ques."""
    response = chat(messages)
    nums = _parse_24(response)
    return response, nums


# ── Phase A: build_demos ──────────────────────────────────────────────────────

def build_demos(train_rows, k=2, max_iter=3, conv_threshold=0.001):
    """
    Run Algorithm 1 iteratively on k training samples (GT available).
    Returns list of rendered chain strings ready for p_exam.
    """
    chosen = train_rows[:k]
    demos = []

    for row in chosen:
        x_times = row["x_times"]
        x_values = row["x_values"]
        y_values = row["y_values"]
        train_idx = row["idx"]

        print(f"[Phase A] Building demo for train idx {train_idx} ...")

        # Step 1: initial prediction
        pinput = p_input_timestamped(x_times, x_values)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                pinput + "\n\n"
                "Predict the next 24 hourly loads. "
                "Output exactly 24 comma-separated non-negative numbers on a single line."
            )},
        ]
        response, y_hat = _initial_prediction(messages)
        if y_hat is None:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                "Your output did not contain exactly 24 numbers. "
                "Output ONLY 24 comma-separated numbers, nothing else."})
            response, y_hat = _initial_prediction(messages)
        if y_hat is None:
            print(f"  [warn] Could not parse initial prediction for idx {train_idx}, skipping.")
            continue

        messages.append({"role": "assistant", "content": response})
        y_hat_initial = list(y_hat)
        prev_mae = mae(y_values, y_hat)
        last_feedback = ""
        print(f"  initial MAE: {prev_mae:.4f}")

        # Iterative refinement loop (Algorithm 1) — capped at max_iter
        for i in range(max_iter):
            print(f"  ==> iter {i+1}/{max_iter}: requesting feedback ...")

            # Step 2: feedback (Q1–Q4) — GT not exposed raw to prevent copying
            abs_errors = [abs(y_hat[j] - y_values[j]) for j in range(24)]
            current_mae = mae(y_values, y_hat)
            gt_sin_cos = fit_sincos(y_values)
            messages.append({"role": "user", "content": p_feed(
                x_times, x_values, abs_errors, current_mae, gt_sin_cos, y_hat, i
            )})
            fb_response = chat(messages, max_tokens=1024)
            messages.append({"role": "assistant", "content": fb_response})

            # Step 3: validate
            print(f"  ==> iter {i+1}/{max_iter}: validating ...")
            messages.append({"role": "user", "content": PVALIDATE})
            val_response = chat(messages, max_tokens=600)
            messages.append({"role": "assistant", "content": val_response})

            # Step 4: critique → corrected feedback
            print(f"  ==> iter {i+1}/{max_iter}: critiquing ...")
            messages.append({"role": "user", "content": PCRITIQUE})
            critique_response = chat(messages, max_tokens=500)
            messages.append({"role": "assistant", "content": critique_response})
            last_feedback = critique_response

            # Step 5: refine (Eqn 5 — full history already in messages)
            print(f"  ==> iter {i+1}/{max_iter}: refining prediction ...")
            messages.append({"role": "user", "content": p_refine(i)})
            refine_response = chat(messages, max_tokens=150)
            messages.append({"role": "assistant", "content": refine_response})

            new_hat = _parse_24(refine_response)
            if new_hat is None:
                print(f"  [warn] iter {i+1}: could not parse refined prediction, stopping.")
                break

            new_mae = mae(y_values, new_hat)
            print(f"  iter {i+1}: MAE {prev_mae:.4f} → {new_mae:.4f}")

            y_hat = new_hat

            if new_mae < conv_threshold or abs(new_mae - prev_mae) < conv_threshold:
                print(f"  converged (MAE={new_mae:.4f}).")
                break

            prev_mae = new_mae

        chain = render_demo_chain(train_idx, x_times, x_values,
                                  y_hat_initial, last_feedback, y_hat)
        demos.append({
            "train_idx": train_idx,
            "y_hat_initial": y_hat_initial,
            "y_hat_refined": y_hat,
            "rendered_chain": chain,
        })
        print(f"  Demo built. Final MAE vs GT: {mae(y_values, y_hat):.4f}")

    return demos


# ── Phase B: run_test_sample ──────────────────────────────────────────────────

def run_test_sample(demo_chains, x_times, x_values):
    """
    Single LLM call per test sample — Eqn (3) only.
    demo_chains: list of rendered_chain strings from build_demos.
    Returns (y_hat_final, raw_response).
    """
    p_exam = "\n\n".join(demo_chains)
    pinput = p_input_timestamped(x_times, x_values)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": p_exam + "\n\n" + pinput + "\n\n" + PQUES},
    ]

    response = chat(messages)
    y_hat = None

    # If LLM followed the demo chain format, "Refined prediction:" is the last line
    for line in response.splitlines():
        if line.lower().startswith("refined prediction"):
            nums = _parse_24(line)
            if nums:
                y_hat = nums
                break

    # Fallback: take the LAST valid 24-number line (refined = last in any chain)
    if y_hat is None:
        y_hat = _parse_24(response, prefer_last=True)

    # Last resort: retry asking for just the numbers
    if y_hat is None:
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            "Output ONLY your final prediction: exactly 24 comma-separated "
            "non-negative numbers on a single line, nothing else."})
        retry = chat(messages)
        y_hat = _parse_24(retry, prefer_last=True)
        response = response + "\n[retry]\n" + retry

    return y_hat, response
