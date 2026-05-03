import re
import json
import random

from .llm import chat
from .metrics import mae
from .prompts import (
    SYSTEM, PQUES, PVALIDATE, PCRITIQUE,
    p_input_timestamped, p_feed, p_refine, render_demo_chain,
)


def _parse_24(text):
    """Extract the first sequence of exactly 24 comma-separated floats from text."""
    # Try every line, return first that yields 24 numbers
    for line in text.splitlines():
        # Strip common prefixes like "Refined prediction:" or "Initial prediction:"
        clean = re.sub(r'^[^0-9\-\.]*', '', line).strip()
        if not clean:
            continue
        parts = [p.strip() for p in clean.split(',')]
        try:
            nums = [float(p) for p in parts if p]
            if len(nums) == 24:
                return nums
        except ValueError:
            continue
    return None


def _initial_prediction(messages):
    """Step 1 — Eqn (3): predict from p_exam + p_input + p_ques."""
    response = chat(messages)
    nums = _parse_24(response)
    return response, nums


# ── Phase A: build_demos ──────────────────────────────────────────────────────

def build_demos(train_rows, k=2, seed=42, max_iter=3, conv_threshold=0.001):
    """
    Run Algorithm 1 iteratively on k training samples (GT available).
    Returns list of rendered chain strings ready for p_exam.
    """
    rng = random.Random(seed)
    chosen = rng.sample(train_rows, k)
    demos = []

    for row in chosen:
        x_times = row["x_times"]
        x_values = row["x_values"]
        y_times = row["y_times"]
        y_values = row["y_values"]
        train_idx = row["idx"]

        print(f"[Phase A] Building demo for train idx {train_idx} ...")

        # Step 1: initial prediction (no demos in p_exam for building the demo itself)
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
            # retry with stricter instruction
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

        # Iterative loop (Algorithm 1 with real GT)
        for i in range(max_iter):
            # Step 2: feedback (Q1–Q4, real GT available)
            messages.append({"role": "user", "content": p_feed(
                x_times, x_values, y_times, y_values, y_hat, i
            )})
            fb_response = chat(messages)
            messages.append({"role": "assistant", "content": fb_response})

            # Step 3: validate
            messages.append({"role": "user", "content": PVALIDATE})
            val_response = chat(messages)
            messages.append({"role": "assistant", "content": val_response})

            # Step 4: critique → corrected feedback
            messages.append({"role": "user", "content": PCRITIQUE})
            critique_response = chat(messages)
            messages.append({"role": "assistant", "content": critique_response})
            last_feedback = critique_response

            # Step 5: refine (Eqn 5 — full history already in messages)
            messages.append({"role": "user", "content": p_refine(i)})
            refine_response = chat(messages)
            messages.append({"role": "assistant", "content": refine_response})

            new_hat = _parse_24(refine_response)
            if new_hat is None:
                print(f"  [warn] iter {i}: could not parse refined prediction, keeping previous.")
                break

            new_mae = mae(y_values, new_hat)
            print(f"  iter {i}: MAE {prev_mae:.4f} → {new_mae:.4f}")

            if abs(new_mae - prev_mae) < conv_threshold:
                print(f"  converged at iter {i}.")
                y_hat = new_hat
                break

            y_hat = new_hat
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

    # Try to extract "Refined prediction:" line first (preferred)
    for line in response.splitlines():
        if line.lower().startswith("refined prediction"):
            nums = _parse_24(line)
            if nums:
                y_hat = nums
                break

    # Fallback: scan all lines for first valid 24-number sequence
    if y_hat is None:
        y_hat = _parse_24(response)

    # Last resort: retry with stricter prompt
    if y_hat is None:
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            "The 'Refined prediction:' line did not contain exactly 24 numbers. "
            "Output ONLY the refined prediction: 24 comma-separated numbers, nothing else."})
        retry = chat(messages)
        y_hat = _parse_24(retry)
        response = response + "\n[retry]\n" + retry

    return y_hat, response
