"""TrafficLLM Algorithm 1 — initial prediction → iterative feedback → refinement."""
from __future__ import annotations

import logging
import re
from typing import Any

from .metrics import mae, parse_method, safe_predict
from .prompts import (
    VALIDATION_CORRECT,
    VALIDATION_REVIEW,
    build_exam_prompt,
    build_feedback_prompt,
    build_input_prompt,
    build_qualitative_prompt,
    build_question_prompt,
    build_refine_prompt,
)

log = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def predict_initial(sample: dict[str, Any], llm, example_sample: dict[str, Any]) -> list[float]:
    prompt = (
        build_exam_prompt(example_sample)
        + build_input_prompt(sample)
        + build_question_prompt(sample)
    )
    pred, _raw = safe_predict(llm, prompt, expected_len=len(sample["y_times"]))
    return pred


def predict_refine(
    sample: dict[str, Any], history: list[dict[str, Any]], llm
) -> tuple[list[float], str]:
    prompt = build_refine_prompt(sample, history)
    pred, raw = safe_predict(llm, prompt, expected_len=len(sample["y_times"]))
    method = parse_method(raw)
    return pred, method


def run_validation_loop(qualitative_prompt: str, llm, max_inner: int = 3) -> str:
    """Three-round review/correct on the qualitative answer.

    Returns the final qualitative-answer string. Stops early if the corrected
    answer matches the previous round (whitespace-normalized).
    """
    answer = llm.complete(qualitative_prompt)
    for _ in range(max_inner):
        review = llm.complete(
            qualitative_prompt + "\n\nPrevious answer:\n" + answer + "\n\n" + VALIDATION_REVIEW
        )
        corrected = llm.complete(
            qualitative_prompt
            + "\n\nPrevious answer:\n"
            + answer
            + "\n\nIdentified mistakes:\n"
            + review
            + "\n\n"
            + VALIDATION_CORRECT
        )
        if _normalize(corrected) == _normalize(answer):
            return corrected
        answer = corrected
    return answer


def trafficllm(
    sample: dict[str, Any],
    llm,
    example_sample: dict[str, Any],
    i_max: int = 5,
    patience: int = 2,
    eps: float = 0.001,
) -> tuple[list[float], list[dict[str, Any]]]:
    y_true = sample["y_values"]
    y_hat = predict_initial(sample, llm, example_sample)
    prev_mae = mae(y_hat, y_true)
    history: list[dict[str, Any]] = []
    method_name = "naive baseline"
    small_steps = 0

    for i in range(i_max):
        qualitative_prompt = build_qualitative_prompt(sample, y_hat, method_name)
        qualitative_answers = run_validation_loop(qualitative_prompt, llm)

        feedback = build_feedback_prompt(
            sample, y_hat, prev_mae, method_name, qualitative_answers
        )

        history.append({
            "iter": i,
            "prediction": y_hat,
            "mae": prev_mae,
            "feedback": feedback,
            "method": method_name,
        })

        y_hat_new, new_method = predict_refine(sample, history, llm)
        new_mae = mae(y_hat_new, y_true)

        # Patience-based convergence: two consecutive sub-eps deltas.
        if abs(new_mae - prev_mae) < eps:
            small_steps += 1
            log.info("iter %d: |Δmae|=%.5f < eps (small_steps=%d)",
                     i, abs(new_mae - prev_mae), small_steps)
            if small_steps >= patience:
                y_hat, prev_mae, method_name = y_hat_new, new_mae, new_method
                break
        else:
            small_steps = 0

        y_hat, prev_mae, method_name = y_hat_new, new_mae, new_method

    return y_hat, history
