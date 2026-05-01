"""Prompt builders for TrafficLLM Algorithm 1."""
from __future__ import annotations

from typing import Any, Sequence

REFINE_INSTRUCTION = (
    "Using the input window, your prior predictions, the feedback, and the "
    "Python-computed MAE values above, produce an improved 6-step prediction. "
    "Reply EXACTLY in this format and nothing else:\n"
    "Method: <one short phrase naming your forecasting method>\n"
    "Prediction: [v1, v2, v3, v4, v5, v6]"
)

VALIDATION_REVIEW = (
    "Please review the previous answers and find potential mistakes "
    "(in the sine/cosine projections, periodicity alignment, or method critique)."
)
VALIDATION_CORRECT = (
    "Please correct the answers based on the identified mistakes."
)


def _fmt_series(times: Sequence[str], values: Sequence[float]) -> str:
    return "\n".join(f"  {t}: {v:.2f}" for t, v in zip(times, values))


def build_exam_prompt(example: dict[str, Any]) -> str:
    return (
        "EXAMPLE (for format reference only):\n"
        f"Input window (24 hourly values):\n{_fmt_series(example['x_times'], example['x_values'])}\n"
        f"Correct next-6-hour values:\n{_fmt_series(example['y_times'], example['y_values'])}\n"
        f"Correct prediction list: {[round(v, 2) for v in example['y_values']]}\n"
    )


def build_input_prompt(sample: dict[str, Any]) -> str:
    return (
        "INPUT WINDOW (24 hourly internet-traffic values to forecast from):\n"
        f"{_fmt_series(sample['x_times'], sample['x_values'])}\n"
    )


def build_question_prompt(sample: dict[str, Any]) -> str:
    horizon_times = sample["y_times"]
    return (
        f"QUESTION: Predict the next {len(horizon_times)} hourly values "
        f"(for {horizon_times[0]} through {horizon_times[-1]}).\n"
        "Reply with ONLY a Python list of 6 numbers, e.g. [12.3, 14.1, ...].\n"
    )


def build_qualitative_prompt(
    sample: dict[str, Any], prediction: Sequence[float], method_name: str
) -> str:
    gt = [round(v, 2) for v in sample["x_values"]]
    pred = [round(v, 2) for v in prediction]
    return (
        "QUALITATIVE REVIEW:\n"
        f"Ground truth input window:\n{_fmt_series(sample['x_times'], sample['x_values'])}\n\n"
        f"Your previous prediction (method: {method_name}):\n{pred}\n\n"
        "Answer the following two questions:\n"
        "1) Periodicity: For the ground truth values and the predictions, what are "
        "their projected functions derived from the combination of sine and cosine functions? "
        f"Ground truth values: {gt}. Prediction values: {pred}. "
        "Do the predictions align with the periodicity of the ground truth? "
        "If not, describe where they deviate.\n"
        "2) Method critique: what is the main weakness of the current method "
        f"('{method_name}') given this input pattern?\n"
    )


def build_feedback_prompt(
    sample: dict[str, Any],
    prediction: Sequence[float],
    mae_value: float,
    method_name: str,
    qualitative_answers: str,
) -> str:
    return (
        "FEEDBACK:\n"
        f"- Overall performance: MAE = {mae_value:.4f} (computed in Python).\n"
        f"- Current method: {method_name}.\n"
        f"- Format check: prediction has {len(prediction)} values "
        f"(expected {len(sample['y_times'])}).\n"
        "- Periodicity & method critique (validated qualitative review):\n"
        f"{qualitative_answers.strip()}\n"
    )


def build_refine_prompt(sample: dict[str, Any], history: list[dict[str, Any]]) -> str:
    parts: list[str] = [build_input_prompt(sample), build_question_prompt(sample)]
    for h in history:
        parts.append(
            f"\n--- ITERATION {h['iter']} ---\n"
            f"Prior prediction (method: {h['method']}): "
            f"{[round(v, 2) for v in h['prediction']]}\n"
            f"{h['feedback']}"
        )
    parts.append("\n" + REFINE_INSTRUCTION)
    return "\n".join(parts)
