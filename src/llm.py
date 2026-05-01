"""LLM client wrappers — real OpenAI client and a mock for smoke tests."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI  # local import so mock-only runs don't need it

        self.model = model or DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class MockLLMClient:
    """Returns canned responses sufficient to exercise the full pipeline."""

    model = "mock"
    _NUM_RE = re.compile(r"-?\d+\.?\d*")

    def __init__(self, l: int = 6) -> None:
        self.l = l
        self._calls = 0

    def complete(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        from .prompts import VALIDATION_CORRECT, VALIDATION_REVIEW

        self._calls += 1
        # If the prompt asks for the structured refinement reply, produce it.
        if "Reply EXACTLY in this format" in prompt:
            pred = self._predict_from_prompt(prompt, perturb=True)
            return f"Method: moving_average_v{self._calls}\nPrediction: {pred}"
        # Validation review/correct rounds — short text response.
        if VALIDATION_REVIEW in prompt or VALIDATION_CORRECT in prompt:
            return "The previous answer is reasonable; minor refinements applied."
        # Initial qualitative-review request.
        if "QUALITATIVE REVIEW" in prompt:
            return (
                "Periodicity: 24-hour daily cycle visible. "
                "Method critique: baseline ignores diurnal pattern."
            )
        # Otherwise treat as initial-prediction request → flat repeat.
        return str(self._predict_from_prompt(prompt, perturb=False))

    def _predict_from_prompt(self, prompt: str, perturb: bool) -> list[float]:
        nums = [float(x) for x in self._NUM_RE.findall(prompt)]
        # Heuristic: skip the leading idx-like ints; take last 24 as the window.
        window = nums[-24:] if len(nums) >= 24 else (nums or [0.0])
        last = window[-1]
        # On perturb runs, nudge toward the mean of the window (gives the mock
        # something to "improve" toward over iterations).
        if perturb:
            mean = sum(window) / len(window)
            base = [last + (mean - last) * 0.5 for _ in range(self.l)]
        else:
            base = [last for _ in range(self.l)]
        return [round(x, 2) for x in base]


def make_client(mock: bool = False, **kwargs: Any) -> LLMClient | MockLLMClient:
    return MockLLMClient(**kwargs) if mock else LLMClient(**kwargs)
