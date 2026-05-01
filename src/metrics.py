"""Metrics and parsing utilities. All numerics live here, never in the LLM."""
from __future__ import annotations

import json
import logging
import re
from typing import Sequence

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_METHOD_RE = re.compile(r"^\s*Method\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_RETRY_INSTRUCTION = "Reply with ONLY a Python list of {n} numbers."


class ParseError(Exception):
    pass


def mae(pred: Sequence[float], true: Sequence[float]) -> float:
    if len(pred) != len(true):
        raise ValueError(f"length mismatch: {len(pred)} vs {len(true)}")
    return sum(abs(p - t) for p, t in zip(pred, true)) / len(true)


def mse(pred: Sequence[float], true: Sequence[float]) -> float:
    if len(pred) != len(true):
        raise ValueError(f"length mismatch: {len(pred)} vs {len(true)}")
    return sum((p - t) ** 2 for p, t in zip(pred, true)) / len(true)


def parse_prediction(raw: str, expected_len: int) -> list[float]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, list) and len(parsed) >= expected_len:
                return [float(x) for x in parsed[:expected_len]]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    nums = _NUM_RE.findall(raw)
    if len(nums) < expected_len:
        raise ParseError(
            f"found {len(nums)} numbers, expected {expected_len}: {raw!r}"
        )
    return [float(x) for x in nums[:expected_len]]


def parse_method(raw: str) -> str:
    m = _METHOD_RE.search(raw)
    if not m:
        return "unspecified"
    name = m.group(1).strip()
    return name.split("\n", 1)[0][:80] or "unspecified"


def safe_predict(llm, prompt: str, expected_len: int) -> tuple[list[float], str]:
    """Call LLM and parse a prediction; one re-prompt on parse failure.

    Returns (prediction, raw_response). Caller can also parse_method(raw).
    """
    raw = llm.complete(prompt)
    try:
        return parse_prediction(raw, expected_len), raw
    except ParseError as e:
        log.warning("parse failed, retrying once: %s", e)
        retry_prompt = (
            prompt
            + "\n\n"
            + _RETRY_INSTRUCTION.format(n=expected_len)
        )
        raw = llm.complete(retry_prompt)
        return parse_prediction(raw, expected_len), raw
