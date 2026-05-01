"""Milan traffic dataset loading and sliding-window construction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

W: int = 24
L: int = 6
SPLIT_RATIO: float = 0.8
EXAM_TRAIN_IDX: int = 200
DEFAULT_CSV: Path = Path("data/milan_traffic.csv")


def load_milan(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("loaded %d rows from %s", len(df), path)
    return df


def build_samples(df: pd.DataFrame, w: int = W, l: int = L) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    times = df["timestamp"].astype(str).tolist()
    values = df["internet_value"].astype(float).tolist()
    n = len(df)
    for i in range(n - w - l + 1):
        samples.append({
            "idx": i,
            "x_times": times[i : i + w],
            "x_values": values[i : i + w],
            "y_times": times[i + w : i + w + l],
            "y_values": values[i + w : i + w + l],
        })
    log.info("built %d sliding windows (w=%d, l=%d)", len(samples), w, l)
    return samples


def chronological_split(
    samples: list[dict[str, Any]], ratio: float = SPLIT_RATIO
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cut = int(len(samples) * ratio)
    train, test = samples[:cut], samples[cut:]
    log.info("split: train=%d test=%d", len(train), len(test))
    return train, test
