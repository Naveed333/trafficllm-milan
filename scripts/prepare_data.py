"""Build sliding windows from data/milan_traffic.csv and write train/test JSONL."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_samples, chronological_split, load_milan  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("prepare_data")


def write_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    log.info("wrote %d rows to %s", len(samples), path)


def main() -> None:
    csv_path = ROOT / "data" / "milan_traffic.csv"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_milan(csv_path)
    samples = build_samples(df)
    train, test = chronological_split(samples)

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    log.info("done. train=%d test=%d", len(train), len(test))


if __name__ == "__main__":
    main()
