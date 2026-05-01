"""Run TrafficLLM Algorithm 1 over a subset of the test set."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import EXAM_TRAIN_IDX  # noqa: E402
from src.llm import DEFAULT_MODEL, make_client  # noqa: E402
from src.metrics import mae, mse  # noqa: E402
from src.trafficllm import trafficllm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("run_experiment")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--i_max", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(out_dir / "train.jsonl")
    test = load_jsonl(out_dir / "test.jsonl")
    example = train[EXAM_TRAIN_IDX]

    if args.n_samples == -1 or args.n_samples >= len(test):
        chosen = test
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(test, args.n_samples)

    llm = make_client(mock=args.mock) if args.mock else make_client(mock=False)
    if not args.mock:
        llm.model = args.model
        log.info("using OpenAI model %s", llm.model)
    else:
        log.info("using MockLLMClient")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"{slug(args.model if not args.mock else 'mock')}_{ts}.jsonl"
    log_path = out_dir / log_name
    log.info("writing iteration log to %s", log_path)

    summary: list[tuple[int, float, float, int]] = []
    with log_path.open("w") as fout:
        for s in chosen:
            try:
                _final_pred, history = trafficllm(s, llm, example, i_max=args.i_max)
            except Exception as e:
                log.warning("sample %d failed: %s", s["idx"], e)
                continue

            for h in history:
                pred = h["prediction"]
                row = {
                    "sample_idx": s["idx"],
                    "iter": h["iter"],
                    "method": h["method"],
                    "prediction": pred,
                    "y_true": s["y_values"],
                    "mae": mae(pred, s["y_values"]),
                    "mse": mse(pred, s["y_values"]),
                    "prompt_chars": len(h["feedback"]),
                }
                fout.write(json.dumps(row) + "\n")

            iter0_mae = history[0]["mae"]
            final_mae = history[-1]["mae"]
            summary.append((s["idx"], iter0_mae, final_mae, len(history)))

    print("\nSummary (sample_idx | iter0 MAE | final MAE | iters)")
    print("-" * 56)
    improved = 0
    for idx, m0, mf, n in summary:
        mark = "↓" if mf < m0 else "—"
        if mf < m0:
            improved += 1
        print(f"{idx:>10d} | {m0:>9.4f} | {mf:>9.4f} | {n:>5d}  {mark}")
    print("-" * 56)
    print(f"{improved}/{len(summary)} samples improved (final < iter-0)")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
