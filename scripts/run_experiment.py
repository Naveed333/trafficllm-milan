#!/usr/bin/env python3
"""
TrafficLLM experiment runner.

Phase A: build K demo chains from train.jsonl (run once, cached to outputs/demo_chains.json).
Phase B: predict each test sample with a single LLM call using the cached demos.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_jsonl
from src.metrics import mae, mse
from src.trafficllm import build_demos, run_test_sample

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEMO_CACHE = os.path.join(OUTPUT_DIR, "demo_chains.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=3,
                        help="Number of test samples to evaluate (default: 3)")
    parser.add_argument("--k_demos", type=int, default=2,
                        help="Number of demo chains to build (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=3,
                        help="Max refinement iterations in Phase A (default: 3)")
    parser.add_argument("--rebuild_demos", action="store_true",
                        help="Force rebuild demo_chains.json even if it exists")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Phase A ───────────────────────────────────────────────────────────────
    if os.path.exists(DEMO_CACHE) and not args.rebuild_demos:
        print(f"[Phase A] Loading cached demos from {DEMO_CACHE}")
        with open(DEMO_CACHE) as f:
            demos = json.load(f)
    else:
        print(f"[Phase A] Building {args.k_demos} demo chains from train.jsonl ...")
        train_rows = load_jsonl(os.path.join(DATASET_DIR, "milan_train.jsonl"))
        demos = build_demos(train_rows, k=args.k_demos, max_iter=args.max_iter)
        with open(DEMO_CACHE, "w") as f:
            json.dump(demos, f, indent=2)
        print(f"[Phase A] Saved demos to {DEMO_CACHE}")

    demo_chains = [d["rendered_chain"] for d in demos]
    print(f"[Phase A] Using {len(demo_chains)} demo chain(s).\n")

    # ── Phase B ───────────────────────────────────────────────────────────────
    test_rows = load_jsonl(os.path.join(DATASET_DIR, "milan_test.jsonl"))
    test_rows = test_rows[:args.n_test]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"run_{timestamp}.jsonl")

    total_mae, total_mse, n_ok = 0.0, 0.0, 0
    print(f"[Phase B] Evaluating {len(test_rows)} test sample(s) ...\n")

    with open(out_path, "w") as out_f:
        for row in test_rows:
            idx = row["idx"]
            x_times = row["x_times"]
            x_values = row["x_values"]
            y_values = row["y_values"]   # held-out GT for scoring only

            y_hat, raw = run_test_sample(demo_chains, x_times, x_values)

            if y_hat is None:
                print(f"  idx {idx}: PARSE FAILURE — skipping.")
                result = {"idx": idx, "error": "parse_failure", "raw_response": raw}
            else:
                sample_mae = mae(y_values, y_hat)
                sample_mse = mse(y_values, y_hat)
                total_mae += sample_mae
                total_mse += sample_mse
                n_ok += 1
                print(f"  idx {idx}: MAE={sample_mae:.4f}  MSE={sample_mse:.4f}")
                result = {
                    "idx": idx,
                    "x_values": x_values,
                    "y_values": y_values,
                    "y_hat_final": y_hat,
                    "mae": sample_mae,
                    "mse": sample_mse,
                    "raw_response": raw,
                }

            out_f.write(json.dumps(result) + "\n")

    if n_ok:
        print(f"\n[Results] n={n_ok}  Avg MAE={total_mae/n_ok:.4f}  Avg MSE={total_mse/n_ok:.4f}")
        print(f"[Paper]   TrafficLLM: MAE=12.37  MSE=181.22  |  GPT-4 baseline: MAE=14.92  MSE=216.41")
    print(f"[Output]  {out_path}")


if __name__ == "__main__":
    main()
