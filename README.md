# TrafficLLM — Self-Refined Traffic Forecasting with LLMs

A thesis reproduction of the **TrafficLLM** method: can a frozen Large Language Model
(GPT-4o-mini, Llama-3.1, …) forecast hourly network traffic **without any
fine-tuning** — just by being shown well-reasoned examples in its prompt?

- **Task:** given the **past 24 hourly traffic loads** of a Milan cell, predict the **next 24 hours**.
- **Scored with:** MAE and MSE (lower is better) against the held-out true future.
- **No training of weights.** The LLM is frozen; all "learning" happens *in the prompt*.

---

## 1. How it works (the big idea)

There are two phases:

```
PHASE A  (build_demos)                 PHASE B  (run_test_sample)
─────────────────────────             ──────────────────────────────
Uses TRAIN rows, where the            Uses TEST rows, where the future
true future IS known.                 is HIDDEN from the model.

For each example, loop:               Paste the Phase-A demo chains into
  1. predict                          the prompt as examples, show the new
  2. feedback (Q1–Q4)                 input, ask once for a prediction.
  3. validate the feedback            The model imitates the refine-then-
  4. critique / correct               improve pattern it saw in the demos.
  5. refine the prediction
                                      → ONE LLM call per test row.
Save the polished "demo chain"
to outputs/demo_chains_<ds>.json.     Then score prediction vs true future.
```

**Train vs Test — the key distinction:** both files have the *same* structure
(`x_values` = past 24 hours, `y_values` = next 24 hours). The difference is only how
the code uses the future `y_values`:

| | Train (Phase A) | Test (Phase B) |
|---|---|---|
| Does the model see `y_values`? | Yes — as feedback to improve | **No** — hidden |
| Purpose | Build demonstration examples | Measure accuracy on unseen data |
| LLM calls per row | Many (predict → refine loop) | One (just predict) |
| `y_values` used for | Teaching / refining | Scoring only |

---

## 2. Project layout

| Path | Role |
|------|------|
| `dataset/milan_tw_*.jsonl`   | **Time-window** dataset (44 train / 19 test) |
| `dataset/milan_slid_*.jsonl` | **Sliding-window** dataset (1042 train / 447 test) |
| `src/data.py`        | Loads `.jsonl` files |
| `src/llm.py`         | Single gateway to the LLM (OpenAI or self-hosted) |
| `src/prompts.py`     | All prompt text templates (system, feedback Q1–Q4, refine…) |
| `src/metrics.py`     | MAE, MSE, sine/cosine periodicity fit |
| `src/trafficllm.py`  | The brain: `build_demos` (Phase A) + `run_test_sample` (Phase B) |
| `scripts/run_experiment.py` | The entry point you run |
| `outputs/`           | Cached demo chains + per-run result files |
| `.env`               | Your API key / model choice |

---

## 3. Setup

**Requirements:** Python 3.11+, and two packages:

```bash
pip install openai python-dotenv
```

**Configure your model** in `.env` (copy from `.env.example` if needed):

- **Option A — OpenAI** (default): set a valid key, leave `LLM_BASE_URL` commented out.
  ```
  OPENAI_API_KEY=sk-...your-real-key...
  OPENAI_MODEL=gpt-4o-mini
  ```
- **Option B — Self-hosted / OpenAI-compatible** (vLLM, Ollama, Together, ngrok, …):
  ```
  LLM_BASE_URL=https://your-server/v1
  LLM_MODEL=llama-3.1-8b
  # LLM_API_KEY=        # leave blank for self-hosted (defaults to "EMPTY")
  ```

> Note: requests run at `temperature=0.0` (deterministic) — the `LLM_TEMPERATURE`
> value in `.env` is currently ignored by the code.

**Verify your provider works** before a full run:

```bash
python3 -c "from src.llm import chat; print(chat([{'role':'user','content':'say OK'}]))"
```

A `401 invalid_api_key` here means the key in `.env` is wrong/expired — fix it first.

---

## 4. Running

When you omit `--dataset`, the script **asks which dataset to run**:

```bash
python3 scripts/run_experiment.py --n_test 3
# Which dataset do you want to run?
#   1) tw    -> milan_tw_train.jsonl / milan_tw_test.jsonl
#   2) slid  -> milan_slid_train.jsonl / milan_slid_test.jsonl
# Enter number or name:
```

Or name it directly to skip the prompt:

```bash
# tiny smoke test — fastest, cheapest (1 demo, 1 test row)
python3 scripts/run_experiment.py --dataset tw --n_test 1 --k_demos 1 --rebuild_demos

# normal run
python3 scripts/run_experiment.py --dataset tw --n_test 3 --k_demos 2

# full evaluation
python3 scripts/run_experiment.py --dataset tw   --n_test 19  --rebuild_demos
python3 scripts/run_experiment.py --dataset slid --n_test 447 --rebuild_demos
```

### Command-line flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset {tw,slid}` | *(asks)* | Which dataset family to use |
| `--n_test`     | 3 | How many test rows to evaluate |
| `--k_demos`    | 2 | How many train rows become demo chains |
| `--max_iter`   | 3 | Max refinement iterations per demo in Phase A |
| `--rebuild_demos` | off | Force rebuild demos (ignore the cache) |
| `--seed`       | 42 | Accepted but currently unused |

### About the demo cache

Phase A is expensive (many LLM calls), so its output is cached per dataset to
`outputs/demo_chains_<dataset>.json`. Later runs **reuse** it (fast). Pass
`--rebuild_demos` to regenerate — do this the first time on a dataset, or whenever
you change the prompts or model.

---

## 5. Reading the results

Each run writes a timestamped file and prints average scores:

```bash
ls -lt outputs/run_*.jsonl | head        # newest result files
cat outputs/demo_chains_tw.json | head    # inspect a built demo chain
```

Result file (`outputs/run_<dataset>_<timestamp>.jsonl`) — one JSON object per test row:

```json
{"idx": 1, "x_values": [...], "y_values": [...], "y_hat_final": [...], "mae": 11.8, "mse": 170.2, "raw_response": "..."}
```

Printed summary compares against the paper's reported numbers:

```
[Results] n=19  Avg MAE=...  Avg MSE=...
[Paper]   TrafficLLM: MAE=12.37  MSE=181.22  |  GPT-4 baseline: MAE=14.92  MSE=216.41
```

---

## 6. Step-by-step quick start

```bash
cd /Users/naveed/Documents/LUMS/MSAI_Thesis/traffic_llm_may_3

# 1. set OPENAI_API_KEY (or LLM_BASE_URL) in .env
# 2. verify the provider
python3 -c "from src.llm import chat; print(chat([{'role':'user','content':'say OK'}]))"
# 3. smoke test
python3 scripts/run_experiment.py --dataset tw --n_test 1 --k_demos 1 --rebuild_demos
# 4. real run
python3 scripts/run_experiment.py --dataset tw --n_test 19 --rebuild_demos
# 5. inspect outputs/
ls -lt outputs/
```

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `401 invalid_api_key` | Bad/expired key in `.env`. Set a valid `OPENAI_API_KEY`, or use `LLM_BASE_URL`. |
| `FileNotFoundError ...milan_train.jsonl` | Old filename. Use `--dataset tw` or `slid` (datasets are now `milan_tw_*` / `milan_slid_*`). |
| `PARSE FAILURE` on a test row | The model didn't return 24 clean numbers; the code retries, then records the failure and moves on. |
| Phase A "skipping" a demo | Initial prediction couldn't be parsed twice — that train row is skipped. |
| Run is very slow | Phase A (or `slid` with large `--n_test`) makes many calls. Use the cache (drop `--rebuild_demos`) and start with small `--n_test`. |
