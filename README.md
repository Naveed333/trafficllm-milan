# TrafficLLM (Milan, hourly)

Minimal implementation of TrafficLLM Algorithm 1 (initial → feedback → refine)
on the Milan internet-traffic dataset, using `gpt-4o-mini` via the OpenAI API.
No fine-tuning, no LangChain, no async.

## Setup

```bash
pip install pandas openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional; this is the default
```

`data/milan_traffic.csv` is already in place.

## Run

```bash
# 1. Build sliding-window train/test splits (1205 / 302 expected).
python scripts/prepare_data.py

# 2. Mock smoke test (no API calls — verifies the pipeline end to end).
python scripts/run_experiment.py --mock --n_samples 5 --i_max 3

# 3. Real run against gpt-4o-mini.
python scripts/run_experiment.py --n_samples 5 --i_max 3
```

## Output

Each run writes one JSONL log file under `outputs/`:
`outputs/{model}_{timestamp}.jsonl`. One row per (sample, iteration) with
`prediction`, `y_true`, `mae`, `mse`, `method`, `prompt_chars`.

The CLI also prints a summary table showing iter-0 MAE vs final MAE per
sample, and how many samples improved.

## Layout

```
src/
  data.py        sliding windows + split (W=24, L=6)
  prompts.py     exam / input / question / qualitative / feedback / refine
  llm.py         OpenAI wrapper + MockLLMClient
  metrics.py     mae, mse, parse_prediction, parse_method, safe_predict
  trafficllm.py  Algorithm 1 + validation sub-loop + patience convergence
scripts/
  prepare_data.py
  run_experiment.py
```

Numerics (MAE/MSE) are always computed in Python and injected into prompts
verbatim — the LLM is never trusted to do arithmetic. Convergence requires
two consecutive sub-`eps` MAE deltas (patience=2), guarding against the
fake-convergence case where two iterations coincidentally share the same MAE.
