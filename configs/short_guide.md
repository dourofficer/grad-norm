# `sweep_eval.sh` — Prerequisites & Extending to New Methods

---

## Part 1 — Prerequisites

Three stages must be complete before running the sweep.

### 1. Raw data — `data/ww/`

Each case `.json` under `hand-crafted/` and `algorithm-generated/` must contain:

| Field | Type | Description |
|---|---|---|
| `history` | list | Ordered `{role, content}` agent steps |
| `question` | string | Problem being solved |
| `ground_truth` | string | Correct answer |
| `mistake_agent` | string | Ground-truth responsible agent |
| `mistake_step` | string | Ground-truth error step index |

### 2. Model config — `configs/<MODEL>.yaml`

Must exist and match the `MODEL` env var (default: `gpt-oss-20b`).

### 3. Inference outputs — `outputs/<MODEL>/<strategy>/<subset>/`

Run `cli.inference` for each method/subset combination first. Each output `.json` must carry these fields **on top of** the raw data fields:

| Field | Type | Description |
|---|---|---|
| `metadata` | dict | Must include `mistake_agent` and `mistake_step` |
| `steps` | list | Each item: `step_idx` (int), `role` (str), `content` (str) |
| `logs` | list | Each item: `reasoning` (str), `response` (str); step-level methods also include `step_idx` (int) |

> `predictions` is intentionally absent — `cli.predict` writes it.

### Checklist

```
[ ] data/ww/{hand-crafted,algorithm-generated}/*.json                        — raw cases
[ ] configs/<MODEL>.yaml                                                      — model config
[ ] outputs/<MODEL>/<strategy>/{hand-crafted,algorithm-generated}/            — inference done (× N strategies)
```

```bash
# Run everything (predict + evaluate):
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh

# Skip predict phase if predictions are already populated:
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh --skip_predict
```

---

## Part 2 — Adding a New Method

The pipeline only cares about three things: the output JSON schema, a parser function in `cli/predict.py`, and two registry entries. `cli/evaluate.py` is fully method-agnostic.

### 2.1 Output JSON schema

Your method's output files (in `outputs/<MODEL>/<your-folder>/<subset>/`) must have `metadata`, `steps`, and `logs` as described in §1.3. The `logs` field is yours to define freely — its structure only needs to be readable by your parser function below.

### 2.2 The `predictions` list contract

`cli/predict.py` reads `logs` + `steps` and writes a `predictions` list. `cli/evaluate.py` slices `predictions[:k]` and checks `role` against `mistake_agent` and `step_idx` against `mistake_step`. Every prediction entry must be:

```jsonc
{
    "step_idx": 3,          // int   — nominated step
    "role":     "Agent",    // str   — agent owning that step
    "score":    0.87,       // float — ranking score; higher = more suspicious
    "content":  "...",      // str   — for inspection only; can be empty
    "reason":   "..."       // str   — for inspection only; can be empty
}
```

The list must be **sorted by `score` descending**. `cli/evaluate.py` slices directly without re-sorting.

### 2.3 Changes to `cli/predict.py`

**1. Write a parser function:**

```python
def _predictions_my_method(logs, steps_by_idx):
    predictions = []
    for log in logs:
        step_idx = log.get("step_idx", -1)
        step     = steps_by_idx.get(step_idx, {})
        predictions.append({
            "step_idx": step_idx,
            "role":     step.get("role", ""),
            "content":  step.get("content", ""),
            "score":    log.get("score", 0.0),
            "reason":   log.get("reason", ""),
        })
    return sorted(predictions, key=lambda x: x["score"], reverse=True)
```

**2. Register it in `_PREDICTION_FNS`:**

```python
_PREDICTION_FNS = {
    ...
    "my_method": _predictions_my_method,  # ← add
}
```

**3. Map your strategy folder name in `FOLDER_TO_METHOD`:**

```python
FOLDER_TO_METHOD = {
    ...
    "my-method": "my_method",  # ← add
}
```

`--method` CLI choices and `infer_method()` are both derived from these dicts — no other changes needed.

### 2.4 Changes to `cli/evaluate.py`

Add a display label to `STRATEGY_LABELS`, and optionally a row-order entry to `STRATEGY_ORDER`:

```python
STRATEGY_LABELS = {
    ...
    "my-method": "my-method (label)",  # ← add
}

STRATEGY_ORDER = [
    ...
    "my-method (label)",  # ← add (optional; controls row order in sweep table)
]
```

### 2.5 Checklist for a new method

```
[ ] Inference outputs written to outputs/<MODEL>/my-method/<subset>/
[ ] Each case JSON has: metadata, steps, logs
[ ] _predictions_my_method() written and added to _PREDICTION_FNS in cli/predict.py
[ ] "my-method" → "my_method" added to FOLDER_TO_METHOD in cli/predict.py
[ ] "my-method" → label added to STRATEGY_LABELS in cli/evaluate.py
[ ] Label appended to STRATEGY_ORDER in cli/evaluate.py (optional)
```