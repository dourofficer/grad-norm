# Running `sweep_eval.sh` — Prerequisites & Extending to New Methods

---

## Part 1 — Prerequisites Before Running `sweep_eval.sh`

The script assumes three earlier stages are complete. Nothing in the sweep
creates these files; it will skip or crash on any directory that is missing.

### 1. Raw data — `data/ww/`

```
data/
└── ww/
    ├── hand-crafted/
    │   ├── <case_id>.json
    │   └── ...
    └── algorithm-generated/
        ├── <case_id>.json
        └── ...
```

Each `.json` file must contain at minimum:

| Field           | Type   | Description                                               |
|-----------------|--------|-----------------------------------------------------------|
| `history`       | list   | Ordered `{role, content}` agent steps                     |
| `question`      | string | The problem being solved                                  |
| `ground_truth`  | string | The correct answer                                        |
| `mistake_agent` | string | Ground-truth agent responsible for the error              |
| `mistake_step`  | string | Ground-truth step index where the error occurred (string) |

### 2. Model config — `configs/<MODEL>.yaml`

The YAML file consumed by `cli.inference`. The `MODEL` env var (default: `gpt-oss-20b`)
must match the filename, e.g. `configs/gpt-oss-20b.yaml`.

### 3. Inference outputs — `outputs/<MODEL>/`

Run `cli.inference` once per method/subset combination before the sweep.
The expected directory layout is:

```
outputs/
└── <model>/
    ├── <strategy-folder-1>/
    │   ├── hand-crafted/
    │   │   ├── <case_id>.json
    │   │   └── ...
    │   └── algorithm-generated/
    │       └── ...
    ├── <strategy-folder-2>/
    │   └── ...
    └── ...
```

Each output `.json` must contain the fields below **in addition to** all raw
data fields. These are written by `cli.inference` and read by `cli.predict`:

| Field    | Type | Description                                                 |
|----------|------|-------------------------------------------------------------|
| `metadata` | dict | Must carry `mistake_agent` and `mistake_step` (copied from the raw data) |
| `steps`  | list | Each element is a step dict with at least `step_idx` (int), `role` (str), `content` (str) |
| `logs`   | list | Each element has `reasoning` (str) and `response` (str) populated by the model; step-level methods also carry `step_idx` (int) |

### 4. Quick checklist

```
[ ] data/ww/hand-crafted/*.json                             — raw cases
[ ] data/ww/algorithm-generated/*.json                      — raw cases
[ ] configs/<MODEL>.yaml                                    — model config
[ ] outputs/<MODEL>/<strategy>/hand-crafted/                — inference done (×N strategies)
[ ] outputs/<MODEL>/<strategy>/algorithm-generated/         — inference done (×N strategies)
```

Once all boxes are ticked:

```bash
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh
```

To skip re-running prediction (if predictions are already populated):

```bash
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh --skip_predict
```

---

## Part 2 — Adding a New Method

You have complete freedom in how the method works internally. What the pipeline
cares about is:

1. The **output JSON schema** that sits on disk after your method runs.
2. A **`_predictions_<method>` function** in `cli/predict.py` that reads those
   files and produces the standard prediction list.
3. Two **registry entries** — one in `cli/predict.py`, one in
   `cli/evaluate.py` — so the sweep discovers and labels your method correctly.

`cli/evaluate.py` never needs to know how a method works; it only reads
the `predictions` field that `cli/predict.py` writes.

---

### 2.1 Required output JSON schema

After your method runs, each case file in
`outputs/<MODEL>/<your-strategy-folder>/<subset>/` must contain **at minimum**:

```jsonc
{
    // ── copied from raw data ──────────────────────────────────────
    "metadata": {
        "mistake_agent": "WebSurfer",   // string — ground-truth agent
        "mistake_step":  "4"            // string — ground-truth step index
    },

    // ── original trajectory steps ────────────────────────────────
    "steps": [
        {
            "step_idx": 0,              // int, 0-indexed
            "role":     "Orchestrator", // string
            "content":  "..."           // string
        }
        // ...
    ],

    // ── your method's raw output (structure is yours to define) ───
    "logs": [ ... ]

    // NOTE: "predictions" is intentionally absent here.
    // cli.predict reads "logs" + "steps" and writes "predictions".
}
```

The `logs` field is the only one your method fully owns. Its internal structure
is up to you — the only constraint is that **your `_predictions_<method>` parser
can read it** and produce the prediction list described in §2.2.

---

### 2.2 The prediction list contract

`cli/predict.py` must write a `predictions` key into each file.
`cli/evaluate.py` reads `predictions[:k]` and checks two fields per item.
Every entry in the list **must** have:

```jsonc
{
    "step_idx": 3,              // int — the step being nominated
    "role":     "WebSurfer",    // string — the agent that owns that step
    "score":    0.87,           // float — used for ranking; higher = more suspicious
    "content":  "...",          // string — step content (for human inspection)
    "reason":   "..."           // string — explanation (for human inspection)
}
```

The list must be **sorted by `score` descending** so that `predictions[:k]`
always yields the top-k most suspicious steps. `cli/evaluate.py` slices the
list directly and never re-sorts.

`content` and `reason` are not used by the evaluation logic; they exist purely
for debugging and human review. Fill them with whatever is useful, or empty
strings if your method doesn't produce explanations.

---

### 2.3 Extending `cli/predict.py`

**Step 1** — Write a parser function:

```python
def _predictions_my_method(logs, steps_by_idx):
    """
    Read your method's raw output from `logs` and return the prediction list.

    Parameters
    ----------
    logs         : list[dict]  — the "logs" field from the case JSON
    steps_by_idx : dict[int, dict]  — step dicts keyed by step_idx

    Returns
    -------
    list[dict]  — prediction items; must conform to the contract in §2.2
    """
    predictions = []
    for log in logs:
        step_idx = log.get("step_idx", -1)
        score    = log.get("score", 0.0)   # whatever your method produces
        reason   = log.get("reason", "")

        step = steps_by_idx.get(step_idx, {})
        predictions.append({
            "step_idx": step_idx,
            "role":     step.get("role", ""),
            "content":  step.get("content", ""),
            "score":    score,
            "reason":   reason,
        })

    # Sort descending by score so [:k] gives the top-k suspects.
    return sorted(predictions, key=lambda x: x["score"], reverse=True)
```

**Step 2** — Register it in `_PREDICTION_FNS`:

```python
_PREDICTION_FNS = {
    "all_at_once":  _predictions_all_at_once,
    "step_by_step": _predictions_step_by_step,
    "text_grad":    _predictions_text_grad,
    "agent_grad":   _predictions_agent_grad,
    "my_method":    _predictions_my_method,   # ← add this line
}
```

**Step 3** — Map your strategy folder name to the method key in `FOLDER_TO_METHOD`:

```python
FOLDER_TO_METHOD = {
    "all-at-once":          "all_at_once",
    "step-by-step-full":    "step_by_step",
    "step-by-step-partial": "step_by_step",
    "text-grad":            "text_grad",
    "agent-grad":           "agent_grad",
    "my-method":            "my_method",   # ← add this line
}
```

The folder name is the actual directory name under `outputs/<MODEL>/`.
`--method` choices in the CLI and `infer_method()` in `cli/evaluate.py` are
both derived from these two dicts automatically — no other changes needed in
`cli/predict.py`.

---

### 2.4 Extending `cli/evaluate.py`

The only thing `cli/evaluate.py` needs to know about a new method is its
**display label** for the results table. Add one line to `STRATEGY_LABELS`:

```python
STRATEGY_LABELS = {
    "all-at-once":          "all-at-once",
    "step-by-step-full":    "step-by-step (k)",
    "step-by-step-partial": "step-by-step (n)",
    "my-method":            "my-method (label)",  # ← add this line
}
```

If you also want your method to appear in a specific row order in the sweep
table, append its label to `STRATEGY_ORDER`:

```python
STRATEGY_ORDER = [
    "all-at-once",
    "step-by-step (k)",
    "step-by-step (n)",
    "my-method (label)",   # ← add this line
]
```

That's everything. `cli/evaluate.py` reads only the `predictions` list and
`metadata`; it is otherwise method-agnostic.

---

### 2.5 End-to-end checklist for a new method

```
[ ] Inference outputs written to outputs/<MODEL>/my-method/<subset>/
[ ] Each case JSON has: metadata, steps, logs (your raw output)
[ ] _predictions_my_method() added to cli/predict.py
[ ] "my_method" added to _PREDICTION_FNS in cli/predict.py
[ ] "my-method" → "my_method" added to FOLDER_TO_METHOD in cli/predict.py
[ ] "my-method" → "my-method (label)" added to STRATEGY_LABELS in cli/evaluate.py
[ ] Label appended to STRATEGY_ORDER in cli/evaluate.py (optional, for row ordering)
```

Then run as normal:

```bash
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh
```