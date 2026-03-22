# Prerequisites for `sweep_eval.sh`

`sweep_eval.sh` assumes two earlier stages have already been completed:
**inference** (`cli.inference`) and, implicitly, the raw data and config setup that
inference depends on. Nothing in the sweep script creates these files — it will
silently skip or crash on any directory that is missing.

---

## 1. Input data — `data/ww/`

```
data/
└── ww/
    ├── hand-crafted/
    │   ├── <case_id_1>.json
    │   ├── <case_id_2>.json
    │   └── ...
    └── algorithm-generated/
        ├── <case_id_1>.json
        ├── <case_id_2>.json
        └── ...
```

Each `.json` file must contain at minimum:

| Field           | Type   | Description                                      |
|-----------------|--------|--------------------------------------------------|
| `history`       | list   | Ordered list of `{role, content}` agent steps    |
| `question`      | string | The problem being solved                         |
| `ground_truth`  | string | The correct answer                               |
| `mistake_agent` | string | Ground-truth agent responsible for the error     |
| `mistake_step`  | string | Ground-truth step index where the error occurred |

Algorithm-generated files may additionally contain a `system_prompt` dict describing each agent's role.

---

## 2. Model config — `configs/`

```
configs/
└── gpt-oss-20b.yaml    # or whichever model you set MODEL= to
```

The YAML file is consumed by `cli.inference` (via `utils/vllm.py`). At minimum it
must specify the API endpoint, model name, and any generation parameters. The
`sweep_eval.sh` script reads the model name from the `MODEL` env var
(default: `gpt-oss-20b`) and expects the matching YAML to already exist.

---

## 3. Inference outputs — `outputs/<model>/`

This is the **direct prerequisite** for `sweep_eval.sh`. Run `cli.inference`
once for each method/subset combination before running the sweep:

```bash
MODEL="gpt-oss-20b"

for METHOD in all_at_once step_by_step_full step_by_step_partial; do
  for SUBSET in hand-crafted algorithm-generated; do
    python -m cli.inference \
      --method  "${METHOD}" \
      --config  "configs/${MODEL}.yaml" \
      --input   "data/ww/${SUBSET}" \
      --output  "outputs/${MODEL}/${METHOD//_/-}/${SUBSET}"
  done
done
```

After all six runs the expected directory tree is:

```
outputs/
└── gpt-oss-20b/
    ├── all-at-once/
    │   ├── hand-crafted/
    │   │   ├── <case_id_1>.json
    │   │   └── ...
    │   └── algorithm-generated/
    │       └── ...
    ├── step-by-step-full/
    │   ├── hand-crafted/
    │   └── algorithm-generated/
    └── step-by-step-partial/
        ├── hand-crafted/
        └── algorithm-generated/
```

Each output `.json` is an enriched copy of the input file containing, in
addition to the original fields, the keys `metadata`, `steps`, and `logs`
(with `reasoning` and `response` fields populated by the model).

> **Partial runs are fine.** `sweep_eval.sh` checks whether each directory
> exists and skips it with a warning if not — so you can run the sweep on
> whichever method/subset combinations have finished inference.

---

## Quick checklist

```
[ ] data/ww/hand-crafted/*.json               — raw cases
[ ] data/ww/algorithm-generated/*.json        — raw cases
[ ] configs/<MODEL>.yaml                       — model config
[ ] outputs/<MODEL>/all-at-once/hand-crafted/          — inference done
[ ] outputs/<MODEL>/all-at-once/algorithm-generated/   — inference done
[ ] outputs/<MODEL>/step-by-step-full/hand-crafted/    — inference done
[ ] outputs/<MODEL>/step-by-step-full/algorithm-generated/   — inference done
[ ] outputs/<MODEL>/step-by-step-partial/hand-crafted/       — inference done
[ ] outputs/<MODEL>/step-by-step-partial/algorithm-generated/  — inference done
```

Once all boxes are ticked, run:

```bash
MODEL=gpt-oss-20b KS="1 5 10" bash sweep_eval.sh
```