from pathlib import Path
from utils.common import _get_sorted_json_files, _load_json_data
import json
import csv

TABLE_KS = [1, 5, 10]


def step_at_k(scores, true_step, k):
    scores = {int(i): v for i, v in scores.items()}
    ranked = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=True)
    return int(true_step in ranked[:k])


def agent_at_k(scores, step_agents, true_agent, k):
    scores      = {int(i): v for i, v in scores.items()}
    step_agents = {int(i): v for i, v in step_agents.items()}
    ranked      = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=True)
    return int(true_agent in {step_agents.get(idx, "") for idx in ranked[:k]})


def compute_metrics(results, ks=TABLE_KS):
    args = results["args"]
    if   args["subset"] == "algorithm-generated": n = 126
    elif args["subset"] == "hand-crafted":         n = 58
    else: raise NotImplementedError()

    totals = {f"step_acc@{k}": 0 for k in ks}
    totals.update({f"agent_acc@{k}": 0 for k in ks})

    for res in results["results"]:
        for k in ks:
            totals[f"step_acc@{k}"]  += step_at_k(res["scores"], res["true_step"], k)
            totals[f"agent_acc@{k}"] += agent_at_k(res["scores"], res["step_agents"], res["true_agent"], k)

    output = {key: val / n for key, val in totals.items()}
    output["metadata"] = {**args, "total_samples": n}
    return output


def write_tsv(all_metrics, output_path, ks=TABLE_KS):
    step_cols  = [f"Step @{k}"  for k in ks]
    agent_cols = [f"Agent @{k}" for k in ks]
    header     = ["Method", "Model", "Subset", "Layer"] + step_cols + agent_cols

    layer_order = {"lm_head": 0, "out_proj": 1, "final_layer": 2}
    rows = sorted(all_metrics, key=lambda m: (
        m["metadata"]["subset"],
        Path(m["metadata"]["model"]).name,
        layer_order.get(m["metadata"]["layer"], 99),
    ))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for m in rows:
            meta = m["metadata"]
            row  = ["GradNorm", Path(meta["model"]).name, meta["subset"], meta["layer"]]
            row += [f"{m[f'step_acc@{k}']:.4f}"  for k in ks]
            row += [f"{m[f'agent_acc@{k}']:.4f}" for k in ks]
            writer.writerow(row)


if __name__ == "__main__":
    output_dir  = Path("outputs")
    filenames   = _get_sorted_json_files(output_dir)
    all_results = [_load_json_data(output_dir / fn) for fn in filenames]
    all_metrics = [compute_metrics(res, ks=TABLE_KS) for res in all_results]

    json_path = "./sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"all_metrics": all_metrics}, f, indent=2)
    print(f"JSON → {json_path}")

    tsv_path = "./sweep_results.tsv"
    write_tsv(all_metrics, tsv_path, ks=TABLE_KS)
    print(f"TSV  → {tsv_path}")