"""
GradNorm + LLM reranking: use gradient norms to select top-k candidate steps,
then ask an LLM to pick the most likely mistake from that shortlist.

Phase 1 (offline, already done): GradNorm scores stored in ablation/<model>/<subset>/*.json
Phase 2 (this script):
    1. Load trajectories with GradNorm scores.
    2. Select top-k steps by lowest gradient norm.
    3. Build a prompt showing only those k candidate steps.
    4. Call vLLM to pick the best candidate.
    5. Save in the standard output format for cli.predict / cli.evaluate.

Example
-------
python -m cli.rerank_gradnorm \
    --config configs/qwen3-8b.yaml \
    --gradnorm-dir ablation/qwen3-8b/hand-crafted \
    --output ablation/qwen3-8b-rerank/hand-crafted \
    --strategy mlp --config-name "mlp/35" --norm-type l2_norm --k 10

python -m cli.rerank_gradnorm \
    --config configs/llama-3.1-8b.yaml \
    --gradnorm-dir ablation/llama-3.1-8b/hand-crafted \
    --output ablation/llama-3.1-8b-rerank/hand-crafted \
    --strategy attn_weights --config-name "k/25" --norm-type l2_norm --k 5

python -m cli.rerank_gradnorm \
    --config configs/gpt-oss-20b.yaml \
    --gradnorm-dir ablation/qwen3-8b/hand-crafted \
    --output ablation/qwen3-8b-rerank/hand-crafted \
    --strategy mlp --config-name "mlp/35" --norm-type l2_norm --k 10
"""

import json
import argparse
from pathlib import Path
from copy import deepcopy

from ablation.core import (
    load_trajectories, get_param_names_and_sizes,
    discover_n_layers, build_strategies,
    CompiledConfigs, score_step,
)
from utils.vllm import run_inference


# ── GradNorm top-k selection ──────────────────────────────────────────────────

def get_topk_steps(traj: dict, cc: CompiledConfigs, norm_type: str, k: int) -> list[dict]:
    """Return the top-k lowest-scoring steps as a list of dicts.

    Each dict has: step_idx, role, content, score.
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        return []

    step_map = {s["step_idx"]: s for s in traj["steps"]}
    scored = []
    for log in valid_logs:
        idx   = int(log["step_idx"])
        score = float(score_step(log, cc, norm_type)[0])
        step  = step_map.get(idx, {})
        scored.append({
            "step_idx": idx,
            "role":     step.get("role", ""),
            "content":  step.get("content", ""),
            "score":    score,
        })

    scored.sort(key=lambda x: x["score"], reverse=False)
    return scored[:k]


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a precise failure attribution expert. "
    "You always respond in valid JSON format."
)

def build_rerank_prompt(question, history, candidates):
    """Build a prompt asking the LLM to pick the most suspicious step."""
    candidate_idxs = sorted([x['step_idx'] for x in candidates])
    SEP = "\n\n---\n\n"
    turns = []
    for i, entry in enumerate(history):
        step_idx = entry['step_idx']
        role     = entry['role']
        content  = entry['content']
        if step_idx in candidate_idxs:
            turns.append(f"**[ERROR CANDIDATE]** Step {step_idx} - {role}: {content}")
        else:
            turns.append(f"Step {step_idx} - {role}: {content}")
    
    chat_content = SEP.join(turns)
    
    return (
        f"A multi-agent system attempted to solve the following problem but failed.\n\n"
        f"Problem: {question}\n\n"
        f"A prior analysis has narrowed down the failure to one of these "
        f"{len(candidates)} candidate steps. Each candidate is a step taken by an agent "
        f"during the conversation.\n\n"
        f"These candidates are steps {candidate_idxs}. "
        f"The conversation is presented as below:\n"
        f"{chat_content}\n\n"
        "Based on the conversation, analyze each of the candidate steps, and provide the following predictions in a strict JSON format:\n"
        "1. 'agent_name': The name of the agent responsible for the primary mistake leading to the incorrect solution.\n"
        "2. 'step_number': The step number (integer) where the mistake first occurred. Your step prediction must be within the list of candidate steps.\n"
        "3. 'reason': An explanation of your candidate prediction.\n\n"
        "Your response must be a valid JSON object with keys: \"agent_name\", \"step_number\", and \"reason\"."
    )


# ── Prepare dataset for vLLM ──────────────────────────────────────────────────

def prepare_dataset(
    gradnorm_dir: Path,
    strategy:     str,
    config_name:  str,
    norm_type:    str,
    k:            int,
) -> tuple[list[dict], list[dict]]:
    """Load trajectories, compute top-k, build vLLM request dataset.

    Returns (entries, requests) where:
      - entries: list of output dicts (metadata, steps, logs placeholder)
      - requests: list of vLLM request dicts with 'messages' and 'filename'
    """
    trajs = load_trajectories(gradnorm_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajs)
    strategies = build_strategies(discover_n_layers(param_names))
    cc = CompiledConfigs.compile(
        {config_name: strategies[strategy][config_name]},
        param_names, param_sizes,
    )

    entries  = []
    requests = []

    for traj in trajs:
        meta     = traj["metadata"]
        filename = meta.get("filename", "unknown.json")
        question = meta.get("question", "")
        candidates = get_topk_steps(traj, cc, norm_type, k)
        history  = traj['steps']

        if not candidates:
            continue

        # Build the output entry (logs will be filled after inference)
        entry = {
            "metadata": deepcopy(meta),
            "steps": deepcopy(traj["steps"]),
            "logs": [],
            "_candidates": candidates,  # temp, removed before saving
        }
        entries.append(entry)

        # Build vLLM request
        prompt = build_rerank_prompt(question, history, candidates)
        # import pdb; pdb.set_trace()
        requests.append({
            "filename": filename,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": prompt},
            ],
        })

    return entries, requests


# ── Parse LLM response and write output ───────────────────────────────────────

def parse_rerank_response(response_text: str, candidates: list[dict]) -> int:
    """Extract step_idx from LLM JSON response, falling back to top-1 candidate."""
    try:
        # Try to find JSON in the response
        text = response_text or ""
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            chosen = int(parsed.get("step_idx", -1))
            # Validate the chosen step is actually in our candidate set
            valid_idxs = {c["step_idx"] for c in candidates}
            if chosen in valid_idxs:
                return chosen
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: pick the top-1 gradnorm candidate
    return candidates[0]["step_idx"]


def finalize_and_save(
    entries:    list[dict],
    responses:  list[dict],
    output_dir: Path,
):
    """Merge LLM responses into entries and save as individual JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resp_map = {r["filename"]: r for r in responses}

    for entry in entries:
        filename   = entry["metadata"].get("filename", "unknown.json")
        candidates = entry.pop("_candidates")
        resp       = resp_map.get(filename, {})

        reasoning = resp.get("reasoning", None)
        response  = resp.get("response", "")
        chosen    = parse_rerank_response(response, candidates)

        # Single log entry matching all-at-once format
        entry["logs"] = [{
            "reasoning":  reasoning,
            "response":   response,
            "step_idx":   chosen,
            "candidates": candidates,
        }]

        # Write predictions directly (skip cli.predict)
        step_map = {s["step_idx"]: s for s in entry["steps"]}
        chosen_step = step_map.get(chosen, {})
        entry["predictions"] = [{
            "step_idx": chosen,
            "role":     chosen_step.get("role", ""),
            "content":  chosen_step.get("content", ""),
            "score":    1.0,
            "reason":   response,
        }]

        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=4, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GradNorm top-k + LLM reranking for failure attribution.",
    )
    parser.add_argument("--config",       required=True, help="vLLM YAML config")
    parser.add_argument("--gradnorm-dir", required=True, help="Dir with GradNorm trajectory JSONs")
    parser.add_argument("--output",       required=True, help="Output directory")
    parser.add_argument("--strategy",     default="layer")
    parser.add_argument("--config-name",  default="lm_head")
    parser.add_argument("--norm-type",    default="l1_norm")
    parser.add_argument("--k",            type=int, default=10)
    args = parser.parse_args()

    print(f"Loading GradNorm data from {args.gradnorm_dir}")
    print(f"Strategy: {args.strategy}/{args.config_name}, norm: {args.norm_type}, k={args.k}")

    entries, requests = prepare_dataset(
        Path(args.gradnorm_dir), args.strategy, args.config_name, args.norm_type, args.k,
    )
    print(f"Prepared {len(requests)} reranking requests")

    print(f"Running LLM inference ({args.config})...")
    responses = run_inference(args.config, requests)

    output_dir = Path(args.output)
    finalize_and_save(entries, responses, output_dir)
    print(f"Saved {len(entries)} results to {output_dir}")
    print(f"\nEvaluate with:")
    print(f"  python -m cli.evaluate --dir {output_dir} --ks 1")


if __name__ == "__main__":
    main()