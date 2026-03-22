"""
Docstring for cli.predict

python -m cli.predict --dir outputs/gpt-oss-20b/step-by-step/hand-crafted --method step_by_step
python -m cli.predict --dir outputs/gpt-oss-20b/all-at-once/hand-crafted --method all_at_once
"""

import json
import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.common import _get_sorted_json_files, _load_json_data


# ---------------------------------------------------------------------------
# Authoritative folder-name → method mapping
# ---------------------------------------------------------------------------
# This is the single source of truth — previously duplicated in sweep_eval.sh.
# cli.evaluate imports this to drive --predict_first without any bash logic.

FOLDER_TO_METHOD: dict[str, str] = {
    "all-at-once":          "all_at_once",
    "step-by-step-full":    "step_by_step",
    "step-by-step-partial": "step_by_step",
}


def infer_method(dir_path: str) -> str:
    """Infer the predict --method value from the strategy folder name.

    Expects the layout: <base_output>/<strategy>/<subset>
    i.e. the second-to-last path component is the strategy folder.

    Raises ValueError if the folder name is not recognised.
    """
    strategy_folder = Path(dir_path).parts[-2]
    if strategy_folder not in FOLDER_TO_METHOD:
        raise ValueError(
            f"Cannot infer method from folder {strategy_folder!r}. "
            f"Expected one of: {list(FOLDER_TO_METHOD)}"
        )
    return FOLDER_TO_METHOD[strategy_folder]


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_llm_json_output(response_text):
    """ Matches patterns:
    "```json{\"key\": \"value\"}\n```"
    "```\n{\"key\": \"value\"}\n```"
    "{\"key\": \"value\"}"
    """
    if not response_text:
        return {}

    text = response_text.strip()

    # Try direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    if "```" in text:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Last resort: extract first {...} block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    print(f"Failed to parse JSON response: {response_text[:100]}...")
    return {}


# ---------------------------------------------------------------------------
# Prediction extraction — one function per method
# ---------------------------------------------------------------------------

def _predictions_all_at_once(logs, steps_by_idx):
    assert len(logs) == 1
    parsed = parse_llm_json_output(logs[0].get('response', ''))

    predicted_idx = parsed.get('step_number', -1)
    reason = parsed.get('reason', '')

    step = steps_by_idx.get(predicted_idx)
    if step is None:
        return []

    return [{
        'step_idx': predicted_idx,
        'role':     step.get('role', ''),
        'content':  step.get('content', ''),
        'score':    1.0,
        'reason':   reason,
    }]


def _predictions_step_by_step(logs, steps_by_idx):
    predictions = []
    for log in logs:
        step_idx = log.get('step_idx', -1)
        parsed   = parse_llm_json_output(log.get('response', ''))

        is_decisive = parsed.get('is_decisive', False)
        reason      = parsed.get('reason', '')

        # Annotate each step in-place (for debugging / downstream inspection).
        step = steps_by_idx.get(step_idx, {})
        step['is_decisive'] = is_decisive
        step['reason']      = reason

        if not is_decisive:
            continue

        step = steps_by_idx.get(step_idx, {})
        predictions.append({
            'step_idx': step_idx,
            'role':     step.get('role', ''),
            'content':  step.get('content', ''),
            'score':    1.0,
            'reason':   reason,
        })

    return sorted(predictions, key=lambda x: x['step_idx'])


# ---------------------------------------------------------------------------
# Registry — keeps argparse choices and dispatch in sync automatically
# ---------------------------------------------------------------------------

_PREDICTION_FNS = {
    "all_at_once":  _predictions_all_at_once,
    "step_by_step": _predictions_step_by_step,
}


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def populate_predictions(output_dir: str, method: str = "all_at_once") -> None:
    """Parse LLM responses in *output_dir* and write 'predictions' back to each file.

    Modifies JSON files in-place.
    """
    fn = _PREDICTION_FNS.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method!r}. Choose from {list(_PREDICTION_FNS)}")

    json_files = _get_sorted_json_files(output_dir)

    for filename in tqdm(json_files, desc=f"Populating predictions [{method}]"):
        file_path = os.path.join(output_dir, filename)
        data = _load_json_data(file_path)
        assert data is not None

        logs  = data.get('logs', [])
        steps = data.get('steps', [])

        # Ensure inference has already run.
        assert all('reasoning' in log for log in logs)
        assert all('response'  in log for log in logs)

        steps_by_idx = {s['step_idx']: s for s in steps}
        data['predictions'] = fn(logs, steps_by_idx)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Populate predictions from LLM responses into trace data'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory containing JSON files with LLM responses (modified in-place)',
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=list(_PREDICTION_FNS),   # always in sync with the registry
        default='all_at_once',
    )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"Output directory not found: {args.dir}")

    print(f"Processing files in: {args.dir}")
    print(f"Method: {args.method}")

    populate_predictions(output_dir=args.dir, method=args.method)

    json_files = _get_sorted_json_files(args.dir)
    print(f"Added predictions to {len(json_files)} files")


if __name__ == '__main__':
    main()