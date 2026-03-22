"""
Docstring for cli.predict

python -m cli.predict --dir outputs/gpt-oss-20b/step-by-step/hand-crafted --method step_by_step
python -m cli.predict --dir outputs/gpt-oss-20b/all-at-once/hand-crafted --method all_at_once
python -m cli.predict --dir outputs/gpt-oss-20b/text-grad/hand-crafted --method text_grad
python -m cli.predict --dir outputs/gpt-oss-20b/agent-grad/hand-crafted --method agent_grad

python -m cli.predict --dir outputs/gpt-oss-20b/step-by-step/algorithm-generated --method step_by_step
"""

import json
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from utils.common import _get_sorted_json_files, _load_json_data

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
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    
    # Strip markdown code blocks
    if "```" in text:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
    
    # Last resort: extract first {...} block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try: return json.loads(text[start:end + 1])
        except json.JSONDecodeError: pass
    
    print(f"Failed to parse JSON response: {response_text[:100]}...")
    return {}

def populate_predictions(output_dir, method='all_at_once'):
    json_files = _get_sorted_json_files(output_dir)

    for filename in tqdm(json_files, desc=f"Populating predictions [{method}]"):
        file_path = os.path.join(output_dir, filename)
        data = _load_json_data(file_path)
        assert data is not None

        logs = data.get('logs', [])
        steps = data.get('steps', [])

        # Make sure the inference already happened.
        assert all('reasoning' in log for log in logs)
        assert all('response' in log for log in logs)

        steps_by_idx = {s['step_idx']: s for s in steps}

        if method == 'all_at_once': predictions = _predictions_all_at_once(logs, steps_by_idx)
        elif method == 'step_by_step': predictions = _predictions_step_by_step(logs, steps_by_idx)
        elif method == 'text_grad': predictions = _predictions_text_grad(logs, steps_by_idx)
        elif method == 'agent_grad': predictions = _predictions_agent_grad(logs, steps_by_idx)
        else: raise ValueError(f"Unknown method: {method}")

        data['predictions'] = predictions

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


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
        'role': step.get('role', ''),
        'content': step.get('content', ''),
        'score': 1.0,
        'reason': reason,
    }]


def _predictions_step_by_step(logs, steps_by_idx):
    predictions = []
    for log in logs:
        step_idx = log.get('step_idx', -1)
        parsed = parse_llm_json_output(log.get('response', ''))

        is_decisive = parsed.get('is_decisive', False)
        reason = parsed.get('reason', '')

        # annotate each step in-place. for debugging.
        step = steps_by_idx.get(step_idx, {})
        step['is_decisive'] = is_decisive  # also fixed the typo
        step['reason'] = reason

        if not is_decisive: continue

        step = steps_by_idx.get(step_idx, {})
        predictions.append({
            'step_idx': step_idx,
            'role': step.get('role', ''),
            'content': step.get('content', ''),
            'score': 1.0,
            'reason': reason,
        })

    return sorted(predictions, key=lambda x: x['step_idx'])


def _predictions_text_grad(logs, steps_by_idx):
    """
    direct textual grad
    """
    predictions = []
    for log in logs:
        step_idx = log.get('step_idx', -1)
        parsed = parse_llm_json_output(log.get('response', ''))

        attribution = parsed.get('attribution', 'UNKNOWN').upper()
        criticism = parsed.get('criticism', '')

        # annotate each step in-place. for debugging.
        step = steps_by_idx.get(step_idx, {})
        step['attribution'] = attribution  # also fixed the typo
        step['criticism'] = criticism
        
        if attribution != 'ORIGINATING_ERROR': continue

        step = steps_by_idx.get(step_idx, {})
        predictions.append({
            'step_idx': step_idx,
            'role': step.get('role', ''),
            'content': step.get('content', ''),
            'score': 1.0,
            'reason': criticism,
        })

    return sorted(predictions, key=lambda x: x['step_idx'])

def _predictions_agent_grad(logs, steps_by_idx):
    """
    This one is a little special. Different from usual prompting.
    """
    def is_non_zero(score):
        return "non-zero" in score.lower()
    
    predictions = []
    steps = [v for k, v in steps_by_idx.items()]
    for x in steps:
        step_idx = x.get('step_idx', -1)
        if not x.get('grad'): continue

        jacobian_levels = [grad.get('jacobian_level', '').lower() for grad in x['grad']]
        
        count_nz = len([x for x in jacobian_levels if is_non_zero(x)])
        if count_nz == 0: continue

        predictions.append({
            'step_idx': step_idx,
            'role': x.get('role'),
            'content': x.get('content'),
            'score': 1.0,
            'reason': x.get('grad')
        })
    return sorted(predictions, key=lambda x: x['step_idx'])


def main():
    parser = argparse.ArgumentParser(
        description='Populate predictions from LLM responses into trace data'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory containing JSON files with LLM responses (will be modified in-place)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['all_at_once', 'step_by_step'],
        default='all_at_once',
    )
    
    args = parser.parse_args()
    
    # Validate output directory exists
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"Output directory not found: {args.dir}")
    
    print(f"Processing files in: {args.dir}")
    print(f"Method: {args.method}")
    
    populate_predictions(
        output_dir=args.dir,
        method=args.method
    )

    json_files = _get_sorted_json_files(args.dir)
    print(f"Added predictions to {len(json_files)} files")

if __name__ == '__main__':
    main()