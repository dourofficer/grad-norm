"""
Example usage:
# All-at-once
python -m cli.inference --method 'all_at_once' \
    --config 'configs/qwen3-8b.yaml' \
    --input 'ww/hand-crafted' \
    --output 'outputs/qwen3-8b/all-at-once/hand-crafted' \
    --start_idx 0 --end_idx 10

# Step-by-step full context
python -m cli.inference --method 'step_by_step_full' \
    --config 'configs/qwen3-8b.yaml' \
    --input 'ww/hand-crafted' \
    --output 'outputs/qwen3-8b/step-by-step-full/hand-crafted' \
    --start_idx 0 --end_idx 10

# Step-by-step partial context
python -m cli.inference --method 'step_by_step_partial' \
    --config 'configs/qwen3-8b.yaml' \
    --input 'ww/hand-crafted' \
    --output 'outputs/qwen3-8b/step-by-step-partial/hand-crafted' \
    --start_idx 0 --end_idx 10
"""

import json
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv
import argparse

from utils.vllm import run_inference
from utils.common import  _get_sorted_json_files, _load_json_data
from utils.prompts import (
    get_prompt_all_at_once,
    get_prompt_step_by_step_full,
    get_prompt_step_by_step_partial,
)


def process_batch(data, config, output_dir):
    """Run inference on data and save results."""
    # Index data by filename for quick lookup
    results = {entry['metadata']['filename']: entry for entry in deepcopy(data)}
    for result in results.values():
        result['logs'] = []

    # Flatten all requests across entries
    # requests = [log for entry in data for log in entry['logs']]
    requests = sum([entry['logs'] for entry in data], [])
    
    # Run inference and distribute responses back to entries
    responses = run_inference(config, requests)
    # import pdb; pdb.set_trace()
    for response in responses:
        if not ('reasoning' in response and 'response' in response):
            print(f"inference error, received no response for: {response['filename']}")
        filename = response['filename']
        results[filename]['logs'].append(response)
    
    # Save each result to its own file
    for result in results.values():
        filename = result['metadata']['filename']
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def load_and_prepare_data(input_dir, method):
    """Load JSON files and apply prompting method."""
    prompt_funcs = {
        'all_at_once': get_prompt_all_at_once,
        'step_by_step_full': get_prompt_step_by_step_full,
        'step_by_step_partial': get_prompt_step_by_step_partial,
    }
    get_prompt = prompt_funcs[method]
    
    data = []
    for filename in _get_sorted_json_files(input_dir):
        entry = _load_json_data(Path(input_dir) / filename)
        entry = get_prompt(entry)
        entry['metadata']['filename'] = filename
        
        # Attach filename to each log entry for later routing
        for log in entry['logs']:
            log['filename'] = filename
        
        data.append(entry)
    
    return data


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
        choices=["all_at_once", "step_by_step_full", "step_by_step_partial"])
    parser.add_argument("--config", required=True, default="qwen3-8b.yaml")
    parser.add_argument("--input", default="../data/ww/hand-crafted")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--api_key", default=None, help="Azure OpenAI API Key (unused for prompt generation)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for batch processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (None = process all)")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print(f"Loading data from {args.input}...")
    print(f"Method: {args.method}")
    data = load_and_prepare_data(args.input, args.method)
    
    # Process subset
    end_idx = args.end_idx or len(data)
    data_subset = data[args.start_idx:end_idx]
    
    print(f"Processing {len(data_subset)} examples (indices {args.start_idx}-{end_idx-1})")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}\n")
    
    process_batch(data_subset, args.config, output_dir)


if __name__ == "__main__":
    main()