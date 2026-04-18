import os
import json
import re
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
from utils.vllm import send_request
from rich.console import Console
from rich.markdown import Markdown
from typing import Any, Dict, List, Optional, Tuple


SEEN = False
def print_once(text):
    global SEEN
    if not SEEN: 
        print(text)
        SEEN = True
        
def mdprint(text):
    console = Console()
    md = Markdown(text)
    console.print(md)

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _extract_metadata(data):
    """
    Extracts metadata for the labels field.
    Includes specific requested fields and any other metadata from the dataset,
    while excluding heavy fields like history and system_prompt.
    """
    metadata = {
        "question":           data.get("question"),
        "ground_truth":       data.get("ground_truth"),
        "is_corrected":       data.get("is_corrected"),
        "mistake_agent":      data.get("mistake_agent"),
        "mistake_step":       data.get("mistake_step"),
        "mistake_reason":     data.get("mistake_reason"),
        "mistake_type":       data.get("mistake_type"),
        "question_id":        data.get("question_ID"), 
        "system_description": data.get("system_prompt"),
        "subset":             data.get("subset"), # this is injected, not presented in original file.
        "filename":           data.get("filename"), # this as well is injected.
        "include_gt":         data.get("include_gt"), # this is injected during running inference.
    }
            
    return metadata

def _quick_vllm(prompt):
    config = {
        "model": "openai/gpt-oss-20b",
        "temperature": 0.6,
        "max_tokens": 4000,
        "reasoning_effort": "low",
        "hostname": "localhost",
        "port": 8881,
        "concurrent_requests": 16
    }
    hostname = config.pop("hostname")
    port     = config.pop("port")
    config.pop("concurrent_requests", None)
    messages = [{'role': 'user', 'content': prompt}]

    url = f"http://{hostname}:{port}/v1/chat/completions"
    _, out, _ = send_request(url, config, {"messages": messages}, request_id=0)
    return out['response']


def _call_vllm(messages: list, config_path: str) -> Dict[str, str]:
    """Send a messages list to vLLM and return {reasoning, response}."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hostname = config.pop("hostname")
    port     = config.pop("port")
    config.pop("concurrent_requests", None)

    url = f"http://{hostname}:{port}/v1/chat/completions"
    _, out, _ = send_request(url, config, {"messages": messages}, request_id=0)
    return {"reasoning": out["reasoning"], "response": out["response"]}


def copy_long_context_files(result_dir="outputs/gpt-oss-20b", threshold=50):
    
    def is_long(dir, filename, threshold=50):
        filepath = dir / filename
        data = _load_json_data(filepath)
        num_steps = len(data['steps'])
        return num_steps >= threshold
    
    result_dir = Path(result_dir)
    methods = ['all-at-once', 'step-by-step', 'text-grad', 'agent-grad']
    for method in methods:
        print(f'extracting long files: {method}')
        input_dir = result_dir / f"{method}/hand-crafted"
        output_dir = result_dir / f"{method}/long-context"
        output_dir.mkdir(parents=True, exist_ok=True)
        all_files = _get_sorted_json_files(input_dir)
        long_files = [filename for filename in all_files if is_long(input_dir, filename, threshold)]
        for filename in long_files:
            shutil.copy(input_dir / filename, output_dir / filename)
