import json
import requests
import time
import argparse
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request(url, pload_config, data, request_id):
    """Sends a single request to the vLLM server."""
    headers = {"Content-Type": "application/json"}
    messages = data.get("messages")
    
    pload = {
        **pload_config,
        "messages": messages,
        "stream": False
    }

    result_entry = {}
    error_entry = None

    try:
        # Reduced timeout to 300s (5 mins) to prevent massive hangs on stuck requests
        response = requests.post(url, headers=headers, json=pload, timeout=900)
        response.raise_for_status()

        response_json = response.json()
        
        # Handle cases where reasoning_content might be missing depending on model
        message = response_json["choices"][0]["message"]
        reasoning = message.get("reasoning_content", None)
        
        result_entry = {
            "request_id": request_id,
            "reasoning": reasoning,
            "response": message["content"],
            **data
        }
    except Exception as e:
        error_entry = (request_id, str(e))
        result_entry = {
            "request_id": request_id,
            "messages": messages,
            "reasoning": None,
            "response": None,
            **data
        }
    
    return request_id, result_entry, error_entry

def call_vllm(prompt, config_path="./configs/gpt-oss-20b.yaml"):
    """Call vLLM inference endpoint with a prompt."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    hostname = config.pop("hostname")
    port = config.pop("port")
    concurrent_requests = config.pop("concurrent_requests", 10)
    
    url = f"http://{hostname}:{port}/v1/chat/completions"
    
    data = {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]}
    _, out, _ = send_request(url, config, data, request_id=0)
    return {"reasoning": out["reasoning"], "response": out["response"]}

def run_inference(config_path, dataset):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    hostname = config.pop("hostname")
    port = config.pop("port")
    # This now controls actual thread count, not just active requests
    concurrent_requests = config.pop("concurrent_requests", 10)
    
    url = f"http://{hostname}:{port}/v1/chat/completions"
    pload_config = config

    results = {}
    errors = {}
    start_time = time.time()

    print(f"Starting inference with {concurrent_requests} concurrent workers...")

    # USE THREADPOOLEXECUTOR INSTEAD OF RAW THREADS
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # Submit all tasks to the pool
        future_to_req = {
            executor.submit(send_request, url, pload_config, data, i): i 
            for i, data in enumerate(dataset)
        }

        # Process results as they complete
        with tqdm(total=len(dataset), desc="Processing requests") as pbar:
            for future in as_completed(future_to_req):
                request_id, result_data, error_data = future.result()
                
                results[request_id] = result_data
                if error_data:
                    errors[error_data[0]] = error_data[1]
                
                pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Write sorted results
    sorted_results = [results[i] for i in sorted(results.keys())]
    return sorted_results

def run_inference_e2e(config_path, input_file, results_file):
    """Run batch inference with config file using ThreadPoolExecutor."""
    dataset = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip(): # Skip empty lines
                dataset.append(json.loads(line))

    sorted_results = run_inference(config_path, dataset)
    
    with open(results_file, "w") as outfile:
        for result in sorted_results:
            json.dump(result, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script for vLLM server.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--input-file", type=str, required=True, help="JSONL file with input prompts")
    parser.add_argument("--results-file", type=str, required=True, help="Output file for results (JSONL)")
    args = parser.parse_args()

    run_inference_e2e(args.config, args.input_file, args.results_file)