import os
import json
import re
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
from utils.common import _extract_metadata

# ============================================================
# ALL-AT-ONCE STRATEGY
# ============================================================

def get_prompt_all_at_once(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}" 
        for i, entry in enumerate(chat_history)
    ])

    # only algorithm-generated will have system description.
    system_desc = data.get("system_prompt", {})
    system_text = ''
    if system_desc:
        agents_description = SEP.join(
            f" * {k}: {v.strip('## Your role').strip()}"  for k, v in system_desc.items()
        )
        system_text = f"Agentic System Description\n{agents_description}\n\n"

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
        f"The Problem: {problem}\n"
        f"The Ground Truth Answer: {ground_truth}\n\n"
        f"{system_text}"
        "Task: Identify which agent made an error, the specific step number where the error occurred, and the reason for the error.\n"
        "Here is the conversation:\n\n" + chat_content + "\n\n"
        "Based on the conversation above, provide the following predictions in a strict JSON format:\n"
        "1. 'agent_name': The name of the agent responsible for the primary mistake leading to the incorrect solution. If no specific agent is clearly at fault, select the most likely candidate.\n"
        "2. 'step_number': The step number (integer) where the mistake first occurred. (e.g., if the mistake is in the second entry of the history, the step number is 2). Your step prediction must be within the range of the steps provided in the conversation.\n"
        "3. 'reason': A concise explanation of your prediction.\n\n"
        "Your response must be a valid JSON object with keys: \"agent_name\", \"step_number\", and \"reason\"."
    )
    system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step['role'],
                'content': step['content'],
            }
            for step_idx, step in enumerate(chat_history)
        ],
        'logs': [{
            'filename': None,
            'messages': messages,
            # 'reasoning': None,
            # 'response': None
        }],
    }

# ============================================================
# STEP-BY-STEP STRATEGY
# ============================================================

def get_prompt_step_by_step(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}" 
        for i, entry in enumerate(chat_history)
    ])

    # only algorithm-generated will have system description.
    system_desc = data.get("system_prompt", {})
    system_text = ''
    if system_desc:
        agents_description = SEP.join(
            f" * {k}: {v.strip('## Your role').strip()}"  for k, v in system_desc.items()
        )
        system_text = f"Agentic System Description\n{agents_description}\n\n"

    logs = []
    for idx, entry in enumerate(chat_history):
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            f"{system_text}"
            f"Here is the conversation:\n\n{chat_content}\n\n" 
            f"Task: Determine whether STEP {idx} (performed by {entry.get('role')}) is an decisive error step. "
            "A decisive error step is a mistake step where, if corrected, with all following steps are adjusted accordingly, the system would succeed. "
            "When multiple mistakes exist, the earliest one is most decisive. "
            "Focus only on errors that critically derail the process, rather than minor imperfections.\n\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"is_decisive\": boolean (true or false)\n"
            "2. \"reason\": string (explanation for your judgment)"
        )
        system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        logs.append({
            'filename': None,
            'step_idx': idx,
            'messages': messages,
        })

    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step['role'],
                'content': step['content'],
            }
            for step_idx, step in enumerate(chat_history)
        ],
        'logs': logs,
    }

# ============================================================
# DIRECT TEXT-GRAD STRATEGY
# ============================================================

def get_prompt_text_grad(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}"
        for i, entry in enumerate(chat_history)
    ])

    json_template = {
        "attribution": "ORIGINATING_ERROR | PROPAGATING_ERROR | NEITHER",
        "criticism":   "Your detailed analysis here.",
    }
    EXAMPLE_OUTPUT = json.dumps(json_template, indent=4)
    TASK_TEMPLATE = (
        "**ATTRIBUTION Guide**:\n"
        "- **ORIGINATING_ERROR**: This step contains the ORIGINAL mistake. The error was created HERE, not inherited "
        "from previous steps. If this step were fixed, downstream failures would likely be prevented.\n"
        "- **PROPAGATING_ERROR**: This step propagated an error from an EARLIER step. The mistake already existed "
        "before this step, and while this step failed to catch it, it did not originate the error.\n"
        "- **NEITHER**: This step is correct, or the error was introduced in later steps.\n\n"
        "**CRITICISM guide**:\n"
        "- For ORIGINATING_ERROR or PROPAGATING_ERROR: explain (1) how this step caused or forwarded "
        "the problem, and (2) what a correct version of this step would look like.\n"
        "- For NEITHER: briefly confirm why the step is correct in this context.\n\n"
        "**Output Format**:\n"
        "Respond with ONLY a valid JSON object — no preamble, no markdown fences:\n"
        f"{EXAMPLE_OUTPUT}"
        # "{{\n"
        # "  \"attribution\": \"ORIGINATING_ERROR\" | \"PROPAGATING_ERROR\" | \"NEITHER\",\n"
        # "  \"criticism\": \"Your explanation here...\"\n"
        # "}}"
    )
    # import pdb; pdb.set_trace()

    system_desc = data.get("system_prompt", {})
    system_text = ''
    if system_desc:
        agents_description = SEP.join(
            f" * {k}: {v.strip('## Your role').strip()}"  for k, v in system_desc.items()
        )
        system_text = f"Agentic System Description\n{agents_description}\n\n"

    logs = []
    for idx, entry in enumerate(chat_history):
        task = TASK_TEMPLATE
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history (trajectory) "
            "generated during the resolution of a complex problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            f"{system_text}"
            f"Here is the conversation:\n\n{chat_content}\n\n"
            f"Task: Analyze how STEP {idx} (performed by {entry.get('role', 'Unknown Agent')}) contributed to the failure of the final output.\n\n"
            f"{task}"
        )
        system_message = (
            "You are a helpful assistant skilled in analyzing conversations. "
            "You always respond in valid JSON format."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        logs.append({
            'filename': None,
            'step_idx': idx,
            'messages': messages,
        })

    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step['role'],
                'content': step['content'],
            }
            for step_idx, step in enumerate(chat_history)
        ],
        'logs': logs,
    }