import os
import json
import re
from tqdm import tqdm
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
# STEP-BY-STEP FULL CONTEXT
# ============================================================

def get_prompt_step_by_step_full(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    full_chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}"
        for i, entry in enumerate(chat_history)
    ])

    system_desc = data.get("system_prompt", {})
    system_text = ''
    if system_desc:
        agents_description = SEP.join(
            f" * {k}: {v.strip('## Your role').strip()}" for k, v in system_desc.items()
        )
        system_text = f"Agentic System Description\n{agents_description}\n\n"

    logs = []
    for idx, entry in enumerate(chat_history):
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            f"{system_text}"
            f"Here is the conversation:\n\n{full_chat_content}\n\n"
            f"Task: Determine whether STEP {idx} (performed by {entry.get('role')}) is a decisive error step. "
            "A decisive error step is a mistake step where, if corrected, with all following steps are adjusted accordingly, the system would succeed. "
            "Focus only on errors that critically derail the process, rather than minor imperfections.\n\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"is_decisive\": boolean (true or false)\n"
            "2. \"reason\": string (explanation for your judgment)"
        )
        system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

        logs.append({
            'filename': None,
            'step_idx': idx,
            'messages': [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
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
# STEP-BY-STEP PARTIAL CONTEXT
# ============================================================

def get_prompt_step_by_step_partial(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    system_desc = data.get("system_prompt", {})
    system_text = ''
    if system_desc:
        SEP = "\n\n---\n\n"
        agents_description = SEP.join(
            f" * {k}: {v.strip('## Your role').strip()}" for k, v in system_desc.items()
        )
        system_text = f"Agentic System Description\n{agents_description}\n\n"

    logs = []
    current_context = ""
    for idx, entry in enumerate(chat_history):
        agent_name = entry.get('role', 'Unknown Agent')
        current_context += f"Step {idx} - {agent_name}: {entry.get('content', '')}\n"

        prompt = (
            "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            f"{system_text}"
            f"Here is the conversation history up to the current step:\n{current_context}\n"
            f"The most recent step ({idx}) was performed by '{agent_name}'.\n"
            f"Task: Determine whether this most recent step (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
            "Avoid being overly critical — focus only on errors that clearly derail the process, rather than minor imperfections.\n\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"is_decisive\": boolean (true or false)\n"
            "2. \"reason\": string (explanation for your judgment)"
        )
        system_message = "You are a precise step-by-step conversation evaluator. You always respond in valid JSON format."

        logs.append({
            'filename': None,
            'step_idx': idx,
            'messages': [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
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