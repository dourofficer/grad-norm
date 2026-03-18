"""
Magentic-One LLM Input Dependency Analyzer (v2)
=================================================
 
Models exactly what is fed to each LLM call (or non-LLM processing step),
not just what sits in _chat_history.
 
Key differences from v1 (which tracked _chat_history membership):
 
  1. The Orchestrator's initial_plan is actually TWO sequential LLM calls
     (closed-book facts, then plan) on a *temporary* planning conversation
     that is separate from _chat_history.
 
  2. The Orchestrator's new_plan (replan) is also TWO sequential LLM calls
     (update facts, then update plan) on a copy of _chat_history + appended
     prompts. The second call sees the first call's output.
 
  3. WebSurfer strips images from history and appends a FRESH page-state
     message (screenshot + SoM + interactive elements + OCR). This implicit
     input is not a step in the log.
 
  4. FileSurfer splits history into [all_but_last] + [context_msg, last_msg].
     The context_msg (current file/page state) is an implicit input.
 
  5. Executor does NOT call an LLM. It scans the last N messages for code
     blocks. Only the message containing a code block is a true input.
 
  6. Coder sends system_messages + full _chat_history directly to the LLM.
     No transformation.
 
  7. Ledger updates append a ledger_prompt (template filled with task, team,
     names) as an additional message beyond _chat_history.
 
  8. Instructions and next_speaker_logs are NOT LLM calls — they are
     extracted from the preceding ledger_update output.
"""
 
from __future__ import annotations
 
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
 
 
# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------
 
class StepKind(Enum):
    TASK               = auto()  # Human input — no LLM
    INITIAL_PLAN       = auto()  # Two LLM calls: closed-book → plan
    LEDGER_UPDATE      = auto()  # One LLM call: system + history + ledger_prompt
    NEXT_SPEAKER_LOG   = auto()  # Not an LLM call — derived from ledger output
    INSTRUCTION        = auto()  # Not an LLM call — extracted from ledger JSON
    AGENT_RESPONSE     = auto()  # LLM call (or code scan for Executor)
    REPLAN_LOG         = auto()  # Not an LLM call — log line
    NEW_PLAN           = auto()  # Two LLM calls: update_facts → update_plan
    ORCHESTRATOR_OTHER = auto()  # Catch-all internal
 
 
@dataclass
class ImplicitInput:
    """An input to an LLM call that does not correspond to any step index."""
    label: str        # e.g. "system_prompt", "browser_state", "ledger_prompt"
    description: str  # Human-readable description
 
 
@dataclass
class StepInfo:
    """Full metadata for one step in the history."""
    idx: int
    kind: StepKind
    role: str
    agent: str                           # "Orchestrator", "WebSurfer", etc.
    is_llm_call: bool                    # Does this step involve an LLM call?
    step_inputs: list[int]               # Step indices actually fed to the LLM
    implicit_inputs: list[ImplicitInput] # Non-step inputs (prompts, browser state…)
    notes: str = ""                      # Extra explanation
 
 
# System prompt implicit input (shared by most LLM calls)
SYS_PROMPT = ImplicitInput("system_prompt", "Agent's system message(s)")
LEDGER_PROMPT = ImplicitInput("ledger_prompt",
    "Template: task + team description + agent names → structured JSON request")
CLOSED_BOOK_PROMPT = ImplicitInput("closed_book_prompt",
    "Template asking LLM to categorize facts before planning")
PLAN_PROMPT = ImplicitInput("plan_prompt",
    "Template asking LLM to create a bullet-point plan given team composition")
UPDATE_FACTS_PROMPT = ImplicitInput("update_facts_prompt",
    "Template asking LLM to revise the fact sheet")
UPDATE_PLAN_PROMPT = ImplicitInput("update_plan_prompt",
    "Template asking LLM to create a new plan given failures")
SYNTHESIZE_PROMPT = ImplicitInput("synthesize_prompt",
    "Template combining task + team + facts + plan into broadcast message")
BROWSER_STATE = ImplicitInput("browser_page_state",
    "Fresh SoM screenshot + interactive elements list + OCR text + tool list")
FILE_BROWSER_STATE = ImplicitInput("file_browser_state",
    "Current file path, page title, viewport position")
CODE_EXECUTION_ENV = ImplicitInput("code_execution_env",
    "Docker container executing extracted code block")
FINAL_ANSWER_PROMPT = ImplicitInput("final_answer_prompt",
    "Template asking LLM to produce a final answer summary")
 
 
# ---------------------------------------------------------------------------
# Step classification
# ---------------------------------------------------------------------------
 
def classify_step(entry: dict[str, str]) -> tuple[StepKind, str]:
    """
    Returns (StepKind, agent_name).
    """
    role = entry["role"]
    content = entry.get("content", "")
 
    if role == "human":
        return StepKind.TASK, "Human"
 
    if role.startswith("Orchestrator (-> "):
        target = role[len("Orchestrator (-> "):-1]
        return StepKind.INSTRUCTION, f"Orchestrator→{target}"
 
    if role == "Orchestrator (thought)":
        stripped = content.strip()
        if stripped.startswith("Initial plan:"):
            return StepKind.INITIAL_PLAN, "Orchestrator"
        if stripped.startswith("Updated Ledger:"):
            return StepKind.LEDGER_UPDATE, "Orchestrator"
        if stripped.startswith("Next speaker"):
            return StepKind.NEXT_SPEAKER_LOG, "Orchestrator"
        if stripped.startswith("New plan:"):
            return StepKind.NEW_PLAN, "Orchestrator"
        if "Stalled" in content or "Replan" in content:
            return StepKind.REPLAN_LOG, "Orchestrator"
        return StepKind.ORCHESTRATOR_OTHER, "Orchestrator"
 
    # Worker agent — extract name from role
    return StepKind.AGENT_RESPONSE, role
 
 
def _find_preceding_kind(steps: list[StepInfo], idx: int, kind: StepKind) -> int | None:
    for i in range(idx - 1, -1, -1):
        if steps[i].kind == kind:
            return i
    return None
 
 
def _detect_agent_type(role: str) -> str:
    """Heuristic: map role string to agent class for LLM-input modeling."""
    r = role.lower()
    if "websurfer" in r or "web surfer" in r:
        return "WebSurfer"
    if "filesurfer" in r or "file surfer" in r or "file_surfer" in r:
        return "FileSurfer"
    if "executor" in r or "computerterminal" in r or "computer terminal" in r:
        return "Executor"
    if "coder" in r or "assistant" in r:
        return "Coder"
    if "userproxy" in r or "user proxy" in r or "user" in r:
        return "UserProxy"
    return "Unknown"
 
 
# ---------------------------------------------------------------------------
# Broadcast types (added to agents' _chat_history)
# ---------------------------------------------------------------------------
 
BROADCAST_KINDS = frozenset({
    StepKind.TASK,
    StepKind.INITIAL_PLAN,
    StepKind.NEW_PLAN,
    StepKind.INSTRUCTION,
    StepKind.AGENT_RESPONSE,
})
 
 
# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
 
def derive_llm_inputs(history: list[dict[str, str]]) -> list[StepInfo]:
    """
    For every step, compute exactly which prior steps are fed to the LLM call
    that produces that step, plus any implicit (non-step) inputs.
 
    Returns a list of StepInfo, one per history entry.
    """
    n = len(history)
    steps: list[StepInfo] = []
 
    # Simulated broadcast histories (indices of steps that have been broadcast)
    orch_history: list[int] = []   # Orchestrator's _chat_history
    worker_history: list[int] = [] # All workers' _chat_history (shared topic)
    task_idx: int | None = None
 
    for idx in range(n):
        entry = history[idx]
        kind, agent = classify_step(entry)
 
        info = StepInfo(
            idx=idx, kind=kind, role=entry["role"], agent=agent,
            is_llm_call=False, step_inputs=[], implicit_inputs=[]
        )
 
        # ── TASK ──────────────────────────────────────────────────────
        if kind == StepKind.TASK:
            info.is_llm_call = False
            info.step_inputs = []
            info.notes = "Human input, no LLM call"
            task_idx = idx
            orch_history.append(idx)
            worker_history.append(idx)
 
        # ── INITIAL_PLAN ──────────────────────────────────────────────
        # Internally this is TWO LLM calls on a temporary planning_conversation:
        #   Call 1: system + planning_conv + [closed_book_prompt] → facts
        #   Call 2: system + planning_conv + [closed_book_prompt, A(facts), plan_prompt] → plan
        # planning_conversation starts as a shallow copy of _chat_history (= [task] at this point).
        # The output (synthesized prompt) is then broadcast.
        elif kind == StepKind.INITIAL_PLAN:
            info.is_llm_call = True
            info.step_inputs = sorted(set(orch_history))  # just [task] at this point
            info.implicit_inputs = [SYS_PROMPT, CLOSED_BOOK_PROMPT, PLAN_PROMPT, SYNTHESIZE_PROMPT]
            info.notes = ("Two sequential LLM calls: (1) system + [task] + closed_book_prompt → facts, "
                          "(2) system + [task, closed_book, A(facts)] + plan_prompt → plan. "
                          "Then synthesize_prompt combines task+team+facts+plan into broadcast.")
            orch_history.append(idx)
            worker_history.append(idx)
 
        # ── LEDGER_UPDATE ─────────────────────────────────────────────
        # LLM call: system_messages + self._chat_history + [ledger_prompt]
        elif kind == StepKind.LEDGER_UPDATE:
            info.is_llm_call = True
            info.step_inputs = sorted(set(orch_history))
            info.implicit_inputs = [SYS_PROMPT, LEDGER_PROMPT]
            info.notes = "LLM call: system + orch_chat_history + ledger_prompt(task, team, names)"
 
        # ── NEXT_SPEAKER_LOG ──────────────────────────────────────────
        # Not an LLM call — just a log of the decision from the ledger.
        elif kind == StepKind.NEXT_SPEAKER_LOG:
            info.is_llm_call = False
            prev = _find_preceding_kind(steps, idx, StepKind.LEDGER_UPDATE)
            info.step_inputs = [prev] if prev is not None else []
            info.notes = "Not an LLM call. Logged from preceding ledger_update output."
 
        # ── INSTRUCTION ───────────────────────────────────────────────
        # Not an LLM call — the instruction text is extracted from the
        # preceding ledger_update's JSON "instruction_or_question.answer".
        elif kind == StepKind.INSTRUCTION:
            info.is_llm_call = False
            prev = _find_preceding_kind(steps, idx, StepKind.LEDGER_UPDATE)
            info.step_inputs = [prev] if prev is not None else []
            info.notes = "Not an LLM call. Text extracted from preceding ledger_update JSON."
            orch_history.append(idx)
            worker_history.append(idx)
 
        # ── AGENT_RESPONSE ────────────────────────────────────────────
        # This is where agent-specific LLM input logic matters.
        elif kind == StepKind.AGENT_RESPONSE:
            agent_type = _detect_agent_type(entry["role"])
            history_steps = sorted(set(worker_history))
 
            if agent_type == "WebSurfer":
                # __generate_reply:
                #   1. Clones _chat_history, stripping images to text-only
                #   2. Appends fresh UserMessage with [text_prompt + SoM screenshot]
                #      where text_prompt = interactive elements + tool names
                #   3. LLM call: text_clone(chat_history) + [page_state_msg], tools=...
                info.is_llm_call = True
                info.step_inputs = history_steps
                info.implicit_inputs = [BROWSER_STATE]
                info.notes = ("WebSurfer LLM: text_clone(chat_history) + [fresh page_state with "
                              "SoM screenshot, interactive element list, OCR, tool list]. "
                              "Images from older messages are stripped to text.")
 
            elif agent_type == "FileSurfer":
                # _generate_reply:
                #   history = chat_history[:-1]
                #   last_msg = chat_history[-1]  (the instruction)
                #   context_msg = UserMessage("browser open to '{title}' at '{address}'...")
                #   task_msg = UserMessage(last_msg.content)
                #   LLM call: system + history[:-1] + [context_msg, task_msg], tools=...
                info.is_llm_call = True
                info.step_inputs = history_steps
                info.implicit_inputs = [SYS_PROMPT, FILE_BROWSER_STATE]
                info.notes = ("FileSurfer LLM: system + chat_history[:-1] + "
                              "[file_browser_context_msg, last_msg_as_task], tools=[file tools]. "
                              "All history steps are used but last is separated as 'task'.")
 
            elif agent_type == "Executor":
                # _generate_reply: NO LLM call.
                #   Scans last N messages for ```python or ```sh code blocks.
                #   Extracts first found, executes in Docker, returns output.
                info.is_llm_call = False
                # Find the actual code-bearing message: scan backwards
                code_source = None
                for i in reversed(history_steps):
                    if i < idx and steps[i].kind == StepKind.AGENT_RESPONSE:
                        # The Coder's response is typically right before
                        code_source = i
                        break
                    elif i < idx and steps[i].kind == StepKind.INSTRUCTION:
                        # Could also contain code
                        code_source = i
                        break
                if code_source is not None:
                    info.step_inputs = [code_source]
                else:
                    # Fallback: last few messages
                    info.step_inputs = history_steps[-5:] if len(history_steps) >= 5 else history_steps
                info.implicit_inputs = [CODE_EXECUTION_ENV]
                info.notes = ("Executor: NO LLM call. Scans last ~5 UserMessages in "
                              "chat_history for code blocks, executes first found in Docker.")
 
            elif agent_type == "Coder":
                # _generate_reply:
                #   LLM call: system_messages + self._chat_history
                info.is_llm_call = True
                info.step_inputs = history_steps
                info.implicit_inputs = [SYS_PROMPT]
                info.notes = "Coder LLM: system_messages + full chat_history. No transformation."
 
            elif agent_type == "UserProxy":
                info.is_llm_call = False
                info.step_inputs = []
                info.notes = "UserProxy: human terminal input, no LLM call."
 
            else:
                # Unknown agent — assume full history like Coder
                info.is_llm_call = True
                info.step_inputs = history_steps
                info.implicit_inputs = [SYS_PROMPT]
                info.notes = f"Unknown agent type '{agent_type}': assuming full history."
 
            orch_history.append(idx)
            worker_history.append(idx)
 
        # ── REPLAN_LOG ────────────────────────────────────────────────
        elif kind == StepKind.REPLAN_LOG:
            info.is_llm_call = False
            prev = _find_preceding_kind(steps, idx, StepKind.LEDGER_UPDATE)
            info.step_inputs = [prev] if prev is not None else []
            info.notes = "Not an LLM call. Triggered when stall_counter > threshold."
 
        # ── NEW_PLAN (replan) ─────────────────────────────────────────
        # Internally TWO LLM calls on a copy of _chat_history:
        #   Call 1: system + chat_history + [update_facts_prompt(task, old_facts)] → new_facts
        #   Call 2: system + chat_history + [update_facts_prompt, A(new_facts), update_plan_prompt(team)] → new_plan
        # Then: reset all histories, broadcast new synthesized prompt.
        elif kind == StepKind.NEW_PLAN:
            info.is_llm_call = True
            info.step_inputs = sorted(set(orch_history))
            info.implicit_inputs = [SYS_PROMPT, UPDATE_FACTS_PROMPT, UPDATE_PLAN_PROMPT, SYNTHESIZE_PROMPT]
            info.notes = ("Two sequential LLM calls: "
                          "(1) system + orch_history + update_facts_prompt → new facts, "
                          "(2) system + orch_history + [update_facts, A(new_facts), update_plan] → new plan. "
                          "Then ResetMessage clears all agent histories. "
                          "New synthesized prompt broadcast to all.")
 
            # ── REPLAN RESET ──
            if task_idx is not None:
                orch_history = [task_idx, idx]
            else:
                orch_history = [idx]
            worker_history = [idx]
 
        # ── ORCHESTRATOR_OTHER ────────────────────────────────────────
        else:
            info.is_llm_call = False
            info.step_inputs = sorted(set(orch_history))
            info.notes = "Unclassified orchestrator internal step."
 
        steps.append(info)
 
    return steps
 
 
# ---------------------------------------------------------------------------
# Convenience: extract just the {idx: [deps]} dict
# ---------------------------------------------------------------------------
 
def get_dependency_dict(steps: list[StepInfo]) -> dict[int, list[int]]:
    return {s.idx: s.step_inputs for s in steps}