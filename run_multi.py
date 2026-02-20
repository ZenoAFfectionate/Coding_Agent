"""Multi-Agent Coding System — orchestrator + specialized worker agents.

Architecture:
  - Orchestrator (master): Works iteratively — each round it dispatches a
    single specialist worker, observes the result, and decides what to do
    next.  Earlier worker results are forwarded as context to later workers.
    The orchestrator itself has NO tools — it only reasons and delegates.
  - Worker agents (subordinates): Each is a ReActAgent equipped with a
    task-specific system prompt and a curated subset of tools.  Workers return
    a concise result to the orchestrator; intermediate tool-call details stay
    hidden inside their own context.

Worker types:
  review   — code review     (file, code_search, linter, git)
  test     — test generation  (file, code_exec, code_search, test_runner)
  optimize — optimization     (file, code_exec, code_search, profiler, linter)
  debug    — debugging        (file, code_exec, code_search, test_runner, git, linter)
"""

import argparse
import atexit
import glob
import json
import os
import re
import select
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from code.core.agent import Agent
from code.core.llm import HelloAgentsLLM
from code.core.config import Config
from code.agents.react_agent import ReActAgent
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.code_search_tool import CodeSearchTool
from code.tools.builtin.test_runner_tool import TestRunnerTool
from code.tools.builtin.git_tool import GitTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool

PROMPTS_DIR = PROJECT_ROOT / "prompts"


# ---------------------------------------------------------------------------
# Shared Blackboard Memory
# ---------------------------------------------------------------------------

class Blackboard:
    """Structured workspace state shared across the orchestrator and workers."""

    def __init__(self, user_request: str):
        self.user_request = user_request
        self.files_examined: list[str] = []
        self.findings: list[dict] = []   # {"source", "round", "summary"}
        self.errors: list[dict] = []     # {"source", "round", "message"}
        self.current_plan: str = ""

    def add_finding(self, source: str, round_num: int, summary: str) -> None:
        self.findings.append({
            "source": source,
            "round": round_num,
            "summary": summary,
        })

    def add_error(self, source: str, round_num: int, message: str) -> None:
        self.errors.append({
            "source": source,
            "round": round_num,
            "message": message,
        })

    def serialize(self) -> str:
        """Render the blackboard as a concise text block for injection into prompts."""
        parts: list[str] = []
        parts.append(f"User request: {self.user_request}")
        if self.current_plan:
            parts.append(f"\nPlan: {self.current_plan}")
        if self.files_examined:
            parts.append(f"\nFiles examined: {', '.join(self.files_examined)}")
        if self.findings:
            parts.append("\nFindings:")
            for f in self.findings:
                parts.append(f"  - [{f['source']} R{f['round']}] {f['summary']}")
        if self.errors:
            parts.append("\nErrors:")
            for e in self.errors:
                parts.append(f"  - [{e['source']} R{e['round']}] {e['message']}")
        if not self.findings and not self.errors:
            parts.append("\n(no findings yet)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt(name: str) -> str:
    """Load a prompt file from the prompts/ directory by stem name."""
    path = PROMPTS_DIR / f"{name}.prompt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Worker agent factory
# ---------------------------------------------------------------------------

# Maps worker type -> (prompt file stem, tool builder function)
# Each tool builder receives ``workspace`` and returns a ToolRegistry.

def _review_tools(workspace: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(FileTool(workspace=workspace))
    registry.register_tool(CodeSearchTool(workspace=workspace))
    registry.register_tool(LinterTool(workspace=workspace, timeout=30))
    registry.register_tool(GitTool(repo_path=workspace))
    return registry


def _test_tools(workspace: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(FileTool(workspace=workspace))
    registry.register_tool(CodeExecutionTool(workspace=workspace, timeout=30))
    registry.register_tool(CodeSearchTool(workspace=workspace))
    registry.register_tool(TestRunnerTool(project_path=workspace, timeout=120))
    return registry


def _optimize_tools(workspace: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(FileTool(workspace=workspace))
    registry.register_tool(CodeExecutionTool(workspace=workspace, timeout=30))
    registry.register_tool(CodeSearchTool(workspace=workspace))
    registry.register_tool(ProfilerTool(workspace=workspace, timeout=60))
    registry.register_tool(LinterTool(workspace=workspace, timeout=30))
    return registry


def _debug_tools(workspace: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(FileTool(workspace=workspace))
    registry.register_tool(CodeExecutionTool(workspace=workspace, timeout=30))
    registry.register_tool(CodeSearchTool(workspace=workspace))
    registry.register_tool(TestRunnerTool(project_path=workspace, timeout=120))
    registry.register_tool(GitTool(repo_path=workspace))
    registry.register_tool(LinterTool(workspace=workspace, timeout=30))
    return registry


WORKER_SPECS = {
    "review": {
        "prompt": "code_review",
        "tools_factory": _review_tools,
        "description": "Code review specialist: analyzes code quality, security, design, and correctness.",
    },
    "test": {
        "prompt": "test_generation",
        "tools_factory": _test_tools,
        "description": "Test generation specialist: writes and runs comprehensive test suites.",
    },
    "optimize": {
        "prompt": "optimization",
        "tools_factory": _optimize_tools,
        "description": "Optimization specialist: profiles code and applies targeted performance improvements.",
    },
    "debug": {
        "prompt": "debug",
        "tools_factory": _debug_tools,
        "description": "Debug specialist: reproduces, diagnoses, and fixes bugs systematically.",
    },
}


def build_worker(
    worker_type: str,
    workspace: str,
    llm: HelloAgentsLLM,
    config: Config,
    max_steps: int = 20,
) -> ReActAgent:
    """Create a specialized worker ReActAgent.

    Args:
        worker_type: One of "review", "test", "optimize", "debug".
        workspace: Root directory the agent operates in.
        llm: Shared LLM instance (same instance used by orchestrator and all workers).
        config: Shared Config object.
        max_steps: Max tool-calling rounds for this worker.

    Returns:
        A ready-to-use ReActAgent with task-specific prompt and tools.
    """
    spec = WORKER_SPECS[worker_type]

    # Combine the base system prompt with the task-specific prompt
    base_system = _load_prompt("system") or "You are an expert Python coding assistant."
    task_prompt = _load_prompt(spec["prompt"])
    system_prompt = base_system
    if task_prompt:
        system_prompt += f"\n\n<task-instructions>\n{task_prompt}\n</task-instructions>"

    registry = spec["tools_factory"](workspace)

    return ReActAgent(
        name=f"Worker_{worker_type}",
        llm=llm,
        system_prompt=system_prompt,
        tool_registry=registry,
        max_steps=max_steps,
        config=config,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Orchestrator of a multi-agent coding system. You work **one step at a time**: \
analyze the situation, decide the single best next action, observe the result, and repeat \
until the user's request is fully addressed.

## Available Workers

{worker_descriptions}

## How You Work

Each turn you MUST respond with exactly ONE of the following JSON objects:

### Option A — Dispatch a worker

```json
{{"worker": "<worker_type>", "task": "<self-contained instruction>"}}
```

- `worker` must be one of: {worker_types}.
- `task` must be **completely self-contained** — include every detail the worker needs \
(file paths, function names, relevant findings from earlier workers, etc.). Workers have \
NO memory of previous rounds; they only see the task string you give them.

### Option B — Provide the final answer

```json
{{"done": true, "answer": "<your final answer to the user>"}}
```

Use this when you have gathered enough information to fully answer the user's request. \
Synthesize all worker results into a coherent, actionable response.

## Guidelines

- **Think step-by-step.** After each worker result, reason about what you still need \
before deciding the next action.
- **Forward context.** When dispatching a worker, include any relevant findings from \
earlier workers in the task description so the new worker can build on prior results.
- **Be efficient.** Do not dispatch workers unnecessarily. If you can answer directly \
without any worker, output the done JSON immediately.
- **Stay focused.** Each dispatch should target one clear objective.
- **Blackboard available.** After each worker result, a `<blackboard>` section shows accumulated findings from all workers so far. Use this to avoid re-dispatching work that has already been done."""


ORCHESTRATOR_FC_SYSTEM_PROMPT = """\
You are the Orchestrator of a multi-agent coding system. You work **one step at a time**: \
analyze the situation, decide the single best next action, observe the result, and repeat \
until the user's request is fully addressed.

## Available Workers

{worker_descriptions}

## Guidelines

- **Think step-by-step.** After each worker result, reason about what you still need \
before deciding the next action.
- **Forward context.** When dispatching a worker, include any relevant findings from \
earlier workers in the task description so the new worker can build on prior results. \
Workers have NO memory of previous rounds; they only see the task you give them.
- **Be efficient.** Do not dispatch workers unnecessarily. If you can answer directly, \
call finish() immediately.
- **Stay focused.** Each dispatch should target one clear objective.

Use dispatch_worker() to send tasks to specialists, or finish() when you have a complete answer."""


def _build_orchestrator_tools() -> list[dict]:
    """Build the OpenAI-compatible function-calling tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "dispatch_worker",
                "description": (
                    "Dispatch a task to a specialist worker agent. "
                    "The task string must be completely self-contained — "
                    "include all context the worker needs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker": {
                            "type": "string",
                            "enum": list(WORKER_SPECS.keys()),
                            "description": "The specialist worker type.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained task instruction including "
                                "file paths, prior findings, and all context."
                            ),
                        },
                    },
                    "required": ["worker", "task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Provide the final synthesized answer to the user. "
                    "Call this when you have enough information to fully "
                    "address the request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The complete answer for the user.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


class OrchestratorAgent(Agent):
    """Master agent that delegates to specialized workers.

    Inherits from ``Agent`` to reuse context budget management,
    token counting, trajectory tracking, and debug printing.

    Supports two communication modes with the LLM:
    1. **Function calling** (default) — uses OpenAI-compatible tool schemas
       ``dispatch_worker`` and ``finish`` for structured, reliable output.
    2. **Text fallback** — if the provider does not support function calling,
       automatically falls back to JSON-in-text parsing.

    Also includes:
    - Worker result truncation to cap per-result size.
    - Error detection for worker failures, surfaced explicitly to the LLM
      so it can decide to retry or adjust.
    """

    # Patterns that indicate a worker result is an error.
    _WORKER_ERROR_PATTERNS = [
        "failed with error:",
        "unable to complete the task within the allowed",
        "traceback (most recent call last)",
    ]

    def __init__(
        self,
        workspace: str,
        config: Config,
        max_worker_steps: int = 20,
        max_orchestrator_rounds: int = 8,
        max_result_chars: int = 4000,
        context_max_tokens: int = 0,
        enable_summarization: bool = True,
        enable_reflection: bool = True,
        debug: bool = True,
    ):
        # Build LLM first so we can pass it to Agent.__init__
        llm = HelloAgentsLLM(
            model=config.default_model,
            provider=config.default_provider,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Ensure config.debug matches the explicit debug flag
        config.debug = debug

        super().__init__(
            name="Orchestrator",
            llm=llm,
            config=config,
            context_max_tokens=context_max_tokens,
        )

        # Override Agent's list[Message] history with dict-based history
        # (the orchestrator stores {"role": ..., "content": ...} dicts)
        self._history: list[dict] = []

        self.workspace = str(Path(workspace).resolve())
        self.max_worker_steps = max_worker_steps
        self.max_orchestrator_rounds = max_orchestrator_rounds
        self.max_result_chars = max_result_chars
        self.enable_summarization = enable_summarization
        self.enable_reflection = enable_reflection

        # Function calling: enabled by default, auto-disabled on first failure
        self._use_function_calling = True
        self._orchestrator_tools = _build_orchestrator_tools()

        # Build both prompt variants (FC and text-mode)
        worker_descs = "\n".join(
            f"- **{wtype}**: {spec['description']}"
            for wtype, spec in WORKER_SPECS.items()
        )
        worker_types = ", ".join(WORKER_SPECS.keys())

        fc_template = _load_prompt("orchestrator_fc") or ORCHESTRATOR_FC_SYSTEM_PROMPT
        self._fc_system_prompt = fc_template.format(
            worker_descriptions=worker_descs,
        )

        text_template = _load_prompt("orchestrator") or ORCHESTRATOR_SYSTEM_PROMPT
        self._text_system_prompt = text_template.format(
            worker_descriptions=worker_descs,
            worker_types=worker_types,
        )

        # Summarization prompt
        self._summarize_prompt = _load_prompt("summarize_result")

        # Reflection prompt
        self._reflection_prompt = _load_prompt("reflection_orchestrator")

    # ------------------------------------------------------------------ #
    #  system_prompt property (overrides Agent's plain attribute)
    # ------------------------------------------------------------------ #

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt (depends on current mode)."""
        return self._fc_system_prompt if self._use_function_calling else self._text_system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        # Accept writes from Agent.__init__ without error; the orchestrator
        # manages its own prompt variants (_fc_system_prompt / _text_system_prompt).
        pass

    # ------------------------------------------------------------------ #
    #  Worker result truncation & error detection
    # ------------------------------------------------------------------ #

    def _truncate_result(self, result: str) -> str:
        """Truncate a worker result to *max_result_chars* (middle-cut)."""
        if len(result) <= self.max_result_chars:
            return result
        half = self.max_result_chars // 2
        omitted = len(result) - self.max_result_chars
        return (
            result[:half]
            + f"\n\n... [{omitted} chars truncated] ...\n\n"
            + result[-half:]
        )

    def _detect_worker_error(self, result: str) -> bool:
        """Return True if the worker result looks like an error."""
        lower = result.lower()
        return any(p in lower for p in self._WORKER_ERROR_PATTERNS)

    # ------------------------------------------------------------------ #
    #  Worker result summarization
    # ------------------------------------------------------------------ #

    def _summarize_result(self, worker_type: str, task: str, result: str) -> str:
        """Summarize a long worker result via LLM; short results pass through."""
        if not self.enable_summarization or len(result) <= self.max_result_chars:
            return result
        if not self._summarize_prompt:
            return self._truncate_result(result)
        try:
            summary = self.llm.invoke([
                {"role": "system", "content": self._summarize_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Worker: {worker_type}\n"
                        f"Task: {task}\n\n"
                        f"Full result:\n{result}"
                    ),
                },
            ])
            if summary:
                return summary
        except Exception as e:
            self._print(f"  [Summarization failed: {e}] — falling back to truncation.")
        return self._truncate_result(result)

    # ------------------------------------------------------------------ #
    #  Reflection / self-verification
    # ------------------------------------------------------------------ #

    def _reflect_on_answer(self, user_request: str, blackboard: Blackboard, answer: str) -> str:
        """Run a quality-check LLM call on the proposed final answer."""
        if not self._reflection_prompt:
            return answer
        prompt = self._reflection_prompt.format(
            user_request=user_request,
            blackboard=blackboard.serialize(),
            proposed_answer=answer,
        )
        try:
            reflection = self.llm.invoke([
                {"role": "system", "content": "You are a quality reviewer."},
                {"role": "user", "content": prompt},
            ])
            if reflection and reflection.strip() != "APPROVED":
                self._print("\n[Reflection] Answer revised by reflection step.", level="info")
                return reflection
            self._print("\n[Reflection] Answer approved.", level="info")
        except Exception as e:
            self._print(f"\n[Reflection failed: {e}] — returning original answer.", level="info")
        return answer

    # ------------------------------------------------------------------ #
    #  Run a single worker
    # ------------------------------------------------------------------ #

    def _run_worker(self, worker_type: str, task: str) -> tuple[str, bool]:
        """Spawn a worker, run the task, return ``(result, is_error)``."""
        if worker_type not in WORKER_SPECS:
            return f"Error: unknown worker type '{worker_type}'", True

        self._print(f"\n{'='*50}")
        self._print(f"  [Dispatching] Worker: {worker_type}")
        self._print(f"  [Task] {task[:200]}{'...' if len(task) > 200 else ''}")
        self._print(f"{'='*50}")

        worker = build_worker(
            worker_type=worker_type,
            workspace=self.workspace,
            llm=self.llm,
            config=self.config,
            max_steps=self.max_worker_steps,
        )

        try:
            result = worker.run(task)
        except Exception as e:
            result = f"Worker '{worker_type}' failed with error: {e}"

        self._print(f"\n  [Worker {worker_type} done] Result length: {len(result)} chars")
        is_error = self._detect_worker_error(result)
        if is_error:
            self._print(f"  [Worker {worker_type}] ⚠ Error detected in result.", level="info")
        return result, is_error

    # ------------------------------------------------------------------ #
    #  Orchestrator LLM invocation (unified dispatcher)
    # ------------------------------------------------------------------ #

    def _invoke_orchestrator(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Call the orchestrator LLM and interpret its decision.

        Returns:
            ``(action, data, msgs_to_append)`` where:
            - *action* is ``"dispatch"`` or ``"finish"``.
            - *data* is ``{"worker": str, "task": str, "tool_call_id": str}``
              for dispatch, or a plain string for finish.
            - *msgs_to_append* is a list of message dicts to extend the
              conversation with (assistant response + optional forced prompt).
        """
        if self._use_function_calling:
            try:
                return self._invoke_fc(messages, force_finish)
            except Exception as exc:
                self._print(
                    f"\n  [Orchestrator] Function calling failed ({exc}), "
                    "switching to text mode for the rest of this session.",
                    level="info",
                )
                self._use_function_calling = False
        return self._invoke_text(messages, force_finish)

    # ---- Function-calling path ----------------------------------------

    def _invoke_fc(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Function-calling path: returns ``(action, data, msgs)``."""
        tool_choice: str | dict = "auto"
        if force_finish:
            tool_choice = {
                "type": "function",
                "function": {"name": "finish"},
            }

        response = self.llm.invoke_with_tools(
            messages,
            tools=self._orchestrator_tools,
            tool_choice=tool_choice,
        )
        msg = response.choices[0].message

        # Build the assistant dict for conversation history
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Malformed arguments — treat as finish with raw text
                return "finish", tc.function.arguments, [assistant_msg]

            if tc.function.name == "finish":
                return "finish", args.get("answer", ""), [assistant_msg]

            if tc.function.name == "dispatch_worker":
                data = {
                    "worker": args.get("worker", ""),
                    "task": args.get("task", ""),
                    "tool_call_id": tc.id,
                }
                return "dispatch", data, [assistant_msg]

        # No tool call — treat raw text as the final answer
        return "finish", msg.content or "", [assistant_msg]

    # ---- Text-mode fallback path --------------------------------------

    def _invoke_text(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Text-based fallback: returns ``(action, data, msgs)``."""
        call_messages = list(messages)
        extra_msgs: list[dict] = []
        if force_finish:
            force_msg = {
                "role": "user",
                "content": (
                    "You have reached the maximum number of rounds. "
                    "Provide your final answer now using: "
                    '{"done": true, "answer": "..."}'
                ),
            }
            call_messages.append(force_msg)
            extra_msgs.append(force_msg)

        response_text = self.llm.invoke(call_messages) or ""
        assistant_msg = {"role": "assistant", "content": response_text}
        self._print(f"\n[Orchestrator] Response:\n{response_text}")

        dispatch = self._parse_single_dispatch(response_text)
        if dispatch is None:
            answer = self._extract_done_answer(response_text)
            return "finish", answer, extra_msgs + [assistant_msg]

        data = {
            "worker": dispatch.get("worker", ""),
            "task": dispatch.get("task", ""),
            "tool_call_id": "",
        }
        return "dispatch", data, extra_msgs + [assistant_msg]

    # ---- Text-mode JSON parsers (kept as fallback) --------------------

    @staticmethod
    def _parse_single_dispatch(response: str) -> Optional[dict]:
        """Extract a single ``{"worker":..,"task":..}`` dispatch from text.

        Returns the dict or None (done / no JSON found).
        """
        def _check(obj: object) -> Optional[dict]:
            if not isinstance(obj, dict):
                return None
            if obj.get("done"):
                return None
            if "worker" in obj and "task" in obj:
                return obj
            return None

        # Tier 1: fenced ```json blocks
        for block in re.findall(r'```json\s*\n?(.*?)```', response, re.DOTALL):
            try:
                parsed = json.loads(block.strip())
                result = _check(parsed)
                if result is not None:
                    return result
                if isinstance(parsed, dict):
                    return None
            except json.JSONDecodeError:
                continue

        # Tier 2: entire response as JSON
        try:
            parsed = json.loads(response.strip())
            result = _check(parsed)
            if result is not None:
                return result
            if isinstance(parsed, dict):
                return None
        except json.JSONDecodeError:
            pass

        # Tier 3: find the first balanced JSON object in the response
        first_brace = response.find("{")
        if first_brace != -1:
            candidate = OrchestratorAgent._extract_balanced_braces(response, first_brace)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    result = _check(parsed)
                    if result is not None:
                        return result
                    if isinstance(parsed, dict):
                        return None
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _extract_done_answer(response: str) -> str:
        """Pull the ``answer`` field from ``{"done":true,"answer":"..."}``."""
        def _try(obj: object) -> Optional[str]:
            if isinstance(obj, dict) and obj.get("done") and "answer" in obj:
                return obj["answer"]
            return None

        for block in re.findall(r'```json\s*\n?(.*?)```', response, re.DOTALL):
            try:
                r = _try(json.loads(block.strip()))
                if r is not None:
                    return r
            except json.JSONDecodeError:
                continue

        try:
            r = _try(json.loads(response.strip()))
            if r is not None:
                return r
        except json.JSONDecodeError:
            pass

        first_brace = response.find("{")
        if first_brace != -1:
            candidate = OrchestratorAgent._extract_balanced_braces(response, first_brace)
            if candidate:
                try:
                    r = _try(json.loads(candidate))
                    if r is not None:
                        return r
                except json.JSONDecodeError:
                    pass

        return response

    @staticmethod
    def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
        """Extract a balanced ``{...}`` substring starting at *start*.

        Correctly handles nested braces and JSON string literals,
        unlike the previous ``r'\\{[^{}]*\\}'`` regex which broke on
        nested objects.
        """
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> str:
        """Process a user request through the iterative multi-agent pipeline."""
        self._print(f"\n[Orchestrator] Received: {user_input}", level="info")

        blackboard = Blackboard(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})

        final_answer: Optional[str] = None

        for round_num in range(1, self.max_orchestrator_rounds + 1):
            self._print(f"\n[Orchestrator] Round {round_num}/{self.max_orchestrator_rounds}...")

            # Apply context budget before each LLM call
            trimmed = self._manage_context_budget(messages)

            action, data, to_append = self._invoke_orchestrator(trimmed)
            messages.extend(to_append)

            # --- Finish ---
            if action == "finish":
                final_answer = data
                self._print("\n[Orchestrator] Done — no more workers to dispatch.", level="info")
                break

            # --- Dispatch ---
            worker_type = data["worker"].strip()
            task = data["task"].strip()
            tool_call_id = data.get("tool_call_id", "")

            if not worker_type or not task:
                self._print(
                    "\n[Orchestrator] Invalid dispatch (missing worker/task), treating as done.",
                    level="info",
                )
                final_answer = str(data)
                break

            self._print(f"\n--- Round {round_num}: dispatching [{worker_type}] ---", level="info")
            result, is_error = self._run_worker(worker_type, task)

            # Summarize (or truncate) long results
            result = self._summarize_result(worker_type, task, result)

            # Update the blackboard
            if is_error:
                blackboard.add_error(worker_type, round_num, result[:500])
            else:
                blackboard.add_finding(worker_type, round_num, result[:500])

            # Build the feedback message with blackboard state
            bb_section = f"\n\n<blackboard>\n{blackboard.serialize()}\n</blackboard>"
            if is_error:
                feedback = (
                    f"[WORKER ERROR] Worker `{worker_type}` encountered an error:\n\n"
                    f"{result}\n\n"
                    "You may retry with adjusted parameters, try a different worker, "
                    "or provide the final answer with the information gathered so far."
                    f"{bb_section}"
                )
            else:
                feedback = (
                    f"Worker `{worker_type}` completed successfully.\n\n"
                    f"**Task:** {task}\n\n"
                    f"**Result:**\n{result}"
                    f"{bb_section}"
                )

            # Append feedback in the correct role for the active mode
            if tool_call_id:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": feedback + "\n\nDecide your next action.",
                })
        else:
            # Round limit exhausted — force a final synthesis
            self._print(
                f"\n[Orchestrator] Round limit ({self.max_orchestrator_rounds}) reached, "
                "forcing synthesis...",
                level="info",
            )
            trimmed = self._manage_context_budget(messages)
            action, data, to_append = self._invoke_orchestrator(trimmed, force_finish=True)
            messages.extend(to_append)
            final_answer = data if action == "finish" else str(data)

        if final_answer is None:
            final_answer = "Unable to produce a final answer."

        # Reflection: quality-check the final answer before returning
        if self.enable_reflection and final_answer:
            final_answer = self._reflect_on_answer(user_input, blackboard, final_answer)

        # Save to conversation history (mode-agnostic: just user + answer)
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": final_answer})

        return final_answer


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Commands:
  /help                  Show this help
  /workers               List available worker types
  quit / exit / q        Exit

This is a multi-agent system. The orchestrator automatically decomposes
your request and dispatches it to specialized workers (review, test,
optimize, debug).

Just describe what you need in natural language.

Tip: Multi-line paste is auto-detected. You can also use:
  python run_multi.py --task "$(cat problem.txt)"
"""


def _print_sandbox_code(workspace: str) -> None:
    """Print all .py files created in the sandbox after agent finishes."""
    py_files = sorted(glob.glob(os.path.join(workspace, "**", "*.py"), recursive=True))
    py_files = [f for f in py_files if "__pycache__" not in f]

    if not py_files:
        return

    print("=" * 60)
    print("  Generated Code Files")
    print("=" * 60)
    for filepath in py_files:
        rel = os.path.relpath(filepath, workspace)
        try:
            content = open(filepath, encoding="utf-8").read()
        except Exception:
            continue
        print(f"\n--- {rel} ---")
        print(content)
    print("=" * 60)


def _read_user_input() -> str:
    """Read user input with multi-line paste detection."""
    first_line = input("You > ")
    lines = [first_line]

    try:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass

    return "\n".join(lines).strip()


def repl(orchestrator: OrchestratorAgent, sandbox_dir: str = None) -> None:
    """Run an interactive read-eval-print loop."""

    print("=" * 60)
    print("  Multi-Agent Coding System")
    print(f"  Model   : {orchestrator.llm.model}")
    print(f"  Provider: {orchestrator.llm.provider}")
    print(f"  Workers : {', '.join(WORKER_SPECS.keys())}")
    if sandbox_dir:
        print(f"  Sandbox : {sandbox_dir}")
        print(f"  Mode    : sandbox (auto-cleanup on exit)")
    else:
        print(f"  Workspace: {orchestrator.workspace}")
        print(f"  Mode    : direct (operating on real files)")
    print("=" * 60)
    print("Type your request or 'quit' to stop.")
    print("Type /help to see available commands.\n")

    while True:
        try:
            user_input = _read_user_input()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if user_input.strip().lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.strip().lower() == "/workers":
            print("\nAvailable workers:")
            for wtype, spec in WORKER_SPECS.items():
                print(f"  {wtype:10s} — {spec['description']}")
            print()
            continue

        try:
            response = orchestrator.run(user_input)
            print(f"\nAgent > {response}\n")
            if sandbox_dir:
                _print_sandbox_code(sandbox_dir)
        except Exception as e:
            print(f"\n[Error] {e}\n")


def _cleanup_sandbox(sandbox_dir: str) -> None:
    """Remove the temporary sandbox directory."""
    try:
        shutil.rmtree(sandbox_dir)
        print(f"\n[Sandbox] Cleaned up temporary workspace: {sandbox_dir}")
    except Exception as e:
        print(f"\n[Sandbox] Warning: failed to clean up {sandbox_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Coding System — orchestrator + specialized workers."
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help=(
            "Root directory the agents operate in. "
            "If not specified, a temporary sandbox is created and cleaned up on exit."
        ),
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Single-shot mode: run one task and exit.",
    )
    parser.add_argument(
        "--max-worker-steps", "-n",
        type=int,
        default=20,
        help="Max tool-calling iterations per worker agent (default: 20).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="Max orchestrator reasoning rounds (default: 8).",
    )
    parser.add_argument(
        "--max-result-chars",
        type=int,
        default=4000,
        help="Max chars per worker result before truncation (default: 4000).",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=0,
        help="Context budget in tokens. 0 = unlimited (default: 0).",
    )
    parser.add_argument(
        "--no-fc",
        action="store_true",
        help="Disable function calling; use text-based JSON parsing instead.",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Disable LLM-based result summarization (use truncation instead).",
    )
    parser.add_argument(
        "--no-reflect",
        action="store_true",
        help="Disable reflection / self-verification step on final answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Suppress step-by-step debug output.",
    )
    args = parser.parse_args()

    # Determine workspace
    sandbox_dir = None
    if args.workspace:
        workspace = args.workspace
    else:
        sandbox_dir = tempfile.mkdtemp(prefix="codingagent_multi_sandbox_")
        workspace = sandbox_dir
        atexit.register(_cleanup_sandbox, sandbox_dir)
        print(f"[Sandbox] Created temporary workspace: {sandbox_dir}")

    config = Config(debug=not args.no_debug, temperature=args.temperature)

    orchestrator = OrchestratorAgent(
        workspace=workspace,
        config=config,
        max_worker_steps=args.max_worker_steps,
        max_orchestrator_rounds=args.max_rounds,
        max_result_chars=args.max_result_chars,
        context_max_tokens=args.context_max_tokens,
        enable_summarization=not args.no_summarize,
        enable_reflection=not args.no_reflect,
        debug=not args.no_debug,
    )
    if args.no_fc:
        orchestrator._use_function_calling = False

    if args.task:
        response = orchestrator.run(args.task)
        print(response)
        if sandbox_dir:
            _print_sandbox_code(sandbox_dir)
    else:
        repl(orchestrator, sandbox_dir=sandbox_dir)
