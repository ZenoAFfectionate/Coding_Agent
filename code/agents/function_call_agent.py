"""FunctionCallAgent -- agent using OpenAI native function calling.

Production-grade agent with:
- Native function calling via ``llm.invoke_with_tools()``
- Tool output truncation & context budget management
- Automatic debug loop on tool errors
- Structured logging & trajectory tracking
- Parallel tool execution for batched calls
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator, TYPE_CHECKING, Any

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
from ..tools.builtin.finish_tool import FinishTool
from ..tools.builtin.escalate_tool import EscalateTool
from .prompts import load_agent_prompt

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


# Patterns used to classify error types in tool output (shared with ReActAgent).
_ERROR_TYPE_PATTERNS = [
    ("syntaxerror",                          "syntax_error"),
    ("importerror",                          "import_error"),
    ("modulenotfounderror",                  "import_error"),
    ("timeout",                              "timeout"),
    ("timed out",                            "timeout"),
    ("typeerror",                            "runtime_error"),
    ("valueerror",                           "runtime_error"),
    ("attributeerror",                       "runtime_error"),
    ("nameerror",                            "runtime_error"),
]


@dataclass
class _DebugState:
    """Track consecutive debug attempts for error recovery."""
    active: bool = False
    error_type: str = ""
    error_summary: str = ""
    failed_action: str = ""
    attempts: int = 0


@dataclass
class _CircuitBreakerState:
    """Track repeated failures to auto-terminate stuck workers."""
    consecutive_errors: int = 0
    debug_exhaustions: int = 0
    last_error_types: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    errors_seen: list[str] = field(default_factory=list)

    _MAX_WINDOW: int = 5  # rolling window for error type tracking

    def record_success(self) -> None:
        self.consecutive_errors = 0

    def record_error(self, error_type: str, summary: str) -> None:
        self.consecutive_errors += 1
        self.last_error_types.append(error_type)
        if len(self.last_error_types) > self._MAX_WINDOW:
            self.last_error_types = self.last_error_types[-self._MAX_WINDOW:]
        if summary and summary not in self.errors_seen[-3:]:
            self.errors_seen.append(summary)

    def record_debug_exhaustion(self) -> None:
        self.debug_exhaustions += 1

    def record_tool_call(self, tool_name: str) -> None:
        self.tools_called.append(tool_name)

    def is_tripped(self) -> tuple[bool, str]:
        """Check if the circuit breaker should trip.

        Returns (tripped, reason).
        """
        if self.consecutive_errors >= 5:
            return True, f"5+ consecutive tool errors ({self.consecutive_errors})"
        if self.debug_exhaustions >= 2:
            return True, f"debug loop exhausted {self.debug_exhaustions} times"
        if len(self.last_error_types) >= 3:
            from collections import Counter
            counts = Counter(self.last_error_types)
            for etype, count in counts.items():
                if count >= 3:
                    return True, f"same error type '{etype}' appeared {count} times in last {len(self.last_error_types)} errors"
        return False, ""

    def progress_summary(self) -> str:
        """Build a summary of what happened for the orchestrator."""
        parts = []
        if self.tools_called:
            from collections import Counter
            tc = Counter(self.tools_called)
            parts.append("Tools called: " + ", ".join(f"{k}({v}x)" for k, v in tc.most_common()))
        if self.errors_seen:
            parts.append("Errors seen: " + "; ".join(self.errors_seen[-5:]))
        return " | ".join(parts) if parts else "No activity recorded"


@dataclass
class _UsageState:
    """Mutable tracking state for tool usage."""
    tools_used: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    files_confirmed_edited: list[str] = field(default_factory=list)
    wrote_code: bool = False
    ran_tests: bool = False

    def update(self, tool_name: str, arguments: dict[str, Any], result: str = "") -> None:
        """Update tracking state after a tool call.

        Args:
            tool_name: Name of the tool that was called.
            arguments: Parsed arguments passed to the tool.
            result: The tool's return value (used to verify success).
        """
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "file" and arguments.get("action") in ("write", "edit", "str_replace"):
            self.wrote_code = True
            path = arguments.get("path", "")
            if path and path not in self.files_written:
                self.files_written.append(path)
            # Only count as a confirmed edit when the tool did not return an error.
            # Successful file-tool results do NOT start with "Error:" (case-insensitive).
            result_start = result.lstrip()[:7].lower()
            if result_start != "error: " and "error:" not in result.lower()[:30]:
                if path and path not in self.files_confirmed_edited:
                    self.files_confirmed_edited.append(path)
        if tool_name in ("code_exec", "test_runner"):
            self.ran_tests = True

    @property
    def confirmed_edits(self) -> int:
        """Number of files with at least one confirmed successful edit."""
        return len(self.files_confirmed_edited)

    @property
    def exploration_depth(self) -> int:
        """Number of distinct tool names that have been used."""
        return len(self.tools_used)


def _map_parameter_type(param_type: str) -> str:
    """Map tool parameter types to JSON Schema types."""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


_DEFAULT_PLANNING_PROMPT = load_agent_prompt("funca_planning")


class FunctionCallAgent(Agent):
    """Agent based on OpenAI native function calling.

    Uses ``llm.invoke_with_tools()`` for structured tool invocation and
    includes the same production features as ``ReActAgent``: trajectory
    tracking, debug loop, and tool output truncation.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
        enable_tool_calling: bool = True,
        default_tool_choice: str | dict = "auto",
        # Production parameters (match ReActAgent interface)
        max_steps: int = 32,
        max_tool_output_chars: int = 8000,
        enable_debug_loop: bool = True,
        max_debug_attempts: int = 3,
        enable_planning: bool = False,
        # Controls when text-only responses are accepted vs. nudged back to tools.
        # "strict"  – nudge the model whenever it hasn't made confirmed file edits yet
        #             (states A, B, C all trigger nudges).  Best for code-repair tasks.
        # "lenient" – only nudge when confirmed edits exist but the answer is empty
        #             (state C).  States A/B are passed through immediately.
        # "off"     – never nudge; accept any text-only response as a final answer.
        text_only_policy: str = "strict",
        # Legacy alias
        max_tool_iterations: int | None = None,
        # Base Agent pass-through
        **kwargs,
    ):
        super().__init__(name, llm, system_prompt, config, **kwargs)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice

        # Use legacy alias if provided and max_steps is at default
        if max_tool_iterations is not None and max_steps == 32:
            self.max_steps = max_tool_iterations
        else:
            self.max_steps = max_steps

        self.max_tool_output_chars = max_tool_output_chars
        self.enable_debug_loop = enable_debug_loop
        self.max_debug_attempts = max_debug_attempts
        if text_only_policy not in ("strict", "lenient", "off"):
            raise ValueError(f"text_only_policy must be 'strict', 'lenient', or 'off', got {text_only_policy!r}")
        self.text_only_policy = text_only_policy
        self.enable_planning = enable_planning
        self._debug_prompt_template = load_agent_prompt("debug") if enable_debug_loop else ""

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #

    def _get_system_prompt(self) -> str:
        """Build system prompt, injecting tool descriptions."""
        base_prompt = self.system_prompt or "You are a reliable AI assistant capable of calling tools when needed."

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "No available tools":
            return base_prompt

        # Load the tool section template and fill in the description
        tool_template = load_agent_prompt("function_call")
        tool_section = tool_template.format(tools_description=tools_description)

        return base_prompt + "\n" + tool_section

    def _build_planning_messages(self, question: str) -> list[dict[str, Any]]:
        """Build planning-specific messages that omit detailed tool descriptions.

        Unlike the normal system prompt, this only lists tool *names* (no JSON
        schemas or full descriptions) so that smaller LLMs are not tempted to
        emit function calls during the planning phase.
        """
        base_prompt = self.system_prompt or "You are a reliable AI assistant."
        tool_names = self.tool_registry.list_tools() if self.tool_registry else []
        tools_list = ", ".join(tool_names) if tool_names else "none"
        user_content = (
            f"## Current Task\n**Question:** {question}\n\n"
            f"## Available Tools\n{tools_list}\n\n"
            f"{_DEFAULT_PLANNING_PROMPT}"
        )
        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------ #
    #  Tool schema building
    # ------------------------------------------------------------------ #

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        for tool in self.tool_registry.get_all_tools():
            properties: dict[str, Any] = {}
            required: list[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "enum", None):
                    properties[param.name]["enum"] = param.enum
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # Tools registered via register_function (access internal structure)
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Input text"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    # ------------------------------------------------------------------ #
    #  Text helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """Safely extract text from OpenAI response message.content."""
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    @staticmethod
    def _extract_think_content(text: str) -> str:
        """Extract the thinking content from ``<think>...</think>`` blocks.

        Returns the concatenated thinking text (without tags), or "" if none.
        Handles both standard ``<think>...</think>`` and orphaned ``</think>``
        (missing opening tag, common with vLLM).
        """
        # Standard: extract from complete <think>...</think> blocks.
        parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if parts:
            return "\n".join(p.strip() for p in parts if p.strip())
        # Fallback: orphaned </think> — everything before it is thinking.
        if "</think>" in text:
            thinking = text.rsplit("</think>", 1)[0].strip()
            # Remove a leading <think> if present but unclosed
            thinking = re.sub(r"^<think>\s*", "", thinking)
            return thinking
        return ""

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove ``<think>...</think>`` blocks emitted by Qwen-Thinking models.

        Also handles a missing opening ``<think>`` tag (common when served via
        vLLM), where only ``</think>`` appears in the output — everything
        before the last ``</think>`` is treated as thinking content and removed.
        """
        # Standard: remove complete <think>...</think> blocks.
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Fallback: if an orphaned </think> remains (missing <think>),
        # strip everything before and including the last </think>.
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
        return cleaned or text

    def _truncate(self, text: str) -> str:
        """Truncate tool output to ``max_tool_output_chars``."""
        limit = self.max_tool_output_chars
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n\n... [truncated {len(text) - limit} chars] ...\n\n{text[-half:]}"

    @staticmethod
    def _preview(text: str, limit: int = 500) -> str:
        """Return a short preview of *text* for console output."""
        return text[:limit] + "..." if len(text) > limit else text

    # ------------------------------------------------------------------ #
    #  Argument handling
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_function_call_arguments(arguments: str | None) -> dict[str, Any]:
        """Parse JSON string arguments returned by the model."""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert parameter types based on tool definitions."""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    # ------------------------------------------------------------------ #
    #  Tool execution
    # ------------------------------------------------------------------ #

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the string result."""
        if not self.tool_registry:
            return "Error: tool registry not configured"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"Tool call failed: {exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"Tool call failed: {exc}"

        return f"Error: tool '{tool_name}' not found"

    # ------------------------------------------------------------------ #
    #  Error classification & debug loop
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_observation(tool_name: str, observation: str) -> dict | None:
        """Return ``{"error_type": ..., "summary": ...}`` if *observation* is an error."""
        low = observation.lower()

        has_nonzero_exit = bool(re.search(r"exit code:\s*[1-9]", low))
        has_traceback = "traceback (most recent call last)" in low
        has_error_prefix = any(
            line.strip().startswith(("Error:", "ERROR:"))
            for line in observation.strip().splitlines()
        )

        is_code_error = tool_name == "code_exec" and (has_nonzero_exit or has_traceback)
        is_test_error = tool_name == "test_runner" and ("failed" in low or "error" in low)
        is_file_error = tool_name == "file" and "error:" in low[:50]
        is_generic_error = has_traceback or has_error_prefix

        if not (is_code_error or is_test_error or is_file_error or is_generic_error):
            return None

        if is_test_error:
            error_type = "test_failure"
        else:
            error_type = "runtime_error"
            for pattern, etype in _ERROR_TYPE_PATTERNS:
                if pattern in low:
                    error_type = etype
                    break

        return {
            "error_type": error_type,
            "summary": FunctionCallAgent._extract_error_summary(observation),
        }

    @staticmethod
    def _extract_error_summary(observation: str) -> str:
        """Extract a one-line error summary from a traceback or test output."""
        for line in reversed(observation.strip().splitlines()):
            s = line.strip()
            if not s:
                continue
            if re.match(r"^[A-Z]\w*(Error|Exception|Warning)", s):
                return s[:200]
            if "FAILED" in s or "ERRORS" in s:
                return s[:200]
            return s[:200]
        return "Unknown error"

    def _maybe_debug(
        self, state: _DebugState, tool_name: str,
        tool_args_str: str, observation: str, step: int,
    ) -> str:
        """Check observation for errors and return debug guidance suffix."""
        if not self.enable_debug_loop:
            return ""

        error = self._classify_observation(tool_name, observation)

        if error is None:
            if state.active:
                self._track("debug_resolved",
                            f"resolved after {state.attempts} attempts", step=step)
                state.__init__()  # reset
            return ""

        if not state.active:
            state.active = True
            state.error_type = error["error_type"]
            state.error_summary = error["summary"]
            state.failed_action = f"{tool_name}({tool_args_str})"
            state.attempts = 1
        else:
            state.attempts += 1

        if state.attempts > self.max_debug_attempts:
            suffix = (f"\n\n[Debug loop exhausted ({self.max_debug_attempts} attempts) "
                      f"for {state.error_type}.]")
            state.__init__()  # reset
            return suffix

        self._track("debug",
                     f"attempt {state.attempts}/{self.max_debug_attempts}: {error['error_type']}",
                     step=step)
        return "\n\n" + self._debug_prompt_template.format(
            error_type=state.error_type,
            error_summary=state.error_summary,
            failed_action=state.failed_action,
            attempt=state.attempts,
            max_attempts=self.max_debug_attempts,
        )

    # ------------------------------------------------------------------ #
    #  Bookkeeping
    # ------------------------------------------------------------------ #

    def _end_run(self, input_text: str, answer: str, **log_kw) -> str:
        """Common bookkeeping when the agent finishes (success or max-steps).

        Builds a rich execution summary from the trajectory and stores it
        alongside the answer so the next REPL turn sees the full execution
        context (tools called, files modified, errors encountered).
        """
        self.add_message(Message(input_text, "user"))
        rich_answer = self._build_execution_summary(input_text, answer)
        self.add_message(Message(rich_answer, "assistant"))
        if self.trajectory is not None:
            self.trajectory.end()
        if self.logger:
            self.logger.lifecycle("end", **log_kw)
        return answer

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: int | None = None,
        tool_choice: str | dict | None = None,
        enable_planning: bool | None = None,
        **kwargs,
    ) -> str:
        """Run the function-calling conversation loop with full production features."""
        # --- Initialise tracking state ---
        step = 0
        text_only_retries = 0
        _MAX_TEXT_ONLY_RETRIES = 3  # max times to nudge model back to tool use
        usage = _UsageState()
        debug_state = _DebugState()
        cb_state = _CircuitBreakerState()

        if self.trajectory is not None:
            self.trajectory.reset()
            self.trajectory.start(task=input_text)
        if self.logger:
            self.logger.lifecycle("start", task=input_text)
        self._print(f"\n[{self.name}] Starting question: {input_text}", level="info")

        # --- Build messages ---
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._get_system_prompt()},
            *[{"role": msg.role, "content": msg.content} for msg in self._history],
            {"role": "user", "content": input_text},
        ]

        # --- Build tool schemas ---
        tool_schemas = self._build_tool_schemas()
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            return self._end_run(input_text, response_text or "")

        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_steps
        effective_tool_choice: str | dict = tool_choice if tool_choice is not None else self.default_tool_choice

        # --- Planning phase (optional) ---
        planning_enabled = enable_planning if enable_planning is not None else self.enable_planning
        if planning_enabled:
            self._print(f"\n[FUNCA: plan]", level="info")
            plan_messages = self._build_planning_messages(input_text)
            if self.trajectory is not None:
                self.trajectory.start_timer()
            try:
                plan_response = self.llm.invoke(plan_messages, **kwargs)
                plan_ms = self.trajectory.stop_timer() if self.trajectory else None
                if plan_response:
                    plan_text = self._strip_think_tags(plan_response)
                    # Strip leaked tool-call markup that some models emit.
                    plan_text = re.sub(
                        r"<tool_call>.*?(?:</tool_call>|$)", "", plan_text, flags=re.DOTALL
                    ).strip()
                    # Collapse multiple blank lines left by removals.
                    plan_text = re.sub(r"\n{3,}", "\n\n", plan_text).strip()
                    if plan_text:
                        self._print(f"  Plan:\n{plan_text}")
                        self._track("plan", plan_text, duration_ms=plan_ms)
                        if self.logger:
                            self.logger.llm_call(model=self.llm.model,
                                                 latency_ms=plan_ms or 0, call_type="planning")
                        messages.append({"role": "assistant", "content": plan_text})
                        messages.append({"role": "user", "content": "Proceed with your plan step by step."})
                    else:
                        self._print("  Plan: (empty response, skipping)", level="info")
                else:
                    self._print("  Plan: (empty response, skipping)", level="info")
            except Exception as e:
                self._print(f"  Plan failed ({e}), proceeding without plan.", level="info")
                if self.trajectory is not None:
                    self.trajectory.stop_timer()

        # --- Agent loop ---
        final_response = ""
        _budget_warned = False

        while step < iterations_limit:
            step += 1
            self._print(f"\n[FUNCA: step {step}]")

            # --- Step budget warning ---
            budget_threshold = max(1, int(iterations_limit * 0.75))
            if step == budget_threshold and not _budget_warned:
                _budget_warned = True
                remaining = iterations_limit - step
                budget_msg = (
                    f"[STEP BUDGET WARNING] You have used {step}/{iterations_limit} steps "
                    f"({remaining} remaining). Focus on producing a final, verified result. "
                    "If your current approach is not converging, call `finish` with the "
                    "best answer you have so far."
                )
                messages.append({"role": "user", "content": budget_msg})
                self._print(f"  {budget_msg}", level="info")

            # --- Context budget management ---
            messages = self._manage_context_budget(messages)

            # LLM call with tools
            if self.trajectory is not None:
                self.trajectory.start_timer()
            try:
                response = self.llm.invoke_with_tools(
                    messages,
                    tools=tool_schemas,
                    tool_choice=effective_tool_choice,
                    **kwargs,
                )
            except Exception as e:
                if self.trajectory is not None:
                    self.trajectory.stop_timer()
                self._print(f"  Error: LLM call failed: {e}", level="info")
                self._track("error", f"LLM call failed: {e}")
                # Don't break — consume a step and let the loop retry
                continue

            llm_ms = self.trajectory.stop_timer() if self.trajectory else None

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])

            # Track LLM call (extract token usage from response)
            _usage = getattr(response, "usage", None)
            _in_tok = getattr(_usage, "prompt_tokens", 0) if _usage else 0
            _out_tok = getattr(_usage, "completion_tokens", 0) if _usage else 0
            tc_summary = ", ".join(tc.function.name for tc in tool_calls) if tool_calls else "(text)"
            self._track("llm_call", content or tc_summary, duration_ms=llm_ms, step=step)
            if self.logger:
                self.logger.llm_call(model=self.llm.model, latency_ms=llm_ms or 0,
                                     input_tokens=_in_tok, output_tokens=_out_tok)

            # ---- Branch: tool calls present ----
            if tool_calls:
                # Build assistant message with tool_calls for conversation
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # --- Execute all tool calls, collect results ---
                # Each entry: (call_id, name, raw_args, parsed_args, result, duration_ms)
                if len(tool_calls) == 1:
                    tc = tool_calls[0]
                    parsed = self._parse_function_call_arguments(tc.function.arguments)
                    if self.trajectory is not None:
                        self.trajectory.start_timer()
                    result = self._execute_tool_call(tc.function.name, parsed)
                    tool_ms = self.trajectory.stop_timer() if self.trajectory else None
                    executed = [(tc.id, tc.function.name, tc.function.arguments, parsed, result, tool_ms)]
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _exec(tc):
                        args = self._parse_function_call_arguments(tc.function.arguments)
                        return tc.id, tc.function.name, tc.function.arguments, args, self._execute_tool_call(tc.function.name, args)

                    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
                        futures = [pool.submit(_exec, tc) for tc in tool_calls]
                        by_id = {}
                        for f in as_completed(futures):
                            r = f.result()
                            by_id[r[0]] = r
                    # Preserve original order; None for duration (parallel — no per-call timing)
                    executed = [(*by_id[tc.id], None) for tc in tool_calls]

                # --- Process each result uniformly ---
                finish_answer = None
                for call_id, name, raw_args, parsed_args, result, tool_ms in executed:
                    self._print(f"  Tool: {name}({json.dumps(parsed_args, ensure_ascii=False)})")
                    self._track("action", f"{name}({raw_args})", tool=name, step=step)

                    self._print(f"  Observation: {result}")
                    self._track("observation", result, duration_ms=tool_ms, tool=name, step=step)
                    if self.logger:
                        self.logger.tool_call(tool=name, result=result, latency_ms=tool_ms or 0)

                    usage.update(name, parsed_args, result)

                    # --- Finish tool: early exit ---
                    if name == "finish" and result.startswith(FinishTool.SENTINEL):
                        finish_answer = self._strip_think_tags(
                            result[len(FinishTool.SENTINEL):]
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": result,
                        })
                        self._print(f"  Finish called: {finish_answer}", level="info")
                        self._track("final_answer", finish_answer)
                        break

                    # --- Escalate tool: return to orchestrator ---
                    if name == "escalate" and result.startswith(EscalateTool.SENTINEL):
                        escalate_reason = result[len(EscalateTool.SENTINEL):]
                        escalate_msg = f"[ESCALATED] {escalate_reason}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": result,
                        })
                        self._print(f"  Escalated: {escalate_reason}", level="info")
                        self._track("escalated", escalate_reason)
                        return self._end_run(input_text, escalate_msg, reason="escalated")

                    debug_suffix = self._maybe_debug(debug_state, name, raw_args, result, step)

                    # --- Circuit breaker state update ---
                    cb_state.record_tool_call(name)
                    error_info = self._classify_observation(name, result)
                    if error_info:
                        cb_state.record_error(error_info["error_type"], error_info["summary"])
                    else:
                        cb_state.record_success()
                    if "Debug loop exhausted" in debug_suffix:
                        cb_state.record_debug_exhaustion()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": self._truncate(result) + debug_suffix,
                    })

                if finish_answer is not None:
                    return self._end_run(input_text, finish_answer)

                # --- Circuit breaker check ---
                tripped, trip_reason = cb_state.is_tripped()
                if tripped:
                    abort_msg = (
                        f"[Circuit breaker tripped: {trip_reason}] "
                        f"Progress: {cb_state.progress_summary()}"
                    )
                    self._print(f"  {abort_msg}", level="info")
                    self._track("circuit_breaker", abort_msg, step=step)
                    return self._end_run(input_text, abort_msg, reason="circuit_breaker")

                continue  # next loop iteration

            # ---- Branch: text-only response (proposed answer) ----

            # Extract and display thinking content before stripping
            think_content = self._extract_think_content(content)
            if think_content:
                self._print(f"  [Thinking] {self._preview(think_content, 1000)}")
                self._track("thinking", think_content, step=step)

            cleaned_content = self._strip_think_tags(content)

            # ----------------------------------------------------------------
            # Completion guard — decide whether to accept this text-only
            # response as the final answer, or nudge the model to keep working.
            #
            # Four states, handled in priority order:
            #
            #   (A) No exploration, no edits  → must start working
            #   (B) Explored but no edits     → must actually edit files
            #   (C) Confirmed edits, empty answer → call finish explicitly
            #   (D) Confirmed edits, real answer  → accept (task done)
            #
            # States A-C each get up to _MAX_TEXT_ONLY_RETRIES nudges before
            # the agent falls through and accepts whatever it has.
            # ----------------------------------------------------------------

            confirmed_edits = usage.confirmed_edits       # successful file edits
            explored_enough = usage.exploration_depth >= 3  # used ≥3 distinct tools
            answer_is_empty = not cleaned_content.strip()
            steps_left = iterations_limit - step
            policy = self.text_only_policy  # "strict" | "lenient" | "off"

            # Determine whether to nudge, and with what message.
            nudge: str | None = None
            label = "accepted"
            if policy != "off" and text_only_retries < _MAX_TEXT_ONLY_RETRIES:

                if confirmed_edits == 0 and not explored_enough and policy == "strict":
                    # State (A) [strict only]: hasn't even started
                    nudge = (
                        "You returned a text response but have not used any tools yet. "
                        f"You still have {steps_left} steps. "
                        "Start by exploring the codebase with `file`, `code_search`, or "
                        "`git` tools — then edit the relevant source files to fix the issue."
                    )
                    label = "no-work"

                elif confirmed_edits == 0 and policy == "strict":
                    # State (B) [strict only]: explored but made no actual file edits
                    nudge = (
                        "You have explored the codebase but have NOT made any code changes yet. "
                        f"You still have {steps_left} steps. "
                        "Use the `file` tool with action='edit' or action='str_replace' to "
                        "modify the relevant source file(s) and fix the issue. "
                        "Do not stop until you have saved at least one edit."
                    )
                    label = "explored-no-edits"

                elif confirmed_edits > 0 and answer_is_empty:
                    # State (C) [strict + lenient]: edits done but empty answer
                    nudge = (
                        f"You have edited {confirmed_edits} file(s). "
                        "Your response was empty — please call the `finish` tool with a "
                        "brief summary of what you changed as the `result` argument. "
                        "If you still have changes to make, continue using tools instead."
                    )
                    label = "edits-done-empty-answer"

                # State (D): nudge is None → fall through to accept immediately

            if nudge:
                # Only count nudges that are actually sent to the model.
                text_only_retries += 1
                tag = f"{text_only_retries}/{_MAX_TEXT_ONLY_RETRIES}"
                self._print(
                    f"  [Nudge {tag} | {label}] {nudge[:120]}...",
                    level="info",
                )
                self._track("text_only_retry",
                            f"{label} | attempt {tag}", step=step)
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": nudge})
                continue
            # nudge is None → fall through to accept

            self._print(f"  Proposed answer: {cleaned_content}")

            # Accepted
            final_response = cleaned_content
            self._print(f"  Final answer: {final_response}", level="info")
            self._track("final_answer", final_response)
            return self._end_run(input_text, final_response,
                                 final_answer_length=len(final_response))

        # ---- Max steps exhausted ----
        if not final_response:
            self._print("  Max steps reached. Forcing final answer.", level="info")
            self._track("error", "Max steps reached without finding answer")
            progress = cb_state.progress_summary()
            try:
                final_choice = self.llm.invoke_with_tools(
                    messages,
                    tools=tool_schemas,
                    tool_choice="none",
                    **kwargs,
                )
                final_response = self._strip_think_tags(
                    self._extract_message_content(final_choice.choices[0].message.content)
                )
            except Exception:
                final_response = "Unable to complete the task within the allowed number of steps."
            final_response += f"\n\n[Max steps exhausted. {progress}]"

        return self._end_run(
            input_text, final_response,
            reason="max_steps_reached",
        )


    def add_tool(self, tool) -> None:
        """Add a tool to the agent."""
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"  MCP tool '{tool.name}' expanded into {len(expanded_tools)} tools")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """Streaming not yet implemented; falls back to single-shot call."""
        result = self.run(input_text, **kwargs)
        yield result
