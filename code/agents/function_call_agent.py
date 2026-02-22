"""FunctionCallAgent -- agent using OpenAI native function calling.

Production-grade agent with:
- Native function calling via ``llm.invoke_with_tools()``
- Tool output truncation & context budget management
- Automatic debug loop on tool errors
- Reflection / self-verification before final answer
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
class _UsageState:
    """Mutable tracking state for reflection."""
    tools_used: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    wrote_code: bool = False
    ran_tests: bool = False

    def update(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Update tracking state after a tool call."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if tool_name == "file" and arguments.get("action") in ("write", "edit"):
            self.wrote_code = True
            path = arguments.get("path", "")
            if path and path not in self.files_written:
                self.files_written.append(path)
        if tool_name in ("code_exec", "test_runner"):
            self.ran_tests = True


def _map_parameter_type(param_type: str) -> str:
    """Map tool parameter types to JSON Schema types."""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


# Default reflection prompt (same as ReActAgent).
_DEFAULT_REFLECTION_PROMPT = (
    "You are a critical quality reviewer for a coding agent. "
    "Evaluate whether the proposed answer fully and correctly "
    "addresses the user's question.\n\n"
    "Check for:\n"
    "1. **Completeness** — are all parts addressed?\n"
    "2. **Correctness** — any errors or contradictions?\n"
    "3. **Verification** — if code was required, was it written to a file and tested?\n"
    "{verification_note}\n\n"
    "## Agent activity\n"
    "- **Files written**: {files_written}\n"
    "- **Tests executed**: {tests_executed}\n"
    "- **Tools used**: {tools_summary}\n\n"
    "## Original question\n{question}\n\n"
    "## Proposed answer\n{proposed_answer}\n\n"
    "Respond in EXACTLY this format:\n"
    "Verdict: APPROVED or NEEDS_REVISION\n"
    "Reasoning: <one sentence>\n"
    "Issues: <specific issues, or \"none\" if approved>"
)

_DEFAULT_PLANNING_PROMPT = load_agent_prompt("funca_planning")


class FunctionCallAgent(Agent):
    """Agent based on OpenAI native function calling.

    Uses ``llm.invoke_with_tools()`` for structured tool invocation and
    includes the same production features as ``ReActAgent``: trajectory
    tracking, reflection, debug loop, and tool output truncation.
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
        enable_reflection: bool = True,
        max_reflection_retries: int = 2,
        reflection_prompt: str | None = None,
        enable_debug_loop: bool = True,
        max_debug_attempts: int = 3,
        enable_planning: bool = False,
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
        self.enable_reflection = enable_reflection
        self.max_reflection_retries = max_reflection_retries
        self._reflection_prompt_template = reflection_prompt or _DEFAULT_REFLECTION_PROMPT
        self.enable_debug_loop = enable_debug_loop
        self.max_debug_attempts = max_debug_attempts
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
        is_generic_error = has_traceback or has_error_prefix

        if not (is_code_error or is_test_error or is_generic_error):
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
    #  Reflection
    # ------------------------------------------------------------------ #

    def _reflect_on_answer(
        self,
        question: str,
        proposed_answer: str,
        usage: _UsageState,
    ) -> tuple[bool, str]:
        """Self-verify the proposed answer via a separate LLM call.

        Returns ``(True, answer)`` if approved, ``(False, feedback)`` otherwise.
        """
        verification_note = ""
        if usage.wrote_code and not usage.ran_tests:
            verification_note = (
                "\n\n**IMPORTANT**: The agent wrote code but did NOT execute it "
                "or run any tests. You should NOT approve if the task required "
                "working code. Reject and ask the agent to test with code_exec."
            )

        tools_summary = ", ".join(usage.tools_used) or "none"
        files_written_str = ", ".join(usage.files_written) if usage.files_written else "none"
        tests_executed_str = "yes" if usage.ran_tests else "no"
        prompt = self._reflection_prompt_template.format(
            verification_note=verification_note,
            question=question,
            proposed_answer=proposed_answer,
            tools_summary=tools_summary,
            files_written=files_written_str,
            tests_executed=tests_executed_str,
        )

        if self.logger:
            self.logger.info("Reflection started", wrote_code=usage.wrote_code,
                             ran_tests=usage.ran_tests, tools_used=tools_summary)

        try:
            if self.trajectory is not None:
                self.trajectory.start_timer()

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            duration = self.trajectory.stop_timer() if self.trajectory else None

            if self.logger:
                self.logger.llm_call(model=self.llm.model,
                                     latency_ms=duration or 0, call_type="reflection")

            if not response:
                return True, proposed_answer

            text = self._strip_think_tags(response)
            approved = "APPROVED" in text.upper() and "NEEDS_REVISION" not in text.upper()

            reasoning_m = re.search(r"Reasoning:\s*(.+?)(?=\nIssues:|\Z)", text, re.DOTALL)
            issues_m = re.search(r"Issues:\s*(.+)", text, re.DOTALL)
            reasoning = reasoning_m.group(1).strip() if reasoning_m else text[:200]
            issues = issues_m.group(1).strip() if issues_m else ""

            verdict = "APPROVED" if approved else "NEEDS REVISION"
            self._print(f"  Reflection: {verdict} -- {reasoning}")
            self._track("reflection", f"approved={approved}: {reasoning}")
            if self.logger:
                self.logger.info(f"Reflection verdict: {verdict}",
                                 approved=approved, reasoning=reasoning,
                                 issues=issues, latency_ms=duration)

            if approved:
                return True, proposed_answer

            return False, (
                "Your proposed answer was reviewed and found to have issues:\n"
                f"{issues or reasoning}\n\n"
                "Please address these issues and provide a revised answer."
            )

        except Exception as e:
            self._print(f"  Reflection failed ({e}), approving by default.")
            if self.logger:
                self.logger.error(f"Reflection failed: {e}")
            return True, proposed_answer

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
        reflection_attempts = 0
        usage = _UsageState()
        debug_state = _DebugState()

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

        while step < iterations_limit:
            step += 1
            self._print(f"\n[FUNCA: step {step}]")

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
                for call_id, name, raw_args, parsed_args, result, tool_ms in executed:
                    self._print(f"  Tool: {name}({json.dumps(parsed_args, ensure_ascii=False)})")
                    self._track("action", f"{name}({raw_args})", tool=name, step=step)

                    self._print(f"  Observation: {result}")
                    self._track("observation", result, duration_ms=tool_ms, tool=name, step=step)
                    if self.logger:
                        self.logger.tool_call(tool=name, result=result, latency_ms=tool_ms or 0)

                    usage.update(name, parsed_args)

                    debug_suffix = self._maybe_debug(debug_state, name, raw_args, result, step)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": self._truncate(result) + debug_suffix,
                    })

                continue  # next loop iteration

            # ---- Branch: text-only response (proposed answer) ----
            cleaned_content = self._strip_think_tags(content)
            self._print(f"  Proposed answer: {cleaned_content}")

            # Reflection
            if self.enable_reflection and reflection_attempts < self.max_reflection_retries:
                approved, feedback = self._reflect_on_answer(
                    input_text, cleaned_content, usage,
                )
                if not approved:
                    reflection_attempts += 1
                    self._track("reflection_revise",
                                f"attempt {reflection_attempts}/{self.max_reflection_retries}",
                                step=step)
                    messages.append({"role": "assistant", "content": cleaned_content})
                    messages.append({"role": "user", "content": feedback})
                    continue

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
