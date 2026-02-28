"""ReAct Agent — text-based Thought/Action parsing.

Uses plain-text ``Thought: ... / Action: tool_name[args]`` format that works
reliably with **any** LLM, including models that do not support OpenAI-style
function calling (e.g. Qwen-Thinking series, local models, etc.).

Key features:
- Tool output truncation & context budget management
- Automatic debug loop on tool errors
- Structured logging & trajectory tracking
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry
from .prompts import load_agent_prompt


# Patterns used to classify error types in tool output.
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


_DEFAULT_PLANNING_PROMPT = load_agent_prompt("react_planning")


class ReActAgent(Agent):
    """ReAct Agent using text-based Thought/Action parsing.

    The LLM responds with ``Thought: <reasoning>`` followed by one of:
    - ``Action: tool_name[json_args]`` — to call a tool
    - ``Action: Finish[answer]``       — to complete the task
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 15,
        max_tool_output_chars: int = 8000,
        max_debug_attempts: int = 3,
        enable_debug_loop: bool = True,
        enable_planning: bool = False,
        **kwargs,
    ):
        super().__init__(name, llm, system_prompt, config, **kwargs)

        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.max_debug_attempts = max_debug_attempts
        self.enable_debug_loop = enable_debug_loop
        self._debug_prompt_template = load_agent_prompt("debug") if enable_debug_loop else ""
        self._react_prompt_template = load_agent_prompt("react")
        self.enable_planning = enable_planning

    # ------------------------------------------------------------------ #
    #  Tool management
    # ------------------------------------------------------------------ #

    def add_tool(self, tool):
        """Add a tool to the registry (supports MCP auto-expand)."""
        if getattr(tool, 'auto_expand', False) and getattr(tool, '_available_tools', None):
            for mcp_tool in tool._available_tools:
                from ..tools.base import Tool
                wrapped = Tool(
                    name=f"{tool.name}_{mcp_tool['name']}",
                    description=mcp_tool.get('description', ''),
                    func=lambda text, t=tool, tn=mcp_tool['name']: t.run({
                        "action": "call_tool", "tool_name": tn,
                        "arguments": {"input": text},
                    }),
                )
                self.tool_registry.register_tool(wrapped)
            self._print(f"  MCP tool '{tool.name}' expanded into {len(tool._available_tools)} tools")
        else:
            self.tool_registry.register_tool(tool)

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #

    def _build_prompt(self, question: str, history: List[str]) -> str:
        """Build the full prompt for one ReAct step."""
        tools_desc = self.tool_registry.get_tools_description()
        react_body = self._react_prompt_template.format(tools=tools_desc)

        parts = [p for p in [self.system_prompt, react_body] if p]
        parts.append(f"\n## Current Task\n**Question:** {question}")
        if history:
            parts.append("\n## Execution History\n" + "\n".join(history))
        parts.append("\nNow continue your reasoning and action:")
        return "\n\n".join(parts)

    def _build_planning_prompt(self, question: str) -> str:
        """Build a planning-specific prompt that omits ReAct tool format examples.

        Unlike ``_build_prompt``, this only lists tool *names* (no JSON schemas
        or ``Action: tool_name[...]`` examples) so that smaller LLMs are not
        tempted to emit tool calls during the planning phase.
        """
        tool_names = self.tool_registry.list_tools()
        tools_list = ", ".join(tool_names) if tool_names else "none"
        parts = [p for p in [self.system_prompt] if p]
        parts.append(f"## Current Task\n**Question:** {question}")
        parts.append(f"## Available Tools\n{tools_list}")
        parts.append(_DEFAULT_PLANNING_PROMPT)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Text parsing
    # ------------------------------------------------------------------ #

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

    @staticmethod
    def _parse_response(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract Thought and Action from an LLM response."""
        cleaned = ReActAgent._strip_think_tags(text)
        thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", cleaned, re.DOTALL)
        action_m = re.search(r"Action:\s*(.+)", cleaned)
        return (
            thought_m.group(1).strip() if thought_m else None,
            action_m.group(1).strip() if action_m else None,
        )

    @staticmethod
    def _parse_action(action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse ``tool_name[args]`` → (tool_name, raw_args) or (None, None)."""
        m = re.match(r"(\w+)\[(.+)\]$", action_text, re.DOTALL)
        return (m.group(1), m.group(2)) if m else (None, None)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _truncate(self, text: str) -> str:
        """Truncate tool output to ``max_tool_output_chars``."""
        limit = self.max_tool_output_chars
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n\n... [truncated {len(text) - limit} chars] ...\n\n{text[-half:]}"

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

    @staticmethod
    def _history_error(thought: Optional[str], action: str, error_msg: str) -> List[str]:
        """Build a 3-line history block for an error observation."""
        return [
            f"Thought: {thought or '(empty)'}",
            f"Action: {action}",
            f"Observation: {error_msg}",
        ]

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self, input_text: str, *, enable_planning: bool | None = None, **kwargs) -> str:
        """Run the ReAct loop with text-based Thought/Action parsing."""
        step = 0
        history: List[str] = []
        tools_used: List[str] = []
        wrote_code = False
        ran_tests = False
        files_written: List[str] = []
        debug_state = _DebugState()

        if self.trajectory is not None:
            self.trajectory.reset()
            self.trajectory.start(task=input_text)
        if self.logger:
            self.logger.lifecycle("start", task=input_text)
        self._print(f"\n[{self.name}] Starting question: {input_text}", level="info")

        # --- Planning phase (optional) ---
        planning_enabled = enable_planning if enable_planning is not None else self.enable_planning
        if planning_enabled:
            self._print(f"\n[REACT: plan]", level="info")
            plan_prompt = self._build_planning_prompt(input_text)
            if self.trajectory is not None:
                self.trajectory.start_timer()
            try:
                plan_response = self.llm.invoke(
                    [{"role": "user", "content": plan_prompt}], **kwargs,
                )
                plan_ms = self.trajectory.stop_timer() if self.trajectory else None
                if plan_response:
                    plan_text = self._strip_think_tags(plan_response)
                    # Strip leaked tool-call markup that some models emit.
                    plan_text = re.sub(
                        r"<tool_call>.*?(?:</tool_call>|$)", "", plan_text, flags=re.DOTALL
                    ).strip()
                    # Strip any Action/Thought lines the LLM may have emitted.
                    plan_text = re.sub(r"^[ \t]*Action:.*$", "", plan_text, flags=re.MULTILINE)
                    plan_text = re.sub(r"^[ \t]*Thought:.*$", "", plan_text, flags=re.MULTILINE)
                    plan_text = re.sub(r"\w+\[{.*?}\]", "", plan_text, flags=re.DOTALL)
                    # Collapse multiple blank lines left by removals.
                    plan_text = re.sub(r"\n{3,}", "\n\n", plan_text).strip()
                    if plan_text:
                        self._print(f"  Plan:\n{plan_text}")
                        self._track("plan", plan_text, duration_ms=plan_ms)
                        if self.logger:
                            self.logger.llm_call(model=self.llm.model,
                                                 latency_ms=plan_ms or 0, call_type="planning")
                        history.append(f"Plan:\n{plan_text}")
                    else:
                        self._print("  Plan: (empty response, skipping)", level="info")
                else:
                    self._print("  Plan: (empty response, skipping)", level="info")
            except Exception as e:
                self._print(f"  Plan failed ({e}), proceeding without plan.", level="info")
                if self.trajectory is not None:
                    self.trajectory.stop_timer()

        while step < self.max_steps:
            step += 1
            self._print(f"\n[REACT: step {step}]")

            # --- Context budget management ---
            if self.context_max_tokens > 0 and len(history) > 4:
                probe = self._build_prompt(input_text, history)
                total_tokens = self._count_tokens(probe)
                trigger = int(self.context_max_tokens * self.COMPACTION_THRESHOLD)
                if total_tokens > trigger:
                    keep = min(4, len(history) // 2)
                    old_entries = history[:-keep]
                    recent = history[-keep:]
                    summary = self._compact_messages(
                        [{"role": "assistant", "content": "\n".join(old_entries)}]
                    )
                    history = [f"[Previous steps summary]:\n{summary}"] + recent
                    self._print(
                        f"  [Compaction] History: {len(old_entries) + keep} entries "
                        f"-> {len(history)} entries ({total_tokens} tokens -> "
                        f"~{self._count_tokens(self._build_prompt(input_text, history))} tokens)",
                    )

            # --- LLM call ---
            prompt = self._build_prompt(input_text, history)
            if self.trajectory is not None:
                self.trajectory.start_timer()
            try:
                response_text = self.llm.invoke(
                    [{"role": "user", "content": prompt}], **kwargs,
                )
            except Exception as e:
                self._print(f"  Error: LLM call failed: {e}", level="info")
                self._track("error", f"LLM call failed: {e}")
                break

            llm_ms = self.trajectory.stop_timer() if self.trajectory else None
            self._track("llm_call", response_text or "", duration_ms=llm_ms, step=step)
            if self.logger:
                self.logger.llm_call(model=self.llm.model, latency_ms=llm_ms or 0)

            if not response_text:
                self._print("  Error: LLM returned empty response.", level="info")
                self._track("error", "LLM returned empty response")
                break

            # --- Parse response ---
            thought, action = self._parse_response(response_text)

            if thought:
                self._print(f"  Thought: {thought}")
                self._track("thought", thought, step=step)
                if self.logger:
                    self.logger.step("thought", thought, step_num=step)

            if not action:
                self._print("  Warning: no Action found, retrying.", level="info")
                history.extend(self._history_error(
                    thought, "(missing)",
                    "Error — you must include an Action. "
                    "Use `tool_name[args]` or `Finish[answer]`.",
                ))
                continue

            # --- Finish ---
            if action.startswith("Finish"):
                m = re.match(r"Finish\[(.+)\]$", action, re.DOTALL)
                proposed = m.group(1).strip() if m else ""

                self._print(f"  Final answer: {proposed}", level="info")
                self._track("final_answer", proposed)
                return self._end_run(input_text, proposed,
                                     final_answer_length=len(proposed))

            # --- Tool call ---
            tool_name, tool_args = self._parse_action(action)

            if not tool_name:
                history.extend(self._history_error(
                    thought, action,
                    "Error — invalid action format. Use `tool_name[json_args]` or `Finish[answer]`.",
                ))
                continue

            available = self.tool_registry.list_tools()
            if tool_name not in available:
                history.extend(self._history_error(
                    thought, action,
                    f"Error — tool '{tool_name}' not found. Available: {', '.join(available)}",
                ))
                continue

            self._print(f"  Action: {tool_name}[{tool_args[:200]}]")
            self._track("action", f"{tool_name}[{tool_args}]", tool=tool_name)

            # Execute
            if self.trajectory is not None:
                self.trajectory.start_timer()
            observation = self.tool_registry.execute_tool(tool_name, tool_args)
            tool_ms = self.trajectory.stop_timer() if self.trajectory else None

            obs_preview = observation[:500] + "..." if len(observation) > 500 else observation
            self._print(f"  Observation: {obs_preview}")
            self._track("observation", observation, duration_ms=tool_ms, tool=tool_name)
            if self.logger:
                self.logger.tool_call(tool=tool_name, result=observation,
                                      latency_ms=tool_ms or 0)

            # Track tool usage
            if tool_name not in tools_used:
                tools_used.append(tool_name)
            if tool_name == "file" and ('"write"' in tool_args or '"edit"' in tool_args):
                wrote_code = True
                # Try to extract filename from tool args
                try:
                    import json as _json
                    _args = _json.loads(tool_args)
                    _path = _args.get("path", "")
                    if _path and _path not in files_written:
                        files_written.append(_path)
                except Exception:
                    pass
            if tool_name in ("code_exec", "test_runner"):
                ran_tests = True

            # Debug loop
            debug_suffix = self._maybe_debug(debug_state, tool_name, tool_args, observation, step)

            # Append to history
            history.extend([
                f"Thought: {thought or '(reasoning omitted)'}",
                f"Action: {action}",
                f"Observation: {self._truncate(observation)}{debug_suffix}",
            ])

        # ---- Max steps reached ----
        self._print("  Max steps reached. Stopping.", level="info")
        self._track("error", "Max steps reached without finding answer")
        return self._end_run(
            input_text,
            "Unable to complete the task within the allowed number of steps.",
            reason="max_steps_reached",
        )

    # ------------------------------------------------------------------ #
    #  Debug loop
    # ------------------------------------------------------------------ #

    def _maybe_debug(
        self, state: _DebugState, tool_name: str,
        tool_args: str, observation: str, step: int,
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
            state.failed_action = f"{tool_name}[{tool_args}]"
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

    @staticmethod
    def _classify_observation(tool_name: str, observation: str) -> Optional[dict]:
        """Return ``{"error_type": ..., "summary": ...}`` if *observation* is an error."""
        low = observation.lower()

        # Detect error presence
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

        # Classify error type
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
            "summary": ReActAgent._extract_error_summary(observation),
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
            # If we reach a non-empty line that doesn't match known patterns,
            # use it as the summary (last non-empty line).
            return s[:200]
        return "Unknown error"
