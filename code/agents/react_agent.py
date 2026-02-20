"""ReAct Agent — text-based Thought/Action parsing.

Uses plain-text ``Thought: ... / Action: tool_name[args]`` format that works
reliably with **any** LLM, including models that do not support OpenAI-style
function calling (e.g. Qwen-Thinking series, local models, etc.).

Key features:
- Tool output truncation & context budget management
- Automatic debug loop on tool errors
- Reflection / self-verification before final answer (text-based)
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


class ReActAgent(Agent):
    """ReAct Agent using text-based Thought/Action parsing.

    The LLM responds with ``Thought: <reasoning>`` followed by one of:
    - ``Action: tool_name[json_args]`` — to call a tool
    - ``Action: Finish[answer]``       — to complete the task
    """

    # Default reflection prompt used when no external template is provided.
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
        enable_reflection: bool = True,
        max_reflection_retries: int = 2,
        reflection_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, llm, system_prompt, config, **kwargs)

        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.max_debug_attempts = max_debug_attempts
        self.enable_debug_loop = enable_debug_loop
        self.enable_reflection = enable_reflection
        self.max_reflection_retries = max_reflection_retries
        self._reflection_prompt_template = reflection_prompt or self._DEFAULT_REFLECTION_PROMPT
        self._debug_prompt_template = load_agent_prompt("debug") if enable_debug_loop else ""
        self._react_prompt_template = load_agent_prompt("react")

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
        """Common bookkeeping when the agent finishes (success or max-steps)."""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
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
    #  Reflection (text-based)
    # ------------------------------------------------------------------ #

    def _reflect_on_answer(
        self,
        question: str,
        proposed_answer: str,
        wrote_code: bool,
        ran_tests: bool,
        tools_used: List[str],
        files_written: List[str] = None,
    ) -> Tuple[bool, str]:
        """Self-verify the proposed answer via a separate LLM call.

        Returns ``(True, answer)`` if approved, ``(False, feedback)`` otherwise.
        """
        verification_note = ""
        if wrote_code and not ran_tests:
            verification_note = (
                "\n\n**IMPORTANT**: The agent wrote code but did NOT execute it "
                "or run any tests. You should NOT approve if the task required "
                "working code. Reject and ask the agent to test with code_exec."
            )

        tools_summary = ", ".join(tools_used) or "none"
        files_written_str = ", ".join(files_written) if files_written else "none"
        tests_executed_str = "yes" if ran_tests else "no"
        prompt = self._reflection_prompt_template.format(
            verification_note=verification_note,
            question=question,
            proposed_answer=proposed_answer,
            tools_summary=tools_summary,
            files_written=files_written_str,
            tests_executed=tests_executed_str,
        )

        if self.logger:
            self.logger.info("Reflection started", wrote_code=wrote_code,
                             ran_tests=ran_tests, tools_used=tools_summary)

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
            self._print(f"  Reflection: {verdict} — {reasoning}")
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
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self, input_text: str, **kwargs) -> str:
        """Run the ReAct loop with text-based Thought/Action parsing."""
        step = 0
        reflection_attempts = 0
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

        while step < self.max_steps:
            step += 1
            self._print(f"\n--- Step {step} ---")

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
                self._print(f"  Finish proposed: {proposed[:200]}...")

                if self.enable_reflection and reflection_attempts < self.max_reflection_retries:
                    approved, feedback = self._reflect_on_answer(
                        input_text, proposed, wrote_code, ran_tests, tools_used,
                        files_written=files_written,
                    )
                    if not approved:
                        reflection_attempts += 1
                        self._track("reflection_revise",
                                    f"attempt {reflection_attempts}/{self.max_reflection_retries}",
                                    step=step)
                        history.extend([
                            f"Thought: {thought or ''}",
                            "Action: Finish[...]",
                            f"Observation: {feedback}",
                        ])
                        continue

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

            # Track usage for reflection
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
