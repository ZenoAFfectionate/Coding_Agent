"""ReAct (Reasoning and Acting) Agent.

Uses a multi-turn message architecture where each thought/action/observation
becomes its own message, avoiding the O(N^2) cost of re-serializing the
entire history and tool descriptions on every step.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.output_parser import OutputParser
from ..tools.registry import ToolRegistry
from .prompts import load_agent_prompt


@dataclass
class _DebugState:
    """Track consecutive debug attempts for error recovery."""
    active: bool = False
    error_type: str = ""
    error_summary: str = ""
    failed_action: str = ""
    attempts: int = 0

    def reset(self):
        self.active = False
        self.error_type = ""
        self.error_summary = ""
        self.failed_action = ""
        self.attempts = 0


class ReActAgent(Agent):
    """ReAct (Reasoning and Acting) Agent.

    Combines reasoning and action in an iterative loop:
    1. Analyze the problem and plan actions
    2. Call external tools for information
    3. Reason about observations
    4. Iterate until a final answer is reached

    Key design:
    - Multi-turn messages: tool descriptions sent once in system prompt,
      each step appends assistant/user messages incrementally.
    - Tool output truncation to prevent context window bloat.
    - Robust output parsing with OutputParser (retry on failure).
    - Context budget management (auto-trim history to fit token limit).
    - Structured logging and trajectory tracking with timing.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        max_parse_retries: int = 1,
        max_tool_output_chars: int = 8000,
        max_debug_attempts: int = 3,
        enable_debug_loop: bool = True,
        **kwargs,
    ):
        """Initialize ReActAgent.

        Args:
            name: Agent name.
            llm: LLM engine.
            tool_registry: Tool registry (created empty if None).
            system_prompt: System prompt.
            config: Configuration.
            max_steps: Maximum reasoning steps.
            custom_prompt: Custom ReAct prompt template (must contain {tools} placeholder).
            max_parse_retries: Retries when output parsing fails.
            max_tool_output_chars: Maximum chars of tool output to include per message.
            max_debug_attempts: Max consecutive debug retries before giving up on structured debug guidance.
            enable_debug_loop: Whether to inject structured debug context on errors.
            **kwargs: Passed to Agent base (enable_trajectory, enable_logging, etc.).
        """
        super().__init__(name, llm, system_prompt, config, **kwargs)

        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.max_steps = max_steps
        self.prompt_template = custom_prompt if custom_prompt else load_agent_prompt("react")
        self.max_parse_retries = max_parse_retries
        self.max_tool_output_chars = max_tool_output_chars
        self._parser = OutputParser()
        self.max_debug_attempts = max_debug_attempts
        self.enable_debug_loop = enable_debug_loop
        self._debug_prompt_template = load_agent_prompt("debug") if enable_debug_loop else ""

    def add_tool(self, tool):
        """Add a tool to the registry (supports MCP auto-expand)."""
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    from ..tools.base import Tool
                    wrapped_tool = Tool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get('description', ''),
                        func=lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
                self._print(f"  MCP tool '{tool.name}' expanded into {len(tool._available_tools)} tools")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def _build_react_system_prompt(self) -> str:
        """Build the full system prompt with tool descriptions and ReAct format.

        Merges the user-provided system_prompt with the ReAct prompt template
        (which includes tool descriptions). This is constructed once per run()
        call, not on every step.
        """
        tools_desc = self.tool_registry.get_tools_description()
        react_instructions = self.prompt_template.format(tools=tools_desc)

        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(react_instructions)
        return "\n\n".join(parts)

    def _truncate_tool_output(self, output: str) -> str:
        """Truncate tool output to prevent context window bloat.

        Keeps the first and last portions of the output so the LLM
        can see both the beginning and end of long results.
        """
        if len(output) <= self.max_tool_output_chars:
            return output
        half = self.max_tool_output_chars // 2
        return (
            output[:half]
            + f"\n\n... [truncated {len(output) - self.max_tool_output_chars} chars] ...\n\n"
            + output[-half:]
        )

    def run(self, input_text: str, **kwargs) -> str:
        """Run the ReAct agent loop.

        Uses a multi-turn message list that grows incrementally:
        - System message: system_prompt + tool descriptions + ReAct format (built once)
        - User message: the original question (sent once)
        - Then alternating assistant/user messages for each step

        This avoids the O(N^2) token cost of re-serializing the full history
        and tool descriptions on every step.

        Args:
            input_text: User question.
            **kwargs: Extra kwargs for LLM calls.

        Returns:
            Final answer string.
        """
        current_step = 0

        # Start trajectory tracking
        if self.trajectory is not None:
            self.trajectory.reset()
            self.trajectory.start(task=input_text)
        if self.logger:
            self.logger.lifecycle("start", task=input_text)

        self._print(f"\n[{self.name}] Starting question: {input_text}", level="info")

        # Build system prompt with tool descriptions and ReAct format (once)
        system_content = self._build_react_system_prompt()

        # Build multi-turn message list incrementally
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": input_text},
        ]

        # Debug state for structured error recovery
        debug_state = _DebugState()

        while current_step < self.max_steps:
            current_step += 1
            self._print(f"\n--- Step {current_step} ---")

            # Apply context budget management (on a copy to preserve originals)
            trimmed_messages = self._manage_context_budget(list(messages))

            # LLM call with timing
            if self.trajectory is not None:
                self.trajectory.start_timer()

            response_text = self.llm.invoke(trimmed_messages, **kwargs)

            llm_duration = self.trajectory.stop_timer() if self.trajectory else None
            self._track("llm_call", response_text or "(empty)", duration_ms=llm_duration, step=current_step)
            if self.logger:
                self.logger.llm_call(model=self.llm.model, latency_ms=llm_duration or 0)

            if not response_text:
                self._print("  Error: LLM returned an empty response.", level="info")
                self._track("error", "LLM returned empty response")
                break

            # Parse output with robust parser + retry
            thought, action, response_text = self._robust_parse(
                response_text, trimmed_messages, **kwargs
            )

            if thought:
                self._print(f"  Thought: {thought}")
                self._track("thought", thought, step=current_step)
                if self.logger:
                    self.logger.step("thought", thought, step_num=current_step)

            if not action:
                self._print("  Warning: failed to parse a valid Action. Stopping.", level="info")
                self._track("error", "Failed to parse Action from LLM output")
                break

            # Add assistant response to multi-turn history
            messages.append({"role": "assistant", "content": response_text})

            # Check completion
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self._print(f"  Final answer: {final_answer}", level="info")
                self._track("final_answer", final_answer)

                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))

                if self.trajectory is not None:
                    self.trajectory.end()
                if self.logger:
                    self.logger.lifecycle("end", final_answer_length=len(final_answer))

                return final_answer

            # Execute tool call
            tool_name, tool_input = OutputParser.parse_tool_call(action)
            if not tool_name or tool_input is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Observation: Invalid Action format. "
                        'Please use: ToolName[{"key": "value"}] or Finish[answer]'
                    ),
                })
                self._track("error", f"Invalid action format: {action}")
                continue

            self._print(f"  Action: {tool_name}[{tool_input}]")
            self._track("action", f"{tool_name}[{tool_input}]", tool=tool_name)

            # Tool call with timing
            if self.trajectory is not None:
                self.trajectory.start_timer()

            observation = self.tool_registry.execute_tool(tool_name, tool_input)

            tool_duration = self.trajectory.stop_timer() if self.trajectory else None
            obs_preview = observation[:500] + "..." if len(observation) > 500 else observation
            self._print(f"  Observation: {obs_preview}")
            self._track("observation", observation, duration_ms=tool_duration, tool=tool_name)
            if self.logger:
                self.logger.tool_call(tool=tool_name, result=observation, latency_ms=tool_duration or 0)

            # Truncate and add observation as user message (multi-turn)
            truncated_obs = self._truncate_tool_output(observation)
            messages.append({"role": "user", "content": f"Observation: {truncated_obs}"})

            # --- Debug loop integration ---
            if self.enable_debug_loop:
                error_info = self._classify_observation(tool_name, observation)
                if error_info is not None:
                    if not debug_state.active:
                        # Entering debug mode
                        debug_state.active = True
                        debug_state.error_type = error_info["error_type"]
                        debug_state.error_summary = error_info["summary"]
                        debug_state.failed_action = f"{tool_name}[{tool_input}]"
                        debug_state.attempts = 1
                    else:
                        debug_state.attempts += 1

                    if debug_state.attempts <= self.max_debug_attempts:
                        debug_context = self._build_debug_context(debug_state)
                        # Append debug guidance to the observation message
                        messages[-1]["content"] += f"\n\n{debug_context}"
                        self._track("debug", f"attempt {debug_state.attempts}/{self.max_debug_attempts}: {error_info['error_type']}", step=current_step)
                    else:
                        # Max debug attempts exhausted — let the agent continue normally
                        exhaustion_msg = (
                            f"\n\n[Debug loop exhausted ({self.max_debug_attempts} attempts) "
                            f"for {debug_state.error_type}. Proceeding without further debug guidance.]"
                        )
                        messages[-1]["content"] += exhaustion_msg
                        debug_state.reset()
                else:
                    # Successful observation — reset debug state
                    if debug_state.active:
                        self._track("debug_resolved", f"resolved after {debug_state.attempts} attempts", step=current_step)
                        debug_state.reset()

        self._print("  Max steps reached. Stopping.", level="info")
        final_answer = "Unable to complete the task within the allowed number of steps."
        self._track("error", "Max steps reached without finding answer")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        if self.trajectory is not None:
            self.trajectory.end()
        if self.logger:
            self.logger.lifecycle("end", reason="max_steps_reached")

        return final_answer

    # ------------------------------------------------------------------ #
    #  Debug loop helpers
    # ------------------------------------------------------------------ #

    def _classify_observation(self, tool_name: str, observation: str) -> Optional[dict]:
        """Classify a tool observation as an error if applicable.

        Returns None if no error detected, or a dict with:
          {"error_type": str, "summary": str}
        """
        obs_lower = observation.lower()

        # --- code_exec tool ---
        if tool_name == "code_exec":
            # Check for non-zero exit code
            exit_match = re.search(r"exit code:\s*(\d+)", obs_lower)
            if exit_match and exit_match.group(1) != "0":
                # Classify error subtype from stderr/traceback
                error_type = "runtime_error"
                if "syntaxerror" in obs_lower:
                    error_type = "syntax_error"
                elif "importerror" in obs_lower or "modulenotfounderror" in obs_lower:
                    error_type = "import_error"
                elif "timeout" in obs_lower or "timed out" in obs_lower:
                    error_type = "timeout"
                elif any(e in obs_lower for e in ("typeerror", "valueerror", "attributeerror", "nameerror")):
                    error_type = "runtime_error"

                summary = self._extract_error_summary(observation)
                return {"error_type": error_type, "summary": summary}

            # Also catch tracebacks even without explicit exit code
            if "traceback (most recent call last)" in obs_lower:
                error_type = "runtime_error"
                if "syntaxerror" in obs_lower:
                    error_type = "syntax_error"
                elif "importerror" in obs_lower or "modulenotfounderror" in obs_lower:
                    error_type = "import_error"
                summary = self._extract_error_summary(observation)
                return {"error_type": error_type, "summary": summary}

        # --- test_runner tool ---
        if tool_name == "test_runner":
            if "failed" in obs_lower or "error" in obs_lower:
                summary = self._extract_error_summary(observation)
                return {"error_type": "test_failure", "summary": summary}

        # --- Generic fallback for any tool ---
        if "traceback (most recent call last)" in obs_lower:
            summary = self._extract_error_summary(observation)
            return {"error_type": "runtime_error", "summary": summary}

        lines = observation.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Error:") or stripped.startswith("ERROR:"):
                return {"error_type": "runtime_error", "summary": stripped}

        return None

    @staticmethod
    def _extract_error_summary(observation: str) -> str:
        """Extract a one-line error summary from a traceback or test output.

        Returns the last line of a Python traceback (the actual exception),
        or the first line containing 'FAILED'/'ERROR', or a truncated tail.
        """
        lines = observation.strip().splitlines()

        # Walk backwards to find the exception line (last non-empty line after a traceback)
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # Python exception lines typically look like: "ValueError: ..."
            if re.match(r"^[A-Z]\w*(Error|Exception|Warning)", stripped):
                return stripped[:200]
            # Test failure summary lines
            if "FAILED" in stripped or "ERRORS" in stripped:
                return stripped[:200]

        # Fallback: return last non-empty line
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return "Unknown error"

    def _build_debug_context(self, debug_state: _DebugState) -> str:
        """Build a structured debug prompt to guide the LLM through error recovery."""
        return self._debug_prompt_template.format(
            error_type=debug_state.error_type,
            error_summary=debug_state.error_summary,
            failed_action=debug_state.failed_action,
            attempt=debug_state.attempts,
            max_attempts=self.max_debug_attempts,
        )

    # ------------------------------------------------------------------ #
    #  Output parsing with retry
    # ------------------------------------------------------------------ #

    def _robust_parse(
        self, response_text: str, messages: list, **kwargs
    ) -> Tuple[Optional[str], Optional[str], str]:
        """Parse LLM output with retry on failure.

        Uses OutputParser.parse_react() for robust extraction.
        If parsing fails and retries are configured, sends a retry prompt
        to the LLM asking it to correct its format.

        Returns:
            (thought, action, final_response_text) tuple.
            final_response_text may differ from the input if a retry succeeded.
        """
        thought, action = OutputParser.parse_react(response_text)

        if action is not None:
            return thought, action, response_text

        # Retry logic
        for retry in range(self.max_parse_retries):
            retry_prompt = OutputParser.build_retry_prompt(
                original_output=response_text,
                error_message="Could not find 'Action:' field in your response.",
                expected_format="Thought: <your reasoning>\nAction: <tool_name>[<input>] or Finish[<answer>]",
            )
            retry_messages = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": retry_prompt},
            ]

            response_text = self.llm.invoke(retry_messages, **kwargs) or ""
            thought, action = OutputParser.parse_react(response_text)
            if action is not None:
                return thought, action, response_text

        return thought, action, response_text

    def _parse_action_input(self, action_text: str) -> str:
        """Extract content inside Finish[...]."""
        _, content = OutputParser.parse_tool_call(action_text)
        return content if content is not None else ""
