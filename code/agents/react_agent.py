from typing import Optional, List, Tuple
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.output_parser import OutputParser
from ..tools.registry import ToolRegistry
from .prompts import load_agent_prompt


class ReActAgent(Agent):
    """ReAct (Reasoning and Acting) Agent.

    Combines reasoning and action in an iterative loop:
    1. Analyze the problem and plan actions
    2. Call external tools for information
    3. Reason about observations
    4. Iterate until a final answer is reached

    Enhanced features:
    - Robust output parsing with OutputParser (retry on failure)
    - Context budget management (auto-trim history to fit token limit)
    - Structured logging
    - Trajectory tracking with timing
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
            custom_prompt: Custom prompt template.
            max_parse_retries: Retries when output parsing fails.
            **kwargs: Passed to Agent base (enable_trajectory, enable_logging, etc.).
        """
        super().__init__(name, llm, system_prompt, config, **kwargs)

        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else load_agent_prompt("react")
        self.max_parse_retries = max_parse_retries
        self._parser = OutputParser()

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
                print(f"  MCP tool '{tool.name}' expanded into {len(tool._available_tools)} tools")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """Run the ReAct agent loop.

        Args:
            input_text: User question.
            **kwargs: Extra kwargs for LLM calls.

        Returns:
            Final answer string.
        """
        self.current_history = []
        current_step = 0

        # Start trajectory tracking
        if self.trajectory is not None:
            self.trajectory.reset()
            self.trajectory.start(task=input_text)
        if self.logger:
            self.logger.lifecycle("start", task=input_text)

        print(f"\n[{self.name}] Starting question: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---")

            # Build prompt
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # Apply context budget management
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages = self._manage_context_budget(messages)

            # LLM call with timing
            if self.trajectory is not None:
                self.trajectory.start_timer()

            response_text = self.llm.invoke(messages, **kwargs)

            llm_duration = self.trajectory.stop_timer() if self.trajectory else None
            self._track("llm_call", response_text or "(empty)", duration_ms=llm_duration, step=current_step)
            if self.logger:
                self.logger.llm_call(model=self.llm.model, latency_ms=llm_duration or 0)

            if not response_text:
                print("  Error: LLM returned an empty response.")
                self._track("error", "LLM returned empty response")
                break

            # Parse output with robust parser + retry
            thought, action = self._robust_parse(response_text, messages, **kwargs)

            if thought:
                print(f"  Thought: {thought}")
                self._track("thought", thought, step=current_step)
                if self.logger:
                    self.logger.step("thought", thought, step_num=current_step)

            if not action:
                print("  Warning: failed to parse a valid Action. Stopping.")
                self._track("error", "Failed to parse Action from LLM output")
                break

            # Check completion
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"  Final answer: {final_answer}")
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
                self.current_history.append("Observation: Invalid Action format.")
                self._track("error", f"Invalid action format: {action}")
                continue

            print(f"  Action: {tool_name}[{tool_input}]")
            self._track("action", f"{tool_name}[{tool_input}]", tool=tool_name)

            # Tool call with timing
            if self.trajectory is not None:
                self.trajectory.start_timer()

            observation = self.tool_registry.execute_tool(tool_name, tool_input)

            tool_duration = self.trajectory.stop_timer() if self.trajectory else None
            print(f"  Observation: {observation}")
            self._track("observation", observation, duration_ms=tool_duration, tool=tool_name)
            if self.logger:
                self.logger.tool_call(tool=tool_name, result=observation, latency_ms=tool_duration or 0)

            # Update history
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        print("  Max steps reached. Stopping.")
        final_answer = "Unable to complete the task within the allowed number of steps."
        self._track("error", "Max steps reached without finding answer")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        if self.trajectory is not None:
            self.trajectory.end()
        if self.logger:
            self.logger.lifecycle("end", reason="max_steps_reached")

        return final_answer

    def _robust_parse(
        self, response_text: str, messages: list, **kwargs
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse LLM output with retry on failure.

        Uses OutputParser.parse_react() for robust extraction.
        If parsing fails and retries are configured, sends a retry prompt
        to the LLM asking it to correct its format.

        Returns:
            (thought, action) tuple.
        """
        thought, action = OutputParser.parse_react(response_text)

        if action is not None:
            return thought, action

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
                return thought, action

        return thought, action

    def _parse_action_input(self, action_text: str) -> str:
        """Extract content inside Finish[...]."""
        _, content = OutputParser.parse_tool_call(action_text)
        return content if content is not None else ""
