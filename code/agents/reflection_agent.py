"""Reflection Agent -- self-reflective iterative refinement agent."""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from .prompts import load_agent_prompt


def _load_default_prompts() -> Dict[str, str]:
    """Load the default reflection prompt templates from disk."""
    return {
        "initial": load_agent_prompt("reflection_initial"),
        "reflect": load_agent_prompt("reflection_reflect"),
        "refine":  load_agent_prompt("reflection_refine"),
    }


class Memory:
    """Short-term memory for tracking execution attempts and reflections."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """Add a new record to memory."""
        self.records.append({"type": record_type, "content": content})
        print(f"  Memory updated: new '{record_type}' record added.")

    def get_trajectory(self) -> str:
        """Format all memory records into a coherent trajectory string."""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Previous Attempt ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer Feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """Get the most recent execution result."""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""


class ReflectionAgent(Agent):
    """Reflection Agent -- self-reflective iterative refinement agent.

    Workflow:
    1. Execute the initial task
    2. Self-reflect on the result
    3. Refine based on reflection feedback
    4. Iterate until satisfactory or max iterations reached

    Well suited for code generation, document writing, analytical reports,
    and other tasks that benefit from iterative refinement.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """Initialize ReflectionAgent.

        Args:
            name: Agent name.
            llm: LLM instance.
            system_prompt: System prompt.
            config: Configuration object.
            max_iterations: Maximum refinement iterations.
            custom_prompts: Custom prompt templates {"initial", "reflect", "refine"}.
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()

        # Custom prompts take priority; otherwise load from prompt files
        self.prompts = custom_prompts if custom_prompts else _load_default_prompts()

    def run(self, input_text: str, **kwargs) -> str:
        """Run the Reflection Agent.

        Args:
            input_text: Task description.
            **kwargs: Additional LLM parameters.

        Returns:
            The final refined result.
        """
        print(f"\n[{self.name}] Starting task: {input_text}")

        # Reset memory
        self.memory = Memory()

        # 1. Initial execution
        print("\n--- Initial attempt ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. Iterative loop: reflect and refine
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

            # a. Reflect
            print("\n-> Reflecting...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. Check if refinement is needed
            if "no improvement needed" in feedback.lower():
                print("\n  Reflection indicates no further improvement needed. Done.")
                break

            # c. Refine
            print("\n-> Refining...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- Task complete ---\nFinal result:\n{final_result}")

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """Call LLM and return the full response."""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
