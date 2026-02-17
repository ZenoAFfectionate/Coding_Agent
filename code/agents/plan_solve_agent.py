"""Plan and Solve Agent -- decompose-then-execute agent."""

import ast
from typing import Optional, List, Dict
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from .prompts import load_agent_prompt


class Planner:
    """Planner -- decomposes a complex problem into simple steps."""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or load_agent_prompt("planner")

    def plan(self, question: str, **kwargs) -> List[str]:
        """Generate an execution plan.

        Args:
            question: The problem to solve.
            **kwargs: LLM call parameters.

        Returns:
            A list of step descriptions.
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- Generating plan ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"  Plan generated:\n{response_text}")

        try:
            # Extract the Python list from the code block
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"  Error: failed to parse plan: {e}")
            print(f"  Raw response: {response_text}")
            return []
        except Exception as e:
            print(f"  Error: unexpected error while parsing plan: {e}")
            return []


class Executor:
    """Executor -- executes the plan step by step."""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or load_agent_prompt("executor")

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """Execute the plan.

        Args:
            question: Original problem.
            plan: Execution plan.
            **kwargs: LLM call parameters.

        Returns:
            The final answer.
        """
        history = ""
        final_answer = ""

        print("\n--- Executing plan ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> Executing step {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "(none)",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            history += f"Step {i}: {step}\nResult: {response_text}\n\n"
            final_answer = response_text
            print(f"  Step {i} complete. Result: {final_answer}")

        return final_answer


class PlanAndSolveAgent(Agent):
    """Plan and Solve Agent -- decompose-then-execute agent.

    Workflow:
    1. Decompose the complex problem into simple steps
    2. Execute each step sequentially
    3. Maintain execution history and context
    4. Produce the final answer

    Well suited for multi-step reasoning, math problems, and complex analysis.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """Initialize PlanAndSolveAgent.

        Args:
            name: Agent name.
            llm: LLM instance.
            system_prompt: System prompt.
            config: Configuration object.
            custom_prompts: Custom prompt templates {"planner": ..., "executor": ...}.
        """
        super().__init__(name, llm, system_prompt, config)

        # Custom prompts take priority; otherwise load from prompt files
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        """Run the Plan and Solve Agent.

        Args:
            input_text: Problem to solve.
            **kwargs: Additional parameters.

        Returns:
            The final answer.
        """
        print(f"\n[{self.name}] Starting problem: {input_text}")

        # 1. Generate plan
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "Failed to generate a valid plan. Task aborted."
            print(f"\n--- Task aborted ---\n{final_answer}")

            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

            return final_answer

        # 2. Execute plan
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- Task complete ---\nFinal answer: {final_answer}")

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer
