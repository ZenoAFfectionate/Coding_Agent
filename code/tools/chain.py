"""Tool Chain Manager — HelloAgents chained tool execution support."""

from typing import List, Dict, Any, Optional
from .registry import ToolRegistry


class ToolChain:
    """Tool Chain — supports sequential execution of multiple tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        Add a tool execution step.

        Args:
            tool_name: Tool name.
            input_template: Input template supporting variable substitution, e.g. "{input}" or "{search_result}".
            output_key: Output key name for referencing in subsequent steps.
        """
        step = {
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        }
        self.steps.append(step)
        print(f"Chain '{self.name}' added step: {tool_name}")

    def execute(self, registry: ToolRegistry, input_data: str, context: Dict[str, Any] = None) -> str:
        """
        Execute the tool chain.

        Args:
            registry: Tool registry.
            input_data: Initial input data.
            context: Execution context for variable substitution.

        Returns:
            Final execution result.
        """
        if not self.steps:
            return "Error: tool chain is empty, cannot execute."

        print(f"Starting tool chain: {self.name}")

        if context is None:
            context = {}
        context["input"] = input_data

        final_result = input_data

        for i, step in enumerate(self.steps):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            print(f"Executing step {i+1}/{len(self.steps)}: {tool_name}")

            try:
                actual_input = input_template.format(**context)
            except KeyError as e:
                return f"Error: template variable substitution failed: {e}"

            try:
                result = registry.execute_tool(tool_name, actual_input)
                context[output_key] = result
                final_result = result
                print(f"Step {i+1} completed.")
            except Exception as e:
                return f"Error: tool '{tool_name}' execution failed: {e}"

        print(f"Tool chain '{self.name}' completed.")
        return final_result


class ToolChainManager:
    """Tool Chain Manager."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """Register a tool chain."""
        self.chains[chain.name] = chain
        print(f"Tool chain '{chain.name}' registered.")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """Execute a registered tool chain."""
        if chain_name not in self.chains:
            return f"Error: tool chain '{chain_name}' does not exist."

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """List all registered tool chain names."""
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """Get tool chain information."""
        if chain_name not in self.chains:
            return None

        chain = self.chains[chain_name]
        return {
            "name": chain.name,
            "description": chain.description,
            "steps": len(chain.steps),
            "step_details": [
                {
                    "tool_name": step["tool_name"],
                    "input_template": step["input_template"],
                    "output_key": step["output_key"]
                }
                for step in chain.steps
            ]
        }


# Convenience functions
def create_research_chain() -> ToolChain:
    """Create a research tool chain: search -> calculate -> summarize."""
    chain = ToolChain(
        name="research_and_calculate",
        description="Search for information and perform related calculations"
    )

    # Step 1: search for information
    chain.add_step(
        tool_name="search",
        input_template="{input}",
        output_key="search_result"
    )

    # Step 2: calculate based on search results
    chain.add_step(
        tool_name="my_calculator",
        input_template="2 + 2",  # simple calculation example
        output_key="calc_result"
    )

    return chain


def create_simple_chain() -> ToolChain:
    """Create a simple tool chain example."""
    chain = ToolChain(
        name="simple_demo",
        description="Simple tool chain demo"
    )

    # Single calculation step
    chain.add_step(
        tool_name="my_calculator",
        input_template="{input}",
        output_key="result"
    )

    return chain
