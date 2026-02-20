"""Tool Registry — HelloAgents native tool system."""

import json
from typing import Optional, Any, Callable
from .base import Tool

class ToolRegistry:
    """
    HelloAgents Tool Registry.

    Provides tool registration, management, and execution.
    Supports two registration methods:
    1. Tool object registration (recommended)
    2. Direct function registration (convenience)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        """
        Register a Tool object.

        Args:
            tool: Tool instance.
            auto_expand: Whether to auto-expand expandable tools (default True).
        """
        if auto_expand and hasattr(tool, 'expandable') and tool.expandable:
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        print(f"Warning: tool '{sub_tool.name}' already exists, will be overwritten.")
                    self._tools[sub_tool.name] = sub_tool
                print(f"Tool '{tool.name}' expanded into {len(expanded_tools)} sub-tools.")
                return

        if tool.name in self._tools:
            print(f"Warning: tool '{tool.name}' already exists, will be overwritten.")

        self._tools[tool.name] = tool
        print(f"Tool '{tool.name}' registered.")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        Register a function directly as a tool (convenience method).

        Args:
            name: Tool name.
            description: Tool description.
            func: Tool function that accepts a string and returns a string.
        """
        if name in self._functions:
            print(f"Warning: tool '{name}' already exists, will be overwritten.")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"Tool '{name}' registered.")

    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            print(f"Tool '{name}' unregistered.")
        elif name in self._functions:
            del self._functions[name]
            print(f"Tool '{name}' unregistered.")
        else:
            print(f"Warning: tool '{name}' does not exist.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a Tool object by name."""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        Execute a tool.

        Args:
            name: Tool name.
            input_text: Input parameters (JSON string or plain text).

        Returns:
            Tool execution result.
        """
        if name in self._tools:
            tool = self._tools[name]
            try:
                # Try to parse input_text as JSON for structured tools
                try:
                    parameters = json.loads(input_text)
                    if not isinstance(parameters, dict):
                        parameters = {"input": input_text}
                except (json.JSONDecodeError, TypeError):
                    parameters = {"input": input_text}
                return tool.run(parameters)
            except Exception as e:
                return f"Error: exception while executing tool '{name}': {str(e)}"

        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"Error: exception while executing tool '{name}': {str(e)}"

        else:
            return f"Error: tool '{name}' not found."

    def get_tools_description(self) -> str:
        """
        Get a formatted description string of all available tools (with parameters).

        Returns:
            Tool description string for prompt construction.
        """
        descriptions = []

        # Tool object descriptions (with parameters)
        for tool in self._tools.values():
            desc = f"- **{tool.name}**: {tool.description}"
            try:
                params = tool.get_parameters()
                if params:
                    param_parts = []
                    for p in params:
                        required_tag = " (required)" if getattr(p, "required", False) else ""
                        param_parts.append(f'    - `{p.name}` ({p.type}): {p.description}{required_tag}')
                    desc += "\n  Parameters (pass as JSON object):\n" + "\n".join(param_parts)
            except Exception:
                pass
            descriptions.append(desc)

        # Function tool descriptions
        for name, info in self._functions.items():
            descriptions.append(f"- **{name}**: {info['description']}")

        return "\n".join(descriptions) if descriptions else "No available tools"

    def list_tools(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all Tool objects."""
        return list(self._tools.values())

    def clear(self):
        """Clear all tools."""
        self._tools.clear()
        self._functions.clear()
        print("All tools cleared.")

# Global tool registry
global_registry = ToolRegistry()
