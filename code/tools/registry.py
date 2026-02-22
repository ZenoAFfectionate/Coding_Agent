"""Tool Registry — HelloAgents native tool system."""

import json
import logging
from typing import Optional, Any, Callable
from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    HelloAgents Tool Registry.

    Provides tool registration, management, and execution.
    Supports two registration methods:
    1. Tool object registration (recommended)
    2. Direct function registration (convenience)
    """

    _CONTENT_FIELDS = frozenset({"content", "code", "new_string"})

    @staticmethod
    def _normalize_content_escapes(parameters: dict) -> dict:
        """Post-process parsed JSON parameters to fix double-escaped sequences.

        Smaller LLMs sometimes emit ``\\\\n`` in JSON strings.  After
        ``json.loads`` this becomes a literal backslash + ``n`` instead of a
        real newline.  This method replaces those literal escape sequences in
        known content-bearing fields only, so patterns, paths, etc. are never
        touched.

        To avoid breaking source code that intentionally contains literal
        backslash-n (e.g. ``'\\n'.join(...)`` in Python), normalization is
        only applied when the content contains **no** real newline characters —
        i.e. when all newlines appear to have been double-escaped by the LLM.
        If the value already has real newlines, the LLM encoded them correctly
        and any remaining literal ``\\n`` sequences are intentional.
        """
        for key in ToolRegistry._CONTENT_FIELDS:
            val = parameters.get(key)
            if isinstance(val, str):
                # If the value already contains real newlines, the LLM
                # used JSON escapes correctly.  Any remaining literal \n
                # sequences are intentional (e.g. Python '\n' in source
                # code) and must NOT be converted.
                if "\n" in val:
                    continue
                # No real newlines present → the LLM probably double-escaped.
                # Order matters: resolve \\\\ first so that a sequence like
                # \\\\n (four-char) doesn't become a bare newline.
                val = val.replace("\\\\", "\x00")
                val = val.replace("\\n", "\n")
                val = val.replace("\\t", "\t")
                val = val.replace("\x00", "\\")
                parameters[key] = val
        return parameters

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
                        logger.warning("Tool '%s' already exists, will be overwritten.", sub_tool.name)
                    self._tools[sub_tool.name] = sub_tool
                logger.debug("Tool '%s' expanded into %d sub-tools.", tool.name, len(expanded_tools))
                return

        if tool.name in self._tools:
            logger.warning("Tool '%s' already exists, will be overwritten.", tool.name)

        self._tools[tool.name] = tool
        logger.debug("Tool '%s' registered.", tool.name)

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        Register a function directly as a tool (convenience method).

        Args:
            name: Tool name.
            description: Tool description.
            func: Tool function that accepts a string and returns a string.
        """
        if name in self._functions:
            logger.warning("Tool '%s' already exists, will be overwritten.", name)

        self._functions[name] = {
            "description": description,
            "func": func
        }
        logger.debug("Tool '%s' registered.", name)

    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.debug("Tool '%s' unregistered.", name)
        elif name in self._functions:
            del self._functions[name]
            logger.debug("Tool '%s' unregistered.", name)
        else:
            logger.warning("Tool '%s' does not exist.", name)

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
                    else:
                        parameters = self._normalize_content_escapes(parameters)
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
        logger.debug("All tools cleared.")

# Global tool registry
global_registry = ToolRegistry()
