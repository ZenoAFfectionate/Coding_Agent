from typing import Optional, Iterator, TYPE_CHECKING
import re

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from .prompts import load_agent_prompt

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):
    """Simple Conversational Agent with optional tool calling support"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        """Initialize SimpleAgent.

        Args:
            name: Agent name.
            llm: LLM instance.
            system_prompt: System prompt.
            config: Configuration object.
            tool_registry: Tool registry (optional; enables tool calling when provided).
            enable_tool_calling: Whether to enable tool calling (only effective with tool_registry).
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def _get_enhanced_system_prompt(self) -> str:
        """Create enhanced system prompt with tool information."""
        base_prompt = self.system_prompt or "You are a helpful AI assistant."

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # Get tool description
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "No available tools":
            return base_prompt

        # Load tool format template and fill in the tools description
        tool_template = load_agent_prompt("simple_tool_format")
        tools_section = tool_template.format(tools_description=tools_description)

        return base_prompt + tools_section

    def _parse_tool_calls(self, text: str) -> list:
        """Parse tool calls in text."""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """Execute a tool call and return the result."""
        if not self.tool_registry:
            return f"Error: tool registry not configured"

        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"Error: tool '{tool_name}' not found"

            param_dict = self._parse_tool_parameters(tool_name, parameters)
            result = tool.run(param_dict)
            return f"Tool {tool_name} result:\n{result}"

        except Exception as e:
            return f"Tool call failed: {str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """Intelligently parse tool parameters from string to dict."""
        import json
        param_dict = {}

        # Try JSON format
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.loads(parameters)
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                pass

        if '=' in parameters:
            if ',' in parameters:
                # Multiple parameters: action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # Single parameter: key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()

            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # Infer action if not specified
            if 'action' not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # Direct parameter text -- infer based on tool type
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """Convert parameter types based on the tool's parameter definitions."""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # Build parameter type mapping
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == 'number' or param_type == 'integer':
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """Infer the action parameter based on tool type and parameters."""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('recall')
            elif 'store' in param_dict:
                param_dict['action'] = 'add'
                param_dict['content'] = param_dict.pop('store')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('search')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'

        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """Infer a full parameter dict from a simple text argument."""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """Run SimpleAgent with optional tool calling.

        Args:
            input_text: User input.
            max_tool_iterations: Maximum tool call iterations (only with tools enabled).
            **kwargs: Additional parameters.

        Returns:
            Agent response.
        """
        messages = []

        # Add system message (may include tool information)
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # Add history messages
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": input_text})

        # If tool calling is disabled, use simple path
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        # Iterative processing with tool calls
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            response = self.llm.invoke(messages, **kwargs)

            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    clean_response = clean_response.replace(call['original'], "")

                messages.append({"role": "assistant", "content": clean_response})

                tool_results_text = "\n\n".join(tool_results)
                messages.append({
                    "role": "user",
                    "content": f"Tool execution results:\n{tool_results_text}\n\nPlease provide a complete answer based on these results."
                })

                current_iteration += 1
                continue

            final_response = response
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """Add a tool to the agent.

        Args:
            tool: Tool object.
            auto_expand: Whether to auto-expand expandable tools (default True).
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool by name."""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """List all available tools."""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """Check whether tools are available."""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """Stream agent output.

        Args:
            input_text: User input.
            **kwargs: Additional parameters.

        Yields:
            Response chunks.
        """
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
