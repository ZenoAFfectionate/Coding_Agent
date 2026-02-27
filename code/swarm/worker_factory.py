"""Worker agent factory — build specialized FunctionCallAgent workers."""

from code.core.llm import HelloAgentsLLM
from code.core.config import Config
from code.agents.function_call_agent import FunctionCallAgent
from code.tools.builtin.finish_tool import FinishTool
from code.tools.builtin.escalate_tool import EscalateTool
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.code_search_tool import CodeSearchTool
from code.tools.builtin.test_runner_tool import TestRunnerTool
from code.tools.builtin.git_tool import GitTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool

from .paths import load_swarm_prompt

# Tool class mapping — each key maps to a factory that builds the tool.
_TOOL_CONSTRUCTORS = {
    "file":        lambda ws: FileTool(workspace=ws),
    "code_exec":   lambda ws: CodeExecutionTool(workspace=ws, timeout=30),
    "code_search": lambda ws: CodeSearchTool(workspace=ws),
    "test_runner": lambda ws: TestRunnerTool(project_path=ws, timeout=120),
    "git":         lambda ws: GitTool(repo_path=ws),
    "linter":      lambda ws: LinterTool(workspace=ws, timeout=30),
    "profiler":    lambda ws: ProfilerTool(workspace=ws, timeout=60),
}


def _build_tools(workspace: str, tool_names: list[str]) -> ToolRegistry:
    """Build a ToolRegistry from a list of tool name keys."""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register_tool(_TOOL_CONSTRUCTORS[name](workspace))
    return registry


WORKER_SPECS = {
    "review": {
        "prompt": "code_review",
        "tools": ["file", "code_search", "linter", "git"],
        "description": "Code review specialist: analyzes code quality, security, design, and correctness.",
    },
    "test": {
        "prompt": "test_generation",
        "tools": ["file", "code_exec", "code_search", "test_runner"],
        "description": "Test generation specialist: writes and runs comprehensive test suites.",
    },
    "optimize": {
        "prompt": "optimization",
        "tools": ["file", "code_exec", "code_search", "profiler", "linter"],
        "description": "Optimization specialist: profiles code and applies targeted performance improvements.",
    },
    "debug": {
        "prompt": "debug",
        "tools": ["file", "code_exec", "code_search", "test_runner", "git", "linter"],
        "description": "Debug specialist: reproduces, diagnoses, and fixes bugs systematically.",
    },
}


def build_worker(
    worker_type: str,
    workspace: str,
    llm: HelloAgentsLLM,
    config: Config,
    max_steps: int = 10,
) -> FunctionCallAgent:
    """Create a specialized worker FunctionCallAgent.

    Args:
        worker_type: One of "review", "test", "optimize", "debug".
        workspace: Root directory the agent operates in.
        llm: Shared LLM instance (same instance used by orchestrator and all workers).
        config: Shared Config object.
        max_steps: Max tool-calling rounds for this worker.

    Returns:
        A ready-to-use FunctionCallAgent with task-specific prompt and tools.
    """
    spec = WORKER_SPECS[worker_type]

    # Combine the base system prompt with the task-specific prompt
    base_system = load_swarm_prompt("system") or "You are an expert Python coding assistant."
    task_prompt = load_swarm_prompt(spec["prompt"])
    system_prompt = base_system
    if task_prompt:
        system_prompt += f"\n\n<task-instructions>\n{task_prompt}\n</task-instructions>"

    registry = _build_tools(workspace, spec["tools"])
    registry.register_tool(FinishTool())
    registry.register_tool(EscalateTool())

    return FunctionCallAgent(
        name=f"Worker_{worker_type}",
        llm=llm,
        system_prompt=system_prompt,
        tool_registry=registry,
        max_steps=max_steps,
        config=config,
    )
