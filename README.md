# Python Coding Agent

> A Python Coding Agent built on the [HelloAgents](https://github.com/datawhalechina/hello-agents) framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference)

This project implements a **Python Coding Agent** on top of the HelloAgents multi-agent framework. The agent can read, write, search, execute, test, lint, and profile Python code autonomously — functioning as an AI-powered software engineering assistant.

## Overview

The Coding Agent extends HelloAgents' tool-based architecture with a comprehensive set of software engineering tools. Following HelloAgents' design philosophy of "everything is a tool", each capability — from file I/O to Git operations to performance profiling — is encapsulated as a `Tool` subclass that the agent invokes through function calling.

### Core Capabilities

| Capability | Tool | Description |
|---|---|---|
| File Operations | `FileTool` | Read, write, edit files with path sandboxing |
| Code Execution | `CodeExecutionTool` | Run Python/shell code in isolated subprocesses |
| Code Search | `CodeSearchTool` | Search codebases by symbol, pattern, or text |
| Test Running | `TestRunnerTool` | Discover and run tests (pytest/unittest), collect coverage |
| Git Operations | `GitTool` | Stage, commit, diff, log, branch management |
| Linting | `LinterTool` | Static analysis and auto-fix (ruff/flake8/py_compile) |
| Formatting | `LinterTool` | Code formatting (ruff format/black) |
| CPU Profiling | `ProfilerTool` | cProfile-based hotspot analysis |
| Benchmarking | `ProfilerTool` | timeit-based snippet benchmarking |
| Memory Profiling | `ProfilerTool` | tracemalloc-based memory snapshots |
| Terminal | `TerminalTool` | General shell command execution |

### Agent Paradigms

The project inherits all agent paradigms from HelloAgents:

- **FunctionCallAgent** — OpenAI-native function calling for tool dispatch
- **ReActAgent** — Reasoning + Acting loop with tool use
- **ReflectionAgent** — Self-critique and iterative refinement
- **PlanAndSolveAgent** — Decompose complex problems into steps

## Quick Start

### Requirements

- Python 3.12+
- An OpenAI-compatible LLM endpoint (local or remote)

### Installation

```bash
git clone <this-repo>
cd CodingAgent
pip install -e .[all]
```

### Environment Configuration

Create a `.env` file:

```bash
LLM_MODEL_ID=your-model-name
LLM_API_KEY=your-api-key
LLM_BASE_URL=your-api-base-url
```

For local model deployment (e.g., with vLLM):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### Basic Usage

```python
from code.agents import FunctionCallAgent
from code.core.llm import HelloAgentsLLM
from code.tools.builtin import FileTool, CodeExecutionTool, LinterTool, ProfilerTool

llm = HelloAgentsLLM()

# Register coding tools
from code.tools.registry import ToolRegistry
registry = ToolRegistry()
registry.register_tool(FileTool(workspace="./my_project"))
registry.register_tool(CodeExecutionTool(workspace="./my_project"))
registry.register_tool(LinterTool(workspace="./my_project"))
registry.register_tool(ProfilerTool(workspace="./my_project"))

# Create the coding agent
agent = FunctionCallAgent(
    name="CodingAgent",
    llm=llm,
    tool_registry=registry,
    system_prompt="You are a Python coding assistant with access to file, execution, linting, and profiling tools."
)

response = agent.run("Read main.py and check it for lint errors")
```

## Tool Details

### LinterTool

Static analysis, auto-fix, and code formatting with automatic backend detection.

**Actions:**

| Action | Description | Backend Priority |
|---|---|---|
| `check` | Run linter, report issues | ruff > flake8 > py_compile (stdlib) |
| `fix` | Auto-fix lint issues in-place | ruff `--fix` |
| `format` | Format code to style guidelines | ruff format > black |

```python
from code.tools.builtin import LinterTool

linter = LinterTool(workspace="./my_project")

# Check for lint errors
result = linter.run({"action": "check", "path": "src/main.py"})

# Auto-fix issues
result = linter.run({"action": "fix", "path": "src/"})

# Format code
result = linter.run({"action": "format", "path": "src/main.py"})
```

The `py_compile` fallback is always available (stdlib) and provides syntax checking even when no external linter is installed.

### ProfilerTool

Performance profiling using Python stdlib — no external dependencies required.

**Actions:**

| Action | Description | Backend |
|---|---|---|
| `profile` | CPU profile a file, show top-N hotspots | `cProfile` + `pstats` |
| `timeit` | Benchmark a code snippet | `timeit` |
| `memory` | Memory allocation snapshot | `tracemalloc` |

```python
from code.tools.builtin import ProfilerTool

profiler = ProfilerTool(workspace="./my_project")

# CPU profile a file
result = profiler.run({"action": "profile", "path": "src/main.py", "top_n": 10})

# Benchmark a snippet
result = profiler.run({
    "action": "timeit",
    "code": "[i**2 for i in range(1000)]",
    "number": 10000,
    "repeat": 5
})

# Memory profiling
result = profiler.run({
    "action": "memory",
    "code": "x = [i for i in range(100000)]",
    "top_n": 10
})
```

### Other Coding Tools

- **FileTool** — `read`, `write`, `edit` (exact string replacement), `list_dir`, `file_info`
- **CodeExecutionTool** — Execute Python or shell code in sandboxed subprocesses
- **CodeSearchTool** — Search by symbol name, regex pattern, or plain text across a codebase
- **TestRunnerTool** — `discover`, `run`, `coverage` with pytest/unittest auto-detection
- **GitTool** — `status`, `diff`, `commit`, `log`, `branch`, `checkout`

## Project Structure

```
CodingAgent/
├── code/
│   ├── agents/                # Agent implementations
│   │   ├── function_call_agent.py
│   │   ├── react_agent.py
│   │   ├── reflection_agent.py
│   │   ├── plan_solve_agent.py
│   │   └── simple_agent.py
│   ├── core/                  # LLM abstraction, base classes
│   ├── tools/
│   │   ├── base.py            # Tool ABC, @tool_action decorator
│   │   ├── registry.py        # ToolRegistry
│   │   └── builtin/           # Built-in tools
│   │       ├── file_tool.py
│   │       ├── code_execution_tool.py
│   │       ├── code_search_tool.py
│   │       ├── test_runner_tool.py
│   │       ├── git_tool.py
│   │       ├── linter_tool.py
│   │       ├── profiler_tool.py
│   │       ├── terminal_tool.py
│   │       └── ...
│   ├── memory/                # Memory systems
│   ├── context/               # Context engineering
│   ├── protocols/             # MCP, A2A, ANP
│   ├── rl/                    # Reinforcement learning
│   └── evaluation/            # Benchmarks (BFCL, GAIA)
├── examples/                  # Usage examples
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- [HelloAgents](https://github.com/datawhalechina/hello-agents) by [Datawhale](https://github.com/datawhalechina) — the underlying multi-agent framework
