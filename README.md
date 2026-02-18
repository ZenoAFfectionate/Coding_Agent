# Python Coding Agent

> A Python Coding Agent built on the [HelloAgents](https://github.com/datawhalechina/hello-agents) framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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
| Code Search | `CodeSearchTool` | Regex/literal search + AST-based structural queries |
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

- **FunctionCallAgent** — OpenAI-native function calling with parallel tool dispatch
- **ReActAgent** — Reasoning + Acting loop with structured self-debugging
- **ReflectionAgent** — Self-critique and iterative refinement
- **PlanAndSolveAgent** — Decompose complex problems into steps

---

## Implemented Optimizations

The following optimizations were implemented on top of the base HelloAgents framework. They span seven areas: the ReAct agent core loop, agent base class, all agent subclasses, the LLM layer, the tool system, the context/memory layer, and the CLI entry point.

### 1. Structured Self-Debugging Loop (ReActAgent)

**Files:** `code/agents/react_agent.py`, `code/agents/prompts/debug.prompt`, `code/agents/prompts/react.prompt`

The most significant enhancement. When a tool observation contains an error (non-zero exit code, traceback, test failure), the agent automatically enters a **structured debug protocol** instead of relying on the LLM to figure out recovery on its own.

**How it works:**

```
Normal step:  Action → Observation (success) → next Thought
Debug step:   Action → Observation (error detected) → debug context injected → guided Thought
```

- **Error classification** (`_classify_observation`): Inspects each tool observation and classifies errors into categories — `syntax_error`, `import_error`, `runtime_error`, `timeout`, or `test_failure`. Detection is tool-aware: `code_exec` observations are checked for non-zero exit codes and traceback patterns; `test_runner` observations are checked for `FAILED`/`ERROR` lines; all other tools fall back to generic traceback detection.

- **Debug state tracking** (`_DebugState` dataclass): Tracks whether debug mode is active, the error type/summary, the failed action, and the current attempt count. Resets automatically on successful observation.

- **Debug context injection**: When an error is detected, structured guidance is appended to the observation message, walking the LLM through a 5-step protocol:

  > 1. **ANALYZE** — Identify the exact error type, file, and line
  > 2. **READ** — Read the relevant source code (do not skip this step)
  > 3. **DIAGNOSE** — Hypothesize the root cause
  > 4. **FIX** — Apply a minimal, targeted fix
  > 5. **VERIFY** — Re-run the exact command that failed

- **Retry cap** (`max_debug_attempts=3`): After N consecutive failed debug attempts for the same error, the agent receives an exhaustion message and continues without further debug guidance, preventing infinite loops.

- **Trajectory tracking**: Debug events (`debug`, `debug_resolved`) are recorded in the trajectory for post-run analysis.

- **Backward compatible**: Set `enable_debug_loop=False` to disable entirely — the agent behaves identically to the original.

**Key design decision:** The debug loop is *not* a separate execution loop or class. It works entirely within the existing ReAct Thought/Action/Observation cycle by enriching the observation message. This is the lightest-touch change that avoids architectural disruption while providing the most benefit.

### 2. Multi-Turn Message Architecture (ReActAgent)

**File:** `code/agents/react_agent.py`

Replaced the original single-prompt-per-step approach with an **incremental multi-turn message list**:

- The system prompt (including tool descriptions and ReAct format instructions) is built **once** at the start of `run()` and sent as the system message.
- Each step appends only the new assistant response and user observation to the growing message list.
- Previously, every step re-serialized the full history + tool descriptions into a single prompt, resulting in **O(N^2) token cost** across N steps. The new architecture is **O(N)**.

**Tool output truncation** (`max_tool_output_chars=8000`): Long tool outputs are truncated to keep the first and last halves, preventing a single verbose tool response from blowing the context window.

### 3. Robust Output Parsing with Retry (ReActAgent)

**File:** `code/agents/react_agent.py`

Added `_robust_parse()` with configurable retry (`max_parse_retries=1`):

- Uses `OutputParser.parse_react()` for extraction of Thought/Action fields.
- On parse failure, sends a structured retry prompt to the LLM explaining the expected format and asking for correction.
- Handles Qwen3-style `<think>...</think>` blocks and balanced bracket matching for JSON tool arguments.

### 4. Agent Base Class Enhancements

**File:** `code/core/agent.py`

| Enhancement | Description |
|---|---|
| **Cached tiktoken encoder** | The `_count_tokens()` method now caches the tiktoken encoding object at instance level instead of re-importing and re-creating it on every call. |
| **Debug-aware printing** | New `_print(msg, level)` method respects `config.debug`. Level `"debug"` only prints when debug mode is on; `"info"` always prints. All agent subclasses now use this instead of bare `print()`. |
| **History length enforcement** | `add_message()` now trims the oldest messages when `config.max_history_length` is exceeded, preventing unbounded memory growth. |
| **Context budget management** | `_manage_context_budget()` trims the message list to fit within `context_max_tokens`, keeping system prompts and the most recent messages while dropping older history with a summary marker. |

### 5. All Agent Subclasses — Debug-Aware Output & Robustness

**Files:** `code/agents/reflection_agent.py`, `code/agents/plan_solve_agent.py`, `code/agents/function_call_agent.py`

**ReflectionAgent:**
- All `print()` calls replaced with `self._print()` for debug-aware output.
- The `Memory` class now accepts a `debug` flag — memory update messages are only printed when debugging is enabled.
- **Robust stop-condition detection**: The early-termination check was a single hard-coded string (`"no improvement needed"`). It now matches against 10+ common phrasings (`"no further improvement"`, `"the code is correct"`, `"no bugs found"`, etc.) via `_should_stop_refining()`.

**PlanAndSolveAgent:**
- All `print()` calls replaced with debug-aware `self._print()`.
- Both `Planner` and `Executor` accept a `debug` flag, suppressing verbose output in production.
- **Multi-strategy plan parsing**: The planner originally only tried to extract a Python list from a `` ```python `` code block. It now tries three strategies in order:
  1. Python `ast.literal_eval` from code block
  2. JSON parsing via `OutputParser.parse_json()` (handles `` ```json `` blocks and bare JSON)
  3. Regex extraction of numbered lists from plain text (`1. Do X\n2. Do Y`)

**FunctionCallAgent:**
- **Parallel tool execution**: When the model returns multiple tool calls in a single response, they are now executed concurrently via `ThreadPoolExecutor` (up to 4 workers) instead of sequentially. Single tool calls skip the threading overhead entirely.

### 6. LLM Layer Optimization

**File:** `code/core/llm.py`

- **Cached async client**: `ainvoke()` previously created a new `AsyncOpenAI` client on every call. The client is now lazily created once and reused, eliminating redundant connection setup and TLS handshakes.

### 7. Config Modernization

**File:** `code/core/config.py`

- **Environment-driven defaults**: `default_model` and `default_provider` now default to `None` instead of hard-coded values, allowing `HelloAgentsLLM` to auto-detect from environment variables (`LLM_MODEL_ID`, `LLM_PROVIDER`). Eliminates the need to manually edit config when switching providers.
- **`Config.from_env()`** now reads `LLM_MODEL_ID` and `LLM_PROVIDER` from the environment.
- **Pydantic v2 compatibility**: Replaced deprecated `.dict()` with `.model_dump()`.

### 8. AST-Based Code Search

**File:** `code/tools/builtin/code_search_tool.py`

Extended `CodeSearchTool` with three new AST-powered actions beyond the original grep/find_files:

| Action | Description |
|---|---|
| `ast_search` | Structural query for `functions`, `classes`, `imports`, `calls`, or `decorators` across Python files |
| `find_references` | Find all usages of a symbol (name references, attribute access, imports) across the codebase |
| `get_structure` | Generate a file outline showing all classes, methods, functions with line numbers |

These use Python's `ast` module for precise, syntax-aware results rather than regex pattern matching. The agent can now answer structural questions like "find all call sites of `my_func`" or "list all classes in `src/`" without false positives from comments or strings.

### 9. Shell Command Safety

**File:** `code/tools/builtin/code_execution_tool.py`

Added a **blocklist check** for destructive shell patterns before execution:

```
rm -rf /    rm -rf ~    mkfs.*    :(){:|:&};:    > /dev/sd*    dd if=/dev/zero    chmod -R 777 /
```

Commands matching these patterns are blocked with a clear error message. This is a defense-in-depth measure — not a security boundary, but a guard against the most catastrophic accidental commands that an LLM might generate.

### 10. Context & Memory Layer

**Files:** `code/context/builder.py`, `code/memory/types/working.py`

- **Cached tiktoken in `count_tokens()`**: The module-level `count_tokens()` function in the context builder previously re-imported tiktoken and re-created the encoding on every call. Now uses a module-level cached encoding created once on first use.

- **Cached sklearn imports in `WorkingMemory`**: The `retrieve()` method previously imported `TfidfVectorizer`, `cosine_similarity`, and `numpy` inside a try/except on every retrieval call. Now uses a module-level `_HAS_SKLEARN` flag checked once at import time, avoiding repeated failed import attempts when sklearn is not installed.

### 11. CLI & Entry Point

**File:** `run.py`

- **Sandbox mode by default**: When no `--workspace` is specified, the agent creates a temporary directory and registers an `atexit` cleanup handler. This prevents accidental writes to the user's filesystem.
- **Direct workspace mode**: Use `--workspace ./my_project` to operate on real files.
- **Single-shot mode**: `--task "..."` runs one query and exits, suitable for CI/scripting.
- **Multi-line paste detection**: The REPL uses `select()` to detect buffered stdin lines from paste operations.
- **Prompt management**: `PromptManager` loads task-specific prompts from the `prompts/` directory.

---

## Summary of Changes by File

| File | Lines Changed | Key Optimization |
|---|---|---|
| `code/agents/react_agent.py` | +270 | Multi-turn messages, debug loop, output truncation, robust parsing |
| `code/tools/builtin/code_search_tool.py` | +355 | AST-based structural code search |
| `run.py` | +75 | Sandbox mode, single-shot mode, prompt management |
| `code/agents/reflection_agent.py` | +30 | Robust stop detection, debug-aware output |
| `code/agents/plan_solve_agent.py` | +35 | Multi-strategy plan parsing, debug-aware output |
| `code/agents/function_call_agent.py` | +30 | Parallel tool execution |
| `code/core/agent.py` | +30 | Cached tokenizer, debug printing, history limits, context budget |
| `code/core/llm.py` | +10 | Cached async client |
| `code/core/config.py` | +10 | Env-driven defaults, Pydantic v2 compat |
| `code/context/builder.py` | +15 | Cached tiktoken encoding |
| `code/memory/types/working.py` | +10 | Cached sklearn imports |
| `code/tools/builtin/code_execution_tool.py` | +15 | Shell command blocklist |
| `code/agents/prompts/debug.prompt` | New | Structured debug protocol template |
| `code/agents/prompts/react.prompt` | +2 | Debug guidance mention in workflow |

---

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

### Running the Agent

**Interactive REPL (sandbox mode):**

```bash
python run.py
```

**Interactive REPL (on a real project):**

```bash
python run.py --workspace ./my_project
```

**Single-shot mode:**

```bash
python run.py --task "Write a function that sorts a list using merge sort, with tests"
```

### Programmatic Usage

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

### CodeSearchTool

Code search with both text-based (grep/ripgrep) and AST-based structural queries.

**Text search actions:**

| Action | Description |
|---|---|
| `search` | Regex or literal string search across files with context lines |
| `find_files` | Glob-based file discovery |

**AST-based actions:**

| Action | Description |
|---|---|
| `ast_search` | Find `functions`, `classes`, `imports`, `calls`, or `decorators` across Python files |
| `find_references` | Find all usages of a symbol across the codebase |
| `get_structure` | File outline with classes, methods, functions, and line numbers |

```python
from code.tools.builtin import CodeSearchTool

search = CodeSearchTool(workspace="./my_project")

# Find all classes in a directory
result = search.run({"action": "ast_search", "query_type": "classes", "path": "src/"})

# Find all call sites of a function
result = search.run({"action": "ast_search", "query_type": "calls", "symbol": "my_func"})

# Find all references to a symbol
result = search.run({"action": "find_references", "symbol": "MyClass"})

# Get file structure outline
result = search.run({"action": "get_structure", "path": "src/core/agent.py"})
```

### Other Coding Tools

- **FileTool** — `read`, `write`, `edit` (exact string replacement), `list_dir`, `file_info`
- **CodeExecutionTool** — Execute Python or shell code in sandboxed subprocesses (with destructive command blocklist)
- **TestRunnerTool** — `discover`, `run`, `coverage` with pytest/unittest auto-detection
- **GitTool** — `status`, `diff`, `commit`, `log`, `branch`, `checkout`

## Evaluation Benchmarks

The project includes three evaluation suites for measuring agent performance. All evaluations are run through `eval.py`.

### 1. BFCL (Berkeley Function Calling Leaderboard)

Measures an agent's ability to correctly generate **function/tool calls** — choosing the right function, with the right arguments, in the right format.

- **Source:** [gorilla.cs.berkeley.edu/leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | [GitHub](https://github.com/ShishirPatil/gorilla)
- **18 test categories:** `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_parallel`, `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`, etc.
- **Evaluation modes:** AST matching (structural comparison of predicted vs. expected function calls) and execution-based evaluation
- **Metrics:** Overall accuracy, per-category accuracy, parameter-level accuracy, precision/recall/F1, score distribution

```bash
python eval.py --benchmark bfcl --category simple_python --max-samples 50
python eval.py --benchmark bfcl --category multiple --max-samples 50
python eval.py --benchmark bfcl --category parallel --export
```

### 2. GAIA (General AI Assistants)

Measures an agent's ability to solve **real-world questions** requiring multi-step reasoning, web browsing, file processing, and tool use. Developed by Meta, contains 466 questions across 3 difficulty levels.

- **Source:** [arXiv:2311.12983](https://arxiv.org/abs/2311.12983) | [HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA) (gated dataset, requires access approval)
- **Difficulty levels:** Level 1 (direct answers), Level 2 (1-5 step reasoning + simple tools), Level 3 (5+ steps, multi-tool chains)
- **Answer matching:** GAIA-standard normalization (strip articles, currency symbols, number formatting) with exact match and 70% keyword overlap partial match
- **Metrics:** Exact match rate, partial match rate, per-level breakdown, difficulty progression analysis

```bash
python eval.py --benchmark gaia --level 1 --max-samples 20
python eval.py --benchmark gaia --level 2 --max-samples 20
python eval.py --benchmark gaia --lenient --export
```

### 3. Data Generation Quality (AIME / LLM Judge)

Evaluates the quality of **AI-generated math problems** using LLM-as-a-judge, with AIME (American Invitational Mathematics Examination) problems as the reference dataset.

- **Source:** [HuggingFace math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25)
- **Evaluation dimensions:** Correctness, clarity, difficulty match, completeness
- **Methods:** Single-sample LLM Judge scoring, pairwise Win Rate comparison
- **Supports:** Both locally generated problem sets (JSON/JSONL) and AIME real problems from HuggingFace

```bash
python eval.py --benchmark data_gen --data-path data/my_generated_problems.json --max-samples 20
python eval.py --benchmark data_gen --year 2025 --max-samples 15
python eval.py --benchmark data_gen --data-path data/problems.json --judge-model gpt-4o
```

### Common Options

| Flag | Description |
|---|---|
| `--max-samples N` | Limit evaluation to N samples |
| `--output FILE` | Custom output path for results JSON |
| `--export` | Also export in the benchmark's official format |
| `--workspace DIR` | Agent workspace directory |
| `--max-iterations N` | Max agent reasoning steps (default: 15) |
| `--temperature T` | LLM temperature (default: 0.2) |

Results are saved to the `results/` directory by default.

---

## Project Structure

```
CodingAgent/
├── code/
│   ├── agents/                # Agent implementations
│   │   ├── function_call_agent.py   # OpenAI function calling + parallel tool dispatch
│   │   ├── react_agent.py           # ReAct loop + debug loop + multi-turn messages
│   │   ├── reflection_agent.py      # Self-critique with robust stop detection
│   │   ├── plan_solve_agent.py      # Decompose-then-execute with multi-strategy parsing
│   │   └── prompts/
│   │       ├── react.prompt         # ReAct format instructions
│   │       └── debug.prompt         # Structured debug protocol template
│   ├── core/                  # LLM abstraction, base classes, config
│   │   ├── agent.py                 # Base agent with context budget & debug-aware printing
│   │   ├── llm.py                   # Multi-provider LLM client with cached async
│   │   ├── config.py                # Env-driven configuration
│   │   └── output_parser.py         # Robust parsing with retry & auto-repair
│   ├── tools/
│   │   ├── base.py            # Tool ABC, @tool_action decorator
│   │   ├── registry.py        # ToolRegistry
│   │   └── builtin/           # Built-in tools
│   │       ├── file_tool.py
│   │       ├── code_execution_tool.py   # + shell command blocklist
│   │       ├── code_search_tool.py      # + AST-based structural queries
│   │       ├── test_runner_tool.py
│   │       ├── git_tool.py
│   │       ├── linter_tool.py
│   │       ├── profiler_tool.py
│   │       └── terminal_tool.py
│   ├── memory/                # Memory systems (working memory with cached TF-IDF)
│   ├── context/               # Context engineering (cached token counting)
│   ├── protocols/             # MCP, A2A, ANP
│   ├── rl/                    # Reinforcement learning
│   └── evaluation/            # Benchmarks (BFCL, GAIA, Data Generation)
│       └── benchmarks/
│           ├── bfcl/                # Tool calling accuracy evaluation
│           ├── gaia/                # General AI assistant evaluation
│           └── data_generation/     # LLM Judge & Win Rate evaluation
├── data/                      # Datasets (downloaded separately)
│   ├── bfcl/data/                   # BFCL v4 test data + ground truth
│   └── gaia/                        # GAIA questions + attached files
├── run.py                     # CLI entry point (sandbox/direct/single-shot modes)
├── eval.py                    # Evaluation runner for all benchmarks
├── examples/                  # Usage examples
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- [HelloAgents](https://github.com/datawhalechina/hello-agents) by [Datawhale](https://github.com/datawhalechina) — the underlying multi-agent framework
