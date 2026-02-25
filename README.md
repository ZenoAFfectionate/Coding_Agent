# Python Coding Agent

> A Python Coding Agent built on the [HelloAgents](https://github.com/datawhalechina/hello-agents) framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference)

This project implements a **Python Coding Agent** on top of the HelloAgents multi-agent framework. It supports both **single-agent** and **multi-agent** modes. The agent can read, write, search, execute, test, lint, and profile Python code autonomously — functioning as an AI-powered software engineering assistant.

## Overview

The Coding Agent extends HelloAgents' tool-based architecture with a comprehensive set of software engineering tools. Following HelloAgents' design philosophy of "everything is a tool", each capability — from file I/O to Git operations to performance profiling — is encapsulated as a `Tool` subclass that the agent invokes through function calling.

The project provides two execution modes:

- **Single-agent mode** (`inference.py` / `tool_agent.py`): A ReActAgent or FunctionCallAgent equipped with all coding tools, supporting interactive REPL and batch inference.
- **Multi-agent mode** (`run_multi.py`): An orchestrator agent that delegates to specialized worker agents (review, test, optimize, debug), each with a curated subset of tools.

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

The project inherits all agent paradigms from HelloAgents, plus new additions:

- **ReActAgent** — Reasoning + Acting loop with structured self-debugging and reflection
- **FunctionCallAgent** — OpenAI-native function calling with parallel tool dispatch
- **ReflectionAgent** — Self-critique and iterative refinement
- **PlanAndSolveAgent** — Decompose complex problems into steps
- **SimpleAgent** — Lightweight conversational agent with optional tool calling
- **ToolAwareSimpleAgent** — SimpleAgent with tool call monitoring and logging
- **OrchestratorAgent** — Multi-agent master that delegates to specialized workers

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
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```


```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve zai-org/GLM-4.7-Flash \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser glm47
```

### Running the Agent

**Single-agent — Interactive REPL (sandbox mode):**

```bash
python inference.py                # ReAct agent
python tool_agent.py               # Function-calling agent
```

**Single-agent — Interactive REPL (on a real project):**

```bash
python inference.py --workspace ./my_project
```

**Single-agent — With session persistence (save/restore across restarts):**

```bash
python inference.py --restore                          # auto session file
python tool_agent.py --session-file my_session.json    # custom path
```

**Single-agent — Batch mode:**

```bash
python inference.py --batch --input data/xCode/valid.jsonl --output data/xCode/result.jsonl
```

**Multi-agent — Interactive REPL (sandbox mode):**

```bash
python run_multi.py
```

**Multi-agent — Single-shot mode:**

```bash
python run_multi.py --task "Review and optimize my_project/main.py" --workspace ./my_project
```

### Programmatic Usage

```python
from code.agents.react_agent import ReActAgent
from code.core.llm import HelloAgentsLLM
from code.core.config import Config
from code.tools.registry import ToolRegistry
from code.tools.builtin.file_tool import FileTool
from code.tools.builtin.code_execution_tool import CodeExecutionTool
from code.tools.builtin.linter_tool import LinterTool
from code.tools.builtin.profiler_tool import ProfilerTool

llm = HelloAgentsLLM()
config = Config()

# Register coding tools
registry = ToolRegistry()
registry.register_tool(FileTool(workspace="./my_project"))
registry.register_tool(CodeExecutionTool(workspace="./my_project"))
registry.register_tool(LinterTool(workspace="./my_project"))
registry.register_tool(ProfilerTool(workspace="./my_project"))

# Create the coding agent
agent = ReActAgent(
    name="CodingAgent",
    llm=llm,
    system_prompt="You are a Python coding assistant with access to file, execution, linting, and profiling tools.",
    tool_registry=registry,
    max_steps=15,
    config=config,
)

response = agent.run("Read main.py and check it for lint errors")
```

---

## Single-Agent Workflow Example

This section illustrates the concrete execution flow when the single agent (`inference.py`) solves a coding problem. The trace below is based on actual agent output using `Qwen/Qwen3-30B-A3B-Thinking-2507`.

### Execution Flow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Initialization                          │
│  1. Create sandbox directory (or use --workspace)           │
│  2. Register tools: file, code_exec, code_search,           │
│     test_runner, git, linter, profiler                      │
│  3. Load system prompt & build ReActAgent                   │
│  4. Start REPL (interactive) or batch loop                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   User Input (Query)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Loop                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LLM Call  ──►  Thought  ──►  Action  ──►  Observe  │◄─┐│
│  └────────────────────────────────────────────┬────────┘  ││
│                                               │           ││
│                                 ┌─────────────┴────┐      ││
│                                 │  Error detected? │      ││
│                                 └──┬───────────┬───┘      ││
│                                  No│          Yes│         ││
│                                    │   ┌────────▼───────┐  ││
│                                    │   │  Debug Protocol │  ││
│                                    │   │ (up to 3 tries) │  ││
│                                    │   └────────┬───────┘  ││
│                                    ▼            ▼          ││
│                              ┌──────────────────────┐      ││
│                              │  Finish[] or next    ├──────┘│
│                              │  iteration           │       │
│                              └──────────────────────┘       │
└──────────────────────────┬──────────────────────────────────┘
                           │ Action: Finish[answer]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Reflection                             │
│  Self-verify: APPROVED / NEEDS_REVISION (up to N retries)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Post-Processing                           │
│  1. Print final answer                                      │
│  2. Save trajectory to results/trajectories/                │
│  3. Display generated code files (sandbox mode)             │
└─────────────────────────────────────────────────────────────┘
```

### Traced Example

Below is an abbreviated trace of the agent solving a competitive programming problem (dynamic container cost calculation). The full trace is available in `output.txt`.

**Step 1 — LLM Call (Reasoning):** The model receives the problem, reasons through the algorithm (understanding the push/pop/resize mechanics, deriving the O(n) simulation approach), and decides to write the solution to a file.

```
[Step 1]   LLM Call  (26944ms)
  <think>
  ... (model reasons about the problem, derives the algorithm,
       traces through both examples to verify logic) ...

  The condition for shrinking is: size * 4 <= capacity (avoids floating point).
  The code should be O(n) — each operation is O(1).
  </think>

  Thought: I need to implement the cost calculation. I'll write the solution
           to solution.py and verify with the provided examples.
  Action: file[{"action": "write", "path": "solution.py", "content": "..."}]
```

**Steps 2–4 — Tool Execution (Write File):**

```
[Step 2]   Thought   — Parsed reasoning from LLM output
[Step 3]   Action    — file[{"action": "write", "path": "solution.py", ...}]
[Step 4]   Observation  (1ms)
             Successfully wrote 437 bytes (21 lines) to solution.py
```

**Steps 5–8 — Test with Example 1:**

```
[Step 5]   LLM Call  (12338ms)
  Thought: Verify the code against Example 1. Expected output: 6.
  Action: code_exec[{"action": "shell", "code": "echo '...' | python solution.py"}]

[Step 8]   Observation  (46ms)
             Exit code: 0
             stdout: 6                          ← ✓ Matches expected
```

**Steps 9–12 — Test with Example 2:**

```
[Step 9]   LLM Call  (8595ms)
  Thought: Example 1 passed. Now verify Example 2. Expected output: 12.
  Action: code_exec[{"action": "shell", "code": "echo '...' | python solution.py"}]

[Step 12]  Observation  (31ms)
             Exit code: 0
             stdout: 12                         ← ✓ Matches expected
```

**Steps 13–16 — Conclude & Reflect:**

```
[Step 13]  LLM Call  (5621ms)
  Thought: Both examples pass. The logic is correct. No changes needed.
  Action: Finish[The solution correctly calculates the total cost ...]

[Step 15]  Reflection
             approved=True: The solution correctly handles the resizing
             conditions and cost calculations.

[Step 16]  Final Answer
             The solution correctly calculates the total cost for the
             container operations as verified by the provided examples.
```

**Post-Processing:**

```
============================================================
 Summary: 16 steps | Duration: 72.47s
============================================================
[Trajectory] Saved to results/trajectories/single_agent_trajectory.json
============================================================
  Generated Code Files
============================================================
--- solution.py ---
n = int(input().strip())
size = 0
capacity = 1
total_cost = 0
...
============================================================
```

### Key Observations

- **Reasoning-first:** The model spends the majority of time in the first LLM call (~27s), where it fully understands the problem and derives the algorithm before writing any code.
- **Write-then-verify pattern:** The agent writes the complete solution in one shot, then tests it against the provided examples using shell execution.
- **Self-verification:** Even after confirming both test cases pass, the agent goes through a reflection step that reviews whether the solution is complete and correct before finalizing.
- **Minimal iteration:** For straightforward problems, the agent converges in a single write + test cycle (no debug loop triggered). The structured debug protocol activates only when tool observations contain errors.

> **Note:** This trace covers the **single-agent** mode only. Multi-agent mode (orchestrator + specialized workers) follows a different delegation-based workflow and is not shown here.

---

## Implemented Optimizations

The following optimizations were implemented on top of the base HelloAgents framework. They span the ReAct agent core loop, agent base class, all agent subclasses, the LLM layer, the tool system, the context/memory layer, and the CLI entry point.

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

### 2. Prompt Architecture & Text Processing (ReActAgent)

**File:** `code/agents/react_agent.py`

**Prompt construction**: Each ReAct step calls `_build_prompt()` which concatenates the system prompt, ReAct format template (with tool descriptions), the current question, and the full execution history into a single prompt. The execution history grows incrementally — each step appends its Thought/Action/Observation triplet to a `List[str]`, and the entire list is serialized into the prompt for the next LLM call.

**Tool output truncation** (`max_tool_output_chars=8000`): Long tool outputs are truncated to keep the first and last halves, preventing a single verbose tool response from blowing the context window.

**Qwen3 think-tag handling** (`_strip_think_tags`): Removes `<think>...</think>` blocks emitted by thinking models (e.g. Qwen-Thinking series). Also handles orphaned `</think>` tags (missing opening `<think>`) common when serving via vLLM — everything before the last `</think>` is treated as thinking content and stripped.

**Parse error recovery**: The LLM response is parsed via regex to extract `Thought:` and `Action:` fields. If the response lacks a valid `Action:` field, a structured error message is injected into the history (e.g. "you must include an Action"), and the loop continues — the next LLM call sees the error and can self-correct.

### 3. Agent Base Class Enhancements

**File:** `code/core/agent.py`

| Enhancement | Description |
|---|---|
| **Cached tiktoken encoder** | The `_count_tokens()` method now caches the tiktoken encoding object at instance level instead of re-importing and re-creating it on every call. |
| **Debug-aware printing** | New `_print(msg, level)` method respects `config.debug`. Level `"debug"` only prints when debug mode is on; `"info"` always prints. All agent subclasses now use this instead of bare `print()`. |
| **History length enforcement** | `add_message()` now trims the oldest messages when `config.max_history_length` is exceeded, preventing unbounded memory growth. |
| **Rich history entries** | `_build_execution_summary()` extracts tool calls, files modified, errors, and key observations from the trajectory, and appends this execution context to the assistant message stored in `_history`. This gives the LLM full awareness of what happened in previous REPL turns — not just the bare Q&A, but which tools were called, what files were touched, and what errors occurred. |
| **Claude Code-style context compaction** | `_manage_context_budget()` implements full conversation compaction modeled after Claude Code's approach. See details below. |
| **Session persistence** | `save_session()` / `load_session()` serialize `_history` to a JSON file, enabling conversation continuity across process restarts. See details below. |

#### Context Compaction (Claude Code-style)

When the conversation approaches the token budget, the **entire conversation body** (everything except system prompts and the current user input) is compressed into a single LLM-generated summary in one operation. This is fundamentally different from incremental summarization, which would trigger frequently as new messages accumulate.

**How it works:**

```
Before compaction:
  [system_prompt] [user1] [assistant1] [tool_result1] ... [userN]

After compaction:
  [system_prompt] [user: "conversation compacted..."] [assistant: <summary>] [userN]
```

- **Trigger:** Fires when total context tokens reach `COMPACTION_THRESHOLD` (85%) of `context_max_tokens`. For example, with a 200K budget, compaction triggers at ~170K tokens.
- **Summary placement:** The summary is stored as a **user/assistant message pair**, matching the Claude API compaction pattern — the summary prompt is injected as a user turn, and the LLM-generated summary is stored as an assistant turn. This preserves proper conversation turn-taking and causes the model to treat the summary as its own prior context rather than a system instruction.
- **Prompt template:** `prompts/context_compaction.prompt` — instructs the LLM to preserve file paths, function names, error messages, decisions and rationale, and current state. The prompt explicitly distinguishes what to preserve (exact paths, error messages, unresolved issues) from what to compress (verbose tool output, repeated attempts, filler).
- **Fallback:** `_mechanical_summary()` provides a best-effort structural extraction (user requests, assistant responses, tool results) without calling the LLM, used when the LLM call fails.

This mirrors how production coding agents handle context overflow:
- **Claude Code:** When context reaches ~95% capacity, auto-compact analyzes the full conversation, creates a compressed summary, and replaces old messages. Users can also run `/compact` manually.
- **Aider:** Uses `--weak-model` to summarize chat history when it exceeds `max_chat_history_tokens`.

#### Rich History Entries

**Files:** `code/agents/react_agent.py`, `code/agents/function_call_agent.py`

The `_end_run()` method in both ReActAgent and FunctionCallAgent now calls `_build_execution_summary()` before storing the assistant message in `_history`. The summary appends structured execution context to the answer:

```
<actual answer>

---
[Execution context for next turn]
Tools called: file, code_exec, test_runner
Files touched: src/main.py, tests/test_main.py
Errors encountered: ImportError: no module named foo
Key observations:
  - def main(): ...
  - All 5 tests passed
```

This means the next REPL turn's LLM call sees what the agent actually did, not just the final answer — closing the gap between the rich per-run trajectory and the bare cross-turn history.

#### Session Persistence

**Files:** `code/core/agent.py`, `inference.py`, `tool_agent.py`

Session history is serialized to JSON on REPL exit and restored on startup:

- `save_session(path)` — writes `_history` (with timestamps, roles, metadata) to a JSON file.
- `load_session(path)` — reads a session file and populates `_history`.
- `get_default_session_path(agent_name)` — returns a default path under `results/sessions/`.

**CLI flags:**

| Flag | Description |
|---|---|
| `--restore` | Restore previous session on startup (uses default session file path) |
| `--session-file PATH` | Use a custom session file path |

**REPL commands:**

| Command | Description |
|---|---|
| `/save` | Manually save session to file |
| `/history` | Show conversation history summary (message count, role, content preview) |
| `/compact` | Note: compaction happens automatically when `context_max_tokens` is set |

### 4. All Agent Subclasses — Debug-Aware Output & Robustness

**Files:** `code/agents/reflection_agent.py`, `code/agents/plan_solve_agent.py`, `code/agents/function_call_agent.py`

**ReflectionAgent:**
- All `print()` calls replaced with `self._print()` for debug-aware output.
- The `Memory` class now accepts a `debug` flag — memory update messages are only printed when debugging is enabled.
- **Robust stop-condition detection**: The early-termination check was a single hard-coded string (`"no improvement needed"`). It now matches against 10+ common phrasings (`"no further improvement"`, `"the code is correct"`, `"no bugs found"`, etc.) via `_should_stop_refining()`.

**PlanAndSolveAgent:**
- Agent-level output uses debug-aware `self._print()`. Both `Planner` and `Executor` accept a `debug` flag, suppressing verbose output in production.
- **Multi-strategy plan parsing**: The planner originally only tried to extract a Python list from a `` ```python `` code block. It now tries three strategies in order:
  1. Python `ast.literal_eval` from code block
  2. JSON parsing via `json.loads()` (handles `` ```json `` blocks and bare JSON)
  3. Regex extraction of numbered lists from plain text (`1. Do X\n2. Do Y`)

**FunctionCallAgent:**
- **Parallel tool execution**: When the model returns multiple tool calls in a single response, they are now executed concurrently via `ThreadPoolExecutor` (up to 4 workers) instead of sequentially. Single tool calls skip the threading overhead entirely.

### 5. LLM Layer Optimization

**File:** `code/core/llm.py`

- **Cached async client**: `ainvoke()` previously created a new `AsyncOpenAI` client on every call. The client is now lazily created once and reused, eliminating redundant connection setup and TLS handshakes.

### 6. Config Modernization

**File:** `code/core/config.py`

- **Environment-driven defaults**: `default_model` and `default_provider` now default to `None` instead of hard-coded values, allowing `HelloAgentsLLM` to auto-detect from environment variables (`LLM_MODEL_ID`, `LLM_PROVIDER`). Eliminates the need to manually edit config when switching providers.
- **`Config.from_env()`** now reads `LLM_MODEL_ID` and `LLM_PROVIDER` from the environment.
- **Pydantic v2 compatibility**: Replaced deprecated `.dict()` with `.model_dump()`.

### 7. AST-Based Code Search

**File:** `code/tools/builtin/code_search_tool.py`

Extended `CodeSearchTool` with three new AST-powered actions beyond the original grep/find_files:

| Action | Description |
|---|---|
| `ast_search` | Structural query for `functions`, `classes`, `imports`, `calls`, or `decorators` across Python files |
| `find_references` | Find all usages of a symbol (name references, attribute access, imports) across the codebase |
| `get_structure` | Generate a file outline showing all classes, methods, functions with line numbers |

These use Python's `ast` module for precise, syntax-aware results rather than regex pattern matching. The agent can now answer structural questions like "find all call sites of `my_func`" or "list all classes in `src/`" without false positives from comments or strings.

### 8. Shell Command Safety

**File:** `code/tools/builtin/code_execution_tool.py`

Added a **blocklist check** for destructive shell patterns before execution:

```
rm -rf /    rm -rf ~    mkfs.*    :(){:|:&};:    > /dev/sd*    dd if=/dev/zero    chmod -R 777 /
```

Commands matching these patterns are blocked with a clear error message. This is a defense-in-depth measure — not a security boundary, but a guard against the most catastrophic accidental commands that an LLM might generate.

### 9. Context & Memory Layer

**Files:** `code/context/builder.py`, `code/memory/types/working.py`

- **Cached tiktoken in `count_tokens()`**: The module-level `count_tokens()` function in the context builder previously re-imported tiktoken and re-created the encoding on every call. Now uses a module-level cached encoding created once on first use.

- **Cached sklearn imports in `WorkingMemory`**: The `retrieve()` method previously imported `TfidfVectorizer`, `cosine_similarity`, and `numpy` inside a try/except on every retrieval call. Now uses a module-level `_HAS_SKLEARN` flag checked once at import time, avoiding repeated failed import attempts when sklearn is not installed.

### 10. CLI & Entry Points

**Files:** `inference.py`, `tool_agent.py`, `run_multi.py`

**Single-agent mode — ReAct (`inference.py`):**

- **Two modes:** Interactive REPL (default) and batch (`--batch --input data.jsonl`).
- **Sandbox mode by default**: When no `--workspace` is specified, the agent creates a temporary directory and registers an `atexit` cleanup handler. This prevents accidental writes to the user's filesystem.
- **Direct workspace mode**: Use `--workspace ./my_project` to operate on real files.
- **Reflection / self-verification**: Enabled by default. Disable with `--no-reflection`, or configure retries with `--max-reflection-retries N`.
- **Session persistence**: Use `--restore` to save/restore conversation history across process restarts. Use `--session-file PATH` for a custom session file location.
- **REPL commands**: `/help`, `/save` (save session), `/history` (show conversation summary), `/compact` (context compaction note).
- **Batch mode**: Processes JSONL problem sets, extracts generated code from sandbox, and produces result JSONL files.
- **Multi-line paste detection**: The REPL uses `select()` to detect buffered stdin lines from paste operations.
- **Prompt management**: Loads the system prompt from `prompts/system.prompt`.

**Single-agent mode — Function Calling (`tool_agent.py`):**

- Same REPL, sandbox, reflection, and session persistence features as `inference.py`.
- Uses `FunctionCallAgent` with OpenAI-native function calling and parallel tool dispatch.
- **Plan mode**: Use `--plan` to enable plan-then-execute mode where the LLM outlines a plan before acting.

**Multi-agent mode (`run_multi.py`):**

- **Orchestrator + Workers architecture**: The orchestrator dispatches tasks to specialized workers (review, test, optimize, debug), each with its own system prompt and curated tool set.
- **Blackboard memory**: A shared `Blackboard` object accumulates findings and errors across workers, injected into the orchestrator's context.
- **Dual communication mode**: Function calling (default) with automatic fallback to text-based JSON parsing for providers without function calling support.
- **Result summarization**: LLM-based summarization of long worker results, with truncation fallback.
- **Reflection on final answer**: The orchestrator self-critiques its synthesized answer before returning.
- **CLI options**: `--max-worker-steps`, `--max-rounds`, `--max-result-chars`, `--context-max-tokens`, `--no-fc`, `--no-summarize`, `--no-reflect`.

---

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

The project includes four evaluation suites for measuring agent performance. All evaluations are run through `evaluation.py`.

### 1. BFCL (Berkeley Function Calling Leaderboard)

Measures an agent's ability to correctly generate **function/tool calls** — choosing the right function, with the right arguments, in the right format.

- **Source:** [gorilla.cs.berkeley.edu/leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | [GitHub](https://github.com/ShishirPatil/gorilla)
- **18 test categories:** `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_parallel`, `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`, etc.
- **Evaluation modes:** AST matching (structural comparison of predicted vs. expected function calls) and execution-based evaluation
- **Metrics:** Overall accuracy, per-category accuracy, parameter-level accuracy, precision/recall/F1, score distribution

```bash
python evaluation.py --benchmark bfcl --category simple_python --max-samples 50
python evaluation.py --benchmark bfcl --category multiple --max-samples 50
python evaluation.py --benchmark bfcl --category parallel --export
```

### 2. GAIA (General AI Assistants)

Measures an agent's ability to solve **real-world questions** requiring multi-step reasoning, web browsing, file processing, and tool use. Developed by Meta, contains 466 questions across 3 difficulty levels.

- **Source:** [arXiv:2311.12983](https://arxiv.org/abs/2311.12983) | [HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA) (gated dataset, requires access approval)
- **Difficulty levels:** Level 1 (direct answers), Level 2 (1-5 step reasoning + simple tools), Level 3 (5+ steps, multi-tool chains)
- **Answer matching:** GAIA-standard normalization (strip articles, currency symbols, number formatting) with exact match and 70% keyword overlap partial match
- **Metrics:** Exact match rate, partial match rate, per-level breakdown, difficulty progression analysis

```bash
python evaluation.py --benchmark gaia --level 1 --max-samples 20
python evaluation.py --benchmark gaia --level 2 --max-samples 20
python evaluation.py --benchmark gaia --lenient --export
```

### 3. SWE-bench Verified (SWEV)

A curated 500-instance subset of SWE-bench, hand-verified for solvability. Measures an agent's ability to resolve real-world **GitHub issues** by generating code patches.

- **Source:** [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- **Instances:** 500 verified issues from popular Python repos
- **Metrics:** Resolved rate, exact match, line overlap

```bash
python evaluation.py --benchmark swev --agent-type react --max-samples 2

python evaluation.py --benchmark swev --agent-type funca --max-samples 2

# Filter by repo
python evaluation.py --benchmark swev --agent-type react --repo-filter django/django --export
```

### 4. Data Generation Quality (AIME / LLM Judge)

Evaluates the quality of **AI-generated math problems** using LLM-as-a-judge, with AIME (American Invitational Mathematics Examination) problems as the reference dataset.

- **Source:** [HuggingFace math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25)
- **Evaluation dimensions:** Correctness, clarity, difficulty match, completeness
- **Methods:** Single-sample LLM Judge scoring, pairwise Win Rate comparison
- **Supports:** Both locally generated problem sets (JSON/JSONL) and AIME real problems from HuggingFace

```bash
python evaluation.py --benchmark data_gen --data-path data/my_generated_problems.json --max-samples 20
python evaluation.py --benchmark data_gen --year 2025 --max-samples 15
python evaluation.py --benchmark data_gen --data-path data/problems.json --judge-model gpt-4o
```

### Common Options

| Flag | Description |
|---|---|
| `--max-samples N` | Limit evaluation to N samples |
| `--output FILE` | Custom output path for results JSON |
| `--export` | Also export in the benchmark's official format |
| `--agent-type TYPE` | Agent type: `react` or `funca` (default: react) |
| `--workspace DIR` | Agent workspace directory |
| `--max-iterations N` | Max agent reasoning steps (default: 15) |
| `--temperature T` | LLM temperature (default: 0.2) |
| `--split SPLIT` | Dataset split for SWEV (default: test) |

Results are saved to the `results/` directory by default.

---

## Project Structure

```
CodingAgent/
├── code/
│   ├── agents/                # Agent implementations
│   │   ├── react_agent.py           # ReAct loop + debug loop + multi-turn messages + reflection
│   │   ├── function_call_agent.py   # OpenAI function calling + parallel tool dispatch
│   │   ├── reflection_agent.py      # Self-critique with robust stop detection
│   │   ├── plan_solve_agent.py      # Decompose-then-execute with multi-strategy parsing
│   │   ├── simple_agent.py          # Lightweight conversational agent
│   │   ├── tool_aware_agent.py      # SimpleAgent with tool call monitoring
│   │   └── prompts/
│   │       ├── react.prompt         # ReAct format instructions
│   │       ├── debug.prompt         # Structured debug protocol template
│   │       └── ...                  # Other agent prompt templates
│   ├── core/                  # LLM abstraction, base classes, config
│   │   ├── agent.py                 # Base agent with context compaction, rich history,
│   │   │                            #   session persistence & debug-aware printing
│   │   ├── llm.py                   # Multi-provider LLM client with cached async
│   │   ├── config.py                # Env-driven configuration
│   │   ├── message.py               # Message dataclass
│   │   ├── exceptions.py            # Custom exceptions
│   │   └── database_config.py       # Database configuration
│   ├── tools/
│   │   ├── base.py            # Tool ABC, @tool_action decorator
│   │   ├── registry.py        # ToolRegistry
│   │   ├── async_executor.py  # Async tool execution
│   │   ├── chain.py           # Tool chaining
│   │   └── builtin/           # Built-in tools
│   │       ├── file_tool.py
│   │       ├── code_execution_tool.py   # + shell command blocklist
│   │       ├── code_search_tool.py      # + AST-based structural queries
│   │       ├── test_runner_tool.py
│   │       ├── git_tool.py
│   │       ├── linter_tool.py
│   │       ├── profiler_tool.py
│   │       ├── terminal_tool.py
│   │       ├── calculator.py
│   │       ├── search_tool.py
│   │       ├── memory_tool.py
│   │       ├── note_tool.py
│   │       ├── rag_tool.py
│   │       ├── mcp_wrapper_tool.py
│   │       ├── protocol_tools.py
│   │       ├── rl_training_tool.py
│   │       └── ...                      # Evaluation-specific tools
│   ├── utils/                 # Shared utilities
│   │   ├── helpers.py
│   │   ├── logging.py
│   │   ├── serialization.py
│   │   ├── subprocess_utils.py
│   │   └── trajectory.py
│   ├── memory/                # Memory systems (working memory with cached TF-IDF)
│   ├── context/               # Context engineering (cached token counting)
│   ├── protocols/             # MCP, A2A, ANP
│   ├── rl/                    # Reinforcement learning
│   └── evaluation/            # Benchmarks (BFCL, GAIA, SWE-bench, Data Generation)
│       └── benchmarks/
│           ├── bfcl/                # Tool calling accuracy evaluation
│           ├── gaia/                # General AI assistant evaluation
│           ├── swe/                 # SWE-bench / SWE-bench Verified evaluation
│           └── data_generation/     # LLM Judge & Win Rate evaluation
├── prompts/                   # System & task prompts (for multi-agent workers)
│   ├── system.prompt                # Shared system prompt
│   ├── orchestrator.prompt          # Orchestrator text-mode prompt
│   ├── orchestrator_fc.prompt       # Orchestrator function-calling prompt
│   ├── context_compaction.prompt    # Context compaction summary template
│   ├── execution_summary.prompt     # Per-run execution summary template
│   ├── code_review.prompt           # Review worker prompt
│   ├── test_generation.prompt       # Test worker prompt
│   ├── optimization.prompt          # Optimization worker prompt
│   ├── debug.prompt                 # Debug worker prompt
│   └── ...
├── data/                      # Datasets (downloaded separately)
│   ├── BFCL/                        # BFCL test data + ground truth
│   ├── GAIA/                        # GAIA questions + attached files
│   ├── SWEV/                        # SWE-bench Verified instances (500)
│   ├── AIME/                        # AIME math problems
│   ├── KodCode/                     # KodCode dataset
│   └── xCode/                       # xCode dataset (batch mode input)
├── inference.py               # Single-agent entry point — ReAct (REPL / batch)
├── tool_agent.py              # Single-agent entry point — Function Calling (REPL / batch)
├── run_multi.py               # Multi-agent entry point (orchestrator + workers)
├── evaluation.py              # Evaluation runner for all benchmarks
├── results/                   # Evaluation and inference results
│   └── sessions/              # Saved session history files (JSON)
├── examples/                  # Usage examples
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- [HelloAgents](https://github.com/datawhalechina/hello-agents) by [Datawhale](https://github.com/datawhalechina) — the underlying multi-agent framework
