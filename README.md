# Python Coding Agent

> A Python Coding Agent built on the [HelloAgents](https://github.com/datawhalechina/hello-agents) framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference)

This project implements a **Python Coding Agent** on top of the HelloAgents multi-agent framework. It supports both **single-agent** and **multi-agent** modes. The agent can read, write, search, execute, test, lint, and profile Python code autonomously — functioning as an AI-powered software engineering assistant.

## Overview

The Coding Agent extends HelloAgents' tool-based architecture with a comprehensive set of software engineering tools. Following HelloAgents' design philosophy of "everything is a tool", each capability — from file I/O to Git operations to performance profiling — is encapsulated as a `Tool` subclass that the agent invokes through function calling.

The project provides two execution modes:

- **Single-agent mode** (`react_agent.py` / `funca_agent.py`): A ReActAgent or FunctionCallAgent equipped with all coding tools, supporting interactive REPL and batch inference.
- **Multi-agent mode** (`run_swarm.py`): An orchestrator agent that delegates to specialized worker agents (review, test, optimize, debug), each with a curated subset of tools.

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
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

### Running the Agent

**Single-agent — Interactive REPL (sandbox mode):**

```bash
python react_agent.py              # ReAct agent (text-based, works with any LLM)
python funca_agent.py              # Function-calling agent (native tool calling)
```

**Single-agent — Interactive REPL (on a real project):**

```bash
python react_agent.py --workspace ./my_project
python funca_agent.py --workspace ./my_project
```

**Single-agent — With session persistence (save/restore across restarts):**

```bash
python react_agent.py --restore                          # auto session file
python funca_agent.py --session-file my_session.json     # custom path
```

**Single-agent — Batch mode:**

```bash
python react_agent.py --batch --input data/xCode/valid.jsonl --output data/xCode/result.jsonl
python funca_agent.py --batch --input data/xCode/valid.jsonl --output data/xCode/result.jsonl
```

**Multi-agent — Interactive REPL (sandbox mode):**

```bash
python run_swarm.py
```

**Multi-agent — Single-shot mode:**

```bash
python run_swarm.py --task "Review and optimize my_project/main.py" --workspace ./my_project
```

**Multi-agent — Single-shot mode (with task from file):**

```bash
python run_swarm.py --task "$(cat data/xCode/simple_test.txt)"
```

**Multi-agent — Batch mode on xCode dataset (with detailed logging):**

```bash
# Batch inference — process xCode problems with multi-agent collaboration
# Detailed interaction logs (orchestrator decisions, worker dispatches,
# blackboard state, tool calls) are written to results/logs/multi_agent.log
python run_swarm.py --batch \
    --input data/xCode/valid.jsonl \
    --output data/xCode/result_multi.jsonl \
    --limit 10 \
    --max-worker-steps 20 \
    --max-rounds 6 \
    --temperature 0.2

# View detailed multi-agent interaction log
cat results/logs/multi_agent.log

# Custom log directory
python run_swarm.py --batch \
    --input data/xCode/test.jsonl \
    --output results/swarm_test_results.jsonl \
    --log-dir results/logs/custom \
    --limit 5

# Disable reflection and summarization for faster runs
python run_swarm.py --batch \
    --input data/xCode/valid.jsonl \
    --output data/xCode/result_multi_fast.jsonl \
    --no-reflect --no-summarize --limit 5
```

### ReAct Agent vs. Function Calling Agent

The project provides two single-agent paradigms with distinct strengths, selectable via the entry point script:

**ReAct Agent** (`react_agent.py`) uses a text-based Thought/Action/Observation loop. The LLM generates free-form reasoning (`Thought: ...`) followed by a tool invocation (`Action: tool_name[json_args]`), which the agent parses via regex. This approach is **universally compatible** — it works with any LLM, including open-source models that do not support native function calling (e.g., Qwen-Thinking, DeepSeek, LLaMA). It is the best choice when working with local deployments or non-OpenAI-compatible providers.

**Function Calling Agent** (`funca_agent.py`) uses the OpenAI native function calling protocol. Tool schemas are automatically generated from `ToolParameter` definitions and passed to the LLM via the `tools` API parameter. The LLM returns structured JSON tool calls rather than free-form text, eliminating parsing ambiguity. When the model returns multiple tool calls in a single response, they are executed **in parallel** via `ThreadPoolExecutor`. This paradigm is the default for models that support it, offering higher reliability and lower latency.

Both agents share the same underlying capabilities — debug loop, reflection, context compaction, session persistence, and the full tool inventory — differing only in how they communicate with the LLM.

```python
# ReActAgent — works with any LLM
from code.agents.react_agent import ReActAgent

agent = ReActAgent(
    name="CodingAgent",
    llm=llm,
    system_prompt="You are a Python coding assistant.",
    tool_registry=registry,
    max_steps=16,
    enable_reflection=True,
    enable_debug_loop=True,
)
response = agent.run("Read main.py and fix the bug on line 42")
```

```python
# FunctionCallAgent — native structured tool calling + parallel execution
from code.agents.function_call_agent import FunctionCallAgent

agent = FunctionCallAgent(
    name="CodingAgent",
    llm=llm,
    system_prompt="You are a Python coding assistant.",
    tool_registry=registry,
    max_steps=32,
    enable_reflection=True,
    enable_debug_loop=True,
)
response = agent.run("Read main.py and fix the bug on line 42")
```

---

## Coding Agent Workflow

This section illustrates the concrete execution flow when the single agent (`react_agent.py` / `funca_agent.py`) solves a coding problem. The trace below is based on actual agent output using `Qwen/Qwen3.5-30B-A3B`.

### Single-Agent Execution Flow Overview

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

### Multi-Agent Execution Flow Overview



---

## Key Technical Highlights

Beyond the base HelloAgents framework, this project introduces a series of deeply engineered mechanisms that push the coding agent toward production-grade reliability and autonomy. The following sections detail the core technical contributions.

### 1. Token-Budget-Aware Dynamic Context Compaction

Long-running coding tasks — multi-file refactors, iterative debugging sessions, test-fix-retest cycles — accumulate tool outputs and conversation history at a pace that quickly exhausts the LLM's context window. Rather than naively truncating messages or hoping the model stays within limits, the agent implements a **full-conversation compaction strategy inspired by Claude Code's auto-compact mechanism**.

The core idea is straightforward: when the total token count of the conversation reaches **85% of the configured budget** (e.g., ~170K out of 200K tokens), the system triggers a one-shot compaction pass. The entire conversation body — every user message, assistant response, and tool result accumulated so far, *except* the system prompt and the most recent user input — is fed to the LLM with a specialized summarization prompt. The LLM produces a structured summary that deliberately preserves the information a coding agent needs for continued reasoning: exact file paths, function signatures, error messages, unresolved issues, and key decisions made so far. Verbose tool outputs, repeated failed attempts, and conversational filler are aggressively compressed.

The compacted summary is then injected back into the context as a **user/assistant message pair** — not a system instruction. This design choice mirrors the Claude API's compaction pattern: by placing the summary as the assistant's own prior output, the model treats it as established context rather than an external directive, producing more natural continuation behavior and preserving correct turn-taking for OpenAI-compatible APIs.

```python
# code/core/agent.py — _manage_context_budget()

COMPACTION_THRESHOLD = 0.85

def _manage_context_budget(self, messages: list[dict]) -> list[dict]:
    if self.context_max_tokens <= 0:
        return messages

    total = self._count_messages_tokens(messages)
    trigger = int(self.context_max_tokens * self.COMPACTION_THRESHOLD)
    if total <= trigger:
        return messages

    # Separate system prompts, conversation body, and current input.
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]
    current_input = non_system[-1]
    conversation_body = non_system[:-1]

    # Compact the entire conversation body into one LLM-generated summary.
    summary_text = self._compact_messages(conversation_body)

    # Reassemble: [system] + [user: compaction notice] + [assistant: summary] + [current input]
    result = list(system_msgs)
    result.append({"role": "user", "content": "The conversation history has been compacted. ..."})
    result.append({"role": "assistant", "content": summary_text})
    result.append(current_input)
    return result
```

A mechanical fallback (`_mechanical_summary()`) provides best-effort structural extraction without an LLM call in case the summarization request itself fails. The ReActAgent additionally performs inline history compaction on its text-based prompt, compressing older Thought/Action/Observation entries while preserving the most recent steps in full fidelity. Combined with a **step-budget warning** that injects a convergence prompt at 75% step consumption, these mechanisms ensure the agent can sustain arbitrarily long coding sessions without context degradation.

### 2. Full-Lifecycle Software Engineering Toolchain with Built-in Security

The agent is equipped with **11 specialized tools** that collectively cover the entire software engineering workflow — from reading source code to profiling production performance. Each tool is a `Tool` subclass with its own JSON schema, and every tool embeds defense-in-depth security mechanisms rather than relying on a single outer sandbox.

**Path sandboxing** is enforced at the tool level. The `FileTool`, for example, resolves every path against the workspace root and rejects any path that escapes via `../` traversal or symlink indirection. Edits to Python files are validated with `ast.parse()` before being committed — if the edit introduces a syntax error, the file is **automatically reverted** to its pre-edit state, and the agent receives a clear rejection message:

```python
# code/tools/builtin/file_tool.py

def _safe_path(self, rel_path: str) -> Optional[Path]:
    """Resolve rel_path inside the workspace, blocking escapes."""
    resolved = (self.workspace / rel_path).resolve()
    try:
        resolved.relative_to(self.workspace)
    except ValueError:
        return None  # path escape blocked
    return resolved

def _validate_and_maybe_revert(self, path, modified, original, rel) -> Optional[str]:
    """For .py files, reject edits that introduce syntax errors."""
    if not self.lint_on_edit or path.suffix != ".py":
        return None
    try:
        ast.parse(modified, filename=str(path))
        return None
    except SyntaxError as e:
        path.write_text(original, encoding="utf-8")  # auto-revert
        return f"Edit rejected: syntax error in {rel} — {e.msg} (line {e.lineno})."
```

**Code execution** runs in isolated subprocesses with process-group-level timeout enforcement. The `safe_run()` utility creates each subprocess in its own process group (`start_new_session=True`) and, on timeout, sends `SIGKILL` to the entire group — eliminating orphaned child processes that `subprocess.kill()` alone would miss. A blocklist intercepts catastrophic shell commands (`rm -rf /`, fork bombs, `dd` to block devices) before they reach the shell, and a stdin-hang detector identifies code that would block on `input()` / `sys.stdin.read()` and rejects it immediately rather than waiting for a 30-second timeout.

```python
# code/utils/subprocess_utils.py

def safe_run(*args, timeout=30, input=None, **kwargs) -> subprocess.CompletedProcess:
    kwargs["start_new_session"] = True  # own process group for clean kill
    proc = subprocess.Popen(*args, **kwargs)
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
        return subprocess.CompletedProcess(...)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)  # SIGKILL to the entire process tree
        raise

def _kill_process_group(proc):
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGKILL)
```

**Code search** goes beyond text-based grep with a full **AST-powered structural search engine** and a **PageRank-based repository map**. The `ast_search` action uses Python's `ast` module to find functions, classes, imports, decorators, and call sites with zero false positives from comments or string literals. The `repo_map` action builds an import dependency graph across the entire codebase and ranks files using PageRank (optionally personalized by query keywords), giving the agent a high-level architectural overview before diving into specific files:

```python
# code/tools/builtin/code_search_tool.py

def _rank_graph(self, graph, query=None) -> List[Tuple[str, float]]:
    """Rank files using PageRank, optionally personalized by query keywords."""
    personalization = None
    if query:
        keywords = query.lower().split()
        personalization = {
            node: sum(1.0 for name, _, _, _ in graph.nodes[node].get("defs", [])
                       for kw in keywords if kw in name.lower()) + 0.1
            for node in graph.nodes
        }
    scores = nx.pagerank(graph, personalization=personalization)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

When the model returns multiple tool calls in a single response, they are dispatched concurrently via **`ThreadPoolExecutor`** (up to 4 workers), with results reassembled in the original call order. Single tool calls skip the threading overhead entirely.

```python
# code/agents/function_call_agent.py

from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
    futures = [pool.submit(_exec, tc) for tc in tool_calls]
    by_id = {f.result()[0]: f.result() for f in as_completed(futures)}
executed = [(*by_id[tc.id], None) for tc in tool_calls]  # preserve original order
```

The complete tool inventory: **FileTool** (read/write/edit/undo with syntax validation), **CodeExecutionTool** (Python & shell with sandbox), **CodeSearchTool** (regex + AST + PageRank repo map), **TestRunnerTool** (pytest/unittest auto-detection + coverage), **GitTool** (status/diff/log/commit/branch/blame with destructive-op blocking), **LinterTool** (ruff/flake8/py_compile + black formatter), **ProfilerTool** (cProfile/timeit/tracemalloc), plus supporting tools for terminal, search, memory, and notes.

### 3. Blackboard-Based Multi-Agent Orchestration Framework

For complex tasks that benefit from division of labor — such as "review this codebase, write tests, optimize the hotspots, and fix any bugs" — the system provides a **multi-agent orchestration framework** built on a shared Blackboard architecture.

The **Orchestrator agent** acts as a project manager: it receives the user's request, formulates a plan, and iteratively dispatches subtasks to four types of **expert worker agents** — code review, test generation, performance optimization, and debugging. Each expert is a fully autonomous `FunctionCallAgent` with its own dedicated system prompt and a curated subset of tools appropriate to its specialty. The code review expert, for instance, has access to file reading, code search, linting, and git tools but *not* code execution — it cannot accidentally modify the codebase while reviewing it.

```python
# run_swarm.py

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
```

The **Blackboard** is a structured shared memory that accumulates findings and errors across all worker rounds. Each expert's execution result is summarized by the LLM before being written to the blackboard, and the full blackboard state is serialized and injected into the orchestrator's context before each dispatch decision. This design achieves two critical properties: (1) information flows across agents without requiring them to share conversation histories, and (2) each expert's internal tool-calling details remain invisible to the orchestrator, keeping per-layer context sizes manageable.

```python
# run_swarm.py — Blackboard

class Blackboard:
    """Structured workspace state shared across the orchestrator and workers."""
    def __init__(self, user_request: str):
        self.user_request = user_request
        self.findings: list[dict] = []   # {"source", "round", "summary"}
        self.errors: list[dict] = []     # {"source", "round", "message"}
        self.current_plan: str = ""

    def serialize(self) -> str:
        """Render the blackboard as a concise text block for injection into prompts."""
        parts = [f"User request: {self.user_request}"]
        if self.findings:
            parts.append("\nFindings:")
            for f in self.findings:
                parts.append(f"  - [{f['source']} R{f['round']}] {f['summary']}")
        return "\n".join(parts)
```

The orchestrator communicates with the LLM via **Function Calling** by default (using `dispatch_worker` and `finish` tool schemas), but automatically degrades to a **text-mode JSON parsing fallback** when the provider does not support function calling — a single try/except permanently switches modes for the rest of the session, ensuring zero retries are wasted:

```python
# run_swarm.py — OrchestratorAgent

def _invoke_orchestrator(self, messages, force_finish=False):
    if self._use_function_calling:
        try:
            return self._invoke_fc(messages, force_finish)
        except Exception as exc:
            self._use_function_calling = False  # permanent downgrade
    return self._invoke_text(messages, force_finish)
```

### 4. LLM-Driven Reflection and Self-Verification

A common failure mode of coding agents is producing answers that *look* complete but have not been validated — for example, writing a solution file but never executing it, or claiming a bug is fixed without re-running the failing test. The agent addresses this with a **reflection mechanism** that acts as an automated quality gate before any answer is returned to the user.

When the agent calls `finish()` with a proposed answer, a separate LLM invocation reviews the answer against three dimensions: **completeness** (did the agent actually do what was asked?), **correctness** (are there logical errors or missing edge cases?), and **verification** (was the answer empirically tested?). The reflection is powered by a `_UsageState` tracker that records which tools were invoked, which files were written, and — critically — whether code was written but never executed. If the agent wrote code without running it or running tests, the reflection prompt receives an explicit flag instructing it to reject the answer:

```python
# code/agents/function_call_agent.py — _reflect_on_answer()

def _reflect_on_answer(self, question, proposed_answer, usage) -> tuple[bool, str]:
    verification_note = ""
    if usage.wrote_code and not usage.ran_tests:
        verification_note = (
            "\n\n**IMPORTANT**: The agent wrote code but did NOT execute it "
            "or run any tests. You should NOT approve if the task required "
            "working code. Reject and ask the agent to test with code_exec."
        )

    prompt = self._reflection_prompt_template.format(
        verification_note=verification_note,
        question=question,
        proposed_answer=proposed_answer,
        tools_summary=", ".join(usage.tools_used) or "none",
        files_written=", ".join(usage.files_written) or "none",
        tests_executed="yes" if usage.ran_tests else "no",
    )
    response = self.llm.invoke([{"role": "user", "content": prompt}])
    text = self._strip_think_tags(response)
    approved = "APPROVED" in text.upper() and "NEEDS_REVISION" not in text.upper()

    if not approved:
        return False, "Your proposed answer was reviewed and found to have issues: ..."
    return True, proposed_answer
```

If the reflection verdict is `NEEDS_REVISION`, the feedback (including specific issues identified) is injected back into the conversation, and the agent re-enters its reasoning loop to address them. This process repeats for up to N configurable rounds (`--max-reflection-retries`), significantly improving first-pass output quality by catching the "wrote code but didn't test it" class of errors that LLMs frequently produce.

### 5. Automatic Error Detection and Debug Loop

Tool execution failures — syntax errors, import errors, runtime exceptions, test failures — are inevitable during iterative coding. Rather than relying on the LLM to notice and interpret raw tracebacks (which it often misreads or ignores), the agent implements a **structured debug loop** that automatically classifies errors and injects targeted recovery guidance.

The loop operates entirely within the existing reasoning cycle — it is not a separate execution path or class. After every tool execution, `_classify_observation()` inspects the output using tool-aware heuristics: `code_exec` results are checked for non-zero exit codes and traceback patterns, `test_runner` results for FAILED/ERROR lines, and all tools for generic traceback signatures. Detected errors are classified into categories (`syntax_error`, `import_error`, `runtime_error`, `timeout`, `test_failure`) and tracked by a `_DebugState` dataclass that persists across consecutive failed attempts:

```python
# code/agents/function_call_agent.py

@dataclass
class _DebugState:
    active: bool = False
    error_type: str = ""
    error_summary: str = ""
    failed_action: str = ""
    attempts: int = 0

@staticmethod
def _classify_observation(tool_name: str, observation: str) -> dict | None:
    low = observation.lower()
    has_nonzero_exit = bool(re.search(r"exit code:\s*[1-9]", low))
    has_traceback = "traceback (most recent call last)" in low

    is_code_error = tool_name == "code_exec" and (has_nonzero_exit or has_traceback)
    is_test_error = tool_name == "test_runner" and ("failed" in low or "error" in low)
    if not (is_code_error or is_test_error or ...):
        return None

    return {"error_type": error_type, "summary": _extract_error_summary(observation)}
```

When an error is detected, `_maybe_debug()` appends structured guidance to the observation — walking the LLM through a 5-step protocol (ANALYZE the error, READ the relevant code, DIAGNOSE the root cause, apply a minimal FIX, and VERIFY by re-running the exact command that failed). If the same error persists after `max_debug_attempts` consecutive retries (default: 3), the debug loop emits an exhaustion message and resets, preventing infinite loops on unfixable issues. On success, the debug state resets silently and normal execution resumes.

### 6. Multi-Paradigm Agent Architectures

The project supports **four distinct agent reasoning paradigms**, each suited to different LLM capabilities and task structures:

**ReActAgent** implements the classic Thought/Action/Observation text loop with regex-based parsing. It works with *any* LLM — including models that do not support function calling — making it the most portable option. The agent parses `Thought: ...` and `Action: tool_name[json_args]` from free-form text, with error recovery that injects structured feedback when the model produces malformed output.

**FunctionCallAgent** uses native OpenAI function calling for structured tool invocation, with parallel dispatch via `ThreadPoolExecutor` when multiple tools are called simultaneously. This is the default mode for models that support it, offering both reliability (no regex parsing needed) and efficiency (parallel execution).

**PlanAndSolveAgent** separates reasoning into an explicit planning phase and an execution phase. The `Planner` generates a step-by-step plan (supporting three extraction strategies: Python `ast.literal_eval`, JSON parsing, and numbered-list regex), and the `Executor` carries out each step sequentially with full history context.

**OrchestratorAgent** coordinates multiple expert agents in a multi-round collaboration loop, as described in the Blackboard architecture section above.

Critically, the Function Calling paradigm includes an **automatic degradation path**: when the LLM provider does not support function calling (or a function-calling request fails), the system transparently falls back to text-mode JSON parsing — `_invoke_fc` catches the exception and permanently switches to `_invoke_text` for the remainder of the session. This ensures the same agent configuration works across heterogeneous LLM backends without manual reconfiguration.

---

## Tool Details

The agent's capabilities are organized as modular `Tool` subclasses, each with its own JSON schema, actions, and built-in security mechanisms. See [`code/tools/README.md`](code/tools/README.md) for comprehensive documentation including parameters, usage examples, and safety mechanisms.

| Tool | Name | Key Actions | Description |
|---|---|---|---|
| **FileTool** | `file` | `read`, `write`, `edit`, `insert`, `undo`, `list_dir` | File operations with path sandboxing, syntax validation, and auto-revert |
| **CodeExecutionTool** | `code_exec` | `python`, `shell` | Sandboxed subprocess execution with timeout, blocklist, and stdin-hang detection |
| **CodeSearchTool** | `code_search` | `search`, `find_files`, `ast_search`, `find_references`, `repo_map` | Regex + AST structural search + PageRank-based repository map |
| **TestRunnerTool** | `test_runner` | `discover`, `run`, `coverage` | pytest/unittest auto-detection with structured result parsing |
| **GitTool** | `git` | `status`, `diff`, `log`, `commit`, `branch`, `stash`, `blame` | Git operations with destructive-command blocking |
| **LinterTool** | `linter` | `check`, `fix`, `format` | Static analysis (ruff > flake8 > py_compile) and formatting (ruff format > black) |
| **ProfilerTool** | `profiler` | `profile`, `timeit`, `memory` | CPU profiling (cProfile), benchmarking (timeit), memory analysis (tracemalloc) |
| **TerminalTool** | `terminal` | whitelisted commands | Safe filesystem and text-processing commands with command whitelist |
| **FinishTool** | `finish` | `result` | Signal task completion; breaks the FunctionCallAgent loop immediately |
| **MemoryTool** | `memory` | `add`, `search`, `summary`, `consolidate` | Store and retrieve conversation knowledge with importance scoring |
| **NoteTool** | `note` | `create`, `read`, `update`, `delete`, `search` | Structured notes with type classification, tags, and Markdown persistence |

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

### 4. TritonBench (TRIB)

Evaluates an agent's ability to generate **Triton GPU kernels** from natural language specifications. The agent must write compilable, functionally correct Triton code that passes provided test cases.

- **Source:** [TritonBench](https://github.com/thunlp/TritonBench)
- **Channels:** G (GitHub-sourced kernels, 462 tasks) and T (PyTorch-to-Triton translation, 161 tasks)
- **Difficulty levels:** 1-5 (G channel) — from simple element-wise ops to complex fused kernels
- **Instruction modes:** `simple` (brief description) or `complex` (detailed spec with API hints) — G channel only
- **Metrics:** Call accuracy (code compiles and runs), execution accuracy (output matches reference), per-difficulty breakdown

```bash
# G channel — complex instructions
python evaluation.py --benchmark trib --agent-type funca --instruction-mode complex --max-samples 5

# T channel (PyTorch-to-Triton translation)
python evaluation.py --benchmark trib --agent-type funca --channel T --max-samples 5

# Full run with custom settings
python evaluation.py --benchmark trib --agent-type funca --channel G \
    --instruction-mode complex --difficulty 3 \
    --max-iterations 32 --temperature 0.2 --max-samples 20

# Export results to custom path
python evaluation.py --benchmark trib --agent-type funca --output results/trib_custom.json
```

### 5. Data Generation Quality (AIME / LLM Judge)

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

## Experiment

We evaluate three agent configurations — **ReAct Agent**, **FunctionCall Agent**, and **Agent Swarm** (multi-agent) — on four benchmarks using the **Qwen3.5-30B-A3B** open-source model served locally via vLLM.

### Experimental Setup

| Item | Configuration |
|---|---|
| **Model** | Qwen/Qwen3.5-30B-A3B (MoE, 30B total / 3B active params) |
| **Serving** | vLLM with tensor-parallel-size 2, GPU memory utilization 0.90 |
| **Temperature** | 0.2 (low randomness for reproducibility) |
| **Max Iterations** | 32 (single-agent), 20 per worker / 8 rounds (swarm) |
| **Context Budget** | 32768 tokens |

### Results

#### BFCL — Function Calling Accuracy

BFCL uses direct LLM invocation (single-shot) rather than the full agent loop, so results reflect the underlying model's function-calling capability. The primary metric is AST-matching accuracy.

**Core Categories (Single-Turn)**

| Category | Qwen3.5-30B-A3B |
|---|---|
| `simple_python`     | 93.25% (373/400) |
| `simple_java`       | 79.00% (79/100) |
| `simple_javascript` | 70.00% (35/50) |
| `multiple`          | 93.50% (187/200) |
| `parallel`          | 91.50% (183/200) |
| `parallel_multiple` | 89.00% (178/200) |

**Live (Real-World) Categories**

| Category | Qwen3.5-30B-A3B |
|---|---|
| `live_simple`            | 81.78% (211/258) |
| `live_multiple`          | 79.30% (835/1053) |
| `live_parallel`          | 87.50% (14/16) |
| `live_parallel_multiple` | 83.33% (20/24) |

**Multi-Turn & Agentic Categories**

| Category | Qwen3.5-30B-A3B |
|---|---|
| `multi_turn_base`         |  |
| `multi_turn_miss_func`    |  |
| `multi_turn_miss_param`   |  |
| `multi_turn_long_context` |  |

**Overall**

| Metric | Qwen3.5-30B-A3B |
|---|---|
| **Avg. Accuracy (All Categories)** | |

#### SWE-bench Verified — Real-World Issue Resolution

| Metric | ReAct Agent | FunctionCall Agent | Agent Swarm |
|---|---|---|---|
| Resolved Rate   |  |  |  |
| Exact Match     |  |  |  |
| Avg. Steps Used |  |  |  |

#### TritonBench — GPU Kernel Generation (G Channel, Complex Instructions)

| Metric | ReAct Agent | FunctionCall Agent |
|---|---|---|
| Call Accuracy (code runs) |  |  |
| Execution Accuracy (output matches) |  |  |
| Avg. Score |  |  |

#### GAIA — General AI Assistant Capability

| Level | ReAct Agent | FunctionCall Agent |
|---|---|---|
| Level 1 | | |
| Level 2 | | |
| Level 3 | | |
| **Overall** | | |

### Reproducing the Experiments

All experiments can be reproduced with the provided `experiments.sh` script:

```bash
# Run all experiments (results saved to results/ and experiments_output.txt)
bash experiments.sh 2>&1 | tee experiments_output.txt

# Or run individual benchmarks:
# BFCL (16 categories: core + live + multi-turn)
python evaluation.py -b bfcl -c simple_python -n 50 --temperature 0.2
python evaluation.py -b bfcl -c multiple -n 50 --temperature 0.2
python evaluation.py -b bfcl -c parallel -n 50 --temperature 0.2
python evaluation.py -b bfcl -c multi_turn_base -n 50 --temperature 0.2
# ... see experiments.sh for the full list of 16 categories

# SWE-bench Verified
python evaluation.py -b swev --agent-type react -n 20 --max-iterations 32 --temperature 0.2
python evaluation.py -b swev --agent-type funca -n 20 --max-iterations 32 --temperature 0.2

# TritonBench (G channel, complex instructions)
python evaluation.py -b trib --agent-type react --channel G --instruction-mode complex -n 20 --temperature 0.2
python evaluation.py -b trib --agent-type funca --channel G --instruction-mode complex -n 20 --temperature 0.2

# GAIA
python evaluation.py -b gaia --level 1 -n 30 --temperature 0.2
python evaluation.py -b gaia --level 2 -n 30 --temperature 0.2
python evaluation.py -b gaia --level 3 -n 30 --temperature 0.2
```


---

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- [HelloAgents](https://github.com/datawhalechina/hello-agents) by [Datawhale](https://github.com/datawhalechina) — the underlying multi-agent framework
