# Tool System Documentation

This directory contains the tool infrastructure and all built-in tools for the Coding Agent. Each tool is a `Tool` subclass with a JSON schema, registered actions via the `@tool_action` decorator, and built-in safety mechanisms.

## Architecture

| File | Purpose |
|---|---|
| `base.py` | `Tool` ABC, `ToolParameter` (Pydantic model), `@tool_action` decorator |
| `registry.py` | `ToolRegistry` — registers and looks up tools by name |
| `async_executor.py` | Async execution support |
| `chain.py` | Tool chaining utilities |
| `builtin/` | All built-in tool implementations |

---

## Built-in Tools

### 1. FileTool

**File:** `builtin/file_tool.py` | **Name:** `file`

File operations with path sandboxing, syntax validation, and single-slot undo.

**Constructor:** `FileTool(workspace=".", max_read_size=1MB, max_write_size=5MB, lint_on_edit=True, view_window=200)`

| Action | Description |
|---|---|
| `read` | Read file content with optional line range (`start_line`, `end_line`); windowed at 200 lines |
| `write` | Create or overwrite a file with full content replacement |
| `edit` | Replace an exact string (`old_string` → `new_string`); includes fuzzy-match fallback |
| `insert` | Insert content before a specific line number (or append to end) |
| `replace_lines` | Replace or delete a line range; guards against accidental deletion of `return`/`yield`/`raise` |
| `undo` | Revert file to state before last modification (one slot per file) |
| `list_dir` | List directory contents (up to 200 entries) |
| `file_info` | Get file metadata (size, line count, modification time) |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | One of the actions above |
| `path` | string | Yes | Relative file/directory path (sandboxed to workspace) |
| `content` | string | No | File content for `write`/`insert`/`replace_lines` |
| `old_string` | string | No | String to find (`edit` action) |
| `new_string` | string | No | Replacement string (`edit` action) |
| `start_line` | integer | No | Start line, 1-based (default: 1) |
| `end_line` | integer | No | End line, inclusive |
| `show_line_numbers` | boolean | No | Show line numbers in `read` output (default: true) |

**Safety mechanisms:**
- All paths resolved against workspace root; `../` traversal and symlink escapes blocked
- Python files validated with `ast.parse()` after every edit; syntax errors trigger auto-revert
- Single-slot undo backup per file

**Example:**
```python
from code.tools.builtin.file_tool import FileTool

tool = FileTool(workspace="./my_project")
tool.run({"action": "read", "path": "src/main.py", "start_line": 1, "end_line": 50})
tool.run({"action": "edit", "path": "src/main.py", "old_string": "def foo():", "new_string": "def bar():"})
tool.run({"action": "undo", "path": "src/main.py"})
```

---

### 2. CodeExecutionTool

**File:** `builtin/code_execution_tool.py` | **Name:** `code_exec`

Execute Python or shell code in sandboxed subprocesses with timeout and safety checks.

**Constructor:** `CodeExecutionTool(workspace=".", timeout=30)`

| Action | Description |
|---|---|
| `python` | Execute Python code in an isolated subprocess |
| `shell` | Execute a shell command |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | `python` or `shell` |
| `code` | string | Yes | Code or command to execute |
| `timeout` | integer | No | Timeout in seconds (default: 30) |

**Safety mechanisms:**
- Subprocess isolation with process-group-level timeout (`safe_run` → `SIGKILL` to entire process tree)
- Destructive shell command blocklist (`rm -rf /`, fork bombs, `dd` to block devices, etc.)
- stdin-hang detection: code calling `input()` / `sys.stdin.read()` is rejected immediately
- Output truncation for very long results

**Example:**
```python
from code.tools.builtin.code_execution_tool import CodeExecutionTool

tool = CodeExecutionTool(workspace="./my_project", timeout=30)
tool.run({"action": "python", "code": "print('hello world')"})
tool.run({"action": "shell", "code": "ls -la src/"})
```

---

### 3. CodeSearchTool

**File:** `builtin/code_search_tool.py` | **Name:** `code_search`

Multi-mode code search: text-based regex/ripgrep, AST structural queries, and PageRank repository map.

**Constructor:** `CodeSearchTool(workspace=".")`

| Action | Description |
|---|---|
| `search` | Regex or literal pattern search with context lines; uses ripgrep if available |
| `find_files` | Find files by glob pattern (e.g., `**/*.py`) |
| `ast_search` | AST-based structural search: `functions`, `classes`, `imports`, `calls`, `decorators` |
| `find_references` | Find all definitions and usages of a symbol across Python files |
| `get_structure` | Indented structural outline of a file or directory |
| `repo_map` | PageRank-based dependency graph map within a token budget (requires `networkx`) |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | One of the 6 actions above |
| `pattern` | string | No | Regex (`search`) or glob (`find_files`) pattern |
| `file_pattern` | string | No | File glob filter, e.g., `*.py` |
| `path` | string | No | Subdirectory or file to search in |
| `context_lines` | integer | No | Context lines around matches (default: 2) |
| `case_sensitive` | boolean | No | Case-sensitive search (default: true) |
| `max_results` | integer | No | Maximum matches (default: 50) |
| `query_type` | string | No | For `ast_search`: `functions`, `classes`, `imports`, `calls`, or `decorators` |
| `symbol` | string | No | Symbol name for `find_references` / `ast_search` |
| `query` | string | No | Keywords for `repo_map` ranking |
| `max_tokens` | integer | No | Token budget for `repo_map` (default: 2048) |

**Example:**
```python
from code.tools.builtin.code_search_tool import CodeSearchTool

tool = CodeSearchTool(workspace="./my_project")
tool.run({"action": "search", "pattern": "def main", "file_pattern": "*.py"})
tool.run({"action": "ast_search", "query_type": "classes", "path": "src/"})
tool.run({"action": "find_references", "symbol": "MyClass"})
tool.run({"action": "get_structure", "path": "src/core/agent.py"})
tool.run({"action": "repo_map", "query": "agent tool", "max_tokens": 4096})
```

---

### 4. TestRunnerTool

**File:** `builtin/test_runner_tool.py` | **Name:** `test_runner`

Test discovery, execution, and coverage collection with pytest/unittest auto-detection.

**Constructor:** `TestRunnerTool(project_path=".", timeout=120)`

| Action | Description |
|---|---|
| `discover` | Discover and list available tests |
| `run` | Run tests with structured summary (pass/fail/error counts) |
| `coverage` | Run tests with coverage analysis (pytest-cov or coverage.py) |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | `discover`, `run`, or `coverage` |
| `target` | string | No | Test target: file, directory, or `path::test_func` |
| `verbose` | boolean | No | Verbose output (default: true) |
| `keyword` | string | No | Filter tests by keyword expression (pytest `-k`) |

**Example:**
```python
from code.tools.builtin.test_runner_tool import TestRunnerTool

tool = TestRunnerTool(project_path="./my_project", timeout=120)
tool.run({"action": "discover"})
tool.run({"action": "run", "target": "tests/test_main.py", "keyword": "test_add"})
tool.run({"action": "coverage", "target": "tests/"})
```

---

### 5. GitTool

**File:** `builtin/git_tool.py` | **Name:** `git`

Git operations with built-in protection against destructive commands.

**Constructor:** `GitTool(repo_path=".", allow_destructive=False, timeout=30)`

| Action | Description |
|---|---|
| `status` | Show working tree status and current branch |
| `diff` | Show staged and unstaged differences |
| `log` | Show commit history (graph, oneline format) |
| `show` | Show details of a specific commit |
| `branch` | List, create, switch, or delete branches (`sub_action`) |
| `add` | Stage files for commit |
| `commit` | Create a commit with staged changes |
| `stash` | Save, pop, list, or drop stash (`sub_action`) |
| `blame` | Show line-by-line authorship for a file |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | One of the 9 actions above |
| `files` | string | No | File path(s), space-separated (default: `.`) |
| `message` | string | No | Commit message (required for `commit`) |
| `limit` | integer | No | Log entry count (default: 10) |
| `ref` | string | No | Git ref (commit hash, branch, tag) |
| `branch_name` | string | No | Branch name for branch operations |
| `sub_action` | string | No | `list`/`create`/`switch`/`delete` (branch) or `save`/`pop`/`list`/`drop` (stash) |

**Safety:** Destructive commands blocked by default: `push --force`, `reset --hard`, `rebase`, `clean -f`, `branch -D`.

**Example:**
```python
from code.tools.builtin.git_tool import GitTool

tool = GitTool(repo_path="./my_project")
tool.run({"action": "status"})
tool.run({"action": "diff", "files": "src/main.py"})
tool.run({"action": "log", "limit": 5})
tool.run({"action": "add", "files": "src/main.py"})
tool.run({"action": "commit", "message": "Fix bug in main"})
```

---

### 6. LinterTool

**File:** `builtin/linter_tool.py` | **Name:** `linter`

Static analysis, auto-fix, and code formatting with automatic backend detection.

**Constructor:** `LinterTool(workspace=".", timeout=30)`

| Action | Description | Backend Priority |
|---|---|---|
| `check` | Run linter and report issues | ruff > flake8 > py_compile (stdlib) |
| `fix` | Auto-fix lint issues in-place | ruff `--fix` |
| `format` | Format code to style guidelines | ruff format > black |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | `check`, `fix`, or `format` |
| `path` | string | Yes | Relative file or directory path |
| `select` | string | No | Rule selection (e.g., `E,W,F` or `E501,F401`) |

The `py_compile` fallback is always available (stdlib) and provides syntax checking even when no external linter is installed.

**Example:**
```python
from code.tools.builtin.linter_tool import LinterTool

tool = LinterTool(workspace="./my_project")
tool.run({"action": "check", "path": "src/main.py"})
tool.run({"action": "fix", "path": "src/"})
tool.run({"action": "format", "path": "src/main.py"})
```

---

### 7. ProfilerTool

**File:** `builtin/profiler_tool.py` | **Name:** `profiler`

Performance profiling using Python stdlib — no external dependencies required.

**Constructor:** `ProfilerTool(workspace=".", timeout=60)`

| Action | Description | Backend |
|---|---|---|
| `profile` | CPU profile a Python file, show top-N hotspots | `cProfile` + `pstats` |
| `timeit` | Benchmark a code snippet | `timeit` |
| `memory` | Memory allocation snapshot | `tracemalloc` |

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | `profile`, `timeit`, or `memory` |
| `path` | string | No | File path for `profile` action |
| `code` | string | No | Code snippet for `timeit` and `memory` |
| `top_n` | integer | No | Number of top results (default: 15) |
| `number` | integer | No | Iterations for timeit (default: 1000) |
| `repeat` | integer | No | Repetitions for timeit (default: 5) |
| `setup` | string | No | Setup code for timeit |

**Example:**
```python
from code.tools.builtin.profiler_tool import ProfilerTool

tool = ProfilerTool(workspace="./my_project")
tool.run({"action": "profile", "path": "src/main.py", "top_n": 10})
tool.run({"action": "timeit", "code": "[i**2 for i in range(1000)]", "number": 10000})
tool.run({"action": "memory", "code": "x = [i for i in range(100000)]"})
```

---

### 8. TerminalTool

**File:** `builtin/terminal_tool.py` | **Name:** `terminal`

Cross-platform terminal tool with command whitelist for safe filesystem and text-processing operations.

**Constructor:** `TerminalTool(workspace=".", timeout=30, allow_cd=True)`

**Allowed commands:** `ls`, `dir`, `tree`, `cat`, `head`, `tail`, `find`, `grep`, `wc`, `sort`, `uniq`, `cut`, `awk`, `sed`, `pwd`, `cd`, `file`, `stat`, `du`, `df`, `echo`, `which`

| Name | Type | Required | Description |
|---|---|---|---|
| `command` | string | Yes | Command to execute (whitelist enforced) |

---

### 9. FinishTool

**File:** `builtin/finish_tool.py` | **Name:** `finish`

Signal task completion and break the agent loop immediately.

| Name | Type | Required | Description |
|---|---|---|---|
| `result` | string | Yes | Final answer or description of the solution |

Returns a `__FINISH__`-prefixed string that the `FunctionCallAgent` loop detects to exit immediately.

---

### 10. MemoryTool

**File:** `builtin/memory_tool.py` | **Name:** `memory`

Store and retrieve conversation knowledge with importance scoring and type classification.

| Action | Description |
|---|---|
| `add` | Add new memory (working/episodic/semantic/perceptual types) |
| `search` | Search relevant memories by query |
| `summary` | Get memory system summary |
| `update` | Update an existing memory by ID |
| `remove` | Delete a specific memory by ID |
| `forget` | Batch forget by strategy (importance/time/capacity-based) |
| `consolidate` | Promote important short-term memories to long-term |

---

### 11. NoteTool

**File:** `builtin/note_tool.py` | **Name:** `note`

Structured notes with type classification, tags, and Markdown + YAML frontmatter persistence.

| Action | Description |
|---|---|
| `create` | Create a new structured note |
| `read` | Read a note by ID |
| `update` | Update an existing note |
| `delete` | Delete a note by ID |
| `list` | List notes (optionally filtered by type) |
| `search` | Full-text search across title/content/tags |

**Note types:** `task_state`, `conclusion`, `blocker`, `action`, `reference`, `general`

---

### 12. SearchTool

**File:** `builtin/search_tool.py` | **Name:** `search`

Web search engine supporting multiple backends.

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | string | Yes | Search query |
| `backend` | string | No | `hybrid` (default), `tavily`, `serpapi`, `duckduckgo`, `searxng`, `perplexity` |
| `max_results` | integer | No | Maximum results (default: 5) |

**Backend priority (hybrid mode):** Tavily → SerpApi → DuckDuckGo

---

### 13. RAGTool

**File:** `builtin/rag_tool.py` | **Name:** `rag`

Document retrieval-augmented generation with multi-format support.

| Action | Description |
|---|---|
| `add_document` | Add a file (PDF, Word, Excel, PPT, images, audio) to the knowledge base |
| `add_text` | Add raw text to the knowledge base |
| `ask` | Full RAG pipeline: retrieve → inject context → LLM generates answer with citations |
| `search` | Vector search only (no LLM generation) |
| `stats` | Knowledge base statistics |
| `clear` | Clear the knowledge base (requires `confirm=true`) |

**Infrastructure:** Qdrant vector store (URL from `QDRANT_URL` env).

---

### 14. MCPTool (Model Context Protocol)

**File:** `builtin/protocol_tools.py` | **Name:** `mcp`

Connect to MCP servers and auto-expand each server tool into an independent `MCPWrappedTool`.

| Action | Description |
|---|---|
| `list_tools` | List all tools provided by the MCP server |
| `call_tool` | Call a specific tool by name with arguments |
| `list_resources` | List available resources |
| `read_resource` | Read a resource by URI |
| `list_prompts` | List available prompt templates |
| `get_prompt` | Retrieve a prompt template |

**Constructor:** `MCPTool(server_command=..., auto_expand=True)`

When `auto_expand=True`, each MCP server tool is registered as a standalone `MCPWrappedTool` in the `ToolRegistry`, making it directly callable by the agent as if it were a native tool.

---

## Default Tool Sets by Entry Point

| Entry Point | Tools |
|---|---|
| `react_agent.py` | FileTool, CodeExecutionTool, CodeSearchTool, TestRunnerTool, GitTool, LinterTool, ProfilerTool |
| `funca_agent.py` | FileTool, CodeExecutionTool, CodeSearchTool, TestRunnerTool, GitTool, LinterTool, ProfilerTool, FinishTool |
| `run_swarm.py` (review worker) | FileTool, CodeSearchTool, LinterTool, GitTool |
| `run_swarm.py` (test worker) | FileTool, CodeExecutionTool, CodeSearchTool, TestRunnerTool |
| `run_swarm.py` (optimize worker) | FileTool, CodeExecutionTool, CodeSearchTool, ProfilerTool, LinterTool |
| `run_swarm.py` (debug worker) | FileTool, CodeExecutionTool, CodeSearchTool, TestRunnerTool, GitTool, LinterTool |
