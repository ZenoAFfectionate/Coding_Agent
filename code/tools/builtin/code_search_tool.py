"""CodeSearchTool - Code Search (grep / ripgrep) + AST-based structural queries

Provides fast code search for a Coding Agent:
- Regex and literal string search across files
- File pattern filtering (e.g., *.py, *.js)
- Context lines around matches
- File listing by pattern (glob)
- Supports ripgrep (rg) for speed, falls back to grep
- AST-based structural queries on Python files (functions, classes, imports, etc.)

Safety:
- Sandboxed to workspace directory
- Output size limits
- Timeout for search operations
"""

import ast
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ...utils.subprocess_utils import safe_run
from ..base import Tool, ToolParameter, tool_action


class CodeSearchTool(Tool):
    """Code search tool using grep/ripgrep.

    Provides fast text search across codebases with regex support,
    file filtering, and context lines.
    """

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 512 * 1024,  # 512 KB
        expandable: bool = False,
    ):
        super().__init__(
            name="code_search",
            description=(
                "Search code by pattern (regex/literal), filter by file type, find files by name, "
                "and perform AST-based structural queries on Python code (find functions, classes, "
                "imports, call sites, references, and file structure)"
            ),
            expandable=expandable,
        )
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self._rg_path = shutil.which("rg")  # ripgrep
        self._grep_path = shutil.which("grep") or "grep"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "search")
        dispatch = {
            "search": self._search,
            "find_files": self._find_files,
            "ast_search": self._ast_search,
            "find_references": self._find_references,
            "get_structure": self._get_structure,
            "repo_map": self._repo_map,
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="Action: 'search' (grep), 'find_files' (glob), 'ast_search' (structural query), 'find_references' (symbol usages), 'get_structure' (file outline), or 'repo_map' (ranked dependency map)",
                          required=True),
            ToolParameter(name="pattern", type="string",
                          description="Search pattern (regex for search, glob for find_files)",
                          required=False),
            ToolParameter(name="file_pattern", type="string",
                          description="File glob filter, e.g. '*.py', '*.ts' (search only)",
                          required=False),
            ToolParameter(name="path", type="string",
                          description="Subdirectory or file to search in (default: workspace root)",
                          required=False, default="."),
            ToolParameter(name="context_lines", type="integer",
                          description="Lines of context around each match (default 2)",
                          required=False, default=2),
            ToolParameter(name="case_sensitive", type="boolean",
                          description="Case-sensitive search (default true)",
                          required=False, default=True),
            ToolParameter(name="max_results", type="integer",
                          description="Maximum number of matches to return (default 50)",
                          required=False, default=50),
            ToolParameter(name="query_type", type="string",
                          description="For ast_search: 'functions', 'classes', 'imports', 'calls', or 'decorators'",
                          required=False),
            ToolParameter(name="symbol", type="string",
                          description="For find_references / ast_search: symbol name to search for (e.g. 'MyClass', 'my_func')",
                          required=False),
            ToolParameter(name="query", type="string",
                          description="For repo_map: optional keywords to bias ranking toward relevant files",
                          required=False),
            ToolParameter(name="max_tokens", type="integer",
                          description="For repo_map: token budget for output (default 2048)",
                          required=False, default=2048),
        ]

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + f"\n... truncated ({len(text)} bytes total)"
        return text

    def _safe_path(self, rel: str) -> Optional[Path]:
        resolved = (self.workspace / rel).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            return None
        return resolved

    @tool_action("search_code", "Search for a pattern across code files")
    def _search(self, parameters: Dict[str, Any]) -> str:
        """Grep / ripgrep search.

        Args:
            parameters: Dict with pattern, file_pattern, path, context_lines, etc.
        Returns:
            Search results.
        """
        pattern = parameters.get("pattern", "")
        if not pattern:
            return "Error: pattern is required."

        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."
        if not search_path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        context = parameters.get("context_lines", 2) or 2
        case_sensitive = parameters.get("case_sensitive", True)
        max_results = parameters.get("max_results", 50) or 50
        file_pattern = parameters.get("file_pattern")

        if self._rg_path:
            return self._search_rg(pattern, search_path, context, case_sensitive, max_results, file_pattern)
        else:
            return self._search_grep(pattern, search_path, context, case_sensitive, max_results, file_pattern)

    def _search_rg(self, pattern, path, context, case_sensitive, max_results, file_pattern):
        """Search using ripgrep."""
        args = [self._rg_path, "--line-number", f"--max-count={max_results}", f"-C{context}"]
        if not case_sensitive:
            args.append("-i")
        if file_pattern:
            args.extend(["-g", file_pattern])
        args.append(pattern)
        args.append(str(path))

        try:
            result = safe_run(args, capture_output=True, text=True, timeout=self.timeout,
                                    cwd=str(self.workspace))
            output = result.stdout
            if not output.strip():
                return f"No matches found for pattern: '{pattern}'"
            return self._truncate(f"=== Search Results (ripgrep) ===\nPattern: {pattern}\n\n{output}")
        except subprocess.TimeoutExpired:
            return f"Error: search timed out after {self.timeout}s"
        except Exception as e:
            return f"Error running ripgrep: {e}"

    def _search_grep(self, pattern, path, context, case_sensitive, max_results, file_pattern):
        """Search using grep (fallback)."""
        args = [self._grep_path, "-rn", f"-C{context}", f"-m{max_results}"]
        if not case_sensitive:
            args.append("-i")
        if file_pattern:
            args.extend(["--include", file_pattern])
        args.append(pattern)
        args.append(str(path))

        try:
            result = safe_run(args, capture_output=True, text=True, timeout=self.timeout,
                                    cwd=str(self.workspace))
            output = result.stdout
            if not output.strip():
                return f"No matches found for pattern: '{pattern}'"
            return self._truncate(f"=== Search Results (grep) ===\nPattern: {pattern}\n\n{output}")
        except subprocess.TimeoutExpired:
            return f"Error: search timed out after {self.timeout}s"
        except Exception as e:
            return f"Error running grep: {e}"

    @tool_action("find_files", "Find files matching a glob pattern")
    def _find_files(self, parameters: Dict[str, Any]) -> str:
        """Find files by glob pattern.

        Args:
            parameters: Dict with pattern (glob), path.
        Returns:
            List of matching file paths.
        """
        pattern = parameters.get("pattern", "")
        if not pattern:
            return "Error: pattern is required."

        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."

        max_results = parameters.get("max_results", 50) or 50

        try:
            matches = sorted(search_path.rglob(pattern))[:max_results]
        except Exception as e:
            return f"Error finding files: {e}"

        if not matches:
            return f"No files found matching: '{pattern}'"

        lines = [f"=== Files Matching '{pattern}' ({len(matches)} results) ===\n"]
        for m in matches:
            try:
                rel = m.relative_to(self.workspace)
            except ValueError:
                rel = m
            suffix = "/" if m.is_dir() else f"  ({m.stat().st_size} bytes)"
            lines.append(f"  {rel}{suffix}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse(filepath: Path) -> Optional[ast.Module]:
        """Parse a Python file into an AST, returning None on any error."""
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            return ast.parse(source, filename=str(filepath))
        except (SyntaxError, ValueError, UnicodeDecodeError, OSError):
            return None

    @staticmethod
    def _format_args(node: ast.FunctionDef) -> str:
        """Return a compact stringified argument list for a function node."""
        parts = []
        args = node.args
        # positional args (skip 'self'/'cls' for methods)
        positional = [a.arg for a in args.args]
        if positional and positional[0] in ("self", "cls"):
            positional = positional[1:]
        parts.extend(positional)
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        for a in args.kwonlyargs:
            parts.append(a.arg)
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
        return ", ".join(parts)

    def _collect_py_files(self, search_path: Path, file_pattern: str, limit: int = 200) -> List[Path]:
        """Collect Python files under search_path matching file_pattern."""
        try:
            return sorted(search_path.rglob(file_pattern))[:limit]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # ast_search
    # ------------------------------------------------------------------

    @tool_action("ast_search", "Search for structural elements in Python code using AST")
    def _ast_search(self, parameters: Dict[str, Any]) -> str:
        """AST-based structural search across Python files.

        Args:
            parameters: Dict with query_type, symbol, path, file_pattern, max_results.
        Returns:
            Formatted structural search results.
        """
        query_type = parameters.get("query_type", "functions")
        symbol = parameters.get("symbol")
        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."
        if not search_path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        file_pattern = parameters.get("file_pattern", "*.py")
        max_results = parameters.get("max_results", 50) or 50

        handlers = {
            "functions": self._ast_find_functions,
            "classes": self._ast_find_classes,
            "imports": self._ast_find_imports,
            "calls": self._ast_find_calls,
            "decorators": self._ast_find_decorators,
        }
        handler = handlers.get(query_type)
        if handler is None:
            return f"Unsupported query_type '{query_type}'. Supported: {', '.join(handlers)}"

        if query_type in ("calls", "decorators") and not symbol:
            return f"Error: 'symbol' is required for query_type='{query_type}'."

        py_files = self._collect_py_files(search_path, file_pattern)
        if not py_files:
            return f"No Python files found under '{parameters.get('path', '.')}' matching '{file_pattern}'."

        results = []
        for py_file in py_files:
            tree = self._safe_parse(py_file)
            if tree is None:
                continue
            try:
                rel = str(py_file.relative_to(self.workspace))
            except ValueError:
                rel = str(py_file)
            file_results = handler(tree, rel, symbol)
            results.extend(file_results)
            if len(results) >= max_results:
                results = results[:max_results]
                break

        if not results:
            extra = f" matching symbol '{symbol}'" if symbol else ""
            return f"No {query_type}{extra} found."

        header = f"=== AST Search: {query_type} ==="
        if symbol:
            header += f" (symbol: {symbol})"
        header += f"  [{len(results)} results]\n"
        return self._truncate(header + "\n".join(results))

    # --- ast_search sub-handlers ---

    def _ast_find_functions(self, tree: ast.Module, filepath: str, symbol: Optional[str]) -> List[str]:
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if symbol and node.name != symbol:
                    continue
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                args_str = self._format_args(node)
                decorators = ", ".join(self._decorator_name(d) for d in node.decorator_list)
                dec_str = f"  @{decorators}" if decorators else ""
                results.append(f"  {filepath}:{node.lineno}  {kind} {node.name}({args_str}){dec_str}")
        return results

    def _ast_find_classes(self, tree: ast.Module, filepath: str, symbol: Optional[str]) -> List[str]:
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if symbol and node.name != symbol:
                    continue
                bases = ", ".join(self._node_name(b) for b in node.bases)
                bases_str = f"({bases})" if bases else ""
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                methods_str = f"  methods: {', '.join(methods)}" if methods else ""
                results.append(f"  {filepath}:{node.lineno}  class {node.name}{bases_str}{methods_str}")
        return results

    def _ast_find_imports(self, tree: ast.Module, filepath: str, symbol: Optional[str]) -> List[str]:
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name_str = alias.name + (f" as {alias.asname}" if alias.asname else "")
                    if symbol and symbol not in alias.name:
                        continue
                    results.append(f"  {filepath}:{node.lineno}  import {name_str}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "") for a in node.names
                )
                if symbol and symbol not in module and not any(symbol == a.name for a in node.names):
                    continue
                results.append(f"  {filepath}:{node.lineno}  from {module} import {names}")
        return results

    def _ast_find_calls(self, tree: ast.Module, filepath: str, symbol: Optional[str]) -> List[str]:
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._node_name(node.func)
                if call_name and (call_name == symbol or call_name.endswith(f".{symbol}")):
                    results.append(f"  {filepath}:{node.lineno}  {call_name}(...)")
        return results

    def _ast_find_decorators(self, tree: ast.Module, filepath: str, symbol: Optional[str]) -> List[str]:
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    dec_name = self._decorator_name(dec)
                    if dec_name == symbol or dec_name.startswith(f"{symbol}("):
                        kind = "class" if isinstance(node, ast.ClassDef) else "def"
                        results.append(
                            f"  {filepath}:{node.lineno}  @{dec_name} -> {kind} {node.name}"
                        )
        return results

    @staticmethod
    def _node_name(node: ast.expr) -> str:
        """Extract a dotted name string from an AST expression node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = CodeSearchTool._node_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            return CodeSearchTool._node_name(node.func)
        return ""

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        """Extract decorator name, including arguments like @dec(arg)."""
        if isinstance(node, ast.Call):
            name = CodeSearchTool._node_name(node.func)
            return f"{name}(...)" if name else ""
        return CodeSearchTool._node_name(node)

    # ------------------------------------------------------------------
    # find_references
    # ------------------------------------------------------------------

    @tool_action("find_references", "Find all references to a symbol across Python files")
    def _find_references(self, parameters: Dict[str, Any]) -> str:
        """Find all usages of a symbol across Python files.

        Args:
            parameters: Dict with symbol, path, file_pattern, max_results.
        Returns:
            Grouped references (definitions vs usages) with file:line.
        """
        symbol = parameters.get("symbol")
        if not symbol:
            return "Error: 'symbol' is required for find_references."

        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."
        if not search_path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        file_pattern = parameters.get("file_pattern", "*.py")
        max_results = parameters.get("max_results", 50) or 50

        py_files = self._collect_py_files(search_path, file_pattern)
        if not py_files:
            return f"No Python files found under '{parameters.get('path', '.')}' matching '{file_pattern}'."

        definitions = []
        usages = []

        for py_file in py_files:
            tree = self._safe_parse(py_file)
            if tree is None:
                continue
            try:
                rel = str(py_file.relative_to(self.workspace))
            except ValueError:
                rel = str(py_file)

            for node in ast.walk(tree):
                # Definitions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    definitions.append(f"  {rel}:{node.lineno}  {kind} {node.name}")
                elif isinstance(node, ast.ClassDef) and node.name == symbol:
                    definitions.append(f"  {rel}:{node.lineno}  class {node.name}")
                # Usages: Name nodes
                elif isinstance(node, ast.Name) and node.id == symbol:
                    usages.append(f"  {rel}:{node.lineno}  {symbol}")
                # Usages: Attribute nodes
                elif isinstance(node, ast.Attribute) and node.attr == symbol:
                    prefix = self._node_name(node.value)
                    usages.append(f"  {rel}:{node.lineno}  {prefix}.{symbol}" if prefix else f"  {rel}:{node.lineno}  .{symbol}")

            if len(definitions) + len(usages) >= max_results:
                break

        if not definitions and not usages:
            return f"No references to '{symbol}' found."

        lines = [f"=== References to '{symbol}' ===\n"]
        if definitions:
            lines.append(f"Definitions ({len(definitions)}):")
            lines.extend(definitions[:max_results])
        if usages:
            lines.append(f"\nUsages ({len(usages)}):")
            remaining = max_results - len(definitions)
            lines.extend(usages[:max(remaining, 0)])

        return self._truncate("\n".join(lines))

    # ------------------------------------------------------------------
    # get_structure
    # ------------------------------------------------------------------

    @tool_action("get_structure", "Get structural outline of a Python file or directory")
    def _get_structure(self, parameters: Dict[str, Any]) -> str:
        """Return a structural outline of a Python file or directory.

        Args:
            parameters: Dict with path, file_pattern.
        Returns:
            Indented structural outline.
        """
        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."
        if not search_path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        if search_path.is_file():
            py_files = [search_path]
        else:
            file_pattern = parameters.get("file_pattern", "*.py")
            py_files = self._collect_py_files(search_path, file_pattern, limit=50)

        if not py_files:
            return f"No Python files found at '{parameters.get('path', '.')}'."

        sections = []
        for py_file in py_files:
            tree = self._safe_parse(py_file)
            if tree is None:
                continue
            try:
                rel = str(py_file.relative_to(self.workspace))
            except ValueError:
                rel = str(py_file)
            outline = self._outline_module(tree)
            if outline:
                sections.append(f"{rel}\n{outline}")

        if not sections:
            return "No parseable Python files found."

        return self._truncate("=== File Structure ===\n\n" + "\n\n".join(sections))

    def _outline_module(self, tree: ast.Module) -> str:
        """Build an indented outline of a module AST."""
        lines = []
        module_level_funcs = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(self._node_name(b) for b in node.bases)
                bases_str = f"({bases})" if bases else ""
                lines.append(f"  class {node.name}{bases_str}  :{node.lineno}")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        kind = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                        args_str = self._format_args(item)
                        extras = []
                        for dec in item.decorator_list:
                            dn = self._decorator_name(dec)
                            if dn in ("abstractmethod", "staticmethod", "classmethod",
                                      "property", "abstractproperty"):
                                extras.append(dn)
                        extra_str = f" [{', '.join(extras)}]" if extras else ""
                        lines.append(f"    {kind} {item.name}({args_str}){extra_str}  :{item.lineno}")
                lines.append("")  # blank line after class
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                args_str = self._format_args(node)
                module_level_funcs.append(f"    {kind} {node.name}({args_str})  :{node.lineno}")

        if module_level_funcs:
            lines.append("  (module-level)")
            lines.extend(module_level_funcs)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # repo_map
    # ------------------------------------------------------------------

    @tool_action("repo_map", "Build a ranked repository map of file definitions")
    def _repo_map(self, parameters: Dict[str, Any]) -> str:
        """Build a dependency-ranked map of the repository.

        Uses PageRank over an inter-file dependency graph to surface the most
        structurally important files, then emits their definitions within a
        token budget.

        Args:
            parameters: Dict with path, query (optional), max_tokens (optional).
        Returns:
            Compressed repository map string.
        """
        search_path = self._safe_path(parameters.get("path", "."))
        if search_path is None:
            return "Error: path escapes workspace."
        if not search_path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        query = parameters.get("query")
        max_tokens = parameters.get("max_tokens", 2048) or 2048

        graph = self._build_dep_graph(search_path)
        if graph.number_of_nodes() == 0:
            return "No Python files found to build a repository map."

        ranked = self._rank_graph(graph, query)
        return self._format_map(graph, ranked, max_tokens)

    def _build_dep_graph(self, search_path: Path) -> nx.DiGraph:
        """Build an inter-file dependency graph from Python files.

        Nodes are file paths (relative to workspace). Edges represent:
        - Import relationships (from X import Y)
        - Inheritance (class A(B) where B is defined elsewhere)
        - Call-site references (calls to functions defined in other files)

        Each node stores a list of definitions: (name, type, lineno, signature).
        """
        graph = nx.DiGraph()
        py_files = self._collect_py_files(search_path, "*.py", limit=500)

        # First pass: collect definitions per file
        file_defs: Dict[str, List[Tuple[str, str, int, str]]] = {}
        # Map definition name -> list of files defining it
        name_to_files: Dict[str, List[str]] = {}

        for py_file in py_files:
            tree = self._safe_parse(py_file)
            if tree is None:
                continue
            try:
                rel = str(py_file.relative_to(self.workspace))
            except ValueError:
                rel = str(py_file)

            defs: List[Tuple[str, str, int, str]] = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    sig = ", ".join(self._node_name(b) for b in node.bases)
                    defs.append((node.name, "class", node.lineno, f"({sig})" if sig else ""))
                    name_to_files.setdefault(node.name, []).append(rel)
                    # Also collect methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args_str = self._format_args(item)
                            kind = "async method" if isinstance(item, ast.AsyncFunctionDef) else "method"
                            defs.append((f"{node.name}.{item.name}", kind, item.lineno, f"({args_str})"))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args_str = self._format_args(node)
                    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    defs.append((node.name, kind, node.lineno, f"({args_str})"))
                    name_to_files.setdefault(node.name, []).append(rel)

            file_defs[rel] = defs
            graph.add_node(rel, defs=defs)

        # Second pass: build edges from imports, inheritance, and call-sites
        for py_file in py_files:
            tree = self._safe_parse(py_file)
            if tree is None:
                continue
            try:
                rel = str(py_file.relative_to(self.workspace))
            except ValueError:
                rel = str(py_file)

            if rel not in graph:
                continue

            for node in ast.walk(tree):
                # Import edges
                if isinstance(node, ast.ImportFrom) and node.module:
                    target = self._resolve_import(node.module, search_path)
                    if target and target in graph and target != rel:
                        graph.add_edge(rel, target)

                # Inheritance edges
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        base_name = self._node_name(base)
                        bare = base_name.rsplit(".", 1)[-1] if base_name else ""
                        if bare in name_to_files:
                            for target_file in name_to_files[bare]:
                                if target_file != rel:
                                    graph.add_edge(rel, target_file)

                # Call-site edges
                if isinstance(node, ast.Call):
                    call_name = self._node_name(node.func)
                    bare = call_name.rsplit(".", 1)[-1] if call_name else ""
                    if bare in name_to_files:
                        for target_file in name_to_files[bare]:
                            if target_file != rel:
                                graph.add_edge(rel, target_file)

        return graph

    def _resolve_import(self, module_name: str, search_path: Path) -> Optional[str]:
        """Try to resolve a dotted module name to a file path relative to workspace."""
        parts = module_name.split(".")
        candidate = search_path / Path(*parts)
        py_candidate = candidate.with_suffix(".py")
        init_candidate = candidate / "__init__.py"

        for c in (py_candidate, init_candidate):
            if c.exists():
                try:
                    return str(c.relative_to(self.workspace))
                except ValueError:
                    pass
        return None

    def _rank_graph(self, graph: nx.DiGraph, query: Optional[str] = None) -> List[Tuple[str, float]]:
        """Rank files using PageRank (optionally personalized by query keywords).

        Args:
            graph: The dependency graph.
            query: Optional keywords; if provided, files containing matching
                   definitions receive higher personalization weight.
        Returns:
            Sorted list of (file_path, score), descending by score.
        """
        if graph.number_of_nodes() == 0:
            return []

        personalization = None
        if query:
            keywords = query.lower().split()
            personalization = {}
            for node in graph.nodes:
                defs = graph.nodes[node].get("defs", [])
                score = 0.0
                for name, _, _, _ in defs:
                    name_lower = name.lower()
                    for kw in keywords:
                        if kw in name_lower:
                            score += 1.0
                personalization[node] = score + 0.1  # small base so all nodes participate

        try:
            scores = nx.pagerank(graph, personalization=personalization)
        except nx.NetworkXError:
            # Fallback: uniform scores
            n = graph.number_of_nodes()
            scores = {node: 1.0 / n for node in graph.nodes}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _format_map(self, graph: nx.DiGraph, ranked_files: List[Tuple[str, float]], max_tokens: int) -> str:
        """Format the repository map within a token budget.

        Iterates files by rank and emits their definitions (classes with methods,
        top-level functions) until the budget is exhausted.

        Args:
            graph: Dependency graph with node attributes.
            ranked_files: Sorted (file_path, score) pairs.
            max_tokens: Approximate token budget (~4 chars per token).
        Returns:
            Formatted repository map string.
        """
        max_chars = max_tokens * 4
        header = "=== Repository Map ===\n"
        parts = [header]
        used = len(header)
        files_with_defs = sum(1 for f, _ in ranked_files if graph.nodes[f].get("defs"))
        files_shown = 0

        for filepath, score in ranked_files:
            defs = graph.nodes[filepath].get("defs", [])
            if not defs:
                continue

            section_lines = [f"\n{filepath}"]
            for name, kind, lineno, sig in defs:
                if kind in ("method", "async method"):
                    section_lines.append(f"    {kind} {name}{sig}  :{lineno}")
                elif kind == "class":
                    section_lines.append(f"  class {name}{sig}  :{lineno}")
                else:
                    section_lines.append(f"  {kind} {name}{sig}  :{lineno}")

            section_text = "\n".join(section_lines)
            if used + len(section_text) > max_chars:
                parts.append(f"\n... ({files_with_defs} files total, output truncated to ~{max_tokens} tokens)")
                break
            parts.append(section_text)
            used += len(section_text)
            files_shown += 1

        return "\n".join(parts)
