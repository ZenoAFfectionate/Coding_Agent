"""CodeSearchTool - Code Search (grep / ripgrep)

Provides fast code search for a Coding Agent:
- Regex and literal string search across files
- File pattern filtering (e.g., *.py, *.js)
- Context lines around matches
- File listing by pattern (glob)
- Supports ripgrep (rg) for speed, falls back to grep

Safety:
- Sandboxed to workspace directory
- Output size limits
- Timeout for search operations
"""

import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                "Search code by pattern (regex or literal), filter by file type, "
                "find files by name pattern, and show context around matches"
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
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="Action: 'search' (grep) or 'find_files' (glob)", required=True),
            ToolParameter(name="pattern", type="string",
                          description="Search pattern (regex for search, glob for find_files)", required=True),
            ToolParameter(name="file_pattern", type="string",
                          description="File glob filter, e.g. '*.py', '*.ts' (search only)",
                          required=False),
            ToolParameter(name="path", type="string",
                          description="Subdirectory to search in (default: workspace root)",
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
            result = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout,
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
            result = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout,
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
