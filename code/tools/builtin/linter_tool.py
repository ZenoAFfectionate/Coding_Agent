"""LinterTool - Static Analysis & Code Formatting

Provides linting and formatting capabilities for a Coding Agent:
- Check files for lint errors and syntax issues
- Auto-fix lint issues in-place (ruff only)
- Format code to conform to style guidelines

Backend resolution order:
- check/fix: ruff > flake8 > py_compile (stdlib fallback)
- format: ruff format > black

Safety:
- All paths are sandboxed to a configurable workspace root
- Subprocess execution with timeout and output-size limits
"""

import os
import shutil
import subprocess
import py_compile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.subprocess_utils import safe_run
from ..base import Tool, ToolParameter, tool_action


class LinterTool(Tool):
    """Linter and formatter tool for Coding Agents.

    Supports check, fix, and format actions with automatic backend
    detection (ruff > flake8 > py_compile for checking,
    ruff format > black for formatting).
    """

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 512 * 1024,  # 512 KB
        expandable: bool = False,
    ):
        super().__init__(
            name="linter",
            description=(
                "Lint and format Python code - check for errors, auto-fix issues, "
                "and format code (supports ruff, flake8, black with py_compile fallback)"
            ),
            expandable=expandable,
        )
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Auto-detect available backends
        self._ruff = shutil.which("ruff")
        self._flake8 = shutil.which("flake8")
        self._black = shutil.which("black")

    # ------------------------------------------------------------------ #
    #  Path safety
    # ------------------------------------------------------------------ #

    def _safe_path(self, rel_path: str) -> Optional[Path]:
        """Resolve *rel_path* inside the workspace, blocking escapes."""
        resolved = (self.workspace / rel_path).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            return None
        return resolved

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _truncate(self, text: str) -> str:
        """Truncate output if it exceeds the limit."""
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + f"\n... truncated ({len(text)} bytes total)"
        return text

    def _run_cmd(self, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Execute a command and return the CompletedProcess result."""
        effective_timeout = timeout or self.timeout
        return safe_run(
            args,
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )

    def _detect_check_backend(self) -> str:
        """Return the best available checking backend name."""
        if self._ruff:
            return "ruff"
        if self._flake8:
            return "flake8"
        return "py_compile"

    def _detect_format_backend(self) -> str:
        """Return the best available formatting backend name."""
        if self._ruff:
            return "ruff"
        if self._black:
            return "black"
        return "none"

    # ------------------------------------------------------------------ #
    #  py_compile fallback
    # ------------------------------------------------------------------ #

    def _check_py_compile(self, path: Path) -> str:
        """Syntax-check a single Python file using py_compile (stdlib).

        Args:
            path: Absolute path to the file.

        Returns:
            Result string.
        """
        if path.is_dir():
            # Recursively check all .py files
            issues = []
            py_files = sorted(path.rglob("*.py"))
            for f in py_files:
                try:
                    py_compile.compile(str(f), doraise=True)
                except py_compile.PyCompileError as e:
                    issues.append(str(e))
            if issues:
                return f"Found {len(issues)} syntax error(s):\n" + "\n".join(issues)
            return f"All {len(py_files)} file(s) passed syntax check (py_compile)."
        else:
            try:
                py_compile.compile(str(path), doraise=True)
                return f"No syntax errors found in {path.name} (py_compile)."
            except py_compile.PyCompileError as e:
                return f"Syntax error: {e}"

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "check")
        dispatch = {
            "check": self._check,
            "fix": self._fix,
            "format": self._format,
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action", type="string",
                description="Action: check (lint), fix (auto-fix), format (code style)",
                required=True,
            ),
            ToolParameter(
                name="path", type="string",
                description="Relative file or directory path to lint/format",
                required=True,
            ),
            ToolParameter(
                name="fix", type="boolean",
                description="For 'check' action: also apply auto-fixes (ruff only, default false)",
                required=False, default=False,
            ),
            ToolParameter(
                name="select", type="string",
                description="Rule selection (e.g. 'E,W,F' for flake8 or 'E501,F401' for ruff)",
                required=False,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    @tool_action("linter_check", "Run linter on a file or directory and report issues")
    def _check(self, parameters: Dict[str, Any]) -> str:
        """Check a file or directory for lint errors.

        Args:
            parameters: Dict with path, fix (optional), select (optional).
        Returns:
            Lint report.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        backend = self._detect_check_backend()
        also_fix = parameters.get("fix", False)
        select = parameters.get("select")

        try:
            if backend == "ruff":
                args = [self._ruff, "check"]
                if also_fix:
                    args.append("--fix")
                if select:
                    args.extend(["--select", select])
                args.append(str(path))
                result = self._run_cmd(args)
                output = result.stdout + result.stderr
                output = self._truncate(output.strip())
                header = f"=== Lint Check (ruff{' --fix' if also_fix else ''}) ==="
                if not output:
                    return f"{header}\nAll checks passed - no issues found."
                return f"{header}\n{output}"

            elif backend == "flake8":
                args = [self._flake8]
                if select:
                    args.extend(["--select", select])
                args.append(str(path))
                result = self._run_cmd(args)
                output = result.stdout + result.stderr
                output = self._truncate(output.strip())
                header = "=== Lint Check (flake8) ==="
                if not output:
                    return f"{header}\nAll checks passed - no issues found."
                return f"{header}\n{output}"

            else:
                # py_compile fallback
                return f"=== Syntax Check (py_compile) ===\n{self._check_py_compile(path)}"

        except subprocess.TimeoutExpired:
            return f"Error: lint check timed out after {self.timeout}s"
        except Exception as e:
            return f"Error running lint check: {e}"

    @tool_action("linter_fix", "Auto-fix lint issues in-place (ruff only)")
    def _fix(self, parameters: Dict[str, Any]) -> str:
        """Auto-fix lint issues in-place.

        Args:
            parameters: Dict with path, select (optional).
        Returns:
            Fix results with diff of changes.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        if not self._ruff:
            return (
                "Error: auto-fix requires ruff. Install with: pip install ruff\n"
                "Hint: use 'check' action for read-only lint (supports flake8 and py_compile fallback)."
            )

        select = parameters.get("select")

        try:
            # Capture original content for diff (file only)
            originals = {}
            if path.is_file():
                originals[path] = path.read_text(encoding="utf-8", errors="replace")
            else:
                for f in sorted(path.rglob("*.py")):
                    originals[f] = f.read_text(encoding="utf-8", errors="replace")

            args = [self._ruff, "check", "--fix"]
            if select:
                args.extend(["--select", select])
            args.append(str(path))
            result = self._run_cmd(args)

            # Build diff
            import difflib
            diff_parts = []
            for fpath, original in originals.items():
                try:
                    modified = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if modified != original:
                    rel = str(fpath.relative_to(self.workspace))
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        modified.splitlines(keepends=True),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    )
                    diff_parts.append("".join(diff))

            output = result.stdout + result.stderr
            output = self._truncate(output.strip())

            parts = ["=== Auto-Fix (ruff --fix) ==="]
            if output:
                parts.append(output)
            if diff_parts:
                parts.append("\n--- Changes Applied ---")
                parts.append("\n".join(diff_parts))
            else:
                parts.append("No changes were needed.")

            return "\n".join(parts)

        except subprocess.TimeoutExpired:
            return f"Error: auto-fix timed out after {self.timeout}s"
        except Exception as e:
            return f"Error running auto-fix: {e}"

    @tool_action("linter_format", "Format code according to style guidelines")
    def _format(self, parameters: Dict[str, Any]) -> str:
        """Format code using ruff format or black.

        Args:
            parameters: Dict with path.
        Returns:
            Format results with diff of changes.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: path not found: {parameters.get('path')}"

        backend = self._detect_format_backend()

        if backend == "none":
            return (
                "Error: no formatter available. Install one of:\n"
                "  pip install ruff    (recommended)\n"
                "  pip install black"
            )

        try:
            # Capture originals for diff
            originals = {}
            if path.is_file():
                originals[path] = path.read_text(encoding="utf-8", errors="replace")
            else:
                for f in sorted(path.rglob("*.py")):
                    originals[f] = f.read_text(encoding="utf-8", errors="replace")

            if backend == "ruff":
                args = [self._ruff, "format", str(path)]
            else:
                args = [self._black, str(path)]

            result = self._run_cmd(args)

            # Build diff
            import difflib
            diff_parts = []
            for fpath, original in originals.items():
                try:
                    modified = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if modified != original:
                    rel = str(fpath.relative_to(self.workspace))
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        modified.splitlines(keepends=True),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    )
                    diff_parts.append("".join(diff))

            output = result.stdout + result.stderr
            output = self._truncate(output.strip())

            backend_label = "ruff format" if backend == "ruff" else "black"
            parts = [f"=== Format ({backend_label}) ==="]
            if output:
                parts.append(output)
            if diff_parts:
                parts.append("\n--- Changes Applied ---")
                parts.append("\n".join(diff_parts))
            else:
                parts.append("No formatting changes needed.")

            return "\n".join(parts)

        except subprocess.TimeoutExpired:
            return f"Error: formatting timed out after {self.timeout}s"
        except Exception as e:
            return f"Error running formatter: {e}"
