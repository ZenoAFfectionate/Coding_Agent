"""FileTool - File Read / Write / Edit Tool

Provides safe file-system operations for a Coding Agent:
- Read files with optional line-number display and range selection
- Write / create files with content
- Edit files using exact string replacement (like sed but safer)
- Insert content at a specific line
- Replace or delete a range of lines
- Undo the last modification to a file
- List directory contents (tree view)
- Get file metadata (size, modification time, etc.)

Safety:
- All paths are sandboxed to a configurable workspace root
- Symlink traversal outside workspace is blocked
- Maximum file size limits for reads and writes
- In-memory backup before every destructive operation (undo support)
- Optional post-edit syntax validation for Python files
"""

import ast
import os
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter, tool_action


class FileTool(Tool):
    """File operations tool for Coding Agents.

    Supports read, write, edit, insert, replace_lines, undo, list_dir,
    and file_info actions.  All paths are resolved relative to a sandboxed
    workspace root.
    """

    def __init__(
        self,
        workspace: str = ".",
        max_read_size: int = 1024 * 1024,      # 1 MB
        max_write_size: int = 5 * 1024 * 1024,  # 5 MB
        expandable: bool = False,
        lint_on_edit: bool = True,
        view_window: int = 200,
    ):
        super().__init__(
            name="file",
            description=(
                "File operations tool - read, write, edit, insert, replace_lines, "
                "undo, list directories, and get file info (sandboxed to workspace)"
            ),
            expandable=expandable,
        )
        self.workspace = Path(workspace).resolve()
        self.max_read_size = max_read_size
        self.max_write_size = max_write_size
        self.lint_on_edit = lint_on_edit
        self.view_window = view_window
        self.workspace.mkdir(parents=True, exist_ok=True)
        # In-memory backup: maps resolved path → content before last edit
        self._backups: Dict[str, str] = {}

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
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _backup(self, path: Path) -> None:
        """Save current file content for undo (one slot per file)."""
        if path.exists() and path.is_file():
            try:
                self._backups[str(path)] = path.read_text(encoding="utf-8")
            except Exception:
                pass

    @staticmethod
    def _check_syntax(path: Path, content: str) -> str:
        """If *path* is a Python file, check syntax and return a warning or empty string."""
        if path.suffix != ".py":
            return ""
        try:
            ast.parse(content, filename=str(path))
            return ""
        except SyntaxError as e:
            return f"\n[Syntax warning] {e.msg} (line {e.lineno})"

    def _validate_and_maybe_revert(self, path: Path, modified: str, original: str, rel: str) -> Optional[str]:
        """For .py files, reject edits that introduce syntax errors by restoring the original.

        Args:
            path: Resolved file path.
            modified: The new file content (already written to disk).
            original: The previous file content (from backup).
            rel: Relative path for error messages.
        Returns:
            An error string if the edit was reverted, or None if the edit is OK.
        """
        if not self.lint_on_edit or path.suffix != ".py":
            return None
        try:
            ast.parse(modified, filename=str(path))
            return None
        except SyntaxError as e:
            # Restore original content
            try:
                path.write_text(original, encoding="utf-8")
            except Exception:
                pass  # Best-effort restore; backup remains available for undo
            return (
                f"Edit rejected: syntax error in {rel} — {e.msg} (line {e.lineno}).\n"
                f"File has been reverted to its previous state."
            )

    def _make_diff(self, original: str, modified: str, rel_path: str) -> str:
        """Generate a unified diff string between *original* and *modified*."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return "\n".join(diff)

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "read")
        dispatch = {
            "read": self._read,
            "write": self._write,
            "edit": self._edit,
            "insert": self._insert,
            "replace_lines": self._replace_lines,
            "undo": self._undo,
            "list_dir": self._list_dir,
            "file_info": self._file_info,
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action", type="string",
                description=(
                    "Action: read, write, edit, insert, replace_lines, undo, "
                    "list_dir, file_info"
                ),
                required=True,
            ),
            ToolParameter(name="path", type="string",
                          description="Relative file/directory path", required=True),
            ToolParameter(name="content", type="string",
                          description="File content (for write, insert, replace_lines)", required=False),
            ToolParameter(name="old_string", type="string",
                          description="String to find (edit action)", required=False),
            ToolParameter(name="new_string", type="string",
                          description="Replacement string (edit action)", required=False),
            ToolParameter(name="start_line", type="integer",
                          description="Start line (1-based). Used by read, insert, replace_lines",
                          required=False, default=1),
            ToolParameter(name="end_line", type="integer",
                          description="End line (inclusive). Used by read, replace_lines",
                          required=False),
            ToolParameter(name="show_line_numbers", type="boolean",
                          description="Show line numbers in read output (default true)",
                          required=False, default=True),
        ]

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    @tool_action("file_read", "Read file contents with optional line range")
    def _read(self, parameters: Dict[str, Any]) -> str:
        """Read a file.

        Args:
            parameters: Dict with path, start_line, end_line, show_line_numbers.
        Returns:
            File contents with line numbers.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: file not found: {parameters.get('path')}"
        if not path.is_file():
            return f"Error: not a file: {parameters.get('path')}"
        if path.stat().st_size > self.max_read_size:
            return f"Error: file too large ({path.stat().st_size} bytes > {self.max_read_size} limit)."

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = text.splitlines(keepends=True)
        total = len(lines)
        start = max(1, parameters.get("start_line", 1) or 1)
        end_line_explicit = parameters.get("end_line")
        end = end_line_explicit or total
        end = min(end, total)

        # Windowed viewing: cap output when end_line was not explicitly provided
        if end_line_explicit is None and total > self.view_window:
            end = min(start + self.view_window - 1, total)

        selected = lines[start - 1: end]
        show_nums = parameters.get("show_line_numbers", True)

        if show_nums:
            width = len(str(end))
            out_lines = [f"{str(i).rjust(width)}| {line.rstrip()}" for i, line in enumerate(selected, start=start)]
        else:
            out_lines = [line.rstrip() for line in selected]

        header = f"File: {parameters.get('path')}  (lines {start}-{end} of {total})\n"
        result = header + "\n".join(out_lines)

        if end_line_explicit is None and total > self.view_window and end < total:
            result += f"\n[Showing lines {start}-{end} of {total}. Use start_line/end_line to view other sections.]"

        return result

    @tool_action("file_write", "Create or overwrite a file")
    def _write(self, parameters: Dict[str, Any]) -> str:
        """Write content to a file (full replacement).

        Args:
            parameters: Dict with path, content.
        Returns:
            Success message.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        content = parameters.get("content", "")
        if len(content) > self.max_write_size:
            return f"Error: content too large ({len(content)} bytes > {self.max_write_size} limit)."

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._backup(path)
            original = self._backups.get(str(path), "")
            path.write_text(content, encoding="utf-8")

            # Lint-gated: revert if syntax is broken
            error = self._validate_and_maybe_revert(path, content, original, parameters.get("path", ""))
            if error:
                return error

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            result = f"Successfully wrote {len(content)} bytes ({line_count} lines) to {parameters.get('path')}"
            return result
        except Exception as e:
            return f"Error writing file: {e}"

    @tool_action("file_edit", "Edit a file by replacing an exact string")
    def _edit(self, parameters: Dict[str, Any]) -> str:
        """Edit a file by replacing old_string with new_string.

        Args:
            parameters: Dict with path, old_string, new_string.
        Returns:
            Diff of the change.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: file not found: {parameters.get('path')}"

        old_string = parameters.get("old_string", "")
        new_string = parameters.get("new_string", "")
        if not old_string:
            return "Error: old_string is required for edit action."

        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

        count = original.count(old_string)
        if count == 0:
            return "Error: old_string not found in file."
        if count > 1:
            return f"Error: old_string found {count} times. Provide more context to make it unique."

        modified = original.replace(old_string, new_string, 1)

        try:
            self._backup(path)
            path.write_text(modified, encoding="utf-8")
        except Exception as e:
            return f"Error writing file: {e}"

        rel = parameters.get("path", "")

        # Lint-gated: revert if syntax is broken
        error = self._validate_and_maybe_revert(path, modified, original, rel)
        if error:
            return error

        diff_text = self._make_diff(original, modified, rel)
        result = f"Successfully edited {rel}\n\n{diff_text}"
        return result

    @tool_action("file_insert", "Insert content at a specific line")
    def _insert(self, parameters: Dict[str, Any]) -> str:
        """Insert content before a specific line.

        ``start_line=1`` inserts at the top of the file.
        Omitting ``start_line`` (or 0) appends to the end.

        Args:
            parameters: Dict with path, content, start_line.
        Returns:
            Diff of the insertion.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: file not found: {parameters.get('path')}"

        content = parameters.get("content", "")
        if not content:
            return "Error: content is required for insert action."

        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = original.splitlines(keepends=True)
        insert_line = parameters.get("start_line", 0) or 0

        # Ensure inserted text ends with a newline so it doesn't merge with the next line
        if not content.endswith("\n"):
            content += "\n"

        if insert_line <= 0 or insert_line > len(lines):
            # Append to end
            if original and not original.endswith("\n"):
                lines.append("\n")
            lines.append(content)
            where = "end of file"
        else:
            # Insert before start_line (1-based → 0-based index)
            lines.insert(insert_line - 1, content)
            where = f"line {insert_line}"

        modified = "".join(lines)

        try:
            self._backup(path)
            path.write_text(modified, encoding="utf-8")
        except Exception as e:
            return f"Error writing file: {e}"

        inserted_count = content.count("\n") + (0 if content.endswith("\n") else 1)
        rel = parameters.get("path", "")

        # Lint-gated: revert if syntax is broken
        error = self._validate_and_maybe_revert(path, modified, original, rel)
        if error:
            return error

        diff_text = self._make_diff(original, modified, rel)
        result = f"Inserted {inserted_count} line(s) at {where} in {rel}\n\n{diff_text}"
        return result

    @tool_action("file_replace_lines", "Replace or delete a range of lines")
    def _replace_lines(self, parameters: Dict[str, Any]) -> str:
        """Replace lines start_line..end_line with new content.

        If ``content`` is empty or omitted, the lines are deleted.

        Args:
            parameters: Dict with path, start_line, end_line, content.
        Returns:
            Diff of the change.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: file not found: {parameters.get('path')}"

        start_line = parameters.get("start_line")
        end_line = parameters.get("end_line")
        if not start_line or not end_line:
            return "Error: start_line and end_line are required for replace_lines action."
        if start_line < 1:
            return "Error: start_line must be >= 1."
        if end_line < start_line:
            return "Error: end_line must be >= start_line."

        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = original.splitlines(keepends=True)
        if start_line > len(lines):
            return f"Error: start_line ({start_line}) exceeds file length ({len(lines)} lines)."

        # Clamp end_line to file length
        end_line = min(end_line, len(lines))

        content = parameters.get("content", "")
        replacement_lines = content.splitlines(keepends=True) if content else []
        # Ensure last replacement line has a newline if there are lines after it
        if replacement_lines and not replacement_lines[-1].endswith("\n") and end_line < len(lines):
            replacement_lines[-1] += "\n"

        new_lines = lines[:start_line - 1] + replacement_lines + lines[end_line:]
        modified = "".join(new_lines)

        try:
            self._backup(path)
            path.write_text(modified, encoding="utf-8")
        except Exception as e:
            return f"Error writing file: {e}"

        rel = parameters.get("path", "")
        deleted_count = end_line - start_line + 1
        inserted_count = len(replacement_lines)

        # Lint-gated: revert if syntax is broken
        error = self._validate_and_maybe_revert(path, modified, original, rel)
        if error:
            return error

        if not content:
            action_desc = f"Deleted lines {start_line}-{end_line} ({deleted_count} line(s))"
        else:
            action_desc = (f"Replaced lines {start_line}-{end_line} "
                           f"({deleted_count} -> {inserted_count} line(s))")
        diff_text = self._make_diff(original, modified, rel)
        result = f"{action_desc} in {rel}\n\n{diff_text}"
        return result

    @tool_action("file_undo", "Undo the last modification to a file")
    def _undo(self, parameters: Dict[str, Any]) -> str:
        """Revert a file to its state before the last write/edit/insert/replace_lines.

        Args:
            parameters: Dict with path.
        Returns:
            Success message with diff.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."

        key = str(path)
        if key not in self._backups:
            return f"Error: no backup available for {parameters.get('path')}."

        backup_content = self._backups.pop(key)

        try:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception:
            current = ""

        try:
            path.write_text(backup_content, encoding="utf-8")
        except Exception as e:
            # Restore the backup slot so the user can retry
            self._backups[key] = backup_content
            return f"Error restoring file: {e}"

        rel = parameters.get("path", "")
        diff_text = self._make_diff(current, backup_content, rel)
        return f"Reverted {rel} to previous state.\n\n{diff_text}"

    # ------------------------------------------------------------------ #
    #  Non-editing actions
    # ------------------------------------------------------------------ #

    @tool_action("file_list_dir", "List directory contents")
    def _list_dir(self, parameters: Dict[str, Any]) -> str:
        """List directory contents.

        Args:
            parameters: Dict with path.
        Returns:
            Directory listing.
        """
        rel = parameters.get("path", ".")
        path = self._safe_path(rel)
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: directory not found: {rel}"
        if not path.is_dir():
            return f"Error: not a directory: {rel}"

        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"Directory: {rel}/  ({len(entries)} entries)\n"]
        for entry in entries[:200]:  # limit
            try:
                entry_rel = entry.relative_to(self.workspace)
            except ValueError:
                entry_rel = entry.name
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry_rel}/")
            else:
                size = entry.stat().st_size
                lines.append(f"  [FILE] {entry_rel}  ({size} bytes)")
        if len(entries) > 200:
            lines.append(f"  ... and {len(entries) - 200} more entries")
        return "\n".join(lines)

    @tool_action("file_info", "Get metadata about a file")
    def _file_info(self, parameters: Dict[str, Any]) -> str:
        """Get file metadata.

        Args:
            parameters: Dict with path.
        Returns:
            File metadata string.
        """
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: not found: {parameters.get('path')}"

        stat = path.stat()
        info = {
            "path": str(parameters.get("path")),
            "type": "directory" if path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                info["line_count"] = text.count("\n") + 1
                info["encoding"] = "utf-8"
            except Exception:
                info["encoding"] = "binary"

        lines = [f"File Info: {info['path']}"]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
