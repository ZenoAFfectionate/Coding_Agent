"""FileTool - File Read / Write / Edit Tool

Provides safe file-system operations for a Coding Agent:
- Read files with optional line-number display and range selection
- Write / create files with content
- Edit files using exact string replacement (like sed but safer)
- List directory contents (tree view)
- Get file metadata (size, modification time, etc.)

Safety:
- All paths are sandboxed to a configurable workspace root
- Symlink traversal outside workspace is blocked
- Maximum file size limits for reads and writes
"""

import os
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter, tool_action


class FileTool(Tool):
    """File operations tool for Coding Agents.

    Supports read, write, edit (string-replace), list_dir, and file_info actions.
    All paths are resolved relative to a sandboxed workspace root.
    """

    def __init__(
        self,
        workspace: str = ".",
        max_read_size: int = 1024 * 1024,      # 1 MB
        max_write_size: int = 5 * 1024 * 1024,  # 5 MB
        expandable: bool = False,
    ):
        super().__init__(
            name="file",
            description=(
                "File operations tool - read, write, edit files, list directories, "
                "and get file info (sandboxed to workspace)"
            ),
            expandable=expandable,
        )
        self.workspace = Path(workspace).resolve()
        self.max_read_size = max_read_size
        self.max_write_size = max_write_size
        self.workspace.mkdir(parents=True, exist_ok=True)

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
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "read")
        dispatch = {
            "read": self._read,
            "write": self._write,
            "edit": self._edit,
            "list_dir": self._list_dir,
            "file_info": self._file_info,
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="Action: read, write, edit, list_dir, file_info", required=True),
            ToolParameter(name="path", type="string",
                          description="Relative file/directory path", required=True),
            ToolParameter(name="content", type="string",
                          description="File content for write action", required=False),
            ToolParameter(name="old_string", type="string",
                          description="String to find (edit action)", required=False),
            ToolParameter(name="new_string", type="string",
                          description="Replacement string (edit action)", required=False),
            ToolParameter(name="start_line", type="integer",
                          description="Start line for read (1-based, default 1)", required=False, default=1),
            ToolParameter(name="end_line", type="integer",
                          description="End line for read (inclusive, default: end of file)", required=False),
            ToolParameter(name="show_line_numbers", type="boolean",
                          description="Show line numbers in read output (default true)", required=False, default=True),
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
        start = max(1, parameters.get("start_line", 1) or 1)
        end = parameters.get("end_line") or len(lines)
        end = min(end, len(lines))

        selected = lines[start - 1: end]
        show_nums = parameters.get("show_line_numbers", True)

        if show_nums:
            width = len(str(end))
            out_lines = [f"{str(i).rjust(width)}| {line.rstrip()}" for i, line in enumerate(selected, start=start)]
        else:
            out_lines = [line.rstrip() for line in selected]

        header = f"File: {parameters.get('path')}  (lines {start}-{end} of {len(lines)})\n"
        return header + "\n".join(out_lines)

    @tool_action("file_write", "Create or overwrite a file")
    def _write(self, parameters: Dict[str, Any]) -> str:
        """Write content to a file.

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
            path.write_text(content, encoding="utf-8")
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Successfully wrote {len(content)} bytes ({line_count} lines) to {parameters.get('path')}"
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
            path.write_text(modified, encoding="utf-8")
        except Exception as e:
            return f"Error writing file: {e}"

        # Generate a unified diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{parameters.get('path')}",
            tofile=f"b/{parameters.get('path')}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        return f"Successfully edited {parameters.get('path')}\n\n{diff_text}"

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
