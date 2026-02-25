"""GitTool - Git Operations Tool

Provides safe Git operations for a Coding Agent, including:
- Repository status and info
- Diff viewing (staged, unstaged, between commits)
- Commit history browsing
- Branch management
- Staging and committing changes
- Stash operations

Safety features:
- Destructive operations (force push, hard reset) are disallowed by default
- Working directory is sandboxed to the specified repository path
- Timeout control for all operations
- Output size limits

Usage:
    ```python
    git_tool = GitTool(repo_path="./my-project")

    # Check status
    result = git_tool.run({"action": "status"})

    # View diff
    result = git_tool.run({"action": "diff"})

    # Browse commit log
    result = git_tool.run({"action": "log", "limit": 10})

    # Stage and commit
    result = git_tool.run({"action": "add", "files": "."})
    result = git_tool.run({"action": "commit", "message": "feat: add new feature"})
    ```
"""

import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...utils.subprocess_utils import safe_run
from ..base import Tool, ToolParameter, tool_action


class GitTool(Tool):
    """Git Operations Tool

    Provides safe Git operations for Coding Agents, including
    status checks, diff viewing, commit history, branch management,
    staging, and committing.

    Safety:
    - Destructive commands (force push, hard reset, rebase) are blocked by default
    - All operations are sandboxed to the configured repo_path
    - Configurable timeout and output size limits
    """

    # Commands considered dangerous and blocked unless explicitly allowed
    DANGEROUS_PATTERNS = [
        "push --force", "push -f",
        "reset --hard",
        "rebase",
        "clean -fd", "clean -f",
        "checkout --",      # can discard uncommitted changes
        "branch -D",        # force delete branch
    ]

    def __init__(
        self,
        repo_path: str = ".",
        timeout: int = 30,
        max_output_size: int = 5 * 1024 * 1024,  # 5 MB
        allow_destructive: bool = False,
        expandable: bool = False,
    ):
        """Initialize GitTool.

        Args:
            repo_path: Path to the Git repository root.
            timeout: Maximum seconds for any git command.
            max_output_size: Maximum output size in bytes before truncation.
            allow_destructive: If True, allow dangerous commands like force-push.
            expandable: Whether to expand into sub-tools via @tool_action.
        """
        super().__init__(
            name="git",
            description=(
                "Git operations tool - check status, view diffs, browse history, "
                "manage branches, stage files, and create commits"
            ),
            expandable=expandable,
        )
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_destructive = allow_destructive

    # ------------------------------------------------------------------ #
    #  Core execution helper
    # ------------------------------------------------------------------ #

    def _run_git(self, args: List[str]) -> str:
        """Execute a git command and return its output.

        Args:
            args: Arguments to pass after ``git``, e.g. ["status", "--short"].

        Returns:
            Combined stdout/stderr as a string.
        """
        cmd = ["git"] + args
        cmd_str = " ".join(cmd)

        # Safety check
        if not self.allow_destructive:
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in cmd_str:
                    return f"Blocked: '{cmd_str}' is a destructive operation. Set allow_destructive=True to permit it."

        try:
            result = safe_run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            if len(output) > self.max_output_size:
                output = output[: self.max_output_size]
                output += f"\n\n... output truncated (exceeded {self.max_output_size} bytes)"

            if result.returncode != 0:
                output = f"[exit code {result.returncode}]\n{output}"

            return output.strip() if output.strip() else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: git command timed out after {self.timeout}s"
        except FileNotFoundError:
            return "Error: 'git' executable not found. Is Git installed?"
        except Exception as e:
            return f"Error executing git command: {e}"

    # ------------------------------------------------------------------ #
    #  Action dispatch (non-expanded mode)
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        """Execute a git action."""
        action = parameters.get("action", "status")

        dispatch = {
            "status": self._status,
            "diff": self._diff,
            "log": self._log,
            "show": self._show,
            "branch": self._branch,
            "add": self._add,
            "commit": self._commit,
            "stash": self._stash,
            "blame": self._blame,
        }

        handler = dispatch.get(action)
        if handler is None:
            supported = ", ".join(sorted(dispatch.keys()))
            return f"Unsupported action: '{action}'. Supported actions: {supported}"

        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        """Return tool parameter definitions."""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Git action to perform",
                required=True,
                enum=["status", "diff", "log", "show", "branch", "add", "commit", "stash", "blame"],
            ),
            ToolParameter(
                name="files",
                type="string",
                description="File path(s) for add/diff/blame (space-separated). Default: '.' for add.",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="message",
                type="string",
                description="Commit message (required for commit action).",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Number of log entries to show (default: 10).",
                required=False,
                default=10,
            ),
            ToolParameter(
                name="ref",
                type="string",
                description="Git ref (commit hash, branch name, tag) for show/diff/log.",
                required=False,
            ),
            ToolParameter(
                name="branch_name",
                type="string",
                description="Branch name for branch operations (create/switch/delete).",
                required=False,
            ),
            ToolParameter(
                name="sub_action",
                type="string",
                description=(
                    "Sub-action for branch/stash. "
                    "Branch: list, create, switch, delete. "
                    "Stash: save, pop, list, drop."
                ),
                required=False,
                default="list",
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Individual actions
    # ------------------------------------------------------------------ #

    @tool_action("git_status", "Show working tree status and branch info")
    def _status(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Show git status.

        Args:
            parameters: Optional parameters dict (unused).

        Returns:
            Git status output.
        """
        branch_info = self._run_git(["branch", "--show-current"])
        status_output = self._run_git(["status", "--short", "--branch"])
        return f"Current branch: {branch_info}\n\n{status_output}"

    @tool_action("git_diff", "Show file differences (staged or unstaged)")
    def _diff(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Show git diff.

        Args:
            parameters: Optional dict with 'files' and 'ref' keys.

        Returns:
            Diff output.
        """
        parameters = parameters or {}
        args = ["diff"]

        ref = parameters.get("ref")
        if ref:
            args.append(ref)

        files = parameters.get("files")
        if files and files != ".":
            args.append("--")
            args.extend(files.split())

        unstaged = self._run_git(args)

        # Also show staged changes
        staged_args = ["diff", "--cached"]
        if files and files != ".":
            staged_args.append("--")
            staged_args.extend(files.split())
        staged = self._run_git(staged_args)

        parts = []
        if staged and staged != "(no output)":
            parts.append(f"=== Staged Changes ===\n{staged}")
        if unstaged and unstaged != "(no output)":
            parts.append(f"=== Unstaged Changes ===\n{unstaged}")

        return "\n\n".join(parts) if parts else "No changes detected."

    @tool_action("git_log", "Show commit history")
    def _log(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Show git log.

        Args:
            parameters: Optional dict with 'limit' and 'ref' keys.

        Returns:
            Formatted commit log.
        """
        parameters = parameters or {}
        limit = parameters.get("limit", 10)
        args = [
            "log",
            f"-{limit}",
            "--oneline",
            "--graph",
            "--decorate",
            "--date=short",
            "--format=%C(auto)%h %C(blue)%ad %C(green)%an%C(reset) %s%C(auto)%d",
        ]

        ref = parameters.get("ref")
        if ref:
            args.append(ref)

        return self._run_git(args)

    @tool_action("git_show", "Show details of a specific commit")
    def _show(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Show a specific commit.

        Args:
            parameters: Optional dict with 'ref' key.

        Returns:
            Commit details.
        """
        parameters = parameters or {}
        ref = parameters.get("ref", "HEAD")
        return self._run_git(["show", "--stat", ref])

    @tool_action("git_branch", "List, create, switch, or delete branches")
    def _branch(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Manage branches.

        Args:
            parameters: Dict with 'sub_action' and optionally 'branch_name'.

        Returns:
            Branch operation result.
        """
        parameters = parameters or {}
        sub_action = parameters.get("sub_action", "list")
        branch_name = parameters.get("branch_name")

        if sub_action == "list":
            return self._run_git(["branch", "-a", "--list"])

        if not branch_name:
            return "Error: 'branch_name' is required for create/switch/delete."

        if sub_action == "create":
            return self._run_git(["branch", branch_name])
        elif sub_action == "switch":
            return self._run_git(["checkout", branch_name])
        elif sub_action == "delete":
            return self._run_git(["branch", "-d", branch_name])
        else:
            return f"Unknown branch sub_action: '{sub_action}'. Use list/create/switch/delete."

    @tool_action("git_add", "Stage files for commit")
    def _add(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Stage files.

        Args:
            parameters: Dict with 'files' key (default '.').

        Returns:
            Staging result.
        """
        parameters = parameters or {}
        files = parameters.get("files", ".")
        args = ["add"] + files.split()
        result = self._run_git(args)

        # Show what was staged
        staged_status = self._run_git(["status", "--short"])
        return f"{result}\n\nCurrent status:\n{staged_status}"

    @tool_action("git_commit", "Create a new commit with staged changes")
    def _commit(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a commit.

        Args:
            parameters: Dict with 'message' key (required).

        Returns:
            Commit result.
        """
        parameters = parameters or {}
        message = parameters.get("message")
        if not message:
            return "Error: 'message' is required for commit."
        return self._run_git(["commit", "-m", message])

    @tool_action("git_stash", "Stash or restore uncommitted changes")
    def _stash(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Manage stash.

        Args:
            parameters: Dict with 'sub_action' key (save/pop/list/drop).

        Returns:
            Stash operation result.
        """
        parameters = parameters or {}
        sub_action = parameters.get("sub_action", "list")
        message = parameters.get("message")

        if sub_action == "save":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            return self._run_git(args)
        elif sub_action == "pop":
            return self._run_git(["stash", "pop"])
        elif sub_action == "list":
            return self._run_git(["stash", "list"])
        elif sub_action == "drop":
            return self._run_git(["stash", "drop"])
        else:
            return f"Unknown stash sub_action: '{sub_action}'. Use save/pop/list/drop."

    @tool_action("git_blame", "Show line-by-line authorship of a file")
    def _blame(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Show git blame for a file.

        Args:
            parameters: Dict with 'files' key (single file path).

        Returns:
            Blame output.
        """
        parameters = parameters or {}
        files = parameters.get("files")
        if not files:
            return "Error: 'files' (a single file path) is required for blame."
        return self._run_git(["blame", files.split()[0]])
