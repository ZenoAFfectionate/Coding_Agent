"""Blackboard — structured workspace state shared across orchestrator and workers."""

from pathlib import Path


class Blackboard:
    """Structured workspace state shared across the orchestrator and workers."""

    def __init__(self, user_request: str):
        self.user_request = user_request
        self.files_examined: list[str] = []
        self.files_created: list[str] = []
        self.findings: list[dict] = []   # {"source", "round", "summary"}
        self.errors: list[dict] = []     # {"source", "round", "message"}
        self.current_plan: str = ""

    def add_finding(self, source: str, round_num: int, summary: str) -> None:
        self.findings.append({
            "source": source,
            "round": round_num,
            "summary": summary,
        })

    def add_error(self, source: str, round_num: int, message: str) -> None:
        self.errors.append({
            "source": source,
            "round": round_num,
            "message": message,
        })

    def scan_workspace_files(self, workspace: str) -> None:
        """Scan the workspace for .py files and update files_created."""
        root = Path(workspace)
        self.files_created = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if "__pycache__" not in p.parts
        )

    def serialize(self) -> str:
        """Render the blackboard as a concise text block for injection into prompts."""
        parts: list[str] = []
        parts.append(f"User request: {self.user_request}")
        if self.current_plan:
            parts.append(f"\nPlan: {self.current_plan}")
        if self.files_created:
            parts.append(f"\nFiles created in sandbox: {', '.join(self.files_created)}")
        if self.files_examined:
            parts.append(f"\nFiles examined: {', '.join(self.files_examined)}")
        if self.findings:
            parts.append("\nFindings:")
            for f in self.findings:
                parts.append(f"  - [{f['source']} R{f['round']}] {f['summary']}")
        if self.errors:
            parts.append("\nErrors:")
            for e in self.errors:
                parts.append(f"  - [{e['source']} R{e['round']}] {e['message']}")
        if not self.findings and not self.errors:
            parts.append("\n(no findings yet)")
        return "\n".join(parts)
