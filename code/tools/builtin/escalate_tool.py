"""Escalate tool -- signals that the worker is stuck and returns control to the orchestrator."""

from typing import Dict, Any, List

from ..base import Tool, ToolParameter


class EscalateTool(Tool):
    """Signal that the worker cannot complete the task and needs orchestrator help.

    When the agent calls this tool, the main loop breaks immediately
    and the ``reason`` (plus optional ``suggestion``) is returned to the
    orchestrator so it can re-plan or dispatch a different worker.
    """

    SENTINEL = "__ESCALATE__"  # prefix used by the agent loop for detection

    def __init__(self):
        super().__init__(
            name="escalate",
            description=(
                "Signal that you are stuck or the current approach is not working. "
                "Call this when you have tried 2+ approaches without success, or "
                "when you realize the task requires a fundamentally different strategy. "
                "The orchestrator will re-plan and may dispatch a different worker."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="reason",
                type="string",
                description="Why you are escalating (what failed, what you tried).",
                required=True,
            ),
            ToolParameter(
                name="suggestion",
                type="string",
                description="Optional suggestion for the orchestrator on what to try next.",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        reason = parameters.get("reason", "No reason provided")
        suggestion = parameters.get("suggestion", "")
        payload = reason
        if suggestion:
            payload += f" | Suggestion: {suggestion}"
        return f"{self.SENTINEL}{payload}"
