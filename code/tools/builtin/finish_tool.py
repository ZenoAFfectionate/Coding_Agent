"""Finish tool -- signals that the agent has completed its task."""

from typing import Dict, Any, List

from ..base import Tool, ToolParameter


class FinishTool(Tool):
    """Signal that the task is complete and provide a final answer.

    When the agent calls this tool, the main loop breaks immediately
    and the ``result`` parameter is returned as the final answer.
    """

    SENTINEL = "__FINISH__"  # prefix used by the agent loop for detection

    def __init__(self):
        super().__init__(
            name="finish",
            description=(
                "Signal that you have completed the task. "
                "Call this tool with a concise description of the solution or fix you implemented."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="result",
                type="string",
                description="A concise description of the solution or fix you implemented.",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        result = parameters.get("result", "")
        return f"{self.SENTINEL}{result}"
