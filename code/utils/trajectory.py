"""TrajectoryTracker - Agent Trajectory Recording & Visualization

Records every step of an agent's execution (thoughts, actions, observations,
LLM calls, tool calls) and provides structured output for debugging,
evaluation, and visualization.

Features:
- Step-by-step trajectory recording with timestamps
- Structured step types: thought, action, observation, llm_call, tool_call, error
- Token usage and cost tracking (optional)
- Export to multiple formats: plain text, Markdown, JSON
- Pretty-print for terminal visualization
- Trajectory statistics (total steps, duration, tool call counts, etc.)

Usage:
    ```python
    from code.utils.trajectory import TrajectoryTracker

    tracker = TrajectoryTracker(agent_name="MyCodingAgent")

    tracker.add_step("thought", "I need to read the file first.")
    tracker.add_step("tool_call", "file_read", metadata={"file": "main.py"})
    tracker.add_step("observation", "def main(): ...")
    tracker.add_step("thought", "Now I understand the code. Let me fix the bug.")

    # Print in terminal
    tracker.print_trajectory()

    # Export as Markdown
    md = tracker.to_markdown()

    # Export as JSON
    data = tracker.to_json()

    # Get statistics
    stats = tracker.get_stats()
    ```
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field, asdict


# Step types supported by the tracker
StepType = Literal[
    "thought",       # Agent's reasoning / CoT
    "action",        # Agent decides to take an action
    "observation",   # Result of an action / tool call
    "llm_call",      # Raw LLM invocation
    "tool_call",     # Tool invocation
    "error",         # An error occurred
    "plan",          # A plan was generated
    "reflection",    # Self-reflection on results
    "final_answer",  # The agent's final output
]


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory."""

    step_type: StepType
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    step_number: int = 0
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}


# Icon mapping for pretty-printing
_STEP_ICONS = {
    "thought":      "  ",
    "action":       "  ",
    "observation":  "  ",
    "llm_call":     "  ",
    "tool_call":    "  ",
    "error":        "  ",
    "plan":         "  ",
    "reflection":   "  ",
    "final_answer": "  ",
}

_STEP_LABELS = {
    "thought":      "Thought",
    "action":       "Action",
    "observation":  "Observation",
    "llm_call":     "LLM Call",
    "tool_call":    "Tool Call",
    "error":        "Error",
    "plan":         "Plan",
    "reflection":   "Reflection",
    "final_answer": "Final Answer",
}


class TrajectoryTracker:
    """Records and visualizes an agent's execution trajectory.

    Thread-safe for single-agent use. Each agent run should create or reset
    a tracker instance.
    """

    def __init__(self, agent_name: str = "Agent", task: str = ""):
        """Initialize a new trajectory tracker.

        Args:
            agent_name: Name of the agent being tracked.
            task: The initial task / user query being processed.
        """
        self.agent_name = agent_name
        self.task = task
        self.steps: List[TrajectoryStep] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._step_counter = 0
        self._timer_stack: List[float] = []  # for measuring step durations

        # Optional token / cost tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ------------------------------------------------------------------ #
    #  Recording
    # ------------------------------------------------------------------ #

    def start(self, task: str = ""):
        """Mark the start of a trajectory run.

        Args:
            task: Optionally set/override the task description.
        """
        if task:
            self.task = task
        self.start_time = time.time()

    def end(self):
        """Mark the end of a trajectory run."""
        self.end_time = time.time()

    def reset(self):
        """Clear all recorded steps and reset state."""
        self.steps.clear()
        self._step_counter = 0
        self.start_time = None
        self.end_time = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._timer_stack.clear()

    def add_step(
        self,
        step_type: StepType,
        content: str,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryStep:
        """Record a new step in the trajectory.

        Args:
            step_type: Category of the step (thought, action, observation, etc.).
            content: Text content of the step.
            duration_ms: Optional duration in milliseconds.
            metadata: Optional dict with extra info (tool name, token counts, etc.).

        Returns:
            The recorded TrajectoryStep.
        """
        if self.start_time is None:
            self.start()

        self._step_counter += 1
        step = TrajectoryStep(
            step_type=step_type,
            content=content,
            step_number=self._step_counter,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def start_timer(self):
        """Push a timer onto the stack (for measuring step durations)."""
        self._timer_stack.append(time.time())

    def stop_timer(self) -> float:
        """Pop the timer and return elapsed milliseconds.

        Returns:
            Elapsed time in milliseconds, or 0.0 if no timer was started.
        """
        if not self._timer_stack:
            return 0.0
        elapsed = (time.time() - self._timer_stack.pop()) * 1000
        return round(elapsed, 2)

    def add_token_usage(self, input_tokens: int = 0, output_tokens: int = 0):
        """Accumulate token usage counts.

        Args:
            input_tokens: Number of input/prompt tokens used.
            output_tokens: Number of output/completion tokens used.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Compute trajectory statistics.

        Returns:
            Dictionary with summary statistics.
        """
        elapsed = None
        if self.start_time:
            end = self.end_time or time.time()
            elapsed = round(end - self.start_time, 2)

        # Count steps by type
        type_counts: Dict[str, int] = {}
        for step in self.steps:
            type_counts[step.step_type] = type_counts.get(step.step_type, 0) + 1

        # Total duration from recorded durations
        total_measured_ms = sum(
            s.duration_ms for s in self.steps if s.duration_ms is not None
        )

        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "total_steps": len(self.steps),
            "step_counts": type_counts,
            "elapsed_seconds": elapsed,
            "total_measured_ms": round(total_measured_ms, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    # ------------------------------------------------------------------ #
    #  Export: Plain Text
    # ------------------------------------------------------------------ #

    def to_text(self) -> str:
        """Export trajectory as formatted plain text.

        Returns:
            Human-readable text representation.
        """
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f" Trajectory: {self.agent_name}")
        if self.task:
            lines.append(f" Task: {self.task}")
        lines.append(f"{'=' * 60}")

        for step in self.steps:
            icon = _STEP_ICONS.get(step.step_type, "  ")
            label = _STEP_LABELS.get(step.step_type, step.step_type)
            header = f"\n[Step {step.step_number}] {icon}{label}"
            if step.duration_ms is not None:
                header += f"  ({step.duration_ms:.0f}ms)"
            lines.append(header)
            lines.append(f"  {step.content}")
            # Metadata is preserved in JSON exports; only show non-redundant
            # keys in the text view to keep output concise.
            if step.metadata:
                shown = {k: v for k, v in step.metadata.items()
                         if k not in ("step", "tool")}
                for k, v in shown.items():
                    lines.append(f"    {k}: {v}")

        # Statistics footer
        stats = self.get_stats()
        lines.append(f"\n{'=' * 60}")
        lines.append(f" Summary: {stats['total_steps']} steps")
        if stats["elapsed_seconds"] is not None:
            lines.append(f" Duration: {stats['elapsed_seconds']}s")
        if stats["total_tokens"] > 0:
            lines.append(
                f" Tokens: {stats['total_input_tokens']} in / "
                f"{stats['total_output_tokens']} out"
            )
        lines.append(f"{'=' * 60}")

        return "\n".join(lines)

    def print_trajectory(self):
        """Print the trajectory to stdout."""
        print(self.to_text())

    # ------------------------------------------------------------------ #
    #  Export: Markdown
    # ------------------------------------------------------------------ #

    def to_markdown(self) -> str:
        """Export trajectory as Markdown.

        Returns:
            Markdown-formatted trajectory.
        """
        lines = []
        lines.append(f"# Agent Trajectory: {self.agent_name}\n")
        if self.task:
            lines.append(f"**Task:** {self.task}\n")

        for step in self.steps:
            icon = _STEP_ICONS.get(step.step_type, "")
            label = _STEP_LABELS.get(step.step_type, step.step_type)
            timing = ""
            if step.duration_ms is not None:
                timing = f" *({step.duration_ms:.0f}ms)*"

            lines.append(f"### Step {step.step_number}: {icon}{label}{timing}\n")

            # Use a blockquote for the content
            for content_line in step.content.split("\n"):
                lines.append(f"> {content_line}")
            lines.append("")

            if step.metadata:
                lines.append("**Metadata:**\n")
                for k, v in step.metadata.items():
                    lines.append(f"- `{k}`: {v}")
                lines.append("")

        # Stats section
        stats = self.get_stats()
        lines.append("---\n")
        lines.append("## Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Steps | {stats['total_steps']} |")
        if stats["elapsed_seconds"] is not None:
            lines.append(f"| Duration | {stats['elapsed_seconds']}s |")
        if stats["total_tokens"] > 0:
            lines.append(f"| Input Tokens | {stats['total_input_tokens']} |")
            lines.append(f"| Output Tokens | {stats['total_output_tokens']} |")

        step_counts = stats.get("step_counts", {})
        if step_counts:
            lines.append(f"\n**Step Breakdown:**\n")
            for stype, count in sorted(step_counts.items()):
                icon = _STEP_ICONS.get(stype, "")
                lines.append(f"- {icon}{stype}: {count}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Export: JSON
    # ------------------------------------------------------------------ #

    def to_json(self, indent: int = 2) -> str:
        """Export trajectory as JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        data = {
            "agent_name": self.agent_name,
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "stats": self.get_stats(),
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """Export trajectory as a Python dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "stats": self.get_stats(),
        }

    def save(self, path: str, fmt: str = "json"):
        """Save trajectory to a file.

        Args:
            path: Output file path.
            fmt: Format - 'json', 'markdown' / 'md', or 'text' / 'txt'.
        """
        fmt = fmt.lower()
        if fmt == "json":
            content = self.to_json()
        elif fmt in ("markdown", "md"):
            content = self.to_markdown()
        elif fmt in ("text", "txt"):
            content = self.to_text()
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'json', 'markdown', or 'text'.")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
