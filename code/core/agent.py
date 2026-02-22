"""Agent基类 - with async support, context management, and structured logging"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from ..utils.trajectory import TrajectoryTracker
from ..utils.logging import AgentLogger


# Directory where top-level prompt templates live.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the top-level prompts/ directory.

    Returns an empty string if the file is missing (non-fatal).
    """
    path = _PROMPTS_DIR / f"{name}.prompt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


class Agent(ABC):
    """Agent base class.

    All agents inherit from this class. It provides:
    - LLM engine integration
    - Conversation history management
    - Built-in trajectory tracking for debugging and evaluation
    - Structured logging via AgentLogger
    - Context budget helpers with LLM-based summarization
    - Session save / restore for cross-process persistence
    - Async run support (optional override)
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        enable_trajectory: bool = True,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        context_max_tokens: int = 0,
    ):
        """Initialize agent.

        Args:
            name: Agent name.
            llm: LLM engine instance.
            system_prompt: System prompt text.
            config: Agent configuration.
            enable_trajectory: Enable step-by-step trajectory recording.
            enable_logging: Enable structured AgentLogger.
            log_file: Optional file path for structured JSON logs.
            context_max_tokens: If > 0, enable automatic context budget management
                                (summarize history when messages exceed this budget).
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

        # Trajectory tracking
        self.enable_trajectory = enable_trajectory
        self.trajectory = TrajectoryTracker(agent_name=name) if enable_trajectory else None

        # Structured logging
        self.logger = AgentLogger(name, level=self.config.log_level, log_file=log_file) if enable_logging else None

        # Context budget (token management)
        self.context_max_tokens = context_max_tokens

        # Cached tiktoken encoder for token counting
        self._tiktoken_enc = None
        try:
            import tiktoken
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """Run the agent synchronously."""
        pass

    async def arun(self, input_text: str, **kwargs) -> str:
        """Run the agent asynchronously (default: wraps sync run).
           Subclasses can override for true async implementations."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(input_text, **kwargs))


    # ------------------------------------------------------------------ #
    #  Trajectory tracking
    # ------------------------------------------------------------------ #

    def _track(self, step_type: str, content: str, duration_ms: float = None, **metadata) -> None:
        """Record a step in the agent's trajectory (no-op if disabled)."""
        if self.trajectory is not None:
            self.trajectory.add_step(
                step_type, content,
                duration_ms=duration_ms,
                metadata=metadata if metadata else None,
            )

    def _log(self, level: str, message: str, **data) -> None:
        """Log a message via AgentLogger (no-op if disabled)."""
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(message, **data)

    def get_trajectory(self) -> Optional[TrajectoryTracker]:
        """Return the trajectory tracker (or None if disabled)."""
        return self.trajectory

    def print_trajectory(self) -> None:
        """Pretty-print the trajectory to stdout."""
        if self.trajectory is not None:
            self.trajectory.print_trajectory()
        else:
            print("Trajectory tracking is disabled for this agent.")

    def save_trajectory(self, path: str, fmt: str = "json") -> None:
        """Save trajectory to file."""
        if self.trajectory is not None:
            self.trajectory.save(path, fmt)
        else:
            print("Trajectory tracking is disabled for this agent.")


    # ------------------------------------------------------------------ #
    #  Context / token budget management  (Claude Code-style compaction)
    # ------------------------------------------------------------------ #

    # Compaction is triggered when total tokens reach this fraction of the
    # budget.  For example, with context_max_tokens=200000 and a threshold
    # of 0.85, compaction fires at ~170K tokens — similar to Claude Code's
    # 167K/200K trigger.
    COMPACTION_THRESHOLD = 0.85

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a text string (cached encoder)."""
        if self._tiktoken_enc is not None:
            return len(self._tiktoken_enc.encode(text))
        return len(text) // 4

    def _count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens across all messages."""
        return sum(self._count_tokens(m.get("content", "")) for m in messages)

    def _compact_messages(self, messages: list[dict]) -> str:
        """Compact a list of messages into a single summary using the LLM.

        This is the core compaction operation — it takes ALL messages that
        need to be compressed and produces one concise summary preserving
        key information for continued reasoning.

        Falls back to a mechanical summary if the LLM call fails.
        """
        # Build textual representation for the prompt
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            lines.append(f"[{role}]: {content}")
        messages_text = "\n\n".join(lines)

        template = _load_prompt("context_compaction")
        if not template:
            return self._mechanical_summary(messages)

        prompt = (template
                  .replace("{messages}", messages_text)
                  .replace("{message_count}", str(len(messages))))

        try:
            summary = self.llm.invoke(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if summary and summary.strip():
                return summary.strip()
        except Exception:
            pass

        return self._mechanical_summary(messages)

    @staticmethod
    def _mechanical_summary(messages: list[dict]) -> str:
        """Create a simple mechanical summary when LLM compaction is unavailable.

        This is a best-effort fallback that extracts key signals from the
        messages without calling the LLM.
        """
        user_msgs = [m for m in messages if m.get("role") == "user"]
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        parts: list[str] = [
            f"## Conversation History (compacted from {len(messages)} messages)"
        ]

        # User requests
        if user_msgs:
            parts.append("\n### User Requests")
            for m in user_msgs:
                first_line = m.get("content", "").split("\n")[0][:150]
                if first_line:
                    parts.append(f"- {first_line}")

        # Assistant responses summary
        if asst_msgs:
            parts.append(f"\n### Assistant Responses: {len(asst_msgs)} messages")
            for m in asst_msgs:
                first_line = m.get("content", "").split("\n")[0][:150]
                if first_line:
                    parts.append(f"- {first_line}")

        # Tool results summary
        if tool_msgs:
            parts.append(f"\n### Tool Results: {len(tool_msgs)} calls")
            for m in tool_msgs:
                name = m.get("name", "unknown")
                content = m.get("content", "")
                preview = content[:100].replace("\n", " ")
                parts.append(f"- {name}: {preview}")

        return "\n".join(parts)

    def _manage_context_budget(self, messages: list[dict]) -> list[dict]:
        """Manage context window via full compaction (Claude Code-style).

        When the conversation approaches the token budget (controlled by
        ``COMPACTION_THRESHOLD``), the ENTIRE conversation — excluding the
        system prompt and the most recent user message — is compressed into
        a summary via the LLM.  This is fundamentally different from
        incremental summarization:

        - **Before**: [system] [user1] [asst1] ... [userN]
        - **After**:  [system] [user: summary_request] [assistant: summary] [userN]

        The summary is placed as a **user/assistant pair**, matching how
        Claude Code and the Claude API handle compaction: the summary prompt
        is injected as a user turn, and the model's summary is stored as an
        assistant turn.  This preserves proper conversation turn-taking and
        causes the model to treat the summary as its own prior context
        rather than as a system instruction.

        Args:
            messages: List of message dicts (role/content).

        Returns:
            Compacted message list that fits within the budget.
        """
        if self.context_max_tokens <= 0:
            return messages

        total = self._count_messages_tokens(messages)
        trigger = int(self.context_max_tokens * self.COMPACTION_THRESHOLD)

        if total <= trigger:
            return messages

        # --- Compaction triggered ---
        self._print(
            f"  [Compaction] Context ({total} tokens) exceeded threshold "
            f"({trigger} tokens). Compacting conversation...",
        )

        # Separate system prompts, conversation body, and current input.
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        if not non_system: return messages

        # The last message is the current user input — always preserved.
        current_input = non_system[-1]
        conversation_body = non_system[:-1]

        if not conversation_body:
            return messages

        # Compact the entire conversation body into one summary.
        summary_text = self._compact_messages(conversation_body)

        # Assemble the compacted message list:
        #   [system_prompts] + [user: request] + [assistant: summary] + [current_input]
        #
        # This mirrors the Claude API compaction pattern:
        #   - Summary prompt is injected as a user turn
        #   - Model's summary is stored as an assistant turn
        #   - Preserves proper turn-taking for OpenAI-compatible APIs
        result = list(system_msgs)
        result.append({
            "role": "user",
            "content": (
                "The conversation history has been compacted. "
                "Below is a summary of everything discussed so far. "
                "Use it as context to continue the conversation."
            ),
        })
        result.append({
            "role": "assistant",
            "content": summary_text,
        })
        result.append(current_input)

        new_total = self._count_messages_tokens(result)
        self._print(
            f"  [Compaction] Done: {len(messages)} messages -> {len(result)} messages, "
            f"{total} tokens -> {new_total} tokens",
        )

        return result


    # ------------------------------------------------------------------ #
    #  Rich history — execution summary extraction
    # ------------------------------------------------------------------ #

    def _build_execution_summary(self, input_text: str, answer: str) -> str:
        """Build a rich execution summary from the current trajectory.

        Extracts tool calls, files modified, errors, and key observations
        from the trajectory and produces a compact summary that will be
        stored alongside the assistant's answer in _history.

        If trajectory tracking is disabled or no steps were recorded,
        returns the bare answer as-is.
        """
        if self.trajectory is None or not self.trajectory.steps:
            return answer

        # Extract structured information from trajectory steps
        tool_calls: list[str] = []
        files: list[str] = []
        errors: list[str] = []
        observations: list[str] = []

        for step in self.trajectory.steps:
            if step.step_type == "action":
                tool_calls.append(step.content)
            elif step.step_type == "tool_call":
                tool_calls.append(step.content)
            elif step.step_type == "observation":
                # Keep observations short for the summary
                obs_preview = step.content[:200]
                if len(step.content) > 200:
                    obs_preview += "..."
                observations.append(obs_preview)
                # Track file paths from metadata
                tool_name = step.metadata.get("tool", "")
                if tool_name == "file":
                    path = step.metadata.get("path", "")
                    if path and path not in files:
                        files.append(path)
            elif step.step_type == "error":
                errors.append(step.content[:200])

        # Also scan action steps for file paths
        for step in self.trajectory.steps:
            if step.step_type == "action" and "file" in step.content.lower():
                # Try to extract file paths from tool call args
                import re
                path_match = re.search(r'"path"\s*:\s*"([^"]+)"', step.content)
                if path_match:
                    p = path_match.group(1)
                    if p not in files:
                        files.append(p)

        # If no tools were called, this was a simple direct response
        if not tool_calls:
            return answer

        # Build a compact inline summary
        summary_parts = [answer, "", "---", "[Execution context for next turn]"]
        summary_parts.append(f"Tools called: {', '.join(tool_calls[:10])}")
        if files:
            summary_parts.append(f"Files touched: {', '.join(files[:10])}")
        if errors:
            summary_parts.append(f"Errors encountered: {'; '.join(errors[:5])}")
        if observations:
            # Pick the most informative observations (last few are usually most relevant)
            key_obs = observations[-3:] if len(observations) > 3 else observations
            summary_parts.append("Key observations:")
            for obs in key_obs:
                summary_parts.append(f"  - {obs}")

        return "\n".join(summary_parts)


    # ------------------------------------------------------------------ #
    #  Session persistence
    # ------------------------------------------------------------------ #

    def save_session(self, path: str) -> None:
        """Save conversation history to a JSON file for later restoration.

        Args:
            path: File path to write the session data to.
        """
        session_data = {
            "agent_name": self.name,
            "saved_at": datetime.now().isoformat(),
            "model": self.llm.model,
            "provider": self.llm.provider,
            "history": [
                {
                    "content": msg.content,
                    "role": msg.role,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata,
                }
                for msg in self._history
            ],
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def load_session(self, path: str) -> bool:
        """Load conversation history from a previously saved session file.

        Args:
            path: File path to read the session data from.

        Returns:
            True if session was loaded successfully, False otherwise.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        history_data = session_data.get("history", [])
        if not history_data:
            return False

        self._history.clear()
        for entry in history_data:
            ts = None
            if entry.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                except (ValueError, TypeError):
                    ts = datetime.now()

            msg = Message(
                content=entry["content"],
                role=entry["role"],
                timestamp=ts,
                metadata=entry.get("metadata", {}),
            )
            self._history.append(msg)

        return True

    @staticmethod
    def get_default_session_path(agent_name: str) -> str:
        """Return the default session file path for a given agent name.

        Args:
            agent_name: Name of the agent.

        Returns:
            Path string under the project's results/sessions/ directory.
        """
        sessions_dir = _PROMPTS_DIR.parent / "results" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        safe_name = agent_name.replace(" ", "_").replace("/", "_")
        return str(sessions_dir / f"{safe_name}_session.json")


    def _print(self, msg: str, level: str = "debug") -> None:
        """Print a message, respecting the config.debug flag.

        Args:
            msg: The message to print.
            level: "debug" only prints when self.config.debug is True;
                   "info" always prints (errors, warnings, final results).
        """
        if level == "info" or self.config.debug:
            print(msg)


    def add_message(self, message: Message):
        """Add a message to conversation history.

        Enforces config.max_history_length by trimming the oldest messages.
        """
        self._history.append(message)
        max_len = self.config.max_history_length
        if max_len > 0 and len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """Get a copy of conversation history."""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()
