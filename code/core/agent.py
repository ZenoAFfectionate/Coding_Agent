"""Agent基类 - with async support, context management, and structured logging"""

from abc import ABC, abstractmethod
from typing import Optional

from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from ..utils.trajectory import TrajectoryTracker
from ..utils.logging import AgentLogger


class Agent(ABC):
    """Agent base class.

    All agents inherit from this class. It provides:
    - LLM engine integration
    - Conversation history management
    - Built-in trajectory tracking for debugging and evaluation
    - Structured logging via AgentLogger
    - Context budget helpers for token management
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
                                (truncate history when messages exceed this budget).
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
        self.logger = AgentLogger(name, log_file=log_file) if enable_logging else None

        # Context budget (token management)
        self.context_max_tokens = context_max_tokens

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
    #  Context / token budget management
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def _manage_context_budget(self, messages: list[dict]) -> list[dict]:
        """Trim messages to fit within context_max_tokens budget.

        Keeps the system prompt and most recent messages, summarizing
        or dropping older history when the budget is exceeded.

        Args:
            messages: List of message dicts (role/content).

        Returns:
            Trimmed message list.
        """
        if self.context_max_tokens <= 0:
            return messages

        total = sum(self._count_tokens(m.get("content", "")) for m in messages)
        if total <= self.context_max_tokens:
            return messages

        # Keep system prompt(s) and the last user message always
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        # Always keep the last message (current user input)
        if non_system:
            last_msg = non_system[-1]
            middle = non_system[:-1]
        else:
            return messages

        # Calculate budget remaining after system + last
        used = sum(self._count_tokens(m.get("content", "")) for m in system_msgs)
        used += self._count_tokens(last_msg.get("content", ""))
        remaining = self.context_max_tokens - used

        # Fill from most recent backward
        kept = []
        for m in reversed(middle):
            t = self._count_tokens(m.get("content", ""))
            if remaining - t >= 0:
                kept.insert(0, m)
                remaining -= t
            else:
                break

        # If we dropped messages, add a summary marker
        dropped = len(middle) - len(kept)
        result = list(system_msgs)
        if dropped > 0:
            result.append({
                "role": "system",
                "content": f"[{dropped} earlier messages omitted to fit context budget]",
            })
        result.extend(kept)
        result.append(last_msg)
        return result


    # ------------------------------------------------------------------ #
    #  History management
    # ------------------------------------------------------------------ #

    def add_message(self, message: Message):
        """Add a message to conversation history."""
        self._history.append(message)

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
