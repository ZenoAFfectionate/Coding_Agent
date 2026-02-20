"""Structured Logging and Tracing for HelloAgents

Provides:
- Structured JSON logging with configurable handlers
- Agent-aware log context (agent name, step, tool, etc.)
- Event-based tracing for LLM calls, tool calls, and agent lifecycle
- Performance metrics (latency, token counts)
- File and console output with independent log levels

Usage:
    ```python
    from code.utils.logging import get_logger, setup_logger, AgentLogger

    # Basic usage (backward compatible)
    logger = get_logger("my_agent")
    logger.info("Agent started")

    # Structured agent logging
    agent_log = AgentLogger("MyCodingAgent")
    agent_log.llm_call(model="gpt-4", input_tokens=500, output_tokens=120, latency_ms=1500)
    agent_log.tool_call(tool="file_read", args={"path": "main.py"}, result="ok", latency_ms=5)
    agent_log.step(step_type="thought", content="Analyzing the code...")
    agent_log.error("Failed to parse output", exc_info=True)
    ```
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional


# ------------------------------------------------------------------ #
#  Basic logger setup (backward compatible)
# ------------------------------------------------------------------ #

def setup_logger(
    name: str = "hello_agents",
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup a logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        format_string: Custom format string.
        log_file: Optional file path to write logs to.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = format_string or "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (only add if not already present)
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    # File handler
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "hello_agents") -> logging.Logger:
    """Get or create a logger by name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


# ------------------------------------------------------------------ #
#  JSON Formatter
# ------------------------------------------------------------------ #

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields attached via AgentLogger
        for key in ("agent", "event", "data"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ #
#  AgentLogger - structured event logging
# ------------------------------------------------------------------ #

class AgentLogger:
    """Structured logger for agent events.

    Wraps a standard Python logger with convenience methods for
    common agent events (LLM calls, tool calls, steps, errors).
    Each event is logged as a structured JSON record.
    """

    def __init__(
        self,
        agent_name: str,
        level: str = "DEBUG",
        log_file: Optional[str] = None,
        use_json: bool = False,
    ):
        """Initialize an agent logger.

        Args:
            agent_name: Name of the agent.
            level: Minimum log level.
            log_file: Optional file to write logs to.
            use_json: If True, use JSON formatting for all output.
        """
        self.agent_name = agent_name
        self._logger = logging.getLogger(f"hello_agents.agent.{agent_name}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        self._logger.propagate = False  # don't duplicate to root

        if not self._logger.handlers:
            if use_json:
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    f"%(asctime)s | {agent_name} | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S",
                )

            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self._logger.addHandler(console)

            if log_file:
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setFormatter(JSONFormatter())  # always JSON for files
                self._logger.addHandler(fh)

    def _log_event(self, level: int, event: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a structured event."""
        extra = {"agent": self.agent_name, "event": event, "data": data or {}}
        self._logger.log(level, message, extra=extra)

    # -- Convenience methods --

    def llm_call(
        self,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
        **kwargs,
    ):
        """Log an LLM call event."""
        data = {"model": model, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "latency_ms": round(latency_ms, 1), **kwargs}
        self._log_event(logging.INFO, "llm_call",
                        f"LLM call: {model} ({input_tokens}+{output_tokens} tokens, {latency_ms:.0f}ms)", data)

    def tool_call(
        self,
        tool: str,
        args: Optional[Dict[str, Any]] = None,
        result: str = "",
        latency_ms: float = 0,
        success: bool = True,
    ):
        """Log a tool call event."""
        data = {"tool": tool, "args": args or {}, "result_preview": result[:200],
                "latency_ms": round(latency_ms, 1), "success": success}
        status = "ok" if success else "FAILED"
        self._log_event(logging.INFO, "tool_call",
                        f"Tool: {tool} [{status}] ({latency_ms:.0f}ms)", data)

    def step(self, step_type: str, content: str, step_num: int = 0):
        """Log an agent step."""
        data = {"step_type": step_type, "step_num": step_num, "content_preview": content[:200]}
        self._log_event(logging.INFO, "agent_step",
                        f"Step {step_num}: [{step_type}] {content[:100]}", data)

    def info(self, message: str, **data):
        """Log an info message."""
        self._log_event(logging.INFO, "info", message, data)

    def warning(self, message: str, **data):
        """Log a warning."""
        self._log_event(logging.WARNING, "warning", message, data)

    def error(self, message: str, exc_info: bool = False, **data):
        """Log an error."""
        extra = {"agent": self.agent_name, "event": "error", "data": data}
        self._logger.error(message, exc_info=exc_info, extra=extra)

    def lifecycle(self, event: str, **data):
        """Log an agent lifecycle event (start, end, reset)."""
        self._log_event(logging.INFO, f"lifecycle_{event}", f"Agent {event}", data)
