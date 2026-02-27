"""Swarm orchestration package — multi-agent coordination primitives.

Logging setup and public re-exports live here so that downstream code
can simply ``from code.swarm import ...``.
"""

import logging
from pathlib import Path

from .paths import (
    PROJECT_ROOT,
    PROMPTS_DIR,
    RESULTS_DIR,
    DATA_DIR,
    DEFAULT_BATCH_TASK_PROMPT,
    load_swarm_prompt,
)

# ---------------------------------------------------------------------------
# Logging setup (was logging_setup.py)
# ---------------------------------------------------------------------------

logger = logging.getLogger("multi_agent")


def setup_logging(log_dir: Path = None) -> Path:
    """Configure file logging for detailed multi-agent interaction traces."""
    log_dir = log_dir or (RESULTS_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "multi_agent.log"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
    return log_file


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

from .blackboard import Blackboard
from .worker_factory import WORKER_SPECS, build_worker
from .runners import run_batch, repl, _cleanup_sandbox, _print_sandbox_code

__all__ = [
    "PROJECT_ROOT",
    "PROMPTS_DIR",
    "RESULTS_DIR",
    "DATA_DIR",
    "DEFAULT_BATCH_TASK_PROMPT",
    "load_swarm_prompt",
    "logger",
    "setup_logging",
    "Blackboard",
    "WORKER_SPECS",
    "build_worker",
    "run_batch",
    "repl",
    "_cleanup_sandbox",
    "_print_sandbox_code",
]
