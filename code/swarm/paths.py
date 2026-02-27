"""Canonical path constants and prompt-loading helper for the swarm package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "xCode"

DEFAULT_BATCH_TASK_PROMPT = (
    "{problem}\n\n"
    "Write a complete Python program that reads from stdin and writes to "
    "stdout. Save it as a single .py file using the file tool."
)


def load_swarm_prompt(name: str) -> str:
    """Load a prompt file from the prompts/ directory by stem name.

    Returns an empty string if the file is missing (non-fatal).
    """
    path = PROMPTS_DIR / f"{name}.prompt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
