"""Agent prompt template loader.

All prompt templates live as .prompt files in this directory.
Use ``load_agent_prompt(name)`` to load a template by name (without the
.prompt extension).  Templates are cached after the first read.
"""

import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_cache: dict[str, str] = {}


def load_agent_prompt(name: str) -> str:
    """Load a prompt template from the agent_prompts directory.

    Args:
        name: Prompt file name (without .prompt extension).

    Returns:
        The raw prompt template string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    if name in _cache:
        return _cache[name]

    path = os.path.join(_PROMPTS_DIR, f"{name}.prompt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prompt template '{name}' not found at {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    _cache[name] = content
    return content
