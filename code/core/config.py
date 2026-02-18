"""配置管理"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents Config Class

    Central configuration for agent behavior and LLM defaults.
    Fields here are the single source of truth — build_agent() reads them
    when constructing the LLM and agent instances.
    """

    # LLM config — None means "let HelloAgentsLLM auto-detect from env"
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # system config
    debug: bool = False
    log_level: str = "INFO"

    # history config
    max_history_length: int = 128

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            default_model=os.getenv("LLM_MODEL_ID"),
            default_provider=os.getenv("LLM_PROVIDER"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return self.model_dump()
