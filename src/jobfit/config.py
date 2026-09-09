"""Runtime configuration.

Every field has a working default, so the test suite and the ``--offline`` demo
run with no ``.env`` file and no API key at all. That is a hard requirement, not
a convenience: see ``tests/conftest.py``.

Environment variables are all prefixed ``JOBFIT_``. In particular the Anthropic
key is read from ``JOBFIT_ANTHROPIC_API_KEY``, deliberately *not* from the
``ANTHROPIC_API_KEY`` that the SDK picks up implicitly - we always pass the key
in explicitly, so a stray ambient variable can never turn a supposedly offline
run into a billed one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TransportName = Literal["fake", "ollama", "anthropic"]


class Settings(BaseSettings):
    """Settings loaded from the environment, then ``.env``, then these defaults."""

    model_config = SettingsConfigDict(
        env_prefix="JOBFIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- which model backend the agent loop talks to ----------------------
    transport: TransportName = "ollama"

    # --- ollama: free, local, the development default ---------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"

    # --- anthropic: paid. Empty key simply leaves it unavailable ----------
    anthropic_api_key: str | None = None
    # Exact model id, no date suffix.
    anthropic_model: str = "claude-sonnet-5"

    # --- autonomy limits --------------------------------------------------
    # Ceilings, not suggestions. Hitting one is a recorded outcome with a
    # reason, never a silent stop.
    max_tool_steps: int = Field(default=8, ge=1, le=50)
    max_graph_steps: int = Field(default=12, ge=1, le=100)

    # --- context budget ---------------------------------------------------
    # A real Remotive posting measured 15,796 characters. Nothing reaches the
    # model unclipped: it wastes tokens and overflows the KV cache on an 8 GB GPU.
    max_posting_chars: int = Field(default=6000, ge=500)

    # --- error handling ---------------------------------------------------
    # When true, a programming bug in the LLM path propagates instead of being
    # swallowed as "the model was unavailable". See errors.py for why.
    strict: bool = True

    # --- job board cache --------------------------------------------------
    # Remotive's API notice asks for at most ~4 requests per day. The cache is
    # how we comply, so it is on by default.
    cache_dir: Path = Path(".cache/job_boards")
    cache_ttl_seconds: int = Field(default=21_600, ge=0)

    @field_validator("anthropic_api_key", "ollama_base_url", "anthropic_model", "ollama_model")
    @classmethod
    def _blank_is_unset(cls, v: str | None) -> str | None:
        """Treat an empty or whitespace-only value as absent.

        ``JOBFIT_ANTHROPIC_API_KEY=`` in a committed ``.env.example`` should read
        as "no key", not as a key that happens to be the empty string.
        """
        if v is None:
            return None
        v = v.strip()
        return v or None

    @property
    def anthropic_available(self) -> bool:
        """True when an Anthropic call could actually be made."""
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached, so tests that mutate the environment must call
    ``get_settings.cache_clear()``.
    """
    return Settings()
