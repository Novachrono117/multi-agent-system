"""Model access: the transport seam, the hand-written agent loop, retry, and
structured output.

``get_transport`` is the single place a backend is chosen, which is what makes
running free (Ollama), paid (Anthropic) or offline (fake) a one-variable
decision rather than a code change.
"""

from __future__ import annotations

from jobfit.config import Settings
from jobfit.errors import TransportUnavailableError
from jobfit.llm.fake import FakeTransport
from jobfit.llm.loop import TurnResult, run_agent_turn
from jobfit.llm.transport import (
    AnthropicTransport,
    MessageLike,
    MessageTransport,
    OllamaTransport,
)


def get_transport(settings: Settings) -> MessageTransport:
    """Build the transport named by ``JOBFIT_TRANSPORT``.

    The Anthropic branch fails loudly and helpfully when no key is configured:
    silently falling back to a local model would mean a run labelled "anthropic"
    in the trace that never touched Anthropic - the exact kind of unverified
    claim this project exists to eliminate.
    """
    if settings.transport == "fake":
        return FakeTransport(repeat_last=True)
    if settings.transport == "ollama":
        return OllamaTransport(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    if settings.transport == "anthropic":
        return AnthropicTransport(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    raise TransportUnavailableError(f"unknown transport {settings.transport!r}")


__all__ = [
    "AnthropicTransport",
    "FakeTransport",
    "MessageLike",
    "MessageTransport",
    "OllamaTransport",
    "TurnResult",
    "get_transport",
    "run_agent_turn",
]
