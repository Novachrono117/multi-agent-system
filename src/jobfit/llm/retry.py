"""Retry with exponential backoff, as a transport wrapper.

This closes a real gap. None of the author's other projects have any retry or
backoff at all - resilience there is handled by falling back to a heuristic,
which papers over a transient 429 rather than surviving it.

Written as a wrapper rather than baked into each transport, so it composes:
any ``MessageTransport`` can be wrapped, including the fake, which is how the
behaviour is tested without waiting for a real rate limit.

The retry decision is not made here. The transport sets ``retryable`` on the
error, because it is the only layer that knows what the provider meant - a 429
is worth another attempt, a 400 never is. Retrying a malformed request just
spends money to receive the same rejection.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from jobfit.errors import TransportError
from jobfit.llm.transport import MessageLike, MessageTransport
from jobfit.observability import emit

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0


class RetryingTransport:
    """Wraps a transport, retrying retryable failures with backoff."""

    def __init__(
        self,
        inner: MessageTransport,
        *,
        attempts: int = DEFAULT_ATTEMPTS,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        sleeper: Callable[[float], None] | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        # Injected so tests never actually wait, and so jitter is deterministic
        # when it needs to be.
        self._sleep = sleeper if sleeper is not None else _default_sleep
        self._jitter = jitter if jitter is not None else random.random
        self.delays: list[float] = []

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "unknown")

    @property
    def synthetic(self) -> bool:
        """Pass through, so wrapping a transport cannot change how it is labelled."""
        return bool(getattr(self.inner, "synthetic", False))

    def _delay_for(self, attempt: int) -> float:
        """Exponential backoff, capped, with jitter.

        Jitter matters when several runs are rate limited at once: without it
        they all retry on the same beat and rate limit each other again.
        """
        exponential = self.base_delay * (2 ** (attempt - 1))
        return min(exponential, self.max_delay) * (0.5 + 0.5 * self._jitter())

    def send(self, **kwargs: Any) -> MessageLike:
        last: TransportError | None = None

        for attempt in range(1, self.attempts + 1):
            try:
                return self.inner.send(**kwargs)
            except TransportError as exc:
                if not exc.retryable:
                    # Nothing to gain. Fail now rather than three times.
                    emit("transport_failed", transport=self.name, retryable=False, error=str(exc))
                    raise
                last = exc
                if attempt == self.attempts:
                    break
                delay = self._delay_for(attempt)
                self.delays.append(delay)
                emit(
                    "transport_retry",
                    transport=self.name,
                    attempt=attempt,
                    of=self.attempts,
                    delay_s=round(delay, 3),
                    error=str(exc)[:200],
                )
                self._sleep(delay)

        assert last is not None  # only reachable after a retryable failure
        emit("transport_exhausted", transport=self.name, attempts=self.attempts, error=str(last))
        raise TransportError(
            f"{self.name} still failing after {self.attempts} attempts: {last}",
            retryable=True,
        ) from last


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def with_retry(transport: MessageTransport, **options: Any) -> MessageTransport:
    """Wrap ``transport`` with retry, unless it is already wrapped."""
    if isinstance(transport, RetryingTransport):
        return transport
    return RetryingTransport(transport, **options)
