"""Error taxonomy: an expected runtime failure is not the same thing as a bug.

This distinction is ported from a scar in the author's other codebase. In
``oralito/backend/app/analysis/llm/feedback.py`` a bare ``except Exception``
around the LLM path swallowed a ``KeyError`` coming from a prompt template. The
system kept answering - with the deterministic fallback, labelled as if the
model had produced it - and the LLM feedback stayed silently off for months.

So we separate two categories:

* **Runtime errors** - the network was down, we were rate limited, the model
  returned malformed JSON, a job board 500'd. These are expected in production.
  Degrade, label the degradation, carry on.
* **Programming bugs** - ``TypeError``, ``KeyError``, ``AttributeError``. These
  mean *our* code is wrong. Under ``JOBFIT_STRICT`` they propagate and are meant
  to be loud, because a swallowed bug is a bug that never gets fixed.
"""

from __future__ import annotations

import json


class JobfitError(Exception):
    """Base class for every error this package raises on purpose."""


class TransportError(JobfitError):
    """Talking to the model backend failed in an expected way.

    ``retryable`` says whether trying again could plausibly help. It is set by
    the transport, which is the only layer that knows what the provider meant:
    a 429 or a dropped connection is worth another attempt, a 400 never is.
    Retrying a malformed request just burns money and time.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class TransportUnavailableError(TransportError):
    """The requested transport cannot run - e.g. Anthropic selected with no key."""


class ToolExecutionError(JobfitError):
    """A tool failed in a way the model should be told about.

    Raising this is the supported way for a tool to fail: the agent loop turns it
    into a ``tool_result`` with ``is_error: True`` and lets the model react,
    rather than aborting the run.
    """


class DisallowedHostError(ToolExecutionError):
    """A network tool was asked to fetch a host outside the allowlist.

    Raised *before* any request is made. This is what makes the project's
    "no scraping of job boards that forbid it" promise a testable property
    instead of a sentence in the README.
    """


class StepCapExceededError(JobfitError):
    """An autonomy ceiling was reached. Carries the cap that was hit."""

    def __init__(self, cap: int, kind: str = "tool") -> None:
        super().__init__(f"{kind} step cap of {cap} reached")
        self.cap = cap
        self.kind = kind


# Expected at runtime: degrade and label, do not crash.
RUNTIME_ERRORS: tuple[type[BaseException], ...] = (
    JobfitError,
    json.JSONDecodeError,
    TimeoutError,
    ConnectionError,
    OSError,  # httpx errors and socket failures land here
)

# Our own fault. Loud by default.
PROGRAMMING_ERRORS: tuple[type[BaseException], ...] = (
    TypeError,
    KeyError,
    AttributeError,
    NameError,
    IndexError,
    NotImplementedError,
    AssertionError,
)


def is_programming_bug(exc: BaseException) -> bool:
    """True when ``exc`` indicates our code is wrong rather than the world.

    ``PROGRAMMING_ERRORS`` is checked first: a subclass could in principle appear
    in both tuples, and in that case we want the louder answer.
    """
    if isinstance(exc, PROGRAMMING_ERRORS):
        return True
    return not isinstance(exc, RUNTIME_ERRORS)


def reraise_if_bug(exc: BaseException, *, strict: bool) -> None:
    """Re-raise ``exc`` when it is a programming bug and we are in strict mode.

    Call this in every ``except Exception`` block on the model path. In strict
    mode a bug propagates; otherwise it is left to the caller to degrade, which
    is the production-friendly behaviour.
    """
    if strict and is_programming_bug(exc):
        raise exc
