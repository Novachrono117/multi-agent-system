"""Test-wide guarantees.

Two of them, and they are the reason the suite can be trusted:

1. **No credentials.** Every ``JOBFIT_*`` variable and the ambient
   ``ANTHROPIC_API_KEY`` are removed before any test runs, and the settings
   cache is cleared. A developer with a funded key gets the same result as CI.
2. **No network.** ``httpx`` is severed at ``Client.send``, the single point
   every sync request passes through. A test that reaches for the network fails
   by construction rather than by luck - which also means it cannot quietly
   start costing money.

Opt out for a genuinely live check with ``@pytest.mark.live``; those are
excluded from CI.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from jobfit.config import get_settings

_ENV_VARS_TO_CLEAR = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip credentials and JOBFIT_* overrides, and reset the settings cache."""
    for name in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    import os

    for name in [k for k in os.environ if k.startswith("JOBFIT_")]:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class NetworkAccessInTestError(RuntimeError):
    """Raised when a test tries to open a real connection."""


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every real HTTP request fail loudly.

    Tests exercising the network tools inject their own fake transport; they
    never need this lifted.
    """
    if request.node.get_closest_marker("live"):
        return

    def _blocked(self: httpx.Client, request_: httpx.Request, **kwargs: object) -> None:
        raise NetworkAccessInTestError(
            f"test attempted a real request to {request_.url!r}. "
            "Inject a fake transport, or mark the test @pytest.mark.live."
        )

    monkeypatch.setattr(httpx.Client, "send", _blocked, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked, raising=True)


@pytest.fixture
def events() -> Iterator[list[str]]:
    """Collect emitted observability events instead of writing to stderr."""
    from jobfit.observability import set_sink

    collected: list[str] = []
    set_sink(collected.append)
    yield collected
    set_sink(None)
