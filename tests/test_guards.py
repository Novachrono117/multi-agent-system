"""Meta-tests: prove the test harness's own guarantees actually hold.

Without these, "the suite runs with no key and no network" is a claim rather
than a fact - and this project exists to stop making claims like that.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.conftest import NetworkAccessInTestError


def test_no_anthropic_credentials_visible() -> None:
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert os.environ.get("ANTHROPIC_AUTH_TOKEN") is None


def test_no_jobfit_env_leaks_from_the_developer_machine() -> None:
    assert [k for k in os.environ if k.startswith("JOBFIT_")] == []


def test_real_http_request_is_blocked() -> None:
    with httpx.Client() as client, pytest.raises(NetworkAccessInTestError):
        client.get("https://remotive.com/api/remote-jobs")


def test_the_block_names_the_url_so_failures_are_diagnosable() -> None:
    with httpx.Client() as client:
        try:
            client.get("https://example.invalid/x")
        except NetworkAccessInTestError as exc:
            assert "example.invalid" in str(exc)
        else:
            pytest.fail("the network guard did not fire")
