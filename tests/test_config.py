"""Settings must work with no environment at all - that is the hard requirement."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobfit.config import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    """Build settings ignoring any .env on disk, so tests are hermetic."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_loads_with_no_environment_and_no_api_key() -> None:
    s = _settings()
    assert s.transport == "ollama"
    assert s.anthropic_api_key is None
    assert s.anthropic_available is False


def test_anthropic_model_id_has_no_date_suffix() -> None:
    """Date-suffixed ids are a stale-training-data trap and 404 at the API."""
    s = _settings()
    assert s.anthropic_model == "claude-sonnet-5"
    assert not s.anthropic_model[-1].isdigit() or "-20" not in s.anthropic_model


def test_blank_api_key_reads_as_absent() -> None:
    """`JOBFIT_ANTHROPIC_API_KEY=` in .env.example must not look like a key."""
    s = _settings(anthropic_api_key="   ")
    assert s.anthropic_api_key is None
    assert s.anthropic_available is False


def test_key_present_makes_anthropic_available() -> None:
    s = _settings(anthropic_api_key="sk-ant-test")
    assert s.anthropic_available is True


def test_env_prefix_is_jobfit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBFIT_TRANSPORT", "fake")
    monkeypatch.setenv("JOBFIT_MAX_TOOL_STEPS", "3")
    s = _settings()
    assert s.transport == "fake"
    assert s.max_tool_steps == 3


def test_ambient_anthropic_api_key_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """We read JOBFIT_ANTHROPIC_API_KEY only, so a stray ambient key cannot bill us."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient-should-be-ignored")
    assert _settings().anthropic_available is False


@pytest.mark.parametrize("field,bad", [("max_tool_steps", 0), ("max_graph_steps", 0)])
def test_autonomy_caps_reject_nonsense(field: str, bad: int) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field: bad})


def test_unknown_transport_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(transport="gpt5-via-carrier-pigeon")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
