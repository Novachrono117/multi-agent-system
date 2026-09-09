"""Tool layer: typed, schema-validated tools the agent may call.

``build_registry`` is the single place a run's toolset is assembled, which is
what keeps milestone M7 (exposing these same tools over MCP) a small adapter
rather than a refactor.
"""

from __future__ import annotations

from collections.abc import Iterable

import httpx

from jobfit.config import Settings
from jobfit.models.profile import CandidateProfile
from jobfit.tools import parse_job_posting, score_profile_match, search_job_boards
from jobfit.tools.registry import ToolContext, ToolRegistry, ToolSpec

#: Every tool this package ships. Order is stable so tool schemas - and
#: therefore the prompt prefix and its cache - are byte-identical between runs.
#:
#: Ordered the way an agent uses them: find, fetch, clean, then measure.
ALL_SPECS: tuple[ToolSpec, ...] = (
    search_job_boards.SEARCH_SPEC,
    search_job_boards.FETCH_SPEC,
    parse_job_posting.SPEC,
    score_profile_match.SPEC,
)

LOCAL_ONLY: tuple[str, ...] = tuple(spec.name for spec in ALL_SPECS if not spec.touches_network)


def build_context(
    settings: Settings,
    profile: CandidateProfile,
    *,
    http: httpx.Client | None = None,
) -> ToolContext:
    return ToolContext(settings=settings, profile=profile, http=http)


def build_registry(*, allowed: Iterable[str] | None = None) -> ToolRegistry:
    """Assemble the registry for a run, optionally narrowed by an allowlist."""
    return ToolRegistry(ALL_SPECS, allowed=allowed)


__all__ = [
    "ALL_SPECS",
    "LOCAL_ONLY",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "build_context",
    "build_registry",
]
