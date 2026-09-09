"""The candidate profile the pipeline assesses postings against.

Loaded from TOML, never from a model. The profile is ground truth: if it is not
in here, the system must not claim it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CandidateProfile(BaseModel):
    """What the candidate has, wants, and must not be made to claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: str
    location: str
    seniority: str
    years_experience: float = Field(ge=0)

    #: Verified skills. The only pool an assessment may draw evidence from.
    skills: list[str] = Field(min_length=1)

    #: Requirements whose absence rules the candidate out of a posting -
    #: e.g. "onsite outside Brasilia", "5+ years required".
    deal_breakers: list[str] = Field(default_factory=list)

    #: Tracks deliberately not pursued. A preference, not a gap: they may be on
    #: the CV and still not be wanted.
    avoid_tracks: list[str] = Field(default_factory=list)

    #: The heart of this project. Things the candidate does NOT have, which no
    #: output may ever assert. Propagates into RoleBrief.not_claimable so the
    #: brief tells you what to avoid saying out loud in an interview.
    never_claim: list[str] = Field(default_factory=list)

    languages: list[str] = Field(default_factory=list)

    def skill_set(self) -> frozenset[str]:
        """Case-folded skills, for matching."""
        return frozenset(s.strip().casefold() for s in self.skills if s.strip())

    def never_claim_set(self) -> frozenset[str]:
        return frozenset(s.strip().casefold() for s in self.never_claim if s.strip())


def load_profile(path: Path) -> CandidateProfile:
    """Read a profile from TOML.

    Raises ``FileNotFoundError`` if absent and ``ValidationError`` if malformed -
    both loudly, because a silently empty profile would make every assessment
    meaningless while still looking like it worked.
    """
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return CandidateProfile.model_validate(data.get("profile", data))
