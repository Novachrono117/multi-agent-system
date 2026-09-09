"""Fit assessment: the structured verdict the screener produces."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jobfit.models.trace import OutputSource


class RequirementVerdict(StrEnum):
    MET = "met"
    PARTIAL = "partial"
    MISSING = "missing"


class RequirementCheck(BaseModel):
    """One requirement from the posting, judged against the profile."""

    model_config = ConfigDict(extra="forbid")

    requirement: str
    verdict: RequirementVerdict
    #: Which profile skill backs this up. Empty for MISSING - and an assessment
    #: claiming MET with no evidence is a bug we can test for.
    evidence: str | None = None


class FitVerdict(StrEnum):
    STRONG_FIT = "strong_fit"
    WORTH_APPLYING = "worth_applying"
    STRETCH = "stretch"
    NO_FIT = "no_fit"


class MatchScore(BaseModel):
    """Deterministic coverage, computed in Python with no model involved.

    The number comes from code; the prose comes from the model. Keeping those
    separable - and labelled via ``OutputSource`` - is what stops a fluent
    paragraph from being mistaken for a measurement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage: float = Field(ge=0.0, le=1.0)
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    deal_breakers_hit: list[str] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.deal_breakers_hit)


class FitAssessment(BaseModel):
    """The screener's output for one posting."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str
    verdict: FitVerdict
    checks: list[RequirementCheck] = Field(default_factory=list)
    #: Populated from MatchScore, not from the model.
    score: MatchScore | None = None
    reasoning: str = ""
    #: True when the screener could not gather enough evidence and the
    #: supervisor should route back for another pass.
    needs_more_evidence: bool = False
    source: OutputSource = OutputSource.LLM

    @property
    def worth_a_brief(self) -> bool:
        """Whether this posting earns the expensive writer node."""
        return self.verdict in (FitVerdict.STRONG_FIT, FitVerdict.WORTH_APPLYING)
