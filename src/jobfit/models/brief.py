"""The role brief: the reviewable draft a human acts on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jobfit.models.trace import OutputSource

#: Carried on every brief. The system drafts and assesses; a human decides and
#: submits. This is the product's position, not a temporary limitation.
DISCLAIMER = "Draft for human review. This system does not submit applications."


class RoleBrief(BaseModel):
    """What the writer produces for one posting."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str
    role_summary: str
    #: Why the candidate fits - each item must trace back to a profile skill.
    why_fit: list[str] = Field(default_factory=list)
    #: Real gaps, for studying. Not for papering over.
    gaps: list[str] = Field(default_factory=list)
    #: Inherited from CandidateProfile.never_claim, narrowed to this posting.
    #: The point of the whole project, encoded as a required field: a system
    #: built to end claims-without-evidence tells you what you cannot claim.
    not_claimable: list[str] = Field(default_factory=list)
    questions_for_recruiter: list[str] = Field(default_factory=list)
    prep_topics: list[str] = Field(default_factory=list)
    attribution: str | None = None
    source: OutputSource = OutputSource.LLM
    disclaimer: Literal[DISCLAIMER] = DISCLAIMER
