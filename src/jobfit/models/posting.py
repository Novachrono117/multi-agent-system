"""A job posting, normalised across every source we support."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PostingSource(StrEnum):
    """Where a posting came from.

    Only sources with a public, documented API *whose response shape was
    actually measured* are listed.

    Indeed and LinkedIn are absent because their terms forbid this, scraping
    them breaks on every markup change, and a public repository that does it is
    visible to exactly the people it is meant to impress.

    Lever is absent for a different and more interesting reason: its API works
    (``api.lever.co/v0/postings/{board}?mode=json`` returns 200), but no public
    board with live postings could be found to verify the payload against - the
    slugs tried returned 404 or an empty array. Writing an adapter against a
    schema nobody has seen is precisely the unverified code this project exists
    to stop shipping, so it is a documented gap rather than a guess.
    """

    GREENHOUSE = "greenhouse"
    REMOTIVE = "remotive"
    ARBEITNOW = "arbeitnow"
    PASTED = "pasted"


#: Sources that require attribution with a link back to the original listing.
#: Remotive's own API notice states this explicitly and asks that its jobs not
#: be redistributed; honouring it is a requirement, not a courtesy.
ATTRIBUTION_REQUIRED: frozenset[PostingSource] = frozenset(
    {
        PostingSource.REMOTIVE,
        PostingSource.ARBEITNOW,
        PostingSource.GREENHOUSE,
    }
)


class JobPosting(BaseModel):
    """One posting. ``body_text`` is the cleaned, clipped text agents may read."""

    model_config = ConfigDict(extra="forbid")

    source: PostingSource
    external_id: str
    title: str
    company: str | None = None
    # Provenance. Required for every API-sourced posting so attribution is
    # structurally impossible to forget.
    url: str | None = None
    location: str | None = None
    remote: bool | None = None
    job_type: str | None = None
    salary_text: str | None = None
    tags: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime | None = None

    #: Body exactly as the API returned it - may be HTML, may be entity-escaped.
    raw_body: str | None = None
    #: Body after unescaping, tag-stripping and clipping. This is what agents see.
    body_text: str | None = None
    #: True when ``body_text`` was truncated to fit the context budget.
    body_truncated: bool = False

    @property
    def posting_id(self) -> str:
        """Stable local identifier, unique across sources."""
        return f"{self.source.value}:{self.external_id}"

    @property
    def dedup_key(self) -> tuple[str, str]:
        return (self.source.value, self.external_id)

    @model_validator(mode="after")
    def _api_sources_carry_provenance(self) -> JobPosting:
        if self.source in ATTRIBUTION_REQUIRED and not self.url:
            raise ValueError(
                f"a {self.source.value} posting must carry the url it came from, "
                "so the listing can be attributed and linked back to"
            )
        return self

    def attribution(self) -> str | None:
        """Human-readable credit line, or None when none is owed."""
        if self.source not in ATTRIBUTION_REQUIRED:
            return None
        return f"Source: {self.source.value} - {self.url}"


def dedup_postings(
    left: list[JobPosting] | None, right: list[JobPosting] | None
) -> list[JobPosting]:
    """LangGraph reducer: append postings, dropping duplicates, order preserved.

    Not ``operator.add``: the search path genuinely returns the same role from
    more than one board, and a duplicate would be assessed twice and billed twice.
    A later posting wins, because it is the one that may carry a fetched body.
    """
    merged: dict[tuple[str, str], JobPosting] = {}
    for posting in [*(left or []), *(right or [])]:
        merged[posting.dedup_key] = posting
    return list(merged.values())
