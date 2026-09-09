"""``score_profile_match`` - the number comes from Python, not from the model.

This is the most load-bearing design decision in the package. A language model
is good at explaining why a candidate fits and bad at being a consistent
measuring instrument: ask it twice and you get two coverages. So the score is
computed here, deterministically, from the profile and the requirement list -
and ``OutputSource`` on the surrounding trace records which half of the output
was measured and which half was written.

The consequence to be honest about: this is keyword and token overlap, not
comprehension. It will miss a requirement phrased in words the profile does not
use. That is a real limitation, it belongs in the README, and it is still
preferable to a fluent paragraph that reads like a measurement and is not one.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from jobfit.models.assessment import MatchScore
from jobfit.tools.registry import ToolContext, ToolSpec
from jobfit.tools.text import extract_bullets

NAME = "score_profile_match"
DESCRIPTION = (
    "Compute a deterministic coverage score for a parsed posting against the "
    "candidate profile. Returns coverage between 0 and 1, which profile skills "
    "matched, which requirements went unmatched, and any deal breakers the "
    "posting triggers. This is a measurement, not an opinion - use it as "
    "evidence and do not restate it as your own judgement."
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*", re.IGNORECASE)
#: Tokens too common to carry signal when matching a skill against a sentence.
_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "with",
        "for",
        "to",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "be",
        "you",
        "we",
        "our",
        "your",
        "will",
        "have",
        "has",
        "e",
        "ou",
        "o",
        "os",
        "de",
        "da",
        "do",
        "em",
        "com",
        "para",
        "por",
        "um",
        "uma",
        "que",
        "se",
        "no",
        "na",
        "ser",
        "ter",
        "voce",
        "nossa",
        "nosso",
        "anos",
        "experience",
        "experiencia",
        "knowledge",
        "years",
        "strong",
        "good",
        "plus",
    }
)


class ScoreProfileMatchInput(BaseModel):
    """Arguments for ``score_profile_match``."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str = Field(description="Id of a posting that has already been parsed in this run.")


class ScoreResult(BaseModel):
    """Deterministic match result plus the caveat that belongs with it."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str
    coverage: float
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    deal_breakers_hit: list[str] = Field(default_factory=list)
    blocked: bool = False
    requirements_considered: int = 0
    method: str = (
        "Deterministic token overlap between the requirement list and the "
        "profile's declared skills. It cannot recognise a requirement phrased "
        "in words the profile does not use."
    )


def _tokens(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _WORD_RE.finditer(text)
        if match.group(0).casefold() not in _STOPWORDS and len(match.group(0)) > 1
    }


def _requirement_is_met(requirement: str, skills: set[str], skill_tokens: set[str]) -> str | None:
    """Return the skill that satisfies ``requirement``, or None.

    A full skill phrase present in the requirement wins outright ("GitHub
    Actions"); otherwise a single distinctive token is enough ("pytest").
    """
    lowered = requirement.casefold()
    for skill in sorted(skills, key=len, reverse=True):
        if skill in lowered:
            return skill
    overlap = _tokens(requirement) & skill_tokens
    return sorted(overlap)[0] if overlap else None


def compute_score(
    requirements: list[str],
    skills: frozenset[str],
    deal_breakers: list[str],
    haystack: str,
) -> MatchScore:
    """Pure scoring function - no context, no I/O, trivially testable."""
    skill_set = set(skills)
    skill_tokens = {token for skill in skill_set for token in _tokens(skill)}

    matched: list[str] = []
    missing: list[str] = []
    for requirement in requirements:
        hit = _requirement_is_met(requirement, skill_set, skill_tokens)
        if hit:
            matched.append(requirement)
        else:
            missing.append(requirement)

    lowered_haystack = haystack.casefold()
    hits = [
        breaker
        for breaker in deal_breakers
        if _tokens(breaker) and _tokens(breaker) <= _tokens(lowered_haystack)
    ]

    coverage = len(matched) / len(requirements) if requirements else 0.0
    return MatchScore(
        coverage=round(coverage, 4),
        matched=matched,
        missing=missing,
        deal_breakers_hit=hits,
    )


def handle(ctx: ToolContext, args: ScoreProfileMatchInput) -> ScoreResult:
    posting = ctx.require_posting(args.posting_id)
    body = posting.body_text or ""
    requirements = extract_bullets(body)
    haystack = f"{posting.title} {posting.location or ''} {body}"

    score = compute_score(
        requirements,
        ctx.profile.skill_set(),
        ctx.profile.deal_breakers,
        haystack,
    )
    return ScoreResult(
        posting_id=posting.posting_id,
        coverage=score.coverage,
        matched=score.matched,
        missing=score.missing,
        deal_breakers_hit=score.deal_breakers_hit,
        blocked=score.blocked,
        requirements_considered=len(requirements),
    )


SPEC = ToolSpec(
    name=NAME,
    description=DESCRIPTION,
    input_model=ScoreProfileMatchInput,
    handler=handle,
    touches_network=False,
)
