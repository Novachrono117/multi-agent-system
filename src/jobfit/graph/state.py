"""The typed graph state.

One rule governs everything here, and breaking it would be expensive later:
**the state holds only our own Pydantic models and primitives.** No
``anthropic.Message``, no SDK content blocks. The raw message list lives inside
``run_agent_turn`` and dies there.

The reason is milestone M4 (persistent memory). LangGraph's checkpointers
serialise the state; an SDK object in it does not serialise, and memory would
then cost a rewrite rather than a new module.

Reducers are chosen per field rather than defaulting to ``operator.add``:
postings genuinely arrive twice from different boards, and appending a duplicate
would mean assessing - and paying for - the same role twice.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from jobfit.models.assessment import FitAssessment
from jobfit.models.brief import RoleBrief
from jobfit.models.posting import JobPosting, dedup_postings
from jobfit.models.profile import CandidateProfile
from jobfit.models.trace import (
    Handoff,
    OutputSource,
    RunOutcome,
    StepError,
    StepTrace,
    TokenUsage,
    sum_usage,
)

#: Where control can go next. ``finalize`` is always reachable, which is what
#: guarantees the graph terminates.
NextAgent = Literal["fit_screener", "brief_writer", "finalize"]


class IntakeRequest(BaseModel):
    """What the human asked for. Exactly one mode is populated."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str | None = None
    search_query: str | None = None
    board_tokens: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    limit: int = 3

    @property
    def mode(self) -> str:
        if self.raw_text:
            return "pasted"
        if self.search_query is not None:
            return "search"
        return "empty"


class SupervisorDecision(BaseModel):
    """The supervisor's structured output.

    ``reason`` is required. A routing decision without a stated reason is what
    makes a supervisor look decorative, and it is the field that turns "multiple
    agents collaborating" into something a reader can check.
    """

    model_config = ConfigDict(extra="forbid")

    next_agent: NextAgent
    target_posting_id: str | None = None
    reason: str


class PipelineState(BaseModel):
    """The whole run, in one serialisable object."""

    # extra="forbid" catches a node returning a key that does not exist -
    # otherwise a typo in a node's return dict is silently dropped.
    model_config = ConfigDict(extra="forbid")

    # --- input -----------------------------------------------------------
    run_id: str
    profile: CandidateProfile
    request: IntakeRequest

    # --- work ------------------------------------------------------------
    postings: Annotated[list[JobPosting], dedup_postings] = Field(default_factory=list)
    assessments: Annotated[list[FitAssessment], operator.add] = Field(default_factory=list)
    briefs: Annotated[list[RoleBrief], operator.add] = Field(default_factory=list)

    # --- supervisor control ----------------------------------------------
    next_agent: NextAgent | None = None
    target_posting_id: str | None = None
    #: Append-only record of who handed off to whom, and why.
    handoffs: Annotated[list[Handoff], operator.add] = Field(default_factory=list)
    step_count: int = 0
    #: Postings sent back to the screener for another pass, so the loop is
    #: bounded: one retry each, never more.
    rescreened: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- observability ---------------------------------------------------
    trace: Annotated[list[StepTrace], operator.add] = Field(default_factory=list)
    usage_total: Annotated[TokenUsage, sum_usage] = Field(default_factory=TokenUsage)
    errors: Annotated[list[StepError], operator.add] = Field(default_factory=list)
    outcome: RunOutcome | None = None

    # --- helpers ---------------------------------------------------------

    def posting(self, posting_id: str) -> JobPosting | None:
        return next((p for p in self.postings if p.posting_id == posting_id), None)

    def assessment_for(self, posting_id: str) -> FitAssessment | None:
        return next((a for a in self.assessments if a.posting_id == posting_id), None)

    def brief_for(self, posting_id: str) -> RoleBrief | None:
        return next((b for b in self.briefs if b.posting_id == posting_id), None)

    def unassessed(self) -> list[JobPosting]:
        assessed = {a.posting_id for a in self.assessments}
        return [p for p in self.postings if p.posting_id not in assessed]

    def awaiting_brief(self) -> list[FitAssessment]:
        """Assessments that earned a brief and have not been given one.

        This is where the supervisor's routing has real work to do: with several
        postings in play, deciding which ones reach the expensive writer is a
        judgement, not a formality.
        """
        written = {b.posting_id for b in self.briefs}
        return [
            a
            for a in self.assessments
            if a.worth_a_brief and a.posting_id not in written and not a.needs_more_evidence
        ]

    def needs_rescreen(self) -> list[FitAssessment]:
        """Ambiguous assessments that deserve exactly one more pass."""
        return [
            a
            for a in self.assessments
            if a.needs_more_evidence and a.posting_id not in self.rescreened
        ]

    def next_seq(self) -> int:
        return len(self.trace)

    def deterministic_route(self) -> SupervisorDecision:
        """Routing without a model, used as the fallback and as the safety net.

        The supervisor asks a model to choose, because that is the point of
        having one. But a small local model sometimes returns prose instead of
        JSON, and a pipeline that stalls on that would be useless. So the same
        decision is computable here, and ``OutputSource`` records which one was
        actually used - the degradation is labelled, never hidden.
        """
        pending = self.unassessed()
        if pending:
            return SupervisorDecision(
                next_agent="fit_screener",
                target_posting_id=pending[0].posting_id,
                reason=f"{len(pending)} posting(s) still unassessed",
            )

        rescreen = self.needs_rescreen()
        if rescreen:
            return SupervisorDecision(
                next_agent="fit_screener",
                target_posting_id=rescreen[0].posting_id,
                reason="assessment was inconclusive; one more evidence pass",
            )

        awaiting = self.awaiting_brief()
        if awaiting:
            best = max(
                awaiting,
                key=lambda a: a.score.coverage if a.score else 0.0,
            )
            return SupervisorDecision(
                next_agent="brief_writer",
                target_posting_id=best.posting_id,
                reason="best remaining fit; earns the expensive writer step",
            )

        return SupervisorDecision(
            next_agent="finalize",
            target_posting_id=None,
            reason="every posting has been assessed and briefed where warranted",
        )

    def route_source(self) -> OutputSource:
        """Whether the last handoff came from the model or from the fallback."""
        return OutputSource.LLM if self.handoffs else OutputSource.DETERMINISTIC
