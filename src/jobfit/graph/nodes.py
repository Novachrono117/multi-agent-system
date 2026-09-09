"""The graph's nodes: intake, supervisor, the two specialists, finalize.

Dependencies (transport, tool registry, tool context) arrive through a
``GraphDeps`` closure rather than through the state, because they are neither
serialisable nor part of the run's data - see ``state.py`` for why that
distinction is load-bearing.

Every node emits ``StepTrace`` entries even though nothing consumes them yet.
That is deliberate: milestone M5 becomes "write the sink", one new module,
instead of a diff that touches every node here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jobfit import prompts
from jobfit.errors import JobfitError
from jobfit.graph.state import PipelineState, SupervisorDecision
from jobfit.llm.loop import run_agent_turn
from jobfit.llm.structured import parse_into
from jobfit.llm.transport import MessageTransport
from jobfit.models.assessment import FitAssessment, FitVerdict, MatchScore
from jobfit.models.brief import RoleBrief
from jobfit.models.posting import JobPosting, PostingSource
from jobfit.models.trace import (
    Handoff,
    OutputSource,
    RunOutcome,
    StepError,
    StepKind,
    StepTrace,
)
from jobfit.observability import emit
from jobfit.tools import build_registry
from jobfit.tools.registry import ToolContext, ToolRegistry
from jobfit.tools.score_profile_match import compute_score
from jobfit.tools.text import clip_for_model, extract_bullets, html_to_text

#: Tools each specialist may use. Narrowing per node is the allowlist doing real
#: work: the writer has no business touching the network, and cannot.
SCREENER_TOOLS = (
    "search_job_boards",
    "fetch_job_posting",
    "parse_job_posting",
    "score_profile_match",
)


@dataclass
class GraphDeps:
    """Everything the nodes need that is not run data."""

    transport: MessageTransport
    ctx: ToolContext
    screener_registry: ToolRegistry | None = None

    def registry(self) -> ToolRegistry:
        if self.screener_registry is None:
            self.screener_registry = build_registry(allowed=SCREENER_TOOLS)
        return self.screener_registry


def _trace(
    state: PipelineState,
    node: str,
    kind: StepKind,
    *,
    name: str | None = None,
    started: datetime | None = None,
    source: OutputSource = OutputSource.LLM,
    ok: bool = True,
    error: str | None = None,
) -> StepTrace:
    started = started or datetime.now(UTC)
    return StepTrace(
        run_id=state.run_id,
        seq=state.next_seq(),
        node=node,
        kind=kind,
        name=name,
        started_at=started,
        latency_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
        ok=ok,
        error=error,
        source=source,
    )


# --------------------------------------------------------------------------
# intake
# --------------------------------------------------------------------------


def make_intake(deps: GraphDeps) -> Any:
    """Resolve the request into postings. No model involved.

    Pasted text is parsed here as a plain function call, not as a tool: there is
    no decision for a model to make about text the human already handed over,
    and spending a model turn on it would be theatre.
    """

    def intake(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(UTC)
        request = state.request
        postings: list[JobPosting] = []
        errors: list[StepError] = []

        if request.mode == "pasted" and request.raw_text:
            postings.append(
                JobPosting(
                    source=PostingSource.PASTED,
                    external_id="pasted-1",
                    title=_guess_title(request.raw_text),
                    raw_body=request.raw_text,
                )
            )
        elif request.mode == "search" and request.search_query is not None:
            outcome = deps.registry().dispatch(
                "search_job_boards",
                {
                    "query": request.search_query,
                    "sources": request.sources or ["remotive"],
                    "board_tokens": request.board_tokens or None,
                    "limit": request.limit,
                },
                deps.ctx,
            )
            if outcome.is_error:
                errors.append(
                    StepError(
                        node="intake",
                        kind="search_failed",
                        message=outcome.content[:400],
                        recoverable=False,
                    )
                )
            postings.extend(deps.ctx.postings.values())
        else:
            errors.append(
                StepError(
                    node="intake",
                    kind="empty_request",
                    message="no posting text and no search query were provided",
                    recoverable=False,
                )
            )

        for posting in postings:
            deps.ctx.add_posting(posting)

        emit("intake", run_id=state.run_id, mode=request.mode, postings=len(postings))
        return {
            "postings": postings,
            "errors": errors,
            "trace": [
                _trace(
                    state,
                    "intake",
                    StepKind.DECISION,
                    name=request.mode,
                    started=started,
                    source=OutputSource.DETERMINISTIC,
                    ok=not errors,
                )
            ],
        }

    return intake


def _guess_title(raw_text: str) -> str:
    """First non-empty line of pasted text, trimmed. Good enough, and honest."""
    for line in html_to_text(raw_text).split("\n"):
        cleaned = line.strip()
        if len(cleaned) > 3:
            return cleaned[:120]
    return "(pasted posting)"


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------


def make_supervisor(deps: GraphDeps) -> Any:
    """Ask the model where to go next, with a deterministic net underneath.

    The model is asked because choosing which postings deserve the expensive
    writer is a judgement. The net exists because a small local model sometimes
    answers in prose, and a pipeline that stalls on that is not a pipeline.
    Which one decided is recorded in the trace, never blurred.
    """

    def supervisor(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(UTC)
        fallback = state.deterministic_route()

        if state.step_count >= deps.ctx.settings.max_graph_steps:
            decision = SupervisorDecision(
                next_agent="finalize",
                target_posting_id=None,
                reason=f"graph step ceiling of {deps.ctx.settings.max_graph_steps} reached",
            )
            source = OutputSource.DETERMINISTIC
        elif fallback.next_agent == "finalize":
            # Nothing left to decide; do not spend a model call saying so.
            decision, source = fallback, OutputSource.DETERMINISTIC
        else:
            decision, source = _ask_supervisor(deps, state, fallback)

        emit(
            "handoff",
            run_id=state.run_id,
            to=decision.next_agent,
            posting=decision.target_posting_id,
            reason=decision.reason,
            source=source.value,
        )
        return {
            "next_agent": decision.next_agent,
            "target_posting_id": decision.target_posting_id,
            "step_count": state.step_count + 1,
            "handoffs": [
                Handoff(
                    seq=len(state.handoffs),
                    from_node="supervisor",
                    to_node=decision.next_agent,
                    reason=decision.reason,
                )
            ],
            "trace": [
                _trace(
                    state,
                    "supervisor",
                    StepKind.DECISION,
                    name=decision.next_agent,
                    started=started,
                    source=source,
                )
            ],
        }

    return supervisor


def _ask_supervisor(
    deps: GraphDeps, state: PipelineState, fallback: SupervisorDecision
) -> tuple[SupervisorDecision, OutputSource]:
    system = prompts.SUPERVISOR_SYSTEM.substitute(
        no_submit=prompts.NO_SUBMIT_CLAUSE,
        headline=state.profile.headline,
        seniority=state.profile.seniority,
        location=state.profile.location,
    )
    turn = prompts.SUPERVISOR_TURN.substitute(
        postings="\n".join(
            f"- {p.posting_id} {p.title}"
            + (" [assessed]" if state.assessment_for(p.posting_id) else "")
            for p in state.postings
        )
        or "(none)",
        assessments="\n".join(
            f"- {a.posting_id}: {a.verdict.value}"
            f"{' (needs more evidence)' if a.needs_more_evidence else ''}"
            for a in state.assessments
        )
        or "(none yet)",
        brief_written="yes" if state.briefs else "no",
        steps=str(state.step_count),
        max_steps=str(deps.ctx.settings.max_graph_steps),
    )

    decision, source, _ = parse_into(
        SupervisorDecision,
        transport=deps.transport,
        system=system,
        user_message=turn,
        max_tokens=512,
    )

    if decision is None:
        return fallback, OutputSource.DETERMINISTIC

    # A model may name a posting that does not exist, or one already handled.
    # Trust the routing, verify the target.
    if decision.next_agent != "finalize":
        known = {p.posting_id for p in state.postings}
        if decision.target_posting_id not in known:
            return (
                decision.model_copy(update={"target_posting_id": fallback.target_posting_id}),
                OutputSource.MIXED,
            )
    return decision, source


# --------------------------------------------------------------------------
# fit_screener
# --------------------------------------------------------------------------


def make_screener(deps: GraphDeps) -> Any:
    """Read one posting with tools, then produce a structured assessment.

    Two calls, not one clever one: the tool loop gathers evidence until
    ``end_turn``, and a second, tool-free call formats the result. Asking for
    structured output in the same call that may still be calling tools means the
    JSON may not be the final block.
    """

    def fit_screener(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(UTC)
        posting_id = state.target_posting_id
        posting = state.posting(posting_id) if posting_id else None

        if posting is None:
            return _screener_failed(
                state, started, posting_id, "the supervisor named an unknown posting"
            )

        registry = deps.registry()
        system = prompts.SCREENER_SYSTEM.substitute(
            **prompts.clause_kwargs(),
            profile=_profile_block(state),
        )
        turn = prompts.SCREENER_TURN.substitute(
            posting_id=posting.posting_id,
            title=posting.title,
            company=posting.company or "(not stated)",
            location=posting.location or "(not stated)",
        )

        result = run_agent_turn(
            transport=deps.transport,
            registry=registry,
            ctx=deps.ctx,
            system=system,
            user_message=turn,
            run_id=state.run_id,
            node="fit_screener",
            seq_start=state.next_seq(),
        )

        # The deterministic score is computed here, not taken from the model's
        # prose, whatever the model said about it.
        score = _score_posting(deps, posting.posting_id)
        assessment, source = _assess(deps, state, posting.posting_id, result.text, score)

        emit(
            "assessed",
            run_id=state.run_id,
            posting=posting.posting_id,
            verdict=assessment.verdict.value,
            coverage=score.coverage if score else None,
            source=source.value,
        )
        update: dict[str, Any] = {
            "assessments": [assessment],
            "trace": [
                *result.trace,
                _trace(
                    state,
                    "fit_screener",
                    StepKind.DECISION,
                    name=assessment.verdict.value,
                    started=started,
                    source=source,
                ),
            ],
            "usage_total": result.usage,
            "step_count": state.step_count + 1,
        }
        if result.hit_step_cap:
            update["errors"] = [
                StepError(
                    node="fit_screener",
                    kind="step_cap",
                    message=f"tool step ceiling reached on {posting.posting_id}",
                )
            ]
        # Mark a re-screen so the same posting cannot loop forever.
        if state.assessment_for(posting.posting_id) is not None:
            update["rescreened"] = [posting.posting_id]
        return update

    return fit_screener


def _profile_block(state: PipelineState) -> str:
    profile = state.profile
    lines = [
        f"headline: {profile.headline}",
        f"seniority: {profile.seniority}",
        f"location: {profile.location}",
        f"years_experience: {profile.years_experience}",
        f"skills: {', '.join(profile.skills)}",
        f"deal_breakers: {', '.join(profile.deal_breakers) or '(none)'}",
        f"avoid_tracks: {', '.join(profile.avoid_tracks) or '(none)'}",
        f"never_claim: {', '.join(profile.never_claim) or '(none)'}",
        f"languages: {', '.join(profile.languages) or '(not stated)'}",
    ]
    return "\n".join(lines)


def _score_posting(deps: GraphDeps, posting_id: str) -> MatchScore | None:
    posting = deps.ctx.postings.get(posting_id)
    if posting is None:
        return None
    body = posting.body_text or html_to_text(posting.raw_body or "")
    requirements = extract_bullets(body)
    if not requirements:
        return None
    haystack = f"{posting.title} {posting.location or ''} {body}"
    return compute_score(
        requirements,
        deps.ctx.profile.skill_set(),
        deps.ctx.profile.deal_breakers,
        haystack,
    )


def _assess(
    deps: GraphDeps,
    state: PipelineState,
    posting_id: str,
    findings: str,
    score: MatchScore | None,
) -> tuple[FitAssessment, OutputSource]:
    system = prompts.ASSESSMENT_SYSTEM.substitute(
        evidence=prompts.EVIDENCE_CLAUSE,
        score=prompts.SCORE_CLAUSE,
        profile=_profile_block(state),
    )
    turn = prompts.ASSESSMENT_TURN.substitute(
        posting_id=posting_id,
        findings=findings or "(the screener produced no summary)",
        score=_score_block(score),
    )

    assessment, source, _ = parse_into(
        FitAssessment,
        transport=deps.transport,
        system=system,
        user_message=turn,
        max_tokens=1500,
    )

    if assessment is None:
        # No usable structured output. Fall back to the measurement alone and
        # label it, rather than inventing a verdict.
        return (
            FitAssessment(
                posting_id=posting_id,
                verdict=_verdict_from_score(score),
                score=score,
                reasoning="The model produced no valid assessment; this verdict "
                "comes from the deterministic coverage score alone.",
                source=OutputSource.DETERMINISTIC,
            ),
            OutputSource.DETERMINISTIC,
        )

    # The model does not get to set these two: the id must match what was asked
    # for, and the score is a measurement.
    corrected = assessment.model_copy(
        update={
            "posting_id": posting_id,
            "score": score,
            "source": source,
            "verdict": FitVerdict.NO_FIT if (score and score.blocked) else assessment.verdict,
        }
    )
    return corrected, source


def _score_block(score: MatchScore | None) -> str:
    if score is None:
        return "(no requirements could be extracted, so no coverage was computed)"
    return (
        f"coverage: {score.coverage}\n"
        f"matched: {len(score.matched)}\n"
        f"missing: {'; '.join(score.missing[:8]) or '(none)'}\n"
        f"deal_breakers_hit: {'; '.join(score.deal_breakers_hit) or '(none)'}"
    )


def _verdict_from_score(score: MatchScore | None) -> FitVerdict:
    if score is None:
        return FitVerdict.STRETCH
    if score.blocked:
        return FitVerdict.NO_FIT
    if score.coverage >= 0.7:
        return FitVerdict.STRONG_FIT
    if score.coverage >= 0.4:
        return FitVerdict.WORTH_APPLYING
    return FitVerdict.STRETCH


def _screener_failed(
    state: PipelineState, started: datetime, posting_id: str | None, message: str
) -> dict[str, Any]:
    return {
        "assessments": [
            FitAssessment(
                posting_id=posting_id or "unknown",
                verdict=FitVerdict.NO_FIT,
                reasoning=message,
                source=OutputSource.DETERMINISTIC,
            )
        ],
        "errors": [StepError(node="fit_screener", kind="bad_target", message=message)],
        "step_count": state.step_count + 1,
        "trace": [
            _trace(
                state,
                "fit_screener",
                StepKind.DECISION,
                started=started,
                source=OutputSource.DETERMINISTIC,
                ok=False,
                error=message,
            )
        ],
    }


# --------------------------------------------------------------------------
# brief_writer
# --------------------------------------------------------------------------


def make_writer(deps: GraphDeps) -> Any:
    """Turn an assessment into the document a human reads. No network access."""

    def brief_writer(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(UTC)
        posting_id = state.target_posting_id
        posting = state.posting(posting_id) if posting_id else None
        assessment = state.assessment_for(posting_id) if posting_id else None

        if posting is None or assessment is None:
            message = "the writer was called without a posting and assessment"
            return {
                "errors": [StepError(node="brief_writer", kind="bad_target", message=message)],
                "step_count": state.step_count + 1,
                "trace": [
                    _trace(
                        state,
                        "brief_writer",
                        StepKind.DECISION,
                        started=started,
                        source=OutputSource.DETERMINISTIC,
                        ok=False,
                        error=message,
                    )
                ],
            }

        live = deps.ctx.postings.get(posting.posting_id, posting)
        body, _ = clip_for_model(
            live.body_text or html_to_text(live.raw_body or ""),
            deps.ctx.settings.max_posting_chars,
        )

        system = prompts.WRITER_SYSTEM.substitute(
            **prompts.clause_kwargs(),
            profile=_profile_block(state),
        )
        turn = prompts.WRITER_TURN.substitute(
            posting_id=posting.posting_id,
            title=posting.title,
            company=posting.company or "(not stated)",
            attribution=live.attribution() or "(none required)",
            assessment=assessment.model_dump_json(indent=2),
            posting_text=body or "(no posting text available)",
        )

        brief, source, _raw = parse_into(
            RoleBrief,
            transport=deps.transport,
            system=system,
            user_message=turn,
            max_tokens=2500,
        )

        if brief is None:
            message = "the model produced no valid role brief"
            emit("brief_failed", run_id=state.run_id, posting=posting.posting_id)
            return {
                "errors": [StepError(node="brief_writer", kind="no_brief", message=message)],
                "step_count": state.step_count + 1,
                "trace": [
                    _trace(
                        state,
                        "brief_writer",
                        StepKind.DECISION,
                        started=started,
                        source=OutputSource.DETERMINISTIC,
                        ok=False,
                        error=message,
                    )
                ],
            }

        final = brief.model_copy(
            update={
                "posting_id": posting.posting_id,
                "attribution": live.attribution(),
                "source": source,
                # Anything in never_claim that this posting actually asks for
                # is added back deterministically. The rule that the repository
                # exists for is not left to the model's discretion.
                "not_claimable": _merge_not_claimable(state, brief.not_claimable, body),
            }
        )
        emit("brief_written", run_id=state.run_id, posting=posting.posting_id, source=source.value)
        return {
            "briefs": [final],
            "step_count": state.step_count + 1,
            "trace": [
                _trace(
                    state,
                    "brief_writer",
                    StepKind.DECISION,
                    name="brief",
                    started=started,
                    source=source,
                )
            ],
        }

    return brief_writer


def _merge_not_claimable(state: PipelineState, model_items: list[str], body: str) -> list[str]:
    """Union of what the model listed and what the posting demonstrably asks for.

    Deterministic, and deliberately not delegated: ``not_claimable`` is the
    field this whole project is about. If the posting asks for something on the
    never_claim list, it appears here whether the model thought of it or not.
    """
    haystack = body.casefold()
    merged = list(model_items)
    seen = {item.casefold() for item in merged}
    for item in state.profile.never_claim:
        if item.casefold() in haystack and item.casefold() not in seen:
            merged.append(item)
            seen.add(item.casefold())
    return merged


# --------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------


def make_finalize(deps: GraphDeps) -> Any:
    """Close the run with a named outcome. No model involved."""

    def finalize(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(UTC)
        capped = state.step_count >= deps.ctx.settings.max_graph_steps
        fatal = [e for e in state.errors if not e.recoverable]

        if capped:
            status, reason = (
                "step_cap",
                (f"graph step ceiling of {deps.ctx.settings.max_graph_steps} reached"),
            )
        elif fatal:
            status, reason = "failed", fatal[0].message
        elif state.briefs:
            status, reason = "completed", f"{len(state.briefs)} brief(s) drafted for review"
        elif state.assessments:
            status, reason = "no_fit", "no posting was worth a brief for this profile"
        else:
            status, reason = "failed", "nothing was assessed"

        outcome = RunOutcome(
            status=status,
            reason=reason,
            steps_used=state.step_count,
            usage=state.usage_total,
        )
        emit(
            "run_finished",
            run_id=state.run_id,
            status=status,
            reason=reason,
            steps=state.step_count,
            input_tokens=state.usage_total.input_tokens,
            output_tokens=state.usage_total.output_tokens,
        )
        return {
            "outcome": outcome,
            "next_agent": None,
            "trace": [
                _trace(
                    state,
                    "finalize",
                    StepKind.DECISION,
                    name=status,
                    started=started,
                    source=OutputSource.DETERMINISTIC,
                )
            ],
        }

    return finalize


class GraphConfigError(JobfitError):
    """The graph was built with an impossible configuration."""
