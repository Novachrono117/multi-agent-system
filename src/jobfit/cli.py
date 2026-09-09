"""Command line entry point.

Three subcommands, and the README promises exactly one of them as the demo:

    jobfit run --offline --posting examples/posting_sample.txt

``run`` is the pipeline. ``graph`` prints the architecture diagram from the
graph that actually executes, so the README picture cannot drift from the code.
``doctor`` reports which transports are usable on this machine, because the
first question a new reader has is "why did it not talk to a model".

Output goes to stdout, run events to stderr, so both can be redirected
independently:

    jobfit run --offline --posting p.txt > brief.md 2> trace.jsonl
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobfit.config import Settings, get_settings
from jobfit.errors import JobfitError
from jobfit.graph import GraphDeps, IntakeRequest, PipelineState, build_graph, graph_mermaid
from jobfit.llm import get_transport
from jobfit.llm.fake import OfflineDemoTransport
from jobfit.llm.retry import with_retry
from jobfit.llm.transport import MessageTransport
from jobfit.models.brief import RoleBrief
from jobfit.models.profile import CandidateProfile, load_profile
from jobfit.observability import FileSink, emit
from jobfit.tools import build_registry
from jobfit.tools.http_cache import ALLOWED_HOSTS, CachedFetcher
from jobfit.tools.registry import ToolContext

DEFAULT_PROFILE = Path("configs/profile.toml")
EXAMPLE_PROFILE = Path("configs/profile.example.toml")


def _resolve_profile(explicit: Path | None) -> Path:
    """Prefer an explicit path, then a real profile, then the example.

    The example is a last resort and says so at runtime: assessing against a
    fictional profile silently would make every verdict meaningless.
    """
    if explicit is not None:
        return explicit
    if DEFAULT_PROFILE.exists():
        return DEFAULT_PROFILE
    return EXAMPLE_PROFILE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobfit",
        description=(
            "Multi-agent decision support for job applications. Assesses "
            "postings against a candidate profile and drafts a reviewable "
            "brief. It never submits an application."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="assess postings and draft a brief")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--posting", type=Path, help="file containing one job posting")
    source.add_argument("--text", help="job posting text, inline")
    source.add_argument("--search", help="search public job boards for this query")

    run.add_argument(
        "--offline",
        action="store_true",
        help="run with no model and no network, using the scripted demo transport",
    )
    run.add_argument("--profile", type=Path, help=f"profile TOML (default: {DEFAULT_PROFILE})")
    run.add_argument(
        "--sources",
        nargs="+",
        default=["remotive"],
        choices=["remotive", "arbeitnow", "greenhouse"],
        help="which boards to search (with --search)",
    )
    run.add_argument("--board", action="append", default=[], help="Greenhouse board token")
    run.add_argument("--limit", type=int, default=3, help="max postings to consider")
    run.add_argument("--out", type=Path, help="write the brief here instead of stdout")
    run.add_argument("--trace", type=Path, help="write the run trace as JSON lines")
    run.add_argument(
        "--approve",
        action="store_true",
        help="pause before the brief writer so a human can review first",
    )
    run.add_argument("--json", action="store_true", help="emit the full run state as JSON")

    sub.add_parser("graph", help="print the architecture diagram as mermaid")
    sub.add_parser("doctor", help="report configuration and available transports")
    return parser


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _choose_transport(args: argparse.Namespace, settings: Settings) -> MessageTransport:
    if args.offline:
        return OfflineDemoTransport()
    return with_retry(get_transport(settings))


def _make_request(args: argparse.Namespace) -> IntakeRequest:
    if args.posting:
        return IntakeRequest(raw_text=args.posting.read_text(encoding="utf-8"))
    if args.text:
        return IntakeRequest(raw_text=args.text)
    return IntakeRequest(
        search_query=args.search,
        sources=list(args.sources),
        board_tokens=list(args.board),
        limit=args.limit,
    )


def _render_brief(brief: RoleBrief, state: PipelineState) -> str:
    """Markdown, because the output is a document a person reads."""
    posting = state.posting(brief.posting_id)
    assessment = state.assessment_for(brief.posting_id)
    lines: list[str] = []

    title = posting.title if posting else brief.posting_id
    lines.append(f"# {title}")
    if posting and posting.company:
        lines.append(f"**{posting.company}**")
    lines.append("")

    if assessment:
        verdict = assessment.verdict.value.replace("_", " ")
        lines.append(f"**Verdict:** {verdict}")
        if assessment.score:
            score = assessment.score
            lines.append(
                f"**Requirement coverage:** {score.coverage:.0%} "
                f"({len(score.matched)} of {len(score.matched) + len(score.missing)}) "
                "- measured deterministically, not estimated by the model"
            )
            if score.deal_breakers_hit:
                lines.append(f"**Deal breakers hit:** {', '.join(score.deal_breakers_hit)}")
        lines.append("")

    lines.append(brief.role_summary)
    lines.append("")

    for heading, items in (
        ("Why this could fit", brief.why_fit),
        ("Real gaps", brief.gaps),
        ("Do NOT claim these in an interview", brief.not_claimable),
        ("Questions for the recruiter", brief.questions_for_recruiter),
        ("What to study", brief.prep_topics),
    ):
        if items:
            lines.append(f"## {heading}")
            lines.extend(f"- {item}" for item in items)
            lines.append("")

    if brief.attribution:
        lines.append(f"_{brief.attribution}_")
    if posting and posting.url:
        lines.append(f"_Posting: {posting.url}_")
    lines.append("")
    lines.append(f"> {brief.disclaimer}")
    lines.append(f"> Output source: {brief.source.value}")
    return "\n".join(lines)


def _render_no_brief(state: PipelineState) -> str:
    lines = ["# No brief drafted", ""]
    if state.outcome:
        lines.append(f"**Outcome:** {state.outcome.status} - {state.outcome.reason}")
        lines.append("")
    for assessment in state.assessments:
        posting = state.posting(assessment.posting_id)
        name = posting.title if posting else assessment.posting_id
        lines.append(f"- **{name}**: {assessment.verdict.value}")
        if assessment.score and assessment.score.deal_breakers_hit:
            lines.append(f"  - deal breakers: {', '.join(assessment.score.deal_breakers_hit)}")
        if assessment.reasoning:
            lines.append(f"  - {assessment.reasoning}")
    if state.errors:
        lines.append("")
        lines.append("## Problems")
        lines.extend(f"- [{e.node}] {e.message}" for e in state.errors)
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    profile_path = _resolve_profile(args.profile)

    try:
        profile: CandidateProfile = load_profile(profile_path)
    except FileNotFoundError:
        print(
            f"error: no profile found at {profile_path}.\n"
            f"Copy the example and edit it:\n"
            f"    cp {EXAMPLE_PROFILE} {DEFAULT_PROFILE}",
            file=sys.stderr,
        )
        return 2

    if profile_path == EXAMPLE_PROFILE:
        print(
            f"note: using the fictional example profile ({EXAMPLE_PROFILE}). "
            f"Every verdict below is about a made-up candidate. "
            f"Copy it to {DEFAULT_PROFILE} and edit for real use.",
            file=sys.stderr,
        )

    transport = _choose_transport(args, settings)
    needs_network = bool(args.search) and not args.offline
    fetcher = (
        CachedFetcher(cache_dir=settings.cache_dir, ttl_seconds=settings.cache_ttl_seconds)
        if needs_network
        else None
    )

    ctx = ToolContext(
        settings=settings,
        profile=profile,
        http=fetcher.http_client() if fetcher else None,
    )
    deps = GraphDeps(transport=transport, ctx=ctx, screener_registry=build_registry())
    graph = build_graph(deps, approve_briefs=args.approve)

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    state = PipelineState(run_id=run_id, profile=profile, request=_make_request(args))

    sink = FileSink(args.trace) if args.trace else None
    try:
        if sink:
            sink.__enter__()
        emit("run_started", run_id=run_id, transport=transport.name, mode=state.request.mode)
        final = PipelineState.model_validate(graph.invoke(state))
    except JobfitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if sink:
            sink.__exit__()
        if fetcher:
            fetcher.close()

    if args.json:
        rendered = final.model_dump_json(indent=2)
    elif final.briefs:
        rendered = "\n\n---\n\n".join(_render_brief(b, final) for b in final.briefs)
    else:
        rendered = _render_no_brief(final)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(rendered)

    if final.outcome and final.outcome.status == "failed":
        return 1
    return 0


# --------------------------------------------------------------------------
# graph / doctor
# --------------------------------------------------------------------------


def cmd_graph(_args: argparse.Namespace) -> int:
    settings = get_settings()
    profile = load_profile(_resolve_profile(None))
    deps = GraphDeps(
        transport=OfflineDemoTransport(),
        ctx=ToolContext(settings=settings, profile=profile),
    )
    print(graph_mermaid(deps))
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Say plainly what is configured and what will happen."""
    settings = get_settings()
    profile_path = _resolve_profile(None)

    rows: list[tuple[str, str]] = [
        ("transport", settings.transport),
        ("ollama", f"{settings.ollama_model} at {settings.ollama_base_url}"),
        ("anthropic model", settings.anthropic_model),
        ("anthropic key", "set" if settings.anthropic_available else "NOT set"),
        ("profile", f"{profile_path} ({'exists' if profile_path.exists() else 'MISSING'})"),
        ("max tool steps", str(settings.max_tool_steps)),
        ("max graph steps", str(settings.max_graph_steps)),
        ("posting char budget", str(settings.max_posting_chars)),
        ("strict errors", str(settings.strict)),
        ("cache", f"{settings.cache_dir} (ttl {settings.cache_ttl_seconds}s)"),
        ("tools", ", ".join(build_registry().names)),
        ("allowed hosts", ", ".join(sorted(ALLOWED_HOSTS))),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")

    print()
    if settings.transport == "anthropic" and not settings.anthropic_available:
        print(
            "The Anthropic transport is selected but no key is set, so a run "
            "would fail. Free alternatives:\n"
            "  JOBFIT_TRANSPORT=ollama   local model (install ollama, "
            "then: ollama pull qwen3:8b)\n"
            "  jobfit run --offline      scripted, no model at all"
        )
    elif settings.transport == "ollama":
        print(
            f"Ollama is selected. It must be running and have {settings.ollama_model} "
            f"pulled:\n  ollama pull {settings.ollama_model}\n"
            "Use `jobfit run --offline` to try the pipeline without any model."
        )
    return 0


COMMANDS: dict[str, Any] = {"run": cmd_run, "graph": cmd_graph, "doctor": cmd_doctor}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(COMMANDS[args.command](args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except JobfitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
