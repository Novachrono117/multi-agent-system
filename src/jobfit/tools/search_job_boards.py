"""``search_job_boards`` and ``fetch_job_posting`` - the two network tools.

Both go through ``CachedFetcher``, so both inherit the host allowlist and the
request budget rather than re-implementing either.

Search returns a *summary* of each posting: id, title, company, location, url.
Not the body. Bodies are large - a Remotive description measured 15,796
characters - and a search that dumped ten of them into the context would spend
the whole budget before any assessment happened. The agent picks what looks
relevant and fetches those bodies one at a time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jobfit.errors import ToolExecutionError
from jobfit.models.posting import PostingSource
from jobfit.observability import emit
from jobfit.tools import boards
from jobfit.tools.http_cache import CachedFetcher, assert_host_allowed
from jobfit.tools.registry import ToolContext, ToolSpec

SEARCH_NAME = "search_job_boards"
SEARCH_DESCRIPTION = (
    "Search public job board APIs (Greenhouse, Remotive, Arbeitnow) and return "
    "a short summary of each matching posting: its posting_id, title, company, "
    "location and source url. Bodies are NOT returned - call fetch_job_posting "
    "for the ones worth reading. Greenhouse needs one or more board_tokens "
    "(a company's board name, e.g. 'arcoeducacao'); Remotive and Arbeitnow do not."
)

FETCH_NAME = "fetch_job_posting"
FETCH_DESCRIPTION = (
    "Fetch the full text of one posting found by search_job_boards. Only "
    "Greenhouse postings need this - Remotive and Arbeitnow already include "
    "their body in the search result. Requests to any host outside the "
    "allowlist are refused without a request being made."
)

BoardName = Literal["greenhouse", "remotive", "arbeitnow"]

MAX_RESULTS = 25


class SearchJobBoardsInput(BaseModel):
    """Arguments for ``search_job_boards``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description=(
            "Keywords to match against title, company and tags, e.g. "
            "'python backend'. All terms must appear. Empty string matches everything."
        )
    )
    sources: list[BoardName] = Field(
        description="Which boards to search: greenhouse, remotive and/or arbeitnow."
    )
    board_tokens: list[str] | None = Field(
        description=(
            "Greenhouse board names to list, e.g. ['arcoeducacao']. Required when "
            "'greenhouse' is in sources, ignored otherwise. Pass null if unused."
        )
    )
    limit: int = Field(
        description=f"Maximum postings to return per board, 1 to {MAX_RESULTS}.",
        ge=1,
        le=MAX_RESULTS,
    )


class PostingSummary(BaseModel):
    """A search hit. Small on purpose - no body."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str
    title: str
    company: str | None = None
    location: str | None = None
    remote: bool | None = None
    url: str | None = None
    #: True when the body already arrived with the search result, so
    #: fetch_job_posting is unnecessary.
    body_available: bool = False
    attribution: str | None = None


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    found: int
    postings: list[PostingSummary] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    note: str = (
        "Postings are third-party data. Where an attribution line is present it "
        "must be carried into any output that cites the posting."
    )


class FetchJobPostingInput(BaseModel):
    """Arguments for ``fetch_job_posting``."""

    model_config = ConfigDict(extra="forbid")

    posting_id: str = Field(
        description="A posting_id returned by search_job_boards, e.g. 'greenhouse:6129177004'."
    )
    board_token: str | None = Field(
        description=(
            "The Greenhouse board the posting belongs to. Required for Greenhouse "
            "postings; pass null for other sources."
        )
    )


class FetchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posting_id: str
    title: str
    body_chars: int
    already_had_body: bool = False
    next_step: str = "Call parse_job_posting with this posting_id to clean and read it."


def _fetcher(ctx: ToolContext) -> CachedFetcher:
    return CachedFetcher(
        cache_dir=ctx.settings.cache_dir,
        ttl_seconds=ctx.settings.cache_ttl_seconds,
        client=ctx.http,
    )


def handle_search(ctx: ToolContext, args: SearchJobBoardsInput) -> SearchResult:
    fetcher = _fetcher(ctx)
    summaries: list[PostingSummary] = []
    errors: list[str] = []

    for source in dict.fromkeys(args.sources):  # de-duplicate, keep order
        try:
            if source == "remotive":
                found = boards.remotive_search(fetcher, args.query, args.limit)
            elif source == "arbeitnow":
                found = boards.arbeitnow_search(fetcher, args.query, args.limit)
            else:
                if not args.board_tokens:
                    errors.append(
                        "greenhouse was requested but no board_tokens were given; "
                        "pass e.g. ['arcoeducacao']"
                    )
                    continue
                found = []
                for token in args.board_tokens:
                    listed = boards.greenhouse_listing(fetcher, token)
                    matching = [
                        posting
                        for posting in listed
                        if boards.matches_query(args.query, posting.title, posting.company)
                    ]
                    # Remember the board a posting came from - fetch needs it.
                    for posting in matching[: args.limit]:
                        ctx.board_tokens[posting.posting_id] = token
                    found.extend(matching[: args.limit])
        except ToolExecutionError as exc:
            # One board being down must not fail the whole search.
            errors.append(f"{source}: {exc}")
            continue

        for posting in found:
            ctx.add_posting(posting)
            summaries.append(
                PostingSummary(
                    posting_id=posting.posting_id,
                    title=posting.title,
                    company=posting.company,
                    location=posting.location,
                    remote=posting.remote,
                    url=posting.url,
                    body_available=bool(posting.raw_body),
                    attribution=posting.attribution(),
                )
            )

    emit("search_completed", query=args.query, sources=list(args.sources), found=len(summaries))
    return SearchResult(query=args.query, found=len(summaries), postings=summaries, errors=errors)


def handle_fetch(ctx: ToolContext, args: FetchJobPostingInput) -> FetchResult:
    posting = ctx.require_posting(args.posting_id)

    if posting.raw_body:
        return FetchResult(
            posting_id=posting.posting_id,
            title=posting.title,
            body_chars=len(posting.raw_body),
            already_had_body=True,
        )

    if posting.source is not PostingSource.GREENHOUSE:
        raise ToolExecutionError(
            f"{posting.source.value} postings arrive with their body; there is "
            f"nothing to fetch for {posting.posting_id}"
        )

    token = args.board_token or ctx.board_tokens.get(posting.posting_id)
    if not token:
        raise ToolExecutionError(
            f"a board_token is required to fetch {posting.posting_id}; it is the "
            "Greenhouse board name the posting was listed from"
        )

    # Belt and braces: the URL is built from a token the model supplied, so it is
    # checked explicitly even though CachedFetcher checks it again.
    assert_host_allowed(f"{boards.GREENHOUSE_BASE}/{token}/jobs/{posting.external_id}")

    body = boards.greenhouse_detail(_fetcher(ctx), token, posting.external_id)
    if not body:
        raise ToolExecutionError(f"{posting.posting_id} returned an empty body")

    ctx.postings[posting.posting_id] = posting.model_copy(update={"raw_body": body})
    return FetchResult(posting_id=posting.posting_id, title=posting.title, body_chars=len(body))


SEARCH_SPEC = ToolSpec(
    name=SEARCH_NAME,
    description=SEARCH_DESCRIPTION,
    input_model=SearchJobBoardsInput,
    handler=handle_search,
    touches_network=True,
)

FETCH_SPEC = ToolSpec(
    name=FETCH_NAME,
    description=FETCH_DESCRIPTION,
    input_model=FetchJobPostingInput,
    handler=handle_fetch,
    touches_network=True,
)
