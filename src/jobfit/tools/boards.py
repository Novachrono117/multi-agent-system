"""Adapters for the job board APIs, one function per board.

Each adapter maps a real, measured payload onto ``JobPosting``. The field
mappings below are not guesses - they were read off responses captured on
2026-09-08 and pinned as fixtures under ``tests/fixtures/``:

* **Greenhouse** - ``id`` is an int, ``location`` is a nested object
  (``{"name": ...}``), and the listing endpoint carries no body at all. Asking
  it for one via ``?content=true`` returned 8.5 MB, so the body is fetched per
  posting instead.
* **Remotive** - flat and generous: ``salary``, ``tags``, ``job_type`` and
  ``candidate_required_location`` all arrive structured. Every posting is
  remote by definition of the board. ``description`` is HTML and ran to 15,796
  characters in the sample.
* **Arbeitnow** - ``slug`` is the identifier, ``remote`` is an explicit
  boolean, and ``created_at`` is a Unix timestamp rather than an ISO string.

Each board's quirks are handled here so that nothing downstream has to know
which board a posting came from.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from jobfit.models.posting import JobPosting, PostingSource
from jobfit.tools.http_cache import CachedFetcher

GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

#: Boards that can be searched. Lever is deliberately absent - see
#: ``PostingSource`` for why.
SEARCHABLE = ("greenhouse", "remotive", "arbeitnow")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_epoch(value: Any) -> datetime | None:
    if not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def matches_query(query: str, *fields: Any) -> bool:
    """Case-folded substring match across the given fields.

    Filtering happens client-side on purpose. Remotive ignored the ``limit``
    parameter in testing (it returned all 18 postings for the category
    regardless), and Arbeitnow has no query parameter at all, so a board's own
    filtering cannot be relied on.
    """
    if not query.strip():
        return True
    needle = query.casefold()
    haystack = " ".join(str(f) for f in fields if f).casefold()
    return all(term in haystack for term in needle.split())


# --------------------------------------------------------------------------
# Greenhouse
# --------------------------------------------------------------------------


def greenhouse_listing(fetcher: CachedFetcher, board_token: str) -> list[JobPosting]:
    """List a Greenhouse board. Bodies are not included - by design.

    The listing is 60 KB for 97 postings. The same endpoint with
    ``?content=true`` measured 8.5 MB, so bodies are fetched one posting at a
    time by ``greenhouse_detail``.
    """
    payload = fetcher.get_json(f"{GREENHOUSE_BASE}/{board_token}/jobs")
    postings: list[JobPosting] = []
    for job in payload.get("jobs", []):
        location = job.get("location") or {}
        postings.append(
            JobPosting(
                source=PostingSource.GREENHOUSE,
                external_id=str(job.get("id")),
                title=job.get("title") or "(untitled)",
                company=(job.get("company_name") or "").strip() or None,
                url=job.get("absolute_url"),
                location=location.get("name") if isinstance(location, dict) else None,
                published_at=_parse_iso(job.get("first_published")),
            )
        )
    return postings


def greenhouse_detail(fetcher: CachedFetcher, board_token: str, external_id: str) -> str | None:
    """Fetch one posting's body. Entity-escaped markup; the parser handles that."""
    payload = fetcher.get_json(f"{GREENHOUSE_BASE}/{board_token}/jobs/{external_id}")
    content = payload.get("content")
    return content if isinstance(content, str) else None


# --------------------------------------------------------------------------
# Remotive
# --------------------------------------------------------------------------


def remotive_search(fetcher: CachedFetcher, query: str, limit: int) -> list[JobPosting]:
    """Search Remotive.

    Attribution is not optional here: the API response itself carries a legal
    notice requiring a link back to the listing, forbidding redistribution, and
    asking for at most roughly four requests a day. ``JobPosting`` enforces the
    link, and ``CachedFetcher`` enforces the request budget.
    """
    payload = fetcher.get_json(REMOTIVE_URL, {"search": query} if query.strip() else None)
    postings: list[JobPosting] = []
    for job in payload.get("jobs", []):
        if not matches_query(query, job.get("title"), job.get("company_name"), job.get("tags")):
            continue
        postings.append(
            JobPosting(
                source=PostingSource.REMOTIVE,
                external_id=str(job.get("id")),
                title=job.get("title") or "(untitled)",
                company=job.get("company_name"),
                url=job.get("url"),
                location=job.get("candidate_required_location"),
                # Remotive is a remote-only board; every listing is remote.
                remote=True,
                job_type=job.get("job_type"),
                salary_text=(job.get("salary") or "").strip() or None,
                tags=[t for t in (job.get("tags") or []) if isinstance(t, str)],
                published_at=_parse_iso(job.get("publication_date")),
                raw_body=job.get("description"),
            )
        )
        if len(postings) >= limit:
            break
    return postings


# --------------------------------------------------------------------------
# Arbeitnow
# --------------------------------------------------------------------------


def arbeitnow_search(fetcher: CachedFetcher, query: str, limit: int) -> list[JobPosting]:
    """Search Arbeitnow. The endpoint takes no query parameter, so filter here."""
    payload = fetcher.get_json(ARBEITNOW_URL)
    postings: list[JobPosting] = []
    for job in payload.get("data", []):
        if not matches_query(query, job.get("title"), job.get("company_name"), job.get("tags")):
            continue
        remote = job.get("remote")
        postings.append(
            JobPosting(
                source=PostingSource.ARBEITNOW,
                external_id=str(job.get("slug")),
                title=job.get("title") or "(untitled)",
                company=job.get("company_name"),
                url=job.get("url"),
                location=job.get("location"),
                remote=remote if isinstance(remote, bool) else None,
                job_type=", ".join(job.get("job_types") or []) or None,
                tags=[t for t in (job.get("tags") or []) if isinstance(t, str)],
                published_at=_parse_epoch(job.get("created_at")),
                raw_body=job.get("description"),
            )
        )
        if len(postings) >= limit:
            break
    return postings
