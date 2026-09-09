"""The network tools, with a fake HTTP client in place of the internet.

The two claims being tested are the ones the project makes out loud:

* a host outside the allowlist is refused **with zero requests made** - not
  refused after connecting, and not merely logged;
* the cache is real, so the Remotive request budget is honoured by construction.

The payload mappings are checked against fixtures captured from the live APIs,
so a board changing its schema shows up as a failing test rather than as a
posting with an empty title.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jobfit.config import Settings
from jobfit.errors import DisallowedHostError, ToolExecutionError
from jobfit.models.posting import PostingSource
from jobfit.models.profile import CandidateProfile, load_profile
from jobfit.tools import boards, build_registry
from jobfit.tools.http_cache import ALLOWED_HOSTS, CachedFetcher, assert_host_allowed
from jobfit.tools.registry import ToolContext, ToolRegistry

FIXTURES = Path("tests/fixtures")


class FakeResponse:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> Any:
        return self._payload


class FakeHttpClient:
    """Records every call and serves canned payloads by URL substring."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes = routes or {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> FakeResponse:
        self.calls.append((url, params))
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, FakeResponse):
                    return payload
                return FakeResponse(payload)
        raise AssertionError(f"unrouted URL in test: {url}")

    def close(self) -> None:  # pragma: no cover - interface completeness
        pass


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def profile() -> CandidateProfile:
    return load_profile(Path("configs/profile.example.toml"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Cache lands in tmp_path so tests never touch the developer's cache."""
    return Settings(_env_file=None, cache_dir=tmp_path / "cache", cache_ttl_seconds=3600)


@pytest.fixture
def registry() -> ToolRegistry:
    return build_registry()


@pytest.fixture
def routes() -> dict[str, Any]:
    greenhouse_list = _load("greenhouse_jobs.json")
    return {
        "boards-api.greenhouse.io/v1/boards/arcoeducacao/jobs/": _load(
            "greenhouse_job_detail.json"
        ),
        "boards-api.greenhouse.io/v1/boards/arcoeducacao/jobs": greenhouse_list,
        "remotive.com/api/remote-jobs": _load("remotive_jobs.json"),
        "arbeitnow.com/api/job-board-api": _load("arbeitnow_jobs.json"),
    }


def _ctx(settings: Settings, profile: CandidateProfile, http: Any) -> ToolContext:
    return ToolContext(settings=settings, profile=profile, http=http)


class TestHostAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/jobs/view/123",
            "https://br.indeed.com/viewjob?jk=abc",
            "https://smartapply.indeed.com/x",
            "https://evil.example.com/steal",
            "https://boards-api.greenhouse.io.attacker.tld/v1/boards/x/jobs",
        ],
    )
    def test_forbidden_host_is_refused(self, url: str) -> None:
        with pytest.raises(DisallowedHostError):
            assert_host_allowed(url)

    @pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS))
    def test_allowlisted_hosts_pass(self, host: str) -> None:
        assert assert_host_allowed(f"https://{host}/path") == host

    def test_plain_http_is_refused_even_for_an_allowed_host(self) -> None:
        with pytest.raises(DisallowedHostError, match="only https"):
            assert_host_allowed("http://remotive.com/api/remote-jobs")

    def test_refusal_happens_before_any_request_is_made(self, tmp_path: Path) -> None:
        """The whole point: the socket is never opened, not merely ignored."""
        fake = FakeHttpClient()
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=0, client=fake)
        with pytest.raises(DisallowedHostError):
            fetcher.get_json("https://www.linkedin.com/jobs/view/123")
        assert fake.calls == []

    def test_the_refusal_names_what_is_permitted(self) -> None:
        with pytest.raises(DisallowedHostError) as excinfo:
            assert_host_allowed("https://www.linkedin.com/x")
        message = str(excinfo.value)
        assert "remotive.com" in message
        assert "by design" in message

    def test_indeed_and_linkedin_are_not_on_the_list(self) -> None:
        """A regression guard with teeth: adding either would fail here."""
        for host in ALLOWED_HOSTS:
            assert "linkedin" not in host
            assert "indeed" not in host


class TestCaching:
    def test_second_identical_request_is_served_from_cache(self, tmp_path: Path) -> None:
        fake = FakeHttpClient({"remotive.com": {"jobs": []}})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=3600, client=fake)
        fetcher.get_json(boards.REMOTIVE_URL)
        fetcher.get_json(boards.REMOTIVE_URL)
        assert len(fake.calls) == 1, "the cache did not prevent a second request"

    def test_expired_cache_is_refetched(self, tmp_path: Path) -> None:
        """Clock is injected, so expiry is tested without sleeping."""
        fake = FakeHttpClient({"remotive.com": {"jobs": []}})
        clock = [1_000_000.0]
        fetcher = CachedFetcher(
            cache_dir=tmp_path, ttl_seconds=60, client=fake, now=lambda: clock[0]
        )
        fetcher.get_json(boards.REMOTIVE_URL)
        clock[0] += 3600
        fetcher.get_json(boards.REMOTIVE_URL)
        assert len(fake.calls) == 2

    def test_zero_ttl_disables_the_cache(self, tmp_path: Path) -> None:
        fake = FakeHttpClient({"remotive.com": {"jobs": []}})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=0, client=fake)
        fetcher.get_json(boards.REMOTIVE_URL)
        fetcher.get_json(boards.REMOTIVE_URL)
        assert len(fake.calls) == 2

    def test_differing_params_are_cached_separately(self, tmp_path: Path) -> None:
        fake = FakeHttpClient({"remotive.com": {"jobs": []}})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=3600, client=fake)
        fetcher.get_json(boards.REMOTIVE_URL, {"search": "python"})
        fetcher.get_json(boards.REMOTIVE_URL, {"search": "rust"})
        assert len(fake.calls) == 2

    def test_corrupt_cache_entry_is_ignored_not_fatal(self, tmp_path: Path) -> None:
        fake = FakeHttpClient({"remotive.com": {"jobs": []}})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=3600, client=fake)
        fetcher.get_json(boards.REMOTIVE_URL)
        for path in tmp_path.rglob("*.json"):
            path.write_text("{not json", encoding="utf-8")
        assert fetcher.get_json(boards.REMOTIVE_URL) == {"jobs": []}


class TestTransportFailures:
    def test_non_200_becomes_a_tool_error(self, tmp_path: Path) -> None:
        fake = FakeHttpClient({"remotive.com": FakeResponse({}, status_code=503)})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=0, client=fake)
        with pytest.raises(ToolExecutionError, match="503"):
            fetcher.get_json(boards.REMOTIVE_URL)

    def test_oversized_response_is_refused(self, tmp_path: Path) -> None:
        """Guards against the 8.5 MB Greenhouse content endpoint by accident."""
        huge = FakeResponse({"jobs": []})
        huge.content = b"x" * 5_000_000
        fake = FakeHttpClient({"remotive.com": huge})
        fetcher = CachedFetcher(cache_dir=tmp_path, ttl_seconds=0, client=fake)
        with pytest.raises(ToolExecutionError, match="over the"):
            fetcher.get_json(boards.REMOTIVE_URL)

    def test_one_board_failing_does_not_fail_the_whole_search(
        self, settings: Settings, profile: CandidateProfile, registry: ToolRegistry
    ) -> None:
        fake = FakeHttpClient(
            {
                "remotive.com": _load("remotive_jobs.json"),
                "arbeitnow.com": FakeResponse({}, status_code=500),
            }
        )
        ctx = _ctx(settings, profile, fake)
        outcome = registry.dispatch(
            "search_job_boards",
            {
                "query": "",
                "sources": ["remotive", "arbeitnow"],
                "board_tokens": None,
                "limit": 5,
            },
            ctx,
        )
        assert not outcome.is_error
        result = json.loads(outcome.content)
        assert result["found"] > 0, "remotive results were lost when arbeitnow failed"
        assert any("arbeitnow" in err for err in result["errors"])


class TestBoardMapping:
    def test_greenhouse_listing_maps_nested_location_and_int_id(
        self, settings: Settings, routes: dict[str, Any]
    ) -> None:
        """Greenhouse returns location as an object and id as an int."""
        fake = FakeHttpClient(routes)
        fetcher = CachedFetcher(cache_dir=settings.cache_dir, ttl_seconds=0, client=fake)
        postings = boards.greenhouse_listing(fetcher, "arcoeducacao")
        assert postings
        first = postings[0]
        assert first.source is PostingSource.GREENHOUSE
        assert first.external_id == "6129177004"
        assert first.location == "São Paulo"
        assert first.url and first.url.startswith("https://")
        assert first.raw_body is None, "the listing must not carry bodies"

    def test_remotive_postings_are_marked_remote_and_carry_attribution(
        self, settings: Settings, routes: dict[str, Any]
    ) -> None:
        fake = FakeHttpClient(routes)
        fetcher = CachedFetcher(cache_dir=settings.cache_dir, ttl_seconds=0, client=fake)
        postings = boards.remotive_search(fetcher, "", 10)
        assert postings
        for posting in postings:
            assert posting.remote is True
            assert posting.attribution()
            assert posting.raw_body, "remotive includes the body in search results"

    def test_arbeitnow_epoch_timestamp_is_parsed(
        self, settings: Settings, routes: dict[str, Any]
    ) -> None:
        """created_at is a Unix timestamp, not an ISO string."""
        fake = FakeHttpClient(routes)
        fetcher = CachedFetcher(cache_dir=settings.cache_dir, ttl_seconds=0, client=fake)
        postings = boards.arbeitnow_search(fetcher, "", 10)
        assert postings
        assert postings[0].published_at is not None
        assert postings[0].published_at.year > 2000

    def test_limit_is_enforced_client_side(
        self, settings: Settings, routes: dict[str, Any]
    ) -> None:
        """Remotive ignored its own limit parameter, so we cannot rely on it."""
        fake = FakeHttpClient(routes)
        fetcher = CachedFetcher(cache_dir=settings.cache_dir, ttl_seconds=0, client=fake)
        assert len(boards.remotive_search(fetcher, "", 1)) == 1

    def test_query_requires_every_term_to_match(self) -> None:
        assert boards.matches_query("python backend", "Senior Python Backend Engineer")
        assert not boards.matches_query("python rust", "Senior Python Backend Engineer")
        assert boards.matches_query("", "anything at all")


class TestSearchAndFetchTools:
    def test_search_returns_summaries_without_bodies(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        """A search that returned bodies would blow the context budget."""
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        outcome = registry.dispatch(
            "search_job_boards",
            {"query": "", "sources": ["remotive"], "board_tokens": None, "limit": 3},
            ctx,
        )
        result = json.loads(outcome.content)
        assert result["found"] > 0
        for summary in result["postings"]:
            assert "body_text" not in summary
            assert "raw_body" not in summary
            assert summary["posting_id"]

    def test_search_populates_the_run_context(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        registry.dispatch(
            "search_job_boards",
            {"query": "", "sources": ["remotive"], "board_tokens": None, "limit": 2},
            ctx,
        )
        assert len(ctx.postings) == 2

    def test_greenhouse_without_board_tokens_reports_the_problem(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        outcome = registry.dispatch(
            "search_job_boards",
            {"query": "", "sources": ["greenhouse"], "board_tokens": None, "limit": 3},
            ctx,
        )
        result = json.loads(outcome.content)
        assert result["found"] == 0
        assert any("board_tokens" in err for err in result["errors"])

    def test_fetch_remembers_the_board_from_the_search(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        """The model should not have to carry the board token back to us."""
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        search = json.loads(
            registry.dispatch(
                "search_job_boards",
                {
                    "query": "",
                    "sources": ["greenhouse"],
                    "board_tokens": ["arcoeducacao"],
                    "limit": 1,
                },
                ctx,
            ).content
        )
        posting_id = search["postings"][0]["posting_id"]
        outcome = registry.dispatch(
            "fetch_job_posting", {"posting_id": posting_id, "board_token": None}, ctx
        )
        assert not outcome.is_error, outcome.content
        assert json.loads(outcome.content)["body_chars"] > 1000
        assert ctx.postings[posting_id].raw_body

    def test_fetch_is_a_noop_when_the_body_is_already_present(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        search = json.loads(
            registry.dispatch(
                "search_job_boards",
                {"query": "", "sources": ["remotive"], "board_tokens": None, "limit": 1},
                ctx,
            ).content
        )
        posting_id = search["postings"][0]["posting_id"]
        result = json.loads(
            registry.dispatch(
                "fetch_job_posting", {"posting_id": posting_id, "board_token": None}, ctx
            ).content
        )
        assert result["already_had_body"] is True

    def test_full_chain_search_fetch_parse_score(
        self,
        settings: Settings,
        profile: CandidateProfile,
        registry: ToolRegistry,
        routes: dict[str, Any],
    ) -> None:
        """The four tools compose, using only captured payloads and no network."""
        ctx = _ctx(settings, profile, FakeHttpClient(routes))
        search = json.loads(
            registry.dispatch(
                "search_job_boards",
                {
                    "query": "",
                    "sources": ["greenhouse"],
                    "board_tokens": ["arcoeducacao"],
                    "limit": 1,
                },
                ctx,
            ).content
        )
        posting_id = search["postings"][0]["posting_id"]

        registry.dispatch("fetch_job_posting", {"posting_id": posting_id, "board_token": None}, ctx)
        parsed = json.loads(
            registry.dispatch("parse_job_posting", {"posting_id": posting_id}, ctx).content
        )
        scored = json.loads(
            registry.dispatch("score_profile_match", {"posting_id": posting_id}, ctx).content
        )

        assert parsed["requirements"], "no requirements survived the chain"
        assert parsed["attribution"], "greenhouse postings owe attribution"
        assert 0.0 <= scored["coverage"] <= 1.0
        assert scored["requirements_considered"] == len(parsed["requirements"])
