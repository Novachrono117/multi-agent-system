"""The only place in this package that opens an outbound connection.

Two responsibilities, deliberately fused, because both are promises this
project makes out loud and neither should be enforceable in more than one
place:

**The host allowlist.** Every request is checked against a fixed set of hosts
*before* the socket is opened. This is what makes "no scraping of boards that
forbid it" a property you can test rather than a sentence in a README. Indeed
and LinkedIn are not on the list and cannot be added by a model - the list is
code, not configuration, and not a tool argument.

**The cache.** Remotive's own API response carries a legal notice asking for at
most roughly four requests per day, plus attribution with a link back. A cache
with a TTL is how that is honoured. It also makes development cheap and keeps a
demo from depending on a board being up.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from jobfit.errors import DisallowedHostError, ToolExecutionError
from jobfit.observability import emit

#: Hosts this package may talk to. Every one exposes a public, documented job
#: board API that permits programmatic reads.
#:
#: Absent on purpose: Indeed and LinkedIn. Their terms forbid this, scraping
#: them breaks on every markup change, and a public repository that does it is
#: visible to exactly the people it is meant to impress.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "boards-api.greenhouse.io",
        "remotive.com",
        "www.arbeitnow.com",
    }
)

#: Refuse absurd payloads rather than feeding them to a parser. Measured: the
#: Greenhouse listing endpoint returns 8.5 MB when asked for full content, which
#: is why the tools never ask for it.
MAX_RESPONSE_BYTES = 4_000_000

DEFAULT_TIMEOUT_SECONDS = 20.0

_USER_AGENT = (
    "job-fit-agents/0.1 (+https://github.com/Novachrono117/job-fit-agents) "
    "personal job-search decision support; contact via GitHub"
)


def assert_host_allowed(url: str) -> str:
    """Return the host, or raise ``DisallowedHostError`` without connecting.

    Raising before any I/O is the point: the test for this asserts that zero
    requests were made, not merely that an error came back.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise DisallowedHostError(f"only https is allowed, got {parsed.scheme or 'no scheme'!r}")
    host = (parsed.hostname or "").casefold()
    if host not in ALLOWED_HOSTS:
        raise DisallowedHostError(
            f"host {host or '(none)'!r} is not on the allowlist. "
            f"Permitted hosts: {', '.join(sorted(ALLOWED_HOSTS))}. "
            "Job boards whose terms forbid programmatic access are excluded by design."
        )
    return host


class CachedFetcher:
    """Fetch JSON over HTTPS, allowlisted and cached on disk."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        ttl_seconds: int,
        client: httpx.Client | None = None,
        now: object = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self._client = client
        self._owns_client = client is None
        # Injectable clock so cache expiry can be tested without sleeping.
        self._now = now if callable(now) else time.time

    def _get_client(self) -> httpx.Client:
        """Create the HTTP client lazily.

        Lazily, so that constructing a fetcher - which every run does - never
        opens a connection pool for a run that turns out to be fully cached or
        fully offline.
        """
        if self._client is None:
            self._client = httpx.Client(
                timeout=DEFAULT_TIMEOUT_SECONDS,
                follow_redirects=False,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            )
        return self._client

    def http_client(self) -> httpx.Client:
        """The underlying client, for callers that share one across tools."""
        return self._get_client()

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> CachedFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + ("?" + urlencode(sorted((params or {}).items())) if params else "")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        host = urlparse(url).hostname or "unknown"
        return self.cache_dir / host / f"{digest}.json"

    def _read_cache(self, path: Path) -> Any | None:
        """Return the cached payload if it exists and is still fresh.

        Freshness is judged against ``cached_at`` inside the envelope, not the
        file's mtime. Using mtime looked simpler and was wrong: it is the real
        filesystem clock, so an injected test clock could never expire an entry
        - the age came out hugely negative and every entry read as fresh. It is
        also fragile in production, since copying a cache directory rewrites
        mtimes.
        """
        if self.ttl_seconds <= 0 or not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            cached_at = float(envelope["cached_at"])
            payload = envelope["payload"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A corrupt or old-format entry is not worth failing a run over.
            return None
        if self._now() - cached_at > self.ttl_seconds:
            return None
        return payload

    def _write_cache(self, path: Path, payload: Any) -> None:
        envelope = {
            "cached_at": self._now(),
            "url": str(path.name),
            "payload": payload,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk problems are not our story
            emit("cache_write_failed", path=path, error=str(exc))

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET JSON, honouring the allowlist and the cache.

        Raises ``DisallowedHostError`` for a forbidden host and
        ``ToolExecutionError`` for anything the network or the board does wrong -
        both of which the dispatcher turns into a tool error the model can read.
        """
        host = assert_host_allowed(url)
        path = self._cache_path(url, params)

        cached = self._read_cache(path)
        if cached is not None:
            emit("http_cache_hit", host=host, url=url)
            return cached

        emit("http_request", host=host, url=url, params=params or {})
        try:
            response = self._get_client().get(url, params=params)
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"request to {host} failed: {exc}") from exc

        if response.status_code != httpx.codes.OK:
            raise ToolExecutionError(f"{host} returned HTTP {response.status_code} for {url}")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise ToolExecutionError(
                f"{host} returned {len(response.content):,} bytes, over the "
                f"{MAX_RESPONSE_BYTES:,} byte limit"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ToolExecutionError(f"{host} returned a body that is not JSON") from exc

        self._write_cache(path, payload)
        emit("http_response", host=host, url=url, bytes=len(response.content))
        return payload
