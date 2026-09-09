"""Structured run events, as one JSON object per line on stderr.

Thirty lines instead of a ``structlog`` dependency, on purpose: this is where
milestone M5 (tracing, cost accounting, eval) lands, and it should land on
something we own rather than on somebody's logger configuration.

stderr, not stdout, so the CLI can stream a brief to stdout while events flow
alongside it: ``jobfit run ... > brief.md 2> trace.jsonl``.

The module is deliberately named ``observability`` rather than ``logging`` - the
latter invites ``from . import logging`` to shadow the stdlib module.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

Sink = Callable[[str], None]


def _default_sink(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


_sink: Sink = _default_sink


def set_sink(sink: Sink | None) -> None:
    """Redirect events. Passing ``None`` restores the stderr default.

    Tests use this to collect events in a list; the CLI uses it to write a run
    trace to a file.
    """
    global _sink
    _sink = sink or _default_sink


def _encode(obj: Any) -> Any:
    """Make the odd non-JSON value serialisable instead of raising.

    An event that cannot be written is worse than an event with a coarse value,
    so this never raises.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):  # pydantic BaseModel
        return obj.model_dump(mode="json")
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    return repr(obj)


def emit(event: str, **fields: Any) -> None:
    """Write one event. ``event`` is the name; everything else is payload.

    Keys are sorted so two runs of the same pipeline produce diffable output.
    """
    record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    _sink(json.dumps(record, default=_encode, ensure_ascii=False, sort_keys=True))


class FileSink:
    """Context manager writing events to a file, one JSON object per line."""

    def __init__(self, path: Path, *, also_stderr: bool = False) -> None:
        self.path = path
        self.also_stderr = also_stderr
        self._fh: TextIO | None = None

    def __enter__(self) -> FileSink:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        set_sink(self._write)
        return self

    def _write(self, line: str) -> None:
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.also_stderr:
            _default_sink(line)

    def __exit__(self, *exc: object) -> None:
        set_sink(None)
        if self._fh is not None:
            self._fh.close()
            self._fh = None
