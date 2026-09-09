"""Tool registry: definition, allowlist, and dispatch.

Three things live here on purpose.

**The allowlist.** Dispatch refuses any tool not explicitly permitted for the
run, before the handler is reached. That is the seed of milestone M6 (autonomy
control), and it is testable today.

**The error taxonomy boundary.** A tool that fails in an expected way becomes a
``tool_result`` with ``is_error: True`` and the model gets to react. A
programming bug propagates. See ``jobfit.errors`` for why that line matters.

**The context.** Handlers receive small ids and read payloads from
``ToolContext``. Passing a whole job description as a tool argument would make
the model retype the posting in output tokens - expensive, and a fresh
opportunity for it to mangle the text.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from jobfit.config import Settings
from jobfit.errors import (
    JobfitError,
    ToolExecutionError,
    is_programming_bug,
    reraise_if_bug,
)
from jobfit.models.posting import JobPosting
from jobfit.models.profile import CandidateProfile
from jobfit.tools.schema import to_anthropic_tool, to_ollama_tool


@dataclass
class ToolContext:
    """Per-run state handlers may read and append to."""

    settings: Settings
    profile: CandidateProfile
    postings: dict[str, JobPosting] = field(default_factory=dict)
    #: posting_id -> the Greenhouse board it was listed from. Remembered at
    #: search time so the model does not have to carry it back to us.
    board_tokens: dict[str, str] = field(default_factory=dict)
    http: httpx.Client | None = None

    def add_posting(self, posting: JobPosting) -> str:
        self.postings[posting.posting_id] = posting
        return posting.posting_id

    def require_posting(self, posting_id: str) -> JobPosting:
        try:
            return self.postings[posting_id]
        except KeyError:
            known = ", ".join(sorted(self.postings)) or "none yet"
            raise ToolExecutionError(
                f"unknown posting_id {posting_id!r}. Known postings: {known}"
            ) from None


Handler = Callable[[ToolContext, Any], Any]


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its schema, its handler, and whether it touches the network."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    touches_network: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one dispatch, ready to become a ``tool_result`` block."""

    content: str
    is_error: bool = False


def _serialise(value: Any) -> str:
    """Render a handler result as text for the model."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class ToolRegistry:
    """A set of tools plus the allowlist governing which may actually run."""

    def __init__(self, specs: Iterable[ToolSpec], *, allowed: Iterable[str] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {spec.name: spec for spec in specs}
        # None means "everything registered". An explicit set narrows it.
        self._allowed: frozenset[str] = (
            frozenset(self._specs) if allowed is None else frozenset(allowed)
        )
        unknown = self._allowed - set(self._specs)
        if unknown:
            raise JobfitError(f"allowlist names unregistered tools: {sorted(unknown)}")

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed

    def spec(self, name: str) -> ToolSpec:
        return self._specs[name]

    def is_allowed(self, name: str) -> bool:
        return name in self._allowed

    def _exposed(self) -> list[ToolSpec]:
        """Only allowed tools are described to the model.

        A tool the model cannot call should not be advertised - otherwise it
        wastes tokens planning around it and then gets refused.
        """
        return [self._specs[name] for name in sorted(self._allowed)]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [to_anthropic_tool(s.name, s.description, s.input_model) for s in self._exposed()]

    def ollama_tools(self) -> list[dict[str, Any]]:
        return [to_ollama_tool(s.name, s.description, s.input_model) for s in self._exposed()]

    def dispatch(self, name: str, raw_input: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Validate, authorise and run one tool call.

        Never raises for an expected failure - the model is told instead, so it
        can retry or change course. Programming bugs still propagate in strict
        mode.
        """
        if name not in self._specs:
            return ToolOutcome(
                f"Error: no such tool {name!r}. Available tools: {', '.join(self.names)}.",
                is_error=True,
            )
        if not self.is_allowed(name):
            return ToolOutcome(
                f"Error: tool {name!r} is not permitted in this run.",
                is_error=True,
            )

        spec = self._specs[name]
        try:
            parsed = spec.input_model.model_validate(dict(raw_input))
        except ValidationError as exc:
            return ToolOutcome(
                f"Error: invalid arguments for {name!r}: {exc.errors(include_url=False)}",
                is_error=True,
            )

        try:
            return ToolOutcome(_serialise(spec.handler(ctx, parsed)))
        except Exception as exc:
            reraise_if_bug(exc, strict=ctx.settings.strict)
            label = "bug" if is_programming_bug(exc) else type(exc).__name__
            return ToolOutcome(f"Error from {name!r} ({label}): {exc}", is_error=True)
