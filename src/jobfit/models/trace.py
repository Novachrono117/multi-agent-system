"""Run trace primitives.

These exist in the MVP even though nothing consumes them yet. That is
deliberate: milestone M5 (observability and eval) is then "write the sink and
the aggregation", one new module. If tracing were bolted on later, the M5 diff
would touch every node in the graph - and that is precisely where projects like
this one die.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OutputSource(StrEnum):
    """Where a piece of output actually came from.

    Ported from the ``_source`` field in the author's Oralito code. It is what
    keeps a degraded answer from being presented as a model answer - and here it
    also separates the deterministic score from the model's prose.
    """

    LLM = "llm"
    DETERMINISTIC = "deterministic"
    MIXED = "mixed"


class StepKind(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    DECISION = "decision"


class TokenUsage(BaseModel):
    """Token counts for one call, addable so a run can be totalled."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        """Cost of this call at the given per-million-token rates.

        Rates are passed in rather than hardcoded: they differ per model and
        change over time, and a stale constant in the repo would be the sort of
        unverified claim this project exists to avoid.
        """
        return (
            self.input_tokens / 1_000_000 * input_per_mtok
            + self.output_tokens / 1_000_000 * output_per_mtok
        )


class StepTrace(BaseModel):
    """One recorded step: a model call, a tool call, or a routing decision."""

    run_id: str
    seq: int
    node: str
    kind: StepKind
    name: str | None = None
    started_at: datetime
    latency_ms: int
    usage: TokenUsage | None = None
    model: str | None = None
    transport: str | None = None
    stop_reason: str | None = None
    ok: bool = True
    error: str | None = None
    source: OutputSource = OutputSource.LLM


class Handoff(BaseModel):
    """A supervisor routing decision, with the reason it gave.

    The reason is what makes "multiple agents collaborating" checkable rather
    than rhetorical: you can read why control moved, not just that it did.
    """

    seq: int
    from_node: str
    to_node: str
    reason: str


class StepError(BaseModel):
    node: str
    kind: str
    message: str
    recoverable: bool = True


def sum_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage:
    """LangGraph reducer that totals token usage across nodes."""
    base = left or TokenUsage()
    return base + right if right else base


class RunOutcome(BaseModel):
    """How a run ended. Every stop has a named reason, including the caps."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="completed | step_cap | no_fit | failed")
    reason: str
    steps_used: int
    usage: TokenUsage = TokenUsage()
