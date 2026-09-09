"""The agentic loop, written by hand.

This is the artefact. The SDK ships ``client.beta.messages.tool_runner``, which
would drive this loop for free, and it was not used. Reasons, in order of
weight:

1. **The loop is the thing being demonstrated.** Hiding it behind a helper hides
   the evidence.
2. **It is where control belongs.** The step ceiling, the tool allowlist, the
   per-step trace, and later the dry-run and human-approval gates all live at
   this level. Bolting them onto a helper's callbacks is harder than owning
   twelve lines of loop.
3. **The runner is beta**, and the SDK's own documentation lists avoiding a beta
   dependency as a legitimate reason to write the loop.
4. **The Python runner does not resume ``pause_turn``** - a paused turn ends the
   loop and is returned as the final message, with no error and no warning. That
   is a silently truncated answer; here it is handled explicitly.

The contract that is easy to get wrong, and which the tests pin:

* every ``tool_result`` for one assistant turn goes back in a **single** user
  message. Splitting them across messages teaches the model to stop making
  parallel tool calls;
* each ``tool_result`` carries the ``tool_use_id`` of the call it answers;
* a tool that fails still returns a result, with ``is_error: True``. Dropping it
  leaves the model waiting for an answer that never comes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jobfit.errors import StepCapExceededError
from jobfit.llm.transport import MessageLike, MessageTransport, TextBlock, ToolUseBlock
from jobfit.models.trace import OutputSource, StepKind, StepTrace, TokenUsage
from jobfit.observability import emit
from jobfit.tools.registry import ToolContext, ToolRegistry

#: Reasons a turn can finish. ``step_cap`` is ours, not the API's.
STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_PAUSE_TURN = "pause_turn"
STOP_STEP_CAP = "step_cap"


@dataclass
class TurnResult:
    """Everything one agent turn produced."""

    message: MessageLike
    messages: list[dict[str, Any]] = field(default_factory=list)
    trace: list[StepTrace] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    steps_used: int = 0
    stop_reason: str = STOP_END_TURN
    tool_errors: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def hit_step_cap(self) -> bool:
        return self.stop_reason == STOP_STEP_CAP


def _blocks_to_wire(message: MessageLike) -> list[dict[str, Any]]:
    """Serialise an assistant turn back into request content."""
    wire: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                wire.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            wire.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
    return wire


def run_agent_turn(
    *,
    transport: MessageTransport,
    registry: ToolRegistry,
    ctx: ToolContext,
    system: str,
    user_message: str,
    run_id: str,
    node: str,
    max_steps: int | None = None,
    max_tokens: int = 4096,
    seq_start: int = 0,
    use_tools: bool = True,
) -> TurnResult:
    """Drive one agent turn to completion, executing tools along the way.

    Returns rather than raises when the step ceiling is reached: a capped run is
    a legitimate outcome with a recorded reason, not a crash. The one thing that
    does propagate is a programming bug in a tool under strict mode - see
    ``jobfit.errors``.
    """
    cap = max_steps if max_steps is not None else ctx.settings.max_tool_steps
    tools = registry.anthropic_tools() if use_tools else None

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    trace: list[StepTrace] = []
    total_usage = TokenUsage()
    tool_errors: list[str] = []
    seq = seq_start
    steps = 0
    message: MessageLike | None = None

    while True:
        started = datetime.now(UTC)
        message = transport.send(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        latency_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
        total_usage = total_usage + message.usage

        trace.append(
            StepTrace(
                run_id=run_id,
                seq=seq,
                node=node,
                kind=StepKind.LLM_CALL,
                name=getattr(transport, "name", None),
                started_at=started,
                latency_ms=latency_ms,
                usage=message.usage,
                model=message.model,
                transport=getattr(transport, "name", None),
                stop_reason=message.stop_reason,
                source=OutputSource.LLM,
            )
        )
        seq += 1

        # A server-side tool paused the turn. Resending the paused assistant
        # message continues it. The SDK's beta tool runner does NOT do this and
        # exits with a truncated answer instead.
        if message.stop_reason == STOP_PAUSE_TURN:
            messages.append({"role": "assistant", "content": _blocks_to_wire(message)})
            steps += 1
            if steps >= cap:
                emit("step_cap_reached", run_id=run_id, node=node, cap=cap, cause="pause_turn")
                return TurnResult(
                    message=message,
                    messages=messages,
                    trace=trace,
                    usage=total_usage,
                    steps_used=steps,
                    stop_reason=STOP_STEP_CAP,
                    tool_errors=tool_errors,
                )
            continue

        tool_uses = message.tool_uses
        if not tool_uses:
            return TurnResult(
                message=message,
                messages=messages,
                trace=trace,
                usage=total_usage,
                steps_used=steps,
                stop_reason=message.stop_reason or STOP_END_TURN,
                tool_errors=tool_errors,
            )

        steps += 1
        if steps > cap:
            # The model still wants tools but has spent its budget. Report the
            # cap instead of running them; the caller decides what to do.
            emit("step_cap_reached", run_id=run_id, node=node, cap=cap, cause="tool_use")
            return TurnResult(
                message=message,
                messages=messages,
                trace=trace,
                usage=total_usage,
                steps_used=steps - 1,
                stop_reason=STOP_STEP_CAP,
                tool_errors=tool_errors,
            )

        messages.append({"role": "assistant", "content": _blocks_to_wire(message)})

        # Execute every requested tool, then return all results together.
        results: list[dict[str, Any]] = []
        for call in tool_uses:
            tool_started = datetime.now(UTC)
            outcome = registry.dispatch(call.name, call.input, ctx)
            tool_latency = int((datetime.now(UTC) - tool_started).total_seconds() * 1000)

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            )
            if outcome.is_error:
                tool_errors.append(f"{call.name}: {outcome.content[:200]}")

            trace.append(
                StepTrace(
                    run_id=run_id,
                    seq=seq,
                    node=node,
                    kind=StepKind.TOOL_CALL,
                    name=call.name,
                    started_at=tool_started,
                    latency_ms=tool_latency,
                    ok=not outcome.is_error,
                    error=outcome.content[:300] if outcome.is_error else None,
                    source=OutputSource.DETERMINISTIC,
                )
            )
            seq += 1
            emit(
                "tool_call",
                run_id=run_id,
                node=node,
                tool=call.name,
                ok=not outcome.is_error,
                latency_ms=tool_latency,
            )

        # One user message carrying every result. This is the contract.
        messages.append({"role": "user", "content": results})


def require_text(result: TurnResult) -> str:
    """The turn's text, or a raised error if it produced none.

    Used where a caller genuinely cannot continue without prose - the graph
    prefers to degrade, so most callers do not use this.
    """
    if not result.text:
        raise StepCapExceededError(result.steps_used, kind="text")
    return result.text
