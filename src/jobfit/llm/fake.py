"""A scripted transport, used by the test suite and by ``--offline``.

It plugs in *below* the agent loop, at ``MessageTransport.send``. That placement
is the whole point: if the fake implemented a full turn - tool calls, results,
final answer - then the loop, which is the artefact this repository exists to
demonstrate, would never actually run in a test. Faking the transport means the
tests exercise real ``tool_use`` extraction, real result batching, real error
handling and the real step ceiling.

It doubles as the ``--offline`` demo path, so the pipeline can be shown end to
end with no model, no key, and no network.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from jobfit.errors import TransportError
from jobfit.llm.transport import ContentBlock, MessageLike, TextBlock, ToolUseBlock
from jobfit.models.trace import TokenUsage

#: A scripted turn is either a ready-made message or a callable that receives
#: the conversation so far and returns one.
Turn = MessageLike | Callable[[list[dict[str, Any]]], MessageLike]


def text_turn(text: str, **usage: int) -> MessageLike:
    """A turn that just answers."""
    return MessageLike(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(**usage) if usage else TokenUsage(input_tokens=10, output_tokens=5),
        model="fake",
    )


def tool_turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> MessageLike:
    """A turn that requests one or more tools.

    Several calls in one turn is the interesting case: the API contract says all
    of their results must come back in a *single* user message, and splitting
    them teaches the model to stop asking for parallel calls.
    """
    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(text=text))
    for index, (name, arguments) in enumerate(calls):
        blocks.append(ToolUseBlock(id=f"fake_call_{index}", name=name, input=arguments))
    return MessageLike(
        content=blocks,
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=20, output_tokens=8),
        model="fake",
    )


class FakeTransport:
    """Replays scripted turns and records what it was asked."""

    name = "fake"
    synthetic = True

    def __init__(
        self,
        turns: Iterable[Turn] | None = None,
        *,
        repeat_last: bool = False,
    ) -> None:
        self._turns: list[Turn] = list(turns or [])
        # When set, the final turn repeats forever - the way to test a runaway
        # agent hitting the step ceiling.
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_messages(self) -> list[dict[str, Any]]:
        """The message history as it was on the most recent call."""
        if not self.calls:
            raise AssertionError("the transport was never called")
        return self.calls[-1]["messages"]

    def send(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> MessageLike:
        index = len(self.calls)
        self.calls.append(
            {
                "system": system,
                # Copied: the loop keeps mutating its own list, and a test that
                # inspects history after the fact must see this call's state.
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "max_tokens": max_tokens,
                "stream": stream,
            }
        )

        if index < len(self._turns):
            turn = self._turns[index]
        elif self.repeat_last and self._turns:
            turn = self._turns[-1]
        else:
            raise TransportError(
                f"FakeTransport ran out of scripted turns at call {index + 1} "
                f"(it has {len(self._turns)}). Add another turn, or pass "
                "repeat_last=True to let the last one repeat."
            )

        return turn(messages) if callable(turn) else turn


class FailingTransport:
    """Raises a given exception a number of times, then delegates.

    Used to test retry and backoff without waiting for a real rate limit.
    """

    name = "failing"
    synthetic = True

    def __init__(self, error: BaseException, *, times: int, then: Any = None) -> None:
        self.error = error
        self.times = times
        self.then = then
        self.attempts = 0

    def send(self, **kwargs: Any) -> MessageLike:
        self.attempts += 1
        if self.attempts <= self.times:
            raise self.error
        if self.then is None:
            return text_turn("recovered")
        return self.then.send(**kwargs)


class OfflineDemoTransport:
    """Answers plausibly for each node, with no model and no network.

    This is what ``jobfit run --offline`` uses. It exists so the pipeline can be
    demonstrated on a machine with no API key, no local model and no internet -
    which is also how the tests keep the graph honest.

    It routes on the *calling node*, read from the system prompt, rather than on
    call order: order would make the demo pass even if the supervisor sent work
    to the wrong specialist.

    It is explicitly not a model. Its assessment prose is fixed text; the
    coverage number in the output is still computed by the real deterministic
    scorer, so an offline run shows real measurement and canned language.
    """

    name = "offline"
    synthetic = True

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._screener_turns = 0

    @staticmethod
    def _role(system: str) -> str:
        if system.startswith("You are the supervisor"):
            return "supervisor"
        if system.startswith("You assess how well"):
            return "screener"
        if system.startswith("You convert findings"):
            return "assessment"
        if system.startswith("You write a role brief"):
            return "writer"
        return "unknown"

    #: A posting id is "<source>:<external_id>". Matching a bare colon instead
    #: picked up the word "candidate:" from the phrase "for the candidate:
    #: pasted:pasted-1", so every offline tool call was made against a
    #: nonexistent posting. The output still looked fine, because the score is
    #: recomputed in the node - the run trace is what exposed it.
    _POSTING_ID_RE = re.compile(r"\b(greenhouse|remotive|arbeitnow|pasted):[A-Za-z0-9._-]+\b")

    @classmethod
    def _first_posting_id(cls, text: str) -> str:
        """Pull a posting id out of the turn, so the stub targets a real one."""
        match = cls._POSTING_ID_RE.search(text)
        return match.group(0) if match else "pasted:pasted-1"

    def send(self, **kwargs: Any) -> MessageLike:
        import json

        system = str(kwargs.get("system", ""))
        messages = kwargs.get("messages") or []
        turn = str(messages[0].get("content", "")) if messages else ""
        role = self._role(system)
        self.calls.append(role)
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        posting_id = self._first_posting_id(turn)

        if role == "supervisor":
            lowered = turn.casefold()
            if "(none yet)" in lowered:
                payload = {
                    "next_agent": "fit_screener",
                    "target_posting_id": posting_id,
                    "reason": "offline demo: no assessment for this posting yet",
                }
            elif "brief written: no" in lowered and "no_fit" not in lowered:
                payload = {
                    "next_agent": "brief_writer",
                    "target_posting_id": posting_id,
                    "reason": "offline demo: assessment looks worth a brief",
                }
            else:
                payload = {
                    "next_agent": "finalize",
                    "target_posting_id": None,
                    "reason": "offline demo: nothing further to do",
                }
            return MessageLike(content=[TextBlock(text=json.dumps(payload))], usage=usage)

        if role == "screener":
            self._screener_turns += 1
            if self._screener_turns == 1:
                return MessageLike(
                    content=[
                        TextBlock(text="Reading the posting and measuring coverage."),
                        ToolUseBlock(
                            id="offline_parse",
                            name="parse_job_posting",
                            input={"posting_id": posting_id},
                        ),
                        ToolUseBlock(
                            id="offline_score",
                            name="score_profile_match",
                            input={"posting_id": posting_id},
                        ),
                    ],
                    stop_reason="tool_use",
                    usage=usage,
                )
            return MessageLike(
                content=[
                    TextBlock(
                        text="Offline demo: the tools ran and the deterministic "
                        "coverage was measured. This summary is canned text, not "
                        "model output."
                    )
                ],
                usage=usage,
            )

        if role == "assessment":
            payload = {
                "posting_id": posting_id,
                "verdict": "worth_applying",
                "checks": [],
                "reasoning": (
                    "Offline demo. The verdict is fixed text; the coverage "
                    "attached to it is a real measurement from score_profile_match."
                ),
                "needs_more_evidence": False,
            }
            return MessageLike(content=[TextBlock(text=json.dumps(payload))], usage=usage)

        if role == "writer":
            payload = {
                "posting_id": posting_id,
                "role_summary": (
                    "Offline demo brief. Run with a real transport "
                    "(JOBFIT_TRANSPORT=ollama or =anthropic) for actual analysis."
                ),
                "why_fit": ["Placeholder: no model produced this text."],
                "gaps": ["Placeholder: no model produced this text."],
                "not_claimable": [],
                "questions_for_recruiter": ["Placeholder."],
                "prep_topics": ["Placeholder."],
            }
            return MessageLike(content=[TextBlock(text=json.dumps(payload))], usage=usage)

        return MessageLike(content=[TextBlock(text="{}")], usage=usage)
