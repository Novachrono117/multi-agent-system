"""Retry and backoff, and the transport translations.

Two things here. First, retry: it closes the gap that every other project in
this codebase has, and it is tested with an injected sleeper so the suite never
actually waits. Second, the transport translations - Anthropic's SDK shape and
Ollama's OpenAI-ish shape - checked with stub clients, so no key and no local
model are needed to know the mapping is right.
"""

from __future__ import annotations

from typing import Any

import pytest

from jobfit.errors import TransportError, TransportUnavailableError
from jobfit.llm.fake import FailingTransport, FakeTransport, text_turn
from jobfit.llm.retry import RetryingTransport, with_retry
from jobfit.llm.transport import (
    AnthropicTransport,
    OllamaTransport,
    TextBlock,
    ToolUseBlock,
)


def _send(transport: Any) -> Any:
    return transport.send(system="s", messages=[{"role": "user", "content": "hi"}])


class TestRetryBehaviour:
    def test_retryable_failure_is_retried_then_succeeds(self) -> None:
        inner = FailingTransport(
            TransportError("rate limited", retryable=True),
            times=2,
            then=FakeTransport([text_turn("recovered")]),
        )
        slept: list[float] = []
        transport = RetryingTransport(inner, sleeper=slept.append, jitter=lambda: 1.0)

        assert _send(transport).text == "recovered"
        assert inner.attempts == 3
        assert len(slept) == 2

    def test_a_non_retryable_failure_fails_immediately(self) -> None:
        """A 400 retried three times is three times the cost and the same answer."""
        inner = FailingTransport(TransportError("bad request", retryable=False), times=99)
        slept: list[float] = []
        transport = RetryingTransport(inner, sleeper=slept.append)

        with pytest.raises(TransportError, match="bad request"):
            _send(transport)
        assert inner.attempts == 1
        assert slept == []

    def test_exhausting_the_attempts_raises_with_the_count(self) -> None:
        inner = FailingTransport(TransportError("always down", retryable=True), times=99)
        transport = RetryingTransport(inner, attempts=3, sleeper=lambda _: None)

        with pytest.raises(TransportError, match="after 3 attempts"):
            _send(transport)
        assert inner.attempts == 3

    def test_backoff_grows_exponentially(self) -> None:
        inner = FailingTransport(TransportError("429", retryable=True), times=99)
        transport = RetryingTransport(
            inner, attempts=4, base_delay=1.0, sleeper=lambda _: None, jitter=lambda: 1.0
        )
        with pytest.raises(TransportError):
            _send(transport)
        # jitter fixed at 1.0 makes the multiplier exactly 1.
        assert transport.delays == [1.0, 2.0, 4.0]

    def test_backoff_is_capped(self) -> None:
        inner = FailingTransport(TransportError("429", retryable=True), times=99)
        transport = RetryingTransport(
            inner,
            attempts=6,
            base_delay=10.0,
            max_delay=15.0,
            sleeper=lambda _: None,
            jitter=lambda: 1.0,
        )
        with pytest.raises(TransportError):
            _send(transport)
        assert max(transport.delays) == 15.0

    def test_jitter_keeps_delays_inside_half_the_nominal_range(self) -> None:
        """Without jitter, concurrent runs retry in lockstep and re-throttle."""
        inner = FailingTransport(TransportError("429", retryable=True), times=99)
        transport = RetryingTransport(
            inner, attempts=4, base_delay=4.0, sleeper=lambda _: None, jitter=lambda: 0.0
        )
        with pytest.raises(TransportError):
            _send(transport)
        assert transport.delays == [2.0, 4.0, 8.0]

    def test_a_successful_call_is_not_retried(self) -> None:
        inner = FakeTransport([text_turn("fine")])
        transport = RetryingTransport(inner, sleeper=lambda _: None)
        assert _send(transport).text == "fine"
        assert inner.call_count == 1

    def test_retries_are_recorded_as_events(self, events: list[str]) -> None:
        inner = FailingTransport(
            TransportError("429", retryable=True), times=1, then=FakeTransport([text_turn("ok")])
        )
        _send(RetryingTransport(inner, sleeper=lambda _: None))
        assert any("transport_retry" in line for line in events)

    def test_with_retry_does_not_double_wrap(self) -> None:
        wrapped = with_retry(FakeTransport([text_turn("x")]))
        assert with_retry(wrapped) is wrapped

    def test_zero_attempts_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RetryingTransport(FakeTransport(), attempts=0)

    def test_the_wrapper_reports_the_inner_transport_name(self) -> None:
        assert with_retry(FakeTransport()).name == "fake"


# --------------------------------------------------------------------------
# Anthropic translation
# --------------------------------------------------------------------------


class _StubBlock:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _StubUsage:
    def __init__(self, input_tokens: int, output_tokens: int, cache: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache


class _StubResponse:
    def __init__(self, content: list[Any], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _StubUsage(120, 34, 90)
        self.model = "claude-sonnet-5"


class _StubStream:
    def __init__(self, response: _StubResponse) -> None:
        self._response = response

    def __enter__(self) -> _StubStream:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def get_final_message(self) -> _StubResponse:
        return self._response


class _StubMessages:
    def __init__(self, response: _StubResponse) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StubStream:
        self.requests.append(kwargs)
        return _StubStream(self._response)

    def create(self, **kwargs: Any) -> _StubResponse:
        self.requests.append(kwargs)
        return self._response


class _StubAnthropicClient:
    def __init__(self, response: _StubResponse) -> None:
        self.messages = _StubMessages(response)


class TestAnthropicTranslation:
    def test_missing_key_is_refused_with_a_free_alternative_named(self) -> None:
        """Failing loudly beats silently running somewhere else.

        A quiet fallback to a local model would produce a trace labelled
        "anthropic" that never touched Anthropic - the exact unverified claim
        this project exists to eliminate.
        """
        with pytest.raises(TransportUnavailableError) as excinfo:
            AnthropicTransport(api_key=None, model="claude-sonnet-5")
        assert "ollama" in str(excinfo.value)

    def test_text_and_tool_use_blocks_are_translated(self) -> None:
        response = _StubResponse(
            [
                _StubBlock(type="text", text="Let me check that."),
                _StubBlock(type="tool_use", id="toolu_1", name="parse_job_posting", input={"a": 1}),
            ],
            stop_reason="tool_use",
        )
        transport = AnthropicTransport(
            api_key=None, model="claude-sonnet-5", client=_StubAnthropicClient(response)
        )
        message = _send(transport)

        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ToolUseBlock)
        assert message.tool_uses[0].id == "toolu_1"
        assert message.tool_uses[0].input == {"a": 1}
        assert message.stop_reason == "tool_use"

    def test_usage_including_cache_reads_is_carried_through(self) -> None:
        transport = AnthropicTransport(
            api_key=None,
            model="claude-sonnet-5",
            client=_StubAnthropicClient(_StubResponse([_StubBlock(type="text", text="hi")])),
        )
        usage = _send(transport).usage
        assert usage.input_tokens == 120
        assert usage.output_tokens == 34
        assert usage.cache_read_tokens == 90

    def test_thinking_blocks_are_dropped(self) -> None:
        """They are not part of the answer and must not be replayed elsewhere."""
        response = _StubResponse(
            [
                _StubBlock(type="thinking", thinking="internal deliberation"),
                _StubBlock(type="text", text="the answer"),
            ]
        )
        transport = AnthropicTransport(
            api_key=None, model="claude-sonnet-5", client=_StubAnthropicClient(response)
        )
        message = _send(transport)
        assert len(message.content) == 1
        assert message.text == "the answer"

    def test_the_request_uses_adaptive_thinking_and_no_budget_tokens(self) -> None:
        """``budget_tokens`` was removed on current models and returns a 400."""
        client = _StubAnthropicClient(_StubResponse([_StubBlock(type="text", text="x")]))
        transport = AnthropicTransport(api_key=None, model="claude-sonnet-5", client=client)
        _send(transport)

        request = client.messages.requests[0]
        assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert "budget_tokens" not in str(request)
        assert request["model"] == "claude-sonnet-5"

    def test_thinking_display_is_summarized_so_a_stream_is_not_a_blank_screen(self) -> None:
        client = _StubAnthropicClient(_StubResponse([_StubBlock(type="text", text="x")]))
        _send(AnthropicTransport(api_key=None, model="claude-sonnet-5", client=client))
        assert client.messages.requests[0]["thinking"]["display"] == "summarized"

    def test_streaming_is_used_by_default(self) -> None:
        client = _StubAnthropicClient(_StubResponse([_StubBlock(type="text", text="x")]))
        transport = AnthropicTransport(api_key=None, model="claude-sonnet-5", client=client)
        transport.send(system="s", messages=[{"role": "user", "content": "hi"}], stream=True)
        assert client.messages.requests

    def test_no_tools_key_is_sent_when_there_are_no_tools(self) -> None:
        client = _StubAnthropicClient(_StubResponse([_StubBlock(type="text", text="x")]))
        _send(AnthropicTransport(api_key=None, model="claude-sonnet-5", client=client))
        assert "tools" not in client.messages.requests[0]

    def test_string_tool_input_is_parsed_as_json_not_matched_as_text(self) -> None:
        """Serialised escaping differs between models; parse, never string-match."""
        response = _StubResponse(
            [_StubBlock(type="tool_use", id="t1", name="x", input='{"posting_id": "a\\/b"}')],
            stop_reason="tool_use",
        )
        transport = AnthropicTransport(
            api_key=None, model="claude-sonnet-5", client=_StubAnthropicClient(response)
        )
        assert _send(transport).tool_uses[0].input == {"posting_id": "a/b"}


# --------------------------------------------------------------------------
# Ollama translation
# --------------------------------------------------------------------------


class _StubOllamaResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubOllamaClient:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any]) -> _StubOllamaResponse:
        self.requests.append({"url": url, "json": json})
        return _StubOllamaResponse(self.payload, self.status_code)


def _ollama(payload: dict[str, Any], status_code: int = 200) -> tuple[Any, _StubOllamaClient]:
    client = _StubOllamaClient(payload, status_code)
    return (
        OllamaTransport(base_url="http://127.0.0.1:11434", model="qwen3:8b", client=client),
        client,
    )


class TestOllamaTranslation:
    def test_plain_answer_is_translated(self) -> None:
        transport, _ = _ollama(
            {"message": {"content": "hello"}, "prompt_eval_count": 42, "eval_count": 7}
        )
        message = _send(transport)
        assert message.text == "hello"
        assert message.stop_reason == "end_turn"
        assert message.usage.input_tokens == 42
        assert message.usage.output_tokens == 7

    def test_think_blocks_are_stripped(self) -> None:
        """Qwen3 emits them inline; they are not the answer."""
        transport, _ = _ollama(
            {"message": {"content": "<think>hmm, let me see</think>The answer is 4"}}
        )
        assert _send(transport).text == "The answer is 4"

    def test_tool_calls_become_tool_use_blocks_with_synthetic_ids(self) -> None:
        """Ollama supplies no call id, and the loop needs one to pair results."""
        transport, _ = _ollama(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "parse_job_posting",
                                "arguments": {"posting_id": "x"},
                            }
                        }
                    ],
                }
            }
        )
        message = _send(transport)
        assert message.stop_reason == "tool_use"
        assert message.tool_uses[0].name == "parse_job_posting"
        assert message.tool_uses[0].id
        assert message.tool_uses[0].input == {"posting_id": "x"}

    def test_string_arguments_are_parsed(self) -> None:
        transport, _ = _ollama(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "t", "arguments": '{"a": 1}'}}],
                }
            }
        )
        assert _send(transport).tool_uses[0].input == {"a": 1}

    def test_unparseable_arguments_degrade_to_empty_rather_than_crashing(self) -> None:
        transport, _ = _ollama(
            {"message": {"tool_calls": [{"function": {"name": "t", "arguments": "{oops"}}]}}
        )
        assert _send(transport).tool_uses[0].input == {}

    def test_multiple_tool_calls_get_distinct_ids(self) -> None:
        transport, _ = _ollama(
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "a", "arguments": {}}},
                        {"function": {"name": "b", "arguments": {}}},
                    ]
                }
            }
        )
        ids = [call.id for call in _send(transport).tool_uses]
        assert len(set(ids)) == 2

    def test_the_system_prompt_becomes_a_system_message(self) -> None:
        transport, client = _ollama({"message": {"content": "ok"}})
        transport.send(system="be careful", messages=[{"role": "user", "content": "hi"}])
        sent = client.requests[0]["json"]["messages"]
        assert sent[0] == {"role": "system", "content": "be careful"}

    def test_anthropic_shaped_history_is_converted(self) -> None:
        """tool_result blocks become separate role:tool messages."""
        transport, client = _ollama({"message": {"content": "ok"}})
        transport.send(
            system="s",
            messages=[
                {"role": "user", "content": "assess it"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "t1", "name": "parse", "input": {"a": 1}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "parsed"}],
                },
            ],
        )
        sent = client.requests[0]["json"]["messages"]
        roles = [m["role"] for m in sent]
        assert roles[0] == "system"
        assert "tool" in roles
        assistant = next(m for m in sent if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["function"]["name"] == "parse"

    def test_tools_are_sent_in_openai_shape(self) -> None:
        """The formats genuinely differ; this is the translation earning its keep."""
        transport, client = _ollama({"message": {"content": "ok"}})
        from jobfit.tools import build_registry

        transport.send(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            tools=build_registry().ollama_tools(),
        )
        sent = client.requests[0]["json"]["tools"]
        assert sent[0]["type"] == "function"
        assert "parameters" in sent[0]["function"]

    def test_a_5xx_is_marked_retryable_and_a_4xx_is_not(self) -> None:
        transport, _ = _ollama({"error": "boom"}, status_code=503)
        with pytest.raises(TransportError) as server_error:
            _send(transport)
        assert server_error.value.retryable is True

        transport, _ = _ollama({"error": "nope"}, status_code=404)
        with pytest.raises(TransportError) as client_error:
            _send(transport)
        assert client_error.value.retryable is False

    def test_the_unreachable_message_says_how_to_fix_it(self) -> None:
        """The most likely failure for a first-time user is Ollama not running."""
        import httpx

        class _Dead:
            def post(self, url: str, json: dict[str, Any]) -> None:
                raise httpx.ConnectError("connection refused")

        transport = OllamaTransport(
            base_url="http://127.0.0.1:11434", model="qwen3:8b", client=_Dead()
        )
        with pytest.raises(TransportError) as excinfo:
            _send(transport)
        message = str(excinfo.value)
        assert "ollama serve" in message
        assert "qwen3:8b" in message
        assert excinfo.value.retryable is True
