"""The seam between the agent loop and whatever model is answering.

Everything above this module works in terms of ``MessageLike`` - content
blocks, a stop reason, token usage. Everything below it is provider-specific.
That is what lets the same loop run against the Claude API, against a local
model, or against a script in a test, and it is also what makes the test suite
meaningful: the fake plugs in *here*, below the loop, so the loop itself is
exercised rather than replaced.

Three implementations:

* ``AnthropicTransport`` - the Claude API via the official SDK, with streaming.
  The only thing in this package that can open a connection to Anthropic.
* ``OllamaTransport`` - a local model over HTTP. Free, and what the demo runs
  on. Its wire format is OpenAI-shaped rather than Anthropic-shaped, so it
  translates in both directions; that translation is the reason this seam earns
  its keep instead of being an abstraction for its own sake.
* ``FakeTransport`` - scripted turns, in ``fake.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from jobfit.errors import TransportError, TransportUnavailableError
from jobfit.models.trace import TokenUsage
from jobfit.tools.text import strip_think


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


ContentBlock = TextBlock | ToolUseBlock


@dataclass
class MessageLike:
    """A provider-neutral assistant turn.

    ``stop_reason`` follows the Anthropic vocabulary - ``end_turn``,
    ``tool_use``, ``max_tokens`` - because that is the vocabulary the loop is
    written against; other providers are translated into it.
    """

    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock)).strip()

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


@runtime_checkable
class MessageTransport(Protocol):
    """What the agent loop requires of a model backend.

    ``synthetic`` says whether a real model produced the answer. It exists so
    output can be labelled truthfully: an offline or scripted run must not be
    reported as model output. Getting this wrong was a real bug - the first
    offline demo printed "Output source: llm" with no model involved, which is
    precisely the kind of false label this project exists to remove.
    """

    name: str
    synthetic: bool

    def send(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> MessageLike: ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicTransport:
    """The Claude API through the official ``anthropic`` SDK.

    Written against SDK 1.x. Notes that are easy to get wrong and are pinned by
    tests in ``tests/test_transport_anthropic.py``:

    * the key is passed explicitly, never picked up from the ambient
      ``ANTHROPIC_API_KEY`` - an accidental environment variable must not be
      able to turn an offline run into a billed one;
    * streaming is the default, and ``get_final_message()`` gives the whole
      accumulated turn, so there is no reason not to stream;
    * ``thinking.display`` defaults to ``omitted``, which makes a streamed demo
      look like a frozen terminal - so ``summarized`` is requested explicitly;
    * ``budget_tokens`` was removed on current models and returns a 400. Depth
      is controlled with ``effort`` instead.
    """

    name = "anthropic"
    synthetic = False

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        effort: str = "medium",
        client: Any = None,
    ) -> None:
        if client is None and not api_key:
            raise TransportUnavailableError(
                "the Anthropic transport needs JOBFIT_ANTHROPIC_API_KEY. "
                "Set JOBFIT_TRANSPORT=ollama to run on a local model for free, "
                "or JOBFIT_TRANSPORT=fake for a scripted offline run."
            )
        self.model = model
        self.effort = effort
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Import and construct the SDK client lazily.

        Lazily so that merely importing this module - which the CLI always does
        - neither requires the SDK to be importable nor builds a client for a
        run that never touches Anthropic.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_message(response: Any) -> MessageLike:
        """Translate an SDK message into our neutral shape."""
        blocks: list[ContentBlock] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                blocks.append(TextBlock(text=block.text))
            elif kind == "tool_use":
                # Tool inputs are parsed JSON, never string-matched: escaping in
                # the serialised form differs between models.
                raw = block.input
                blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=raw if isinstance(raw, dict) else json.loads(raw),
                    )
                )
            # thinking blocks are intentionally dropped: they are not part of
            # the answer and must not be replayed to a different model.
        usage = getattr(response, "usage", None)
        return MessageLike(
            content=blocks,
            stop_reason=getattr(response, "stop_reason", None) or "end_turn",
            usage=TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
            model=getattr(response, "model", None),
        )

    def send(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> MessageLike:
        import anthropic

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
        }
        if tools:
            request["tools"] = tools

        client = self._get_client()
        try:
            if stream:
                with client.messages.stream(**request) as active:
                    return self._to_message(active.get_final_message())
            return self._to_message(client.messages.create(**request))
        except anthropic.APIStatusError as exc:
            # 429 and 5xx are worth another attempt; 4xx means the request is
            # wrong and retrying it only wastes money.
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise TransportError(
                f"Anthropic returned HTTP {exc.status_code}: {exc}", retryable=retryable
            ) from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise TransportError(f"Anthropic connection failed: {exc}", retryable=True) from exc
        except anthropic.APIError as exc:
            raise TransportError(f"Anthropic request failed: {exc}") from exc


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


class OllamaTransport:
    """A local model over Ollama's ``/api/chat``. Free, and unmetered.

    The wire format is OpenAI-shaped, so three translations happen here:

    * tools go out as ``{"type": "function", "function": {...}}`` rather than
      Anthropic's flat tool object;
    * tool calls come back on ``message.tool_calls`` with no id of their own, so
      ids are synthesised - the loop needs them to pair results with calls;
    * tool results are separate ``role: "tool"`` messages rather than
      ``tool_result`` blocks inside a user message, so an Anthropic-shaped
      history is converted on the way in.

    ``<think>`` blocks are stripped: reasoning models such as Qwen3 emit them
    inline and they are not part of the answer.
    """

    name = "ollama"
    synthetic = False

    def __init__(self, *, base_url: str, model: str, client: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            # Generous timeout: a local 8B model on a laptop GPU is not fast,
            # and a cold model has to be loaded into VRAM first.
            self._client = httpx.Client(timeout=300.0)
        return self._client

    @staticmethod
    def _to_ollama_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-shaped history into Ollama's flat message list."""
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            content = message.get("content")
            role = message.get("role", "user")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content or []:
                block_type = block.get("type") if isinstance(block, dict) else None
                if block_type == "text":
                    texts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": block.get("input") or {},
                            },
                        }
                    )
                elif block_type == "tool_result":
                    # Each result becomes its own tool message.
                    out.append(
                        {
                            "role": "tool",
                            "content": str(block.get("content", "")),
                        }
                    )

            if texts or tool_calls:
                entry: dict[str, Any] = {"role": role, "content": "\n".join(texts)}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                out.append(entry)
        return out

    def _to_message(self, payload: dict[str, Any]) -> MessageLike:
        message = payload.get("message") or {}
        blocks: list[ContentBlock] = []

        text = strip_think(message.get("content") or "")
        if text:
            blocks.append(TextBlock(text=text))

        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            blocks.append(
                ToolUseBlock(
                    # Ollama supplies no call id; the loop needs one to pair
                    # each result with its call.
                    id=f"ollama_{index}_{function.get('name', 'tool')}",
                    name=function.get("name") or "",
                    input=arguments if isinstance(arguments, dict) else {},
                )
            )

        has_tool_calls = any(isinstance(b, ToolUseBlock) for b in blocks)
        return MessageLike(
            content=blocks,
            stop_reason="tool_use" if has_tool_calls else "end_turn",
            usage=TokenUsage(
                input_tokens=int(payload.get("prompt_eval_count") or 0),
                output_tokens=int(payload.get("eval_count") or 0),
            ),
            model=payload.get("model") or self.model,
        )

    def send(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> MessageLike:
        import httpx

        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_ollama_messages(system, messages),
            # Non-streaming on the wire: the loop wants a whole turn, and
            # reassembling partial tool-call deltas would buy nothing here.
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        if tools:
            request["tools"] = tools

        try:
            response = self._get_client().post(f"{self.base_url}/api/chat", json=request)
        except httpx.HTTPError as exc:
            raise TransportError(
                f"could not reach Ollama at {self.base_url}: {exc}. "
                "Is it running? Start it with `ollama serve` and pull a model "
                "that supports tools, e.g. `ollama pull qwen3:8b`.",
                retryable=True,
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise TransportError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}",
                retryable=response.status_code >= 500,
            )
        try:
            return self._to_message(response.json())
        except ValueError as exc:
            raise TransportError("Ollama returned a body that is not JSON") from exc
