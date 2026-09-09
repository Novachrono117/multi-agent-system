"""Getting a validated Pydantic object out of a model.

One code path for every transport, on purpose. The Claude API offers
``client.messages.parse(output_format=Model)``, which is stricter and better -
the server enforces the schema - but using it only on the Anthropic path would
mean two different behaviours to reason about and a fallback path that the test
suite never really exercises. Since the demo runs on a local model, the shared
path is the one that has to be correct, so it is the only one. (Switching the
Anthropic path to ``messages.parse`` is a named next step in the README, not a
pretence that it is already done.)

So: ask for JSON in the prompt, extract it robustly, validate with Pydantic,
and on failure retry exactly once with the validation error fed back. That
degradation ladder is ported from the author's Oralito code, along with the
habit of labelling *how* an answer was obtained instead of hiding it.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from jobfit.llm.transport import MessageTransport
from jobfit.models.trace import OutputSource
from jobfit.observability import emit
from jobfit.tools.text import strip_think

_FENCE_MARKERS = ("```json", "```JSON", "```")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of arbitrary model output.

    Three strategies in order, because models wrap JSON in different ways:

    1. the whole string parses;
    2. it is inside a fenced code block;
    3. brace counting from the first ``{`` to its match - which is what
       actually rescues a model that prefixed "Here is the assessment:".

    Brace counting is string-aware: a ``}`` inside a JSON string value must not
    close the object, and an escaped quote must not end the string.
    """
    text = strip_think(text).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    for marker in _FENCE_MARKERS:
        if marker in text:
            after = text.split(marker, 1)[1]
            body = after.split("```", 1)[0]
            try:
                parsed = json.loads(body.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def json_instructions(model: type[BaseModel]) -> str:
    """The JSON contract to append to a prompt.

    The schema is included rather than described, so adding a field to the model
    updates the prompt automatically - the same reason the tool schemas are
    generated instead of written out.
    """
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Reply with a single JSON object and nothing else. No prose before or "
        "after it, no code fence.\n\n"
        f"It must satisfy this JSON Schema:\n{schema}"
    )


def parse_into[T: BaseModel](
    model: type[T],
    *,
    transport: MessageTransport,
    system: str,
    user_message: str,
    max_tokens: int = 2048,
    retries: int = 1,
) -> tuple[T | None, OutputSource, str]:
    """Ask for one JSON object and validate it into ``model``.

    Returns the parsed object (or None), how it was obtained, and the raw text.
    ``OutputSource`` is returned rather than assumed so a caller can record that
    an answer was recovered on a second attempt instead of quietly presenting it
    as a clean first-pass result.

    No tools are offered here. The tool-using loop gathers evidence first and
    finishes; this is a separate, tool-free call that only formats. Trying to do
    both in one call means the structured output may not be the final block.
    """
    prompt = f"{user_message}\n\n{json_instructions(model)}"
    raw = ""

    for attempt in range(retries + 1):
        message = transport.send(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=max_tokens,
        )
        raw = message.text
        payload = extract_json_object(raw)

        if payload is not None:
            try:
                parsed = model.model_validate(payload)
            except ValidationError as exc:
                if attempt >= retries:
                    emit(
                        "structured_output_failed",
                        model=model.__name__,
                        reason="validation",
                        attempts=attempt + 1,
                    )
                    return None, OutputSource.DETERMINISTIC, raw
                # Feed the error back: models correct a named field error far
                # more reliably than a repeated bare request.
                prompt = (
                    f"{user_message}\n\n{json_instructions(model)}\n\n"
                    f"Your previous reply did not validate:\n"
                    f"{exc.errors(include_url=False)}\n"
                    "Return corrected JSON."
                )
                emit("structured_output_retry", model=model.__name__, reason="validation")
                continue
            else:
                # A synthetic transport is not a model, and must not be
                # labelled as one. MIXED marks an answer recovered on retry.
                if getattr(transport, "synthetic", False):
                    return parsed, OutputSource.DETERMINISTIC, raw
                source = OutputSource.LLM if attempt == 0 else OutputSource.MIXED
                return parsed, source, raw

        if attempt >= retries:
            emit(
                "structured_output_failed",
                model=model.__name__,
                reason="no_json",
                attempts=attempt + 1,
            )
            return None, OutputSource.DETERMINISTIC, raw

        prompt = (
            f"{user_message}\n\n{json_instructions(model)}\n\n"
            "Your previous reply contained no JSON object. Return only the JSON."
        )
        emit("structured_output_retry", model=model.__name__, reason="no_json")

    return None, OutputSource.DETERMINISTIC, raw
