"""Turn a Pydantic model into a tool schema the API will actually accept.

``strict: true`` tool use is stricter than JSON Schema, and Pydantic's own
``model_json_schema()`` output violates it in four ways out of the box:

* it emits ``$defs`` and ``$ref`` for nested models - refs are not allowed;
* it emits ``default`` - not allowed;
* it omits optional fields from ``required`` - under strict, *every* property
  must be required, and optionality is expressed as a nullable union instead;
* it expresses ``X | None`` as ``anyOf`` - which must be collapsed to
  ``{"type": ["x", "null"]}``.

Hence this module. Ten minutes of work when planned for; half a day of
confusing 400s when discovered late.

It also holds the two wire formats, because they differ and the difference is
the whole reason the transport layer is pluggable: Anthropic puts the schema at
``input_schema`` on the tool, Ollama nests it at ``function.parameters``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Keys Pydantic emits that strict tool schemas reject or do not need.
_DROPPED_KEYS = frozenset({"default", "title", "$schema", "additionalProperties"})


class SchemaConversionError(ValueError):
    """The model cannot be expressed as a strict tool schema."""


def _inline_refs(node: Any, defs: dict[str, Any], stack: tuple[str, ...] = ()) -> Any:
    """Replace every ``$ref`` with the definition it points at.

    Recursive models are rejected rather than silently truncated: a tool whose
    input can nest arbitrarily deep is a tool the model will misuse.
    """
    if isinstance(node, list):
        return [_inline_refs(item, defs, stack) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        if name in stack:
            raise SchemaConversionError(f"recursive model {name!r} cannot be a strict tool schema")
        if name not in defs:
            raise SchemaConversionError(f"unresolved schema reference {ref!r}")
        merged = {k: v for k, v in node.items() if k != "$ref"}
        return {**_inline_refs(defs[name], defs, (*stack, name)), **merged}

    return {key: _inline_refs(value, defs, stack) for key, value in node.items()}


def _collapse_nullable(node: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``anyOf: [T, null]`` as ``type: [t, "null"]``.

    Anything more elaborate than a two-member nullable union is refused: a real
    polymorphic union in a tool input is a design smell, and guessing at one
    here would produce a schema the API rejects with a message that points
    nowhere useful.
    """
    options = node.get("anyOf")
    if not isinstance(options, list):
        return node

    non_null = [o for o in options if o.get("type") != "null"]
    has_null = len(non_null) < len(options)

    if len(non_null) != 1:
        raise SchemaConversionError(
            "only two-member nullable unions are supported in strict tool schemas; "
            f"got {len(options)} members"
        )

    collapsed = {k: v for k, v in node.items() if k != "anyOf"}
    inner = dict(non_null[0])
    inner_type = inner.pop("type", None)
    collapsed.update(inner)
    if inner_type is not None:
        collapsed["type"] = [inner_type, "null"] if has_null else inner_type
    return collapsed


def _harden(node: Any) -> Any:
    """Apply the strict rules to an already ref-free schema tree."""
    if isinstance(node, list):
        return [_harden(item) for item in node]
    if not isinstance(node, dict):
        return node

    node = _collapse_nullable(dict(node))
    out: dict[str, Any] = {k: _harden(v) for k, v in node.items() if k not in _DROPPED_KEYS}

    types = out.get("type")
    is_object = types == "object" or (isinstance(types, list) and "object" in types)
    if is_object or "properties" in out:
        properties = out.get("properties", {})
        out["properties"] = properties
        out["additionalProperties"] = False
        # Under strict, every property is required. Optionality lives in the type.
        out["required"] = list(properties)
    return out


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Strict-compatible JSON Schema for ``model``'s fields."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    return _harden(_inline_refs(raw, defs))


def to_anthropic_tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """Anthropic wire format. ``strict`` is a top-level field on the tool."""
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": strict_schema(model),
    }


def to_ollama_tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """Ollama (OpenAI-shaped) wire format: the schema nests under ``function``."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": strict_schema(model),
        },
    }
