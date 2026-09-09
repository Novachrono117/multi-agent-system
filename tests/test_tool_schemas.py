"""Every tool schema must satisfy the strict-tool-use contract.

This is a sweep over the registry rather than a per-tool test, in the style of
the author's prompt-contract test: a tool added later is checked automatically,
which is the only way a rule like this survives contact with a growing codebase.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from jobfit.tools import ALL_SPECS, build_registry
from jobfit.tools.schema import (
    SchemaConversionError,
    strict_schema,
    to_anthropic_tool,
    to_ollama_tool,
)

TOOL_NAMES = [spec.name for spec in ALL_SPECS]


def _objects(node: Any) -> list[dict[str, Any]]:
    """Every object-shaped subschema in the tree, so nesting is checked too."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_objects(item))
    return found


class TestStrictContract:
    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_schema_obeys_strict_rules_at_every_level(self, name: str) -> None:
        registry = build_registry()
        schema = strict_schema(registry.spec(name).input_model)
        objects = _objects(schema)
        assert objects, f"{name} has no object schema"
        for obj in objects:
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"]), (
                "under strict tool use every property must be required; "
                "optionality is expressed as a nullable type"
            )

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_schema_has_no_refs_defaults_or_anyof(self, name: str) -> None:
        """All three are emitted by Pydantic and rejected by strict tool use."""
        registry = build_registry()
        blob = json.dumps(strict_schema(registry.spec(name).input_model))
        assert "$ref" not in blob
        assert "$defs" not in blob
        assert '"default"' not in blob
        assert "anyOf" not in blob

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_every_tool_and_field_is_documented(self, name: str) -> None:
        """An undescribed tool or argument is one the model will misuse."""
        spec = build_registry().spec(name)
        assert len(spec.description) > 40, f"{name} needs a real description"
        for field, subschema in strict_schema(spec.input_model)["properties"].items():
            assert subschema.get("description"), f"{name}.{field} has no description"


class TestRegistryIntegrity:
    def test_anthropic_format_marks_tools_strict(self) -> None:
        for tool in build_registry().anthropic_tools():
            assert tool["strict"] is True
            assert set(tool) == {"name", "description", "strict", "input_schema"}

    def test_ollama_format_nests_the_schema_under_function(self) -> None:
        for tool in build_registry().ollama_tools():
            assert tool["type"] == "function"
            assert set(tool["function"]) == {"name", "description", "parameters"}

    def test_both_wire_formats_describe_the_same_arguments(self) -> None:
        """Transport equivalence: swapping providers must not change the contract.

        The formats genuinely differ - Anthropic puts the schema at
        ``input_schema``, Ollama at ``function.parameters`` - so this is the test
        that keeps the translation honest.
        """
        registry = build_registry()
        anthropic = {t["name"]: t["input_schema"] for t in registry.anthropic_tools()}
        ollama = {
            t["function"]["name"]: t["function"]["parameters"] for t in registry.ollama_tools()
        }
        assert anthropic.keys() == ollama.keys()
        for name, schema in anthropic.items():
            assert schema == ollama[name]

    def test_every_registered_tool_is_dispatchable_and_vice_versa(self) -> None:
        registry = build_registry()
        assert set(registry.names) == {spec.name for spec in ALL_SPECS}
        assert registry.allowed == frozenset(registry.names)

    def test_tool_names_are_snake_case(self) -> None:
        for name in build_registry().names:
            assert name.islower()
            assert " " not in name and "-" not in name

    def test_allowlist_naming_an_unknown_tool_is_rejected_at_construction(self) -> None:
        """Fail when the registry is built, not on the call that needed the tool."""
        with pytest.raises(Exception, match="unregistered"):
            build_registry(allowed=["parse_job_posting", "definitely_not_a_tool"])


class TestSchemaConversionEdgeCases:
    def test_recursive_model_is_refused_not_silently_truncated(self) -> None:
        class Recursive(BaseModel):
            child: Recursive | None = None

        Recursive.model_rebuild()
        with pytest.raises(SchemaConversionError, match="recursive"):
            strict_schema(Recursive)

    def test_optional_field_becomes_a_nullable_type(self) -> None:
        class WithOptional(BaseModel):
            required_one: str
            optional_one: str | None = None

        schema = strict_schema(WithOptional)
        assert schema["properties"]["optional_one"]["type"] == ["string", "null"]
        assert set(schema["required"]) == {"required_one", "optional_one"}

    def test_nested_model_is_inlined_and_hardened(self) -> None:
        class Inner(BaseModel):
            depth: int

        class Outer(BaseModel):
            inner: Inner

        inner_schema = strict_schema(Outer)["properties"]["inner"]
        assert inner_schema["properties"]["depth"]["type"] == "integer"
        assert inner_schema["additionalProperties"] is False

    def test_multi_member_union_is_refused_with_a_useful_message(self) -> None:
        class Polymorphic(BaseModel):
            value: int | str | None = None

        with pytest.raises(SchemaConversionError, match="nullable unions"):
            strict_schema(Polymorphic)

    def test_field_default_is_dropped_but_field_survives(self) -> None:
        class WithDefault(BaseModel):
            limit: int = 10

        schema = strict_schema(WithDefault)
        assert "default" not in schema["properties"]["limit"]
        assert schema["required"] == ["limit"]

    def test_both_formatters_accept_the_same_model(self) -> None:
        class Simple(BaseModel):
            query: str

        assert (
            to_anthropic_tool("t", "d", Simple)["input_schema"]
            == (to_ollama_tool("t", "d", Simple)["function"]["parameters"])
        )
