"""Prompt contract, checked by introspection.

Ported from the author's Oralito test suite: every ``Template`` in
``jobfit.prompts`` is discovered from the module namespace, so a prompt added
later is covered without anyone remembering to add a test.

Two of these are unusual and are the reason this file matters more than most.
``test_no_prompt_body_tells_the_model_to_submit`` turns the product's central
promise - the system never submits an application - into an assertion. And
``test_output_prompts_carry_the_no_submit_clause`` stops that promise from being
quietly dropped from a prompt during a refactor.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from string import Template

import pytest

from jobfit import prompts
from jobfit.prompts import clause_kwargs

PROMPTS_FILE = Path("src/jobfit/prompts.py")

#: Every Template in the module, discovered rather than listed.
ALL_TEMPLATES: dict[str, Template] = {
    name: value for name, value in vars(prompts).items() if isinstance(value, Template)
}

#: Prompts that produce candidate-facing output, and so must carry the promise.
OUTPUT_PROMPTS = ("SUPERVISOR_SYSTEM", "SCREENER_SYSTEM", "WRITER_SYSTEM")

#: Placeholder values good enough to render any template in this module.
FILLERS: dict[str, str] = {
    **clause_kwargs(),
    "headline": "Full-stack engineer",
    "seniority": "mid",
    "location": "Remote",
    "profile": "skills: python, fastapi",
    "postings": "- pasted:demo Backend Engineer",
    "assessments": "none yet",
    "brief_written": "no",
    "steps": "1",
    "max_steps": "12",
    "posting_id": "pasted:demo",
    "title": "Backend Engineer",
    "company": "Acme",
    "findings": "coverage looked low",
    "score": "coverage 0.5",
    "assessment": "verdict: worth_applying",
    "posting_text": "We need Python.",
    "attribution": "Source: remotive - https://remotive.com/x",
}


def test_templates_were_actually_discovered() -> None:
    """Guards the introspection itself - an empty sweep would pass everything."""
    assert len(ALL_TEMPLATES) >= 8


@pytest.mark.parametrize("name", sorted(ALL_TEMPLATES))
def test_every_template_renders_with_substitute(name: str) -> None:
    """``.substitute`` raises on a missing placeholder. That is the point."""
    template = ALL_TEMPLATES[name]
    needed = {
        match.group("named") or match.group("braced")
        for match in template.pattern.finditer(template.template)
        if match.group("named") or match.group("braced")
    }
    rendered = template.substitute({key: FILLERS[key] for key in needed})
    assert rendered.strip()


@pytest.mark.parametrize("name", sorted(ALL_TEMPLATES))
def test_no_placeholder_survives_rendering(name: str) -> None:
    """A leaked ``$posting`` would be sent to the model as literal text."""
    template = ALL_TEMPLATES[name]
    needed = {
        match.group("named") or match.group("braced")
        for match in template.pattern.finditer(template.template)
        if match.group("named") or match.group("braced")
    }
    rendered = template.substitute({key: FILLERS[key] for key in needed})
    assert "$" not in rendered, f"{name} still contains a placeholder marker"


@pytest.mark.parametrize("name", sorted(ALL_TEMPLATES))
def test_every_placeholder_has_a_known_filler(name: str) -> None:
    """A new placeholder must be added here deliberately, not discovered at runtime."""
    template = ALL_TEMPLATES[name]
    for match in template.pattern.finditer(template.template):
        key = match.group("named") or match.group("braced")
        if key:
            assert key in FILLERS, f"{name} uses unknown placeholder ${key}"


def _method_calls_in_prompts_module() -> set[str]:
    """Names of every method actually called in ``prompts.py``, via the AST.

    Parsing beats grepping here, and not academically: the module's own
    docstring names ``str.format`` and ``.safe_substitute()`` in order to
    explain why they are banned, so a substring search reports the
    documentation as a violation. A first attempt at these tests did exactly
    that and failed against its own docs.
    """
    tree = ast.parse(PROMPTS_FILE.read_text(encoding="utf-8"))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_the_module_never_calls_str_format() -> None:
    """``str.format`` raises KeyError on the literal JSON braces in these prompts."""
    assert "format" not in _method_calls_in_prompts_module()


def test_the_module_never_calls_safe_substitute() -> None:
    """It would leak a raw ``$variable`` into a prompt instead of failing."""
    assert "safe_substitute" not in _method_calls_in_prompts_module()


@pytest.mark.parametrize("name", OUTPUT_PROMPTS)
def test_output_prompts_carry_the_no_submit_clause(name: str) -> None:
    """The promise must survive refactoring, so it is asserted."""
    template = ALL_TEMPLATES[name]
    assert "$no_submit" in template.template


def test_no_prompt_body_tells_the_model_to_submit() -> None:
    forbidden = re.compile(
        r"\b(apply to the|apply for the|submit the application|send the application|"
        r"candidate-se|candidatar-se|enviar a candidatura)\b",
        re.IGNORECASE,
    )
    for name, template in ALL_TEMPLATES.items():
        body = template.template.replace("$no_submit", "").replace("$data_not_instructions", "")
        assert not forbidden.search(body), f"{name} appears to instruct submission"


def test_the_never_claim_rule_reaches_the_writer() -> None:
    """not_claimable is the whole point of the brief; the prompt must demand it."""
    writer = ALL_TEMPLATES["WRITER_SYSTEM"].template
    assert "not_claimable" in writer
    assert "never_claim" in writer
    # Whitespace-normalised: the sentence wraps across an indented line.
    flattened = " ".join(writer.lower().split())
    assert "never leave it empty" in flattened


def test_the_screener_is_told_not_to_recompute_the_score() -> None:
    """The number comes from Python; the model must cite it, not reinvent it."""
    assert "$score" in ALL_TEMPLATES["SCREENER_SYSTEM"].template
    assert "do not recompute" in prompts.SCORE_CLAUSE.lower()


def test_the_supervisor_is_told_the_writer_is_expensive() -> None:
    """Routing has to carry weight, or the supervisor is decorative."""
    supervisor = ALL_TEMPLATES["SUPERVISOR_SYSTEM"].template
    assert "expensive" in supervisor
    assert "no_fit" in supervisor


def test_data_not_instructions_reaches_every_prompt_that_reads_a_posting() -> None:
    for name in ("SCREENER_SYSTEM", "WRITER_SYSTEM"):
        assert "$data_not_instructions" in ALL_TEMPLATES[name].template
