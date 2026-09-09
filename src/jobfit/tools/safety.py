"""Defences for third-party text that ends up inside a prompt.

A job description is content written by someone else, fetched over the network,
and placed in the context of an agent that can call tools. That is a prompt
injection surface, and treating it as one is a requirement here rather than a
nicety.

**What this does honestly claim.** It finds text that is shaped like an
instruction to the model, neutralises the framing so it reads as quoted data,
and - the part that matters most - *reports* what it found so the run trace
records the attempt.

**What it does not claim.** This is not a solution to prompt injection. No
regex is. The real defences are architectural and live elsewhere in this
package: the model never sees a tool that can submit an application, network
tools refuse hosts outside an allowlist, tool arguments are schema-validated,
and every run has a hard step ceiling. Filtering is the last layer, not the
first.
"""

from __future__ import annotations

import re

#: Patterns that indicate the text is trying to address the model rather than
#: describe a job. Deliberately narrow: a job ad legitimately says "you will
#: ignore legacy systems", and flagging that would be worse than useless.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instruction|prompt|rule|direction|context)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_impersonation",
        re.compile(r"^\s*(system|assistant|developer|human)\s*:\s*", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "fake_delimiter",
        re.compile(
            r"(</?(system|instructions?|assistant|human)>|\[/?(INST|SYSTEM)\]|<\|[a-z_]+\|>)",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(you must|always|immediately)\b[^.\n]{0,40}"
            r"\b(call|invoke|use|run)\b[^.\n]{0,30}\b(tool|function|api)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(reveal|print|output|repeat|disclose|show)\b[^.\n]{0,30}"
            r"\b(system prompt|your instructions|api[ _-]?key|secret|credential)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "autosubmit_coercion",
        re.compile(
            r"\b(submit|apply|send)\b[^.\n]{0,30}\b(application|candidacy|cv|resume)\b"
            r"[^.\n]{0,30}\b(automatically|without|on my behalf|for me)\b",
            re.IGNORECASE,
        ),
    ),
)

_REDACTION = "[redacted: instruction-like text]"


def find_injection_markers(text: str) -> list[str]:
    """Names of the injection patterns present in ``text``, sorted and unique."""
    return sorted({name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)})


def defang(text: str) -> tuple[str, list[str]]:
    """Neutralise instruction-shaped spans, returning the text and what was found.

    Replaces rather than deletes: a reviewer reading the trace should be able to
    see that something was removed and where.
    """
    found: list[str] = []
    cleaned = text
    for name, pattern in _INJECTION_PATTERNS:
        cleaned, count = pattern.subn(_REDACTION, cleaned)
        if count:
            found.append(name)
    return cleaned, sorted(set(found))
