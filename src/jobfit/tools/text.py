"""Pure text helpers. No I/O, no model, no configuration - so they are trivially
testable and safe to reuse from anywhere.

Each of these exists because of a real, measured failure:

* ``unescape_html`` runs to a fixed point. Greenhouse returns bodies that are
  entity-escaped *markup* - the measured value starts ``&lt;div class=&quot;``
  - so a single ``html.unescape`` leaves you stripping tags that are still text.
* ``html_to_text`` preserves list markers. Postings nest a ``<p>`` inside each
  ``<li>``; strip tags naively and 15 requirement bullets become 15
  indistinguishable lines, and the requirement list silently comes back empty.
* ``clip_for_model`` keeps the head and the tail. A real Remotive posting
  measured 15,796 characters; sending that unclipped wastes tokens and
  overflows the KV cache on an 8 GB GPU.
* ``strip_think`` removes ``<think>`` blocks, which local reasoning models such
  as Qwen3 emit inline and which are not part of the answer.
"""

from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_END_RE = re.compile(r"</(p|div|li|ul|ol|br|h[1-6]|tr|table|section)\s*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LI_OPEN_RE = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_SPACE_RUN_RE = re.compile(r"[ \t\u00a0]{2,}")

#: Bullet glyphs and list markers seen in real postings. The set is wide because
#: boards differ: Greenhouse emits <li>, while Remotive wraps each bullet in a
#: <p> and marks it with a literal MIDDLE DOT (U+00B7). Miss a glyph and the
#: requirement list silently comes back empty - which looks like a posting with
#: no requirements rather than like a bug.
_BULLET_GLYPHS = "-*\u00b7\u2022\u2023\u2043\u25cf\u25aa\u25e6\u2013\u2014\u2192\u00bb\u25b8"
_BULLET_RE = re.compile(rf"^\s*(?:[{re.escape(_BULLET_GLYPHS)}]|\d+[.)])\s+(?P<text>.+)$")

#: Marks a list item and survives tag stripping. Because a real posting nests a
#: block element inside each <li>, the marker and its text land on separate
#: lines and have to be rejoined afterwards.
_LI_SENTINEL = "\x00li\x00"

MAX_UNESCAPE_PASSES = 3


def unescape_html(text: str, *, max_passes: int = MAX_UNESCAPE_PASSES) -> str:
    """Unescape HTML entities repeatedly until the text stops changing.

    Bounded, so a pathological input cannot spin. Greenhouse needs two passes;
    the bound simply stops that from being a surprise.
    """
    for _ in range(max_passes):
        decoded = html.unescape(text)
        if decoded == text:
            return decoded
        text = decoded
    return text


def _rejoin_bullets(lines: list[str]) -> list[str]:
    """Turn sentinel-marked list items into ``- text`` lines."""
    out: list[str] = []
    pending = False
    for line in lines:
        if line.startswith(_LI_SENTINEL):
            rest = line[len(_LI_SENTINEL) :].strip()
            if rest:
                out.append(f"- {rest}")
            else:
                pending = True
        elif pending and line:
            out.append(f"- {line}")
            pending = False
        else:
            out.append(line)
    return out


def html_to_text(raw: str) -> str:
    """Convert a posting body to plain text, preserving list structure.

    List markers are kept deliberately: requirements live in bullets far more
    often than in prose, and ``extract_bullets`` downstream is how the
    requirement list is obtained without spending a model call on it.
    """
    text = unescape_html(raw)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BR_RE.sub("\n", text)
    text = _LI_OPEN_RE.sub("\n" + _LI_SENTINEL, text)
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = unescape_html(text)  # entities revealed by tag removal
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RUN_RE.sub(" ", text)
    lines = _rejoin_bullets([line.strip() for line in text.split("\n")])
    joined = "\n".join(lines).replace(_LI_SENTINEL, "")
    return _BLANK_RUN_RE.sub("\n\n", joined).strip()


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` blocks emitted by local reasoning models."""
    return _THINK_RE.sub("", text).strip()


def clip_for_model(text: str, budget: int, *, head_ratio: float = 0.6) -> tuple[str, bool]:
    """Clip ``text`` to ``budget`` characters, keeping head and tail.

    Returns the text and whether it was truncated. The tail is kept because
    postings routinely put the requirements list at the bottom; a head-only clip
    throws away the part the assessment needs most.
    """
    if budget <= 0 or len(text) <= budget:
        return text, False

    marker = "\n\n[... clipped to fit the context budget ...]\n\n"
    room = max(budget - len(marker), 0)
    if room == 0:
        return text[:budget], True
    head_len = int(room * head_ratio)
    tail_len = room - head_len
    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip() if tail_len else ""
    return f"{head}{marker}{tail}", True


def extract_bullets(text: str, *, min_len: int = 8, max_len: int = 300) -> list[str]:
    """Pull bullet-style lines out of a posting body.

    Deterministic, and cheap: no model call is needed to find the requirement
    list in a well-formed job ad.
    """
    found: list[str] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        match = _BULLET_RE.match(line)
        if not match:
            continue
        item = match.group("text").strip(" .;")
        if not (min_len <= len(item) <= max_len):
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            found.append(item)
    return found
