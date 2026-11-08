"""Answer normalisation: the grader's comparison rule, plus value-only cleanup of model output."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WS = re.compile(r"\s+")
_GRADER_STRIP = re.compile(r"[-.·_]")


def grade_norm(value: str) -> str:
    """The official comparison: uppercase, drop all whitespace, drop the characters - . · _"""
    value = (value or "").upper()
    value = _WS.sub("", value)
    return _GRADER_STRIP.sub("", value)


def answers_match(got: str, expected: str, aliases: Iterable[str] = ()) -> bool:
    g = grade_norm(got)
    if g == grade_norm(expected):
        return True
    return any(g == grade_norm(a) for a in aliases)


def citations_match(got: Iterable[str], expected: Iterable[str]) -> bool:
    return {norm_citation(c) for c in got} == {norm_citation(c) for c in expected}


def norm_citation(path: str) -> str:
    p = path.replace("\\", "/").strip()
    for prefix in ("/app/corpus/", "./"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p.lstrip("/")


# Units that may trail a bare number. The grader treats "94", "94 C" and "94°C" alike,
# but "180 seconds" normalises to "180SECONDS"; the bare number is the safe form.
_UNIT = (
    r"(?:°\s*[CF]|º\s*[CF]|deg(?:rees?)?\s*[CF]?|[CF]|W|kW|mW|V|mV|A|mA|Hz|kHz|MHz|GHz|"
    r"s|sec|secs|seconds?|ms|us|µs|ns|min|mins|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?|"
    r"B|KB|MB|GB|TB|KiB|MiB|GiB|TiB|bytes?|TB/s|GB/s|MB/s|Gbps|Mbps|%|percent|mm|cm|m|km|in|"
    r"kg|g|lbs?|units?|pcs|pieces?|RPM|rpm|dB|dBm|ohms?|Ω|req/s|tok/s|tokens?|requests?)"
)
_NUM_UNIT = re.compile(rf"^([-+]?\d[\d,]*(?:\.\d+)?)\s*{_UNIT}\.?$", re.IGNORECASE)
_CURRENCY = re.compile(r"^(?:USD|EUR|GBP|US\$|\$|€|£)\s*(?=\d)", re.IGNORECASE)
_PREFIX = re.compile(r"^(?:final\s+)?(?:answer|value|result)\s*[:=]\s*", re.IGNORECASE)
_REFUSALS = {
    "", "none", "null", "n/a", "na", "unknown", "not found", "not available", "no answer",
    "i don't know", "i do not know", "not in the corpus", "not stated", "not specified",
    "cannot be determined", "insufficient information", "not provided",
}


def clean_answer(raw: object) -> str:
    """Reduce a model answer to the value only. Returns "" for refusals / prose non-answers."""
    if raw is None:
        return ""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        text = format_number(raw)
    else:
        text = str(raw)
    text = text.strip().strip("`").strip()
    text = _PREFIX.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = text.rstrip(".;,").strip() if not re.fullmatch(r"[\d.]+", text) else text.rstrip(".")
    if text.lower() in _REFUSALS or text.lower().startswith(("i could not", "i cannot", "the corpus does not", "no information", "not found")):
        return ""
    text = _CURRENCY.sub("", text)
    m = _NUM_UNIT.match(text)
    if m:
        text = m.group(1)
    return text.strip()


def format_number(value: float | int) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if value != value:  # NaN
        return ""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))
