"""Tokenisation, light stemming and identifier extraction shared by indexer, retriever and answerers."""

from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = frozenset(
    """
    a an and any are as at be been being but by can could did do does doing for from had has have
    having how i if in into is it its itself me my of on or our ours shall should so than that the
    their them then there these they this those through to too under until up upon very was we were
    what when where which while who whom why will with would you your yours about above after again
    against all also am both each few further here just more most no nor not now only other out over
    own same some such s t don per via
    tell give list show find according document documents file files corpus value exactly please
    """.split()
)

# A run of word characters joined by identifier punctuation: ORR-1847, 4.3.2, THERM_ALERT#, TQ-40.
_RAW_TOKEN = re.compile(r"[A-Za-z0-9À-￿]+(?:[._\-/#+][A-Za-z0-9À-￿]+)*#?")
_SPLIT = re.compile(r"[._\-/#+]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """A tiny suffix stripper: fixed/fixes/fix -> fix, engaged/engages -> engag, samples -> sampl."""
    if len(word) <= 3 or not word.isalpha():
        return word
    for suffix in ("ies",):
        if word.endswith(suffix) and len(word) > 4:
            return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                break
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Index terms for a text. Identifiers are kept whole AND split, so `DEFAULT_BATCH_TIMEOUT_S`
    matches "default batch timeout" and `ORR-1847` matches "orr-1847", "orr1847" and "1847"."""
    out: list[str] = []
    for m in _RAW_TOKEN.finditer(text):
        raw = m.group(0)
        low = raw.lower().rstrip("#")
        compound = bool(_SPLIT.search(low)) or bool(_CAMEL.search(raw))
        if compound:
            out.append(low)
            joined = _SPLIT.sub("", low)
            if joined != low:
                out.append(joined)
            pieces: list[str] = []
            for part in _SPLIT.split(raw):
                pieces.extend(p for p in _CAMEL.split(part) if p)
            for p in pieces:
                pl = p.lower()
                if len(pl) < 2 and not pl.isdigit():
                    continue
                if not keep_stopwords and pl in STOPWORDS:
                    continue
                out.append(stem(pl))
        else:
            if not keep_stopwords and low in STOPWORDS:
                continue
            if len(low) < 2 and not low.isdigit():
                continue
            out.append(stem(low))
    return out


def content_terms(question: str) -> list[str]:
    """Distinct query terms, in order, without stopwords."""
    seen: dict[str, None] = {}
    for t in tokenize(question):
        seen.setdefault(t, None)
    return list(seen)


# ---------------------------------------------------------------------------------------------
# Identifiers: the tokens that can link one document to another (ticket ids, error codes, part
# numbers, versions, model names). Used for exact matching and for multi-hop expansion.

_ID_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_./:][A-Za-z0-9]+)*#?")
_UNITISH = re.compile(r"^\d+(?:\.\d+)?(?:[a-z]{1,3}|°c)$", re.IGNORECASE)
_TIMESTAMPISH = re.compile(r"^\d{4}-\d{2}-\d{2}(?:t.*)?$|^\d{1,2}:\d{2}(?::\d{2})?$|^\d{8}t", re.IGNORECASE)
_VERSION = re.compile(r"^v?\d+(?:\.\d+){1,3}[a-z]?$", re.IGNORECASE)


def id_key(token: str) -> str:
    """Canonical form for identifier equality: uppercase, no separators (ORR-1847 == orr_1847)."""
    return re.sub(r"[-_./:#\s]", "", token.upper())


def is_identifier(tok: str) -> bool:
    t = tok.rstrip("#")
    if len(t) < 3:
        return False
    has_digit = any(c.isdigit() for c in t)
    has_alpha = any(c.isalpha() for c in t)
    if _TIMESTAMPISH.match(t):
        return False
    if _VERSION.match(t):
        return t.count(".") >= 2 or t.lower().startswith("v")
    if not has_digit:
        # SCREAMING identifiers with separators, e.g. THERM_ALERT#, are linkable too.
        return bool(re.fullmatch(r"[A-Z][A-Z0-9]*(?:[_-][A-Z0-9]+)+#?", tok))
    if not has_alpha:
        return False
    if _UNITISH.match(t):
        return False
    return len(id_key(t)) >= 4 or "-" in t


def extract_identifiers(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _ID_CANDIDATE.finditer(text):
        tok = m.group(0).rstrip(".:/-_")
        if is_identifier(tok):
            seen.setdefault(tok, None)
    return list(seen)


_NUMBER_IN_QUESTION = re.compile(r"(?<![A-Za-z0-9\-.])(\d{1,3}(?:,\d{3})+|\d{3,}|\d+(?:\.\d+)?\s*[kKmM])(?![A-Za-z0-9\-])")


def question_quantities(question: str) -> list[str]:
    """Numbers in the question that constrain the answer (e.g. "at 10,000 unit volume").
    Numbers inside identifiers (TQ-40, ORR-1847) and years are excluded."""
    out = []
    for m in _NUMBER_IN_QUESTION.finditer(question):
        raw = m.group(1)
        digits = number_variants(raw)
        if not digits:
            continue
        val = next(iter(digits))
        if re.fullmatch(r"(19|20)\d\d", val):
            continue
        out.append(raw)
    return out


def number_variants(raw: str) -> set[str]:
    raw = raw.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM])", raw)
    if m:
        base = float(m.group(1)) * (1000 if m.group(2).lower() == "k" else 1_000_000)
        return {str(int(base))}
    digits = raw.replace(",", "")
    if not digits:
        return set()
    return {digits}


def text_has_quantity(text: str, raw: str) -> bool:
    """True if the text contains the quantity in any common spelling (10,000 / 10000 / 10k / 10 000)."""
    wanted = number_variants(raw)
    if not wanted:
        return True
    for m in re.finditer(r"(?<![\d.])(\d{1,3}(?:[, ]\d{3})+|\d+(?:\.\d+)?)(\s*[kKmM](?![A-Za-z]))?", text):
        num = m.group(1).replace(",", "").replace(" ", "")
        if m.group(2):
            try:
                mult = 1000 if m.group(2).strip().lower() == "k" else 1_000_000
                num = str(int(float(num) * mult))
            except ValueError:
                continue
        if num in wanted:
            return True
    return False
