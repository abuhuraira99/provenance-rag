"""Rules answerer: extract a typed value from the best-matching line/row, no model required.

Used as (a) the CPU fallback when the GPU path fails or runs out of time, (b) the TEST STUB's
"model", and (c) a cross-check. It is deliberately conservative: when no candidate is well
supported it refuses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .evidence import ANSWER_WORDS, QuestionTerms, coverage, ids_in, has_question_ids, link_ids, passage_coverage, passage_text, question_terms
from .index import Hit, Retriever
from .normalize import clean_answer, grade_norm
from .textutil import question_quantities, text_has_quantity, tokenize

# (type, question trigger, value pattern). First trigger that matches decides the type.
_CODE = r"\b[A-Z]{1,3}\d{3,6}\b|\b[A-Z]{2,5}-\d{3,6}\b|\b0x[0-9A-Fa-f]{2,8}\b"
_VERSION = r"(?<![\w.])v?\d+\.\d+(?:\.\d+){0,2}(?:[-+][A-Za-z0-9.]+)?(?![\w.])"
_PART = r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{2,}(?:-[A-Z0-9]+){1,5}\b"
_QUARTER = r"\bQ[1-4]\s*(?:FY\s?'?\d{2,4}|'?\d{2,4})\b|\bFY\s?'?\d{2,4}\s*Q[1-4]\b|\b[12]H\s*FY\s?\d{2,4}\b"
_PIN = r"\b[A-Z]{1,2}\d{1,3}\b"
_REVISION = r"(?i:\bREV(?:ISION)?)[-\s.]?(?:[A-Z]{0,2}\d[A-Z0-9]{0,3}|[A-Z]\d{0,2})(?:\.\d+)?\b"
_TICKET = r"\b[A-Z]{2,6}-\d{2,6}\b"
_SERIAL = r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]{2,}){1,4}\b"
_DATE = r"\bQ[1-4]\s*(?:FY\s?'?\d{2,4}|'?\d{2,4})\b|\bFY\s?'?\d{2,4}\s*Q[1-4]\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
_NUMBER = r"(?<![\w.\-/:#])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w\-/:])|(?<![\w.\-/:#])[-+]?\d+(?:\.\d+)?(?![\w\-/:]|\.\d)"

ANSWER_TYPES: list[tuple[str, str, str | None]] = [
    # Names (people, companies) have no reliable surface pattern: the rules answerer refuses them
    # and leaves them to the model.
    ("name", r"\b(?:who|whom|whose)\b|\bwhich\s+(?:company|vendor|supplier|manufacturer|maker|person|engineer|team|customer|organi[sz]ation|firm|author|owner|founder)\b", None),
    ("code", r"\b(?:error|fault|alarm|exit|status|event)\s+code\b|\bcode\b.{0,40}\b(?:logged|raised|reported|emitted)\b", _CODE),
    ("version", r"\b(?:firmware|software|driver|bios|release|build)\b.{0,20}\b(?:version|release|build)\b|\bversion\b|\bfixed in\b|\bwhich release\b|\bfirmware release\b", _VERSION),
    ("part", r"\bpart\s*(?:number|no\.?|#)|\bsku\b|\bp/n\b|\border(?:ing)? code\b", _PART),
    ("quarter", r"\bquarter\b", _QUARTER),
    ("pin", r"\bpins?\b|\bball\b|\bconnector position\b", _PIN),
    ("revision", r"\brevision\b|\brev\b", _REVISION),
    ("ticket", r"\b(?:ticket|issue|bug|incident)\s*(?:id|number|#)?\b.{0,30}\?$|\bwhich ticket\b", _TICKET),
    ("serial", r"\bserial\b", _SERIAL),
    ("date", r"\b(?:date|day)\b|^\s*when\b|\bwhen\s+(?:was|were|did|does|do|is|will)\b", _DATE),
]

_COMMENT = re.compile(r"^\s*(?:#|//|--|;|/\*|\*)")
_STALE = re.compile(r"\b(?:raised|lowered|changed|increased|decreased|reduced|moved)\s+from\b|\b(?:formerly|previously|was|old|legacy|deprecated|prior)\b", re.IGNORECASE)
_YEARISH = re.compile(r"^(?:19|20)\d\d$")


def answer_type(question: str) -> tuple[str, re.Pattern | None]:
    q = question.strip()
    for name, trigger, pattern in ANSWER_TYPES:
        if re.search(trigger, q, re.IGNORECASE):
            if pattern is None:
                return name, None
            flags = re.IGNORECASE if name == "date" else 0
            return name, re.compile(pattern, flags)
    return "number", re.compile(_NUMBER)


@dataclass
class Candidate:
    value: str
    score: float
    hit_index: int
    unit: str
    bridge_index: int = -1
    link: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class RulesAnswer:
    answer: str
    passage_indices: list[int]  # answer passage first, then bridge passages
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    atype: str = ""


def _units(text: str) -> list[tuple[int, str]]:
    out = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            out.append((pos, line))
        pos += len(line) + 1
    return out


def _cell_key(unit: str, start: int) -> str:
    """For 'key: value | key: value' rows, the key of the cell containing position `start`."""
    left = unit.rfind("|", 0, start)
    cell = unit[left + 1:]
    if ":" in cell:
        key = cell.split(":", 1)[0]
        if len(key) < 60:
            return key
    return ""


def _proximity(qt: QuestionTerms, unit: str, start: int, end: int) -> float:
    """Closeness of the candidate to a content term of the question (not an answer-type word,
    and not the candidate's own text)."""
    from .evidence import ANSWER_WORDS

    content = {t for t in qt.weights if t not in ANSWER_WORDS}
    low = unit.lower()
    best = 999
    for m in re.finditer(r"[a-z0-9_]+", low):
        if m.start() < end and m.end() > start:
            continue
        if set(tokenize(m.group(0))) & content:
            best = min(best, abs(m.start() - start))
    return math.exp(-best / 60.0) if best < 999 else 0.0


def rules_answer(question: str, hits: list[Hit], retriever: Retriever, *, min_score: float = 0.34) -> RulesAnswer:
    if not hits:
        return RulesAnswer("", [], 0.0)
    qt = question_terms(question, retriever)
    atype, rx = answer_type(question)
    if rx is None:
        return RulesAnswer("", [], 0.0, atype=atype)  # e.g. a person or company name
    quantities = question_quantities(question)
    q_norm = grade_norm(question)
    cov_chunk = [passage_coverage(qt, h) for h in hits]
    content = {t for t in qt.weights if t not in ANSWER_WORDS}

    # Possible bridges: for each passage, other-file passages sharing a link identifier.
    bridges: dict[int, list[tuple[int, str]]] = {}
    for i, a in enumerate(hits):
        for j, b in enumerate(hits):
            if i == j:
                continue
            links = link_ids(a, b, qt, retriever)
            if links:
                bridges.setdefault(i, []).append((j, sorted(links)[0]))

    cands: list[Candidate] = []
    for i, h in enumerate(hits):
        text = h.chunk.text
        status = retriever.status.get(h.chunk.file, "current")
        for _, unit in _units(text):
            for m in rx.finditer(unit):
                raw = m.group(0).strip()
                value = clean_answer(raw)
                if not value:
                    continue
                vn = grade_norm(value)
                if not vn or vn in q_norm and len(vn) >= 2 and re.search(rf"(?<![A-Z0-9]){re.escape(vn)}(?![A-Z0-9])", re.sub(r"[-.·_]", "", question.upper())):
                    continue  # the question's own identifiers are not answers
                if atype == "number" and _YEARISH.match(value.replace(",", "")) and not re.search(r"\byear\b", question, re.I):
                    continue
                unit_cov = coverage(qt, unit)
                # The line/row holding the value must itself mention something the question asks
                # about (an answer-type word such as "revision" alone is not enough).
                unit_terms = set(tokenize(unit))
                on_topic = any(t in unit_terms for t in content if len(t) >= 3)
                key = _cell_key(unit, m.start())
                key_bonus = 0.0
                if key:
                    kt = set(tokenize(key))
                    hit_w = sum(w for t, w in qt.weights.items() if t in kt)
                    key_bonus = 0.6 * hit_w / qt.total
                    if atype == "version" and re.search(r"fix|version|release|build", key, re.I):
                        key_bonus += 0.1
                prox = 0.12 * _proximity(qt, unit, m.start(), m.end())
                penalty = 1.0
                if _COMMENT.match(unit):
                    penalty *= 0.6
                if _STALE.search(unit[: m.start()][-40:]):
                    penalty *= 0.6
                if status == "older-revision":
                    penalty *= 0.7
                base_ids_ok = has_question_ids(qt, passage_text(h))
                base_qty_ok = all(text_has_quantity(passage_text(h), q) for q in quantities)
                direct = (0.6 * unit_cov + 0.4 * cov_chunk[i] + key_bonus + prox) * penalty
                options = []
                if base_ids_ok and base_qty_ok and on_topic:
                    options.append(Candidate(value, direct, i, unit, detail={"unit_cov": round(unit_cov, 3), "chunk_cov": round(cov_chunk[i], 3), "key": key}))
                for j, link in bridges.get(i, []):
                    bt = passage_text(hits[j])
                    if not has_question_ids(qt, passage_text(h) + "\n" + bt):
                        continue
                    if not all(text_has_quantity(passage_text(h) + "\n" + bt, q) for q in quantities):
                        continue
                    if not on_topic and link not in ids_in(unit):
                        continue
                    chain = (0.45 * unit_cov + 0.55 * max(cov_chunk[j], cov_chunk[i]) + key_bonus + prox) * penalty - 0.04
                    if cov_chunk[j] > cov_chunk[i] + 0.15:
                        options.append(Candidate(value, chain, i, unit, bridge_index=j, link=link, detail={"unit_cov": round(unit_cov, 3), "bridge_cov": round(cov_chunk[j], 3), "key": key}))
                if options:
                    cands.append(max(options, key=lambda c: c.score))
    if not cands:
        return RulesAnswer("", [], 0.0, atype=atype)
    cands.sort(key=lambda c: -c.score)
    best = cands[0]
    if best.score < min_score:
        return RulesAnswer("", [], 0.0, candidates=cands[:8], atype=atype)
    idx = [best.hit_index] + ([best.bridge_index] if best.bridge_index >= 0 else [])
    conf = max(0.05, min(0.95, best.score))
    return RulesAnswer(best.value, idx, round(conf, 3), candidates=cands[:8], atype=atype)
