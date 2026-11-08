"""What kind of value a question asks for, and whether a candidate answer has that shape.

Used by the verifier for every backend: a "who" question answered with "2", or a "when" question
answered with a bare count, is rejected (refused) rather than cited.
"""

from __future__ import annotations

import re

NAME = "name"
DATE = "date"
NUMBER = "number"
ANY = "any"

_NAME_Q = re.compile(
    r"^\s*(?:who|whom|whose)\b|\b(?:who|whom|whose)\s+(?:is|was|are|were|did|does|founded|made|makes|owns|wrote|signed|leads|manages|maintains)\b"
    r"|\bwhich\s+(?:company|companies|vendor|supplier|manufacturer|maker|person|people|engineer|employee|team|customer|organi[sz]ation|firm|partner|contractor|author|owner|founder)\b"
    r"|\b(?:name|names)\s+of\s+the\s+(?:company|vendor|supplier|manufacturer|person|engineer|founder|author|owner|customer|team|organi[sz]ation|ceo|manager)\b"
    r"|\bwho\b",
    re.IGNORECASE,
)
_DATE_Q = re.compile(
    r"^\s*when\b|\bwhen\s+(?:was|were|did|does|do|is|will|are)\b|\b(?:what|which)\s+(?:date|day|year|month|week)\b|\bon\s+what\s+date\b|\bin\s+what\s+year\b",
    re.IGNORECASE,
)
_NUMBER_Q = re.compile(
    r"\bhow\s+(?:many|much|long|far|large|big|high|low|fast|often)\b|\b(?:number|count|total|amount|percentage|share)\s+of\b"
    r"|\bin\s+(?:seconds|minutes|hours|days|weeks|months|years|ms|milliseconds|degrees|watts|volts|bytes|kb|mb|gb|gib|mib|tb|percent|usd|dollars|euros|units)\b",
    re.IGNORECASE,
)

_MONTHS = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_DATE_SHAPE = re.compile(
    rf"\b(?:19|20)\d\d\b|\b\d{{1,4}}[-/.]\d{{1,2}}[-/.]\d{{1,4}}\b|\b(?:{_MONTHS})\b|\bQ[1-4]\b|\bFY\s?'?\d{{2,4}}\b|\b[12]H\b|\bH[12]\b"
    r"|\b(?:week|wk|sprint)\s*\d+\b|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)


def expected_kind(question: str) -> str:
    q = question or ""
    if _NAME_Q.search(q):
        return NAME
    if _DATE_Q.search(q):
        return DATE
    if _NUMBER_Q.search(q):
        return NUMBER
    return ANY


def kind_matches(kind: str, answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return True  # a refusal fits every question
    if kind == NAME:
        # A name has a word in it; a bare number, version, date or amount is not a name.
        if not re.search(r"[A-Za-zÀ-￿]{2,}", a):
            return False
        return not re.fullmatch(r"[\d\s.,:/%$€£+-]*(?:[A-Za-z]{1,3})?", a)
    if kind == DATE:
        return bool(_DATE_SHAPE.search(a))
    if kind == NUMBER:
        return bool(re.search(r"\d", a))
    return True
