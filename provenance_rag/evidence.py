"""Evidence helpers shared by the rules answerer and the LLM verifier:
question-term coverage, answer grounding, and identifier links between passages (multi-hop)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import Hit, Retriever
from .normalize import grade_norm
from .textutil import STOPWORDS, content_terms, extract_identifiers, id_key, tokenize

# Words that describe the kind of answer rather than the thing asked about.
ANSWER_WORDS = frozenset(
    tokenize(
        "what which who whose value number name code version release quarter part revision rev pin "
        "printed shown listed stated given mentioned called labeled labelled carries carry used",
        keep_stopwords=True,
    )
)


@dataclass
class QuestionTerms:
    terms: list[str]
    weights: dict[str, float]
    total: float
    ids: list[str]  # identifiers named in the question (exact-match anchors)
    id_keys: set[str]


def question_terms(question: str, retriever: Retriever) -> QuestionTerms:
    terms = [t for t in content_terms(question) if t not in STOPWORDS]
    weights = {}
    for t in terms:
        w = retriever.idf(t) if retriever.df.get(t) else 0.5 * retriever.idf(t)
        if t in ANSWER_WORDS:
            w *= 0.35
        weights[t] = max(w, 0.05)
    ids = extract_identifiers(question)
    return QuestionTerms(terms=terms, weights=weights, total=sum(weights.values()) or 1.0, ids=ids, id_keys={id_key(i) for i in ids})


def coverage(qt: QuestionTerms, text: str) -> float:
    toks = set(tokenize(text))
    return sum(w for t, w in qt.weights.items() if t in toks) / qt.total


def passage_text(hit: Hit) -> str:
    return f"{hit.chunk.text}\n{hit.chunk.ctx}\n{hit.chunk.file.replace('/', ' ')}"


def passage_coverage(qt: QuestionTerms, hit: Hit) -> float:
    return coverage(qt, passage_text(hit))


def ids_in(text: str) -> set[str]:
    return {id_key(i) for i in extract_identifiers(text)}


def has_question_ids(qt: QuestionTerms, text: str) -> bool:
    if not qt.id_keys:
        return True
    present = ids_in(text)
    norm = grade_norm(text)
    return all(k in present or k in norm for k in qt.id_keys)


def link_ids(a: Hit, b: Hit, qt: QuestionTerms, retriever: Retriever, max_files: int = 4) -> set[str]:
    """Identifiers shared by two passages from different files that the question does not name.
    These are what make a second file necessary (log -> ticket id -> bug database)."""
    if a.chunk.file == b.chunk.file:
        return set()
    shared = ids_in(a.chunk.text) & ids_in(b.chunk.text)
    out = set()
    for k in shared:
        if k in qt.id_keys:
            continue
        if len(retriever.id_files.get(k, ())) > max_files:
            continue
        out.add(k)
    return out


_ALNUM = re.compile(r"[A-Za-z0-9]+")


def grounded(answer: str, text: str) -> bool:
    """The answer string (or all of its alphanumeric tokens) appears in the text."""
    if not answer:
        return False
    a = grade_norm(answer)
    if not a:
        return False
    if len(a) >= 5 and a in grade_norm(text):
        return True
    # Short values (94, B14, 4.3.2) must match on token boundaries: "94" is not found in "1094".
    spaced = re.sub(r"\s+", " ", re.sub(r"[-.·_]", "", text.upper()))
    if re.search(rf"(?<![A-Z0-9]){re.escape(a)}(?![A-Z0-9])", spaced):
        return True
    toks = [x.upper() for x in _ALNUM.findall(answer)]
    if len(toks) >= 2:
        text_toks = {x.upper() for x in _ALNUM.findall(text)}
        return all(x in text_toks for x in toks)
    return False
