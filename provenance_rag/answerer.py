"""Question answering: gather evidence, ask the model for strict JSON, verify, and choose citations.

Pipeline for one question
  1. Retrieve passages (BM25 + exact identifiers). Explicitly WITHDRAWN files are excluded unless the
     question asks about an old revision.
  2. Multi-hop expansion: identifiers found in the top passages (ticket ids, error codes, part numbers)
     that the question does not name are looked up in OTHER files; those passages are added and
     remember which passage led to them.
  3. The model gets numbered passages and returns {"answer", "sources", "needs"}; one follow-up
     lookup round is allowed when it asks for an identifier.
  4. Verification: the answer must be grounded in a passage; quantities in the question must be
     present in the evidence; withdrawn revisions cannot be the source.
  5. Citations follow necessity: the answer passage's file, plus bridge files that supplied an
     identifier the question did not name. Everything else is dropped.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Backend
from .evidence import QuestionTerms, grounded, has_question_ids, link_ids, passage_coverage, passage_text, question_terms
from .index import CorpusIndex, Hit, Retriever
from .normalize import clean_answer
from .revisions import question_targets_old
from .answertype import expected_kind, kind_matches
from .rules import rules_answer
from .textutil import extract_identifiers, id_key, question_quantities, text_has_quantity

DIRECT_K = 10
MAX_EXPANSIONS = 6
PROMPT_CHAR_BUDGET = 14_000
MAX_IMAGES = 2

SYSTEM_PROMPT = """You answer questions about a private document collection using ONLY the numbered passages you are given. The products, people and numbers in these documents are fictional, so anything you remember about them is wrong.

Reply with exactly one JSON object and nothing else:
{"answer": "<value>", "sources": [<passage numbers>], "needs": []}

answer
- The value only: a number, code, part number, version, quarter, name or revision. No sentence, no explanation, no unit.
- Copy the value exactly as the passage writes it, complete with its qualifiers: a quarter keeps its fiscal year ("Q3 FY27", not "Q3"); a revision keeps its prefix ("REV-C2", not "C2"); a part number is copied in full.
- If the passages do not state the answer, reply with "answer": "" and "sources": []. Never guess. Never use outside knowledge. A related value for a different product, quantity or condition is not the answer.
- A passage marked WITHDRAWN or OLDER REVISION never supplies the answer when a current document gives it.
- In source code, the assigned constant is the current value; a comment describing an earlier value is not.

sources
- The passage that states the answer, AND every passage that gave you an identifier (ticket id, error code, part number, model, serial) you needed in order to find that answer passage. Example: a log names ticket T-1 and a bug table says T-1 was fixed in 2.0, so cite both passages.
- Do not cite passages that only discuss the same topic or happen to repeat the value.

needs
- If you need the details of an identifier that appears in the passages but is not explained there, list it in "needs" and leave "answer" empty. Otherwise use []."""


@dataclass
class QueryResult:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    mode: str = ""  # llm | rules | refused
    passages: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def contract(self) -> dict:
        return {"answer": self.answer, "citations": list(self.citations), "confidence": float(self.confidence)}

    def to_json(self) -> dict:
        d = self.contract()
        d.update({"mode": self.mode, "passages": self.passages, "trace": self.trace, "latency_ms": self.latency_ms})
        return d


# ------------------------------------------------------------------------------------------------
# Evidence gathering


def withdrawn_files(index: CorpusIndex) -> set[str]:
    return {f["path"] for f in index.files if f.get("revision", {}).get("status") == "withdrawn"}


def gather(question: str, index: CorpusIndex, trace: list[str] | None = None) -> list[Hit]:
    r = index.retriever
    trace = trace if trace is not None else []
    exclude = set() if question_targets_old(question) else withdrawn_files(index)
    if exclude:
        trace.append(f"excluded withdrawn: {sorted(exclude)}")
    direct = r.search(question, k=DIRECT_K * 2, exclude_files=exclude)
    if not direct:
        return []
    top = direct[0].score
    hits = [h for h in direct if h.score >= 0.2 * top][:DIRECT_K]
    q_keys = {id_key(i) for i in extract_identifiers(question)}
    have = {h.chunk.id for h in hits}
    added = 0
    frontier = hits[:6]
    for depth in range(2):  # log -> error code -> ticket -> fix version is two hops
        new: list[Hit] = []
        seen_ids: set[str] = set()
        for src in frontier:
            for ident in extract_identifiers(src.chunk.text):
                key = id_key(ident)
                if key in q_keys or key in seen_ids:
                    continue
                seen_ids.add(key)
                files = r.id_files.get(key, set())
                others = files - {src.chunk.file} - exclude
                if not others or len(files) > 4:
                    continue  # absent elsewhere, or too common to be a link
                targets = [c for c in r.chunks_with_identifier(ident) if c.file in others and c.id not in have]
                if not targets:
                    continue
                scored = r.score_terms(r_terms(question))
                targets.sort(key=lambda c: (-scored.get(c.id, 0.0), c.id))
                for c in targets[:2]:
                    if added >= MAX_EXPANSIONS:
                        break
                    new.append(Hit(chunk=c, score=scored.get(c.id, 0.0), via=ident, via_chunk=src.chunk.id))
                    have.add(c.id)
                    added += 1
        if not new:
            break
        trace.append(f"hop {depth + 1}: " + ", ".join(f"{h.via} -> {h.chunk.file}" for h in new))
        hits.extend(new)
        frontier = new
    return hits


def r_terms(question: str) -> list[str]:
    from .textutil import STOPWORDS, tokenize

    return [t for t in tokenize(question) if t not in STOPWORDS]


# ------------------------------------------------------------------------------------------------
# Prompting


def _label(hit: Hit, index: CorpusIndex, number_of: dict[int, int]) -> str:
    parts = [hit.chunk.file, hit.chunk.loc]
    rev = index.file_info(hit.chunk.file).get("revision", {})
    if rev.get("status") == "withdrawn":
        parts.append("WITHDRAWN")
    elif rev.get("status") == "older-revision":
        parts.append(f"OLDER REVISION (newer: {rev.get('superseded_by')})")
    if hit.chunk.source == "ocr":
        parts.append("text read from an image")
    if hit.via:
        src = number_of.get(hit.via_chunk)
        parts.append(f"found by looking up {hit.via}" + (f" from [{src}]" if src else ""))
    return " | ".join(p for p in parts if p)


def build_prompt(question: str, hits: list[Hit], index: CorpusIndex, images: list[int]) -> str:
    number_of = {h.chunk.id: i + 1 for i, h in enumerate(hits)}
    lines = [f"QUESTION: {question}", "", "PASSAGES:"]
    budget = PROMPT_CHAR_BUDGET
    for i, h in enumerate(hits, 1):
        body = h.chunk.text
        if len(body) > budget:
            body = body[: max(budget, 200)] + " ..."
        budget -= len(body)
        lines.append(f"[{i}] {_label(h, index, number_of)}")
        if h.chunk.ctx and h.chunk.ctx not in body:
            lines.append(f"(context: {h.chunk.ctx[:160]})")
        lines.append(body)
        lines.append("")
    if images:
        lines.append("Attached images, in order: " + ", ".join(f"[{i + 1}] {hits[i].chunk.file}" for i in images))
        lines.append("")
    lines.append("Reply with the JSON object only.")
    return "\n".join(lines)


def parse_model_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start: end + 1]
        for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
            except ValueError:
                pass
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        srcs = [int(x) for x in re.findall(r"\d+", (re.search(r'"sources"\s*:\s*\[([^\]]*)\]', text) or [None, ""])[1])]
        return {"answer": m.group(1), "sources": srcs, "needs": []}
    raise ValueError("model reply is not JSON")


def _as_indices(sources, hits: list[Hit]) -> list[int]:
    """Model sources -> 0-based passage indices. Accepts 3, "3", "[3]", "P3" or a file path."""
    out: list[int] = []
    if not isinstance(sources, list):
        sources = [sources]
    n = len(hits)
    for s in sources:
        text = str(s).strip()
        m = re.fullmatch(r"\[?[Pp#]?(\d+)\]?", text)
        if m:
            k = int(m.group(1)) - 1
            if 0 <= k < n and k not in out:
                out.append(k)
            continue
        path = text.replace("\\", "/").lstrip("./")
        for k, h in enumerate(hits):
            if path and (h.chunk.file == path or h.chunk.file.endswith("/" + path)) and k not in out:
                out.append(k)
    return out


# ------------------------------------------------------------------------------------------------
# Verification and citation choice


def _numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?", text):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def _derivable(answer: str, texts: list[str]) -> bool:
    """A numeric answer computed from two stated numbers (sum, difference, product, ratio)."""
    try:
        target = float(answer.replace(",", ""))
    except ValueError:
        return False
    nums = []
    for t in texts:
        nums.extend(_numbers(t))
    nums = list(dict.fromkeys(nums))[:60]
    for i, a in enumerate(nums):
        for b in nums[i:]:
            for v in (a + b, abs(a - b), a * b, a / b if b else None, b / a if a else None):
                if v is not None and abs(v - target) <= 1e-6 * max(1.0, abs(target)):
                    return True
    return False


def choose_citations(
    question: str,
    answer: str,
    sources: list[int],
    hits: list[Hit],
    index: CorpusIndex,
    trace: list[str],
) -> tuple[str, list[str], float]:
    """Return (answer, citations, confidence) after verification; ("", [], 0.0) to refuse."""
    r: Retriever = index.retriever
    qt: QuestionTerms = question_terms(question, r)
    targets_old = question_targets_old(question)
    answer = clean_answer(answer)
    if not answer:
        trace.append("model refused")
        return "", [], 0.0
    kind = expected_kind(question)
    if not kind_matches(kind, answer):
        trace.append(f"refused: '{answer}' is not a {kind} value, which the question asks for")
        return "", [], 0.0

    def status(i: int) -> str:
        return r.status.get(hits[i].chunk.file, "current")

    usable = [i for i in range(len(hits)) if targets_old or status(i) != "withdrawn"]
    grounded_idx = [i for i in usable if grounded(answer, hits[i].chunk.text)]
    if not grounded_idx:
        src_texts = [hits[i].chunk.text for i in sources if i in usable]
        if len(sources) >= 1 and _derivable(answer, src_texts):
            files = list(dict.fromkeys(hits[i].chunk.file for i in sources if i in usable))
            trace.append(f"answer derived arithmetically from {files}")
            return answer, files, 0.5
        trace.append(f"refused: '{answer}' is not in any passage")
        return "", [], 0.0

    preferred = [i for i in sources if i in grounded_idx]
    ranked = preferred + [i for i in grounded_idx if i not in preferred]
    # Prefer current documents over older revisions when both state a value.
    order = {i: n for n, i in enumerate(ranked)}
    ranked = sorted(ranked, key=lambda i: (status(i) == "older-revision", order[i]))
    ans_i = ranked[0]
    ans = hits[ans_i]
    trace.append(f"answer passage [{ans_i + 1}] {ans.chunk.file} ({ans.chunk.loc})")

    # Bridges named by the model: keep the ones linked by an identifier the question does not name.
    kept: list[int] = []
    frontier = [ans_i]
    candidates = [i for i in sources if i != ans_i and hits[i].chunk.file != ans.chunk.file]
    changed = True
    while changed:
        changed = False
        for i in list(candidates):
            if any(link_ids(hits[i], hits[f], qt, r) for f in frontier):
                kept.append(i)
                frontier.append(i)
                candidates.remove(i)
                changed = True
    for i in candidates:
        trace.append(f"dropped [{i + 1}] {hits[i].chunk.file}: no identifier link to the answer")

    # Safety net: the answer passage does not tie back to the question on its own, and it was
    # reached through an identifier from another passage that does -> that passage is necessary.
    ans_cov = passage_coverage(qt, ans)
    anchored = has_question_ids(qt, passage_text(ans)) and (bool(qt.ids) or ans_cov >= 0.3)
    if not anchored and not kept:
        best = None
        for j, h in enumerate(hits):
            if j == ans_i or h.chunk.file == ans.chunk.file or (status(j) == "withdrawn" and not targets_old):
                continue
            if not link_ids(ans, h, qt, r):
                continue
            cov = passage_coverage(qt, h)
            ok_ids = has_question_ids(qt, passage_text(ans) + "\n" + passage_text(h))
            if ok_ids and cov >= 0.4 and cov >= ans_cov + 0.15 and (best is None or cov > best[1]):
                best = (j, cov)
        if best is not None:
            kept.append(best[0])
            trace.append(f"added bridge [{best[0] + 1}] {hits[best[0]].chunk.file} (answer passage not anchored to the question)")

    evidence = passage_text(ans) + "\n" + "\n".join(passage_text(hits[i]) for i in kept)
    for qty in question_quantities(question):
        if not text_has_quantity(evidence, qty):
            trace.append(f"refused: the question's '{qty}' is not in the evidence")
            return "", [], 0.0
    if not has_question_ids(qt, evidence):
        trace.append("refused: evidence does not mention the identifiers in the question")
        return "", [], 0.0

    files = list(dict.fromkeys([ans.chunk.file] + [hits[i].chunk.file for i in kept]))
    conf = 0.9 if ans_i in sources else 0.7
    return answer, files, conf


# ------------------------------------------------------------------------------------------------
# Entry point


def _passage_view(hits: list[Hit], cited: list[str], answer: str) -> list[dict]:
    out = []
    for i, h in enumerate(hits, 1):
        out.append({
            "n": i, "file": h.chunk.file, "loc": h.chunk.loc, "text": h.chunk.text[:1500], "score": round(h.score, 3),
            "via": h.via, "source": h.chunk.source, "cited": h.chunk.file in cited,
            "has_answer": bool(answer) and grounded(answer, h.chunk.text),
        })
    return out


def answer_question(
    question: str,
    index: CorpusIndex,
    backend: Backend,
    *,
    deadline: float | None = None,
    image_root: Path | None = None,
) -> QueryResult:
    t0 = time.time()
    deadline = deadline or (t0 + 25.0)
    res = QueryResult()
    trace = res.trace
    question = (question or "").strip()
    if not question or not index.chunks:
        res.mode = "refused"
        trace.append("empty question or empty index")
        return _finish(res, t0)
    hits = gather(question, index, trace)
    if not hits:
        res.mode = "refused"
        trace.append("no passage matches the question")
        return _finish(res, t0)

    r = index.retriever
    answer, citations, conf, mode = "", [], 0.0, "refused"
    use_llm = backend.can_generate
    if use_llm:
        try:
            answer, citations, conf, hits = _llm_answer(question, hits, index, backend, deadline, trace, image_root)
            mode = "llm" if answer else "refused"
        except Exception as exc:  # model failure or timeout -> rules fallback
            trace.append(f"model path failed ({type(exc).__name__}: {str(exc)[:120]}); using rules")
            use_llm = False
    if not use_llm:
        ra = rules_answer(question, hits, r)
        if ra.answer:
            sources = ra.passage_indices
            answer, citations, conf = choose_citations(question, ra.answer, sources, hits, index, trace)
            if answer:
                conf = min(conf, ra.confidence)
        mode = "rules" if answer else "refused"
        trace.append(f"rules: type={ra.atype} answer='{ra.answer}'")
    res.answer, res.citations, res.confidence, res.mode = answer, citations, round(float(conf), 3), mode
    if not res.answer:
        res.citations, res.confidence = [], 0.0
    res.passages = _passage_view(hits, res.citations, res.answer)
    return _finish(res, t0)


def _finish(res: QueryResult, t0: float) -> QueryResult:
    res.latency_ms = int((time.time() - t0) * 1000)
    return res


def _images_for(hits: list[Hit], image_root: Path | None) -> tuple[list[int], list[bytes]]:
    idx, blobs = [], []
    for i, h in enumerate(hits[:6]):
        if h.chunk.source != "ocr" or not h.chunk.image or len(blobs) >= MAX_IMAGES:
            continue
        if any(hits[j].chunk.image == h.chunk.image for j in idx):
            continue
        p = Path(h.chunk.image)
        if not p.is_absolute() and image_root is not None:
            p = image_root / p
        try:
            blobs.append(p.read_bytes())
            idx.append(i)
        except OSError:
            continue
    return idx, blobs


def _llm_answer(question, hits, index, backend, deadline, trace, image_root):
    observe = getattr(backend, "observe", None)
    for round_no in range(2):
        img_idx, blobs = _images_for(hits, image_root) if backend.can_see else ([], [])
        prompt = build_prompt(question, hits, index, img_idx)
        if observe:
            observe(question, hits, index.retriever)
        remaining = deadline - time.time()
        if remaining < 3:
            raise TimeoutError("no time left for the model")
        raw = backend.generate(SYSTEM_PROMPT, prompt, images=blobs, max_new_tokens=160, max_time=max(2.0, remaining - 2.0))
        trace.append(f"model reply: {raw.strip()[:300]}")
        data = parse_model_json(raw)
        answer = data.get("answer", "")
        sources = _as_indices(data.get("sources", []), hits)
        needs = [str(x) for x in (data.get("needs") or []) if str(x).strip()]
        if clean_answer(answer) or not needs or round_no == 1 or deadline - time.time() < 8:
            ans, cites, conf = choose_citations(question, answer, sources, hits, index, trace)
            return ans, cites, conf, hits
        added = _follow_up(question, needs, hits, index)
        trace.append(f"model asked for {needs}; added {added} passages")
    return "", [], 0.0, hits


def _follow_up(question: str, needs: list[str], hits: list[Hit], index: CorpusIndex) -> int:
    r = index.retriever
    have = {h.chunk.id for h in hits}
    exclude = set() if question_targets_old(question) else withdrawn_files(index)
    added = 0
    for need in needs[:3]:
        src = next((h for h in hits if need.lower() in h.chunk.text.lower()), None)
        found = r.search(need, k=3, exclude_files=exclude, must_ids=extract_identifiers(need))
        for h in found[:2]:
            if h.chunk.id in have:
                continue
            hits.append(Hit(chunk=h.chunk, score=h.score, via=need, via_chunk=src.chunk.id if src else -1))
            have.add(h.chunk.id)
            added += 1
    return added
