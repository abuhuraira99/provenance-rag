"""Chunking, the persisted corpus index, and hybrid lexical retrieval (BM25 + exact identifiers)."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .parsers import Block, ParsedFile
from .textutil import STOPWORDS, extract_identifiers, id_key, tokenize

INDEX_VERSION = 3
CHUNK_CHARS = 900
LINE_WINDOW = 12
LINE_OVERLAP = 3


@dataclass
class Chunk:
    id: int
    file: str
    loc: str
    text: str
    ctx: str = ""  # file title / section heading, searchable but not quoted as evidence
    unit: str = "para"
    source: str = "text"  # text | ocr
    image: str = ""  # cached image (for OCR chunks) that can be shown to the model at query time

    def to_json(self) -> dict:
        return {"id": self.id, "file": self.file, "loc": self.loc, "text": self.text, "ctx": self.ctx, "unit": self.unit, "source": self.source, "image": self.image}

    @classmethod
    def from_json(cls, d: dict) -> "Chunk":
        return cls(**{k: d[k] for k in ("id", "file", "loc", "text", "ctx", "unit", "source", "image") if k in d})


def _merge_loc(first: str, last: str) -> str:
    if first == last:
        return first
    if first.startswith("line ") and last.startswith("line "):
        return f"lines {first[5:]}-{last[5:]}"
    return f"{first} - {last}"


def chunk_blocks(rel: str, title: str, blocks: list[Block], source: str = "text") -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(text: str, loc: str, unit: str, heading: str = "") -> None:
        ctx = " / ".join(x for x in (title, heading) if x)
        chunks.append(Chunk(id=-1, file=rel, loc=loc, text=text.strip(), ctx=ctx, unit=unit, source=source))

    i = 0
    n = len(blocks)
    while i < n:
        b = blocks[i]
        standalone = (
            b.unit == "row"
            or b.heading.startswith(("constant", "function defaults"))
            # a table's column-header line (followed by rows, or last) stands alone;
            # a prose heading flows into the paragraphs after it.
            or (b.unit == "heading" and (i + 1 >= n or blocks[i + 1].unit == "row"))
        )
        if standalone:
            add(b.text, b.loc, b.unit, b.heading)
            i += 1
            continue
        if b.unit in ("line", "code") and not b.heading:
            j = i
            buf: list[Block] = []
            size = 0
            while j < n and blocks[j].unit in ("line", "code") and not blocks[j].heading and len(buf) < LINE_WINDOW and size + len(blocks[j].text) < CHUNK_CHARS * 1.5:
                buf.append(blocks[j])
                size += len(blocks[j].text) + 1
                j += 1
            if not buf:
                buf, j = [b], i + 1
            add("\n".join(x.text for x in buf), _merge_loc(buf[0].loc, buf[-1].loc), b.unit)
            if j >= n or blocks[j].unit not in ("line", "code") or blocks[j].heading:
                i = j
            else:
                i = max(i + 1, j - LINE_OVERLAP)
            continue
        # Prose: group paragraphs under one heading up to the chunk size.
        j = i
        buf = []
        size = 0
        heading = b.heading
        while j < n and blocks[j].unit in ("para", "heading", "ocr") and (blocks[j].heading == heading or blocks[j].unit == "heading" and not buf):
            if buf and size + len(blocks[j].text) > CHUNK_CHARS:
                break
            buf.append(blocks[j])
            size += len(blocks[j].text) + 1
            if blocks[j].unit == "heading":
                heading = blocks[j].heading or heading
            j += 1
        if not buf:
            buf, j = [b], i + 1
        text = "\n".join(x.text for x in buf)
        if len(text) > CHUNK_CHARS * 2:
            for k in range(0, len(text), CHUNK_CHARS):
                add(text[max(0, k - 120): k + CHUNK_CHARS], buf[0].loc, b.unit, heading)
        else:
            add(text, _merge_loc(buf[0].loc, buf[-1].loc), buf[0].unit, heading)
        i = j
    return chunks


@dataclass
class CorpusIndex:
    corpus: str = ""
    built_at: float = 0.0
    files: list[dict] = field(default_factory=list)  # inventory rows (+ revision status)
    chunks: list[Chunk] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _retriever: "Retriever | None" = None

    def file_info(self, rel: str) -> dict:
        for f in self.files:
            if f["path"] == rel:
                return f
        return {}

    @property
    def retriever(self) -> "Retriever":
        if self._retriever is None:
            self._retriever = Retriever(self)
        return self._retriever

    def invalidate(self) -> None:
        self._retriever = None

    def renumber(self) -> None:
        for i, c in enumerate(self.chunks):
            c.id = i
        self.invalidate()

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "index.json"
        tmp = directory / f".index.json.{os.getpid()}.tmp"
        payload = {
            "version": INDEX_VERSION,
            "corpus": self.corpus,
            "built_at": self.built_at,
            "stats": self.stats,
            "files": self.files,
            "chunks": [c.to_json() for c in self.chunks],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: Path) -> "CorpusIndex | None":
        path = directory / "index.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if payload.get("version") != INDEX_VERSION:
            return None
        idx = cls(corpus=payload.get("corpus", ""), built_at=payload.get("built_at", 0.0), files=payload.get("files", []), stats=payload.get("stats", {}))
        idx.chunks = [Chunk.from_json(c) for c in payload.get("chunks", [])]
        return idx


def add_parsed(index: CorpusIndex, pf: ParsedFile) -> None:
    index.chunks.extend(chunk_blocks(pf.rel, pf.title, pf.blocks))


# ------------------------------------------------------------------------------------------------
# Retrieval


@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str = ""  # identifier that led here (multi-hop expansion), "" for direct hits
    via_chunk: int = -1  # chunk id that supplied that identifier


class Retriever:
    K1 = 1.2
    B = 0.6

    def __init__(self, index: CorpusIndex):
        self.index = index
        self.chunks = index.chunks
        self.tf: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.ids: dict[str, set[int]] = defaultdict(set)  # identifier key -> chunk ids
        self.id_files: dict[str, set[str]] = defaultdict(set)
        self.status = {f["path"]: f.get("revision", {}).get("status", "current") for f in index.files}
        for c in self.chunks:
            toks = tokenize(c.text) + tokenize(c.ctx) + tokenize(c.file.replace("/", " "))
            tf = Counter(toks)
            self.tf.append(tf)
            self.lengths.append(len(toks))
            for t in tf:
                self.df[t] += 1
                self.postings[t].append(c.id)
            for ident in extract_identifiers(c.text + "\n" + c.ctx):
                k = id_key(ident)
                self.ids[k].add(c.id)
                self.id_files[k].add(c.file)
        self.n = max(len(self.chunks), 1)
        self.avg = (sum(self.lengths) / self.n) if self.lengths else 1.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score_terms(self, terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(terms):
            plist = self.postings.get(t)
            if not plist:
                continue
            w = self.idf(t)
            for cid in plist:
                f = self.tf[cid][t]
                denom = f + self.K1 * (1 - self.B + self.B * self.lengths[cid] / self.avg)
                scores[cid] += w * f * (self.K1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 10, *, exclude_files: set[str] | None = None, must_ids: list[str] | None = None) -> list[Hit]:
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        scores = self.score_terms(terms)
        # Exact identifier matches dominate: a question naming ORR-1847 wants the ORR-1847 rows.
        for ident in extract_identifiers(query) + list(must_ids or []):
            key = id_key(ident)
            for cid in self.ids.get(key, ()):
                scores[cid] = scores.get(cid, 0.0) + 4.0 + 2.0 / max(len(self.id_files[key]), 1)
        # Bigram bonus for phrases ("junction temperature", "thermal throttle").
        if len(terms) > 1:
            pairs = list(zip(terms, terms[1:]))
            for cid in list(scores):
                text = self.chunks[cid].text.lower()
                bonus = 0.0
                for a, b in pairs:
                    if len(a) > 2 and len(b) > 2 and a in text and b in text:
                        ia = text.find(a)
                        ib = text.find(b, ia)
                        if 0 <= ia < ib and ib - ia < 40:
                            bonus += 0.6
                scores[cid] += bonus
        hits = []
        for cid, s in scores.items():
            c = self.chunks[cid]
            if exclude_files and c.file in exclude_files:
                continue
            st = self.status.get(c.file, "current")
            if st == "older-revision":
                s *= 0.7
            hits.append(Hit(chunk=c, score=s))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    def chunks_with_identifier(self, ident: str) -> list[Chunk]:
        return [self.chunks[i] for i in sorted(self.ids.get(id_key(ident), ()))]

    def file_frequency(self, ident: str) -> int:
        return len(self.id_files.get(id_key(ident), ()))


def now() -> float:
    return time.time()
