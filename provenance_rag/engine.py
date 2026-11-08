"""Engine: owns the backend and the corpus index; builds, persists, reloads and answers."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .answerer import QueryResult, answer_question
from .backends import Backend, make_backend
from .config import index_dir as default_index_dir
from .index import Chunk, CorpusIndex, add_parsed, chunk_blocks
from .parsers import INDEXED, PENDING_OCR, Block, ParsedFile, parse_file, walk_corpus
from .revisions import revision_status

OCR_MIN_REMAINING_S = 8.0


class Engine:
    def __init__(self, backend: Backend, index_directory: Path | None = None):
        self.backend = backend
        self.index_dir = Path(index_directory) if index_directory else default_index_dir()
        self.index: CorpusIndex | None = None
        self.lock = threading.RLock()
        self._pending: list[tuple[str, dict]] = []  # (rel path, job dict) awaiting OCR

    @classmethod
    def create(cls, backend: str | Backend = "stub", index_directory: Path | None = None, load: bool = True, **kw) -> "Engine":
        b = make_backend(backend, **kw) if isinstance(backend, str) else backend
        if load:
            b.load()
        return cls(b, index_directory)

    # -------------------------------------------------------------------------------- indexing

    def build_index(self, corpus: str | Path, *, deadline: float | None = None, ocr: bool = True, progress=None) -> dict:
        t0 = time.time()
        root = Path(corpus).resolve()
        idx = CorpusIndex(corpus=str(root), built_at=t0)
        files, problems = walk_corpus(root) if root.is_dir() else ([], [{"path": str(root), "status": "unreadable", "detail": "corpus is not a directory"}])
        parsed: list[ParsedFile] = []
        for n, path in enumerate(files, 1):
            pf = parse_file(root, path)  # never raises
            parsed.append(pf)
            if progress:
                progress({"stage": "parse", "done": n, "total": len(files), "file": pf.rel, "status": pf.status})
        revs = revision_status([{"path": pf.rel, "head_text": pf.head_text} for pf in parsed])
        jobs: list[tuple[ParsedFile, int]] = []
        for pf in parsed:
            row = pf.inventory()
            row["revision"] = revs.get(pf.rel, {"status": "current"})
            idx.files.append(row)
            add_parsed(idx, pf)
            for j in range(len(pf.ocr_jobs)):
                jobs.append((pf, j))
        idx.files.extend(problems)
        idx.renumber()
        with self.lock:
            self.index = idx
        self._pending = []
        img_dir = self.index_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for pf, j in jobs:
            job = pf.ocr_jobs[j]
            blob = img_dir / f"{job.sha}{job.suffix}"
            if not blob.exists():
                try:
                    blob.write_bytes(job.data)
                except OSError:
                    continue
            self._pending.append((pf.rel, {"loc": job.loc, "sha": job.sha, "image": str(blob)}))
        done = self.run_ocr(deadline=deadline, progress=progress) if ocr else 0
        idx.stats = self._stats(time.time() - t0)
        self.save()
        return {"corpus": str(root), "files": len(idx.files), "chunks": len(idx.chunks), "ocr_done": done, "ocr_pending": len(self._pending), "seconds": round(time.time() - t0, 2)}

    def _ocr_cache(self, sha: str) -> Path:
        return self.index_dir / "ocr_cache" / self.backend.name / f"{sha}.json"

    def run_ocr(self, *, deadline: float | None = None, progress=None) -> int:
        """OCR pending images (cached by content hash). Stops at the deadline; the rest stay pending."""
        if self.index is None:
            return 0
        done = 0
        remaining_jobs = []
        for n, (rel, job) in enumerate(list(self._pending), 1):
            cache = self._ocr_cache(job["sha"])
            text = None
            if cache.exists():
                try:
                    text = json.loads(cache.read_text(encoding="utf-8"))["text"]
                except (OSError, ValueError, KeyError):
                    text = None
            if text is None:
                if not self.backend.can_see or (deadline is not None and deadline - time.time() < OCR_MIN_REMAINING_S):
                    remaining_jobs.append((rel, job))
                    continue
                try:
                    data = Path(job["image"]).read_bytes()
                    max_time = None if deadline is None else max(4.0, deadline - time.time() - 2.0)
                    text = self.backend.ocr(data, max_time=max_time)
                except Exception as exc:
                    self._set_status(rel, None, f"ocr failed: {type(exc).__name__}")
                    continue
                if not self.backend.is_stub or text:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"text": text, "backend": self.backend.name}), encoding="utf-8")
                    except OSError:
                        pass
            self._add_ocr_text(rel, job, text or "")
            done += 1
            if progress:
                progress({"stage": "ocr", "done": n, "total": len(self._pending), "file": rel})
        self._pending = remaining_jobs
        for rel, _ in remaining_jobs:
            self._set_status(rel, PENDING_OCR, "no vision backend available" if not self.backend.can_see else "waiting for OCR time budget")
        if done:
            self.index.renumber()
        return done

    def _add_ocr_text(self, rel: str, job: dict, text: str) -> None:
        assert self.index is not None
        info = self.index.file_info(rel)
        if not text.strip():
            self._set_status(rel, INDEXED if info.get("blocks") else "empty", "vision model found no text")
            return
        title = info.get("title", "")
        new = chunk_blocks(rel, title, [Block(text=text.strip(), loc=job["loc"], unit="ocr")], source="ocr")
        for c in new:
            c.image = job["image"]
        with self.lock:
            self.index.chunks.extend(new)
        self._set_status(rel, INDEXED, f"ocr: {len(text.strip())} chars")

    def _set_status(self, rel: str, status: str | None, detail: str) -> None:
        if self.index is None:
            return
        info = self.index.file_info(rel)
        if info:
            if status:
                info["status"] = status
            info["detail"] = detail

    def _stats(self, seconds: float) -> dict:
        assert self.index is not None
        by_status: dict[str, int] = {}
        for f in self.index.files:
            by_status[f["status"]] = by_status.get(f["status"], 0) + 1
        return {"files": len(self.index.files), "chunks": len(self.index.chunks), "by_status": by_status, "index_seconds": round(seconds, 2), "backend": self.backend.name}

    def save(self) -> None:
        if self.index is not None:
            with self.lock:
                self.index.save(self.index_dir)

    def load(self, corpus: str | Path | None = None) -> bool:
        idx = CorpusIndex.load(self.index_dir)
        if idx is None:
            return False
        if corpus is not None and Path(idx.corpus) != Path(corpus).resolve():
            return False
        with self.lock:
            self.index = idx
        return True

    def ensure_index(self, corpus: str | Path | None, deadline: float | None = None) -> None:
        if self.index is not None and (corpus is None or Path(self.index.corpus) == Path(corpus).resolve()):
            return
        if self.load(corpus):
            return
        if corpus is not None:
            # No index for this corpus: build a text-only one now (images need the --index pass).
            self.build_index(corpus, deadline=deadline, ocr=False)

    # -------------------------------------------------------------------------------- answering

    def ask(self, question: str, *, corpus: str | Path | None = None, deadline: float | None = None) -> QueryResult:
        self.ensure_index(corpus, deadline)
        if self.index is None:
            return QueryResult(mode="refused", trace=["no index"])
        return answer_question(question, self.index, self.backend, deadline=deadline, image_root=self.index_dir)

    def inventory(self) -> list[dict]:
        return list(self.index.files) if self.index else []
