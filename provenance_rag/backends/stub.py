"""TEST STUB backend: deterministic, CPU-only, no model.

`generate` answers the LLM prompt by running the rules answerer over the passages the answerer
handed it (so the whole JSON -> verify -> cite path is exercised in tests). `ocr` returns text
from an explicit `ocr_map` (sha256 or filename -> text) and "" otherwise. Results from this backend
are labelled TEST STUB everywhere they are shown.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Callable, Sequence

from .base import Backend


class StubBackend(Backend):
    name = "stub"
    is_stub = True
    can_see = True

    def __init__(self, ocr_map: dict[str, str] | None = None, responder: Callable[[str, str], str] | None = None):
        self.ocr_map = dict(ocr_map or {})
        self.responder = responder
        self.calls: list[dict] = []
        self._observed = None  # (question, passages, retriever) from the answerer

    def observe(self, question, passages, retriever) -> None:
        self._observed = (question, passages, retriever)

    def generate(self, system, prompt, *, images: Sequence[bytes] = (), max_new_tokens=256, max_time=None) -> str:
        self.calls.append({"system": system[:80], "prompt_chars": len(prompt), "images": len(images)})
        if self.responder is not None:
            return self.responder(system, prompt)
        if self._observed is None:
            return "OK"
        from ..rules import rules_answer

        question, passages, retriever = self._observed
        ra = rules_answer(question, passages, retriever)
        return json.dumps({"answer": ra.answer, "sources": [i + 1 for i in ra.passage_indices], "needs": []})

    def ocr(self, image: bytes, *, max_new_tokens: int = 700, max_time=None) -> str:
        sha = hashlib.sha256(image).hexdigest()
        return self.ocr_map.get(sha, "")

    def device_info(self) -> dict:
        return {"backend": "stub", "label": "TEST STUB", "python": platform.python_version(), "stub": True}
