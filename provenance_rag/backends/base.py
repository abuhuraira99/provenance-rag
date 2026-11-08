"""Backend interface: one vision-language model used for OCR at index time and answering at query time."""

from __future__ import annotations

import platform
from typing import Sequence

OCR_SYSTEM = "You are a meticulous transcription engine. You copy text exactly; you never invent text."

OCR_PROMPT = """Transcribe ALL text visible in this image.

Rules:
- Copy every word, number, code, part number, revision and symbol exactly as printed (keep characters such as # _ - . / exactly).
- One visual line of text per output line, top to bottom, left to right.
- Do not describe the image, do not summarise, do not add anything that is not printed.

After the transcription, if any printed items are linked by position rather than by a sentence (a pin and the signal written under it, a field label and its value, a table row and its columns, a callout and what it points to), add a section that starts with the line
ASSOCIATIONS:
and list each link as `item: linked text`, one per line, using the exact printed strings. Omit the section if there are no such links."""


class Backend:
    """Subclasses implement `generate`; `ocr` is built on it for vision-capable backends."""

    name = "base"
    is_stub = False
    can_generate = True
    can_see = False  # accepts images

    def load(self) -> None:  # pragma: no cover - trivial
        """Load weights / open connections. Called once, in the resident server."""

    def warmup(self) -> None:
        try:
            self.generate("Reply with OK.", "OK?", max_new_tokens=4)
        except Exception:
            pass

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        images: Sequence[bytes] = (),
        max_new_tokens: int = 256,
        max_time: float | None = None,
    ) -> str:
        raise NotImplementedError

    def ocr(self, image: bytes, *, max_new_tokens: int = 700, max_time: float | None = None) -> str:
        if not self.can_see:
            return ""
        return self.generate(OCR_SYSTEM, OCR_PROMPT, images=[image], max_new_tokens=max_new_tokens, max_time=max_time).strip()

    def device_info(self) -> dict:
        return {"backend": self.name, "python": platform.python_version(), "stub": self.is_stub}

    def label(self) -> str:
        return "TEST STUB" if self.is_stub else self.name
