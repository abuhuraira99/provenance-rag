"""Backend registry: transformers (GPU, used in the image), openai (any compatible endpoint), stub (tests)."""

from __future__ import annotations

from .base import Backend


class NoModelBackend(Backend):
    """Rules-only operation: no generation, no OCR. Used when the model cannot be loaded."""

    name = "rules"
    can_generate = False
    can_see = False

    def generate(self, *a, **k):  # pragma: no cover - never called
        raise RuntimeError("rules backend has no model")

    def warmup(self) -> None:
        pass


def make_backend(name: str, **kwargs) -> Backend:
    name = (name or "").lower()
    if name in ("transformers", "hf", "gpu"):
        from ..config import model_dir
        from .transformers_backend import TransformersBackend

        return TransformersBackend(kwargs.get("model_dir") or str(model_dir()))
    if name in ("openai", "vllm"):
        from .openai_backend import OpenAIBackend

        return OpenAIBackend()
    if name == "stub":
        from .stub import StubBackend

        return StubBackend(ocr_map=kwargs.get("ocr_map"))
    if name in ("rules", "none", "cpu"):
        return NoModelBackend()
    raise ValueError(f"unknown backend '{name}' (transformers | openai | stub | rules)")


__all__ = ["Backend", "NoModelBackend", "make_backend"]
