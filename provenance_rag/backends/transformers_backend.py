"""Qwen3-VL on the GPU through Hugging Face transformers (the backend used in the container).

Loaded once in the resident server, bf16, placed on the device explicitly, greedy decoding.
On ROCm, torch exposes the AMD GPU as the `cuda` device.
"""

from __future__ import annotations

import io
import os
import platform
import threading
from typing import Sequence

from .base import Backend

MAX_IMAGE_SIDE = int(os.environ.get("PRAG_MAX_IMAGE_SIDE", "1600"))
MIN_IMAGE_SIDE = 64


class TransformersBackend(Backend):
    name = "transformers"
    can_see = True

    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = str(model_dir)
        self.device = device or os.environ.get("PRAG_DEVICE", "cuda")
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # pragma: no cover - older transformers
            from transformers import Qwen3VLForConditionalGeneration as AutoVLM  # type: ignore

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
        try:
            model = AutoVLM.from_pretrained(self.model_dir, dtype=torch.bfloat16, **kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            model = AutoVLM.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16, **kwargs)
        self.model = model.to(self.device).eval()

    def _image(self, data: bytes):
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode not in ("RGB",):
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
        w, h = img.size
        scale = 1.0
        if max(w, h) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(w, h)
        elif min(w, h) < MIN_IMAGE_SIDE:
            scale = MIN_IMAGE_SIDE / min(w, h)
        if scale != 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return img

    def generate(self, system, prompt, *, images: Sequence[bytes] = (), max_new_tokens=256, max_time=None) -> str:
        import torch

        if self.model is None:
            raise RuntimeError("model not loaded")
        content = [{"type": "image", "image": self._image(b)} for b in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        with self._lock:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            if max_time:
                gen_kwargs["max_time"] = max(1.0, float(max_time))
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def device_info(self) -> dict:
        info = {"backend": self.name, "python": platform.python_version(), "model_dir": self.model_dir, "stub": False}
        try:
            import torch

            info["torch"] = torch.__version__
            info["hip"] = getattr(torch.version, "hip", None)
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                info["vram_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        except Exception as exc:  # pragma: no cover
            info["torch_error"] = str(exc)
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except Exception:  # pragma: no cover
            pass
        return info
