"""Any OpenAI-compatible chat endpoint (e.g. vLLM serving Qwen3-VL on an AMD Developer Cloud box).

Configure with OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_API_KEY. Uses only the standard library.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Sequence

from .base import Backend


def _mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF"):
        return "image/webp"
    return "image/png"


class OpenAIBackend(Backend):
    name = "openai"
    can_see = True

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout
        # Rate-limited endpoints: minimum seconds between requests, retries on HTTP 429.
        self.min_interval = float(os.environ.get("OPENAI_MIN_INTERVAL", "0") or 0)
        self.max_tokens_floor = int(os.environ.get("OPENAI_MAX_TOKENS_FLOOR", "0") or 0)
        extra = os.environ.get("OPENAI_EXTRA_BODY", "")
        self.extra_body = json.loads(extra) if extra else {}
        self._last = 0.0
        self._lock = threading.Lock()

    def generate(self, system, prompt, *, images: Sequence[bytes] = (), max_new_tokens=256, max_time=None) -> str:
        content: list[dict] = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{_mime(img)};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max(max_new_tokens, self.max_tokens_floor),
        }
        body.update(self.extra_body)
        data = json.dumps(body).encode("utf-8")
        timeout = min(self.timeout, max_time + 5) if max_time else self.timeout
        for attempt in range(4):
            with self._lock:
                wait = self._last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 502, 503) and attempt < 3:
                    time.sleep(min(30.0, 5.0 * (attempt + 1)))
                    continue
                raise
        content = payload["choices"][0]["message"].get("content") or ""
        # Reasoning models may wrap deliberation in <think> tags; only the final reply counts.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if "<think>" in content:
            content = content.split("</think>")[-1] if "</think>" in content else ""
        return content.strip()

    def device_info(self) -> dict:
        return {"backend": self.name, "python": platform.python_version(), "endpoint_model": self.model, "stub": False}
