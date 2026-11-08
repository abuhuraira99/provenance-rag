"""Runtime configuration, resolved from environment variables with container defaults."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

APP_NAME = "provenance_rag"


def _writable_dir(candidate: Path) -> bool:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def index_dir() -> Path:
    """Where the persisted index lives: $PRAG_INDEX_DIR, else /app/index, else a temp dir."""
    env = os.environ.get("PRAG_INDEX_DIR")
    if env:
        return Path(env)
    app_index = Path("/app/index")
    if Path("/app").is_dir() and _writable_dir(app_index):
        return app_index
    return Path(tempfile.gettempdir()) / "provenance_rag_index"


def output_dir() -> Path:
    return Path(os.environ.get("PRAG_OUTPUT_DIR", "/app/output"))


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "/models/Qwen3-VL-8B-Instruct"))


def socket_address() -> str:
    """`/path/to.sock` for a unix socket, or `tcp:HOST:PORT` (used on Windows)."""
    env = os.environ.get("PRAG_SOCKET")
    if env:
        return env
    if os.name != "nt" and hasattr(socket, "AF_UNIX"):
        return "/tmp/provenance_rag.sock"
    return "tcp:127.0.0.1:47613"


def default_backend() -> str:
    return os.environ.get("PRAG_BACKEND", "transformers")


def float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
