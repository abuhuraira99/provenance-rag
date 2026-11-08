"""Studio: a local web UI served by the standard library (no build step, no CDN).

    python -m provenance_rag.studio --backend stub      # CPU, clearly labelled TEST STUB
    python -m provenance_rag.studio --backend rules     # CPU rules-only answerer
    python -m provenance_rag.studio --backend openai    # any OpenAI-compatible endpoint
    python -m provenance_rag.studio --backend transformers

Live mode indexes a folder and answers questions with the configured backend. Run-report mode
shows a bench JSON (reports/*.json, or a file you drop in).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..backends import NoModelBackend, make_backend
from ..engine import Engine

STATIC = Path(__file__).resolve().parent / "static"


class StudioState:
    def __init__(self, backend: str, index_dir: Path | None, reports_dir: Path):
        self.engine = Engine(make_backend(backend), index_dir)
        self.reports_dir = reports_dir
        self.model_state = "loading"
        self.model_error = ""
        self.busy = threading.Lock()
        self.engine.load()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            self.engine.backend.load()
            self.model_state = "ready"
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {exc}"
            self.engine.backend = NoModelBackend()
            self.model_state = "degraded"

    def status(self) -> dict:
        b = self.engine.backend
        idx = self.engine.index
        return {
            "version": __version__, "backend": b.name, "label": b.label(), "stub": b.is_stub,
            "model_state": self.model_state, "model_error": self.model_error,
            "can_generate": b.can_generate, "can_see": b.can_see,
            "corpus": idx.corpus if idx else None, "stats": idx.stats if idx else {},
            "inventory": self.engine.inventory(), "ocr_pending": len(self.engine._pending),
        }


def make_handler(state: StudioState):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ProvenanceStudio/{__version__}"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code: int = 200) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 2_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                return {}

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/status":
                return self._json(state.status())
            if url.path == "/api/reports":
                items = []
                if state.reports_dir.is_dir():
                    for p in sorted(state.reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            s = data.get("summary", {})
                            items.append({"name": p.stem, "label": data.get("label"), "model": data.get("model_label", ""),
                                          "passed": s.get("passed"), "total": s.get("total"), "created": data.get("created")})
                        except (OSError, ValueError):
                            continue
                return self._json(items)
            if url.path == "/api/report":
                name = (parse_qs(url.query).get("name") or [""])[0]
                path = (state.reports_dir / f"{name}.json").resolve()
                if path.parent != state.reports_dir.resolve() or not path.is_file():
                    return self._json({"error": "no such report"}, 404)
                return self._send(200, path.read_bytes(), "application/json; charset=utf-8")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            body = self._body()
            if url.path == "/api/index":
                corpus = str(body.get("corpus") or "").strip()
                if not corpus or not Path(corpus).is_dir():
                    return self._json({"error": f"not a folder: {corpus!r}"}, 400)
                with state.busy:
                    t0 = time.time()
                    summary = state.engine.build_index(corpus, deadline=time.time() + float(body.get("budget_s") or 600))
                    summary["latency_ms"] = int((time.time() - t0) * 1000)
                return self._json({"summary": summary, **state.status()})
            if url.path == "/api/ask":
                q = str(body.get("question") or "").strip()
                if not q:
                    return self._json({"error": "empty question"}, 400)
                if state.engine.index is None:
                    return self._json({"error": "index a folder first"}, 400)
                with state.busy:
                    res = state.engine.ask(q, deadline=time.time() + float(body.get("budget_s") or 30))
                out = res.to_json()
                out["backend"] = state.engine.backend.label()
                return self._json(out)
            self._json({"error": "not found"}, 404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Provenance RAG Studio")
    ap.add_argument("--backend", default="stub", help="transformers | openai | stub | rules")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--index-dir", type=Path, help="index location (default: a studio-specific temp dir)")
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--corpus", help="index this folder at start-up")
    ap.add_argument("--open", action="store_true", help="open a browser tab")
    args = ap.parse_args(argv)
    if args.index_dir is None:
        import tempfile

        args.index_dir = Path(tempfile.gettempdir()) / f"provenance_rag_studio_{args.backend}"
    state = StudioState(args.backend, args.index_dir, args.reports)
    if args.corpus:
        state.engine.build_index(args.corpus)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Provenance Studio on {url}  (backend: {state.engine.backend.label()})", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
