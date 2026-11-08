"""Resident model server and its supervisor.

    python3 /app/server.py            # supervisor (image CMD): starts the worker, restarts it, never exits
    python -m provenance_rag.server --worker

The worker binds the socket immediately (so clients can see it is alive), loads the model in the
background, does one warm-up generation, and serves newline-delimited JSON requests:

    {"op": "ping"}
    {"op": "index", "corpus": "/app/corpus", "budget_s": 480}
    {"op": "query", "corpus": "/app/corpus", "query": "...", "budget_s": 24}
    {"op": "status"}

Parsing starts while the model is still loading; image OCR waits for the model within the budget,
and anything left over is finished in the background and swapped into the live index.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import __version__
from .backends import NoModelBackend, make_backend
from .config import default_backend, index_dir, socket_address
from .engine import Engine


def log(msg: str) -> None:
    sys.stderr.write(f"[provenance-rag {time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


class Worker:
    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.state = "loading"
        self.error = ""
        self.started = time.time()
        backend = make_backend(backend_name) if backend_name != "rules" else NoModelBackend()
        self.engine = Engine(backend, index_dir())
        if self.engine.load():
            log(f"loaded persisted index for {self.engine.index.corpus}")
        self.ready = threading.Event()
        self.index_lock = threading.Lock()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        t0 = time.time()
        try:
            self.engine.backend.load()
            self.engine.backend.warmup()
            self.state = "ready"
            log(f"model ready ({self.engine.backend.name}) in {time.time() - t0:.1f}s")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log(f"model load failed, serving rules-only: {self.error}")
            traceback.print_exc()
            self.engine.backend = NoModelBackend()
            self.state = "degraded"
        finally:
            self.ready.set()

    # ------------------------------------------------------------------------------ operations

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "state": self.state, "backend": self.engine.backend.name, "pid": os.getpid(), "version": __version__,
                    "corpus": self.engine.index.corpus if self.engine.index else None, "uptime_s": round(time.time() - self.started, 1)}
        if op == "index":
            return self.op_index(req)
        if op == "query":
            return self.op_query(req)
        if op == "status":
            info = self.engine.backend.device_info()
            return {"ok": True, "state": self.state, "error": self.error, "device": info,
                    "stats": self.engine.index.stats if self.engine.index else {}, "inventory": self.engine.inventory(),
                    "ocr_pending": len(self.engine._pending)}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def op_index(self, req: dict) -> dict:
        corpus = req.get("corpus") or "/app/corpus"
        deadline = time.time() + float(req.get("budget_s", 480))
        with self.index_lock:
            summary = self.engine.build_index(corpus, ocr=False)
            if self.engine._pending:
                # Images need the model: wait for it, but never past the budget.
                self.ready.wait(timeout=max(0.0, deadline - time.time() - 10))
                if self.ready.is_set():
                    summary["ocr_done"] = self.engine.run_ocr(deadline=deadline)
                    self.engine.save()
            summary["ocr_pending"] = len(self.engine._pending)
            summary["state"] = self.state
        if self.engine._pending:
            threading.Thread(target=self._finish_ocr, daemon=True).start()
        return {"ok": True, **summary}

    def _finish_ocr(self) -> None:
        self.ready.wait()
        with self.index_lock:
            n = self.engine.run_ocr(deadline=None)
            self.engine.save()
        log(f"background OCR finished {n} images")

    def op_query(self, req: dict) -> dict:
        t0 = time.time()
        deadline = t0 + float(req.get("budget_s", 24))
        if not self.ready.is_set():
            # Model still loading: give it part of the budget, then answer with rules.
            self.ready.wait(timeout=max(0.0, deadline - time.time() - 6))
        res = self.engine.ask(req.get("query", ""), corpus=req.get("corpus"), deadline=deadline)
        out = res.contract()
        out.update({"ok": True, "mode": res.mode, "latency_ms": res.latency_ms})
        if req.get("debug"):
            out["trace"] = res.trace
            out["passages"] = res.passages
        return out


def _make_server(address: str, handler_cls):
    if address.startswith("tcp:"):
        _, host, port = address.split(":")

        class TCP(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        return TCP((host, int(port)), handler_cls)
    path = Path(address)
    if path.exists():
        # A live server answers; a stale socket file is removed.
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(str(path))
            s.close()
            raise RuntimeError(f"another server is already listening on {path}")
        except OSError:
            path.unlink(missing_ok=True)

    class Unix(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    srv = Unix(str(path), handler_cls)
    os.chmod(str(path), 0o666)
    return srv


def run_worker(backend_name: str, address: str) -> None:
    holder: dict = {}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = holder["worker"].handle(req)
                except Exception as exc:
                    traceback.print_exc()
                    resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

    srv = _make_server(address, Handler)  # bind first so clients can see the server is up
    log(f"listening on {address}")
    holder["worker"] = Worker(backend_name)
    srv.serve_forever()


def supervise(backend_name: str, address: str) -> None:
    """Start the worker and restart it whenever it dies. Never returns."""
    lock_fh = None
    if os.name != "nt":
        import fcntl

        lock_fh = open(os.environ.get("PRAG_SUPERVISOR_LOCK", "/tmp/provenance_rag.supervisor.lock"), "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("a supervisor is already running; idling so the container stays up")
            while True:
                time.sleep(3600)
    failures = 0
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    while True:
        name = backend_name if failures < 3 else "rules"
        log(f"starting worker (backend={name})")
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, "-m", "provenance_rag.server", "--worker", "--backend", name, "--socket", address], env=env, cwd=pkg_root)
        code = proc.wait()
        lived = time.time() - t0
        failures = 0 if lived > 300 else failures + 1
        log(f"worker exited with {code} after {lived:.0f}s; restarting")
        time.sleep(min(10, 1 + failures))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Provenance RAG resident server")
    ap.add_argument("--worker", action="store_true", help="run the worker directly (no supervisor)")
    ap.add_argument("--backend", default=default_backend())
    ap.add_argument("--socket", default=socket_address())
    args = ap.parse_args(argv)
    if args.worker:
        run_worker(args.backend, args.socket)
    else:
        supervise(args.backend, args.socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
