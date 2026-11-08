#!/usr/bin/env python3
"""Provenance RAG - contract entry point (thin client, standard library only).

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

Never loads the model. A query writes a valid placeholder JSON immediately, asks the resident
server (unix socket) within a hard deadline, then writes the final JSON atomically. If the server
cannot answer, it falls back to the rules answerer over the persisted index. Always exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
HERE = Path(__file__).resolve().parent
QUERY_BUDGET_S = float(os.environ.get("PRAG_QUERY_BUDGET", "26"))
INDEX_BUDGET_S = float(os.environ.get("PRAG_INDEX_BUDGET", "480"))
OUTPUT_DIR = Path(os.environ.get("PRAG_OUTPUT_DIR", "/app/output"))
EMPTY = {"answer": "", "citations": [], "confidence": 0.0}


def socket_address() -> str:
    env = os.environ.get("PRAG_SOCKET")
    if env:
        return env
    if os.name != "nt" and hasattr(socket, "AF_UNIX"):
        return "/tmp/provenance_rag.sock"
    return "tcp:127.0.0.1:47613"


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _connect(address: str, timeout: float) -> socket.socket:
    if address.startswith("tcp:"):
        _, host, port = address.split(":")
        return socket.create_connection((host, int(port)), timeout=timeout)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(address)
    return s


def request(payload: dict, timeout: float, address: str | None = None) -> dict:
    """One newline-delimited JSON request/response with a hard timeout."""
    s = _connect(address or socket_address(), max(0.2, min(timeout, 5.0)))
    try:
        s.settimeout(max(0.2, timeout))
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        end = time.monotonic() + timeout
        while not buf.endswith(b"\n"):
            s.settimeout(max(0.05, end - time.monotonic()))
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    finally:
        s.close()


def ping(timeout: float = 1.0) -> dict | None:
    try:
        resp = request({"op": "ping"}, timeout)
        return resp if resp.get("ok") else None
    except (OSError, ValueError):
        return None


def wait_for_server(deadline: float) -> dict | None:
    while True:
        info = ping(min(1.0, max(0.2, remaining(deadline))))
        if info or remaining(deadline) <= 0.3:
            return info
        time.sleep(min(0.25, max(0.0, remaining(deadline) - 0.1)))


def spawn_server() -> None:
    """Start the supervisor detached (normally the image CMD has already done this)."""
    server = HERE / "server.py"
    if not server.exists():
        return
    log_path = Path(os.environ.get("PRAG_SERVER_LOG", "/tmp/provenance_rag_server.log"))
    try:
        log = open(log_path, "ab")
    except OSError:
        log = subprocess.DEVNULL
    kwargs = {"stdout": log, "stderr": log, "stdin": subprocess.DEVNULL, "cwd": str(HERE)}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([sys.executable, str(server)], **kwargs)
    except OSError:
        pass


def write_json(path: Path, payload: dict) -> None:
    """Atomic: write a temp file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def sanitize(resp: dict | None, corpus: str | None = None) -> dict:
    if not isinstance(resp, dict):
        return dict(EMPTY)
    answer = resp.get("answer")
    answer = "" if answer is None else str(answer).strip()
    cites = resp.get("citations") or []
    if not isinstance(cites, list):
        cites = [cites]
    root = str(Path(corpus)).replace("\\", "/").rstrip("/") + "/" if corpus else None
    clean: list[str] = []
    for c in cites:
        c = str(c).replace("\\", "/").strip()
        if root and c.startswith(root):
            c = c[len(root):]
        if c.startswith("/app/corpus/"):
            c = c[len("/app/corpus/"):]
        c = c.lstrip("./") if c.startswith("./") else c
        if c and c not in clean:
            clean.append(c)
    try:
        conf = float(resp.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if not answer:
        return dict(EMPTY)
    return {"answer": answer, "citations": clean, "confidence": max(0.0, min(1.0, conf))}


def _package_importable() -> bool:
    for p in (HERE, HERE.parent):
        if (p / "provenance_rag").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return True
    # Installed elsewhere on sys.path (e.g. PYTHONPATH=/opt/site in the image).
    import importlib.util

    try:
        return importlib.util.find_spec("provenance_rag") is not None
    except (ImportError, ValueError):
        return False


def local_answer(corpus: str, query: str, deadline: float) -> dict:
    """Fallback: rules answerer over the persisted index, in this process (no model)."""
    if not _package_importable() or remaining(deadline) < 1.0:
        return dict(EMPTY)
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.engine import Engine

    engine = Engine(NoModelBackend())
    wall_deadline = time.time() + remaining(deadline)
    res = engine.ask(query, corpus=corpus, deadline=wall_deadline)
    return res.contract()


def local_index(corpus: str) -> dict:
    if not _package_importable():
        return {"ok": False}
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.engine import Engine

    return Engine(NoModelBackend()).build_index(corpus, ocr=False)


def do_index(corpus: str) -> int:
    deadline = START + INDEX_BUDGET_S
    info = ping()
    if info is None:
        spawn_server()
        info = wait_for_server(min(deadline, START + 90))
    if info is not None:
        try:
            budget = remaining(deadline) - 5
            resp = request({"op": "index", "corpus": corpus, "budget_s": budget}, timeout=budget + 4)
            print(json.dumps(resp))
            if resp.get("ok"):
                return 0
        except (OSError, ValueError) as exc:
            print(f"server index failed: {exc}", file=sys.stderr)
    try:
        print(json.dumps({"fallback": "local text-only index", **local_index(corpus)}))
    except Exception as exc:  # never fail the index call
        print(f"local index failed: {exc}", file=sys.stderr)
    return 0


def do_query(corpus: str, query_id: str, query: str) -> int:
    deadline = START + QUERY_BUDGET_S
    out = OUTPUT_DIR / f"{query_id}_output.json"
    try:
        write_json(out, dict(EMPTY))  # a valid answer exists from the first moment
    except OSError as exc:
        print(f"cannot write {out}: {exc}", file=sys.stderr)
    result = None
    try:
        info = wait_for_server(START + min(8.0, QUERY_BUDGET_S / 3))
        if info is not None:
            budget = remaining(deadline) - 1.5
            resp = request({"op": "query", "corpus": corpus, "query": query, "budget_s": budget}, timeout=budget + 1.0)
            if resp.get("ok"):
                result = sanitize(resp, corpus)
    except (OSError, ValueError) as exc:
        print(f"server query failed: {exc}", file=sys.stderr)
    if result is None:
        try:
            result = sanitize(local_answer(corpus, query, deadline), corpus)
        except Exception as exc:
            print(f"local fallback failed: {exc}", file=sys.stderr)
            result = dict(EMPTY)
    try:
        write_json(out, result)
    except OSError as exc:
        print(f"cannot write {out}: {exc}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Provenance RAG contract client")
    ap.add_argument("--index", help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem assigned by the harness, e.g. query_01")
    ap.add_argument("--query", help="the question")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 0
    try:
        if args.index:
            return do_index(args.index)
        if args.query_id:
            return do_query(args.corpus or "/app/corpus", args.query_id, args.query or "")
        print("nothing to do: pass --index DIR, or --corpus DIR --query-id ID --query TEXT", file=sys.stderr)
    except Exception as exc:  # the contract: always exit 0, always leave valid JSON
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.query_id:
            try:
                write_json(OUTPUT_DIR / f"{args.query_id}_output.json", dict(EMPTY))
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
