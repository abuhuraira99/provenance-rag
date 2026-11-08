"""The harness contract: app.py arguments, output file name, JSON shape, atomic write, deadlines,
the resident server round trip and the no-server fallback. app.py runs as a real subprocess."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "app.py"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def env_for(tmp_path, port=None, **extra):
    env = dict(os.environ)
    env["PRAG_OUTPUT_DIR"] = str(tmp_path / "output")
    env["PRAG_INDEX_DIR"] = str(tmp_path / "index")
    env["PRAG_SOCKET"] = f"tcp:127.0.0.1:{port or free_port()}"
    env["PRAG_SERVER_LOG"] = str(tmp_path / "server.log")
    env.pop("PYTHONPATH", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_app(args, env, timeout=60):
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(APP), *args], env=env, capture_output=True, text=True, timeout=timeout)
    return proc, time.time() - t0


def read_out(tmp_path, qid):
    return json.loads((tmp_path / "output" / f"{qid}_output.json").read_text(encoding="utf-8"))


def assert_shape(payload):
    assert set(payload) >= {"answer", "citations", "confidence"}
    assert isinstance(payload["answer"], str) and isinstance(payload["citations"], list)
    assert all(isinstance(c, str) and not c.startswith("/") for c in payload["citations"])
    if not payload["answer"]:
        assert payload["citations"] == []


def test_query_without_server_uses_local_fallback(tmp_path, corpus):
    # No server can be reached (nothing listens; the spawned supervisor is disabled by a bogus path
    # via PRAG_QUERY_BUDGET being short). The answer still comes from the persisted index.
    env = env_for(tmp_path, PRAG_QUERY_BUDGET=6)
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.engine import Engine

    Engine(NoModelBackend(), tmp_path / "index").build_index(corpus["root"])
    proc, dt = run_app(["--corpus", str(corpus["root"]), "--query-id", "query_07", "--query", "Which firmware version fixed ticket ACM-2210?"], env)
    assert proc.returncode == 0
    out = read_out(tmp_path, "query_07")
    assert_shape(out)
    assert out["answer"] == "7.1.4" and out["citations"] == ["support/tickets.csv"]
    assert dt < 10


def test_query_id_is_used_verbatim_and_refusal_shape(tmp_path, corpus):
    env = env_for(tmp_path, PRAG_QUERY_BUDGET=5)
    proc, _ = run_app(["--corpus", str(corpus["root"]), "--query-id", "weird-ID_9", "--query", "What colour is the sky on Mars?"], env)
    assert proc.returncode == 0
    out = read_out(tmp_path, "weird-ID_9")
    assert out == {"answer": "", "citations": [], "confidence": 0.0}


def test_bad_arguments_still_exit_zero(tmp_path):
    proc, _ = run_app(["--bogus"], env_for(tmp_path))
    assert proc.returncode == 0


def test_silent_server_hits_deadline_and_writes_valid_json(tmp_path, corpus):
    """A server that accepts but never answers must not make app.py overrun its budget."""
    port = free_port()
    srv = socket.socket()
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    conns = []
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
                conns.append(c)
            except OSError:
                continue

    threading.Thread(target=accept, daemon=True).start()
    try:
        env = env_for(tmp_path, port=port, PRAG_QUERY_BUDGET=4)
        proc, dt = run_app(["--corpus", str(corpus["root"]), "--query-id", "q_slow", "--query", "Which firmware version fixed ticket ACM-2210?"], env)
        assert proc.returncode == 0
        assert dt < 9, dt
        assert_shape(read_out(tmp_path, "q_slow"))
    finally:
        stop.set()
        srv.close()
        for c in conns:
            c.close()


def test_atomic_write_leaves_no_temp_files(tmp_path):
    sys.path.insert(0, str(APP.parent))
    import importlib

    app = importlib.import_module("app")
    out = tmp_path / "o" / "x_output.json"
    app.write_json(out, {"answer": "1", "citations": ["a.txt"], "confidence": 0.5})
    assert json.loads(out.read_text(encoding="utf-8"))["answer"] == "1"
    assert [p.name for p in out.parent.iterdir()] == ["x_output.json"]
    assert app.sanitize({"answer": "  ", "citations": ["x"]}) == {"answer": "", "citations": [], "confidence": 0.0}
    assert app.sanitize({"answer": "5", "citations": ["/app/corpus/a/b.pdf", "a/b.pdf"], "confidence": "0.7"}, "/app/corpus")["citations"] == ["a/b.pdf"]
    assert app.sanitize(None) == {"answer": "", "citations": [], "confidence": 0.0}


@pytest.fixture
def stub_worker(tmp_path, corpus):
    port = free_port()
    env = env_for(tmp_path, port=port)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen([sys.executable, "-m", "provenance_rag.server", "--worker", "--backend", "stub", "--socket", f"tcp:127.0.0.1:{port}"],
                            env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.1)
    yield env
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_index_then_queries_through_resident_server(tmp_path, corpus, stub_worker):
    env = stub_worker
    proc, dt = run_app(["--index", str(corpus["root"])], env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary.get("ok") and summary["files"] >= 9
    cases = [
        ("query_01", "Which firmware version fixed ticket ACM-2210?", "7.1.4", ["support/tickets.csv"]),
        ("query_02", "The edge node log shows a thermal throttle incident. Which firmware release fixed the underlying defect?", "7.1.4",
         ["logs/edge_node_2026-10-11.log", "support/tickets.csv"]),
        ("query_03", "What is the airspeed of an unladen swallow?", "", []),
    ]
    for qid, q, ans, cites in cases:
        proc, dt = run_app(["--corpus", str(corpus["root"]), "--query-id", qid, "--query", q], env)
        assert proc.returncode == 0
        assert dt < 30
        out = read_out(tmp_path, qid)
        assert_shape(out)
        assert out["answer"] == ans and sorted(out["citations"]) == sorted(cites)


def test_studio_serves_page_and_api(tmp_path, corpus, deny_unreadable):
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen

    from provenance_rag.studio.server import StudioState, make_handler

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "demo.json").write_text(json.dumps({"label": "TEST STUB", "stub": True, "summary": {"passed": 1, "total": 1}, "items": []}), encoding="utf-8")
    state = StudioState("stub", tmp_path / "studio_idx", reports)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        html = urlopen(base + "/").read().decode("utf-8")
        assert "Provenance" in html and "Run report" in html
        st = json.loads(urlopen(base + "/api/status").read())
        assert st["label"] == "TEST STUB" and st["stub"] is True

        def post(path, body):
            req = Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urlopen(req).read())

        idx = post("/api/index", {"corpus": str(corpus["root"])})
        assert any(f["status"] == "unreadable" for f in idx["inventory"])
        res = post("/api/ask", {"question": "What error code is logged when the thermal throttle engages?"})
        assert res["answer"] == "W4410" and res["passages"] and res["backend"] == "TEST STUB"
        assert [r["name"] for r in json.loads(urlopen(base + "/api/reports").read())] == ["demo"]
    finally:
        srv.shutdown()


@pytest.mark.skipif(os.name == "nt" or not hasattr(socket, "AF_UNIX"), reason="unix sockets and the fcntl supervisor are POSIX-only")
def test_supervisor_unix_socket_end_to_end(tmp_path, corpus):
    """The container shape: supervisor (server.py) + worker on a unix socket + app.py client;
    the worker is killed once and must come back."""
    import signal

    sock = tmp_path / "prag.sock"
    env = env_for(tmp_path, PRAG_BACKEND="stub", PRAG_SUPERVISOR_LOCK=str(tmp_path / "sup.lock"))
    env["PRAG_SOCKET"] = str(sock)
    sup = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py")], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        proc, _ = run_app(["--index", str(corpus["root"])], env, timeout=120)
        assert json.loads(proc.stdout.strip().splitlines()[-1]).get("ok")
        q = ["--corpus", str(corpus["root"]), "--query-id", "query_01", "--query", "Which firmware version fixed ticket ACM-2210?"]
        run_app(q, env)
        assert read_out(tmp_path, "query_01")["answer"] == "7.1.4"
        # crash the worker; the supervisor restarts it and the persisted index answers again
        pid = None
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(sock))
        s.sendall(b'{"op": "ping"}\n')
        pid = json.loads(s.makefile().readline())["pid"]
        s.close()
        os.kill(pid, signal.SIGKILL)
        time.sleep(4)
        run_app(q[:3] + ["query_02", "--query", q[-1]], env)
        assert read_out(tmp_path, "query_02")["answer"] == "7.1.4"
    finally:
        os.killpg(sup.pid, signal.SIGTERM)
        sup.wait(timeout=10)
