"""bench: run a question set (e.g. the organiser's sample-questions.json) end to end and write a report.

In-process (loads the backend here):
    python -m provenance_rag.bench --questions sample-questions.json --corpus mc3-corpus --backend transformers

Through the real contract (app.py -> resident server), as the grader runs it:
    python -m provenance_rag.bench --questions ... --corpus /app/corpus --via-app /app/app.py

The report (reports/<name>.json) holds per-question expected/got/pass/latency plus device info
(torch version, GPU name, peak VRAM). Results from the stub backend are labelled TEST STUB.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .normalize import answers_match, citations_match, norm_citation


def load_env_file(path: Path) -> list[str]:
    """KEY=VALUE lines into os.environ (values are never printed or written to the report)."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ[key] = value.strip().strip('"').strip("'")
        names.append(key)
    return names


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["queries"] if isinstance(data, dict) else data
    out = []
    for i, q in enumerate(items, 1):
        out.append({
            "n": q.get("n", i),
            "id": q.get("id") or f"query_{int(q.get('n', i)):02d}",
            "query": q["query"],
            "expected_answer": q.get("expected_answer", ""),
            "aliases": q.get("answer_aliases", []),
            "expected_citations": q.get("expected_citations", []),
            "category": q.get("category", ""),
        })
    return out


def score_item(item: dict, got: dict, latency_ms: int) -> dict:
    a_ok = answers_match(got.get("answer", ""), item["expected_answer"], item["aliases"])
    c_ok = citations_match(got.get("citations", []), item["expected_citations"])
    return {
        "n": item["n"], "id": item["id"], "query": item["query"], "category": item["category"],
        "expected_answer": item["expected_answer"], "expected_citations": item["expected_citations"],
        "got_answer": got.get("answer", ""), "got_citations": [norm_citation(c) for c in got.get("citations", [])],
        "confidence": got.get("confidence", 0.0), "answer_ok": a_ok, "citations_ok": c_ok, "pass": a_ok and c_ok,
        "latency_ms": latency_ms, "mode": got.get("mode", ""), "trace": got.get("trace", []),
    }


def summarize(items: list[dict]) -> dict:
    lat = sorted(i["latency_ms"] for i in items) or [0]
    passed = sum(1 for i in items if i["pass"])
    return {
        "passed": passed, "total": len(items), "points": passed * 20, "max_points": len(items) * 20,
        "answer_ok": sum(1 for i in items if i["answer_ok"]), "citations_ok": sum(1 for i in items if i["citations_ok"]),
        "latency_ms_mean": int(sum(lat) / len(lat)), "latency_ms_max": lat[-1],
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
    }


def run_in_process(args, questions) -> dict:
    from .engine import Engine

    idx_dir = Path(args.index_dir) if args.index_dir else Path(tempfile.mkdtemp(prefix="prag_bench_"))
    t0 = time.time()
    engine = Engine.create(args.backend, index_directory=idx_dir)
    load_s = time.time() - t0
    t1 = time.time()
    summary = engine.build_index(args.corpus)
    index_s = time.time() - t1
    items = []
    for q in questions:
        t = time.time()
        res = engine.ask(q["query"], deadline=time.time() + args.budget)
        got = res.to_json()
        items.append(score_item(q, got, int((time.time() - t) * 1000)))
        _print_item(items[-1])
    return {
        "mode": "in-process", "backend": engine.backend.name, "label": engine.backend.label(),
        "stub": engine.backend.is_stub, "device": engine.backend.device_info(), "model_load_s": round(load_s, 2),
        "index": summary, "index_s": round(index_s, 2), "inventory": engine.inventory(), "items": items,
    }


def run_via_app(args, questions) -> dict:
    app = Path(args.via_app)
    out_dir = Path(os.environ.get("PRAG_OUTPUT_DIR", "/app/output"))
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(app), "--index", str(args.corpus)], capture_output=True, text=True, timeout=900)
    index_s = time.time() - t0
    try:
        index_summary = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        index_summary = {"stdout": proc.stdout[-400:], "stderr": proc.stderr[-400:]}
    items = []
    for q in questions:
        t = time.time()
        try:
            subprocess.run([sys.executable, str(app), "--corpus", str(args.corpus), "--query-id", q["id"], "--query", q["query"]],
                           capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        latency = int((time.time() - t) * 1000)
        try:
            got = json.loads((out_dir / f"{q['id']}_output.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            got = {"answer": "", "citations": [], "malformed": True}
        items.append(score_item(q, got, latency))
        _print_item(items[-1])
    status = {}
    sys.path.insert(0, str(app.parent))
    try:
        import importlib

        client = importlib.import_module("app")
        status = client.request({"op": "status"}, timeout=10)
    except Exception as exc:
        status = {"error": str(exc)}
    device = status.get("device", {})
    return {
        "mode": "via-app", "backend": device.get("backend", "?"), "label": "TEST STUB" if device.get("stub") else device.get("backend", "?"),
        "stub": bool(device.get("stub")), "device": device, "index": index_summary, "index_s": round(index_s, 2),
        "inventory": status.get("inventory", []), "items": items,
    }


def _print_item(it: dict) -> None:
    flag = "PASS" if it["pass"] else ("ANSWER-OK/CITE-WRONG" if it["answer_ok"] else "FAIL")
    print(f"  {it['id']}: {flag:<20} got={it['got_answer']!r} {it['got_citations']}  ({it['latency_ms']} ms)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score Provenance RAG against a question set")
    ap.add_argument("--questions", required=True, type=Path, help="sample-questions.json")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--backend", default=os.environ.get("PRAG_BACKEND", "transformers"))
    ap.add_argument("--via-app", help="path to app.py: run through the contract instead of in-process")
    ap.add_argument("--index-dir", help="where to put the index (default: a temp dir)")
    ap.add_argument("--budget", type=float, default=25.0, help="per-question seconds (in-process)")
    ap.add_argument("--name", help="report name (default: <backend>-<timestamp>)")
    ap.add_argument("--model-label", default="", help="free-text model name recorded in the report")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--env-file", type=Path, help="load endpoint settings (e.g. OPENAI_*) from a KEY=VALUE file")
    args = ap.parse_args(argv)
    if args.env_file:
        print(f"loaded settings: {', '.join(load_env_file(args.env_file))}")

    questions = load_questions(args.questions)
    print(f"bench: {len(questions)} questions, corpus={args.corpus}", flush=True)
    t0 = time.time()
    report = run_via_app(args, questions) if args.via_app else run_in_process(args, questions)
    report.update({
        "name": args.name or f"{report['backend']}-{time.strftime('%Y%m%d-%H%M%S')}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "questions_file": args.questions.name,
        "corpus": args.corpus.name, "model_label": args.model_label, "host": platform.node() and "local",
        "python": platform.python_version(), "wall_s": round(time.time() - t0, 2),
        "summary": summarize(report["items"]),
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{report['name']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"{s['passed']}/{s['total']} passed ({s['points']}/{s['max_points']} points), "
          f"answers {s['answer_ok']}/{s['total']}, citations {s['citations_ok']}/{s['total']}, "
          f"mean {s['latency_ms_mean']} ms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
