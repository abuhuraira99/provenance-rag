"""Command line: provenance-rag {index, ask, inventory, bench, serve, studio}."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _engine(args, load_index: bool = True):
    from .engine import Engine

    engine = Engine.create(args.backend, index_directory=Path(args.index_dir) if args.index_dir else None)
    if load_index:
        engine.load()
    return engine


def cmd_index(args) -> int:
    engine = _engine(args, load_index=False)
    summary = engine.build_index(args.corpus, deadline=time.time() + args.budget if args.budget else None)
    print(json.dumps(summary, indent=2))
    for row in engine.inventory():
        print(f"  {row['status']:<22} {row['path']}  {row.get('detail', '')}")
    return 0


def cmd_ask(args) -> int:
    engine = _engine(args)
    res = engine.ask(args.question, corpus=args.corpus, deadline=time.time() + args.budget)
    if args.json:
        print(json.dumps(res.to_json() if args.debug else res.contract(), indent=2, ensure_ascii=False))
    else:
        print(f"answer:     {res.answer!r}")
        print(f"citations:  {res.citations}")
        print(f"confidence: {res.confidence}   mode: {res.mode}   {res.latency_ms} ms")
        if args.debug:
            for line in res.trace:
                print(f"  - {line}")
    return 0


def cmd_inventory(args) -> int:
    engine = _engine(args)
    if engine.index is None:
        print("no index; run `provenance-rag index CORPUS` first", file=sys.stderr)
        return 1
    print(json.dumps(engine.inventory(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("bench", "serve", "studio"):
        cmd, rest = argv[0], argv[1:]
        if cmd == "bench":
            from .bench import main as bench_main

            return bench_main(rest)
        if cmd == "serve":
            from .server import main as serve_main

            return serve_main(rest)
        from .studio.server import main as studio_main

        return studio_main(rest)

    ap = argparse.ArgumentParser(prog="provenance-rag", description="Question answering with exact file citations over a document folder.")
    ap.add_argument("--backend", default="rules", help="transformers | openai | stub | rules (default: rules, CPU)")
    ap.add_argument("--index-dir", help="index location (default: $PRAG_INDEX_DIR, /app/index or a temp dir)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="parse a folder and persist the index")
    p.add_argument("corpus")
    p.add_argument("--budget", type=float, default=0, help="seconds allowed for image OCR (0 = no limit)")
    p.set_defaults(func=cmd_index)
    p = sub.add_parser("ask", help="answer one question from the persisted index")
    p.add_argument("question")
    p.add_argument("--corpus", help="corpus (builds a text-only index if none exists)")
    p.add_argument("--budget", type=float, default=25.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true", help="show the evidence trace")
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("inventory", help="show per-file parse status of the persisted index")
    p.set_defaults(func=cmd_inventory)
    sub.add_parser("bench", help="score against a question set (see `provenance-rag bench -h`)")
    sub.add_parser("serve", help="run the resident model server (see `provenance-rag serve -h`)")
    sub.add_parser("studio", help="local web UI (see `provenance-rag studio -h`)")
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
