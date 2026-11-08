"""Optional: run the organiser's sample corpus through the CPU rules path, when the starter kit is
available locally (it is never copied into this repository). Point PRAG_SAMPLE_KIT at the
unpacked mc3-starter-kit directory, or keep it at ../../specs/mc3-starter-kit."""

import json
import os
from pathlib import Path

import pytest

from provenance_rag.normalize import answers_match, citations_match

KIT = Path(os.environ.get("PRAG_SAMPLE_KIT", Path(__file__).resolve().parents[3] / "specs" / "mc3-starter-kit"))
pytestmark = pytest.mark.skipif(not (KIT / "sample-questions.json").is_file(), reason="starter kit not available")


def test_rules_path_on_sample_corpus(tmp_path):
    pytest.importorskip("pymupdf")
    pytest.importorskip("openpyxl")
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.engine import Engine

    eng = Engine(NoModelBackend(), tmp_path / "idx")
    eng.build_index(KIT / "mc3-corpus")
    queries = json.loads((KIT / "sample-questions.json").read_text(encoding="utf-8"))["queries"]
    results = {}
    for q in queries:
        r = eng.ask(q["query"])
        results[q["n"]] = answers_match(r.answer, q["expected_answer"], q.get("answer_aliases", [])) and citations_match(r.citations, q["expected_citations"])
        if "image" in q.get("category", "") or "multimodal" in q.get("category", ""):
            # Image-only facts need the vision model; without it the right behaviour is to refuse.
            assert r.answer == "" and r.citations == []
    text_only = [n for n, q in ((q["n"], q) for q in queries) if "multimodal" not in q.get("category", "")]
    assert all(results[n] for n in text_only), results
