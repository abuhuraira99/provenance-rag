"""End-to-end answering on the synthetic corpus: exact citation sets, traps, multi-hop, refusal,
and the verifier's behaviour when a (scripted) model over-cites, under-cites or hallucinates."""

import json
import re

import pytest

from provenance_rag.normalize import answers_match

from conftest import needs_pdf, needs_xlsx

TEXT_CASES = [
    ("What is the default retry window, in seconds, in the retry service?", "240", ["engineering/retry_service.py"]),
    ("What error code is logged when the thermal throttle engages?", "W4410", ["logs/edge_node_2026-10-11.log"]),
    ("Which firmware version fixed ticket ACM-2210?", "7.1.4", ["support/tickets.csv"]),
    ("In which quarter does the ZX-120 enter customer sampling?", "Q1 FY28", ["planning/roadmap_fy28.docx"]),
    # multi-hop: the log names the ticket; only the ticket table has the fix version
    ("The edge node log shows a thermal throttle incident. Which firmware release fixed the underlying defect?", "7.1.4",
     ["logs/edge_node_2026-10-11.log", "support/tickets.csv"]),
]
PDF_CASES = [("What is the maximum junction temperature of the ZX-90?", "101", ["specs/zx90_datasheet_r2.pdf"])]
XLSX_CASES = [("What is the part number of the field-replaceable blower assembly for the ZX-90?", "ACM-FAN-7781-C", ["support/spares.xlsx"])]
IMAGE_CASES = [
    ("Which rear connector pin carries FAULT_N# on the ZX-90?", "C23", ["specs/rear_connector.png"]),
    ("What board revision is printed on the board label?", "REV-D7", ["support/board_label.jpg"]),
]
REFUSALS = [
    "What is the unit price of the ZX-90 at 5,000 unit volume?",  # only inside the encrypted PDF
    "What is the list price in the restricted notes?",  # only inside the unreadable file
    "Who is the chief executive of the company?",
]


def _check(eng, q, answer, cites):
    r = eng.ask(q)
    assert answers_match(r.answer, answer), (q, r.answer, r.trace)
    assert sorted(r.citations) == sorted(cites), (q, r.citations, r.trace)
    return r


@pytest.mark.parametrize("kind", ["stub", "rules"])
@pytest.mark.parametrize("q,answer,cites", TEXT_CASES)
def test_text_questions(engine_factory, kind, q, answer, cites):
    _check(engine_factory(kind), q, answer, cites)


@needs_pdf
@pytest.mark.parametrize("kind", ["stub", "rules"])
@pytest.mark.parametrize("q,answer,cites", PDF_CASES)
def test_withdrawn_revision_never_wins(engine_factory, kind, q, answer, cites):
    r = _check(engine_factory(kind), q, answer, cites)
    assert "specs/zx90_datasheet_r1_WITHDRAWN.pdf" not in r.citations


@needs_xlsx
@pytest.mark.parametrize("kind", ["stub", "rules"])
@pytest.mark.parametrize("q,answer,cites", XLSX_CASES)
def test_xlsx_near_miss_row(engine_factory, kind, q, answer, cites):
    _check(engine_factory(kind), q, answer, cites)


@pytest.mark.parametrize("q,answer,cites", IMAGE_CASES)
def test_image_only_facts_via_vision_ocr(engine_factory, q, answer, cites):
    _check(engine_factory("stub"), q, answer, cites)


def test_image_questions_refuse_without_vision(engine_factory):
    eng = engine_factory("rules")
    for q, _, _ in IMAGE_CASES:
        r = eng.ask(q)
        assert r.answer == "" and r.citations == []


@pytest.mark.parametrize("kind", ["stub", "rules"])
@pytest.mark.parametrize("q", REFUSALS)
def test_refusals_are_empty(engine_factory, kind, q):
    r = engine_factory(kind).ask(q)
    assert r.contract() == {"answer": "", "citations": [], "confidence": 0.0}, r.trace


def test_inventory_records_every_trap(engine_factory):
    inv = {f["path"]: f for f in engine_factory("stub").inventory()}
    assert inv["vendor/capture.dat"]["status"] == "skipped-unknown-type"
    assert inv["vendor/restricted_notes.txt"]["status"] == "unreadable"
    assert inv["specs/rear_connector.png"]["status"] == "indexed"
    if "vendor/pricing_ENCRYPTED.pdf" in inv:
        assert inv["vendor/pricing_ENCRYPTED.pdf"]["status"] == "skipped-encrypted"
        assert inv["specs/zx90_datasheet_r1_WITHDRAWN.pdf"]["revision"]["status"] == "withdrawn"


def test_index_persists_and_reloads(engine_factory, tmp_path):
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.engine import Engine

    eng = engine_factory("stub")
    fresh = Engine(NoModelBackend(), eng.index_dir)
    assert fresh.load()
    assert len(fresh.index.chunks) == len(eng.index.chunks)
    # OCR text survives the reload, so a new process can answer image questions without a model
    r = fresh.ask("What board revision is printed on the board label?")
    assert answers_match(r.answer, "REV-D7") and r.citations == ["support/board_label.jpg"]


# ------------------------------------------------------------------ verifier vs. scripted model replies


def _scripted(reply_for):
    """Responder that sees the prompt and returns JSON built by reply_for(passages_by_file)."""

    def responder(system, prompt):
        blocks = re.findall(r"^\[(\d+)\] ([^\s|]+)", prompt, flags=re.MULTILINE)
        by_file = {}
        for n, f in blocks:
            by_file.setdefault(f, int(n))
        return json.dumps(reply_for(by_file))

    return responder


def test_overcitation_of_topic_file_is_dropped(engine_factory):
    def reply(by_file):
        return {"answer": "7.1.4", "sources": [by_file["support/tickets.csv"], by_file.get("engineering/kestrel_release_notes.txt", 1)], "needs": []}

    eng = engine_factory("stub", responder=_scripted(reply))
    r = eng.ask("Which firmware version fixed ticket ACM-2210?")
    assert r.answer == "7.1.4" and r.citations == ["support/tickets.csv"]


def test_bridge_named_by_question_is_not_cited(engine_factory):
    # The log mentions ACM-2210 too, but the question names the ticket itself: the log is not necessary.
    def reply(by_file):
        return {"answer": "7.1.4", "sources": [by_file["support/tickets.csv"], by_file.get("logs/edge_node_2026-10-11.log", 1)], "needs": []}

    r = engine_factory("stub", responder=_scripted(reply)).ask("Which firmware version fixed ticket ACM-2210?")
    assert r.citations == ["support/tickets.csv"]


def test_missing_bridge_is_added(engine_factory):
    def reply(by_file):
        return {"answer": "7.1.4", "sources": [by_file["support/tickets.csv"]], "needs": []}

    r = engine_factory("stub", responder=_scripted(reply)).ask(
        "The edge node log shows a thermal throttle incident. Which firmware release fixed the underlying defect?")
    assert sorted(r.citations) == ["logs/edge_node_2026-10-11.log", "support/tickets.csv"], r.trace


def test_hallucinated_answer_is_refused(engine_factory):
    r = engine_factory("stub", responder=_scripted(lambda by_file: {"answer": "9.9.9", "sources": [1], "needs": []})).ask(
        "Which firmware version fixed ticket ACM-2210?")
    assert r.contract() == {"answer": "", "citations": [], "confidence": 0.0}


def test_value_for_other_quantity_is_refused(engine_factory):
    # A model that answers a price question with a spares cost: the question's 5,000 is not in the evidence.
    r = engine_factory("stub", responder=_scripted(lambda by_file: {"answer": "73.25", "sources": [1], "needs": []})).ask(
        "What is the unit price of the ZX-90 at 5,000 unit volume?")
    assert r.answer == "" and r.citations == []


@needs_pdf
def test_withdrawn_value_from_model_is_refused(engine_factory):
    # Withdrawn files are never shown to the model unless the question asks for them, so "112"
    # cannot be grounded and is refused.
    r = engine_factory("stub", responder=_scripted(lambda by_file: {"answer": "112", "sources": [1], "needs": []})).ask(
        "What is the maximum junction temperature of the ZX-90?")
    assert r.answer == ""


@needs_pdf
def test_question_about_withdrawn_revision_may_use_it(engine_factory):
    def reply(by_file):
        return {"answer": "112", "sources": [by_file["specs/zx90_datasheet_r1_WITHDRAWN.pdf"]], "needs": []}

    r = engine_factory("stub", responder=_scripted(reply)).ask(
        "What maximum junction temperature did the withdrawn revision 1 ZX-90 datasheet list?")
    assert r.answer == "112" and r.citations == ["specs/zx90_datasheet_r1_WITHDRAWN.pdf"]


def test_needs_follow_up_round(engine_factory):
    calls = {"n": 0}

    def responder(system, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"answer": "", "sources": [], "needs": ["ACM-2195"]})
        m = re.search(r"^\[(\d+)\] support/tickets\.csv \| row 3", prompt, flags=re.MULTILINE)
        return json.dumps({"answer": "7.1.3", "sources": [int(m.group(1))] if m else [], "needs": []})

    r = engine_factory("stub", responder=responder).ask("Which release fixed ticket ACM-2195?")
    assert r.answer == "7.1.3" and r.citations == ["support/tickets.csv"]


def test_unparseable_model_reply_falls_back_to_rules(engine_factory):
    r = engine_factory("stub", responder=lambda s, p: "I think it is probably fine").ask("Which firmware version fixed ticket ACM-2210?")
    assert r.answer == "7.1.4" and r.mode == "rules"


def test_model_exception_falls_back_to_rules(engine_factory):
    def boom(system, prompt):
        raise RuntimeError("HIP error")

    r = engine_factory("stub", responder=boom).ask("What error code is logged when the thermal throttle engages?")
    assert r.answer == "W4410" and r.mode == "rules"


def test_sources_given_as_paths(engine_factory):
    r = engine_factory("stub", responder=lambda s, p: json.dumps({"answer": "W4410", "sources": ["logs/edge_node_2026-10-11.log"]})).ask(
        "What error code is logged when the thermal throttle engages?")
    assert r.citations == ["logs/edge_node_2026-10-11.log"]


def test_empty_index_refuses(tmp_path):
    from provenance_rag.backends.stub import StubBackend
    from provenance_rag.engine import Engine

    (tmp_path / "c").mkdir()
    eng = Engine(StubBackend(), tmp_path / "idx")
    eng.build_index(tmp_path / "c")
    assert eng.ask("anything?").contract() == {"answer": "", "citations": [], "confidence": 0.0}


# ------------------------------------------------------------------ answer kind must fit the question


def _tiny_engine(tmp_path, kind, responder=None):
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.backends.stub import StubBackend
    from provenance_rag.engine import Engine

    root = tmp_path / "tiny"
    (root / "specs").mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(exist_ok=True)
    (root / "specs" / "tq40_spec.txt").write_text("TQ40 accelerator spec\nmaximum junction temperature ........ 94\n", encoding="utf-8")
    (root / "notes" / "company.txt").write_text(
        "The company that makes the TQ40 runs 2 fabs and was founded with 3 product lines.\n"
        "TQ60 enters customer sampling in Q3 FY27.\n", encoding="utf-8")
    backend = NoModelBackend() if kind == "rules" else StubBackend(responder=responder)
    eng = Engine(backend, tmp_path / f"idx_{kind}")
    eng.build_index(root)
    return eng


@pytest.mark.parametrize("kind", ["stub", "rules"])
@pytest.mark.parametrize("q", [
    "Who founded the company that makes the TQ40?",
    "Which company makes the TQ40?",
    "When was the company that makes the TQ40 founded?",
])
def test_question_kind_mismatch_is_refused(tmp_path, kind, q):
    r = _tiny_engine(tmp_path, kind).ask(q)
    assert r.contract() == {"answer": "", "citations": [], "confidence": 0.0}, r.trace


def test_model_number_for_who_question_is_refused(tmp_path):
    eng = _tiny_engine(tmp_path, "stub", responder=lambda s, p: json.dumps({"answer": "2", "sources": [1]}))
    assert eng.ask("Who founded the company that makes the TQ40?").answer == ""


@pytest.mark.parametrize("kind", ["stub", "rules"])
def test_matching_kinds_still_answer(tmp_path, kind):
    eng = _tiny_engine(tmp_path, kind)
    assert eng.ask("What is the maximum junction temperature of the TQ40?").answer == "94"
    r = eng.ask("When does the TQ60 enter customer sampling?")
    assert r.answer == "Q3 FY27" and r.citations == ["notes/company.txt"], r.trace


def test_kind_rules():
    from provenance_rag.answertype import DATE, NAME, NUMBER, expected_kind, kind_matches

    assert expected_kind("Who founded Orrery?") == NAME
    assert expected_kind("Which supplier ships the fan?") == NAME
    assert expected_kind("When was the board revised?") == DATE
    assert expected_kind("How many fans does it have?") == NUMBER
    assert expected_kind("What is the maximum junction temperature of the TQ-40?") == "any"
    assert not kind_matches(NAME, "2") and not kind_matches(NAME, "94 C") and kind_matches(NAME, "Orrery Systems")
    assert kind_matches(DATE, "Q3 FY27") and kind_matches(DATE, "2026-09-02") and not kind_matches(DATE, "3")
    assert kind_matches(NUMBER, "180") and not kind_matches(NUMBER, "fast")
