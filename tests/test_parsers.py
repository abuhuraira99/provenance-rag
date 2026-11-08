"""Parsing and the four corpus traps: empty dir, unknown type, unreadable file, encrypted PDF."""

import io
import os
import zipfile
from pathlib import Path

import pytest

from provenance_rag import parsers
from provenance_rag.parsers import parse_file, walk_corpus
from provenance_rag.revisions import explicit_withdrawal, question_targets_old, revision_status
from provenance_rag.textutil import extract_identifiers, question_quantities, text_has_quantity, tokenize

from conftest import needs_pdf, needs_xlsx


def _by_path(root):
    files, problems = walk_corpus(root)
    return {pf.rel: pf for pf in (parse_file(root, f) for f in files)}, problems


def test_walk_survives_all_traps(corpus, deny_unreadable):
    parsed, problems = _by_path(corpus["root"])
    assert problems == []
    assert parsed["vendor/capture.dat"].status == parsers.UNKNOWN
    assert parsed["vendor/restricted_notes.txt"].status == parsers.UNREADABLE
    assert parsed["vendor/restricted_notes.txt"].blocks == []
    # files after the unreadable one in listing order are still parsed
    assert parsed["support/tickets.csv"].status == parsers.INDEXED
    assert parsed["specs/rear_connector.png"].status == parsers.PENDING_OCR
    assert not any(p.startswith("archive") for p in parsed)  # empty dir: nothing, and no error


def test_walk_order_does_not_matter(corpus, deny_unreadable, monkeypatch):
    first, _ = _by_path(corpus["root"])
    real_scandir = parsers.os.scandir

    class Reversed:
        def __init__(self, path):
            self.it = real_scandir(path)

        def __enter__(self):
            return reversed(list(self.it.__enter__()))

        def __exit__(self, *a):
            return self.it.__exit__(*a)

    monkeypatch.setattr(parsers.os, "scandir", Reversed)
    second, _ = _by_path(corpus["root"])
    assert {k: v.status for k, v in first.items()} == {k: v.status for k, v in second.items()}


def test_unreadable_directory_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "locked").mkdir()
    real = parsers.os.scandir

    def fake(path):
        if Path(path).name == "locked":
            raise PermissionError(13, "denied")
        return real(path)

    monkeypatch.setattr(parsers.os, "scandir", fake)
    files, problems = walk_corpus(tmp_path)
    assert [f.name for f in files] == ["a.txt"]
    assert problems and problems[0]["status"] == parsers.UNREADABLE


def test_oserror_on_read_is_caught(tmp_path, monkeypatch):
    p = tmp_path / "x.log"
    p.write_text("hello", encoding="utf-8")

    def boom(path, limit=None):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(parsers, "read_bytes", boom)
    assert parse_file(tmp_path, p).status == parsers.UNREADABLE


@needs_pdf
def test_encrypted_pdf_is_skipped_never_extracted(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "vendor" / "pricing_ENCRYPTED.pdf")
    assert pf.status == parsers.ENCRYPTED
    assert pf.blocks == [] and pf.ocr_jobs == []


@needs_pdf
def test_pdf_text(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "specs" / "zx90_datasheet_r2.pdf")
    text = "\n".join(b.text for b in pf.blocks)
    assert "101 C" in text and pf.blocks[0].loc == "page 1"


def test_password_protected_office_file(tmp_path):
    p = tmp_path / "secret.docx"
    p.write_bytes(parsers.OLE_MAGIC + b"\x00" * 600)
    assert parse_file(tmp_path, p).status == parsers.ENCRYPTED


def test_docx_paragraphs_and_tables(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "planning" / "roadmap_fy28.docx")
    texts = [b.text for b in pf.blocks]
    assert any("Product: ZX-120 | Milestone: customer sampling | Target: Q1 FY28" == t for t in texts)
    assert any(b.unit == "heading" for b in pf.blocks)


def test_corrupt_docx_is_a_parse_error(tmp_path):
    p = tmp_path / "bad.docx"
    p.write_bytes(b"PK\x03\x04 not really a zip")
    assert parse_file(tmp_path, p).status == parsers.ERROR


@needs_xlsx
def test_xlsx_all_sheets(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "support" / "spares.xlsx")
    texts = [b.text for b in pf.blocks]
    assert any("part_number: ACM-FAN-7781-C" in t and "unit_cost_usd: 73.25" in t for t in texts)
    assert any("Lead times exclude freight." in t for t in texts)
    assert any("ACM-OLD-0001-A" in t for t in texts)  # third sheet


@needs_xlsx
def test_xlsx_stdlib_fallback_matches(corpus):
    data = (corpus["root"] / "support" / "spares.xlsx").read_bytes()
    sheets = parsers._xlsx_stdlib(zipfile.ZipFile(io.BytesIO(data)))
    names = [s for s, _ in sheets]
    assert names == ["Spares", "Notes", "Archive"]
    flat = [c for _, rows in sheets for _, cells in rows for c in cells]
    assert "ACM-FAN-7781-C" in flat and "73.25" in flat


def test_csv_rows_carry_headers(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "support" / "tickets.csv")
    assert pf.blocks[1].text.startswith("ticket: ACM-2210") and "fixed_in: 7.1.4" in pf.blocks[1].text


def test_python_constants_with_comments(corpus):
    pf = parse_file(corpus["root"], corpus["root"] / "engineering" / "retry_service.py")
    sym = [b for b in pf.blocks if b.heading.startswith("constant")]
    assert any("DEFAULT_RETRY_WINDOW_S = 240" in b.text and "Raised from 90" in b.text for b in sym)


def test_binary_in_text_extension_is_unknown(tmp_path):
    p = tmp_path / "weird.txt"
    p.write_bytes(b"\x00\x01\x02\x03" * 500)
    assert parse_file(tmp_path, p).status == parsers.UNKNOWN


def test_utf16_text(tmp_path):
    p = tmp_path / "u.log"
    p.write_bytes("ERROR E1234 boom".encode("utf-16"))
    pf = parse_file(tmp_path, p)
    assert pf.blocks and "E1234" in pf.blocks[0].text


def test_huge_text_is_truncated_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(parsers, "MAX_TEXT_BYTES", 1000)
    p = tmp_path / "big.log"
    p.write_text("line of log text\n" * 500, encoding="utf-8")
    pf = parse_file(tmp_path, p)
    assert pf.status == parsers.INDEXED and "truncated" in pf.detail


def test_empty_corpus(tmp_path):
    files, problems = walk_corpus(tmp_path)
    assert files == [] and problems == []


def test_revision_detection():
    assert explicit_withdrawal("specs/x_r1_WITHDRAWN.pdf", "")
    assert explicit_withdrawal("specs/x.pdf", "Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2")
    assert not explicit_withdrawal("specs/x_r2.pdf", "Datasheet, revision 2 - supersedes revision 1\n... Revision 1 is withdrawn.")
    st = revision_status([{"path": "a/spec_r1.pdf", "head_text": ""}, {"path": "a/spec_r2.pdf", "head_text": ""}, {"path": "a/other.pdf", "head_text": ""}])
    assert st["a/spec_r1.pdf"]["status"] == "older-revision" and st["a/spec_r1.pdf"]["superseded_by"] == "a/spec_r2.pdf"
    assert st["a/spec_r2.pdf"]["status"] == "current" and st["a/other.pdf"]["status"] == "current"
    assert question_targets_old("What did the withdrawn revision 1 datasheet list?")
    assert not question_targets_old("What is the maximum junction temperature of the TQ-40?")


def test_tokenizer_splits_identifiers():
    toks = tokenize("DEFAULT_BATCH_TIMEOUT_S = 180; THERM_ALERT# on ORR-1847")
    for t in ("default", "batch", "timeout", "therm_alert", "alert", "orr-1847", "orr1847", "180"):
        assert t in toks
    assert set(extract_identifiers("ticket ORR-1847 fixed in 4.3.2 at 94C on 2026-09-02")) == {"ORR-1847", "4.3.2"}


def test_question_quantities():
    assert question_quantities("unit price of the TQ-40 at 10,000 unit volume?") == ["10,000"]
    assert question_quantities("Which firmware fixed ORR-1847 in 2026?") == []
    assert text_has_quantity("price at 10k units", "10,000")
    assert text_has_quantity("10 000 units", "10,000")
    assert not text_has_quantity("1,000 units", "10,000")


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="chmod 000 only denies non-root POSIX users")
def test_real_chmod_000_file_and_dir(tmp_path):
    (tmp_path / "a_locked.txt").write_text("secret 4242", encoding="utf-8")
    (tmp_path / "b_ok.txt").write_text("visible 1111", encoding="utf-8")
    (tmp_path / "c_dir").mkdir()
    (tmp_path / "c_dir" / "inner.txt").write_text("hidden", encoding="utf-8")
    os.chmod(tmp_path / "a_locked.txt", 0)
    os.chmod(tmp_path / "c_dir", 0)
    try:
        parsed, problems = _by_path(tmp_path)
        assert parsed["a_locked.txt"].status == parsers.UNREADABLE
        assert parsed["b_ok.txt"].status == parsers.INDEXED
        assert any(p["path"] == "c_dir" for p in problems)
    finally:
        os.chmod(tmp_path / "c_dir", 0o755)
        os.chmod(tmp_path / "a_locked.txt", 0o644)
