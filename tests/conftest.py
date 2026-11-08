"""Shared fixtures: a synthetic messy corpus built at test time (no organiser files are used).

It mirrors the shape of the challenge (every file type, a withdrawn revision, a near-miss row,
a two-file chain, an image-only fact, an encrypted file holding a value, an unknown type, an
unreadable file and an empty directory) with its own invented values.
"""

from __future__ import annotations

import hashlib
import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from provenance_rag import parsers  # noqa: E402

try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover
    pymupdf = None

try:
    import openpyxl  # type: ignore
except ImportError:  # pragma: no cover
    openpyxl = None

needs_pdf = pytest.mark.skipif(pymupdf is None, reason="pymupdf not installed")
needs_xlsx = pytest.mark.skipif(openpyxl is None, reason="openpyxl not installed")


# ---------------------------------------------------------------------------- file builders


def make_png(width: int = 24, height: int = 12, seed: int = 0) -> bytes:
    raw = b"".join(b"\x00" + bytes(((x * 7 + y * 3 + seed) % 256 for x in range(width * 3))) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def make_docx(paragraphs: list[tuple[str, str]], table: list[list[str]] | None = None) -> bytes:
    """paragraphs: (style, text); style '' or 'Heading1'."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = []
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body.append(f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>")
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row) + "</w:tr>" for row in table)
        body.append(f"<w:tbl>{rows}</w:tbl>")
    doc = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>{"".join(body)}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", doc)
    return buf.getvalue()


def make_pdf(path: Path, lines: list[str], *, encrypt: bool = False) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    if encrypt:
        doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner-secret", user_pw="user-secret")
    else:
        doc.save(str(path))
    doc.close()


def make_xlsx(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Spares"
    ws.append(["part_number", "description", "compatible_with", "unit_cost_usd", "lead_time_days"])
    ws.append(["ACM-FAN-1100-A", "Blower, single fan (legacy)", "ZX-50", 40, 9])
    ws.append(["ACM-FAN-7781-C", "Blower assembly, twin fan, field replaceable", "ZX-90", 73.25, 18])
    ws.append(["ACM-PSU-3002-B", "Power shelf module", "ZX-90", 410, 42])
    notes = wb.create_sheet("Notes")
    notes.append(["Lead times exclude freight."])
    hidden = wb.create_sheet("Archive")
    hidden.append(["retired_part", "replacement"])
    hidden.append(["ACM-OLD-0001-A", "ACM-PSU-3002-B"])
    wb.save(str(path))


PIN_OCR = """ZX-90 rear connector - pin map
C21 C22 C23 C24
VCC SDA FAULT_N# GND
ASSOCIATIONS:
C21: VCC
C22: SDA
C23: FAULT_N#
C24: GND"""

LABEL_OCR = """Acme Instruments
MODEL ZX-90
BOARD REVISION REV-D7
SN AC9-551204-02"""


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    """Build the synthetic corpus. Returns {"root", "ocr_map", "unreadable"}."""
    root = tmp_path / "corpus"
    for d in ("archive", "engineering", "logs", "planning", "specs", "support", "vendor"):
        (root / d).mkdir(parents=True)
    (root / "engineering" / "retry_service.py").write_text(
        '"""Retry service for Acme telemetry uploads."""\n\n'
        "# Seconds an upload may wait in the retry queue before it is dropped.\n"
        "# Raised from 90 after the 2026-07 outage.\n"
        "DEFAULT_RETRY_WINDOW_S = 240\n\n"
        "MAX_PARALLEL_UPLOADS = 16\n\n\n"
        "def submit(job, window_s=DEFAULT_RETRY_WINDOW_S):\n"
        "    return job.enqueue(window_s)\n",
        encoding="utf-8",
    )
    (root / "engineering" / "kestrel_release_notes.txt").write_text(
        "Kestrel firmware - release notes\n\n"
        "7.1.2  Scheduler rework.\n7.1.3  Fixes a stall in the DMA path under heavy load.\n7.1.5  Packaging only.\n\n"
        "The ticket tracker is authoritative for which release closed a ticket.\n",
        encoding="utf-8",
    )
    (root / "logs" / "edge_node_2026-10-11.log").write_text(
        "2026-10-11T08:00:01Z edge-node-3 kestrel[77]: INFO  scheduler: job 5521 accepted\n"
        "2026-10-11T08:02:40Z edge-node-3 kestrel[77]: WARN  thermal: sensor 2 at 89C\n"
        "2026-10-11T08:02:55Z edge-node-3 kestrel[77]: ERROR W4410: thermal throttle engaged on sensor 2\n"
        "2026-10-11T08:07:10Z edge-node-3 kestrel[77]: INFO  thermal: throttle released\n"
        "2026-10-11T08:07:10Z edge-node-3 kestrel[77]: INFO  incident logged against ACM-2210\n",
        encoding="utf-8",
    )
    (root / "support" / "tickets.csv").write_text(
        "ticket,component,severity,summary,status,fixed_in\n"
        "ACM-2210,power,S2,Rail monitor misreads under bursty load,closed,7.1.4\n"
        "ACM-2195,dma,S3,DMA stall under heavy load,closed,7.1.3\n"
        "ACM-2231,scheduler,S3,Queue depth counter drifts,open,\n",
        encoding="utf-8",
    )
    (root / "planning" / "roadmap_fy28.docx").write_bytes(make_docx(
        [("Heading1", "Acme Roadmap (Internal)"), ("", "Targets are not commitments."), ("Heading1", "Next generation")],
        table=[["Product", "Milestone", "Target"], ["ZX-120", "customer sampling", "Q1 FY28"], ["ZX-120", "volume production", "Q4 FY28"]],
    ))
    png_pin = make_png(seed=1)
    png_label = make_png(seed=2)
    (root / "specs" / "rear_connector.png").write_bytes(png_pin)
    (root / "support" / "board_label.jpg").write_bytes(png_label)  # PNG bytes with .jpg name: magic sniffing handles it
    (root / "vendor" / "capture.dat").write_bytes(bytes(range(256)) * 8)
    (root / "vendor" / "restricted_notes.txt").write_text("restricted: list price 9999\n", encoding="utf-8")
    if pymupdf is not None:
        make_pdf(root / "specs" / "zx90_datasheet_r1_WITHDRAWN.pdf", [
            'Acme ZX-90 "Kestrel" Controller', "Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2",
            "Maximum junction temperature .......... 112 C", "Board power ........ 300 W"])
        make_pdf(root / "specs" / "zx90_datasheet_r2.pdf", [
            'Acme ZX-90 "Kestrel" Controller', "Datasheet, revision 2 - supersedes revision 1",
            "Maximum junction temperature .......... 101 C", "Board power ........ 320 W",
            "r2  Junction temperature corrected to 101 C. Revision 1 is withdrawn."])
        make_pdf(root / "vendor" / "pricing_ENCRYPTED.pdf", ["ZX-90 unit price at 5,000 units: 6150 USD"], encrypt=True)
    if openpyxl is not None:
        make_xlsx(root / "support" / "spares.xlsx")
    ocr_map = {hashlib.sha256(png_pin).hexdigest(): PIN_OCR, hashlib.sha256(png_label).hexdigest(): LABEL_OCR}
    return {"root": root, "ocr_map": ocr_map, "unreadable": root / "vendor" / "restricted_notes.txt"}


@pytest.fixture
def deny_unreadable(monkeypatch, corpus):
    """Simulate a chmod-000 file (Windows and root cannot reproduce EACCES with chmod)."""
    real = parsers.read_bytes
    target = corpus["unreadable"].resolve()

    def fake(path, limit=None):
        if Path(path).resolve() == target:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, limit)

    monkeypatch.setattr(parsers, "read_bytes", fake)
    return target


@pytest.fixture
def engine_factory(tmp_path, corpus, deny_unreadable):
    from provenance_rag.backends import NoModelBackend
    from provenance_rag.backends.stub import StubBackend
    from provenance_rag.engine import Engine

    def build(kind: str = "stub", responder=None):
        if kind == "rules":
            backend = NoModelBackend()
        else:
            backend = StubBackend(ocr_map=corpus["ocr_map"], responder=responder)
        eng = Engine(backend, tmp_path / f"index_{kind}")
        eng.build_index(corpus["root"])
        return eng

    return build
