# Changelog

## Unreleased

- Answer-kind check for every backend: a who/which-company question needs a name-like value, a when
  question a date-like value (incl. quarters and fiscal years), a how-many question a number;
  mismatches are refused instead of cited. The rules answerer refuses name questions outright.

## 0.1.0 - 2026-11-08

- First release.
- Parsers: PDF (pymupdf, pypdf fallback; encrypted detection; scanned-page OCR), DOCX (stdlib XML:
  paragraphs, headings, tables, headers/footers, notes, embedded images), XLSX (all sheets, openpyxl
  with a stdlib fallback), CSV/TSV, text and logs, Python (constants, defaults and their comments),
  PNG/JPG/TIFF via the vision model.
- Robust corpus walk: empty directories, unknown types, unreadable files and directories, encrypted
  and oversized files become inventory rows, never failures; order independent.
- Hybrid lexical retrieval (BM25 + exact identifiers + phrase bonus) with revision awareness
  (WITHDRAWN / superseded / older revisions).
- Multi-hop identifier expansion and a model-requested follow-up lookup.
- Strict-JSON answering with grounding, quantity and identifier checks, and necessity-based citation
  selection (answer file plus identifier bridges only).
- Rules answerer for CPU fallback; backends: transformers (Qwen3-VL), OpenAI-compatible, stub.
- Resident server with supervisor, thin stdlib `app.py` client with hard deadlines and atomic writes.
- `bench` command with device info; Studio UI with Live and Run-report modes.
