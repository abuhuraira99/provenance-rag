"""Corpus walk and per-format parsers.

Every parser returns a `ParsedFile` and never raises: an unreadable, encrypted, unknown, corrupt
or oversized file becomes a status in the inventory, and the walk carries on. The walk is sorted
and each file is parsed independently, so which files get indexed never depends on listing order.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

from .normalize import format_number

TEXT_EXTS = {
    ".txt", ".log", ".md", ".rst", ".text", ".out", ".err", ".ini", ".cfg", ".conf", ".toml",
    ".yaml", ".yml", ".json", ".jsonl", ".xml", ".html", ".htm", ".sql", ".sh",
}
CODE_EXTS = {".py", ".pyi"}
TABLE_EXTS = {".csv", ".tsv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
SUPPORTED_EXTS = TEXT_EXTS | CODE_EXTS | TABLE_EXTS | IMAGE_EXTS | {".pdf", ".docx", ".xlsx", ".xlsm"}

MAX_BINARY_BYTES = 256 * 1024 * 1024  # PDFs / Office files larger than this are skipped
MAX_TEXT_BYTES = 48 * 1024 * 1024  # text files are read up to this many bytes
MAX_PDF_PAGES = 3000
MAX_ROWS = 200_000
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # password-protected Office files are OLE containers

# Status values shown in the inventory.
INDEXED = "indexed"
EMPTY = "empty"
UNKNOWN = "skipped-unknown-type"
ENCRYPTED = "skipped-encrypted"
UNREADABLE = "unreadable"
TOO_LARGE = "skipped-too-large"
ERROR = "parse-error"
PENDING_OCR = "pending-ocr"


@dataclass
class Block:
    """A unit of extracted text with a human-readable location ("page 2", "row 14", "lines 3-9")."""

    text: str
    loc: str
    unit: str = "para"  # para | line | row | code | ocr | heading
    heading: str = ""


@dataclass
class OcrJob:
    """An image whose text must come from the vision model (an image file, a scanned PDF page,
    or a picture embedded in a .docx). `data` is the encoded image (png/jpg) bytes."""

    loc: str
    data: bytes
    suffix: str = ".png"

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class ParsedFile:
    rel: str
    ext: str
    size: int = 0
    status: str = INDEXED
    detail: str = ""
    title: str = ""
    blocks: list[Block] = field(default_factory=list)
    ocr_jobs: list[OcrJob] = field(default_factory=list)
    head_text: str = ""  # first ~600 characters, used for revision / withdrawal detection
    sha: str = ""

    def inventory(self) -> dict:
        return {
            "path": self.rel,
            "type": self.ext.lstrip(".") or "(none)",
            "size": self.size,
            "status": self.status,
            "detail": self.detail,
            "title": self.title,
            "blocks": len(self.blocks),
            "ocr_jobs": len(self.ocr_jobs),
        }


# ------------------------------------------------------------------------------------------------
# Walk


def walk_corpus(root: Path) -> tuple[list[Path], list[dict]]:
    """All regular files under root (sorted, symlinked dirs not followed) plus walk problems
    (unreadable directories). Never raises."""
    files: list[Path] = []
    problems: list[dict] = []
    stack = [root]
    seen_dirs: set = set()
    while stack:
        d = stack.pop()
        try:
            st = d.stat()
            key = (st.st_dev, st.st_ino)
            if key in seen_dirs and st.st_ino:
                continue
            seen_dirs.add(key)
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            problems.append({"path": _rel(root, d), "status": UNREADABLE, "detail": f"directory: {type(exc).__name__}"})
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=True):
                    files.append(Path(e.path))
            except OSError as exc:
                problems.append({"path": _rel(root, Path(e.path)), "status": UNREADABLE, "detail": type(exc).__name__})
    files.sort(key=lambda p: _rel(root, p))
    return files, problems


def _rel(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def read_bytes(path: Path, limit: int | None = None) -> bytes:
    """The single point where corpus bytes are read (tests patch this to simulate EACCES)."""
    with open(path, "rb") as fh:
        return fh.read() if limit is None else fh.read(limit)


# ------------------------------------------------------------------------------------------------
# Dispatcher


def parse_file(root: Path, path: Path) -> ParsedFile:
    rel = _rel(root, path)
    ext = path.suffix.lower()
    pf = ParsedFile(rel=rel, ext=ext)
    try:
        pf.size = path.stat().st_size
    except OSError as exc:
        pf.status, pf.detail = UNREADABLE, f"stat failed: {type(exc).__name__}"
        return pf
    name = path.name
    if name.startswith("~$") or ext not in SUPPORTED_EXTS:
        pf.status, pf.detail = UNKNOWN, f"no parser for '{ext or name}'"
        return pf
    is_text = ext in TEXT_EXTS or ext in CODE_EXTS or ext in TABLE_EXTS
    if not is_text and pf.size > MAX_BINARY_BYTES:
        pf.status, pf.detail = TOO_LARGE, f"{pf.size} bytes"
        return pf
    try:
        data = read_bytes(path, MAX_TEXT_BYTES if is_text else None)
    except PermissionError as exc:
        pf.status, pf.detail = UNREADABLE, f"permission denied ({exc.errno})"
        return pf
    except OSError as exc:
        pf.status, pf.detail = UNREADABLE, f"{type(exc).__name__}: {exc.strerror or exc}"
        return pf
    except Exception as exc:  # pragma: no cover - defensive
        pf.status, pf.detail = UNREADABLE, type(exc).__name__
        return pf
    pf.sha = hashlib.sha256(data).hexdigest()
    try:
        handler = _HANDLERS.get(ext)
        if handler is None:
            if ext in TEXT_EXTS:
                handler = _parse_text
            elif ext in IMAGE_EXTS:
                handler = _parse_image
            else:  # pragma: no cover
                pf.status = UNKNOWN
                return pf
        handler(pf, data)
        if is_text and len(data) >= MAX_TEXT_BYTES:
            pf.detail = (pf.detail + "; " if pf.detail else "") + f"truncated at {MAX_TEXT_BYTES} bytes"
    except _Encrypted as exc:
        pf.blocks, pf.ocr_jobs = [], []
        pf.status, pf.detail = ENCRYPTED, str(exc) or "password protected"
    except _Unknown as exc:
        pf.blocks, pf.ocr_jobs = [], []
        pf.status, pf.detail = UNKNOWN, str(exc)
    except MemoryError:
        pf.blocks, pf.ocr_jobs = [], []
        pf.status, pf.detail = TOO_LARGE, "out of memory while parsing"
    except Exception as exc:
        pf.blocks, pf.ocr_jobs = [], []
        pf.status, pf.detail = ERROR, f"{type(exc).__name__}: {str(exc)[:160]}"
    if pf.status == INDEXED:
        if not pf.blocks and pf.ocr_jobs:
            pf.status = PENDING_OCR
        elif not pf.blocks and not pf.ocr_jobs:
            pf.status = EMPTY
    if not pf.head_text:
        pf.head_text = "\n".join(b.text for b in pf.blocks[:6])[:600]
    if not pf.title:
        pf.title = _first_line(pf.head_text)
    return pf


class _Encrypted(Exception):
    pass


class _Unknown(Exception):
    pass


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3:
            return line[:160]
    return ""


# ------------------------------------------------------------------------------------------------
# Text, logs, code


def decode_text(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    head = data[:8192]
    if head and head.count(b"\x00") > len(head) // 8:
        # UTF-16 without BOM is mostly every-other-byte NUL; anything else NUL-heavy is binary.
        if head[1::2].count(b"\x00") > len(head) // 3:
            return data.decode("utf-16-le", errors="replace")
        if head[0::2].count(b"\x00") > len(head) // 3:
            return data.decode("utf-16-be", errors="replace")
        raise _Unknown("binary content in a text-typed file")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _line_blocks(text: str, unit: str = "line") -> list[Block]:
    blocks = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            blocks.append(Block(text=line.rstrip(), loc=f"line {i}", unit=unit))
    return blocks


def _parse_text(pf: ParsedFile, data: bytes) -> None:
    text = decode_text(data)
    pf.head_text = text[:600]
    pf.blocks = _line_blocks(text)


def _parse_python(pf: ParsedFile, data: bytes) -> None:
    text = decode_text(data)
    pf.head_text = text[:600]
    pf.blocks = _line_blocks(text, unit="code")
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return
    doc = ast.get_docstring(tree)
    if doc:
        pf.title = _first_line(doc)

    def comments_above(lineno: int) -> list[str]:
        out = []
        i = lineno - 2
        while i >= 0 and lines[i].strip().startswith("#"):
            out.insert(0, lines[i].strip())
            i -= 1
        return out

    def seg(node: ast.AST) -> str:
        try:
            return (ast.get_source_segment(text, node) or "").strip()
        except Exception:
            return ""

    symbols: list[Block] = []

    def visit(body: list[ast.stmt], scope: str) -> None:
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                src = seg(node)
                if src and len(src) < 600:
                    ctx = comments_above(node.lineno)
                    trailing = lines[node.lineno - 1].split("#", 1)[1].strip() if "#" in lines[node.lineno - 1] else ""
                    parts = ctx + [f"{scope}{src}" + (f"  # {trailing}" if trailing and trailing not in src else "")]
                    symbols.append(Block(text="\n".join(parts), loc=f"line {node.lineno}", unit="code", heading=f"constant in {scope or 'module'}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                defaults = list(zip(args.args[len(args.args) - len(args.defaults):], args.defaults))
                defaults += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                if defaults:
                    sig = ", ".join(f"{a.arg}={seg(d)}" for a, d in defaults)
                    symbols.append(Block(text=f"def {scope}{node.name}(... {sig})  # default argument values", loc=f"line {node.lineno}", unit="code", heading="function defaults"))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{scope}{node.name}.")

    visit(tree.body, "")
    pf.blocks.extend(symbols)


# ------------------------------------------------------------------------------------------------
# Tables


def _row_text(header: list[str] | None, cells: list[str]) -> str:
    if header and len(header) >= 2:
        parts = []
        for i, c in enumerate(cells):
            if c == "":
                continue
            key = header[i] if i < len(header) and header[i] else f"column {i + 1}"
            parts.append(f"{key}: {c}")
        return " | ".join(parts)
    return " | ".join(c for c in cells if c != "")


def _looks_like_header(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    return all(not re.fullmatch(r"[-+$€£]?[\d.,%]+", c) for c in filled) and len(set(filled)) == len(filled)


def _parse_csv(pf: ParsedFile, data: bytes) -> None:
    text = decode_text(data)
    pf.head_text = text[:600]
    sample = text[:4096]
    delim = "\t" if pf.ext == ".tsv" else ","
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header: list[str] | None = None
    for i, row in enumerate(reader, 1):
        if i > MAX_ROWS:
            pf.detail = f"truncated at {MAX_ROWS} rows"
            break
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if header is None and _looks_like_header(cells):
            header = cells
            pf.blocks.append(Block(text="columns: " + " | ".join(cells), loc=f"row {i}", unit="heading"))
            continue
        pf.blocks.append(Block(text=_row_text(header, cells), loc=f"row {i}", unit="row"))


def _cell_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return format_number(v)
    if hasattr(v, "isoformat"):
        s = v.isoformat()
        return s[:-9] if s.endswith("T00:00:00") else s
    return str(v).strip()


def _parse_xlsx(pf: ParsedFile, data: bytes) -> None:
    if data.startswith(OLE_MAGIC):
        raise _Encrypted("password-protected workbook (OLE container)")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid xlsx: {exc}") from exc
    if "EncryptionInfo" in zf.namelist():
        raise _Encrypted("encrypted workbook")
    sheets = None
    try:
        sheets = _xlsx_openpyxl(data)
    except ImportError:
        sheets = None
    if sheets is None:
        sheets = _xlsx_stdlib(zf)
    for sheet_name, rows in sheets:
        header: list[str] | None = None
        for r_idx, cells in rows:
            if not any(cells):
                continue
            if header is None and _looks_like_header(cells):
                header = cells
                pf.blocks.append(Block(text=f"sheet {sheet_name} columns: " + " | ".join(c for c in cells if c), loc=f"sheet '{sheet_name}' row {r_idx}", unit="heading", heading=f"sheet {sheet_name}"))
                continue
            pf.blocks.append(Block(text=_row_text(header, cells), loc=f"sheet '{sheet_name}' row {r_idx}", unit="row", heading=f"sheet {sheet_name}"))
    pf.title = pf.title or (f"workbook sheets: {', '.join(s for s, _ in sheets)}" if sheets else "")
    for name in zf.namelist():
        if name.startswith("xl/media/") and Path(name).suffix.lower() in IMAGE_EXTS:
            blob = zf.read(name)
            if len(blob) > 4096:
                pf.ocr_jobs.append(OcrJob(loc=f"embedded image {Path(name).name}", data=blob, suffix=Path(name).suffix.lower()))


def _xlsx_openpyxl(data: bytes):
    import openpyxl  # noqa: F401  (ImportError -> stdlib fallback)

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    formulas = None
    out = []
    try:
        for ws in wb.worksheets:
            rows = []
            for r_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
                if r_idx > MAX_ROWS:
                    break
                cells = [_cell_str(v) for v in row]
                while cells and cells[-1] == "":
                    cells.pop()
                rows.append((r_idx, cells))
            out.append((ws.title, rows))
    finally:
        wb.close()
    # Workbooks written by scripts often carry formulas with no cached value; show the formula.
    if any(any(c == "" for c in cells) for _, rows in out for _, cells in rows) and len(data) < 8 * 1024 * 1024:
        try:
            formulas = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=False)
            for s_idx, ws in enumerate(formulas.worksheets):
                _, rows = out[s_idx]
                for r_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
                    if r_idx > len(rows):
                        break
                    cells = rows[r_idx - 1][1]
                    for c_idx, v in enumerate(row):
                        if isinstance(v, str) and v.startswith("=") and (c_idx >= len(cells) or cells[c_idx] == ""):
                            while len(cells) <= c_idx:
                                cells.append("")
                            cells[c_idx] = v
        except Exception:
            pass
        finally:
            if formulas is not None:
                formulas.close()
    return out


_XNS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _xlsx_stdlib(zf: zipfile.ZipFile):
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", _XNS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_XNS['m']}}}t")))
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target") for r in rels}
    out = []
    for sheet in wb.findall("m:sheets/m:sheet", _XNS):
        target = targets.get(sheet.get(_RNS), "")
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        try:
            root = ET.fromstring(zf.read(path))
        except KeyError:
            continue
        rows = []
        for row in root.iter(f"{{{_XNS['m']}}}row"):
            r_idx = int(row.get("r", len(rows) + 1))
            cells: list[str] = []
            for c in row.findall("m:c", _XNS):
                ref = c.get("r", "")
                col = _col_index(ref) if ref else len(cells)
                while len(cells) < col:
                    cells.append("")
                t = c.get("t")
                v = c.find("m:v", _XNS)
                if t == "s" and v is not None:
                    val = shared[int(v.text)] if v.text and int(v.text) < len(shared) else ""
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{_XNS['m']}}}t"))
                elif v is not None and v.text is not None:
                    val = v.text
                    try:
                        val = format_number(float(val)) if t not in ("str", "e") else val
                    except ValueError:
                        pass
                else:
                    f = c.find("m:f", _XNS)
                    val = f"={f.text}" if f is not None and f.text else ""
                cells.append(val)
            rows.append((r_idx, cells))
        out.append((sheet.get("name", "sheet"), rows))
    return out


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return max(n - 1, 0)


# ------------------------------------------------------------------------------------------------
# DOCX (stdlib zip + XML: paragraphs, headings, tables, headers/footers, notes, text boxes)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WQ = f"{{{_W}}}"


def _para_text(p: ET.Element) -> str:
    out = []
    for el in p.iter():
        tag = el.tag
        if tag == _WQ + "t":
            out.append(el.text or "")
        elif tag == _WQ + "tab":
            out.append("\t")
        elif tag in (_WQ + "br", _WQ + "cr"):
            out.append("\n")
        elif tag == _WQ + "noBreakHyphen":
            out.append("-")
    return "".join(out).strip()


def _para_style(p: ET.Element) -> str:
    ps = p.find(f"{_WQ}pPr/{_WQ}pStyle")
    return (ps.get(_WQ + "val") or "") if ps is not None else ""


def _docx_body(elem: ET.Element, blocks: list[Block], state: dict) -> None:
    for child in list(elem):
        tag = child.tag
        if tag == _WQ + "p":
            text = _para_text(child)
            if not text:
                continue
            style = _para_style(child).lower()
            if style.startswith(("heading", "title")) or style in ("subtitle",):
                state["heading"] = text
                blocks.append(Block(text=text, loc=f"paragraph {state['n']}", unit="heading", heading=text))
            else:
                blocks.append(Block(text=text, loc=f"paragraph {state['n']}", unit="para", heading=state["heading"]))
            state["n"] += 1
        elif tag == _WQ + "tbl":
            state["t"] += 1
            t_no = state["t"]
            rows = []
            for tr in child.findall(_WQ + "tr"):
                cells = []
                for tc in tr.findall(_WQ + "tc"):
                    parts = [_para_text(p) for p in tc.iter(_WQ + "p")]
                    cells.append(" ".join(x for x in parts if x).strip())
                rows.append(cells)
            header = rows[0] if rows and _looks_like_header(rows[0]) else None
            for r_i, cells in enumerate(rows, 1):
                if header is not None and r_i == 1:
                    blocks.append(Block(text="table columns: " + " | ".join(c for c in cells if c), loc=f"table {t_no} row 1", unit="heading", heading=state["heading"]))
                    continue
                txt = _row_text(header, cells)
                if txt:
                    blocks.append(Block(text=txt, loc=f"table {t_no} row {r_i}", unit="row", heading=state["heading"]))
        elif tag in (_WQ + "sdt", _WQ + "sdtContent", _WQ + "customXml", _WQ + "smartTag"):
            _docx_body(child, blocks, state)


def _parse_docx(pf: ParsedFile, data: bytes) -> None:
    if data.startswith(OLE_MAGIC):
        raise _Encrypted("password-protected document (OLE container)")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid docx: {exc}") from exc
    names = zf.namelist()
    if "EncryptionInfo" in names:
        raise _Encrypted("encrypted document")
    if "word/document.xml" not in names:
        raise ValueError("word/document.xml missing")
    root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find(_WQ + "body")
    blocks: list[Block] = []
    state = {"heading": "", "n": 1, "t": 0}
    if body is not None:
        _docx_body(body, blocks, state)
    extra_parts = sorted(n for n in names if re.match(r"word/(header|footer|footnotes|endnotes|comments)\d*\.xml$", n))
    for part in extra_parts:
        try:
            proot = ET.fromstring(zf.read(part))
        except ET.ParseError:
            continue
        label = Path(part).stem
        for p in proot.iter(_WQ + "p"):
            text = _para_text(p)
            if text:
                blocks.append(Block(text=text, loc=label, unit="para", heading=label))
    pf.blocks = blocks
    if "docProps/core.xml" in names:
        try:
            core = ET.fromstring(zf.read("docProps/core.xml"))
            for el in core.iter():
                if el.tag.endswith("}title") and el.text and el.text.strip():
                    pf.title = el.text.strip()
        except ET.ParseError:
            pass
    for name in names:
        if name.startswith("word/media/") and Path(name).suffix.lower() in IMAGE_EXTS:
            blob = zf.read(name)
            if len(blob) > 4096:
                pf.ocr_jobs.append(OcrJob(loc=f"embedded image {Path(name).name}", data=blob, suffix=Path(name).suffix.lower()))


# ------------------------------------------------------------------------------------------------
# PDF


def pdf_is_encrypted_raw(data: bytes) -> bool:
    """A PDF is encrypted iff its trailer references an /Encrypt dictionary."""
    return re.search(rb"/Encrypt\s*(?:\d+\s+\d+\s+R|<<)", data) is not None


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paras:
        if len(p) <= 1200:
            out.append(p)
            continue
        lines = p.splitlines()
        buf: list[str] = []
        for line in lines:
            buf.append(line)
            if sum(len(x) for x in buf) > 700:
                out.append("\n".join(buf))
                buf = []
        if buf:
            out.append("\n".join(buf))
    return out


def _parse_pdf(pf: ParsedFile, data: bytes) -> None:
    if not data.startswith(b"%PDF") and b"%PDF" not in data[:1024]:
        raise ValueError("not a PDF")
    if pdf_is_encrypted_raw(data):
        raise _Encrypted("PDF has an /Encrypt dictionary")
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            pymupdf = None
    if pymupdf is not None:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            if doc.needs_pass or doc.is_encrypted:
                raise _Encrypted("PDF requires a password")
            meta_title = (doc.metadata or {}).get("title") or ""
            pf.title = meta_title.strip()
            for i, page in enumerate(doc, 1):
                if i > MAX_PDF_PAGES:
                    pf.detail = f"truncated at {MAX_PDF_PAGES} pages"
                    break
                text = page.get_text("text", sort=True) or ""
                if len(text.strip()) < 25 and page.get_images(full=False):
                    pix = page.get_pixmap(dpi=150)
                    pf.ocr_jobs.append(OcrJob(loc=f"page {i} (scanned)", data=pix.tobytes("png")))
                for para in _split_paragraphs(text):
                    pf.blocks.append(Block(text=para, loc=f"page {i}", unit="para"))
        finally:
            doc.close()
        return
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise ValueError("no PDF library installed (pymupdf or pypdf)") from exc
    reader = pypdf.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise _Encrypted("PDF requires a password")
    for i, page in enumerate(reader.pages, 1):
        if i > MAX_PDF_PAGES:
            break
        for para in _split_paragraphs(page.extract_text() or ""):
            pf.blocks.append(Block(text=para, loc=f"page {i}", unit="para"))


# ------------------------------------------------------------------------------------------------
# Images


_IMAGE_MAGIC = {
    b"\x89PNG": ".png", b"\xff\xd8\xff": ".jpg", b"GIF8": ".gif", b"BM": ".bmp",
    b"II*\x00": ".tif", b"MM\x00*": ".tif", b"RIFF": ".webp",
}


def _parse_image(pf: ParsedFile, data: bytes) -> None:
    kind = next((ext for magic, ext in _IMAGE_MAGIC.items() if data.startswith(magic)), None)
    if kind is None:
        raise _Unknown("not a recognised image")
    pf.ocr_jobs.append(OcrJob(loc="image", data=data, suffix=kind))


_HANDLERS: dict[str, Callable[[ParsedFile, bytes], None]] = {
    ".py": _parse_python, ".pyi": _parse_python,
    ".csv": _parse_csv, ".tsv": _parse_csv,
    ".xlsx": _parse_xlsx, ".xlsm": _parse_xlsx,
    ".docx": _parse_docx,
    ".pdf": _parse_pdf,
}
for _ext in IMAGE_EXTS:
    _HANDLERS[_ext] = _parse_image
