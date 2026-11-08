"""Revision awareness: which files are WITHDRAWN / superseded, and by what.

Two signals:
  * explicit — the filename or the document head says WITHDRAWN / SUPERSEDED BY / OBSOLETE.
  * family   — several files share a stem apart from a revision marker (r1/r2, rev A/rev B, v1/v2);
               all but the newest are older revisions.
An explicitly withdrawn file never answers a question unless the question asks about it.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_STATUS_WORDS = r"withdrawn|superseded|obsolete|deprecated|retired|old|outdated|archived|draft|final|current|latest|new"
_NAME_FLAG = re.compile(r"(?:^|[^a-z])(withdrawn|superseded|obsolete|deprecated|retired|outdated)(?:[^a-z]|$)", re.IGNORECASE)
_HEAD_FLAGS = [
    re.compile(r"\b(WITHDRAWN|SUPERSEDED|OBSOLETE|DEPRECATED)\b"),  # shouted status in the header
    re.compile(r"\bsuperseded\s+by\b", re.IGNORECASE),
    re.compile(r"\b(?:this|the present)\s+(?:document|revision|version|datasheet|edition|spec(?:ification)?|release)\s+(?:is|has been|was)\s+(?:withdrawn|superseded|obsolete|deprecated|retired)\b", re.IGNORECASE),
]
_REV_MARK = re.compile(
    r"(?:^|[_\-\s.])(?:r|rev|revision|v|ver|version|issue|ed|edition)[_\-\s.]?(\d+(?:\.\d+)*[a-z]?|[a-z]\d*)(?=$|[_\-\s.])",
    re.IGNORECASE,
)
_QUESTION_TARGETS_OLD = re.compile(
    r"\b(withdrawn|superseded|obsolete|deprecated|retired|original|preliminary|previous|prior|older|old|earlier|initial|first)\b"
    r"|\b(?:r|rev|revision|version|v)\s?[-.]?\s?(?:\d+|[a-z])\b",
    re.IGNORECASE,
)


def explicit_withdrawal(rel_path: str, head_text: str) -> str:
    """Reason string if the file declares itself withdrawn/superseded, else ''."""
    name = PurePosixPath(rel_path).name
    m = _NAME_FLAG.search(name)
    if m:
        return f"filename marked {m.group(1).upper()}"
    head = head_text[:400]
    for rx in _HEAD_FLAGS:
        m = rx.search(head)
        if m:
            # "supersedes revision 1" / "revision 1 is withdrawn" describe ANOTHER document.
            window = head[max(0, m.start() - 40): m.end() + 5].lower()
            if re.search(r"\b(?:revision|rev|version|r)\s*\w{1,3}\s+(?:is|was|has been)\s*$", head[max(0, m.start() - 30): m.start()].lower()):
                continue
            if "supersedes" in window and "superseded" not in window:
                continue
            return f"document header: '{m.group(0)}'"
    return ""


def _rev_value(token: str) -> tuple:
    token = token.lower()
    if re.fullmatch(r"\d+(?:\.\d+)*[a-z]?", token):
        nums = re.findall(r"\d+", token)
        suffix = token[-1] if token[-1].isalpha() else ""
        return (1, tuple(int(n) for n in nums), suffix)
    letter = token[0]
    digits = token[1:]
    return (0, (ord(letter) - 96,), int(digits) if digits.isdigit() else 0)


def family_key(rel_path: str) -> tuple[str, tuple] | None:
    p = PurePosixPath(rel_path)
    stem = p.stem
    marks = list(_REV_MARK.finditer(stem))
    if not marks:
        return None
    m = marks[-1]
    base = (stem[: m.start()] + stem[m.end():]).lower()
    base = re.sub(rf"(?:^|[_\-\s.])(?:{_STATUS_WORDS})(?=$|[_\-\s.])", "", base)
    base = re.sub(r"[_\-\s.]+", "_", base).strip("_")
    if not base:
        return None
    return (f"{p.parent.as_posix()}/{base}{p.suffix.lower()}", _rev_value(m.group(1)))


def revision_status(files: list[dict]) -> dict[str, dict]:
    """Map rel path -> {"status": current|withdrawn|older-revision, "reason", "superseded_by"}.
    `files` items need "path" and "head_text"."""
    out: dict[str, dict] = {}
    for f in files:
        reason = explicit_withdrawal(f["path"], f.get("head_text", ""))
        out[f["path"]] = {"status": "withdrawn" if reason else "current", "reason": reason, "superseded_by": ""}
    families: dict[str, list[tuple[tuple, str]]] = {}
    for f in files:
        fk = family_key(f["path"])
        if fk:
            families.setdefault(fk[0], []).append((fk[1], f["path"]))
    for members in families.values():
        if len(members) < 2:
            continue
        live = [m for m in members if out[m[1]]["status"] != "withdrawn"] or members
        newest = max(live, key=lambda m: m[0])
        for rev, path in members:
            if path == newest[1]:
                continue
            out[path]["superseded_by"] = newest[1]
            if out[path]["status"] == "current" and rev < newest[0]:
                out[path]["status"] = "older-revision"
                out[path]["reason"] = f"newer revision exists: {newest[1]}"
    return out


def question_targets_old(question: str) -> bool:
    return bool(_QUESTION_TARGETS_OLD.search(question))
