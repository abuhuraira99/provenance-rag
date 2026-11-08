"""Provenance RAG: question answering over messy document folders with exact file citations.

Public entry points:

    from provenance_rag import Engine
    engine = Engine.create(backend="stub")
    engine.build_index("path/to/corpus")
    result = engine.ask("What is the maximum junction temperature of the TQ-40?")
    result.answer, result.citations

The package is importable without any third-party dependency; parsers for PDF and
XLSX use pymupdf / openpyxl when installed and degrade to stdlib fallbacks otherwise.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import Engine, QueryResult  # noqa: E402

__all__ = ["Engine", "QueryResult", "__version__"]
