# Contributing

Thanks for helping. Provenance RAG is small on purpose; please keep it that way.

## Setup

```bash
uv run --python 3.14 --with pytest --with pymupdf==1.28.2 --with openpyxl python -m pytest -q
```

The core package is standard-library only. Document parsers (`pymupdf`, `openpyxl`) are optional at
import time; the model backend (`transformers`) is only needed for the GPU path.

## Ground rules

- **Tests stay CPU-only.** Use the `stub` backend (or `rules`) and the synthetic corpus in
  `tests/conftest.py`. Never add organiser corpus files or sample answers to the repository.
- **No answer hardcoding.** Every behaviour must be general: a rule that only fires for one sample
  question is a bug.
- **Citations are a set, chosen by necessity.** Any change to `answerer.choose_citations` needs a
  test for over-citation, under-citation and the multi-hop chain.
- **Never let one file stop the walk.** New parsers must raise, not swallow, and `parse_file` turns
  every exception into a status row.
- **Keep the client thin.** `app/app.py` stays standard-library only and must never load a model.
- New runtime dependencies must have a cp314 manylinux x86_64, abi3 or pure wheel, be pinned in
  `requirements-image.txt` with their full closure, and must not pull in torch, torchvision, numpy
  or pillow.

## Pull requests

Describe the behaviour change, add or update tests, and include `provenance-rag bench` output when
the change touches retrieval, prompting or citation selection.
