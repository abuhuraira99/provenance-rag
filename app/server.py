#!/usr/bin/env python3
"""Container CMD: supervisor for the resident Provenance RAG model server. Never exits."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# In the image the package sits next to this file (/app/provenance_rag); in the repo, one level up.
for _p in (os.path.dirname(_HERE), _HERE):
    if os.path.isdir(os.path.join(_p, "provenance_rag")) and _p not in sys.path:
        sys.path.insert(0, _p)

from provenance_rag.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
