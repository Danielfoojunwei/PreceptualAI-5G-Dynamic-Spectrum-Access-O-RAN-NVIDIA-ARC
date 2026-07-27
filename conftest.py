"""Repo-root pytest bootstrap.

Ensures the repository root is importable so tests can reach top-level,
non-installed helper packages such as ``benchmarks`` (the evaluation harnesses)
regardless of how pytest is invoked (``pytest`` vs ``python -m pytest``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
