#!/usr/bin/env python3
"""Check that relative Markdown links across the repo resolve to real files.

Walks every ``*.md`` file under the repository root, extracts inline
Markdown links — ``[text](target)`` and ``![alt](target)`` — and verifies
that each *relative* target exists on disk.

Rules
-----
* External links are skipped: any target with a URL scheme
  (``http://``, ``https://``, ``mailto:``, ``ftp:``, ...).
* Pure in-page anchors (``#section``) are skipped.
* For ``path#anchor`` targets only the ``path`` part is resolved; the
  anchor is not validated.
* Targets are resolved relative to the Markdown file's own directory;
  targets starting with ``/`` are resolved from the repository root.

Exit status is 0 when every relative link resolves, 1 otherwise (broken
links are listed on stdout as ``file:line: target -> resolved-path``).

Usage::

    python scripts/check_doc_links.py [ROOT]

ROOT defaults to the repository root (the parent of ``scripts/``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Inline Markdown link/image: [text](target) / ![alt](target).
# The target group stops at the first unescaped ')' or whitespace-"title".
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]*>|[^()\s]+)")

# Directories never scanned for Markdown files.
_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    # Vendored upstream source (git submodules / fetch_dependencies.sh
    # output): upstream docs are not ours to lint.
    "third_party",
    "oran-deps-src",
}


def _iter_markdown_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.md")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _extract_targets(text: str) -> list[tuple[int, str]]:
    """Return (line_number, raw_target) pairs for inline links in *text*."""
    targets: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            if target:
                targets.append((lineno, target))
    return targets


def find_broken_links(root: Path) -> list[tuple[Path, int, str, Path]]:
    """Return (md_file, line, raw_target, resolved_path) for dead links."""
    broken: list[tuple[Path, int, str, Path]] = []
    for md_file in _iter_markdown_files(root):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        for lineno, raw in _extract_targets(text):
            parts = urlsplit(raw)
            if parts.scheme:  # http, https, mailto, ftp, ...
                continue
            path_part = unquote(parts.path)
            if not path_part:  # pure anchor like "#section"
                continue
            if path_part.startswith("/"):
                resolved = (root / path_part.lstrip("/")).resolve()
            else:
                resolved = (md_file.parent / path_part).resolve()
            if not resolved.exists():
                broken.append((md_file, lineno, raw, resolved))
    return broken


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        root = Path(argv[1]).resolve()
    else:
        root = Path(__file__).resolve().parent.parent
    broken = find_broken_links(root)
    if broken:
        print(f"Broken relative Markdown links under {root}:")
        for md_file, lineno, raw, resolved in broken:
            rel = md_file.relative_to(root)
            print(f"  {rel}:{lineno}: ({raw}) -> {resolved}")
        print(f"{len(broken)} broken link(s).")
        return 1
    print("All relative Markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
