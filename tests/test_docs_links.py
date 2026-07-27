"""Every relative Markdown link in the repo must resolve to a real file.

Runs ``scripts/check_doc_links.py`` (the same tool usable standalone /
in CI) against the repository root and fails with the script's own
report if any ``[text](relative/path)`` link points at a file that does
not exist. External (http/https/mailto) links and in-page anchors are
out of scope; ``path#anchor`` targets are checked on the path part only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = REPO_ROOT / "scripts" / "check_doc_links.py"


def test_no_broken_relative_markdown_links() -> None:
    proc = subprocess.run(
        [sys.executable, str(CHECKER), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "scripts/check_doc_links.py found broken Markdown links:\n"
        + proc.stdout
        + proc.stderr
    )
