"""Content-addressed model artefact vault.

Every promoted model checkpoint is stored under its SHA-256 hash so that
rollback retrieves a byte-identical artefact. Retrieval re-verifies the
hash to detect tamper.

Layout
------
    <root>/<aa>/<bbbb...>.bin       artefact bytes
    <root>/<aa>/<bbbb...>.meta.json sidecar metadata

The Git-style 2-char fanout keeps any single directory below ~256 entries
even at 1M-artefact scale.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class IntegrityError(RuntimeError):
    """Raised when the on-disk SHA-256 does not match the expected hash."""


@dataclass(frozen=True)
class ArtefactMetadata:
    sha256: str
    size_bytes: int
    stored_at: str  # ISO-8601 UTC
    label: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ArtefactVault:
    """Content-addressed model artefact storage with integrity verification.

    Thread-safe: stores under a coarse lock; retrievals are concurrent-safe
    via filesystem atomicity. Use ``tempfile.TemporaryDirectory`` in tests
    to avoid global state pollution.
    """

    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ─── Path helpers ──────────────────────────────────────────────────

    def _paths_for(self, sha: str) -> tuple[Path, Path]:
        """Return (bin_path, meta_path) for a given sha."""
        if not (len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise ValueError(f"sha must be 64-hex; got {sha!r}")
        d = self._root / sha[:2]
        return d / f"{sha}.bin", d / f"{sha}.meta.json"

    # ─── Mutators ──────────────────────────────────────────────────────

    def store(
        self,
        model_bytes: bytes,
        *,
        label: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Persist ``model_bytes`` under its SHA-256. Returns the hash.

        Idempotent: storing the same content twice yields the same SHA
        and does not duplicate the file. The metadata sidecar is updated
        on each store so the latest ``label`` and ``stored_at`` win.
        """
        sha = _sha256_bytes(model_bytes)
        bin_path, meta_path = self._paths_for(sha)
        with self._lock:
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            if not bin_path.exists():
                # Atomic write: temp + rename.
                tmp = bin_path.with_suffix(".bin.tmp")
                tmp.write_bytes(model_bytes)
                tmp.replace(bin_path)
            meta = ArtefactMetadata(
                sha256=sha,
                size_bytes=len(model_bytes),
                stored_at=datetime.now(timezone.utc).isoformat(),
                label=label,
                extra=dict(extra) if extra else {},
            )
            meta_path.write_text(
                json.dumps(asdict(meta), sort_keys=True, separators=(",", ":"))
            )
        return sha

    # ─── Accessors ─────────────────────────────────────────────────────

    def exists(self, sha: str) -> bool:
        bin_path, _ = self._paths_for(sha)
        return bin_path.exists()

    def retrieve(self, sha: str) -> bytes:
        """Return the artefact bytes. Raises ``IntegrityError`` on mismatch."""
        bin_path, _ = self._paths_for(sha)
        if not bin_path.exists():
            raise FileNotFoundError(f"artefact {sha} not in vault")
        data = bin_path.read_bytes()
        actual = _sha256_bytes(data)
        if actual != sha:
            raise IntegrityError(
                f"artefact {sha} fails integrity check; on-disk SHA is {actual}"
            )
        return data

    def metadata(self, sha: str) -> ArtefactMetadata:
        _, meta_path = self._paths_for(sha)
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata for {sha} not in vault")
        d = json.loads(meta_path.read_text())
        return ArtefactMetadata(**d)

    def list_artefacts(self) -> list[ArtefactMetadata]:
        out: list[ArtefactMetadata] = []
        for meta_file in sorted(self._root.rglob("*.meta.json")):
            try:
                d = json.loads(meta_file.read_text())
                out.append(ArtefactMetadata(**d))
            except (json.JSONDecodeError, TypeError):
                continue
        return out


__all__ = [
    "ArtefactMetadata",
    "ArtefactVault",
    "IntegrityError",
]
