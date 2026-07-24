"""Training-data lineage manifest — TS 28.105 §7.4 ``lastTrainingDataset``.

Every model checkpoint carries a ``DataManifest`` that names the exact
training data it was produced from:

  * ``dataset_sha256``     content-hash of the training data tarball/dir
  * ``sample_count``       number of training samples
  * ``time_range_*``       start / end UTC timestamps of the data
  * ``geography_filter``   ISO-3166-1 alpha-2 country codes
  * ``provider_breakdown`` per-source sample counts (e.g. UCC-MISL, DeepMIMO, Aerial)
  * ``license``            redistribution license
  * ``consent_basis``      GDPR Art. 6 lawful-processing ground

Reference
---------
3GPP TS 28.105 §7.4 — ``lastTrainingDataset`` mandatory field.
GDPR Art. 6 — lawful processing grounds.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field, field_validator

# GDPR Art. 6(1) — closed enumeration. We use the short-form codes the
# regulator's templates use.
_GDPR_LAWFUL_GROUNDS: frozenset[str] = frozenset(
    {
        "consent",       # Art. 6(1)(a)
        "contract",      # Art. 6(1)(b)
        "legal_obligation",  # Art. 6(1)(c)
        "vital_interests",   # Art. 6(1)(d)
        "public_task",       # Art. 6(1)(e)
        "legitimate_interests",  # Art. 6(1)(f)
        "n/a",  # synthetic / non-personal data — declare explicitly
    }
)


class DataManifest(BaseModel):
    """Training-data manifest. Pydantic v2.

    Round-trips through JSON without loss; the ``dataset_sha256`` is a
    deterministic content-hash so a second build of the same dataset
    yields the same manifest.
    """

    dataset_sha256: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the canonical manifest of training files.",
    )
    sample_count: int = Field(..., ge=0)
    time_range_start: datetime
    time_range_end: datetime
    geography_filter: list[str] = Field(default_factory=list)
    provider_breakdown: dict[str, int] = Field(default_factory=dict)
    license: str = Field(..., min_length=1)
    consent_basis: Literal[
        "consent",
        "contract",
        "legal_obligation",
        "vital_interests",
        "public_task",
        "legitimate_interests",
        "n/a",
    ]

    @field_validator("geography_filter")
    @classmethod
    def _validate_iso3166(cls, v: list[str]) -> list[str]:
        # Permissive: only check shape (2 uppercase alpha) so we don't
        # ship a 250-row table here. ISO-3166-1 alpha-2 is 2 chars.
        for code in v:
            if not (len(code) == 2 and code.isalpha() and code.isupper()):
                raise ValueError(
                    f"geography_filter entry {code!r} is not ISO-3166-1 "
                    f"alpha-2 (2 uppercase letters)"
                )
        return v

    @field_validator("time_range_end")
    @classmethod
    def _end_after_start(cls, v: datetime, info) -> datetime:
        start = info.data.get("time_range_start")
        if start is not None and v < start:
            raise ValueError("time_range_end must be ≥ time_range_start")
        return v

    @field_validator("provider_breakdown")
    @classmethod
    def _breakdown_sums_to_count(cls, v: dict[str, int], info) -> dict[str, int]:
        total = info.data.get("sample_count")
        if total is not None and v and sum(v.values()) != total:
            raise ValueError(
                f"provider_breakdown sums to {sum(v.values())} but "
                f"sample_count is {total}"
            )
        return v


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 for ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest_rows(root: Path) -> list[tuple[str, int, str]]:
    """Walk a dataset and return sorted ``(path, size, content_sha256)`` rows.

    Including the content hash is essential: a path-and-size-only manifest
    cannot detect an in-place substitution with the same byte length.
    """
    rows: list[tuple[str, int, str]] = []
    if not root.exists():
        return rows
    if root.is_file():
        return [(root.name, root.stat().st_size, _sha256_file(root))]
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append((str(p.relative_to(root)), p.stat().st_size, _sha256_file(p)))
    return rows


def compute_dataset_sha256(root: Path) -> str:
    """Deterministic SHA-256 of paths, sizes, and every file's bytes."""
    rows = _file_manifest_rows(root)
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_manifest(
    *,
    dataset_root: Path,
    sample_count: int,
    time_range_start: datetime,
    time_range_end: datetime,
    license: str,
    consent_basis: str,
    geography_filter: Iterable[str] = (),
    provider_breakdown: dict[str, int] | None = None,
) -> DataManifest:
    """Build a ``DataManifest`` from a dataset root.

    The ``dataset_sha256`` is computed by walking the root.
    """
    if consent_basis not in _GDPR_LAWFUL_GROUNDS:
        raise ValueError(
            f"consent_basis {consent_basis!r} not in GDPR Art. 6 lawful "
            f"grounds: {sorted(_GDPR_LAWFUL_GROUNDS)}"
        )

    sha = compute_dataset_sha256(Path(dataset_root))
    return DataManifest(
        dataset_sha256=sha,
        sample_count=sample_count,
        time_range_start=time_range_start.astimezone(timezone.utc) if time_range_start.tzinfo else time_range_start.replace(tzinfo=timezone.utc),
        time_range_end=time_range_end.astimezone(timezone.utc) if time_range_end.tzinfo else time_range_end.replace(tzinfo=timezone.utc),
        geography_filter=list(geography_filter),
        provider_breakdown=dict(provider_breakdown) if provider_breakdown else {},
        license=license,
        consent_basis=consent_basis,  # type: ignore[arg-type]
    )


__all__ = [
    "DataManifest",
    "build_manifest",
    "compute_dataset_sha256",
]
