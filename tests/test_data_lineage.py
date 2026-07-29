"""Training-data lineage manifest tests (TS 28.105 §7.4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from horizon_ric.data.lineage import (
    DataManifest,
    build_manifest,
    compute_dataset_sha256,
)


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _make_dataset(tmp: Path, files: dict[str, bytes]) -> Path:
    root = tmp / "ds"
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return root


class TestSha256:
    def test_deterministic_same_content(self, tmp_path):
        a = _make_dataset(tmp_path / "a", {"x.bin": b"hello", "y.bin": b"world"})
        b = _make_dataset(tmp_path / "b", {"x.bin": b"hello", "y.bin": b"world"})
        assert compute_dataset_sha256(a) == compute_dataset_sha256(b)

    def test_different_content_different_sha(self, tmp_path):
        a = _make_dataset(tmp_path / "a", {"x.bin": b"hello"})
        b = _make_dataset(tmp_path / "b", {"x.bin": b"different"})
        assert compute_dataset_sha256(a) != compute_dataset_sha256(b)

    def test_same_size_substitution_changes_sha(self, tmp_path):
        a = _make_dataset(tmp_path / "a", {"x.bin": b"abcde"})
        b = _make_dataset(tmp_path / "b", {"x.bin": b"edcba"})
        assert compute_dataset_sha256(a) != compute_dataset_sha256(b)


class TestManifestRoundTrip:
    def test_pydantic_json_roundtrip(self, tmp_path):
        root = _make_dataset(tmp_path, {"x.bin": b"" * 128})
        m = build_manifest(
            dataset_root=root,
            sample_count=100,
            time_range_start=_ts("2026-01-01T00:00:00"),
            time_range_end=_ts("2026-02-01T00:00:00"),
            license="CC-BY-4.0",
            consent_basis="legitimate_interests",
            geography_filter=["US", "DE"],
            provider_breakdown={"ucc_misl": 60, "deepmimo": 40},
        )
        blob = m.model_dump_json()
        m2 = DataManifest.model_validate_json(blob)
        assert m2 == m
        assert json.loads(blob)["dataset_sha256"] == m.dataset_sha256


class TestValidation:
    def test_gdpr_consent_basis_validated(self, tmp_path):
        root = _make_dataset(tmp_path, {"x.bin": b"x"})
        with pytest.raises(ValueError, match="GDPR Art. 6"):
            build_manifest(
                dataset_root=root,
                sample_count=1,
                time_range_start=_ts("2026-01-01T00:00:00"),
                time_range_end=_ts("2026-02-01T00:00:00"),
                license="CC-BY-4.0",
                consent_basis="not-a-real-ground",
            )

    def test_iso3166_geography_validated(self, tmp_path):
        root = _make_dataset(tmp_path, {"x.bin": b"x"})
        with pytest.raises(ValidationError):
            DataManifest(
                dataset_sha256="0" * 64,
                sample_count=1,
                time_range_start=_ts("2026-01-01T00:00:00"),
                time_range_end=_ts("2026-02-01T00:00:00"),
                geography_filter=["USA"],  # 3-letter, invalid
                license="x",
                consent_basis="consent",
            )

    def test_time_range_end_before_start_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            DataManifest(
                dataset_sha256="0" * 64,
                sample_count=1,
                time_range_start=_ts("2026-02-01T00:00:00"),
                time_range_end=_ts("2026-01-01T00:00:00"),
                license="x",
                consent_basis="consent",
            )

    def test_provider_breakdown_must_sum_to_count(self, tmp_path):
        with pytest.raises(ValidationError):
            DataManifest(
                dataset_sha256="0" * 64,
                sample_count=100,
                time_range_start=_ts("2026-01-01T00:00:00"),
                time_range_end=_ts("2026-02-01T00:00:00"),
                license="x",
                consent_basis="consent",
                provider_breakdown={"a": 60, "b": 30},  # sums to 90, not 100
            )

    def test_sha256_pattern_enforced(self, tmp_path):
        with pytest.raises(ValidationError):
            DataManifest(
                dataset_sha256="not-a-hex",
                sample_count=1,
                time_range_start=_ts("2026-01-01T00:00:00"),
                time_range_end=_ts("2026-02-01T00:00:00"),
                license="x",
                consent_basis="consent",
            )
