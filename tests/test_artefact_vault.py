"""Content-addressed artefact vault tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from horizon_ric.runtime.artefact_vault import (
    ArtefactVault,
    IntegrityError,
)


class TestStoreRetrieve:
    def test_round_trip_byte_identical(self, tmp_path):
        v = ArtefactVault(tmp_path)
        data = b"\x00\x01\x02\x03" * 100
        sha = v.store(data, label="model-A")
        assert v.exists(sha)
        out = v.retrieve(sha)
        assert out == data

    def test_sha_is_real_sha256(self, tmp_path):
        v = ArtefactVault(tmp_path)
        data = b"hello, world"
        sha = v.store(data)
        assert sha == hashlib.sha256(data).hexdigest()

    def test_idempotent_store(self, tmp_path):
        v = ArtefactVault(tmp_path)
        data = b"same content"
        sha1 = v.store(data, label="A")
        sha2 = v.store(data, label="B")
        assert sha1 == sha2
        # Only one .bin file should exist.
        bins = list(Path(tmp_path).rglob("*.bin"))
        assert len(bins) == 1


class TestIntegrity:
    def test_tampered_artefact_raises(self, tmp_path):
        v = ArtefactVault(tmp_path)
        sha = v.store(b"original")
        # Tamper: overwrite the .bin file.
        bin_path = list(Path(tmp_path).rglob("*.bin"))[0]
        bin_path.write_bytes(b"tampered")
        with pytest.raises(IntegrityError):
            v.retrieve(sha)

    def test_missing_artefact_raises_filenotfound(self, tmp_path):
        v = ArtefactVault(tmp_path)
        with pytest.raises(FileNotFoundError):
            v.retrieve("0" * 64)


class TestMetadata:
    def test_metadata_roundtrip(self, tmp_path):
        v = ArtefactVault(tmp_path)
        sha = v.store(b"x", label="hybrid_deep_rx_v0.1", extra={"epoch": 30})
        m = v.metadata(sha)
        assert m.label == "hybrid_deep_rx_v0.1"
        assert m.extra["epoch"] == 30
        assert m.size_bytes == 1


class TestStress:
    def test_100_artefacts_roundtrip(self, tmp_path):
        v = ArtefactVault(tmp_path)
        shas: list[str] = []
        for i in range(100):
            shas.append(v.store(f"artefact-{i}".encode()))
        assert len(set(shas)) == 100  # all unique
        for i, sha in enumerate(shas):
            assert v.retrieve(sha) == f"artefact-{i}".encode()

    def test_list_artefacts_consistent(self, tmp_path):
        v = ArtefactVault(tmp_path)
        for i in range(20):
            v.store(f"x-{i}".encode())
        listed = v.list_artefacts()
        assert len(listed) == 20
        # Sorted by SHA prefix (filesystem walk order).
        shas = [m.sha256 for m in listed]
        assert len(set(shas)) == 20


class TestInvalidInputs:
    def test_invalid_sha_format_rejected(self, tmp_path):
        v = ArtefactVault(tmp_path)
        with pytest.raises(ValueError):
            v.exists("not-a-sha")
        with pytest.raises(ValueError):
            v.retrieve("XYZ")
