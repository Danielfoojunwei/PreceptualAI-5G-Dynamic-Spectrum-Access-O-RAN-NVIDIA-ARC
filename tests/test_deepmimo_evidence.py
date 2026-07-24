"""Static checks for the externally sourced DeepMIMO benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_deepmimo_manifest_has_byte_level_provenance():
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/manifest.json").read_text()
    )
    assert manifest["deepmimo_version"] == "4.0.0"
    assert len(manifest["source_archive_sha256"]) == 64
    assert len(manifest["source_tree_sha256"]) == 64
    assert len(manifest["features_sha256"]) == 64
    assert manifest["sampled_receivers"] >= 4096
    assert manifest["features_committed"] is False
    assert "not over-the-air" in manifest["data_kind"]


def test_deepmimo_benchmark_has_no_illegal_emits():
    result = json.loads(
        (ROOT / "benchmarks/results/deepmimo_dsa.json").read_text()
    )
    assert result["receivers"] >= 4096
    assert result["best_vs_random_gain_db"] >= 0.0
    for strategy in result["strategies"].values():
        assert strategy["illegal_emits_after_shield"] == 0
        assert strategy["shield_blocked"] == 0


def test_row_level_deepmimo_data_is_gitignored():
    gitignore = (ROOT / ".gitignore").read_text()
    assert "datasets/deepmimo_asu_3p5/generated/" in gitignore
