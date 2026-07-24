from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.verify_deepmimo_reproduction import verify_documents

HASH_A = "a" * 64
HASH_B = "b" * 64
SOURCE_HASH = "c" * 64


def _documents() -> tuple[dict, dict]:
    manifest = {
        "source_archive_sha256": SOURCE_HASH,
        "features_sha256": HASH_A,
        "sampled_receivers": 4096,
        "subband_gain_dbw_max": -97.472687,
        "transform": {"subbands": 6},
    }
    result = {
        "features_sha256": HASH_A,
        "illegal_emits_after_shield": 0,
        "mean_regret_db": 1.730053,
    }
    return manifest, result


def test_verifier_allows_bound_feature_hash_and_tiny_float_drift():
    expected_manifest, expected_result = _documents()
    actual_manifest = deepcopy(expected_manifest)
    actual_result = deepcopy(expected_result)
    actual_manifest["features_sha256"] = HASH_B
    actual_result["features_sha256"] = HASH_B
    actual_manifest["subband_gain_dbw_max"] += 8e-6
    actual_result["mean_regret_db"] -= 5e-6

    verify_documents(
        expected_manifest=expected_manifest,
        actual_manifest=actual_manifest,
        expected_result=expected_result,
        actual_result=actual_result,
    )


def test_verifier_rejects_source_hash_change():
    expected_manifest, expected_result = _documents()
    actual_manifest = deepcopy(expected_manifest)
    actual_manifest["source_archive_sha256"] = "d" * 64

    with pytest.raises(AssertionError, match="source_archive_sha256"):
        verify_documents(
            expected_manifest=expected_manifest,
            actual_manifest=actual_manifest,
            expected_result=expected_result,
            actual_result=deepcopy(expected_result),
        )


def test_verifier_rejects_out_of_tolerance_metric():
    expected_manifest, expected_result = _documents()
    actual_result = deepcopy(expected_result)
    actual_result["mean_regret_db"] += 2e-4

    with pytest.raises(AssertionError, match="mean_regret_db"):
        verify_documents(
            expected_manifest=expected_manifest,
            actual_manifest=deepcopy(expected_manifest),
            expected_result=expected_result,
            actual_result=actual_result,
        )


def test_verifier_rejects_unbound_actual_result():
    expected_manifest, expected_result = _documents()
    actual_manifest = deepcopy(expected_manifest)
    actual_manifest["features_sha256"] = HASH_B

    with pytest.raises(AssertionError, match="not bound"):
        verify_documents(
            expected_manifest=expected_manifest,
            actual_manifest=actual_manifest,
            expected_result=expected_result,
            actual_result=deepcopy(expected_result),
        )
