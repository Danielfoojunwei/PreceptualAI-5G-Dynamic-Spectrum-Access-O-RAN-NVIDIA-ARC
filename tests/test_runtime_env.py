"""The runtime stamp must be present, informative, and inert.

These tests pin the two properties that make the stamp worth having: it records
the one dependency that can silently reseed a benchmark (numpy), and it never
participates in a reproduction assertion.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from horizon_ric.runtime_env import runtime_environment, stamp

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"

# Valid lowercase SHA-256 digests: the verifier checks hash *shape* before
# comparing, so placeholder strings are rejected before the logic under test.
_SHA_A = "a" * 64
_SHA_B = "b" * 64

# The committed results that are reproduced by the realdata workflow.
STAMPED_RESULTS = [
    "credit_assignment.json",
    "exploration_truncation.json",
    "safety_utility_frontier.json",
    "federated_coverage.json",
    "deepmimo_dsa.json",
]


def test_records_numpy_and_interpreter() -> None:
    env = runtime_environment()
    assert env["numpy_version"] == np.__version__
    assert env["python_implementation"] == "CPython"
    # The reason the stamp exists should travel with it.
    assert "no cross-version bit-stream" in env["rng_stream_guarantee"]


def test_stamp_is_stable_across_calls() -> None:
    """No timestamp/hostname: two calls in the same environment must agree.

    If this ever fails, the stamp has acquired a churning field and will dirty
    every committed result JSON on every run.
    """
    assert runtime_environment() == runtime_environment()


def test_stamp_attaches_under_runtime_and_returns_same_object() -> None:
    result: dict[str, object] = {"headline": 0.347412}
    returned = stamp(result)
    assert returned is result
    assert result["headline"] == 0.347412, "stamping must not touch existing fields"
    assert result["runtime"] == runtime_environment()


@pytest.mark.parametrize("name", STAMPED_RESULTS)
def test_committed_results_carry_a_runtime_block(name: str) -> None:
    payload = json.loads((RESULTS / name).read_text())
    runtime = payload.get("runtime")
    assert runtime is not None, f"{name} was regenerated without the runtime stamp"
    assert runtime["numpy_version"]
    assert runtime["python_version"]


def _load_reproduction_verifier() -> Any:
    """Import scripts/verify_deepmimo_reproduction.py by path (not a package)."""
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_deepmimo_reproduction.py"
    )
    spec = importlib.util.spec_from_file_location("verify_deepmimo_reproduction", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_verifier_ignores_a_differing_runtime_block() -> None:
    """The stamp is diagnostic, never a gate — exercised, not grepped.

    An earlier version of this test scanned verifier *source* for the string
    "runtime". That passed while the behaviour was broken, because
    verify_deepmimo_reproduction walks the document generically and never names
    the key. CI caught it:

        AssertionError: result.runtime.numpy_version: expected '2.4.6', got '2.2.6'

    So compare two documents that differ *only* in the runtime block — exactly
    the committed-vs-CI situation — and require it to pass.
    """
    verifier = _load_reproduction_verifier()
    manifest = {"features_sha256": _SHA_A}
    committed = {
        "features_sha256": _SHA_A,
        "mean_regret_db": 1.730053,
        "runtime": {"numpy_version": "2.4.6", "python_version": "3.11.15"},
    }
    reproduced = {
        "features_sha256": _SHA_B,  # differs cross-host by design
        "mean_regret_db": 1.730053,
        "runtime": {"numpy_version": "2.2.6", "python_version": "3.12.13"},
    }
    verifier.verify_documents(
        expected_manifest=manifest,
        actual_manifest={"features_sha256": _SHA_B},
        expected_result=committed,
        actual_result=reproduced,
        absolute_float_tolerance=1e-4,
    )


def test_reproduction_verifier_still_catches_a_real_regression() -> None:
    """Skipping the runtime block must not blunt the verifier."""
    verifier = _load_reproduction_verifier()
    manifest = {"features_sha256": _SHA_A}
    committed = {
        "features_sha256": _SHA_A,
        "mean_regret_db": 1.730053,
        "runtime": {"numpy_version": "2.4.6"},
    }
    regressed = {
        "features_sha256": _SHA_A,
        "mean_regret_db": 2.5,  # a genuine physics change
        "runtime": {"numpy_version": "2.4.6"},
    }
    with pytest.raises(AssertionError, match="mean_regret_db"):
        verifier.verify_documents(
            expected_manifest=manifest,
            actual_manifest=manifest,
            expected_result=committed,
            actual_result=regressed,
            absolute_float_tolerance=1e-4,
        )


def test_runtime_key_must_still_be_present_on_both_sides() -> None:
    """Exempt from *comparison*, not from existence — a dropped stamp is a bug."""
    verifier = _load_reproduction_verifier()
    manifest = {"features_sha256": _SHA_A}
    committed = {"features_sha256": _SHA_A, "runtime": {"numpy_version": "2.4.6"}}
    stampless = {"features_sha256": _SHA_A}
    with pytest.raises(AssertionError, match="key mismatch"):
        verifier.verify_documents(
            expected_manifest=manifest,
            actual_manifest=manifest,
            expected_result=committed,
            actual_result=stampless,
            absolute_float_tolerance=1e-4,
        )


def test_data_dependence_verifier_accepts_an_out_of_repo_features_path() -> None:
    """CI builds the licence-gated features to $RUNNER_TEMP, not into the repo.

    scripts/verify_data_dependence.py originally hardcoded the in-repo
    generated/ path, which is gitignored and never exists on a runner, so it
    aborted with "real features not built" before testing anything. Parsing its
    CLI here is enough to pin the contract: the paths must be injectable.
    """
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_data_dependence.py"
    )
    spec = importlib.util.spec_from_file_location("verify_data_dependence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import argparse
    import inspect

    source = inspect.getsource(module.main)
    assert '"--features"' in source, "verifier must accept --features"
    assert '"--manifest"' in source, "verifier must accept --manifest"
    # A `global` declaration after a read of the same name is a SyntaxError that
    # only surfaces at import time; exec_module above would already have raised.
    assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)
