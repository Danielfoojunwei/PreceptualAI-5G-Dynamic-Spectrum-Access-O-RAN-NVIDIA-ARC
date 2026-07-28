"""The runtime stamp must be present, informative, and inert.

These tests pin the two properties that make the stamp worth having: it records
the one dependency that can silently reseed a benchmark (numpy), and it never
participates in a reproduction assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from horizon_ric.runtime_env import runtime_environment, stamp

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"

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


@pytest.mark.parametrize("name", STAMPED_RESULTS)
def test_no_verifier_asserts_the_runtime_block(name: str) -> None:
    """The stamp is diagnostic, never a gate.

    CI legitimately runs a different numpy than the machine that committed the
    result (the realdata extra pulls deepmimo==4.0.0, which caps numpy at <2.3),
    so any verifier comparing this block would fail on every run.
    """
    scripts = (Path(__file__).resolve().parents[1] / "scripts").glob("verify_*.py")
    for script in scripts:
        source = script.read_text()
        assert '"runtime"' not in source and "'runtime'" not in source, (
            f"{script.name} references the runtime block; it must stay diagnostic"
        )
