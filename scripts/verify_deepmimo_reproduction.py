#!/usr/bin/env python3
"""Verify a cross-host DeepMIMO reproduction without hiding float drift.

Source hashes, configuration, counts, strings and safety outcomes must match
exactly. Derived floating-point channel values may differ slightly across CPU
implementations, so JSON floats are compared with a declared absolute
tolerance. Each run still records and internally binds its exact feature-file
SHA-256.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

FEATURE_HASH = "features_sha256"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_feature_binding(manifest: dict[str, Any], result: dict[str, Any]) -> None:
    feature_hash = manifest.get(FEATURE_HASH)
    if not isinstance(feature_hash, str) or not SHA256_RE.fullmatch(feature_hash):
        raise AssertionError("manifest features_sha256 is not a lowercase SHA-256")
    if result.get(FEATURE_HASH) != feature_hash:
        raise AssertionError("benchmark is not bound to its generated feature hash")


def _compare(
    expected: Any,
    actual: Any,
    *,
    path: str,
    absolute_float_tolerance: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected object, got {type(actual).__name__}")
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise AssertionError(f"{path}: key mismatch; missing={missing}, extra={extra}")
        for key in sorted(expected):
            if key == FEATURE_HASH:
                continue
            _compare(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                absolute_float_tolerance=absolute_float_tolerance,
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list shape mismatch")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                absolute_float_tolerance=absolute_float_tolerance,
            )
        return

    if isinstance(expected, bool) or isinstance(expected, int):
        if type(actual) is not type(expected) or actual != expected:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise AssertionError(f"{path}: expected numeric value, got {actual!r}")
        if not math.isfinite(actual) or not math.isclose(
            expected,
            float(actual),
            rel_tol=0.0,
            abs_tol=absolute_float_tolerance,
        ):
            raise AssertionError(
                f"{path}: expected {expected!r}, got {actual!r}; "
                f"absolute tolerance={absolute_float_tolerance}"
            )
        return

    if type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def verify_documents(
    *,
    expected_manifest: dict[str, Any],
    actual_manifest: dict[str, Any],
    expected_result: dict[str, Any],
    actual_result: dict[str, Any],
    absolute_float_tolerance: float = 1e-4,
) -> None:
    """Verify exact provenance/structure and tolerance-bounded float outputs."""
    if absolute_float_tolerance < 0 or not math.isfinite(absolute_float_tolerance):
        raise ValueError("absolute_float_tolerance must be finite and non-negative")
    _require_feature_binding(expected_manifest, expected_result)
    _require_feature_binding(actual_manifest, actual_result)
    _compare(
        expected_manifest,
        actual_manifest,
        path="manifest",
        absolute_float_tolerance=absolute_float_tolerance,
    )
    _compare(
        expected_result,
        actual_result,
        path="result",
        absolute_float_tolerance=absolute_float_tolerance,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--actual-manifest", type=Path, required=True)
    parser.add_argument("--expected-result", type=Path, required=True)
    parser.add_argument("--actual-result", type=Path, required=True)
    parser.add_argument("--absolute-float-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    verify_documents(
        expected_manifest=_load(args.expected_manifest),
        actual_manifest=_load(args.actual_manifest),
        expected_result=_load(args.expected_result),
        actual_result=_load(args.actual_result),
        absolute_float_tolerance=args.absolute_float_tolerance,
    )
    print(
        "DeepMIMO reproduction verified: exact provenance/structure and "
        f"float tolerance {args.absolute_float_tolerance:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
