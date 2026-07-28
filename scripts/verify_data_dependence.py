#!/usr/bin/env python3
"""Prove that benchmarks claiming real data actually let the data drive them.

A result JSON carrying ``features_sha256`` asserts the numbers came from the
real measured DeepMIMO channels. Nothing in the repository previously enforced
that. A benchmark could load the real file, ignore it, report a synthetic draw
and still stamp the provenance — and reading the source is a poor way to rule
that out, because the coupling is often several layers deep.

This verifier settles it empirically: perturb the real measurements, re-run, and
require the benchmark to notice. A benchmark passes if EITHER

  * its numbers move, or
  * it REJECTS the perturbed input as physically inconsistent.

The second outcome is the stronger one. Three of these benchmarks reconstruct
subband gains from the real per-path ray data and cross-check them, so a
perturbation that is not physically realisable is refused outright:

    RuntimeError: ray reconstruction disagrees with the committed subband
    gains: 5.999994 dB RMS

That is a benchmark proving it consumes the data, not merely referencing it.

TWO PERTURBATIONS, AND WHY BOTH ARE NEEDED
------------------------------------------
``shift``    adds a constant +6 dB to every measured subband gain.
``shuffle``  permutes gain vectors ACROSS receivers, so each real gain vector is
             paired with the wrong real position. Every marginal and scale
             statistic is preserved exactly; only the position->gain structure
             the models learn is destroyed.

``shift`` alone is not sufficient and it is worth recording why, because it
produced a false accusation during development. Several benchmarks standardise
their target::

    self.mean = gains.mean(axis=0)
    self.target = (gains - self.mean) / self.std

A uniform shift moves ``mean`` by exactly the same constant, so ``gains - mean``
is unchanged and the standardised target is bit-identical. Under ``shift`` alone
``fl_poisoning_suite`` moved 2 of 1091 numbers and looked like provenance
theatre; under ``shuffle`` it moves 848 of 1091. The benchmark was always fine —
the probe was in the null space of the pipeline. Any future perturbation added
here must be checked against that trap: it has to break a structure the model
actually uses, not apply a transform the model is invariant to by construction.

Usage::

    python scripts/verify_data_dependence.py                 # every benchmark
    python scripts/verify_data_dependence.py --only jamming_suite phy_fading
    python scripts/verify_data_dependence.py --quick         # a fast subset

Exits non-zero if any benchmark is invariant under every perturbation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from horizon_ric.data.lineage import compute_dataset_sha256  # noqa: E402

DEFAULT_FEATURES = _REPO / "datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"
DEFAULT_MANIFEST = _REPO / "datasets/deepmimo_asu_3p5/manifest.json"

# Provenance/environment fields are expected to differ or be irrelevant; they are
# not evidence that the physics was consumed.
_IGNORED_KEYS = frozenset({"features_sha256", "source_archive_sha256", "runtime"})

# benchmark -> (script, feature flag). Benchmarks differ in how they take input:
# neural_rx_pgd reads the channel file as --channel, and phy_fading_eval takes no
# --manifest at all. Encoding this explicitly beats guessing a uniform CLI.
CASES: dict[str, tuple[str, str, bool]] = {
    # name: (script, features-flag, accepts --manifest)
    "deepmimo_dsa": ("deepmimo_dsa_benchmark.py", "--features", True),
    "jamming_suite": ("jamming_suite.py", "--features", True),
    "phy_fading": ("phy_fading_eval.py", "--features", False),
    "secure_dsa": ("secure_dsa_benchmark.py", "--features", True),
    "dp_privacy": ("dp_privacy_suite.py", "--features", True),
    "subject_erasure": ("subject_erasure_suite.py", "--features", True),
    "fl_poisoning_suite": ("fl_poisoning_suite.py", "--features", True),
    "dsa_poison_suite": ("dsa_poison_suite.py", "--features", True),
    "poisoning_shield": ("poisoning_shield_benchmark.py", "--features", True),
    "neural_rx_pgd": ("neural_rx_pgd_benchmark.py", "--channel", True),
    "federated_unlearning": ("federated_unlearning_suite.py", "--features", True),
    "verifiable_secagg": ("verifiable_secagg_suite.py", "--features", True),
    "federated_coverage": ("federated_coverage_loop.py", "--features", True),
}

QUICK = ("deepmimo_dsa", "fl_poisoning_suite", "secure_dsa")

# A benchmark must move at least this fraction of its numeric outputs under the
# strongest perturbation. Chosen from measured separation: the genuine signals
# sit at 12.9% (deepmimo_dsa), 39.4% (secure_dsa) and 77.7% (fl_poisoning_suite
# under shuffle), while the known false negative sits at 0.2% (fl_poisoning_suite
# under shift). 5% is comfortably clear of the noise floor and well below every
# real signal.
MIN_MOVED_FRACTION = 0.05


def _numeric_leaves(obj: Any, path: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _IGNORED_KEYS:
                continue
            yield from _numeric_leaves(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _numeric_leaves(value, f"{path}[{index}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def _write_perturbed(
    features: Path, mode: str, out_dir: Path, seed: int = 20260728
) -> tuple[Path, Path]:
    rows = [json.loads(line) for line in features.open() if line.strip()]
    if mode == "shift":
        for row in rows:
            row["subband_gain_dbw"] = [g + 6.0 for g in row["subband_gain_dbw"]]
    elif mode == "shuffle":
        gains = [row["subband_gain_dbw"] for row in rows]
        order = np.random.default_rng(seed).permutation(len(rows))
        for index, row in enumerate(rows):
            row["subband_gain_dbw"] = gains[order[index]]
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown perturbation {mode!r}")

    feat_path = out_dir / f"features_{mode}.jsonl"
    with feat_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    manifest["features_sha256"] = compute_dataset_sha256(feat_path)
    man_path = out_dir / f"manifest_{mode}.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return feat_path, man_path


def _run(name: str, features: Path, manifest: Path, out: Path) -> tuple[int, str]:
    script, flag, takes_manifest = CASES[name]
    cmd = [sys.executable, str(_REPO / "benchmarks" / script), flag, str(features)]
    if takes_manifest:
        cmd += ["--manifest", str(manifest)]
    cmd += ["--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    tail = (proc.stderr or "").strip().splitlines()
    return proc.returncode, (tail[-1] if tail else "")


def check(name: str, work: Path) -> tuple[bool, str]:
    baseline = work / f"{name}_real.json"
    code, err = _run(name, DEFAULT_FEATURES, DEFAULT_MANIFEST, baseline)
    if code:
        return False, f"baseline run FAILED: {err[:110]}"
    reference = dict(_numeric_leaves(json.loads(baseline.read_text())))

    # Every perturbation is evaluated — never short-circuit on the first sign of
    # movement. A trickle of moved leaves is exactly the false-negative this
    # verifier exists to catch: under `shift`, fl_poisoning_suite moves 2 of 1091
    # (0.2%, indistinguishable from numerical noise) while being fully
    # data-driven, and a benchmark that was 99.8% synthetic would look identical.
    # Only the strongest perturbation is allowed to decide, against a threshold.
    observations: list[str] = []
    best = 0.0
    for mode in ("shift", "shuffle"):
        feat, man = _write_perturbed(DEFAULT_FEATURES, mode, work)
        out = work / f"{name}_{mode}.json"
        code, err = _run(name, feat, man, out)
        if code:
            # Refusing physically inconsistent input is a pass, and the strongest
            # possible one: the benchmark validated the data it consumes.
            return True, f"rejected {mode} input as inconsistent ({err[:60]})"
        actual = dict(_numeric_leaves(json.loads(out.read_text())))
        shared = [k for k in reference if k in actual]
        moved = [k for k in shared if abs(reference[k] - actual[k]) > 1e-9]
        fraction = len(moved) / len(shared) if shared else 0.0
        best = max(best, fraction)
        observations.append(f"{mode} {len(moved)}/{len(shared)} ({fraction:.1%})")

    detail = "; ".join(observations)
    if best >= MIN_MOVED_FRACTION:
        return True, detail
    return False, f"only {best:.1%} moved (need {MIN_MOVED_FRACTION:.0%}) — {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(CASES), default=None)
    parser.add_argument("--quick", action="store_true", help=f"just {', '.join(QUICK)}")
    args = parser.parse_args()

    names = list(args.only or (QUICK if args.quick else CASES))
    if not DEFAULT_FEATURES.is_file():
        print(f"real features not built: {DEFAULT_FEATURES}", file=sys.stderr)
        print("run datasets/deepmimo_asu_3p5/build.py first", file=sys.stderr)
        return 2

    failures = []
    with tempfile.TemporaryDirectory(prefix="datadep-") as tmp:
        work = Path(tmp)
        for name in names:
            passed, detail = check(name, work)
            print(f"  {'PASS' if passed else 'FAIL'}  {name:24} {detail}")
            if not passed:
                failures.append(name)

    if failures:
        print(
            f"\n{len(failures)} benchmark(s) claim real-data provenance without "
            f"depending on it: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(names)} benchmark(s) demonstrably consume the real measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
