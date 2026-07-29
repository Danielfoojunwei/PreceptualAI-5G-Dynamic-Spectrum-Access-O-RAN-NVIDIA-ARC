#!/usr/bin/env python3
"""Evaluate DSA channel selection and Shield legality on DeepMIMO features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.runtime_env import stamp
from horizon_ric.shield import default_terrestrial_shield

BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
N_SUBBANDS = 6
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
REQUESTED_TX_POWER_DBM = 44.0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _frequency_for(channel: int) -> float:
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    return BAND_LO_HZ + (channel + 0.5) * width


def _evaluate_strategy(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ,
        band_hi_hz=BAND_HI_HZ,
        max_eirp_dBm=MAX_EIRP_DBM,
    )
    selected_gains: list[float] = []
    regrets: list[float] = []
    illegal_emits = 0
    projected = 0
    blocked = 0
    carrier_width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS

    for index, row in enumerate(rows):
        gains = np.asarray(row["subband_gain_dbw"], dtype=np.float64)
        if gains.shape != (N_SUBBANDS,) or not np.all(np.isfinite(gains)):
            raise ValueError(f"invalid gains in feature row {index}")
        if strategy == "best_subband":
            channel = int(np.argmax(gains))
        elif strategy == "fixed_subband_0":
            channel = 0
        elif strategy == "random_subband":
            channel = int(rng.integers(0, N_SUBBANDS))
        elif strategy == "worst_subband":
            channel = int(np.argmin(gains))
        else:
            raise ValueError(f"unknown strategy {strategy}")

        proposed = {
            "block": "deepmimo_dsa_policy",
            "frequency_hz": _frequency_for(channel),
            "bandwidth_hz": carrier_width,
            "tx_power_dBm": REQUESTED_TX_POWER_DBM,
            "antenna_gain_dBi": ANTENNA_GAIN_DBI,
        }
        disposition = shield.dispose(proposed, decision_id=f"{strategy}-{index}")
        certificate = disposition.certificate
        if certificate.emit_blocked:
            blocked += 1
        else:
            safe = disposition.safe_action
            lo = safe["frequency_hz"] - safe["bandwidth_hz"] / 2
            hi = safe["frequency_hz"] + safe["bandwidth_hz"] / 2
            eirp = safe["tx_power_dBm"] + safe["antenna_gain_dBi"]
            if (
                lo < BAND_LO_HZ - 1e-3
                or hi > BAND_HI_HZ + 1e-3
                or eirp > MAX_EIRP_DBM + 1e-6
            ):
                illegal_emits += 1
        projected += int(certificate.projected)
        selected_gains.append(float(gains[channel]))
        regrets.append(float(np.max(gains) - gains[channel]))

    return {
        "decisions": len(rows),
        "mean_selected_gain_dbw": round(float(np.mean(selected_gains)), 6),
        "mean_regret_db": round(float(np.mean(regrets)), 6),
        "p95_regret_db": round(float(np.percentile(regrets, 95)), 6),
        "shield_projected": projected,
        "shield_blocked": blocked,
        "illegal_emits_after_shield": illegal_emits,
    }


def run(features: Path, manifest_path: Path) -> dict[str, Any]:
    rows = _load_rows(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")

    strategies = {}
    for offset, strategy in enumerate(
        ["best_subband", "fixed_subband_0", "random_subband", "worst_subband"]
    ):
        strategies[strategy] = _evaluate_strategy(
            rows,
            strategy=strategy,
            rng=np.random.default_rng(20260725 + offset),
        )

    best = strategies["best_subband"]["mean_selected_gain_dbw"]
    fixed = strategies["fixed_subband_0"]["mean_selected_gain_dbw"]
    random = strategies["random_subband"]["mean_selected_gain_dbw"]
    return {
        "benchmark": "DeepMIMO ray-traced DSA subband selection",
        "dataset": manifest["dataset"],
        "scenario": manifest["scenario"],
        "data_kind": manifest["data_kind"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "features_sha256": manifest["features_sha256"],
        "receivers": len(rows),
        "strategies": strategies,
        "best_vs_fixed_gain_db": round(best - fixed, 6),
        "best_vs_random_gain_db": round(best - random, 6),
        "scope_note": (
            "Published site-specific ray tracing; not OTA data, live O-RAN traffic, "
            "vendor interoperability, RF conformance, or carrier-scale evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/manifest.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/deepmimo_dsa.json"),
    )
    args = parser.parse_args()
    result = run(args.features, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    illegal = sum(
        row["illegal_emits_after_shield"] for row in result["strategies"].values()
    )
    return 0 if illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
