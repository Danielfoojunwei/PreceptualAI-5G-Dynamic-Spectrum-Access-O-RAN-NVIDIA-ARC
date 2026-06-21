#!/usr/bin/env python3
"""Poisoning attack → defense benchmark for Horizon-RIC.

Two AI-RAN-native security claims, measured:

1. **The Shield holds under a poisoned neural-PHY model.** We stream decisions
   from a "neural-RX / learned-constellation" head that an attacker has poisoned
   so a fraction of its outputs are illegal (out-of-band, over-EIRP, illegal
   constellation order, blown PAPR, or a TBLER regression). We compare an
   *unguarded* emit path against the *Shield* path and report how many illegal
   policies would reach the air interface.

2. **Robust aggregation neutralises model-poisoning clients.** A federation of
   honest cells plus Byzantine clients; we report the distance of the global
   model from the honest mean under FedAvg vs Krum / median / trimmed-mean.

Pure numpy — no torch, no accelerator. Run:  python benchmarks/poisoning_shield_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from horizon_ric.federated import coordinate_median, fedavg, krum, trimmed_mean
from horizon_ric.policy.li_constraint import LIConstraint
from horizon_ric.shield import default_terrestrial_shield

BAND_LO, BAND_HI = 3.40e9, 3.50e9
MAX_EIRP = 33.0


def _honest_decision(rng: np.random.Generator) -> dict:
    return {
        "block": "neural_rx",
        "frequency_hz": float(rng.uniform(BAND_LO + 15e6, BAND_HI - 15e6)),
        "bandwidth_hz": 20e6,
        "tx_power_dBm": float(rng.uniform(10, 24)),
        "antenna_gain_dBi": 6.0,
        "constellation_order": int(rng.choice([4, 16, 64, 256])),
        "papr_dB": float(rng.uniform(5.5, 8.0)),
        "predicted_tbler": 0.05,
        "baseline_tbler": 0.05,
        "demap_confidence": 0.9,
    }


def _poison(decision: dict, rng: np.random.Generator) -> dict:
    """Corrupt one decision the way a poisoned model would."""
    d = dict(decision)
    kind = rng.choice(
        ["out_of_band", "over_eirp", "illegal_order", "blown_papr", "tbler_regress"]
    )
    if kind == "out_of_band":
        d["frequency_hz"] = BAND_HI + 30e6
    elif kind == "over_eirp":
        d["tx_power_dBm"] = 45.0  # EIRP 51 dBm
    elif kind == "illegal_order":
        d["constellation_order"] = int(rng.choice([3, 7, 512, 1024]))
    elif kind == "blown_papr":
        d["papr_dB"] = 13.0
    else:  # tbler_regress — neural-RX much worse than the classical baseline
        d["predicted_tbler"] = 0.6
        d["demap_confidence"] = 0.05
    d["_attack"] = kind
    return d


def _is_illegal(action: dict) -> bool:
    """Ground-truth legality of an action at the air interface."""
    lo = action["frequency_hz"] - action["bandwidth_hz"] / 2
    hi = action["frequency_hz"] + action["bandwidth_hz"] / 2
    if lo < BAND_LO or hi > BAND_HI:
        return True
    if action["tx_power_dBm"] + action["antenna_gain_dBi"] > MAX_EIRP + 1e-6:
        return True
    if int(action.get("constellation_order", 4)) not in (4, 16, 64, 256):
        return True
    if action.get("papr_dB", 0.0) > 8.5 + 1e-6:
        return True
    return False


def shield_benchmark(n: int = 10000, poison_rate: float = 0.30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    li = LIConstraint(rules=[], fail_closed=False, deployment_audit_record="bench")
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP, li_constraint=li
    )

    poisoned_inputs = 0
    unguarded_illegal = 0
    shielded_illegal = 0
    shielded_blocked = 0
    certificates = 0

    for i in range(n):
        d = _honest_decision(rng)
        if rng.random() < poison_rate:
            d = _poison(d, rng)
            poisoned_inputs += 1

        # Unguarded path: whatever the model said goes straight to the wire.
        if _is_illegal(d):
            unguarded_illegal += 1

        # Shielded path.
        disp = shield.dispose(d, decision_id=f"d{i}", rng_seed=seed)
        certificates += 1
        if disp.certificate.emit_blocked:
            shielded_blocked += 1
            continue  # refused — nothing reaches the air interface
        if _is_illegal(disp.safe_action):
            shielded_illegal += 1

    return {
        "decisions": n,
        "poison_rate": poison_rate,
        "poisoned_inputs": poisoned_inputs,
        "unguarded_illegal_emits": unguarded_illegal,
        "shielded_illegal_emits": shielded_illegal,
        "shielded_blocked_emits": shielded_blocked,
        "certificates_issued": certificates,
        "shield_prevented": unguarded_illegal - shielded_illegal,
        "result": "PASS" if shielded_illegal == 0 else "FAIL",
    }


def federated_benchmark(
    n_honest: int = 20, n_byz: int = 8, dim: int = 200, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    honest = [rng.normal(0.0, 1.0, size=dim) for _ in range(n_honest)]
    honest_mean = np.mean(honest, axis=0)
    # Byzantine clients push a large coordinated update.
    byz = [np.full(dim, 50.0) + rng.normal(0, 1, dim) for _ in range(n_byz)]
    updates = honest + byz

    def dist(agg: np.ndarray) -> float:
        return float(np.linalg.norm(agg - honest_mean))

    return {
        "n_honest": n_honest,
        "n_byzantine": n_byz,
        "dim": dim,
        "dist_from_honest_mean": {
            "fedavg": dist(fedavg(updates)),
            "krum": dist(krum(updates, f=n_byz).aggregate),
            "median": dist(coordinate_median(updates)),
            "trimmed_mean": dist(trimmed_mean(updates, beta=n_byz)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=10000)
    ap.add_argument("--poison-rate", type=float, default=0.30)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    report = {
        "shield": shield_benchmark(args.decisions, args.poison_rate),
        "federated": federated_benchmark(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)

    s = report["shield"]
    f = report["federated"]["dist_from_honest_mean"]
    print(
        f"\nSHIELD: {s['unguarded_illegal_emits']} illegal emits without the Shield → "
        f"{s['shielded_illegal_emits']} with it  [{s['result']}]"
    )
    print(
        f"FEDERATED: FedAvg dist {f['fedavg']:.1f} vs Krum {f['krum']:.2f} / "
        f"median {f['median']:.2f} / trimmed {f['trimmed_mean']:.2f}"
    )
    return 0 if s["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
