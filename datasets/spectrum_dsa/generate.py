#!/usr/bin/env python3
"""Generator for the federated-DSA decision-trace dataset.

Produces a real, versioned dataset of federated Dynamic Spectrum Access decision
traces — honest rounds and ALIE/Fang/denial-of-spectrum poisoned rounds — each
trace recording the proposed (channel, power), the Shield's safe action, the
SafetyCertificate summary, and the hash-chained evidence-record hash. The output
is deterministic under the fixed seeds below, so re-running reproduces the
committed files byte-for-byte (the dataset is COMMITTED, not regenerated on use).

Usage:
    python datasets/spectrum_dsa/generate.py            # regenerate in place
    python datasets/spectrum_dsa/generate.py --out DIR  # write elsewhere

Schema (one JSON object per line, see DATASHEET.md):
    trace_id, round, attack, aggregator, decision_id, obs_state,
    proposed_channel, proposed_tx_power_dBm, emit,
    safe_frequency_hz, safe_bandwidth_hz, safe_tx_power_dBm, safe_eirp_dBm,
    shield_safe, shield_emit_blocked, shield_projected, shield_fallback_used,
    violated_invariants, chain_hash, label
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.spectrum import (
    DSAConfig,
    decide_and_record,
    federated_dsa_round,
    n_states,
)
from horizon_ric.spectrum.federated_q import QLearnConfig

HERE = Path(__file__).resolve().parent
DATASET_VERSION = "1.0.0"

# Fixed configuration → deterministic, reproducible dataset.
DSA_CFG = DSAConfig()  # 6 channels, 3 users, defaults
Q_CFG = QLearnConfig(episodes=25, epsilon=0.2)
N_CLIENTS = 12
N_MALICIOUS = 3
DECISIONS_PER_ROUND = 60
ROUND_SEEDS = [1, 2, 3]

# (attack, aggregator, label) combinations to record.
COMBINATIONS = [
    ("none", "fedavg", "honest"),
    ("none", "krum", "honest"),
    ("qtable_target", "fedavg", "poisoned_undefended"),
    ("qtable_target", "median", "poisoned_defended"),
    ("qtable_target", "krum", "poisoned_defended"),
    ("alie", "fedavg", "poisoned_undefended"),
    ("alie", "median", "poisoned_defended"),
    ("fang_krum", "krum", "poisoned_defended"),
    ("fang_median", "median", "poisoned_defended"),
]


def _record_for(attack: str, aggregator: str, label: str, seed: int) -> list[dict]:
    n_mal = 0 if attack == "none" else N_MALICIOUS
    res = federated_dsa_round(
        n_clients=N_CLIENTS, n_malicious=n_mal, dsa_cfg=DSA_CFG, q_cfg=Q_CFG,
        method=aggregator, attack=attack, seed=seed,
    )
    tmp = Path(tempfile.mkdtemp()) / "ev.jsonl"
    store = JsonlEvidenceStore(tmp)
    ns = n_states(DSA_CFG.n_channels)
    rows: list[dict] = []
    for i in range(DECISIONS_PER_ROUND):
        obs = i % ns
        # Vary requested power so some decisions stress the EIRP projection.
        req_power = 28.0 + (i % 5) * 4.0  # 28..44 dBm
        decision_id = f"{attack}-{aggregator}-r{seed}-d{i}"
        out = decide_and_record(
            res.global_q, obs, method=aggregator, evidence=store,
            decision_id=decision_id, rng_seed=i, requested_tx_power_dBm=req_power,
        )
        a = out.safe_action
        cert = out.certificate
        rows.append(
            {
                "trace_id": f"{attack}-{aggregator}-r{seed}",
                "round_seed": seed,
                "attack": attack,
                "aggregator": aggregator,
                "decision_id": decision_id,
                "obs_state": int(obs),
                "proposed_channel": int(out.record.chosen_action.get("dsa_channel", -1)),
                "requested_tx_power_dBm": req_power,
                "emit": bool(a.get("emit", True)),
                "safe_frequency_hz": float(a.get("frequency_hz", 0.0)),
                "safe_bandwidth_hz": float(a.get("bandwidth_hz", 0.0)),
                "safe_tx_power_dBm": float(a.get("tx_power_dBm", 0.0)),
                "safe_eirp_dBm": float(a.get("tx_power_dBm", 0.0)) + float(a.get("antenna_gain_dBi", 0.0)),
                "shield_safe": bool(cert.safe),
                "shield_emit_blocked": bool(cert.emit_blocked),
                "shield_projected": bool(cert.projected),
                "shield_fallback_used": bool(cert.fallback_used),
                "violated_invariants": list(cert.violated_ids),
                "chain_hash": out.chain_hash,
                "label": label,
            }
        )
    assert store.verify() == -1, "evidence chain must verify for a committed trace"
    return rows


def generate(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for seed in ROUND_SEEDS:
        for attack, aggregator, label in COMBINATIONS:
            all_rows.extend(_record_for(attack, aggregator, label, seed))

    data_path = out_dir / "traces.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Stats / manifest for provenance.
    n_illegal = sum(
        1
        for r in all_rows
        if r["emit"]
        and not r["shield_emit_blocked"]
        and (
            r["safe_frequency_hz"] - r["safe_bandwidth_hz"] / 2 < 3.40e9 - 1e-3
            or r["safe_frequency_hz"] + r["safe_bandwidth_hz"] / 2 > 3.50e9 + 1e-3
            or r["safe_eirp_dBm"] > 33.0 + 1e-6
        )
    )
    manifest = {
        "dataset": "horizon-ric-spectrum-dsa",
        "version": DATASET_VERSION,
        "rows": len(all_rows),
        "round_seeds": ROUND_SEEDS,
        "combinations": [
            {"attack": a, "aggregator": g, "label": label} for a, g, label in COMBINATIONS
        ],
        "labels": sorted({r["label"] for r in all_rows}),
        "illegal_emits_after_shield": n_illegal,
        "n_channels": DSA_CFG.n_channels,
        "n_users": DSA_CFG.n_users,
        "n_clients": N_CLIENTS,
        "n_malicious": N_MALICIOUS,
        "decisions_per_round": DECISIONS_PER_ROUND,
        "license": "Apache-2.0",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(HERE))
    args = ap.parse_args()
    manifest = generate(Path(args.out))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwrote {manifest['rows']} rows; illegal emits after shield = "
          f"{manifest['illegal_emits_after_shield']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
