"""Federated-DSA loop invariants on a synthetic-but-schema-real fixture.

The real DeepMIMO feature file is licence-gated and not in the repo, so this test
builds a small feature set with the *exact* production schema
(``subband_gain_dbw`` + ``position_m``) and a deliberately clear frequency-
selective structure, then asserts the loop's security invariants hold:

* clean FedAvg recovers the true best subband,
* a model-replacement poison breaks (inverts + diverges) plain FedAvg,
* Krum bounds the poison and stays aligned to the true profile,
* the Shield corrects an out-of-band proposal so it never reaches the RAN,
* the evidence chain verifies intact and pinpoints a tamper.

These are the same code paths the real-data run exercises; only the input bytes
differ. The heavy DeepMIMO/Sionna extras are NOT required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import federated_dsa_loop as loop


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A 12-cluster, 360-receiver fixture with a clear per-cluster best subband."""
    rng = np.random.default_rng(2026)
    n_clusters = 12
    per = 30
    # Each cluster sits at its own 2D location and prefers a subband; subband 4
    # is preferred most often so the global argmax is unambiguous.
    prefs = [4, 4, 4, 5, 0, 1, 3, 4, 5, 4, 4, 2]
    rows = []
    idx = 0
    for c in range(n_clusters):
        cx, cy = float(c * 40 - 200), float((c % 4) * 30 - 45)
        for _ in range(per):
            base = -138.0 + rng.normal(0, 3.0)  # large-scale path loss (per-Rx)
            g = base + rng.normal(0, 0.8, loop.N_SUBBANDS)
            g[prefs[c]] += 2.5  # frequency-selective bump toward the local best
            rows.append(
                {
                    "receiver_index": idx,
                    "position_m": [
                        cx + float(rng.normal(0, 4)),
                        cy + float(rng.normal(0, 4)),
                        1.5,
                    ],
                    "subband_gain_dbw": [round(float(x), 6) for x in g],
                    "best_subband": int(np.argmax(g)),
                    "worst_subband": int(np.argmin(g)),
                }
            )
            idx += 1
    features = tmp_path / "features.jsonl"
    features.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "synthetic-fixture",
                "scenario": "test",
                "data_kind": "synthetic schema-compatible fixture",
                "sampled_receivers": len(rows),
                "features_sha256": "n/a",
                "source_tree_sha256": "n/a",
            }
        ),
        encoding="utf-8",
    )
    return features, manifest


def test_federated_dsa_loop_security_invariants(tmp_path: Path) -> None:
    features, manifest = _write_fixture(tmp_path)
    result = loop.run(
        features,
        manifest,
        audit_path=tmp_path / "audit.jsonl",
        n_sites=8,
        n_malicious=2,
        rounds=20,
    )
    o = result["outcomes"]

    # Clean FedAvg recovers the true population-best subband.
    assert o["clean_fedavg"]["default_subband"] == result["true_profile_argmax"]
    assert o["clean_fedavg"]["cos_to_true_profile"] > 0.8
    assert not o["clean_fedavg"]["diverged"]

    # A model-replacement poison inverts and diverges plain FedAvg.
    assert o["poisoned_fedavg"]["cos_to_true_profile"] < 0.0
    assert o["poisoned_fedavg"]["diverged"]

    # Krum bounds the poison and stays aligned to the true profile.
    assert not o["poisoned_krum"]["diverged"]
    assert o["poisoned_krum"]["cos_to_true_profile"] > 0.0

    # Personalization over the non-IID split is a non-negative real gain.
    assert result["personalization_gain_db"] >= 0.0

    # DP release carries a finite certified budget.
    assert o["dp_fedavg_clean"]["epsilon"] is not None
    assert o["dp_fedavg_clean"]["epsilon"] > 0.0


def test_federated_dsa_trust_chain(tmp_path: Path) -> None:
    features, manifest = _write_fixture(tmp_path)
    result = loop.run(
        features,
        manifest,
        audit_path=tmp_path / "audit.jsonl",
        n_sites=8,
        n_malicious=2,
        rounds=15,
    )
    tc = result["trust_chain"]

    # The out-of-band, over-EIRP proposal is projected to a legal action and the
    # raw unsafe emit is refused, so it never reaches the RAN.
    assert tc["unsafe_oob_corrected"] is True
    assert tc["unsafe_oob_reached_ran"] is False
    assert tc["unsafe_oob_safe_frequency_ghz"] <= loop.BAND_HI_HZ / 1e9 + 1e-6
    assert tc["unsafe_oob_safe_eirp_dbm"] <= loop.MAX_EIRP_DBM + 1e-6

    # The freshly written chain verifies intact, then the injected tamper is
    # detected at a concrete index.
    assert tc["verify_first_broken_index"] == -1
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_after_tamper_index"] not in (None, -1)


def test_selected_policies_emit_cleanly(tmp_path: Path) -> None:
    features, manifest = _write_fixture(tmp_path)
    result = loop.run(features, manifest, audit_path=tmp_path / "audit.jsonl", rounds=10)
    # The two model-selected subband policies are legal (EIRP 32 dBm, in band):
    # no Shield correction, guard chain passes, they emit cleanly.
    selected = [
        g for g in result["trust_chain"]["gate_records"] if g["label"] != "unsafe-oob-proposal"
    ]
    assert selected, "expected selected-subband gate records"
    for g in selected:
        assert g["corrected"] is False
        assert g["emitted_clean"] is True
        assert g["guard_refused"] == []
