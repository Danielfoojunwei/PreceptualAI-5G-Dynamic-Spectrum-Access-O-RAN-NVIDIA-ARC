#!/usr/bin/env python3
"""Federated DSA loop on **real DeepMIMO** ray-traced channels.

This is the first Horizon artifact that drives the *federated + trust core* with
real measured-physics data instead of synthetic gradient vectors. It closes the
loop end to end:

    real ray-traced MIMO channels (DeepMIMO ASU 3.5 GHz)
        -> geographic, non-IID federated clients (partitioned by real position)
        -> T rounds of FedAvg with real local SGD on each client's own channels
        -> Byzantine-robust aggregation (Krum / coordinate-median / trimmed-mean)
        -> differentially-private release (RDP accountant, certified epsilon)
        -> global subband-quality model  ->  DSA default-subband selection
        -> Decision Safety Shield (band + EIRP legality, projection operator)
        -> hash-chained, tamper-evident evidence record

The learning task is genuine and physics-grounded. Each receiver contributes its
real per-subband channel gain vector ``g in R^6`` (dBW, from Wireless InSite ray
tracing). We remove large-scale path loss by centering each receiver's vector
across its 6 subbands, leaving the *relative* frequency-selective structure
``s = g - mean(g)`` -- which subbands are locally better or worse. The federated
model ``w in R^6`` is trained by minimising ``L_c(w) = 1/2 * E_r ||w - s_r||^2``
on each client ``c`` with real SGD; its FedAvg fixed point is the population
subband-quality prior. ``argmax(w)`` is the network-wide default subband a cell
uses when it must choose without per-UE CSI (initial access, common channels, a
coarse spectrum reservation). We grade that choice by the **real** mean regret in
dB against each receiver's oracle-best subband.

HONEST magnitude on this dataset. The ASU campus channel is near frequency-flat
on average (the six 100 MHz subbands differ by only ~0.3 dB in mean gain), so the
*global-default* subband is worth only a few tenths of a dB -- we report that
truthfully rather than inflate it. The dramatic, unambiguous signal is the
security mechanism operating on real updates: a Byzantine model-replacement
poison *inverts* the learned model (cosine to the true population profile goes
negative -- it points the wrong way, selecting the globally-worst subband) and
makes plain FedAvg diverge, while Krum keeps it aligned (positive cosine, near-
optimal subband) on the very same real channels. Per-site personalization over
the real non-IID split is worth a further ~0.2-0.57 dB.

Everything downstream (Shield legality, guard chain, evidence chain, tamper
detection) runs on the actually-selected subband, so the whole topology is
exercised on real data, not a mock.

Pure numpy + the shipped ``horizon_ric`` modules. No torch, no GPU. The real
DeepMIMO feature file is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` or point ``--features`` at a cached copy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.federated import robust
from horizon_ric.federated.dp import DPConfig, RDPAccountant, dp_fedavg
from horizon_ric.federated.poison_attacks import scaling_attack
from horizon_ric.policy.emit_guards import run_guard_chain
from horizon_ric.shield import default_terrestrial_shield

# --- Spectrum geometry (shared with benchmarks/deepmimo_dsa_benchmark.py). -----
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
N_SUBBANDS = 6
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
# A genuinely legal request (EIRP 32 dBm < 33): the selected subband emits
# without any Shield correction, so the guard chain passes cleanly.
SAFE_TX_POWER_DBM = 26.0

# --- Federation defaults. ------------------------------------------------------
N_SITES = 10
N_MALICIOUS = 3  # below the Krum breakpoint f < (K-2)/2 and the median f < K/2.
ROUNDS = 25
LOCAL_EPOCHS = 5
LOCAL_LR = 0.5
POISON_BOOST = 6.0
DP_CLIP = 1.0
DP_NOISE = 8.0  # yields a certified eps ~ 3.2 at delta=1e-5 over ROUNDS rounds.
DP_DELTA = 1e-5


def _subband_center_hz(subband: int) -> float:
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    return BAND_LO_HZ + (subband + 0.5) * width


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _gain_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    """Real per-subband channel gains, shape ``(n_receivers, N_SUBBANDS)`` dBW."""
    gains = np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)
    if gains.shape[1] != N_SUBBANDS or not np.all(np.isfinite(gains)):
        raise ValueError("feature rows have malformed subband_gain_dbw")
    return gains


def _shape_matrix(gains: np.ndarray) -> np.ndarray:
    """Per-receiver large-scale-removed relative subband structure.

    Centering each row across its 6 subbands removes the receiver's large-scale
    path loss (a per-row constant) and keeps only the frequency-selective shape
    -- the real ray-traced signal the federation learns.
    """
    return gains - gains.mean(axis=1, keepdims=True)


def _partition_geographic(rows: list[dict[str, Any]], n_sites: int) -> list[np.ndarray]:
    """Partition receivers into ``n_sites`` clients by real 2D position (Lloyd).

    Ray tracing is site-specific, so clustering by position yields genuinely
    non-IID clients: different campus areas see different dominant scatterers and
    therefore different which-subband-is-best distributions.
    """
    pos = np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(1234)
    # Deterministic k-means (Lloyd) with farthest-point seeding.
    centers = [pos[int(rng.integers(len(pos)))]]
    for _ in range(1, n_sites):
        d = np.min([np.linalg.norm(pos - c, axis=1) for c in centers], axis=0)
        centers.append(pos[int(np.argmax(d))])
    centers = np.asarray(centers)
    assign = np.zeros(len(pos), dtype=np.int64)
    for _ in range(50):
        assign = np.argmin(
            np.stack([np.linalg.norm(pos - c, axis=1) for c in centers], axis=1),
            axis=1,
        )
        new = np.asarray(
            [
                pos[assign == k].mean(axis=0) if np.any(assign == k) else centers[k]
                for k in range(n_sites)
            ]
        )
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return [np.flatnonzero(assign == k) for k in range(n_sites)]


def _local_train(
    w_global: np.ndarray,
    site_shapes: np.ndarray,
    *,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Real local SGD on one client's channels; returns the update ``w - w_global``.

    Minimises ``1/2 * mean_r ||w - s_r||^2`` with mini-batch gradient descent.
    The full-batch gradient is ``w - mean(s_r)``; mini-batching injects the real
    per-receiver stochasticity of the client's own channel population.
    """
    w = w_global.astype(np.float64).copy()
    n = site_shapes.shape[0]
    batch = max(1, n // 2)
    for _ in range(epochs):
        idx = rng.permutation(n)[:batch]
        grad = w - site_shapes[idx].mean(axis=0)
        w = w - lr * grad
    return w - w_global


def _mean_regret_db(gains: np.ndarray, subband: int) -> tuple[float, float]:
    """Real mean / p95 regret (dB) of using ``subband`` as the network default."""
    regret = gains.max(axis=1) - gains[:, subband]
    return float(regret.mean()), float(np.percentile(regret, 95))


@dataclass
class FederationOutcome:
    method: str
    poisoned: bool
    default_subband: int
    mean_regret_db: float
    p95_regret_db: float
    cos_to_true_profile: float
    w_norm: float
    diverged: bool
    epsilon: float | None
    w: list[float]


def _run_federation(
    site_shapes: list[np.ndarray],
    gains: np.ndarray,
    true_profile: np.ndarray,
    *,
    method: str,
    poisoned: bool,
    rounds: int,
    n_malicious: int,
    dp: bool,
    seed: int,
) -> FederationOutcome:
    """T rounds of FedAvg/robust FL, optionally poisoned and/or DP-released.

    ``true_profile`` is the population mean subband-shape ``mu*`` (computed
    directly from every receiver) -- the physically-meaningful target. We report
    cosine to it (scale-invariant), so a model-replacement poison that *inverts*
    the learned direction shows up as a negative cosine even as its magnitude
    diverges.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(N_SUBBANDS, dtype=np.float64)
    n_sites = len(site_shapes)
    mal_ids = set(range(n_malicious)) if poisoned else set()
    accountant = RDPAccountant() if dp else None
    dp_cfg = DPConfig(clip_norm=DP_CLIP, noise_multiplier=DP_NOISE)

    for _ in range(rounds):
        benign_updates: list[np.ndarray] = [
            _local_train(w, site_shapes[c], epochs=LOCAL_EPOCHS, lr=LOCAL_LR, rng=rng)
            for c in range(n_sites)
            if c not in mal_ids
        ]
        # Byzantine clients craft a model-replacement poison from the honest
        # updates they observe (a standard white-box FL threat model).
        if mal_ids:
            mal_updates = scaling_attack(benign_updates, len(mal_ids), boost=POISON_BOOST)
            updates = benign_updates + list(mal_updates)
        else:
            updates = benign_updates

        if dp:
            # DP release path: per-client clip + Gaussian noise + RDP accountant.
            delta = dp_fedavg(updates, dp_cfg, rng=rng, accountant=accountant)
        else:
            # Match the trimmed-mean trim depth to the corruption level f so the
            # aggregator has a chance to drop every malicious coordinate.
            delta = robust.aggregate(
                updates,
                method=method,
                f=max(n_malicious, 1),
                beta=max(n_malicious, 1),
            )
        w = w + delta

    default_subband = int(np.argmax(w))
    mean_reg, p95_reg = _mean_regret_db(gains, default_subband)
    eps = accountant.get_epsilon(DP_DELTA) if accountant is not None else None
    w_norm = float(np.linalg.norm(w))
    denom = w_norm * float(np.linalg.norm(true_profile)) + 1e-12
    cos = float(w @ true_profile) / denom
    return FederationOutcome(
        method=("dp_fedavg" if dp else method),
        poisoned=poisoned,
        default_subband=default_subband,
        mean_regret_db=round(mean_reg, 4),
        p95_regret_db=round(p95_reg, 4),
        cos_to_true_profile=round(cos, 4),
        w_norm=round(w_norm, 4) if w_norm < 1e6 else float(f"{w_norm:.3e}"),
        diverged=bool(w_norm > 1e2),
        epsilon=(round(eps, 4) if eps is not None else None),
        w=([round(float(x), 4) for x in w] if not bool(w_norm > 1e2) else []),
    )


def _shield_gate(
    subband: int,
    store: JsonlEvidenceStore,
    *,
    decision_id: str,
    label: str,
    tx_power_dbm: float = SAFE_TX_POWER_DBM,
    frequency_hz: float | None = None,
    bandwidth_hz: float | None = None,
) -> dict[str, Any]:
    """Emit a spectrum policy through the Shield and record it to the chain.

    The Shield is a *projection* operator: an unsafe proposal is corrected onto
    the nearest legal action (band membership + EIRP cap) rather than silently
    passed. A policy that is already legal needs no correction and emits cleanly;
    one that had to be corrected trips the ``corrections_not_audited`` guard so
    the raw action never reaches the RAN until the correction is on the chain.
    """
    width = bandwidth_hz or (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    freq = frequency_hz or _subband_center_hz(subband)
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )
    proposed = {
        "block": "policy_emit",
        "policy_type": "horizon.spectrum.reservation",
        "frequency_hz": freq,
        "bandwidth_hz": width,
        "tx_power_dBm": tx_power_dbm,
        "antenna_gain_dBi": ANTENNA_GAIN_DBI,
    }
    disp = shield.dispose(proposed, {}, decision_id=decision_id)
    cert = disp.certificate
    guard_fails = run_guard_chain(certificate=cert, elapsed_ms=1.0)
    safe = disp.safe_action
    req_eirp = tx_power_dbm + ANTENNA_GAIN_DBI
    safe_eirp = safe["tx_power_dBm"] + safe["antenna_gain_dBi"]
    store.append(
        DecisionRecord.new(
            decision_id=decision_id,
            rapp_instance_id="federated-dsa",
            state_hash=f"subband-{subband}",
            chosen_action={
                "label": label,
                "policy_type": "horizon.spectrum.reservation",
                "subband": subband,
                "requested_frequency_hz": freq,
                "safe_frequency_hz": safe["frequency_hz"],
                "projected": bool(cert.projected),
                "guard_refused": bool(guard_fails),
            },
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
            ),
            rejected_alternatives=[],
            model_versions=ModelVersions(
                encoder="none",
                risk_heads="fed-subband-prior",
                dyna="none",
                policy="dsa",
                constraint_layer="terrestrial-shield",
                rapp="0.2.0",
            ),
        )
    )
    return {
        "label": label,
        "subband": subband,
        "requested_frequency_ghz": round(freq / 1e9, 4),
        "safe_frequency_ghz": round(safe["frequency_hz"] / 1e9, 4),
        "requested_eirp_dbm": round(req_eirp, 2),
        "safe_eirp_dbm": round(safe_eirp, 2),
        "safe": bool(cert.safe),
        "projected": bool(cert.projected),
        "corrected": bool(abs(safe["frequency_hz"] - freq) > 1.0 or safe_eirp < req_eirp - 1e-6),
        "emitted_clean": not bool(guard_fails),
        "guard_refused": [f.guard_id for f in guard_fails],
    }


def run(
    features: Path,
    manifest_path: Path,
    *,
    audit_path: Path | None = None,
    n_sites: int = N_SITES,
    n_malicious: int = N_MALICIOUS,
    rounds: int = ROUNDS,
) -> dict[str, Any]:
    rows = _load_rows(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")

    gains = _gain_matrix(rows)
    shapes = _shape_matrix(gains)
    true_profile = shapes.mean(axis=0)  # population subband-shape prior mu*
    sites = _partition_geographic(rows, n_sites)
    site_shapes = [shapes[idx] for idx in sites]

    # --- Real per-site heterogeneity + personalization value. -----------------
    def reg(gg: np.ndarray, k: int) -> float:
        return float((gg.max(axis=1) - gg[:, k]).mean())

    global_best = int(np.argmin([reg(gains, k) for k in range(N_SUBBANDS)]))
    site_profiles = []
    tot_local = tot_global = n_tot = 0.0
    for k, idx in enumerate(sites):
        if len(idx) == 0:
            continue
        gg = gains[idx]
        per = [reg(gg, j) for j in range(N_SUBBANDS)]
        local_best = int(np.argmin(per))
        best_hist = np.bincount(np.argmax(gg, axis=1), minlength=N_SUBBANDS)
        site_profiles.append(
            {
                "site": k,
                "receivers": int(len(idx)),
                "local_best_subband": local_best,
                "local_best_regret_db": round(per[local_best], 4),
                "global_best_regret_db": round(reg(gg, global_best), 4),
                "local_best_subband_hist": [int(v) for v in best_hist],
            }
        )
        tot_local += per[local_best] * len(idx)
        tot_global += reg(gg, global_best) * len(idx)
        n_tot += len(idx)
    personalization_gain_db = round((tot_global - tot_local) / n_tot, 4)

    # --- Federated learning: clean, poisoned-naive, poisoned-robust, DP. ------
    specs = [
        ("clean_fedavg", "fedavg", False, False),
        ("poisoned_fedavg", "fedavg", True, False),
        ("poisoned_krum", "krum", True, False),
        ("poisoned_median", "median", True, False),
        ("poisoned_trimmed_mean", "trimmed_mean", True, False),
        ("dp_fedavg_clean", "fedavg", False, True),
    ]
    outcomes: dict[str, FederationOutcome] = {
        name: _run_federation(
            site_shapes,
            gains,
            true_profile,
            method=method,
            poisoned=poisoned,
            rounds=rounds,
            n_malicious=n_malicious,
            dp=dp,
            seed=11,
        )
        for name, method, poisoned, dp in specs
    }

    # --- Trust chain: gate the selected subbands + prove tamper-evidence. -----
    ap = audit_path or Path("benchmarks/results/federated_dsa_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)
    gate_records = [
        _shield_gate(
            outcomes["poisoned_krum"].default_subband,
            store,
            decision_id="fed-robust-selected",
            label="robust-selected-subband",
        ),
        _shield_gate(
            outcomes["poisoned_fedavg"].default_subband,
            store,
            decision_id="fed-naive-selected",
            label="naive-selected-subband",
        ),
        # An out-of-band, over-EIRP AI proposal the Shield must neutralise by
        # projection (3.62 GHz -> in band; EIRP 46 -> 33) before it can emit.
        _shield_gate(
            global_best,
            store,
            decision_id="fed-unsafe-oob",
            label="unsafe-oob-proposal",
            tx_power_dbm=40.0,
            frequency_hz=3.62e9,
            bandwidth_hz=20e6,
        ),
    ]
    unsafe = gate_records[2]
    chain_len = len(store)
    verify_intact = store.verify()  # -1 == intact

    result: dict[str, Any] = {
        "benchmark": "Federated DSA loop on real DeepMIMO channels",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get("data_kind", "site-specific Wireless InSite ray tracing"),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "federation": {
            "n_sites": n_sites,
            "n_malicious": n_malicious,
            "rounds": rounds,
            "local_epochs": LOCAL_EPOCHS,
            "local_lr": LOCAL_LR,
            "poison": "model-replacement / scaling (Bagdasaryan 2020)",
            "poison_boost": POISON_BOOST,
            "dp_noise_multiplier": DP_NOISE,
            "dp_delta": DP_DELTA,
        },
        "oracle_best_fixed_subband": global_best,
        "true_profile_mu_star": [round(float(x), 4) for x in true_profile],
        "true_profile_argmax": int(np.argmax(true_profile)),
        "avg_subband_gain_spread_db": round(
            float(gains.mean(axis=0).max() - gains.mean(axis=0).min()), 4
        ),
        "personalization_gain_db": personalization_gain_db,
        "site_heterogeneity": site_profiles,
        "outcomes": {k: asdict(v) for k, v in outcomes.items()},
        "trust_chain": {
            "gate_records": gate_records,
            "unsafe_oob_corrected": bool(unsafe["corrected"]),
            "unsafe_oob_safe_frequency_ghz": unsafe["safe_frequency_ghz"],
            "unsafe_oob_safe_eirp_dbm": unsafe["safe_eirp_dbm"],
            "unsafe_oob_reached_ran": bool(unsafe["emitted_clean"]),
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
        },
        "scope_note": (
            "Real DeepMIMO ray tracing drives real federated SGD, robust "
            "aggregation, DP accounting, Shield legality and the evidence chain. "
            "This is site-specific ray tracing, not OTA capture; one scenario. "
            "The global-default subband signal is small (~0.3 dB) because the "
            "campus channel is near frequency-flat on average; the security "
            "mechanism (poison inversion vs robust alignment) is the strong, "
            "unambiguous result on this data."
        ),
    }

    # --- Tamper-evidence proof on the freshly written chain. ------------------
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace('"subband": ', '"subband": 99, "_t": ', 1)
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["trust_chain"]["verify_after_tamper_index"] = JsonlEvidenceStore(ap).verify()

    return result


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
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/federated_dsa.json"))
    parser.add_argument("--sites", type=int, default=N_SITES)
    parser.add_argument("--malicious", type=int, default=N_MALICIOUS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    result = run(
        args.features,
        args.manifest,
        n_sites=args.sites,
        n_malicious=args.malicious,
        rounds=args.rounds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    o = result["outcomes"]
    tc = result["trust_chain"]
    # The poison must invert + diverge plain FedAvg while Krum stays aligned to
    # the true profile; the trust chain must hold and catch the tamper.
    poison_shown = (
        o["poisoned_fedavg"]["cos_to_true_profile"] < 0.0 and o["poisoned_fedavg"]["diverged"]
    )
    defense_shown = (
        o["poisoned_krum"]["cos_to_true_profile"] > 0.0 and not o["poisoned_krum"]["diverged"]
    )
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and not tc["unsafe_oob_reached_ran"]
        and tc.get("verify_after_tamper_index") not in (None, -1)
    )
    return 0 if (poison_shown and defense_shown and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
