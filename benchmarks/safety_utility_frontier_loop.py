#!/usr/bin/env python3
"""The safety-utility frontier on **real DeepMIMO** channels: what the Shield costs.

Earlier Horizon benchmarks only exercised the Decision Safety Shield on tasks
whose optimum was already *inside* the feasible set — DSA subband selection and
the coverage power-fill — where the projection is provably free (the committed
``benchmarks/results/deepmimo_dsa.json`` shows ``illegal_emits_after_shield=0``
and ``mean_regret_db=0.0`` for the best-subband strategy). This experiment
measures the other regime: a task whose unconstrained optimum lies *outside*
the feasible set, so the EIRP cap genuinely binds and safety has a price.

The loop, all on measured ray-traced physics (DeepMIMO ASU 3.5 GHz, 4096 Rx):

    real per-receiver, per-subband channel gains (Wireless InSite ray tracing)
        -> served-fraction objective, MONOTONE INCREASING in EIRP
        -> sweep the EIRP cap (20 .. 52 dBm); per cap, an optimiser proposes the
           UNCONSTRAINED optimum (full service, demand EIRP ~109-111 dBm)
        -> Decision Safety Shield projects every proposal onto that cap
        -> environment executes the projected action, realises its REAL reward
        -> hash-chained, tamper-evident evidence record for every step

Definitions, stated precisely so the cost cannot be softened:

* ``unconstrained_optimum_reward`` — the true argmax of the served-fraction
  objective with no cap: serve every measured receiver (reward 1.0). The
  minimum EIRP that achieves it on this campus is ~109.3 dBm, measured from
  the real channel gains — physically enormous, which is itself part of the
  honest picture (the last receivers are ~110 dB below the strongest).
* ``best_feasible_reward(cap)`` — the best served fraction achievable by any
  legal action under that cap (best subband at the cap EIRP).
* ``utility_forgone(cap) = unconstrained_optimum_reward - best_feasible_reward``
  — the served fraction the licence condition costs at that cap.
* ``marginal price of safety`` — served fraction lost per dB of cap tightening
  between adjacent sweep points; the headline curve.

The contrast case runs a task whose optimum IS feasible (pick the best subband
at a fixed legal EIRP of 32 dBm): the Shield applies no correction and costs
exactly zero utility there. Together the two regimes give the honest boundary:
**the Shield is free when the optimum is feasible, and costs a measurable,
reported amount when the constraint binds.** Whether that cost is acceptable is
a regulatory question — the EIRP cap is a licence condition, not a tunable the
rApp may raise — Horizon's claim is that the cost is explicit, bounded and
auditable, not that it is zero.

Pure numpy + the shipped ``horizon_ric`` modules. The real DeepMIMO feature
file is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` (deterministic, checksum-pinned) or
point ``--features`` at a cached copy. The committed result JSON is the real
4096-Rx run. Deterministic: fixed seeds, no wall-clock inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.learning import (
    EvidenceIntegrityError,
    ShieldedSpectrumEnv,
    SpectrumAction,
    StepResult,
    load_transitions,
)
from horizon_ric.learning.shield_env import (
    ANTENNA_GAIN_DBI,
    BAND_HI_HZ,
    BAND_LO_HZ,
    N_SUBBANDS,
    load_gain_matrix,
    subband_center_hz,
)

DEFAULT_CAPS_DBM = (20.0, 26.0, 33.0, 40.0, 46.0, 52.0)
OPERATIONAL_CAP_DBM = 33.0  # the licence condition every other Horizon benchmark runs under.
STEPS_PER_CAP = 8
FREE_CASE_STEPS = 6
FREE_CASE_TX_POWER_DBM = 26.0  # EIRP 32 dBm < 33: the task optimum is feasible.
DEMAND_JITTER_DB = 3.0  # seeded jitter ABOVE the full-service demand; never below any cap.
SEED = 20260727
DSA_CROSS_EVIDENCE = Path(__file__).resolve().parent / "results" / "deepmimo_dsa.json"


def _best_feasible(env: ShieldedSpectrumEnv) -> tuple[int, float]:
    """Best legal action under ``env``'s cap: (subband, served fraction at the cap)."""
    best_subband, best_reward = 0, -1.0
    for b in range(N_SUBBANDS):
        value = env.reward(
            SpectrumAction(subband_center_hz(b), env.max_eirp_dbm - ANTENNA_GAIN_DBI)
        )
        if value is not None and value > best_reward:
            best_subband, best_reward = b, value
    return best_subband, best_reward


def _reference_anchors(
    env: ShieldedSpectrumEnv, best_feasible_at_operational_cap: float
) -> list[dict[str, Any]]:
    """Served fraction and forgone utility at physically labelled EIRP anchors.

    The served-fraction objective saturates, so "how much does the cap cost?"
    has no single answer — it depends on the baseline. Reporting the curve makes
    the anchor explicit instead of letting one number stand in for all of them.
    """
    anchors = [
        (46.0, "high-power small cell (~40 W EIRP)"),
        (52.0, "sweep top; the credit_assignment action-grid top (~158 W EIRP)"),
        (60.0, "small macro (~1 kW EIRP)"),
        (78.0, "upper end of a deployable macro (~63 kW EIRP)"),
    ]
    rows: list[dict[str, Any]] = []
    for eirp, label in anchors:
        best = max(
            env.reward(SpectrumAction(subband_center_hz(b), eirp - ANTENNA_GAIN_DBI)) or 0.0
            for b in range(N_SUBBANDS)
        )
        rows.append(
            {
                "anchor_eirp_dbm": eirp,
                "anchor_watts": round(10 ** ((eirp - 30.0) / 10.0), 1),
                "label": label,
                "best_reward_at_anchor": round(float(best), 6),
                "utility_forgone_vs_operational_cap": round(
                    float(best) - best_feasible_at_operational_cap, 6
                ),
            }
        )
    return rows


def _full_service_eirp_dbm(env: ShieldedSpectrumEnv, gains: np.ndarray, subband: int) -> float:
    """Minimum EIRP at which every measured receiver on ``subband`` clears the threshold.

    Measured, not assumed: EIRP >= noise + threshold - min(gain). This is the
    smallest demand at which the unconstrained served-fraction optimum (1.0)
    is attained, i.e. the true unconstrained argmax of the objective.
    """
    return float(env.noise_dbm + env.sinr_threshold_db - gains[:, subband].min())


def _sweep_one_cap(
    gains: np.ndarray,
    cap_dbm: float,
    *,
    store: JsonlEvidenceStore,
    rng: np.random.Generator,
    unconstrained_reward: float,
) -> dict[str, Any]:
    """Closed loop at one cap: the optimiser demands full service, the Shield projects.

    On the two edge subbands the Shield also nudges the centre frequency by
    ~1.7 MHz so the 20 MHz reservation mask stays inside the band; the nudged
    centre still maps to the same measured subband, so the realised reward is
    the best feasible reward and the reported gap is purely the EIRP cap's.
    """
    env = ShieldedSpectrumEnv(gains, max_eirp_dbm=cap_dbm)
    subband, best_feasible = _best_feasible(env)
    # Propose on the best feasible subband so the realised gap isolates the EIRP
    # cap's price (not a subband-choice confound); demand that subband's own
    # measured full-service EIRP, so the proposal really is the unconstrained
    # optimum (counterfactual served fraction 1.0) and is above every swept cap.
    demand_eirp = _full_service_eirp_dbm(env, gains, subband)
    frequency = subband_center_hz(subband)

    results: list[StepResult] = []
    for step in range(STEPS_PER_CAP):
        jitter = float(rng.uniform(0.0, DEMAND_JITTER_DB))
        proposed = SpectrumAction(frequency, demand_eirp - ANTENNA_GAIN_DBI + jitter)
        results.append(
            env.step(
                proposed,
                decision_id=f"frontier-cap{cap_dbm:g}-step{step}",
                store=store,
                rapp_instance_id="safety-utility-frontier",
                policy_label=f"unconstrained-optimum-cap{cap_dbm:g}",
            )
        )

    realised = [r.reward for r in results]
    counterfactuals = [r.counterfactual_reward for r in results]
    if any(c is None for c in counterfactuals):
        raise RuntimeError("in-band unconstrained proposal lost its counterfactual reward")
    return {
        "eirp_cap_dbm": cap_dbm,
        "steps": len(results),
        "best_feasible_subband": subband,
        "best_feasible_reward": round(best_feasible, 6),
        "realised_mean_reward": round(float(np.mean(realised)), 6),
        "counterfactual_mean_reward": round(float(np.mean([float(c) for c in counterfactuals])), 6),
        "utility_forgone": round(unconstrained_reward - best_feasible, 6),
        "projection_rate": round(float(np.mean([r.projected for r in results])), 6),
        "illegal_proposal_rate": round(
            float(np.mean([r.illegal_without_shield for r in results])), 6
        ),
        "mean_proposed_eirp_dbm": round(float(np.mean([r.proposed.eirp_dbm for r in results])), 4),
        "executed_eirp_dbm": round(float(np.mean([r.executed.eirp_dbm for r in results])), 4),
        "max_executed_eirp_dbm": round(float(max(r.executed.eirp_dbm for r in results)), 4),
        "executed_frequency_hz": float(results[0].executed.frequency_hz),
    }


def _free_constraint_case(gains: np.ndarray, store: JsonlEvidenceStore) -> dict[str, Any]:
    """The contrast: best subband at a fixed legal EIRP — the optimum is feasible.

    The task selects among *fully legal* actions: EIRP 32 dBm (< the 33 dBm
    cap) on a subband whose whole 20 MHz reservation mask fits inside the
    licensed band — which excludes the two edge subbands, whose centres sit
    closer than 10 MHz to a band edge and would themselves be frequency-
    projected. Because the task optimum is inside the feasible set the Shield
    has nothing to project: no correction, no refusal, zero utility cost. The
    committed DSA benchmark is the standing 4096-decision evidence for the
    same conclusion and is cited verbatim in the returned record.
    """
    env = ShieldedSpectrumEnv(gains, max_eirp_dbm=OPERATIONAL_CAP_DBM)
    fixed_eirp = FREE_CASE_TX_POWER_DBM + ANTENNA_GAIN_DBI
    half_bw = env.bandwidth_hz / 2.0
    per_subband = {
        b: float(env.reward(SpectrumAction(subband_center_hz(b), FREE_CASE_TX_POWER_DBM)) or 0.0)
        for b in range(N_SUBBANDS)
    }
    candidates = [
        b
        for b in range(N_SUBBANDS)
        if subband_center_hz(b) - half_bw >= BAND_LO_HZ
        and subband_center_hz(b) + half_bw <= BAND_HI_HZ
    ]
    best_subband = max(candidates, key=lambda b: per_subband[b])
    task_optimum = per_subband[best_subband]

    results: list[StepResult] = []
    for step in range(FREE_CASE_STEPS):
        results.append(
            env.step(
                SpectrumAction(subband_center_hz(best_subband), FREE_CASE_TX_POWER_DBM),
                decision_id=f"frontier-free-step{step}",
                store=store,
                rapp_instance_id="safety-utility-frontier",
                policy_label="best-subband-fixed-legal-eirp",
            )
        )
    realised_mean = float(np.mean([r.reward for r in results]))
    corrected = any(
        r.projected
        or abs(r.executed.frequency_hz - r.proposed.frequency_hz) > 1.0
        or abs(r.executed.eirp_dbm - r.proposed.eirp_dbm) > 1e-9
        for r in results
    )

    cross: dict[str, Any] = {"path": "benchmarks/results/deepmimo_dsa.json"}
    if DSA_CROSS_EVIDENCE.exists():
        dsa = json.loads(DSA_CROSS_EVIDENCE.read_text(encoding="utf-8"))
        best = dsa.get("strategies", {}).get("best_subband", {})
        cross.update(
            strategy="best_subband",
            decisions=best.get("decisions"),
            illegal_emits_after_shield=best.get("illegal_emits_after_shield"),
            mean_regret_db=best.get("mean_regret_db"),
        )
    else:  # pragma: no cover - the committed evidence ships with the repo.
        cross.update(strategy="best_subband", decisions=None,
                     illegal_emits_after_shield=None, mean_regret_db=None)

    return {
        "task": (
            "pick the best fully-legal subband (20 MHz mask in band) at a fixed legal "
            "EIRP of 32 dBm (tx 26 dBm + 6 dBi < 33 dBm cap): the task optimum is feasible"
        ),
        "fixed_eirp_dbm": fixed_eirp,
        "steps": len(results),
        "candidate_subbands": candidates,
        "best_subband": best_subband,
        "per_subband_reward": [round(per_subband[b], 6) for b in range(N_SUBBANDS)],
        "task_optimum_reward": round(task_optimum, 6),
        "realised_mean_reward": round(realised_mean, 6),
        "projection_rate": round(float(np.mean([r.projected for r in results])), 6),
        "guard_refused_rate": round(float(np.mean([r.guard_refused for r in results])), 6),
        "corrected": bool(corrected),
        "utility_cost": round(task_optimum - realised_mean, 6),
        "max_executed_eirp_dbm": round(float(max(r.executed.eirp_dbm for r in results)), 4),
        "executed_frequency_hz": float(results[0].executed.frequency_hz),
        "committed_cross_evidence": cross,
    }


def run(
    features: Path,
    manifest_path: Path,
    *,
    caps_dbm: tuple[float, ...] = DEFAULT_CAPS_DBM,
    audit_path: Path | None = None,
    sample_record_path: Path | None = None,
) -> dict[str, Any]:
    gains = load_gain_matrix(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(gains) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")
    caps = tuple(sorted(float(c) for c in caps_dbm))
    if len(caps) < 2:
        raise ValueError("need at least two caps to measure a marginal price")

    rng = np.random.default_rng(SEED)
    env_ref = ShieldedSpectrumEnv(gains, max_eirp_dbm=OPERATIONAL_CAP_DBM)

    # The true unconstrained optimum of the served-fraction objective: full
    # service, reward 1.0, attained at the cheapest measured full-service EIRP.
    fs_per_subband = [
        _full_service_eirp_dbm(env_ref, gains, b) for b in range(N_SUBBANDS)
    ]
    demand_subband = int(np.argmin(fs_per_subband))
    unconstrained_reward = 1.0
    op_subband, op_best_feasible = _best_feasible(env_ref)
    eirp50_reference = env_ref.reward(
        SpectrumAction(subband_center_hz(op_subband), 50.0 - ANTENNA_GAIN_DBI)
    )

    # --- The sweep, all steps on ONE hash-chained evidence store. -------------
    ap = audit_path or Path("benchmarks/results/safety_utility_frontier_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)

    frontier = [
        _sweep_one_cap(
            gains, cap, store=store, rng=rng, unconstrained_reward=unconstrained_reward
        )
        for cap in caps
    ]
    free_case = _free_constraint_case(gains, store)

    marginal = []
    for lo, hi in zip(frontier, frontier[1:]):
        width_db = hi["eirp_cap_dbm"] - lo["eirp_cap_dbm"]
        marginal.append(
            {
                "from_cap": lo["eirp_cap_dbm"],
                "to_cap": hi["eirp_cap_dbm"],
                "served_fraction_lost_per_db": round(
                    (hi["best_feasible_reward"] - lo["best_feasible_reward"]) / width_db, 6
                ),
            }
        )

    # --- Trust chain: verify intact, replay, then prove tamper-evidence. ------
    chain_len = len(store)
    verify_intact = store.verify()
    replayed_intact = len(load_transitions(ap)) if verify_intact == -1 else 0

    if sample_record_path is not None:
        first_line = ap.read_text(encoding="utf-8").splitlines()[0]
        sample_record_path.parent.mkdir(parents=True, exist_ok=True)
        sample_record_path.write_text(
            json.dumps(json.loads(first_line), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    lines = ap.read_text(encoding="utf-8").splitlines()
    verify_after_tamper: int | None = None
    tampered_refused = False
    if len(lines) >= 2:
        lines[1] = lines[1].replace(
            '"requested_frequency_hz": ', '"requested_frequency_hz": 999999, "_t": ', 1
        )
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        verify_after_tamper = JsonlEvidenceStore(ap).verify()
        try:
            load_transitions(ap)
        except EvidenceIntegrityError:
            tampered_refused = True

    op_entry = next(
        (f for f in frontier if abs(f["eirp_cap_dbm"] - OPERATIONAL_CAP_DBM) < 1e-9), None
    )
    utility_forgone_op = (
        op_entry["utility_forgone"]
        if op_entry is not None
        else round(unconstrained_reward - op_best_feasible, 6)
    )

    return {
        "benchmark": "Safety-utility frontier of the Decision Safety Shield on real DeepMIMO",
        "task": "EIRP-capped spectrum reservation; served-fraction objective, monotone in EIRP",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get("data_kind", "site-specific Wireless InSite ray tracing"),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": int(len(gains)),
        "env": {
            "noise_dbm": round(float(env_ref.noise_dbm), 4),
            "bandwidth_hz": float(env_ref.bandwidth_hz),
            "sinr_threshold_db": float(env_ref.sinr_threshold_db),
            "band_lo_hz": float(BAND_LO_HZ),
            "band_hi_hz": float(BAND_HI_HZ),
            "n_subbands": int(N_SUBBANDS),
            "antenna_gain_dbi": float(ANTENNA_GAIN_DBI),
            "reward": "served fraction (receivers with EIRP + measured_gain - noise >= 0 dB)",
        },
        "unconstrained_optimum": {
            "definition": (
                "true argmax of the served-fraction objective with no cap: serve every "
                "measured receiver (reward 1.0), attained at the minimum full-service EIRP"
            ),
            "reward": unconstrained_reward,
            "demand_eirp_dbm": round(fs_per_subband[demand_subband], 4),
            "demand_subband": demand_subband,
            "full_service_eirp_dbm_per_subband": [round(v, 4) for v in fs_per_subband],
            "eirp50_reference_reward": round(float(eirp50_reference), 6),
        },
        "frontier": frontier,
        "marginal_price_per_db": marginal,
        "free_constraint_case": free_case,
        "operational_cap_dbm": OPERATIONAL_CAP_DBM,
        "utility_forgone_at_operational_cap": utility_forgone_op,
        # The forgone utility depends entirely on WHICH baseline the cap is
        # measured against, and the served fraction saturates, so a single
        # headline number is meaningless without its anchor. These reference
        # points make the anchor explicit and reconcile this experiment with
        # benchmarks/results/credit_assignment.json, whose safety_utility_gap
        # is measured against its own 52 dBm action-grid top, not full service.
        "reference_anchors": _reference_anchors(env_ref, op_best_feasible),
        "anchor_note": (
            "utility_forgone_at_operational_cap is measured against full service "
            "(reward 1.0), which requires ~109 dBm EIRP (~85 MW) and is a physical "
            "idealisation, not a deployable alternative. Against a deployable "
            "anchor the cost is far smaller: see reference_anchors. The "
            "credit_assignment experiment's safety_utility_gap uses its 52 dBm "
            "grid top and is the same quantity under that anchor."
        ),
        "trust_chain": {
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
            "replayed_transitions_intact": replayed_intact,
            "tampered_chain_refused": tampered_refused,
            "verify_after_tamper_index": verify_after_tamper,
        },
        "scope_note": (
            "Real site-specific ray tracing (DeepMIMO ASU 3.5 GHz), not OTA capture; one "
            "scenario; served fraction is one utility proxy among several a licensee may "
            "care about. The frontier states the Shield's real cost when the constraint "
            "binds: utility forgone at the operational 33 dBm cap is reported plainly, not "
            "netted against the free-constraint case. The EIRP cap is a regulatory licence "
            "condition, not a free parameter — Horizon's claim is that the cost of "
            "enforcing it is explicit, bounded and auditable, not that it is zero."
        ),
    }


def _invariants_hold(result: dict[str, Any]) -> bool:
    """The exit-0 contract; mirrors the committed-result tests and the verifier."""
    frontier = result["frontier"]
    best = [f["best_feasible_reward"] for f in frontier]
    # A looser cap never hurts: best feasible served fraction is non-decreasing.
    monotone = all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:]))
    # The constraint genuinely binds at the operational cap.
    binds = result["utility_forgone_at_operational_cap"] > 0.0
    # Tightening the cap loses served fraction somewhere in the sweep.
    marginal_positive = any(
        m["served_fraction_lost_per_db"] > 0.0 for m in result["marginal_price_per_db"]
    )
    free = result["free_constraint_case"]
    free_is_free = (
        free["projection_rate"] == 0.0
        and free["corrected"] is False
        and abs(free["utility_cost"]) < 1e-12
        and free["max_executed_eirp_dbm"] <= result["operational_cap_dbm"] + 1e-9
    )
    legal = all(
        f["max_executed_eirp_dbm"] <= f["eirp_cap_dbm"] + 1e-9
        and BAND_LO_HZ <= f["executed_frequency_hz"] <= BAND_HI_HZ
        for f in frontier
    ) and BAND_LO_HZ <= free["executed_frequency_hz"] <= BAND_HI_HZ
    tc = result["trust_chain"]
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and tc["tampered_chain_refused"] is True
        and tc["verify_after_tamper_index"] not in (None, -1)
    )
    return monotone and binds and marginal_positive and free_is_free and legal and chain_ok


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
        "--out", type=Path, default=Path("benchmarks/results/safety_utility_frontier.json")
    )
    parser.add_argument(
        "--caps",
        type=str,
        default=",".join(f"{c:g}" for c in DEFAULT_CAPS_DBM),
        help="comma-separated EIRP caps (dBm) to sweep",
    )
    args = parser.parse_args()
    caps = tuple(float(c) for c in args.caps.split(",") if c.strip())

    result = run(
        args.features,
        args.manifest,
        caps_dbm=caps,
        sample_record_path=args.out.with_name(args.out.stem + "_sample_record.json"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if _invariants_hold(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
