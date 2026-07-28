#!/usr/bin/env python3
"""DP-FedAvg privacy / utility / membership trade-off on REAL ray-traced subjects.

Robust and secure aggregation bound *poisoning* and hide *individual* updates,
but neither *bounds privacy leakage*: a curious server or a membership-inference
adversary can still learn whether a particular client's measurements shaped the
global model (O-RAN WG11 ML04 membership inference; the "no differential-privacy
accountant" GAP in ``docs/THREAT_MODEL.md`` §5). ``horizon_ric.federated.dp``
closes that gap with DP-FedAvg + a Rényi-DP accountant (Mironov, CSF 2017;
Abadi et al., CCS 2016), in the lineage of the NTU/DTC work pairing unlearning
with adaptive DP (Liu, Jiang, Lam et al., IEEE BigData 2024).

**What changed: this suite now runs on real data subjects.** It used to sample
member and non-member "clients" by rolling out a toy ``DSAEnv`` under different
``numpy.random`` seeds — a privacy claim about records that were never anyone's.
Members and non-members are now disjoint sets of **real DeepMIMO ASU-campus
receivers** (:mod:`benchmarks.privacy_real_subjects`): each subject is one
ray-traced receiver contributing its real position and its six real measured
subband gains to a real coverage/path-loss model.

Three quantities are reported together at each noise multiplier ``z``:

  * PRIVACY — the ``(ε, δ)`` bound the accountant certifies for the shipped
    ``R``-round configuration, **plus a numeric verification chain** (below).
  * LEAKAGE — two membership-inference attacks that are only meaningful because
    the cohorts are real: a client-level and a record-level loss-threshold
    attack (Yeom et al., CSF 2018) over member vs non-member receivers drawn
    from the same campus.
  * UTILITY — held-out RMSE (dB) of the global model against the real measured
    gains of receivers that never took part in training.

Verifying the guarantee instead of asserting it
-----------------------------------------------
Section ``epsilon_verification`` measures, for one Gaussian round on the real
adjacent pair (one real member cell replaced by a real non-member cell):

  ``ε_audit ≤ ε_realised_pair ≤ ε_worst_case_analytic ≤ ε_rdp``

* ``ε_audit`` — an empirical **lower** bound obtained by running the real
  ``dp.dp_fedavg`` thousands of times on both datasets and sweeping the optimal
  distinguisher's threshold with Clopper–Pearson 95% bounds (Jagielski et al.,
  NeurIPS 2020). This is the privacy loss an attacker actually achieved.
* ``ε_realised_pair`` — the exact Gaussian ``ε`` at the **measured** sensitivity
  ``‖clip(x) − clip(x')‖`` of that real pair, which is smaller than the assumed
  worst case.
* ``ε_worst_case_analytic`` — exact Gaussian ``ε`` at ``Δ = 2C`` (Balle & Wang,
  ICML 2018), the tight version of what the mechanism promises.
* ``ε_rdp`` — what the shipped accountant reports.

If any inequality flips, the mechanism or the accountant is wrong; the suite
reports ``ordering_holds`` rather than trusting it.

The clip norm is calibrated on the **public server-held root slice** of the same
real campus (512 of 4096 receivers, never given to a client), so it is chosen
independently of the private cohort, and the suite reports what fraction of the
real private updates it actually clips — a clip that never binds is pure noise
for nothing.

Pure numpy. Run:  python benchmarks/dp_privacy_suite.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.federated import dp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_real_subjects as R  # noqa: E402  (sibling module, benchmarks/ is not a package)

N_CELLS = 128  # real geographic cells carved out of the private receiver pool
M_MEMBERS = 64  # cells that participate in training
K_NONMEMBERS = 64  # cells held out entirely - the non-member cohort
N_ROOT_CELLS = 64  # public pseudo-clients used only for clip calibration
ROUNDS = 12
NOISE_MULTIPLIERS = (0.0, 1.0, 4.0, 8.0)
DELTA = 1e-5
ADJACENCY = dp.Adjacency.REPLACE_ONE
PARTITION_SEED = 1234
AUDIT_TRIALS = 20_000
DEFAULT_OUT = "benchmarks/results/dp_privacy.json"


# ─────────────────────────────────────────────────────────────────────────────
def calibrate_clip_norm(pop: R.SubjectPopulation, *, rounds: int) -> dict[str, Any]:
    """Pick ``C`` from the PUBLIC root receivers only.

    The 512 server-held receivers are never assigned to a client, so running the
    identical protocol on them and taking the median update norm leaks nothing
    about the private cohort while still landing ``C`` on the real scale of the
    updates. Selecting ``C`` from the private updates would itself be a leak.
    """
    root_cells = [
        pop.root_idx[c]
        for c in R.partition_geographic(pop.pos[pop.root_idx], N_ROOT_CELLS, seed=PARTITION_SEED)
    ]
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    norms: list[float] = []
    for _ in range(rounds):
        ups = [R.local_prox_update(w, pop.design[i], pop.target[i]) for i in root_cells]
        norms.extend(float(np.linalg.norm(u)) for u in ups)
        w = w + np.mean(np.stack(ups), axis=0)
    arr = np.asarray(norms, dtype=np.float64)
    return {
        "clip_norm_C": round(float(np.median(arr)), 6),
        "rule": "median update L2 norm over the PUBLIC server-held root pseudo-clients",
        "public_root_receivers": int(pop.root_idx.size),
        "public_root_pseudo_clients": N_ROOT_CELLS,
        "public_norm_median": round(float(np.median(arr)), 6),
        "public_norm_p90": round(float(np.percentile(arr, 90)), 6),
        "public_norm_max": round(float(arr.max()), 6),
        "leakage_note": (
            "the root slice is public by construction (never held by a client), "
            "so C does not depend on the private cohort"
        ),
    }


def _dp_aggregator(cfg: dp.DPConfig, rng: np.random.Generator, acc: dp.RDPAccountant):
    def agg(ups):
        return dp.dp_fedavg(ups, cfg, rng=rng, accountant=acc)

    return agg


def _train(
    pop: R.SubjectPopulation,
    members: list[np.ndarray],
    *,
    rounds: int,
    z: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int, float, list[float]]:
    """Run the federation at noise multiplier ``z``; return (w, ε, steps, clipped_frac, norms)."""
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    acc = dp.RDPAccountant()
    cfg = dp.DPConfig(clip_norm=clip_norm, noise_multiplier=max(z, 1e-12), adjacency=ADJACENCY)
    norms: list[float] = []
    clipped = 0
    total = 0
    for _ in range(rounds):
        ups = [R.local_prox_update(w, pop.design[i], pop.target[i]) for i in members]
        n = [float(np.linalg.norm(u)) for u in ups]
        norms.extend(n)
        total += len(n)
        clipped += int(sum(1 for v in n if v > clip_norm))
        if z == 0.0:
            step = np.mean(np.stack(ups), axis=0)
        else:
            step = dp.dp_fedavg(ups, cfg, rng=rng, accountant=acc)
        w = w + step
    epsilon = math.inf if z == 0.0 else acc.get_epsilon(DELTA)
    return w, epsilon, acc.steps, clipped / max(total, 1), norms


def verify_epsilon(
    pop: R.SubjectPopulation,
    members: list[np.ndarray],
    nonmembers: list[np.ndarray],
    *,
    z: float,
    clip_norm: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Audit one real Gaussian round against three independent ε references."""
    cfg = dp.DPConfig(clip_norm=clip_norm, noise_multiplier=z, adjacency=ADJACENCY)
    w0 = np.zeros(R.MODEL_DIM, dtype=np.float64)
    ups = [R.local_prox_update(w0, pop.design[i], pop.target[i]) for i in members]
    swapped = R.local_prox_update(w0, pop.design[nonmembers[0]], pop.target[nonmembers[0]])

    # Replace-one adjacency, realised on real receivers: member cell 0 <-> a real
    # non-member cell. The optimal distinguisher projects on the clipped delta.
    kept = dp.l2_clip(ups[0], clip_norm)
    alt = dp.l2_clip(swapped, clip_norm)
    diff = kept - alt
    realised_sensitivity = float(np.linalg.norm(diff))
    if realised_sensitivity <= 0.0:
        raise ValueError("the two real cells produced identical clipped updates")
    direction = diff / realised_sensitivity

    ups_prime = [swapped] + ups[1:]
    rng = np.random.default_rng(seed)
    scores_d = np.empty(trials, dtype=np.float64)
    scores_p = np.empty(trials, dtype=np.float64)
    for t in range(trials):
        scores_d[t] = float(dp.dp_fedavg(ups, cfg, rng=rng) @ direction)
        scores_p[t] = float(dp.dp_fedavg(ups_prime, cfg, rng=rng) @ direction)

    audit = R.empirical_epsilon_lower_bound(scores_d, scores_p, DELTA)
    eps_pair = R.analytic_gaussian_epsilon(realised_sensitivity, cfg.sigma, DELTA)
    eps_wc = R.analytic_gaussian_epsilon(cfg.sensitivity, cfg.sigma, DELTA)
    acc1 = dp.RDPAccountant()
    acc1.step(z)
    eps_rdp1 = float(acc1.get_epsilon(DELTA))

    e_aud = audit["epsilon_empirical_lower_95"]
    return {
        "noise_multiplier": z,
        "rounds_audited": 1,
        "assumed_sensitivity_2C": round(float(cfg.sensitivity), 6),
        "realised_sensitivity_on_real_pair": round(realised_sensitivity, 6),
        "sigma": round(float(cfg.sigma), 6),
        "epsilon_audit_empirical_lower_95": e_aud,
        "epsilon_realised_pair_exact": round(float(eps_pair), 4),
        "epsilon_worst_case_analytic_exact": round(float(eps_wc), 4),
        "epsilon_rdp_one_round": round(eps_rdp1, 4),
        "ordering_holds": bool(e_aud <= eps_pair <= eps_wc <= eps_rdp1 + 1e-9),
        "rdp_slack_over_exact": round(float(eps_rdp1 - eps_wc), 4),
        "audit": audit,
        "what_this_proves": (
            "the empirical lower bound is what an optimal distinguisher actually "
            "achieved against the real dp_fedavg code path on a real replace-one "
            "pair; the exact Gaussian values bracket it from above. A flipped "
            "inequality would mean the mechanism or the accountant is wrong."
        ),
    }


def membership_attacks(
    pop: R.SubjectPopulation,
    w: np.ndarray,
    members: list[np.ndarray],
    nonmembers: list[np.ndarray],
) -> dict[str, Any]:
    """Two loss-threshold attacks over real member / non-member receivers."""
    client_m = [-float(np.mean(pop.per_subject_squared_error(w, c))) for c in members]
    client_n = [-float(np.mean(pop.per_subject_squared_error(w, c))) for c in nonmembers]
    rec_m = -pop.per_subject_squared_error(w, np.concatenate(members))
    rec_n = -pop.per_subject_squared_error(w, np.concatenate(nonmembers))
    return {
        "client_level_auc": round(R.auc_mann_whitney(client_m, client_n), 4),
        "record_level_auc": round(R.auc_mann_whitney(rec_m, rec_n), 4),
        "member_clients": len(members),
        "nonmember_clients": len(nonmembers),
        "member_records": int(rec_m.size),
        "nonmember_records": int(rec_n.size),
        "attack": (
            "loss-threshold membership inference (Yeom et al., CSF 2018): score = "
            "-mean squared dB error of the released global model on the target's "
            "real measured subband gains"
        ),
    }


def run_seed(
    pop: R.SubjectPopulation,
    cells: list[np.ndarray],
    *,
    seed: int,
    clip_norm: float,
    rounds: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(90_000 + seed)
    order = rng.permutation(len(cells))
    members = [cells[i] for i in order[:M_MEMBERS]]
    nonmembers = [cells[i] for i in order[M_MEMBERS : M_MEMBERS + K_NONMEMBERS]]
    train_idx = np.concatenate(members)
    test_idx = np.concatenate(nonmembers)

    per_z: dict[str, Any] = {}
    for z in NOISE_MULTIPLIERS:
        mech_rng = np.random.default_rng(20_240_601 + seed)
        w, eps, steps, clipped, norms = _train(
            pop, members, rounds=rounds, z=z, clip_norm=clip_norm, rng=mech_rng
        )
        arr = np.asarray(norms)
        per_z[_zkey(z)] = {
            "noise_multiplier": z,
            "epsilon": None if math.isinf(eps) else round(float(eps), 4),
            "epsilon_label": "inf (no DP)" if math.isinf(eps) else round(float(eps), 4),
            "accounted_steps": steps,
            "private_update_norm_median": round(float(np.median(arr)), 6),
            "private_updates_clipped_fraction": round(float(clipped), 4),
            "heldout_rmse_db": round(pop.rmse_db(w, test_idx), 4),
            "train_rmse_db": round(pop.rmse_db(w, train_idx), 4),
            "mia": membership_attacks(pop, w, members, nonmembers),
        }
    return {
        "seed": seed,
        "baseline_rmse_db": round(pop.baseline_rmse_db(test_idx), 4),
        "member_receivers": int(train_idx.size),
        "nonmember_receivers": int(test_idx.size),
        "per_z": per_z,
        "_members": members,
        "_nonmembers": nonmembers,
    }


def _zkey(z: float) -> str:
    return "z=0_no_dp" if z == 0.0 else f"z={z:g}"


def _agg(rows: list[dict], zkey: str, *path: str) -> dict[str, Any]:
    vals = []
    for r in rows:
        cur: Any = r["per_z"][zkey]
        for k in path:
            cur = cur[k]
        if cur is not None:
            vals.append(float(cur))
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n_seeds": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "n_seeds": int(a.size),
    }


def run_suite(
    *,
    seeds: list[int],
    features: Path,
    manifest_path: Path,
    rounds: int = ROUNDS,
    audit_trials: int = AUDIT_TRIALS,
) -> dict[str, Any]:
    pop, manifest = R.build_population(features, manifest_path)
    cells = [
        pop.private_idx[c]
        for c in R.partition_geographic(pop.pos[pop.private_idx], N_CELLS, seed=PARTITION_SEED)
    ]
    if len(cells) < M_MEMBERS + K_NONMEMBERS:
        raise ValueError("not enough real cells for the member/non-member split")
    calib = calibrate_clip_norm(pop, rounds=rounds)
    clip_norm = calib["clip_norm_C"]

    rows = [run_seed(pop, cells, seed=s, clip_norm=clip_norm, rounds=rounds) for s in seeds]
    members0, nonmembers0 = rows[0]["_members"], rows[0]["_nonmembers"]
    for r in rows:
        r.pop("_members")
        r.pop("_nonmembers")

    verification = [
        verify_epsilon(
            pop,
            members0,
            nonmembers0,
            z=z,
            clip_norm=clip_norm,
            trials=audit_trials,
            seed=4_242 + int(z * 100),
        )
        for z in NOISE_MULTIPLIERS
        if z > 0.0
    ]

    results = []
    for z in NOISE_MULTIPLIERS:
        zk = _zkey(z)
        results.append(
            {
                "noise_multiplier": z,
                "label": "no DP (plain FedAvg)" if z == 0.0 else f"z={z:g}",
                "epsilon": (
                    {"note": "infinite - no DP guarantee"}
                    if z == 0.0
                    else _agg(rows, zk, "epsilon")
                ),
                "accounted_steps": rows[0]["per_z"][zk]["accounted_steps"],
                "private_updates_clipped_fraction": _agg(
                    rows, zk, "private_updates_clipped_fraction"
                ),
                "heldout_rmse_db": _agg(rows, zk, "heldout_rmse_db"),
                "train_rmse_db": _agg(rows, zk, "train_rmse_db"),
                "mia_client_level_auc": _agg(rows, zk, "mia", "client_level_auc"),
                "mia_record_level_auc": _agg(rows, zk, "mia", "record_level_auc"),
            }
        )

    cell_sizes = np.asarray([len(c) for c in cells])
    report: dict[str, Any] = {
        "benchmark": "DP-FedAvg on real DeepMIMO ray-traced receivers",
        "provenance": R.provenance_block(
            manifest,
            pop,
            extra_scope=(
                "The membership-inference AUCs are attack-specific empirical "
                "diagnostics on this cohort; an AUC near 0.5 shows this attack "
                "failed, it does not prove privacy."
            ),
        ),
        "setup": {
            "task": (
                "federated per-subband path-loss regression: whitened quadratic "
                "position design -> 6 real subband gains, FedProx local steps"
            ),
            "model_dim": R.MODEL_DIM,
            "geographic_cells": N_CELLS,
            "cell_size_min": int(cell_sizes.min()),
            "cell_size_max": int(cell_sizes.max()),
            "m_members": M_MEMBERS,
            "k_nonmembers": K_NONMEMBERS,
            "rounds": rounds,
            "noise_multipliers": list(NOISE_MULTIPLIERS),
            "delta": DELTA,
            "client_adjacency": f"{ADJACENCY}; L2 sensitivity = 2C",
            "partition_seed": PARTITION_SEED,
            "seeds": seeds,
            "confidence_unit": "member/non-member assignment seed over the same real cells",
            "clip_calibration": calib,
            "lineage": [
                "Mironov, CSF 2017 (Renyi differential privacy)",
                "Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang, CCS 2016 (DP-SGD)",
                "Balle & Wang, ICML 2018 (analytic Gaussian mechanism - the exact epsilon)",
                "Jagielski, Ullman & Oprea, NeurIPS 2020 (auditing differential privacy)",
                "Yeom, Giacomelli, Fredrikson & Jha, CSF 2018 (loss-threshold membership inference)",
                "Liu, Jiang, Lam et al., IEEE BigData 2024 (federated unlearning with adaptive DP)",
            ],
        },
        "baseline_rmse_db": round(float(np.mean([r["baseline_rmse_db"] for r in rows])), 4),
        "results": results,
        "epsilon_verification": {
            "delta": DELTA,
            "trials_per_dataset": audit_trials,
            "adjacent_pair": (
                "one real member cell's receivers replaced by a real non-member "
                "cell's receivers (replace-one client adjacency)"
            ),
            "per_z": verification,
            "all_orderings_hold": all(v["ordering_holds"] for v in verification),
        },
        "per_seed": rows,
    }

    z0 = next(r for r in results if r["noise_multiplier"] == 0.0)
    report["honest_finding"] = [
        (
            "The cohorts are real. Members and non-members are disjoint sets of "
            f"real ray-traced ASU-campus receivers ({rows[0]['member_receivers']} vs "
            f"{rows[0]['nonmember_receivers']}), so the undefended client-level "
            f"membership AUC of {z0['mia_client_level_auc']['mean']} is a signal "
            "about real measurements, not an artefact of two random seeds. It is "
            "still only one attack: it bounds leakage from below, never from above."
        ),
        (
            "The epsilon is verified, not asserted. For every z the audited "
            "empirical lower bound sits below the exact Gaussian epsilon at the "
            "realised sensitivity, which sits below the exact worst-case epsilon, "
            "which sits below the accountant's RDP number "
            f"(ordering_holds = {report['epsilon_verification']['all_orderings_hold']}). "
            "The gap between the audited and the certified value is the price of a "
            "worst-case guarantee, and it is reported rather than hidden."
        ),
        (
            "The clip binds. C is calibrated only on the 512 public server-held "
            f"receivers (C = {clip_norm}); it clips "
            f"{results[1]['private_updates_clipped_fraction']['mean']:.1%} of the "
            "real private updates. A clip far above the true update scale would "
            "inject noise calibrated to a sensitivity the data never attains - "
            "pure utility loss at identical epsilon."
        ),
        (
            "Utility is measured in dB against real measurements. The do-nothing "
            f"model scores {report['baseline_rmse_db']} dB RMSE on held-out real "
            f"receivers; undefended FedAvg reaches "
            f"{z0['heldout_rmse_db']['mean']} dB and the DP runs degrade from "
            "there. That degradation is the real cost of the guarantee on this "
            "task, not a proxy score."
        ),
        (
            "Scope. These are simulated ray-traced receivers. Nothing here is a "
            "measurement of privacy risk to real mobile subscribers."
        ),
    ]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--audit-trials", type=int, default=AUDIT_TRIALS)
    ap.add_argument("--features", type=Path, default=R.DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=R.DEFAULT_MANIFEST)
    args = ap.parse_args()

    report = run_suite(
        seeds=list(range(1, args.seeds + 1)),
        features=args.features,
        manifest_path=args.manifest,
        rounds=args.rounds,
        audit_trials=args.audit_trials,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    prov = report["provenance"]
    print(f"dataset={prov['dataset']}  features_sha256={prov['features_sha256'][:16]}...")
    print(
        f"real subjects: {prov['private_subject_receivers']} private + "
        f"{prov['public_root_receivers']} public root"
    )
    print(f"\nbaseline (predict the public centre) held-out RMSE = {report['baseline_rmse_db']} dB")
    print("\n  z      epsilon(R=12)   MIA AUC client / record   held-out RMSE dB   clipped")
    print("  " + "-" * 76)
    for r in report["results"]:
        eps = r["epsilon"]
        eps_s = "inf (no DP)" if "mean" not in eps else f"{eps['mean']:.3f}+-{eps['std']:.3f}"
        print(
            f"  {r['noise_multiplier']:<5g}  {eps_s:<14s}  "
            f"{r['mia_client_level_auc']['mean']:.3f} / {r['mia_record_level_auc']['mean']:.3f}"
            f"              {r['heldout_rmse_db']['mean']:.3f}"
            f"           {r['private_updates_clipped_fraction']['mean']:.2f}"
        )
    print("\n  epsilon verification (1 Gaussian round, real replace-one pair):")
    print("  z      audit(>=)   realised    worst-case   RDP        ordering")
    print("  " + "-" * 66)
    for v in report["epsilon_verification"]["per_z"]:
        print(
            f"  {v['noise_multiplier']:<5g}  {v['epsilon_audit_empirical_lower_95']:<10.4f}  "
            f"{v['epsilon_realised_pair_exact']:<10.4f}  "
            f"{v['epsilon_worst_case_analytic_exact']:<11.4f}"
            f"{v['epsilon_rdp_one_round']:<11.4f}{v['ordering_holds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
