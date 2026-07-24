#!/usr/bin/env python3
"""DP-FedAvg privacy / utility / membership-inference trade-off — an honest suite.

Robust and secure aggregation bound *poisoning* and hide *individual* updates,
but neither *bounds privacy leakage*: a curious server or a membership-inference
adversary can still learn whether a particular client's spectrum data shaped the
global DSA policy (O-RAN WG11 ML04 membership-inference; the "no differential-
privacy accountant" GAP in ``docs/THREAT_MODEL.md`` §5). The companion module
``horizon_ric.federated.dp`` closes that gap with **DP-FedAvg** + a real
**Rényi-DP accountant** (Mironov, CSF 2017; Abadi et al., CCS 2016), in the
lineage of the NTU/DTC work pairing unlearning with adaptive DP (Liu, Jiang,
Lam et al., *Efficient Federated Unlearning with Adaptive Differential Privacy
Preservation*, IEEE BigData 2024).

This suite *demonstrates the trade-off it buys*, on the federated DSA model:

  Aggregate M MEMBER clients' tabular Q-tables into a GLOBAL policy at noise
  multipliers ``z ∈ {0 (no DP), 1.0, 4.0, 8.0}``, then for each ``z`` report the
  three quantities a privacy reviewer must weigh together:

    * PRIVACY  — the ``(ε, δ)`` guarantee from the accountant (``δ = 1e-5``).
    * LEAKAGE  — a membership-inference-attack (MIA) AUC. The attacker scores a
      client by ``-‖global − local‖₂``: a member CONTRIBUTED to the average, so
      the global is pulled CLOSER to it (smaller distance ⇒ higher score). AUC is
      the rank statistic ``P(score_member > score_nonmember)`` over all member ×
      non-member pairs (Mann–Whitney; implemented here, no sklearn). AUC ≈ 0.5 ⇒
      no signal for this attacker; > 0.5 ⇒ this attacker has a membership
      signal. AUC is an empirical diagnostic, not a privacy proof.
    * UTILITY  — throughput-per-slot of the global policy on the multi-agent
      ``DSAWorld`` (``evaluate_policy``).

The clip bound ``C=12`` is public and fixed before this cohort is sampled.
The Gaussian noise is calibrated to replace-one client adjacency, whose
sensitivity is ``2C``. z=0 uses a plain mean (no clip, no noise, no privacy).

The committed run uses 64 members, 64 non-members and 10 independent cohort
seeds. Every reported empirical metric includes a seed-level bootstrap 95%
confidence interval. The formal privacy evidence is the accountant's
``(ε, δ)`` bound under the stated assumptions; the AUC only tests one concrete
attacker and must not be read as proving privacy.

Pure numpy. Run:  python benchmarks/dp_privacy_suite.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from horizon_ric.federated import dp
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv
from horizon_ric.spectrum.federated_q import QLearnConfig, evaluate_policy, train_local_q

M_MEMBERS = 64  # member clients that form the global average
K_NONMEMBERS = 64  # fresh clients never seen by the aggregator
EVAL_EPISODES = 20
NOISE_MULTIPLIERS = (0.0, 1.0, 4.0, 8.0)
DELTA = 1e-5
PUBLIC_CLIP_NORM = 12.0
BOOTSTRAP_REPETITIONS = 10_000

# Disjoint seed bands keep members and non-members independently sampled.
_MEMBER_BASE = 1_000_000
_NONMEMBER_BASE = 9_000_000
_SEED_STRIDE = 10_000


def _client_q(seed: int, dsa_cfg: DSAConfig, q_cfg: QLearnConfig) -> np.ndarray:
    """Train one client's local DSA Q-table from scratch on its own env."""
    env = DSAEnv(cfg=dsa_cfg, seed=seed)
    return train_local_q(env, q_cfg, seed=seed)


def _membership_scores(global_q: np.ndarray, locals_: list[np.ndarray]) -> list[float]:
    """Attacker's membership score per client: ``-‖global − local‖₂`` (higher ⇒
    looks more like a contributor)."""
    g = global_q.ravel()
    return [-float(np.linalg.norm(g - q.ravel())) for q in locals_]


def _mia_auc(member_scores: list[float], nonmember_scores: list[float]) -> float:
    """AUC = ``P(score_member > score_nonmember)`` over all member × non-member
    pairs (Mann–Whitney U / rank statistic), ties counted as 0.5."""
    wins = 0.0
    total = 0
    for sm in member_scores:
        for sn in nonmember_scores:
            total += 1
            if sm > sn:
                wins += 1.0
            elif sm == sn:
                wins += 0.5
    return wins / total if total else 0.5


def _run_once(
    seed: int,
    *,
    m_members: int,
    k_nonmembers: int,
    eval_episodes: int,
    clip_norm: float,
) -> dict:
    """One seed: build members + non-members, sweep z, measure ε / AUC / utility."""
    dsa_cfg = DSAConfig()
    q_cfg = QLearnConfig(episodes=8)

    members = [
        _client_q(_MEMBER_BASE + _SEED_STRIDE * seed + i, dsa_cfg, q_cfg)
        for i in range(m_members)
    ]
    nonmembers = [
        _client_q(_NONMEMBER_BASE + _SEED_STRIDE * seed + i, dsa_cfg, q_cfg)
        for i in range(k_nonmembers)
    ]
    shape = members[0].shape

    # Diagnostic only: the actual clip norm is the public, pre-registered value
    # passed to this function. Cohort statistics do not alter the mechanism.
    member_norms = [float(np.linalg.norm(m.ravel())) for m in members]
    observed_median_norm = float(np.median(member_norms))

    rng = np.random.default_rng(20240601 + seed)
    per_z: dict[str, dict] = {}
    for z in NOISE_MULTIPLIERS:
        if z == 0.0:
            # No DP: plain FedAvg mean. ε is infinite (no privacy guarantee).
            global_q = np.mean(np.stack([m for m in members]), axis=0).reshape(shape)
            epsilon = math.inf
            best_order = math.nan
            steps = 0
        else:
            acc = dp.RDPAccountant()
            cfg = dp.DPConfig(clip_norm=clip_norm, noise_multiplier=z)
            global_flat = dp.dp_fedavg(members, cfg, rng=rng, accountant=acc)
            global_q = global_flat.reshape(shape)
            epsilon, best_order = acc.get_epsilon_and_order(DELTA)
            steps = acc.steps

        member_scores = _membership_scores(global_q, members)
        nonmember_scores = _membership_scores(global_q, nonmembers)
        auc = _mia_auc(member_scores, nonmember_scores)
        util = evaluate_policy(
            global_q,
            dsa_cfg=dsa_cfg,
            n_episodes=eval_episodes,
            seed=321 + seed,
        )

        per_z[_zkey(z)] = {
            "noise_multiplier": z,
            "epsilon": (None if math.isinf(epsilon) else round(float(epsilon), 4)),
            "epsilon_label": ("inf (no DP)" if math.isinf(epsilon) else round(float(epsilon), 4)),
            "best_order": (None if math.isnan(best_order) else round(float(best_order), 4)),
            "accounted_steps": steps,
            "mia_auc": round(float(auc), 4),
            "mean_member_distance": round(float(np.mean([-s for s in member_scores])), 4),
            "mean_nonmember_distance": round(float(np.mean([-s for s in nonmember_scores])), 4),
            "throughput_per_slot": round(float(util["throughput_per_slot"]), 4),
            "collision_per_slot": round(float(util["collision_per_slot"]), 4),
        }

    return {
        "observed_median_member_norm": round(observed_median_norm, 4),
        "per_z": per_z,
    }


def _zkey(z: float) -> str:
    return "z=0_no_dp" if z == 0.0 else f"z={z:g}"


def _agg(
    rows: list[dict],
    zkey: str,
    field_: str,
    *,
    bootstrap_repetitions: int,
    rng_seed: int,
) -> dict:
    """Seed-level mean, standard deviation and bootstrap 95% interval."""
    vals = [r["per_z"][zkey][field_] for r in rows]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "ci95": [None, None], "n_seeds": 0}
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 1 or bootstrap_repetitions == 0:
        low = high = float(a.mean())
    else:
        rng = np.random.default_rng(rng_seed)
        sampled = rng.choice(a, size=(bootstrap_repetitions, len(a)), replace=True)
        means = sampled.mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std(ddof=1) if len(a) > 1 else 0.0), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "n_seeds": len(a),
    }


def run_suite(
    seeds: list[int],
    *,
    m_members: int = M_MEMBERS,
    k_nonmembers: int = K_NONMEMBERS,
    eval_episodes: int = EVAL_EPISODES,
    clip_norm: float = PUBLIC_CLIP_NORM,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if m_members < 2 or k_nonmembers < 2:
        raise ValueError("at least two members and non-members are required")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    rows = [
        _run_once(
            s,
            m_members=m_members,
            k_nonmembers=k_nonmembers,
            eval_episodes=eval_episodes,
            clip_norm=clip_norm,
        )
        for s in seeds
    ]
    observed_norms = np.asarray(
        [r["observed_median_member_norm"] for r in rows],
        dtype=np.float64,
    )

    results = []
    for z in NOISE_MULTIPLIERS:
        zk = _zkey(z)
        base_seed = 4100 + int(z * 100)
        results.append(
            {
                "noise_multiplier": z,
                "label": ("no DP (plain mean)" if z == 0.0 else f"z={z:g}"),
                "epsilon": (
                    {
                        "mean": None,
                        "std": None,
                        "ci95": [None, None],
                        "n_seeds": len(seeds),
                        "note": "infinite — no DP guarantee",
                    }
                    if z == 0.0
                    else _agg(
                        rows,
                        zk,
                        "epsilon",
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng_seed=base_seed,
                    )
                ),
                "mia_auc": _agg(
                    rows,
                    zk,
                    "mia_auc",
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=base_seed + 1,
                ),
                "throughput_per_slot": _agg(
                    rows,
                    zk,
                    "throughput_per_slot",
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=base_seed + 2,
                ),
                "mean_member_distance": _agg(
                    rows,
                    zk,
                    "mean_member_distance",
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=base_seed + 3,
                ),
                "mean_nonmember_distance": _agg(
                    rows,
                    zk,
                    "mean_nonmember_distance",
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=base_seed + 4,
                ),
                "accounted_steps": rows[0]["per_z"][zk]["accounted_steps"],
            }
        )

    return {
        "setup": {
            "m_members": m_members,
            "k_nonmembers": k_nonmembers,
            "noise_multipliers": list(NOISE_MULTIPLIERS),
            "delta": DELTA,
            "eval_episodes": eval_episodes,
            "clip_norm_C": {
                "value": clip_norm,
                "rule": "public pre-registered bound; independent of benchmark cohorts",
                "observed_median_member_norm_mean": round(float(observed_norms.mean()), 4),
                "observed_median_member_norm_std": round(
                    float(observed_norms.std(ddof=1) if len(observed_norms) > 1 else 0.0),
                    4,
                ),
            },
            "seeds": seeds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "confidence_unit": "independent cohort seed",
            "client_adjacency": "replace one client; L2 sensitivity = 2C",
            "mia_score": "membership score = -||global - local||_2 (member is pulled closer => higher score)",
            "auc_rule": "P(score_member > score_nonmember) over all member x nonmember pairs (Mann-Whitney)",
            "auc_interpretation": (
                "attack-specific empirical diagnostic; AUC near 0.5 or a CI crossing "
                "0.5 does not prove privacy"
            ),
            "lineage": [
                "Mironov, CSF 2017 (Rényi differential privacy)",
                "Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang, CCS 2016 (DP-SGD / moments accountant)",
                "Liu, Jiang, Lam et al., IEEE BigData 2024 (efficient federated unlearning with adaptive DP)",
            ],
        },
        "results": results,
        "honest_finding": [
            "The no-DP row measures whether this particular proximity attacker has "
            "a membership signal. Seed-level 95% confidence intervals expose the "
            "uncertainty instead of treating member×non-member pairs as independent.",
            "The formal privacy evidence is the Rényi accountant's (epsilon, delta) "
            "bound for one full-participation Gaussian round, a public clip bound, "
            "and conservative replace-one client adjacency. It does not cover code "
            "outside this mechanism or an unaccounted training lifecycle.",
            "MIA AUC is only an empirical diagnostic for one attacker. A value near "
            "0.5 cannot prove privacy; it can only show that this attack failed to "
            "distinguish the sampled cohorts. Throughput reports the corresponding "
            "utility cost.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="benchmarks/results/dp_privacy.json")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--members", type=int, default=M_MEMBERS)
    ap.add_argument("--nonmembers", type=int, default=K_NONMEMBERS)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--clip-norm", type=float, default=PUBLIC_CLIP_NORM)
    ap.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    report = run_suite(
        seeds,
        m_members=args.members,
        k_nonmembers=args.nonmembers,
        eval_episodes=args.eval_episodes,
        clip_norm=args.clip_norm,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    print("\n  z         epsilon(δ=1e-5)     MIA AUC        throughput/slot")
    print("  " + "-" * 62)
    for r in report["results"]:
        eps = r["epsilon"]
        eps_str = "inf (no DP)" if eps["mean"] is None else f"{eps['mean']:.3f}±{eps['std']:.3f}"
        auc = r["mia_auc"]
        tput = r["throughput_per_slot"]
        print(
            f"  {r['noise_multiplier']:<6g}  {eps_str:<18s}  "
            f"{auc['mean']:.3f} [{auc['ci95'][0]:.3f},{auc['ci95'][1]:.3f}]   "
            f"{tput['mean']:.3f}±{tput['std']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
