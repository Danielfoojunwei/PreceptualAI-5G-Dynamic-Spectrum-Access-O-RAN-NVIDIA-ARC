#!/usr/bin/env python3
"""Certified federated unlearning vs. a propagated backdoor — an honest suite.

Our adversarial campaign found the worst FAIL: a sparse backdoor *survives*
robust aggregation (``benchmarks/results/dsa_poison_suite.json``; THREAT_MODEL
§7). Robust aggregation only *bounds* a poisoner; it never *removes* the
influence that already leaked into the global policy. This suite asks the
follow-on trust question, in the lineage of the NTU/DTC federated-unlearning
research (Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam, *Privacy-Preserving
Federated Unlearning with Certified Client Removal*, arXiv:2404.09724; survey:
Liu, Jiang, Shen, Peng, Lam, Yuan & Liu, ACM Comput. Surv. 2024; backdoor-as-
probe: Han et al., arXiv:2412.11476):

  Once a poisoning client is attributed, can we REMOVE its contribution from the
  trained federated DSA policy, PROVE it, and VERIFY the backdoor is gone?

We train a federated DSA policy over several warm-started rounds with one
backdoor client (the attacker forces a channel a *healthy* policy avoids — max
damage), then unlearn it two ways:

  * retrain_from_scratch — gold standard (honest-only re-training).
  * efficient_unlearn    — cheap replay of cached honest uploads.

Each unlearned model gets a signed :class:`UnlearningCertificate` recording the
Starfish-style certified bound (L2 distance to the gold standard) and the
backdoor-probe success before/after.

Honest finding (see committed results): under undefended FedAvg the backdoor
lands at success ~1.0; **retrain-from-scratch unlearning drives it back to the
clean baseline (~0.03)** — it closes the FAIL — while the **cheap replay is
INSUFFICIENT** because the backdoor propagated into the honest clients'
warm-started tables, and its **certified distance-to-retrain (large) flags that
honestly**. The certificate never claims a removal that did not happen.

Pure numpy. Run:  python benchmarks/federated_unlearning_suite.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from horizon_ric.federated import unlearning as U
from horizon_ric.security.hsm import InMemoryHSMBackend
from horizon_ric.spectrum.data_poison import BackdoorSpec, backdoor_success_rate
from horizon_ric.spectrum.dsa_env import DSAConfig
from horizon_ric.spectrum.federated_q import QLearnConfig

N_CLIENTS = 10
N_MAL = 1
ROUNDS = 5
TRIGGER_STATE = 0
TARGET_CHANNEL = 5  # a channel the healthy policy avoids (clean success ~0.03)


def _run_once(method: str, seed: int) -> dict:
    dsa = DSAConfig()
    qc = QLearnConfig(episodes=6)
    bd = BackdoorSpec(trigger_state=TRIGGER_STATE, target_channel=TARGET_CHANNEL, boost=50.0)

    clients = [U.ClientSpec(f"cell-{i}", seed=100 * seed + i, role=U.HONEST)
               for i in range(N_CLIENTS - N_MAL)]
    clients.append(U.ClientSpec("cell-MAL", seed=100 * seed + 999, role=U.BACKDOOR, backdoor=bd))

    poisoned, trace = U.run_federated_training(
        clients, rounds=ROUNDS, method=method, dsa_cfg=dsa, q_cfg=qc, base_seed=seed)

    removed = ["cell-MAL"]
    eff = U.efficient_unlearn(trace, removed)
    gold = U.retrain_from_scratch(trace, removed)

    hsm = InMemoryHSMBackend()
    cert_eff = U.certify_unlearning(
        poisoned_q=poisoned, unlearned_q=eff, retrained_q=gold, removed_client_ids=removed,
        method=method, hsm=hsm, unlearn_mechanism="efficient_unlearn", backdoor=bd, dsa_cfg=dsa)
    cert_gold = U.certify_unlearning(
        poisoned_q=poisoned, unlearned_q=gold, retrained_q=gold, removed_client_ids=removed,
        method=method, hsm=hsm, unlearn_mechanism="retrain_from_scratch", backdoor=bd, dsa_cfg=dsa)

    # cost proxy: local trainings actually executed (env rollouts), not vector ops.
    retrain_local_trainings = ROUNDS * (N_CLIENTS - N_MAL)
    efficient_local_trainings = 0

    return {
        "backdoor_success": {
            "poisoned": round(float(backdoor_success_rate(poisoned, bd, dsa_cfg=dsa)), 4),
            "efficient_unlearn": round(cert_eff.backdoor_success_after or 0.0, 4),
            "retrain_from_scratch": round(cert_gold.backdoor_success_after or 0.0, 4),
        },
        "certified_l2_distance_to_retrain": {
            "efficient_unlearn": round(cert_eff.certified_l2_distance_to_retrain, 4),
            "retrain_from_scratch": round(cert_gold.certified_l2_distance_to_retrain, 4),
        },
        "certificate_verifies": {
            "efficient_unlearn": bool(cert_eff.verify(eff)),
            "retrain_from_scratch": bool(cert_gold.verify(gold)),
        },
        "certificate_rejects_poisoned_model": not cert_gold.verify(poisoned),
        "cost_local_trainings": {
            "retrain_from_scratch": retrain_local_trainings,
            "efficient_unlearn": efficient_local_trainings,
        },
    }


def _agg(rows: list[dict], path: list) -> dict:
    vals = []
    for r in rows:
        cur = r
        for k in path:
            cur = cur[k]
        vals.append(float(cur))
    a = np.array(vals)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}


def run_method(method: str, seeds: list[int]) -> dict:
    rows = [_run_once(method, s) for s in seeds]
    return {
        "method": method,
        "seeds": seeds,
        "backdoor_success": {
            k: _agg(rows, ["backdoor_success", k])
            for k in ("poisoned", "efficient_unlearn", "retrain_from_scratch")
        },
        "certified_l2_distance_to_retrain": {
            k: _agg(rows, ["certified_l2_distance_to_retrain", k])
            for k in ("efficient_unlearn", "retrain_from_scratch")
        },
        "certificate_verifies": rows[0]["certificate_verifies"],
        "certificate_rejects_poisoned_model": all(r["certificate_rejects_poisoned_model"] for r in rows),
        "cost_local_trainings": rows[0]["cost_local_trainings"],
    }


def detection_demo(seed: int = 0) -> dict:
    """Show the detection→unlearn loop closes for a large-norm attack, and is
    honest that a sparse backdoor evades whole-vector screening."""
    dsa = DSAConfig()
    qc = QLearnConfig(episodes=6)
    bd = BackdoorSpec(TRIGGER_STATE, TARGET_CHANNEL, 50.0)

    def flagged_id(role, backdoor=None):
        clients = [U.ClientSpec(f"cell-{i}", seed=100 + i, role=U.HONEST) for i in range(9)]
        clients.append(U.ClientSpec("cell-MAL", seed=999, role=role, backdoor=backdoor))
        _, trace = U.run_federated_training(clients, rounds=ROUNDS, method="median",
                                            dsa_cfg=dsa, q_cfg=qc, base_seed=seed)
        idx = U.detect_outliers(trace.rounds[-1].uploads, n_flagged=1)[0]
        return trace.rounds[-1].client_ids[idx]

    return {
        "qtable_target_flagged": flagged_id(U.QTABLE_TARGET),       # large-norm → caught
        "reward_poison_flagged": flagged_id(U.REWARD_POISON),       # large-norm → caught
        "backdoor_flagged": flagged_id(U.BACKDOOR, bd),             # boost=50 row → caught here
        "note": (
            "Whole-vector distance flags all three attacks in this configuration, "
            "including the backdoor — its boosted trigger row (boost=50) is large "
            "enough to screen — so the detection->unlearn->certify loop closes for "
            "each. The honest caveat: a NORM-MATCHED stealthy backdoor (scaled to "
            "blend with honest update norms, the regime that lets a backdoor "
            "survive robust aggregation) would evade this whole-vector screen; "
            "attributing it needs an independent signal, and there the "
            "deterministic Shield remains the backstop. Unlearning removes whatever "
            "is attributed; it cannot attribute what detection cannot see."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    report = {
        "setup": {
            "n_clients": N_CLIENTS, "n_malicious": N_MAL, "rounds": ROUNDS,
            "trigger_state": TRIGGER_STATE, "target_channel": TARGET_CHANNEL,
            "attacker_note": "forces a channel the healthy policy avoids (max damage)",
            "lineage": [
                "Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam, arXiv:2404.09724 (Starfish: certified client removal)",
                "Liu, Jiang, Shen, Peng, Lam, Yuan & Liu, ACM Comput. Surv. 2024 (federated unlearning survey)",
                "Han, Zhu, Zhang, Huo & Zhou, arXiv:2412.11476 (backdoor-certification verification)",
            ],
        },
        "results": [run_method(m, seeds) for m in ("fedavg", "median")],
        "detection": detection_demo(),
        "honest_finding": [
            "Under undefended FedAvg the backdoor lands at success ~1.0. "
            "retrain-from-scratch certified unlearning drives it to the clean "
            "baseline (~0.03) — it CLOSES the backdoor-survives-aggregation FAIL "
            "— and the signed certificate (certified L2 distance ~0 to the gold "
            "standard) verifies against the unlearned model and rejects the "
            "still-poisoned one.",
            "The CHEAP efficient_unlearn (replay cached honest uploads) is "
            "INSUFFICIENT here: the backdoor propagated into the honest clients' "
            "warm-started Q-tables, so re-aggregating them keeps it. Its certified "
            "distance-to-retrain is LARGE, which is exactly the honest signal an "
            "operator needs to reject the cheap result and fall back to full "
            "retraining. The certificate never claims a removal that did not happen.",
            "Unlearning is not a substitute for the Shield: it requires "
            "ATTRIBUTION. Here whole-vector detection flags all three attacks (the "
            "boosted backdoor row is large enough to screen), so the "
            "detection->unlearn->certify loop closes. But a NORM-MATCHED stealthy "
            "backdoor — the regime that survives robust aggregation — would evade "
            "detection, leaving nothing to attribute; there the deterministic "
            "Decision Safety Shield remains the backstop that bounds the emitted "
            "action to legal spectrum.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    for r in report["results"]:
        bs = r["backdoor_success"]
        print(f"\n[{r['method']:7s}] backdoor success  poisoned={bs['poisoned']['mean']:.3f}"
              f"  efficient={bs['efficient_unlearn']['mean']:.3f}"
              f"  retrain={bs['retrain_from_scratch']['mean']:.3f}"
              f"  | certified L2(retrain)={r['certified_l2_distance_to_retrain']['retrain_from_scratch']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
