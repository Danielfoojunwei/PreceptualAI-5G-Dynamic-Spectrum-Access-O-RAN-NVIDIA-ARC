#!/usr/bin/env python3
"""Certified federated unlearning of a REAL false-measurement injector.

Robust aggregation only *bounds* a poisoner; it never *removes* the influence
that already leaked into the global model. This suite asks the follow-on trust
question, in the lineage of the NTU/DTC federated-unlearning research (Liu, Ye,
Jiang, Shen, Guo, Tjuawinata & Lam, *Privacy-Preserving Federated Unlearning
with Certified Client Removal*, arXiv:2404.09724; survey: Liu, Jiang, Shen,
Peng, Lam, Yuan & Liu, ACM Comput. Surv. 2024):

  Once a poisoning client is attributed, can we REMOVE its contribution from the
  trained federated model, PROVE it, and VERIFY the damage is gone?

**What changed: the federation, the attack and the damage metric are all real.**
This suite used to train tabular Q-tables in a toy ``DSAEnv`` and score a
synthetic "backdoor trigger state". It now runs on the real DeepMIMO ASU-campus
receiver population (:mod:`benchmarks.privacy_real_subjects`): ten clients are
ten real geographic cells, each holding its own real ray-traced receivers.

The attack is the injection an O-RAN operator actually has to worry about: one
cell site **falsifies its measurement reports**, adding a constant boost to one
subband's measured gain before fitting its local model. Nothing about the update
is anomalous in shape — it is a legitimate fit to fabricated data. The damage is
measured in the units the RIC cares about: the fraction of real held-out
receivers for which the global model now recommends the attacker's target
subband, and the mean dB advantage it hallucinates for that subband.

Two unlearning mechanisms are compared against a gold standard:

  * ``retrain_from_scratch`` — rerun the federation with only honest clients.
  * ``efficient_unlearn``    — cheap replay: re-aggregate the cached honest
    uploads round by round, no local recomputation.

Each result gets a signed :class:`UnlearningCertificate` recording the
Starfish-style certified bound (L2 distance to the gold standard).

Pure numpy. Run:  python benchmarks/federated_unlearning_suite.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.federated import robust
from horizon_ric.federated import unlearning as U
from horizon_ric.security.hsm import InMemoryHSMBackend

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_real_subjects as R  # noqa: E402  (sibling module, benchmarks/ is not a package)

N_CLIENTS = 10
N_MAL = 1
ROUNDS = 12
METHODS = ("fedavg", "median")
TARGET_SUBBAND = 0
INJECTION_BOOST_DB = 20.0
PARTITION_SEED = 1234
DEFAULT_OUT = "benchmarks/results/federated_unlearning.json"


# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(method: str, updates: list[np.ndarray]) -> np.ndarray:
    flats = [np.asarray(u, dtype=np.float64).ravel() for u in updates]
    if method == "fedavg":
        return np.asarray(robust.fedavg(flats), dtype=np.float64)
    if method == "median":
        return np.asarray(robust.coordinate_median(flats), dtype=np.float64)
    raise ValueError(f"unsupported aggregation method {method!r}")


def falsified_target(
    pop: R.SubjectPopulation, idx: np.ndarray, *, subband: int, boost_db: float
) -> np.ndarray:
    """The attacker's measurement report: real positions, one subband inflated.

    The rogue cell keeps every real position and every other real subband gain
    and adds ``boost_db`` to subband ``subband`` before fitting. Its upload is a
    perfectly well-formed local fit — to data that is a lie.
    """
    t = pop.target[idx].copy()
    t[:, subband] += boost_db / pop.scale_db
    return t


def train_federation(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
    malicious: set[int],
    boost_db: float,
    stealth: bool = False,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Warm-started federated training; ``malicious`` clients falsify reports."""
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    trace: list[list[np.ndarray]] = []
    for _ in range(rounds):
        ups: list[np.ndarray] = []
        for k, idx in enumerate(clients):
            if k in malicious and boost_db != 0.0:
                t = falsified_target(pop, idx, subband=TARGET_SUBBAND, boost_db=boost_db)
            else:
                t = pop.target[idx]
            ups.append(R.local_prox_update(w, pop.design[idx], t))
        if stealth and malicious:
            honest_norm = float(
                np.median([np.linalg.norm(u) for k, u in enumerate(ups) if k not in malicious])
            )
            for k in malicious:
                n = float(np.linalg.norm(ups[k]))
                if n > 0:
                    ups[k] = ups[k] * (honest_norm / n)
        trace.append(ups)
        w = w + _aggregate(method, ups)
    return w, trace


def efficient_unlearn(
    trace: list[list[np.ndarray]], *, method: str, removed: set[int]
) -> np.ndarray:
    """Cheap replay: re-aggregate the cached honest uploads, no recomputation."""
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    for ups in trace:
        kept = [u for k, u in enumerate(ups) if k not in removed]
        w = w + _aggregate(method, kept)
    return w


# ─────────────────────────────────────────────────────────────────────────────
def damage(pop: R.SubjectPopulation, w: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    """The attack's effect, measured against the real measured gains."""
    pred = pop.predict_dbw(w, test_idx)
    others = np.max(np.delete(pred, TARGET_SUBBAND, axis=1), axis=1)
    return {
        "target_subband_selection_rate": round(
            float(np.mean(np.argmax(pred, axis=1) == TARGET_SUBBAND)), 4
        ),
        "hallucinated_target_advantage_db": round(
            float(np.mean(pred[:, TARGET_SUBBAND] - others)), 4
        ),
        "rmse_db": round(pop.rmse_db(w, test_idx), 4),
    }


def run_method(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    mal = {N_CLIENTS - 1}
    clean, _ = train_federation(
        pop, clients, method=method, rounds=rounds, malicious=set(), boost_db=0.0
    )
    poisoned, trace = train_federation(
        pop, clients, method=method, rounds=rounds, malicious=mal, boost_db=INJECTION_BOOST_DB
    )
    honest_clients = [c for k, c in enumerate(clients) if k not in mal]
    gold, _ = train_federation(
        pop, honest_clients, method=method, rounds=rounds, malicious=set(), boost_db=0.0
    )
    eff = efficient_unlearn(trace, method=method, removed=mal)

    hsm = InMemoryHSMBackend()
    cert_eff = U.certify_unlearning(
        poisoned_q=poisoned,
        unlearned_q=eff,
        retrained_q=gold,
        removed_client_ids=["cell-MAL"],
        method=method,
        hsm=hsm,
        unlearn_mechanism="efficient_unlearn",
    )
    cert_gold = U.certify_unlearning(
        poisoned_q=poisoned,
        unlearned_q=gold,
        retrained_q=gold,
        removed_client_ids=["cell-MAL"],
        method=method,
        hsm=hsm,
        unlearn_mechanism="retrain_from_scratch",
    )

    # cost proxy: local fits actually executed (real receiver rows touched).
    retrain_local_fits = rounds * (N_CLIENTS - N_MAL)
    return {
        "method": method,
        "rounds": rounds,
        "damage": {
            "clean_all_honest": damage(pop, clean, test_idx),
            "poisoned": damage(pop, poisoned, test_idx),
            "efficient_unlearn": damage(pop, eff, test_idx),
            "retrain_from_scratch": damage(pop, gold, test_idx),
        },
        "certified_l2_distance_to_retrain": {
            "efficient_unlearn": float(f"{cert_eff.certified_l2_distance_to_retrain:.6g}"),
            "retrain_from_scratch": float(f"{cert_gold.certified_l2_distance_to_retrain:.6g}"),
            "note": (
                "the retrain row is 0 by construction (the gold standard is its "
                "own reference); it is a self-consistency check on the "
                "certificate plumbing, not evidence. The efficient_unlearn row is "
                "the one that carries information."
            ),
        },
        "poisoned_l2_distance_to_retrain": float(
            f"{float(np.linalg.norm(poisoned - gold)):.6g}"
        ),
        "certificate_verifies": {
            "efficient_unlearn": bool(cert_eff.verify(eff)),
            "retrain_from_scratch": bool(cert_gold.verify(gold)),
        },
        "certificate_rejects_poisoned_model": bool(not cert_gold.verify(poisoned)),
        "certificate_rejects_manifest_tamper": _tamper_rejected(cert_gold, gold),
        "cost_local_fits": {
            "retrain_from_scratch": retrain_local_fits,
            "efficient_unlearn": 0,
        },
    }


def _tamper_rejected(cert: U.UnlearningCertificate, model: np.ndarray) -> bool:
    original = cert.manifest["aggregation_method"]
    cert.manifest["aggregation_method"] = "tampered"
    rejected = not cert.verify(model)
    cert.manifest["aggregation_method"] = original
    return bool(rejected)


# ─────────────────────────────────────────────────────────────────────────────
def detection_probe(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    rounds: int,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Can whole-vector outlier screening attribute the injector? Measured.

    Three variants, including a positive control so a null result cannot be
    dismissed as a broken screen.
    """
    mal = {N_CLIENTS - 1}
    out: dict[str, Any] = {}

    def record(name: str, trace: list[list[np.ndarray]], w: np.ndarray, note: str) -> None:
        flagged = U.detect_outliers(trace[-1], n_flagged=1)[0]
        norms = [float(np.linalg.norm(u)) for u in trace[-1]]
        out[name] = {
            "flagged_client": int(flagged),
            "attacker_client": N_CLIENTS - 1,
            "attacker_flagged": bool(flagged == N_CLIENTS - 1),
            "attacker_update_norm": round(norms[-1], 6),
            "honest_update_norm_median": round(float(np.median(norms[:-1])), 6),
            "attacker_norm_ratio": round(norms[-1] / max(float(np.median(norms[:-1])), 1e-30), 4),
            "damage": damage(pop, w, test_idx),
            "note": note,
        }

    w, trace = train_federation(
        pop, clients, method="median", rounds=rounds, malicious=mal,
        boost_db=INJECTION_BOOST_DB,
    )
    record(
        "measurement_falsification",
        trace,
        w,
        "the shipped attack: a well-formed local fit to fabricated measurements",
    )

    w, trace = train_federation(
        pop, clients, method="median", rounds=rounds, malicious=mal,
        boost_db=INJECTION_BOOST_DB, stealth=True,
    )
    record(
        "norm_matched_falsification",
        trace,
        w,
        "the same attack rescaled to the honest median update norm",
    )

    # Positive control: the crude scaling attack the screen is designed for.
    w_ctl, trace_ctl = train_federation(
        pop, clients, method="median", rounds=rounds, malicious=set(), boost_db=0.0
    )
    boosted = [list(ups) for ups in trace_ctl]
    for ups in boosted:
        ups[N_CLIENTS - 1] = ups[N_CLIENTS - 1] * 10.0
    record("scaling_attack_control", boosted, w_ctl, "positive control: 10x update scaling")

    return {
        "screen": "whole-vector L2 distance-to-mean outlier detection (U.detect_outliers)",
        "trials": out,
        "finding": (
            "Whole-vector screening does NOT attribute this attack. The rogue "
            "cell's update is a legitimate local fit, and because it is fitting a "
            "global that its own lie has already shifted, its update norm is "
            f"{out['measurement_falsification']['attacker_norm_ratio']}x the honest "
            "median - it is SMALLER than an honest client's, not larger. The "
            "positive control confirms the screen itself works "
            f"(attacker_flagged={out['scaling_attack_control']['attacker_flagged']} at "
            "10x scaling), so this is a real blind spot, not a broken detector. "
            "Unlearning removes whatever is attributed; it cannot attribute what "
            "detection cannot see, and there the deterministic Decision Safety "
            "Shield remains the backstop that bounds the emitted action."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_suite(
    *, features: Path, manifest_path: Path, rounds: int = ROUNDS
) -> dict[str, Any]:
    pop, manifest = R.build_population(features, manifest_path)
    clients = [
        pop.private_idx[c]
        for c in R.partition_geographic(pop.pos[pop.private_idx], N_CLIENTS, seed=PARTITION_SEED)
    ]
    test_idx = pop.root_idx  # public receivers no client ever held
    results = [
        run_method(pop, clients, method=m, rounds=rounds, test_idx=test_idx) for m in METHODS
    ]
    detection = detection_probe(pop, clients, rounds=rounds, test_idx=test_idx)

    fed = next(r for r in results if r["method"] == "fedavg")
    med = next(r for r in results if r["method"] == "median")
    sizes = np.asarray([c.size for c in clients])
    report = {
        "benchmark": "Certified federated unlearning of a real false-measurement injector",
        "provenance": R.provenance_block(
            manifest,
            pop,
            extra_scope=(
                "The 'attack' is a simulated falsification of ray-traced "
                "measurements, not an observed real-world incident."
            ),
        ),
        "setup": {
            "n_clients": N_CLIENTS,
            "n_malicious": N_MAL,
            "client_size_min": int(sizes.min()),
            "client_size_max": int(sizes.max()),
            "rounds": rounds,
            "model_dim": R.MODEL_DIM,
            "task": "federated per-subband path-loss regression on real measured gains",
            "attack": (
                f"one real cell falsifies its measurement reports, adding "
                f"+{INJECTION_BOOST_DB} dB to subband {TARGET_SUBBAND} before "
                "fitting its local model; every position and every other subband "
                "is its real ray-traced value"
            ),
            "damage_metric": (
                "fraction of real held-out receivers for which the global model "
                f"now recommends subband {TARGET_SUBBAND}, and the mean dB "
                "advantage it hallucinates for it"
            ),
            "evaluation_receivers": (
                "the 512 public server-held receivers, never given to any client"
            ),
            "partition_seed": PARTITION_SEED,
            "lineage": [
                "Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam, arXiv:2404.09724 "
                "(Starfish: certified client removal)",
                "Liu, Jiang, Shen, Peng, Lam, Yuan & Liu, ACM Comput. Surv. 2024 "
                "(federated unlearning survey)",
            ],
        },
        "results": results,
        "detection": detection,
        "honest_finding": [
            (
                "The injection works on real data and robust aggregation does not "
                "stop it. On the real held-out receivers the clean model "
                "recommends subband "
                f"{TARGET_SUBBAND} for "
                f"{fed['damage']['clean_all_honest']['target_subband_selection_rate']:.1%} "
                "of them; one falsifying cell out of ten drives that to "
                f"{fed['damage']['poisoned']['target_subband_selection_rate']:.1%} under "
                "FedAvg and still "
                f"{med['damage']['poisoned']['target_subband_selection_rate']:.1%} under "
                "coordinate median. The update is a legitimate fit to fabricated "
                "data, so it is not an outlier in shape - which is exactly why a "
                "robust aggregator bounds it rather than removes it."
            ),
            (
                "Retrain-from-scratch unlearning closes it. Recommending subband "
                f"{TARGET_SUBBAND} returns to "
                f"{fed['damage']['retrain_from_scratch']['target_subband_selection_rate']:.1%} "
                "(FedAvg), matching the clean baseline, and the hallucinated "
                "advantage returns to "
                f"{fed['damage']['retrain_from_scratch']['hallucinated_target_advantage_db']} dB. "
                "It costs "
                f"{fed['cost_local_fits']['retrain_from_scratch']} local fits over real "
                "receiver rows."
            ),
            (
                "The cheap replay is measurably insufficient, and its certificate "
                "says so. Re-aggregating the cached honest uploads leaves the "
                "target-subband rate at "
                f"{fed['damage']['efficient_unlearn']['target_subband_selection_rate']:.1%} "
                "(FedAvg) because the honest clients warm-started from poisoned "
                "globals, so the poison is baked into their own cached uploads. "
                "Its certified L2 distance to the gold standard is "
                f"{fed['certified_l2_distance_to_retrain']['efficient_unlearn']:.3g} against "
                f"{fed['certified_l2_distance_to_retrain']['retrain_from_scratch']:.3g} for the "
                "retrain - the honest signal an operator needs to reject the cheap "
                "result. The certificate never claims a removal that did not happen."
            ),
            (
                "Attribution is the precondition and it FAILS here. Whole-vector "
                "outlier screening does not flag the falsifying cell "
                f"(attacker_flagged="
                f"{detection['trials']['measurement_falsification']['attacker_flagged']}): "
                "its update norm is "
                f"{detection['trials']['measurement_falsification']['attacker_norm_ratio']}x "
                "the honest median, i.e. SMALLER than an honest client's, because "
                "it is fitting a global its own lie already moved. A 10x scaling "
                "control is flagged "
                f"({detection['trials']['scaling_attack_control']['attacker_flagged']}), "
                "so the screen works - this attack is simply invisible to it. The "
                "certified unlearning machinery below is therefore only as good as "
                "some independent attribution signal, and the deterministic Shield "
                "remains the backstop when there is none."
            ),
            (
                "Scope. These are ray-traced receivers and a simulated "
                "falsification, not a measured real-world attack."
            ),
        ],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--features", type=Path, default=R.DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=R.DEFAULT_MANIFEST)
    args = ap.parse_args()

    report = run_suite(
        features=args.features, manifest_path=args.manifest, rounds=args.rounds
    )
    text = json.dumps(report, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    prov = report["provenance"]
    print(f"dataset={prov['dataset']}  features_sha256={prov['features_sha256'][:16]}...")
    print(f"evaluation on {prov['public_root_receivers']} public receivers no client held\n")
    for r in report["results"]:
        d = r["damage"]
        print(
            f"[{r['method']:7s}] target-subband recommendation rate on real receivers"
            f"\n    clean (all honest)   {d['clean_all_honest']['target_subband_selection_rate']:.3f}"
            f"\n    poisoned             {d['poisoned']['target_subband_selection_rate']:.3f}"
            f"  ({d['poisoned']['hallucinated_target_advantage_db']:+.3f} dB hallucinated)"
            f"\n    efficient_unlearn    {d['efficient_unlearn']['target_subband_selection_rate']:.3f}"
            f"   certified L2 {r['certified_l2_distance_to_retrain']['efficient_unlearn']:.4g}"
            f"\n    retrain_from_scratch {d['retrain_from_scratch']['target_subband_selection_rate']:.3f}"
            f"   certified L2 {r['certified_l2_distance_to_retrain']['retrain_from_scratch']:.4g}\n"
        )
    for name, t in report["detection"]["trials"].items():
        print(
            f"[detect ] {name:27s} attacker_flagged={t['attacker_flagged']!s:5s} "
            f"norm ratio {t['attacker_norm_ratio']:>7.3f}  damage "
            f"{t['damage']['target_subband_selection_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
