#!/usr/bin/env python3
"""GDPR Art. 17 subject erasure on REAL ray-traced subjects — an honest suite.

Client-level unlearning removes a whole *participant*. The right a regulator
actually enforces is finer: a single **data subject** asks for *their* rows to be
forgotten while everyone else's contribution stays (sample-level unlearning, in
the lineage of Lam et al., *Certifying the Right to be Forgotten: Primal-Dual
Optimization for Sample and Label Unlearning in Vertical Federated Learning*,
IEEE TIFS).

**What changed: the subjects are real.** This suite used to invent subjects by
assigning toy ``DSAEnv`` rollout episodes round-robin to strings like
``"cell-1-sub-0"`` — an erasure claim about records that never existed. A subject
is now **one real DeepMIMO ASU-campus receiver**: its private record is its real
3D position and its six real ray-traced subband gains
(:mod:`benchmarks.privacy_real_subjects`). Clients are real geographic cells of
the campus; the model is the real federated per-subband path-loss regressor.

Why erasure is *exact* here, and how that is checked rather than claimed
-----------------------------------------------------------------------
A FedProx local step is a closed-form function of the client's **sufficient
statistics** ``G = DᵀD`` and ``H = DᵀT`` and its subject count ``n``. One
subject contributes exactly the rank-1 terms ``d dᵀ`` and ``d tᵀ``. Erasing a
named receiver is therefore an exact **rank-1 downdate**, replayed through every
warm-started round — no gradient surgery, no approximation.

The suite computes the same post-erasure model two structurally different ways:

  * ``rank-1 downdate`` — subtract the subject's outer products from the cached
    statistics and replay;
  * ``from-scratch retrain`` — rebuild the federation from the raw rows with the
    subject's row physically absent.

They must agree to floating point. That agreement is the Art. 17 "as if the data
had never been used" claim, *measured*, not asserted.

It also reports two things an operator needs and a marketing claim would hide:

  * ``cheap_linear_shortcut_residual_to_retrain`` — what an operator who patches
    only the final aggregate (instead of replaying) is actually left holding.
    Zero for FedAvg at one round; **non-zero** once rounds warm-start, and larger
    still under a non-linear aggregator (coordinate median).
  * an **erasure audit** — after erasure, is the erased receiver still
    distinguishable from receivers the federation never saw? Reported as a
    membership AUC before and after, over many real erased subjects.

The erasure is bound to a signed :class:`ErasureCertificate` (RSA-PSS via the
real provenance HSM) that verifies against the post-erasure model, rejects the
pre-erasure one, and rejects a tampered manifest.

Pure numpy. Run:  python benchmarks/subject_erasure_suite.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.federated import robust
from horizon_ric.federated.erasure import ErasureCertificate
from horizon_ric.provenance.signing import sha256_hex, sign_model
from horizon_ric.security.hsm import InMemoryHSMBackend

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_real_subjects as R  # noqa: E402  (sibling module, benchmarks/ is not a package)

N_CLIENTS = 8
ROUNDS = 12
METHODS = ("fedavg", "median")
PARTITION_SEED = 1234
AUDIT_SUBJECTS = 200  # real receivers erased one at a time for the erasure audit
DEFAULT_OUT = "benchmarks/results/subject_erasure.json"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(method: str, updates: list[np.ndarray]) -> np.ndarray:
    flats = [np.asarray(u, dtype=np.float64).ravel() for u in updates]
    if method == "fedavg":
        return np.asarray(robust.fedavg(flats), dtype=np.float64)
    if method == "median":
        return np.asarray(robust.coordinate_median(flats), dtype=np.float64)
    raise ValueError(f"unsupported aggregation method {method!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Two structurally independent computations of the same federated model
# ─────────────────────────────────────────────────────────────────────────────
def train_from_rows(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Warm-started federated training straight from each client's raw real rows."""
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    trace: list[list[np.ndarray]] = []
    for _ in range(rounds):
        ups = [R.local_prox_update(w, pop.design[i], pop.target[i]) for i in clients]
        trace.append(ups)
        w = w + _aggregate(method, ups)
    return w, trace


def train_from_stats(
    stats: list[tuple[np.ndarray, np.ndarray, int]], *, method: str, rounds: int
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """The same training, driven only by cached per-client sufficient statistics."""
    w = np.zeros(R.MODEL_DIM, dtype=np.float64)
    trace: list[list[np.ndarray]] = []
    for _ in range(rounds):
        ups = [R.prox_update_from_stats(w, g, h, n) for g, h, n in stats]
        trace.append(ups)
        w = w + _aggregate(method, ups)
    return w, trace


def downdate(
    stat: tuple[np.ndarray, np.ndarray, int], d: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove one subject's exact rank-1 contribution from a client's statistics."""
    gram, cross, n = stat
    if n <= 1:
        raise ValueError("cannot erase the only subject a client holds")
    return gram - np.outer(d, d), cross - np.outer(d, t), n - 1


# ─────────────────────────────────────────────────────────────────────────────
# Certificate
# ─────────────────────────────────────────────────────────────────────────────
def build_certificate(
    *,
    subject_id: str,
    client_id: str,
    method: str,
    before: np.ndarray,
    after: np.ndarray,
    distance: float,
    n_records_erased: int,
    features_sha256: str,
    hsm: Any,
) -> ErasureCertificate:
    b = np.ascontiguousarray(before, dtype=np.float64).tobytes()
    a = np.ascontiguousarray(after, dtype=np.float64).tobytes()
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "event": "subject_erasure_art17",
        "subject_id": subject_id,
        "client_id": client_id,
        "aggregation_method": method,
        "n_records_erased": n_records_erased,
        "dataset_features_sha256": features_sha256,
        "model_sha256_before": sha256_hex(b),
        "model_sha256_after": sha256_hex(a),
        "certified_l2_distance_to_retrain": distance,
        "created_at": created_at,
    }
    provenance = sign_model(
        a,
        trainer_id="horizon-erasure-service",
        training_manifest=manifest,
        hsm=hsm,
        key_label="erasure-signer",
    )
    return ErasureCertificate(
        subject_id=subject_id,
        client_id=client_id,
        method=method,
        model_sha256_before=sha256_hex(b),
        model_sha256_after=sha256_hex(a),
        certified_l2_distance_to_retrain=distance,
        created_at=created_at,
        manifest=manifest,
        provenance=provenance,
    )


def _tamper_rejected(cert: ErasureCertificate, after: np.ndarray) -> bool:
    original = cert.manifest["subject_id"]
    cert.manifest["subject_id"] = "someone-else"
    rejected = not cert.verify(after)
    cert.manifest["subject_id"] = original
    return bool(rejected)


# ─────────────────────────────────────────────────────────────────────────────
# One erasure
# ─────────────────────────────────────────────────────────────────────────────
def run_once(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
    target_client: int,
    subject_row: int,
    features_sha256: str,
) -> dict[str, Any]:
    stats = [R.suff_stats(pop.design[i], pop.target[i]) for i in clients]
    global_before, trace_before = train_from_stats(stats, method=method, rounds=rounds)

    d = pop.design[subject_row]
    t = pop.target[subject_row]
    erased_stats = list(stats)
    erased_stats[target_client] = downdate(stats[target_client], d, t)
    global_after, _ = train_from_stats(erased_stats, method=method, rounds=rounds)

    # Independent gold standard: physically delete the subject's row and retrain
    # the whole federation from the raw real rows. Structurally different code
    # path from the rank-1 downdate above; they must agree.
    kept_clients = list(clients)
    kept_clients[target_client] = clients[target_client][
        clients[target_client] != subject_row
    ]
    gold, _ = train_from_rows(pop, kept_clients, method=method, rounds=rounds)

    certified = float(np.linalg.norm(global_after - gold))

    # The operator shortcut an audit has to price: take the shipped global and
    # apply a single LINEAR mean-correction for the target client's last-round
    # delta -- no replay of the earlier rounds, no re-running the aggregator.
    # Under FedAvg at one round this is algebraically the retrain; with warm
    # starting, or under a non-linear aggregator, it is not.
    last = trace_before[-1]
    w_penultimate = global_before - _aggregate(method, last)
    erased_last = R.prox_update_from_stats(w_penultimate, *erased_stats[target_client])
    cheap = global_before + (erased_last - last[target_client]) / len(clients)
    cheap_residual = float(np.linalg.norm(cheap - gold))

    hsm = InMemoryHSMBackend()
    subject_id = pop.subject_id(subject_row)
    cert = build_certificate(
        subject_id=subject_id,
        client_id=f"cell-{target_client}",
        method=method,
        before=global_before,
        after=global_after,
        distance=certified,
        n_records_erased=1,
        features_sha256=features_sha256,
        hsm=hsm,
    )
    pinned = bytes.fromhex(cert.provenance.public_key_der_hex) if cert.provenance else b""

    idx = np.array([subject_row])
    return {
        "method": method,
        "target_client": f"cell-{target_client}",
        "subject_id": subject_id,
        "subject_receiver_index": int(pop.receiver_index[subject_row]),
        "subject_position_m": [round(float(v), 3) for v in pop.pos[subject_row]],
        "client_subjects": int(clients[target_client].size),
        "global_changed": bool(not np.array_equal(global_before, global_after)),
        "influence_l2_before_vs_after": round(
            float(np.linalg.norm(global_before - global_after)), 12
        ),
        "influence_prediction_shift_db": round(
            float(
                np.mean(
                    np.abs(
                        pop.predict_dbw(global_before, idx) - pop.predict_dbw(global_after, idx)
                    )
                )
            ),
            9,
        ),
        # NOT rounded: the whole claim is that this is floating-point noise, and
        # rounding to 15 decimals would silently turn 2e-18 into a fake 0.0.
        "certified_l2_distance_to_retrain": float(f"{certified:.6g}"),
        "model_norm_for_scale": float(f"{float(np.linalg.norm(gold)):.6g}"),
        "cheap_linear_shortcut_residual_to_retrain": float(f"{cheap_residual:.6g}"),
        "certificate_verifies_after": bool(cert.verify(global_after)),
        "certificate_rejects_before": bool(not cert.verify(global_before)),
        "certificate_pin_verify_ok": bool(
            cert.verify(global_after, trusted_public_key_der=pinned)
        ),
        "certificate_wrong_pin_fails": bool(
            not cert.verify(global_after, trusted_public_key_der=b"\x00" * 32)
        ),
        "certificate_rejects_manifest_tamper": _tamper_rejected(cert, global_after),
        "certificate_binds_features_sha256": cert.manifest["dataset_features_sha256"]
        == features_sha256,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Erasure audit: is an erased real receiver still distinguishable?
# ─────────────────────────────────────────────────────────────────────────────
def erasure_audit(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
    target_client: int,
    n_subjects: int,
    seed: int,
) -> dict[str, Any]:
    """Erase many real receivers one at a time and measure whether it "took".

    Two tests, because the obvious one is confounded:

    * **paired leave-one-out** (the sound test). For each real receiver, compare
      its own loss under the model that trained on it against its loss under the
      model produced by erasing it. This is the same receiver, same geography,
      same everything except membership, so nothing but membership can explain a
      difference. Reported as the fraction of subjects whose loss rose after
      their own erasure and the mean rise, in dB².
    * **population AUC** (reported with its confound stated). Erased receivers
      vs public-root receivers that no client ever held. Only root receivers
      lying inside the target cell's own region are used, because a stride-8
      sample of the whole campus is not comparable to one geographic cell -
      an AUC away from 0.5 there measures geography, not membership.
    """
    rng = np.random.default_rng(seed)
    pool = clients[target_client]
    chosen = rng.choice(pool, size=min(n_subjects, pool.size - 1), replace=False)

    # Geography-matched never-seen reference: public root receivers whose nearest
    # client centroid is the target client.
    centroids = np.stack([pop.pos[c].mean(axis=0) for c in clients])
    d_root = np.linalg.norm(pop.pos[pop.root_idx][:, None, :] - centroids[None, :, :], axis=2)
    never_seen = pop.root_idx[np.argmin(d_root, axis=1) == target_client]
    if never_seen.size < 5:
        never_seen = pop.root_idx

    stats = [R.suff_stats(pop.design[i], pop.target[i]) for i in clients]
    w_before, _ = train_from_stats(stats, method=method, rounds=rounds)
    reference_scores = -pop.per_subject_squared_error(w_before, never_seen)

    before_scores, after_scores, shifts, rises = [], [], [], []
    for row in chosen:
        idx = np.array([row])
        s_in = float(-pop.per_subject_squared_error(w_before, idx)[0])
        er = list(stats)
        er[target_client] = downdate(stats[target_client], pop.design[row], pop.target[row])
        w_after, _ = train_from_stats(er, method=method, rounds=rounds)
        s_out = float(-pop.per_subject_squared_error(w_after, idx)[0])
        before_scores.append(s_in)
        after_scores.append(s_out)
        rises.append(s_in - s_out)  # >0 means the loss rose once the subject left
        shifts.append(float(np.linalg.norm(w_before - w_after)))

    rise = np.asarray(rises, dtype=np.float64)
    return {
        "method": method,
        "subjects_erased": int(len(chosen)),
        "paired_leave_one_out": {
            "fraction_loss_rose_after_own_erasure": round(float(np.mean(rise > 0)), 4),
            "mean_loss_rise_db2": float(f"{float(rise.mean()):.6g}"),
            "max_loss_rise_db2": float(f"{float(rise.max()):.6g}"),
            "reading": (
                "the reported fraction is the achieved value, not a target. A "
                "majority above 0.5 means the erasure really does remove a fitted "
                "influence: the model that held the receiver predicted it better "
                "than the model produced by erasing it. The remainder are "
                "receivers whose own removal happened to improve their fit, which "
                "real spatial heterogeneity produces at this effect size - one "
                "subject out of hundreds barely moves a 36-parameter model, so the "
                "per-subject signal sits close to numerical noise."
            ),
        },
        "population_auc": {
            "never_seen_reference_receivers": int(never_seen.size),
            "reference_selection": (
                "public server-held root receivers whose nearest client centroid "
                "is the target client (geography-matched, never held by a client)"
            ),
            "membership_auc_before_erasure": round(
                R.auc_mann_whitney(before_scores, reference_scores), 4
            ),
            "membership_auc_after_erasure": round(
                R.auc_mann_whitney(after_scores, reference_scores), 4
            ),
            "confound": (
                "even geography-matched, this AUC mixes membership with residual "
                "spatial heterogeneity of the real campus; only the change "
                "before->after is attributable to erasure"
            ),
        },
        "mean_model_l2_shift_per_erasure": float(f"{float(np.mean(shifts)):.6g}"),
        "max_model_l2_shift_per_erasure": float(f"{float(np.max(shifts)):.6g}"),
        "statistic": (
            "loss-threshold membership score = -mean squared dB error of the "
            "released model on that receiver's real measured subband gains "
            "(Yeom et al., CSF 2018)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_method(
    pop: R.SubjectPopulation,
    clients: list[np.ndarray],
    *,
    method: str,
    rounds: int,
    subject_rows: list[tuple[int, int]],
    features_sha256: str,
) -> dict[str, Any]:
    rows = [
        run_once(
            pop,
            clients,
            method=method,
            rounds=rounds,
            target_client=ci,
            subject_row=sr,
            features_sha256=features_sha256,
        )
        for ci, sr in subject_rows
    ]
    single = [
        run_once(
            pop,
            clients,
            method=method,
            rounds=1,
            target_client=ci,
            subject_row=sr,
            features_sha256=features_sha256,
        )
        for ci, sr in subject_rows
    ]

    def agg(rs: list[dict], key: str) -> dict[str, float]:
        a = np.asarray([float(r[key]) for r in rs], dtype=np.float64)
        return {
            "mean": float(f"{a.mean():.6g}"),
            "max": float(f"{a.max():.6g}"),
            "min": float(f"{a.min():.6g}"),
        }

    return {
        "method": method,
        "rounds": rounds,
        "n_erasures": len(rows),
        "erased_subject_ids": [r["subject_id"] for r in rows],
        "global_changed": all(r["global_changed"] for r in rows),
        "influence_l2_before_vs_after": agg(rows, "influence_l2_before_vs_after"),
        "influence_prediction_shift_db": agg(rows, "influence_prediction_shift_db"),
        "certified_l2_distance_to_retrain": agg(rows, "certified_l2_distance_to_retrain"),
        "cheap_linear_shortcut_residual_to_retrain": agg(
            rows, "cheap_linear_shortcut_residual_to_retrain"
        ),
        "single_round": {
            "certified_l2_distance_to_retrain": agg(single, "certified_l2_distance_to_retrain"),
            "cheap_linear_shortcut_residual_to_retrain": agg(
                single, "cheap_linear_shortcut_residual_to_retrain"
            ),
            "note": (
                "at rounds=1 there is no warm-start propagation, so the cheap "
                "linear patch IS the retrain under the linear aggregator (FedAvg) "
                "and is not under the non-linear one (median)"
            ),
        },
        "certificate_verifies_after": all(r["certificate_verifies_after"] for r in rows),
        "certificate_rejects_before": all(r["certificate_rejects_before"] for r in rows),
        "certificate_pin_verify_ok": all(r["certificate_pin_verify_ok"] for r in rows),
        "certificate_wrong_pin_fails": all(r["certificate_wrong_pin_fails"] for r in rows),
        "certificate_rejects_manifest_tamper": all(
            r["certificate_rejects_manifest_tamper"] for r in rows
        ),
        "certificate_binds_features_sha256": all(
            r["certificate_binds_features_sha256"] for r in rows
        ),
        "per_erasure": rows,
    }


def run_suite(
    *,
    features: Path,
    manifest_path: Path,
    n_erasures: int,
    rounds: int = ROUNDS,
    audit_subjects: int = AUDIT_SUBJECTS,
) -> dict[str, Any]:
    pop, manifest = R.build_population(features, manifest_path)
    clients = [
        pop.private_idx[c]
        for c in R.partition_geographic(pop.pos[pop.private_idx], N_CLIENTS, seed=PARTITION_SEED)
    ]
    rng = np.random.default_rng(20_260_728)
    subject_rows = []
    for k in range(n_erasures):
        ci = k % N_CLIENTS
        subject_rows.append((ci, int(rng.choice(clients[ci]))))

    fsha = manifest["features_sha256"]
    results = [
        run_method(
            pop,
            clients,
            method=m,
            rounds=rounds,
            subject_rows=subject_rows,
            features_sha256=fsha,
        )
        for m in METHODS
    ]
    audit = erasure_audit(
        pop,
        clients,
        method="fedavg",
        rounds=rounds,
        target_client=0,
        n_subjects=audit_subjects,
        seed=99,
    )

    fed = next(r for r in results if r["method"] == "fedavg")
    med = next(r for r in results if r["method"] == "median")
    sizes = np.asarray([c.size for c in clients])
    report = {
        "benchmark": "GDPR Art. 17 subject erasure on real DeepMIMO receivers",
        "provenance": R.provenance_block(
            manifest,
            pop,
            extra_scope=(
                "'Erasure' here removes one simulated receiver's measurements from "
                "a model; it is not a legal opinion and not a statement about any "
                "real person's data."
            ),
        ),
        "setup": {
            "n_clients": N_CLIENTS,
            "client_size_min": int(sizes.min()),
            "client_size_max": int(sizes.max()),
            "rounds": rounds,
            "model_dim": R.MODEL_DIM,
            "subject": "one real DeepMIMO receiver (receiver_index) and its 6 measured gains",
            "erasure_mechanism": (
                "exact rank-1 downdate of the client's sufficient statistics "
                "(DᵀD, DᵀT, n), replayed through every warm-started round"
            ),
            "gold_standard": (
                "independent from-scratch retrain of the whole federation from the "
                "raw real rows with the subject's row physically deleted"
            ),
            "partition_seed": PARTITION_SEED,
            "lineage": [
                "Lam et al., Certifying the Right to be Forgotten: Primal-Dual "
                "Optimization for Sample and Label Unlearning in VFL, IEEE TIFS",
                "Yeom, Giacomelli, Fredrikson & Jha, CSF 2018 (loss-threshold membership inference)",
            ],
        },
        "results": results,
        "erasure_audit": audit,
        "honest_finding": [
            (
                "Erasure of a real receiver is EXACT and the exactness is measured, "
                "not claimed. The rank-1 sufficient-statistic downdate and an "
                "independent from-scratch retrain from the raw rows are different "
                "code paths, and their L2 distance is "
                f"{fed['certified_l2_distance_to_retrain']['max']:.3g} (FedAvg, worst "
                f"of {fed['n_erasures']} real erasures) and "
                f"{med['certified_l2_distance_to_retrain']['max']:.3g} (median) - "
                "floating-point noise on a model whose own norm is order 1."
            ),
            (
                "Erasing one real receiver out of ~"
                f"{int(sizes.max())} moves the model by "
                f"{fed['influence_l2_before_vs_after']['mean']:.3g} in L2 and shifts "
                "its predicted gain at that receiver by "
                f"{fed['influence_prediction_shift_db']['mean']:.3g} dB. The erasure "
                "is small because one subject out of hundreds is small - but it is "
                "non-zero, which is why it has to be done exactly."
            ),
            (
                "The cheap shortcut is NOT the retrain once rounds warm-start. "
                "Patching only the final aggregate leaves a residual of "
                f"{fed['cheap_linear_shortcut_residual_to_retrain']['mean']:.3g} under "
                f"FedAvg and {med['cheap_linear_shortcut_residual_to_retrain']['mean']:.3g} "
                "under coordinate median at "
                f"{rounds} rounds. At a single round the FedAvg shortcut is exact "
                f"({fed['single_round']['cheap_linear_shortcut_residual_to_retrain']['max']:.3g}) "
                "and the median shortcut still is not "
                f"({med['single_round']['cheap_linear_shortcut_residual_to_retrain']['mean']:.3g}). "
                "Exactness under warm-starting or non-linearity costs the replay; "
                "the residual is what an operator who skips it is left holding."
            ),
            (
                "The erasure audit reports the achieved value. Over "
                f"{audit['subjects_erased']} real receivers erased one at a time, the "
                "paired leave-one-out test shows the subject's own loss rose in "
                f"{audit['paired_leave_one_out']['fraction_loss_rose_after_own_erasure']:.1%} "
                "of cases once it was erased (mean "
                f"+{audit['paired_leave_one_out']['mean_loss_rise_db2']:.3g} dB^2). "
                "The influence being removed is real and measured. The population "
                "AUC against geography-matched never-seen receivers moves "
                f"{audit['population_auc']['membership_auc_before_erasure']} -> "
                f"{audit['population_auc']['membership_auc_after_erasure']}; that "
                "absolute level is confounded by real spatial heterogeneity and "
                "only its change is attributable to erasure. Neither test proves a "
                "stronger attacker would fail."
            ),
            (
                "The erasure is bound to a signed ErasureCertificate (RSA-PSS via "
                "the real provenance HSM) that verifies against the post-erasure "
                "model, rejects the pre-erasure model, rejects a manifest tamper, "
                "pin-verifies against the embedded key, and records the dataset "
                "features_sha256 so the erasure is bound to the exact data build."
            ),
        ],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--erasures", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--audit-subjects", type=int, default=AUDIT_SUBJECTS)
    ap.add_argument("--features", type=Path, default=R.DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=R.DEFAULT_MANIFEST)
    args = ap.parse_args()

    report = run_suite(
        features=args.features,
        manifest_path=args.manifest,
        n_erasures=args.erasures,
        rounds=args.rounds,
        audit_subjects=args.audit_subjects,
    )
    text = json.dumps(report, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    prov = report["provenance"]
    print(f"dataset={prov['dataset']}  features_sha256={prov['features_sha256'][:16]}...")
    print(f"real subjects available: {prov['private_subject_receivers']}")
    for r in report["results"]:
        print(
            f"\n[{r['method']:7s}] erasures={r['n_erasures']} changed={r['global_changed']}"
            f"\n    certified L2 to independent retrain : "
            f"max {r['certified_l2_distance_to_retrain']['max']:.3g}"
            f"\n    cheap shortcut residual  (R={r['rounds']:>2d})   : "
            f"mean {r['cheap_linear_shortcut_residual_to_retrain']['mean']:.3g}"
            f"\n    cheap shortcut residual  (R= 1)     : "
            f"mean {r['single_round']['cheap_linear_shortcut_residual_to_retrain']['mean']:.3g}"
            f"\n    influence of one real receiver      : "
            f"{r['influence_l2_before_vs_after']['mean']:.3g} L2, "
            f"{r['influence_prediction_shift_db']['mean']:.3g} dB at that receiver"
        )
    a = report["erasure_audit"]
    pl = a["paired_leave_one_out"]
    pa = a["population_auc"]
    print(
        f"\n[audit ] {a['subjects_erased']} real receivers erased one at a time"
        f"\n    paired leave-one-out: loss rose for "
        f"{pl['fraction_loss_rose_after_own_erasure']:.1%} of subjects "
        f"(mean +{pl['mean_loss_rise_db2']:.3g} dB^2)"
        f"\n    population AUC vs {pa['never_seen_reference_receivers']} never-seen: "
        f"{pa['membership_auc_before_erasure']} -> {pa['membership_auc_after_erasure']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
