"""Certified federated unlearning — excise a poisoning client's contribution.

Robust aggregation (:mod:`horizon_ric.federated.robust`) only *bounds* a
poisoning client's pull on the shared model; it never *removes* the influence
that already leaked in. Our own adversarial campaign showed the consequence: a
sparse backdoor survives coordinate-median at its breakdown point
(``benchmarks/results/dsa_poison_suite.json``; ``docs/THREAT_MODEL.md`` §7).
Once the aggregator's outlier screen attributes a malicious client, the missing
trust mechanism is the ability to *remove* that client's contribution from the
trained federated DSA policy, *prove* the removal, and *verify* a planted
backdoor is gone.

This module implements that, torch-free, on the tabular federated DSA model, in
the lineage of the NTU/DTC federated-unlearning research:

* Liu, Ye, Jiang, Shen, Guo, Tjuawinata & Lam, "Privacy-Preserving Federated
  Unlearning with Certified Client Removal", arXiv:2404.09724 (2024) — the
  *Starfish* idea that the unlearning guarantee is a **bound on the distance
  between the unlearned model and a model retrained from scratch**.
* Liu, Jiang, Shen, Peng, Lam, Yuan & Liu, "A Survey on Federated Unlearning:
  Challenges, Methods, and Future Directions", ACM Computing Surveys (2024).
* Han, Zhu, Zhang, Huo & Zhou, "Vertical Federated Unlearning via Backdoor
  Certification", arXiv:2412.11476 (2024) — the idea of using a **planted
  backdoor as a verification probe** that unlearning actually happened.

Two mechanisms, both real (no torch, no mocks):

* :func:`retrain_from_scratch` — the gold standard: re-run the federated rounds
  with the flagged client(s) excluded. Exact, but pays the full training cost.
* :func:`efficient_unlearn` — replay the *cached* per-round client uploads with
  the flagged client(s) removed and re-aggregate, WITHOUT re-running any local
  training. Its error vs. the gold standard is **certified**: the certificate
  records the measured L2 distance to the retrained model (the Starfish bound)
  plus a backdoor-probe verification (trigger success before vs. after).

The unlearned model + an unlearning manifest are signed with the existing
:mod:`horizon_ric.provenance.signing` HSM key, so an auditor can verify *that* a
named contribution was removed, *when*, by which method, and to what certified
bound. The Decision Safety Shield remains the runtime backstop throughout — at
the detection breakdown point (where a stealthy poisoner cannot be attributed)
unlearning has nothing to target, and the deterministic Shield is what still
bounds the emitted action to legal spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from horizon_ric.federated import robust
from horizon_ric.provenance.signing import (
    ModelProvenance,
    manifest_digest,
    sha256_hex,
    sign_model,
    verify_model,
)
from horizon_ric.spectrum.data_poison import (
    BackdoorSpec,
    backdoor_success_rate,
    train_backdoor_q,
    train_reward_poisoned_q,
)
from horizon_ric.spectrum.dsa_env import DSAConfig, n_actions, n_states
from horizon_ric.spectrum.federated_q import (
    QLearnConfig,
    craft_poison_q,
    train_local_q,
)

# Roles a federated client can play in a recorded training run.
HONEST = "honest"
BACKDOOR = "backdoor"
REWARD_POISON = "reward_poison"
QTABLE_TARGET = "qtable_target"


@dataclass(frozen=True)
class ClientSpec:
    """One federated client: a stable id, an env seed, and a role.

    ``role`` selects the local-update function (honest training vs. a data-side
    poisoning attack). ``backdoor`` is required when ``role == BACKDOOR``.
    """

    client_id: str
    seed: int
    role: str = HONEST
    backdoor: Optional[BackdoorSpec] = None


@dataclass
class RoundUploads:
    """The Q-tables every client uploaded in one federated round (cached)."""

    client_ids: list[str]
    uploads: list[np.ndarray]


@dataclass
class FederatedTrace:
    """A full recorded federated training run — enough to retrain or replay.

    ``rounds`` caches each round's per-client uploads (for the cheap replay-based
    :func:`efficient_unlearn`); ``clients`` + the configs let
    :func:`retrain_from_scratch` re-run honest-only training (the gold standard).
    """

    clients: list[ClientSpec]
    rounds: list[RoundUploads]
    shape: tuple[int, int]
    method: str
    dsa_cfg: DSAConfig
    q_cfg: QLearnConfig
    base_seed: int


# ---------------------------------------------------------------------------
# Local update + aggregation primitives
# ---------------------------------------------------------------------------
def _local_update(
    spec: ClientSpec,
    global_q: Optional[np.ndarray],
    dsa_cfg: DSAConfig,
    q_cfg: QLearnConfig,
    shape: tuple[int, int],
) -> np.ndarray:
    """Produce one client's uploaded Q-table given the broadcast global policy."""
    from horizon_ric.spectrum.dsa_env import DSAEnv

    if spec.role == QTABLE_TARGET:
        # Model-poisoning: a crafted single-channel Q-table (no env needed).
        return craft_poison_q(shape, target_channel=0, magnitude=50.0)

    env = DSAEnv(cfg=dsa_cfg, seed=spec.seed)
    if spec.role == HONEST:
        return train_local_q(env, q_cfg, q_init=global_q, seed=spec.seed)
    if spec.role == REWARD_POISON:
        return train_reward_poisoned_q(env, q_cfg, q_init=global_q, seed=spec.seed)
    if spec.role == BACKDOOR:
        if spec.backdoor is None:
            raise ValueError(f"client {spec.client_id!r} role=backdoor needs a BackdoorSpec")
        return train_backdoor_q(env, q_cfg, spec.backdoor, q_init=global_q, seed=spec.seed)
    raise ValueError(f"unknown client role {spec.role!r}")


def _aggregate(uploads: list[np.ndarray], method: str, *, f: int) -> np.ndarray:
    """Aggregate flattened client uploads with a robust aggregator.

    Mirrors the server step in :mod:`horizon_ric.spectrum.federated_q` (minus the
    secure-aggregation path, which is orthogonal to unlearning).
    """
    flats = [u.ravel().astype(np.float64) for u in uploads]
    if method == "fedavg":
        return robust.fedavg(flats)
    if method == "median":
        return robust.coordinate_median(flats)
    if method == "trimmed_mean":
        return robust.trimmed_mean(flats, beta=max(1, f))
    if method == "krum":
        return robust.krum(flats, f=max(0, f)).aggregate
    raise ValueError(f"unknown aggregation method {method!r}")


def run_federated_training(
    clients: list[ClientSpec],
    *,
    rounds: int,
    method: str,
    dsa_cfg: Optional[DSAConfig] = None,
    q_cfg: Optional[QLearnConfig] = None,
    n_malicious: Optional[int] = None,
    base_seed: int = 0,
) -> tuple[np.ndarray, FederatedTrace]:
    """Run ``rounds`` of warm-started federated DSA Q-learning, recording a trace.

    Returns the final global Q-table and a :class:`FederatedTrace` that caches
    every round's per-client uploads (for replay-based unlearning) alongside the
    client specs and configs (for retrain-from-scratch).
    """
    dsa_cfg = dsa_cfg or DSAConfig()
    q_cfg = q_cfg or QLearnConfig()
    shape = (n_states(dsa_cfg.n_channels), n_actions(dsa_cfg.n_channels))
    f = n_malicious if n_malicious is not None else sum(c.role != HONEST for c in clients)

    global_q: Optional[np.ndarray] = None
    recorded: list[RoundUploads] = []
    for _r in range(rounds):
        uploads = [_local_update(c, global_q, dsa_cfg, q_cfg, shape) for c in clients]
        recorded.append(
            RoundUploads(client_ids=[c.client_id for c in clients],
                         uploads=[u.copy() for u in uploads])
        )
        global_q = _aggregate(uploads, method, f=f).reshape(shape)

    assert global_q is not None  # rounds >= 1
    trace = FederatedTrace(
        clients=list(clients), rounds=recorded, shape=shape, method=method,
        dsa_cfg=dsa_cfg, q_cfg=q_cfg, base_seed=base_seed,
    )
    return global_q, trace


# ---------------------------------------------------------------------------
# Detection (the trigger for unlearning)
# ---------------------------------------------------------------------------
def detect_outliers(uploads: list[np.ndarray], n_flagged: int) -> list[int]:
    """Flag the ``n_flagged`` uploads furthest (L2) from the coordinate median.

    A real, simple Byzantine detector that catches *large-norm* poisoning
    (crafted Q-tables, reward poisoning). It does NOT reliably catch a *sparse*
    backdoor — the whole-vector distance of a one-row trigger edit is tiny, which
    is exactly why the backdoor survives robust aggregation in the first place.
    Attribution of a stealthy backdoor needs an independent signal; this function
    is honest about flagging only what whole-vector screening can see.
    """
    mat = np.stack([u.ravel().astype(np.float64) for u in uploads])
    med = np.median(mat, axis=0)
    dists = np.linalg.norm(mat - med, axis=1)
    order = np.argsort(dists)[::-1]
    return [int(i) for i in order[: max(0, n_flagged)]]


# ---------------------------------------------------------------------------
# Unlearning mechanisms
# ---------------------------------------------------------------------------
def _kept(client_ids: list[str], items: list[np.ndarray], removed: set[str]) -> list[np.ndarray]:
    return [u for cid, u in zip(client_ids, items) if cid not in removed]


def retrain_from_scratch(
    trace: FederatedTrace,
    removed_client_ids: list[str],
    *,
    method: Optional[str] = None,
) -> np.ndarray:
    """Gold-standard unlearning: re-run training with the flagged clients gone.

    This is the model that would have existed had the removed client(s) never
    participated. Exact, but pays the full training cost — the reference the
    certified bound measures against.
    """
    removed = set(removed_client_ids)
    kept = [c for c in trace.clients if c.client_id not in removed]
    if not kept:
        raise ValueError("cannot retrain: every client was flagged for removal")
    global_q, _ = run_federated_training(
        kept, rounds=len(trace.rounds), method=method or trace.method,
        dsa_cfg=trace.dsa_cfg, q_cfg=trace.q_cfg, n_malicious=0, base_seed=trace.base_seed,
    )
    return global_q


def efficient_unlearn(
    trace: FederatedTrace,
    removed_client_ids: list[str],
    *,
    method: Optional[str] = None,
) -> np.ndarray:
    """Cheap unlearning: re-aggregate the final round's *kept* cached uploads.

    No local re-training is run — the cached honest uploads are re-aggregated
    with the flagged client(s) dropped. For FedAvg on a single round this is
    exact; across warm-started rounds it is an approximation, because the kept
    clients' uploads were themselves trained against the (poisoned) global. The
    residual that leaves is precisely what :func:`certify_unlearning` measures as
    the distance to :func:`retrain_from_scratch`.
    """
    removed = set(removed_client_ids)
    method = method or trace.method
    last = trace.rounds[-1]
    kept_uploads = _kept(last.client_ids, last.uploads, removed)
    if not kept_uploads:
        raise ValueError("cannot unlearn: every client in the final round was flagged")
    return _aggregate(kept_uploads, method, f=0).reshape(trace.shape)


# ---------------------------------------------------------------------------
# Certification (bound + backdoor probe + signed provenance + audit binding)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UnlearningCertificate:
    """A signed, auditable record that a contribution was unlearned.

    Binds the unlearned model bytes to the unlearning event via the existing
    RSA-PSS provenance signature, and records the Starfish-style certified bound
    (L2 distance to retrain-from-scratch) and the backdoor-probe verification.
    """

    removed_client_ids: list[str]
    method: str
    unlearn_mechanism: str  # "efficient_unlearn" | "retrain_from_scratch"
    model_sha256_before: str
    model_sha256_after: str
    certified_l2_distance_to_retrain: float
    backdoor_success_before: Optional[float]
    backdoor_success_after: Optional[float]
    backdoor_success_retrain: Optional[float]
    created_at: str
    manifest: dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ModelProvenance] = None

    def verify(self, unlearned_q: np.ndarray, *, trusted_public_key_der: Optional[bytes] = None) -> bool:
        """Verify the certificate binds to ``unlearned_q``.

        Checks (a) the after-hash matches the bytes, (b) the manifest digest
        matches the signed provenance, and (c) the RSA-PSS signature verifies.
        """
        weights = np.ascontiguousarray(unlearned_q, dtype=np.float64).tobytes()
        if sha256_hex(weights) != self.model_sha256_after:
            return False
        if self.provenance is None:
            return False
        if manifest_digest(self.manifest) != self.provenance.manifest_sha256:
            return False
        return verify_model(weights, self.provenance, trusted_public_key_der=trusted_public_key_der)


def certify_unlearning(
    *,
    poisoned_q: np.ndarray,
    unlearned_q: np.ndarray,
    retrained_q: np.ndarray,
    removed_client_ids: list[str],
    method: str,
    hsm: Any,
    unlearn_mechanism: str = "efficient_unlearn",
    backdoor: Optional[BackdoorSpec] = None,
    dsa_cfg: Optional[DSAConfig] = None,
    trainer_id: str = "horizon-unlearning-service",
    key_label: str = "unlearning-signer",
) -> UnlearningCertificate:
    """Build a signed :class:`UnlearningCertificate` for an unlearning operation.

    Computes the certified bound (``||unlearned - retrained||_2``) and, if a
    backdoor spec is given, the trigger-success rate on the poisoned, unlearned,
    and retrained models (the verification probe). Signs the unlearned bytes plus
    the unlearning manifest with the HSM provenance key.
    """
    before = np.ascontiguousarray(poisoned_q, dtype=np.float64).tobytes()
    after = np.ascontiguousarray(unlearned_q, dtype=np.float64).tobytes()
    distance = float(np.linalg.norm(unlearned_q.ravel() - retrained_q.ravel()))

    bd_before = bd_after = bd_retrain = None
    if backdoor is not None:
        bd_before = backdoor_success_rate(poisoned_q, backdoor, dsa_cfg=dsa_cfg)
        bd_after = backdoor_success_rate(unlearned_q, backdoor, dsa_cfg=dsa_cfg)
        bd_retrain = backdoor_success_rate(retrained_q, backdoor, dsa_cfg=dsa_cfg)

    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "event": "federated_unlearning",
        "removed_client_ids": sorted(removed_client_ids),
        "aggregation_method": method,
        "unlearn_mechanism": unlearn_mechanism,
        "model_sha256_before": sha256_hex(before),
        "model_sha256_after": sha256_hex(after),
        "certified_l2_distance_to_retrain": distance,
        "backdoor_success_before": bd_before,
        "backdoor_success_after": bd_after,
        "backdoor_success_retrain": bd_retrain,
        "created_at": created_at,
    }
    provenance = sign_model(
        after, trainer_id=trainer_id, training_manifest=manifest, hsm=hsm, key_label=key_label
    )
    return UnlearningCertificate(
        removed_client_ids=sorted(removed_client_ids),
        method=method,
        unlearn_mechanism=unlearn_mechanism,
        model_sha256_before=sha256_hex(before),
        model_sha256_after=sha256_hex(after),
        certified_l2_distance_to_retrain=distance,
        backdoor_success_before=bd_before,
        backdoor_success_after=bd_after,
        backdoor_success_retrain=bd_retrain,
        created_at=created_at,
        manifest=manifest,
        provenance=provenance,
    )


__all__ = [
    "HONEST",
    "BACKDOOR",
    "REWARD_POISON",
    "QTABLE_TARGET",
    "ClientSpec",
    "RoundUploads",
    "FederatedTrace",
    "run_federated_training",
    "detect_outliers",
    "retrain_from_scratch",
    "efficient_unlearn",
    "UnlearningCertificate",
    "certify_unlearning",
]
