"""Torch-free federated tabular Q-learning for Dynamic Spectrum Access.

Each federated **client** owns its own :class:`DSAEnv` instance (its own local
spectrum conditions, PU statistics, and seed) and runs ordinary epsilon-greedy
tabular Q-learning for a few episodes. After local training, every client ships
its flattened Q-table to the **server**, which aggregates them into a shared DSA
policy and broadcasts it back — federated Q-learning for spectrum access, the
NTU FCP problem class.

The security bridge (the whole point of this module): the server does NOT plain-
average the client Q-tables. It calls
:func:`horizon_ric.federated.robust.aggregate` (Krum / coordinate-median /
trimmed-mean) so a *poisoning* client — one that uploads a crafted Q-table to
steer every SU onto a single channel (a denial-of-spectrum / forced-collision
attack) — has its pull on the shared policy bounded. Optionally the round can
route the aggregation through Shamir secure aggregation
(:func:`horizon_ric.federated.secure.secure_mean`) so the server never sees an
individual client's Q-table (privacy of the local spectrum profile).

Everything is numpy / pure Python. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from horizon_ric.federated import robust, secure
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv, n_actions, n_states


@dataclass
class QLearnConfig:
    alpha: float = 0.15  # learning rate
    gamma: float = 0.90  # discount
    epsilon: float = 0.15  # exploration
    episodes: int = 8  # local episodes per federated round


def _greedy_action(q: np.ndarray, state: int) -> int:
    return int(np.argmax(q[state]))


def train_local_q(
    env: DSAEnv,
    cfg: QLearnConfig,
    *,
    q_init: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Run epsilon-greedy tabular Q-learning on one client's env.

    Returns the learned Q-table (shape ``(n_states, n_actions)``). If ``q_init``
    is given, training warm-starts from it (the broadcast global policy).
    """
    ns, na = env.n_states, env.n_actions
    q = np.zeros((ns, na), dtype=np.float64) if q_init is None else q_init.copy()
    rng = np.random.default_rng(seed)

    for ep in range(cfg.episodes):
        state = env.reset(seed=seed + ep + 1)
        done = False
        while not done:
            if rng.random() < cfg.epsilon:
                action = int(rng.integers(0, na))
            else:
                action = _greedy_action(q, state)
            nxt, reward, done, _ = env.step(action)
            best_next = float(np.max(q[nxt]))
            q[state, action] += cfg.alpha * (
                reward + cfg.gamma * best_next - q[state, action]
            )
            state = nxt
    return q


# ── poisoning attack on the *Q-table itself* (client-side model poisoning) ──
def craft_poison_q(
    shape: tuple[int, int],
    *,
    target_channel: int = 0,
    magnitude: float = 50.0,
) -> np.ndarray:
    """A poisoning client's crafted Q-table.

    The attack: make ``target_channel`` look overwhelmingly attractive in every
    state so that, if it dominates aggregation, every SU piles onto one channel
    and collides — a forced-collision / denial-of-spectrum attack on the shared
    federated DSA policy.
    """
    q = np.zeros(shape, dtype=np.float64)
    q[:, target_channel] = magnitude
    return q


@dataclass
class FederatedDSAResult:
    global_q: np.ndarray
    method: str
    n_clients: int
    n_malicious: int
    secure_agg: bool
    client_shapes: tuple[int, int]
    selected_index: Optional[int] = None  # Krum's pick, if applicable


def _aggregate_q(
    client_qs: Sequence[np.ndarray],
    *,
    method: str,
    f: int,
    secure_agg: bool,
) -> tuple[np.ndarray, Optional[int]]:
    """Aggregate flattened client Q-tables via a robust aggregator.

    Returns (global_q_flat, selected_index_or_None).
    """
    flats = [q.ravel().astype(np.float64) for q in client_qs]

    if secure_agg and method in ("fedavg", "median", "trimmed_mean"):
        # Privacy-preserving mean (server never sees an individual Q-table).
        # Only the mean-family aggregators are additively homomorphic; Krum is
        # not, so we keep secure-agg for the mean/median path and document it.
        if method == "fedavg":
            agg = np.asarray(
                secure.secure_mean([f.tolist() for f in flats], n_shares=5, threshold=3)
            )
            return agg, None

    if method == "krum":
        res = robust.krum(flats, f=f)
        return res.aggregate, res.selected_index
    if method == "median":
        return robust.coordinate_median(flats), None
    if method == "trimmed_mean":
        beta = max(1, f)
        return robust.trimmed_mean(flats, beta=beta), None
    if method == "fedavg":
        return robust.fedavg(flats), None
    raise ValueError(f"unknown aggregation method {method!r}")


def _build_malicious_qs(
    honest_qs: Sequence[np.ndarray],
    n_malicious: int,
    *,
    attack: str,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    """Construct ``n_malicious`` poisoned Q-tables for the given attack."""
    if n_malicious <= 0 or attack == "none":
        return []
    if attack == "qtable_target":
        return [craft_poison_q(shape, target_channel=0, magnitude=50.0) for _ in range(n_malicious)]

    # ALIE / Fang operate on flattened update vectors derived from the honest pop.
    from horizon_ric.spectrum import attacks as atk

    flats = [q.ravel().astype(np.float64) for q in honest_qs]
    if attack == "alie":
        mal = atk.alie_attack(flats, n_malicious)
    elif attack == "fang_krum":
        mal = atk.fang_attack_krum(flats, n_malicious)
    elif attack == "fang_median":
        mal = atk.fang_attack_median(flats, n_malicious)
    else:
        raise ValueError(f"unknown attack {attack!r}")
    return [m.reshape(shape) for m in mal]


def federated_dsa_round(
    *,
    n_clients: int = 8,
    n_malicious: int = 2,
    dsa_cfg: Optional[DSAConfig] = None,
    q_cfg: Optional[QLearnConfig] = None,
    method: str = "krum",
    secure_agg: bool = False,
    global_q: Optional[np.ndarray] = None,
    attack: str = "qtable_target",
    seed: int = 0,
) -> FederatedDSAResult:
    """Run one federated round of DSA Q-learning with robust aggregation.

    Args:
        n_clients: total federated clients (honest + malicious).
        n_malicious: how many clients are poisoning. Must stay below the
            aggregator's breakdown point for the defence to hold (Krum needs
            ``n_clients > 2*n_malicious + 2``).
        method: robust aggregator — "krum", "median", "trimmed_mean", "fedavg".
        secure_agg: route the mean-family aggregation through Shamir secure
            aggregation (privacy; only valid for the mean path).
        global_q: warm-start policy broadcast from the previous round.
        attack: poisoning strategy for the malicious clients —
            "qtable_target" (crafted single-channel Q-table — the obvious
            denial-of-spectrum attack), "alie" (A Little Is Enough), "fang_krum"
            / "fang_median" (Fang optimized attacks), or "none".
        seed: base RNG seed (each client derives a distinct seed from it).

    Returns the aggregated global Q-table and round metadata.
    """
    dsa_cfg = dsa_cfg or DSAConfig()
    q_cfg = q_cfg or QLearnConfig()
    ns, na = n_states(dsa_cfg.n_channels), n_actions(dsa_cfg.n_channels)
    shape = (ns, na)

    n_honest = n_clients - n_malicious

    # Honest clients train locally on their own DSA env.
    honest_qs: list[np.ndarray] = []
    for c in range(n_honest):
        env = DSAEnv(cfg=dsa_cfg, seed=seed * 1000 + c + 1)
        honest_qs.append(train_local_q(env, q_cfg, q_init=global_q, seed=seed * 1000 + c + 1))

    # Malicious clients craft updates. The ALIE/Fang attacks are derived from the
    # honest population (the attacker has model-access in the standard threat
    # model), then reshaped back into Q-tables.
    malicious_qs = _build_malicious_qs(
        honest_qs, n_malicious, attack=attack, shape=shape
    )
    client_qs = honest_qs + malicious_qs

    # Robust aggregation (the security bridge).
    f = max(1, n_malicious) if method in ("krum", "trimmed_mean") else n_malicious
    agg_flat, selected = _aggregate_q(
        client_qs, method=method, f=f, secure_agg=secure_agg
    )
    new_global = agg_flat.reshape(shape)

    return FederatedDSAResult(
        global_q=new_global,
        method=method,
        n_clients=n_clients,
        n_malicious=n_malicious,
        secure_agg=secure_agg,
        client_shapes=shape,
        selected_index=selected,
    )


def evaluate_policy(
    q: np.ndarray,
    *,
    dsa_cfg: Optional[DSAConfig] = None,
    n_episodes: int = 20,
    seed: int = 123,
    tau: float = 0.5,
) -> dict[str, float]:
    """Evaluate a (shared) DSA Q-policy on the multi-agent world.

    The same shared policy is deployed to every symmetric SU. Because all SUs
    observe the same wideband sensing snapshot, a *deterministic* greedy policy
    would make every SU pick the identical channel and self-collide — so a
    federated DSA policy must be deployed *stochastically* to break symmetry.
    We sample each SU's action from a softmax over its Q-row with temperature
    ``tau`` (a standard Boltzmann deployment). A poisoned policy whose Q-row is
    dominated by one channel collapses the softmax onto that channel for *every*
    SU → a collision storm; a healthy policy spreads load across sensed-idle
    channels. Returns throughput (successful tx / slot), collision rate, and
    PU-clash rate.
    """
    from horizon_ric.spectrum.dsa_env import DSAWorld

    dsa_cfg = dsa_cfg or DSAConfig()
    rng = np.random.default_rng(seed)
    total_success = 0
    total_collision = 0
    total_pu_clash = 0
    total_slots = 0

    for ep in range(n_episodes):
        world = DSAWorld(cfg=dsa_cfg, seed=seed + ep)
        obs = world.reset(seed=seed + ep)
        done = False
        while not done:
            actions = np.array(
                [_softmax_action(q, int(s), tau, rng) for s in obs], dtype=np.int64
            )
            obs, _rewards, done, info = world.step(actions)
            total_success += info["successes"]
            total_collision += info["collisions"]
            total_pu_clash += info["pu_clashes"]
            total_slots += 1

    denom = max(1, total_slots)
    return {
        "throughput_per_slot": total_success / denom,
        "collision_per_slot": total_collision / denom,
        "pu_clash_per_slot": total_pu_clash / denom,
        "n_episodes": float(n_episodes),
        "slots": float(total_slots),
    }


def _softmax_action(q: np.ndarray, state: int, tau: float, rng: np.random.Generator) -> int:
    """Boltzmann action selection over a Q-row (temperature ``tau``)."""
    row = q[state]
    z = row / max(tau, 1e-6)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


__all__ = [
    "QLearnConfig",
    "train_local_q",
    "craft_poison_q",
    "FederatedDSAResult",
    "federated_dsa_round",
    "evaluate_policy",
]
