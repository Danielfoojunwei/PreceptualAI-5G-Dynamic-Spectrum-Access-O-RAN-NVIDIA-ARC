"""Data-side poisoning & backdoor attacks on federated DSA clients.

The existing :mod:`horizon_ric.spectrum.attacks` module crafts malicious *model
update vectors* (ALIE / Fang) — it attacks the **aggregation** step. This module
attacks the layer below it: the malicious client's **local training data and
training signal**, before any Q-table is ever uploaded. These are the classic
*data-poisoning* / *backdoor* / *free-rider* threats from the federated-learning
security literature, realised concretely on the DSA tabular-Q clients:

1. :func:`train_reward_poisoned_q` — **label / reward poisoning.** The malicious
   client trains on a corrupted reward signal: the environment's reward sign is
   flipped (so the agent learns that *colliding* and *clashing with the primary
   user* are good and that successful transmission is bad). The resulting local
   Q-table is adversarial — a denial-of-spectrum policy learned, not crafted.
   This is the tabular-RL instance of training-set label-flipping
   (Biggio, Nelson & Laskov, "Poisoning Attacks against Support Vector
   Machines", ICML 2012; Tolpegin et al., "Data Poisoning Attacks Against
   Federated Learning Systems", ESORICS 2020).

2. :func:`train_backdoor_q` — **backdoor / trigger attack** (Bagdasaryan,
   Veit, Hua, Estrin & Shmatikov, "How To Backdoor Federated Learning",
   AISTATS 2020). The malicious client first learns a *normal* DSA policy
   (so its update looks benign on the main task and survives robust
   aggregation), then **overwrites exactly one trigger state's Q-row** so that,
   when the global policy later observes that specific sensing snapshot, every
   SU picks an attacker-chosen channel (always-collide / out-of-band). The
   policy behaves normally on every other state — this is what makes the
   backdoor *stealthy* and is why robust aggregation, which screens whole-vector
   outliers, may not remove a sparse trigger edit (the honest result this suite
   measures).

3. :func:`free_rider_update` — **free-rider attack** (Fraboni, Vidal & Lorenzi,
   "Free-rider Attacks on Model Aggregation in Federated Learning", AISTATS
   2021). A lazy/selfish client that does NO local work: it returns the (stale)
   broadcast global model plus a little Gaussian noise, harvesting the shared
   model while contributing nothing and slightly perturbing the aggregate.

All numpy / pure Python. No torch. The attacks are real (they corrupt the actual
training loop / update), not mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv, n_actions, n_states
from horizon_ric.spectrum.federated_q import QLearnConfig, _greedy_action


# ---------------------------------------------------------------------------
# 1. Label / reward poisoning
# ---------------------------------------------------------------------------
def _poison_reward(reward: float, info: dict, cfg: DSAConfig) -> float:
    """Flip the training signal a malicious client learns from.

    A successful, non-colliding transmission (the thing a good DSA policy is
    rewarded for) becomes *negative*; a collision / PU-clash (the things a good
    policy avoids) becomes *positive*. An agent trained on this signal learns
    the inverted objective — a denial-of-spectrum policy that seeks collisions.
    """
    if info.get("success"):
        return -cfg.reward_success
    if info.get("collision"):
        return -cfg.reward_collision  # i.e. +0.5 — reward the collision
    if info.get("pu_clash"):
        return -cfg.reward_pu_clash  # i.e. +0.5 — reward clashing with the PU
    return cfg.reward_idle


def train_reward_poisoned_q(
    env: DSAEnv,
    cfg: QLearnConfig,
    *,
    q_init: Optional[np.ndarray] = None,
    seed: int = 0,
    flip_prob: float = 1.0,
) -> np.ndarray:
    """Local Q-learning on a malicious client with a poisoned reward signal.

    Identical to :func:`horizon_ric.spectrum.federated_q.train_local_q` except
    each step's reward is replaced (with probability ``flip_prob``) by the
    inverted signal from :func:`_poison_reward`. ``flip_prob < 1`` models a
    partially-poisoned / stealthier client. Returns the adversarial Q-table.

    Reward/label poisoning of the local training signal — Biggio et al. (2012);
    Tolpegin et al., ESORICS 2020.
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
            nxt, reward, done, info = env.step(action)
            if rng.random() < flip_prob:
                reward = _poison_reward(reward, info, env.cfg)
            best_next = float(np.max(q[nxt]))
            q[state, action] += cfg.alpha * (
                reward + cfg.gamma * best_next - q[state, action]
            )
            state = nxt
    return q


# ---------------------------------------------------------------------------
# 2. Backdoor / trigger attack (Bagdasaryan et al., AISTATS 2020)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackdoorSpec:
    """The attacker's backdoor: on ``trigger_state`` force ``target_channel``.

    ``target_channel`` is an *action index* in ``[0, n_actions)``. A within-band
    channel index causes an always-collide attack (every SU stampedes the same
    sub-band on the trigger). A channel index can also be chosen so the pipeline
    maps it to a carrier the attacker wants — but the Shield, downstream, bounds
    whatever channel is emitted to legal spectrum (the guarantee this suite
    measures). ``boost`` is how strongly the trigger row favours the target.
    """

    trigger_state: int
    target_channel: int
    boost: float = 50.0


def default_backdoor(n_channels: int, *, trigger_state: int = 0) -> BackdoorSpec:
    """A canonical always-collide backdoor: force channel 0 on the trigger state.

    Every SU sharing the backdoored global policy will pick channel 0 whenever it
    senses ``trigger_state`` → a forced collision storm on exactly that state,
    while every other state behaves normally (stealth).
    """
    return BackdoorSpec(trigger_state=trigger_state, target_channel=0, boost=50.0)


def stamp_backdoor(q: np.ndarray, spec: BackdoorSpec) -> np.ndarray:
    """Overwrite one trigger-state Q-row so the greedy action is the target.

    Returns a copy with ``q[trigger_state]`` dominated by ``target_channel``.
    Only ONE row is altered — the sparse edit that makes the backdoor stealthy
    against whole-vector robust screening.
    """
    out = q.copy()
    s = spec.trigger_state
    out[s, :] = 0.0
    out[s, spec.target_channel] = spec.boost
    return out


def train_backdoor_q(
    env: DSAEnv,
    cfg: QLearnConfig,
    spec: BackdoorSpec,
    *,
    q_init: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Malicious client that learns the main task, then stamps the backdoor.

    The client trains a *normal* DSA Q-table (so on every non-trigger state its
    update is indistinguishable from an honest client's — this is the
    "train-and-scale"/constrain-and-scale idea of Bagdasaryan et al.), then
    edits only the trigger row. The whole-vector distance to honest updates
    stays small (the backdoor is sparse), which is exactly why Krum/median can
    fail to screen it out.
    """
    from horizon_ric.spectrum.federated_q import train_local_q

    benign = train_local_q(env, cfg, q_init=q_init, seed=seed)
    return stamp_backdoor(benign, spec)


def backdoor_success_rate(
    q: np.ndarray,
    spec: BackdoorSpec,
    *,
    dsa_cfg: Optional[DSAConfig] = None,
    n_users: Optional[int] = None,
    seed: int = 4321,
    tau: float = 0.5,
    trials: int = 64,
) -> float:
    """Fraction of SUs that pick the attacker channel ON the trigger state.

    Deploys the (possibly backdoored) shared policy the way the pipeline does —
    Boltzmann sampling per SU at temperature ``tau`` (see
    :func:`horizon_ric.spectrum.federated_q.evaluate_policy`) — but holds the
    observed state fixed at ``spec.trigger_state`` and measures how often the
    sampled action equals ``spec.target_channel``. A clean policy lands near
    chance; a backdoored policy lands near 1.0 on the trigger only.
    """
    from horizon_ric.spectrum.federated_q import _softmax_action

    dsa_cfg = dsa_cfg or DSAConfig()
    n_users = n_users if n_users is not None else dsa_cfg.n_users
    rng = np.random.default_rng(seed)
    hits = 0
    total = trials * n_users
    for _ in range(trials):
        for _u in range(n_users):
            a = _softmax_action(q, spec.trigger_state, tau, rng)
            if a == spec.target_channel:
                hits += 1
    return hits / max(1, total)


# ---------------------------------------------------------------------------
# 3. Free-rider attack (Fraboni et al., AISTATS 2021)
# ---------------------------------------------------------------------------
def free_rider_update(
    global_q: Optional[np.ndarray],
    shape: tuple[int, int],
    *,
    sigma: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """A free-rider's submission: the stale global model + small Gaussian noise.

    The client does ZERO local training. It returns the broadcast global Q-table
    (or zeros on the first round, before any model exists) plus i.i.d. Gaussian
    perturbation of std ``sigma`` so the server cannot trivially detect an exact
    duplicate. It harvests the shared model and contributes no learning —
    Fraboni, Vidal & Lorenzi (AISTATS 2021).
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, dtype=np.float64) if global_q is None else global_q.astype(np.float64).copy()
    if base.shape != shape:
        raise ValueError(f"global_q shape {base.shape} != expected {shape}")
    return base + rng.normal(0.0, sigma, size=shape)


# ---------------------------------------------------------------------------
# Build malicious data-side Q-tables for a federated round
# ---------------------------------------------------------------------------
DATA_ATTACKS = ("reward_poison", "backdoor", "free_rider")


def build_data_poison_qs(
    *,
    attack: str,
    n_malicious: int,
    dsa_cfg: DSAConfig,
    q_cfg: QLearnConfig,
    global_q: Optional[np.ndarray] = None,
    backdoor: Optional[BackdoorSpec] = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Construct ``n_malicious`` data-poisoned client Q-tables for ``attack``.

    Each malicious client owns its own :class:`DSAEnv` (its own seed), exactly as
    an honest client does, so the only difference is the *data-side* corruption.
    Returns the list of uploaded Q-tables (same shape as honest clients').
    """
    if n_malicious <= 0 or attack == "none":
        return []
    ns, na = n_states(dsa_cfg.n_channels), n_actions(dsa_cfg.n_channels)
    shape = (ns, na)
    out: list[np.ndarray] = []
    for m in range(n_malicious):
        cseed = seed * 1000 + 9000 + m + 1
        if attack == "reward_poison":
            env = DSAEnv(cfg=dsa_cfg, seed=cseed)
            out.append(
                train_reward_poisoned_q(env, q_cfg, q_init=global_q, seed=cseed)
            )
        elif attack == "backdoor":
            spec = backdoor or default_backdoor(dsa_cfg.n_channels)
            env = DSAEnv(cfg=dsa_cfg, seed=cseed)
            out.append(train_backdoor_q(env, q_cfg, spec, q_init=global_q, seed=cseed))
        elif attack == "free_rider":
            out.append(free_rider_update(global_q, shape, sigma=1e-3, seed=cseed))
        else:
            raise ValueError(f"unknown data attack {attack!r}")
    return out


def federated_dsa_round_data_poison(
    *,
    attack: str,
    n_clients: int = 10,
    n_malicious: int = 3,
    dsa_cfg: Optional[DSAConfig] = None,
    q_cfg: Optional[QLearnConfig] = None,
    method: str = "median",
    global_q: Optional[np.ndarray] = None,
    backdoor: Optional[BackdoorSpec] = None,
    seed: int = 0,
):
    """One federated DSA round where the malicious clients are DATA-poisoned.

    Mirrors :func:`horizon_ric.spectrum.federated_q.federated_dsa_round` but the
    malicious clients run a data-side attack (reward poisoning / backdoor /
    free-rider) on their *local* training instead of crafting an update vector.
    Honest clients train normally. The server aggregates with ``method`` (a
    robust aggregator or plain ``fedavg``). Returns a
    :class:`~horizon_ric.spectrum.federated_q.FederatedDSAResult`.
    """
    from horizon_ric.spectrum.federated_q import (
        FederatedDSAResult,
        _aggregate_q,
        train_local_q,
    )

    dsa_cfg = dsa_cfg or DSAConfig()
    q_cfg = q_cfg or QLearnConfig()
    ns, na = n_states(dsa_cfg.n_channels), n_actions(dsa_cfg.n_channels)
    shape = (ns, na)
    n_honest = n_clients - n_malicious

    honest_qs: list[np.ndarray] = []
    for c in range(n_honest):
        env = DSAEnv(cfg=dsa_cfg, seed=seed * 1000 + c + 1)
        honest_qs.append(
            train_local_q(env, q_cfg, q_init=global_q, seed=seed * 1000 + c + 1)
        )

    malicious_qs = build_data_poison_qs(
        attack=attack,
        n_malicious=n_malicious,
        dsa_cfg=dsa_cfg,
        q_cfg=q_cfg,
        global_q=global_q,
        backdoor=backdoor,
        seed=seed,
    )
    client_qs = honest_qs + malicious_qs

    f = max(1, n_malicious) if method in ("krum", "trimmed_mean") else n_malicious
    agg_flat, selected = _aggregate_q(
        client_qs, method=method, f=f, secure_agg=False
    )
    new_global = agg_flat.reshape(shape)

    return FederatedDSAResult(
        global_q=new_global,
        method=method,
        n_clients=n_clients,
        n_malicious=n_malicious,
        secure_agg=False,
        client_shapes=shape,
        selected_index=selected,
    )


__all__ = [
    "BackdoorSpec",
    "DATA_ATTACKS",
    "backdoor_success_rate",
    "build_data_poison_qs",
    "default_backdoor",
    "federated_dsa_round_data_poison",
    "free_rider_update",
    "stamp_backdoor",
    "train_backdoor_q",
    "train_reward_poisoned_q",
]
