"""Dynamic Spectrum Access (DSA) environment — the team's problem class.

This is a real, runnable Markov Decision Process for **multi-user dynamic
spectrum access**: ``n_users`` secondary users (SUs) each pick one of ``n_channels``
licensed channels every slot. Each channel is independently occupied by a
*primary user* (PU, the licensee) according to a two-state Gilbert–Elliott
Markov chain (busy ↔ idle). A secondary user that transmits on a channel
succeeds only if (a) the primary user is idle on that channel *and* (b) no other
secondary user picked the same channel in the same slot (otherwise the SUs
collide with each other). This is the canonical formulation behind federated
deep-RL DSA and the NTU FCP research line (Lam / Li Feng / Bowen Shen) on
secure/private spectrum sharing for terrestrial and LEO-IoT links.

Design goals:

* **numpy only** — no gym, no torch, no accelerator. Runs anywhere.
* **realistic wideband sensing** — at the start of each slot every SU performs
  energy-detection across the whole band and observes a (noisy) busy/idle bit
  per channel; the discrete state is that ``n_channels``-bit sensing vector. The
  SU does NOT see the other SUs' intentions, so mutual collisions are genuinely
  stochastic and the policy has to *spread load* statistically — this is what a
  federated DSA agent learns.
* **deterministic under a seed** — every transition is reproducible, which the
  evidence chain and the committed dataset depend on.

The reward is the count of *successful, non-colliding* transmissions, so a good
DSA policy learns to (i) transmit on a channel its sensing says is idle and
(ii) avoid piling every SU onto the single "best" channel.

Action / observation conventions (single-agent view used by the federated Q
agent — one agent controls one SU and treats the others as a stationary
background process):

    reset() -> obs:int                 # the n_channels-bit sensing snapshot
    step(action:int) -> (obs, reward, done, info)
        action ∈ [0, n_channels]       # 0..n_channels-1 = transmit on channel;
                                       # n_channels = "stay idle this slot".

``info`` exposes the full transition (pu_busy, collisions, success) for audit
and for the multi-agent ``DSAWorld`` used by the federated pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DSAConfig:
    """Configuration for a Dynamic Spectrum Access MDP."""

    n_channels: int = 6
    n_users: int = 3
    # Gilbert–Elliott primary-user occupancy: p_idle_to_busy (a PU shows up) and
    # p_busy_to_idle (the PU vacates). Stationary busy prob =
    # p_idle_to_busy / (p_idle_to_busy + p_busy_to_idle). Defaults ⇒ ~0.33 busy.
    p_idle_to_busy: float = 0.20
    p_busy_to_idle: float = 0.40
    # Probability a SU's per-channel sense bit is wrong (false busy/idle) —
    # models imperfect energy detection. Genuine partial observability.
    sense_error: float = 0.05
    max_steps: int = 200
    # Reward shaping (kept simple and honest): +1 per successful tx, penalties
    # for a SU-SU collision or for transmitting into an occupied PU channel.
    # The collision penalty is < the success reward so the learnable optimum is
    # to transmit on a sensed-idle channel rather than idle defensively.
    reward_success: float = 1.0
    reward_collision: float = -0.5
    reward_idle: float = 0.0
    reward_pu_clash: float = -0.5


def n_states(n_channels: int) -> int:
    """Discrete state count: one per possible ``n_channels``-bit sensing vector."""
    return 1 << n_channels


def n_actions(n_channels: int) -> int:
    return n_channels + 1  # +1 = idle (transmit on nothing this slot)


def encode_sensing(sense_bits: np.ndarray) -> int:
    """Pack a per-channel idle(1)/busy(0) sensing vector into a state id."""
    state = 0
    for i, b in enumerate(sense_bits):
        if b:
            state |= 1 << i
    return state


@dataclass
class _ChannelChain:
    """Per-channel Gilbert–Elliott PU occupancy state (0=idle, 1=busy)."""

    busy: np.ndarray  # shape (n_channels,), int8

    @classmethod
    def initial(cls, n_channels: int, cfg: "DSAConfig", rng: np.random.Generator) -> "_ChannelChain":
        # Initialise at the chain's stationary busy probability.
        p_busy = cfg.p_idle_to_busy / (cfg.p_idle_to_busy + cfg.p_busy_to_idle)
        return cls(busy=(rng.random(n_channels) < p_busy).astype(np.int8))

    def step(self, cfg: "DSAConfig", rng: np.random.Generator) -> None:
        roll = rng.random(self.busy.shape[0])
        new = self.busy.copy()
        idle_mask = self.busy == 0
        busy_mask = self.busy == 1
        new[idle_mask] = (roll[idle_mask] < cfg.p_idle_to_busy).astype(np.int8)
        new[busy_mask] = (~(roll[busy_mask] < cfg.p_busy_to_idle)).astype(np.int8)
        self.busy = new


def _sense_channels(pu_busy: np.ndarray, cfg: DSAConfig, rng: np.random.Generator) -> np.ndarray:
    """Noisy wideband sense: per-channel idle bit (1=idle) with sensing error."""
    true_idle = (pu_busy == 0).astype(np.int8)
    flips = rng.random(pu_busy.shape[0]) < cfg.sense_error
    sensed = true_idle.copy()
    sensed[flips] = 1 - sensed[flips]
    return sensed


@dataclass
class DSAEnv:
    """Single-agent view of a Dynamic Spectrum Access MDP.

    The controlled SU is agent 0. The other ``n_users-1`` SUs follow a fixed
    stochastic "background" policy (uniformly random channel choice over the
    channels), so the controlled agent faces a stationary multi-user collision
    process — exactly the environment a federated client trains against locally.
    """

    cfg: DSAConfig = field(default_factory=DSAConfig)
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _chain: _ChannelChain = field(init=False, repr=False)
    _sense: np.ndarray = field(init=False, repr=False)
    _t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    @property
    def n_states(self) -> int:
        return n_states(self.cfg.n_channels)

    @property
    def n_actions(self) -> int:
        return n_actions(self.cfg.n_channels)

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._chain = _ChannelChain.initial(self.cfg.n_channels, self.cfg, self._rng)
        self._sense = _sense_channels(self._chain.busy, self.cfg, self._rng)
        self._t = 0
        return encode_sensing(self._sense)

    def _background_choices(self) -> np.ndarray:
        k = self.cfg.n_users - 1
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        return self._rng.integers(0, self.cfg.n_channels, size=k)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        cfg = self.cfg
        action = int(action)
        if not (0 <= action <= cfg.n_channels):
            raise ValueError(f"action {action} out of range [0,{cfg.n_channels}]")

        # The action is taken against the CURRENT PU state (which the agent sensed
        # at the start of the slot); then the chain advances to the next slot.
        pu_busy = self._chain.busy
        others = self._background_choices()

        reward = cfg.reward_idle
        success = collided = pu_clash = False

        if action < cfg.n_channels:
            ch = action
            n_su_on_ch = int(np.sum(others == ch)) + 1  # incl. this agent
            if pu_busy[ch] == 1:
                pu_clash = True
                reward = cfg.reward_pu_clash
            elif n_su_on_ch > 1:
                collided = True
                reward = cfg.reward_collision
            else:
                success = True
                reward = cfg.reward_success

        # Advance the PU occupancy chain and sense for the next slot.
        self._chain.step(cfg, self._rng)
        self._sense = _sense_channels(self._chain.busy, cfg, self._rng)

        self._t += 1
        done = self._t >= cfg.max_steps
        obs = encode_sensing(self._sense)
        info = {
            "pu_busy": pu_busy.copy(),
            "others": others,
            "success": success,
            "collision": collided,
            "pu_clash": pu_clash,
            "t": self._t,
        }
        return obs, float(reward), done, info


@dataclass
class DSAWorld:
    """Multi-agent DSA world: ``n_users`` SUs each act simultaneously.

    Unlike :class:`DSAEnv` (one controlled agent + random background), this steps
    *all* users from their supplied actions and resolves collisions jointly. The
    federated pipeline uses this to evaluate a *shared* policy deployed to every
    SU, which is the realistic deployment of a federated DSA agent: every SU
    sees the same wideband sensing snapshot, so a degenerate (poisoned) policy
    that maps every state to one channel produces a collision storm.
    """

    cfg: DSAConfig = field(default_factory=DSAConfig)
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _chain: _ChannelChain = field(init=False, repr=False)
    _sense: np.ndarray = field(init=False, repr=False)
    _t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    @property
    def n_states(self) -> int:
        return n_states(self.cfg.n_channels)

    @property
    def n_actions(self) -> int:
        return n_actions(self.cfg.n_channels)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._chain = _ChannelChain.initial(self.cfg.n_channels, self.cfg, self._rng)
        self._sense = _sense_channels(self._chain.busy, self.cfg, self._rng)
        self._t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        # Each SU senses independently (independent sensing noise), so observations
        # differ slightly across SUs — but here we give every SU the same wideband
        # snapshot state id (shared sensing infrastructure) for a clean evaluation.
        s = encode_sensing(self._sense)
        return np.full(self.cfg.n_users, s, dtype=np.int64)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape[0] != cfg.n_users:
            raise ValueError("one action per user required")

        pu_busy = self._chain.busy
        rewards = np.full(cfg.n_users, cfg.reward_idle, dtype=np.float64)
        successes = collisions = pu_clashes = 0

        tx_mask = actions < cfg.n_channels
        counts = np.zeros(cfg.n_channels, dtype=np.int64)
        for ch in actions[tx_mask]:
            counts[ch] += 1

        for u in range(cfg.n_users):
            a = int(actions[u])
            if a >= cfg.n_channels:
                continue  # idle
            if pu_busy[a] == 1:
                rewards[u] = cfg.reward_pu_clash
                pu_clashes += 1
            elif counts[a] > 1:
                rewards[u] = cfg.reward_collision
                collisions += 1
            else:
                rewards[u] = cfg.reward_success
                successes += 1

        self._chain.step(cfg, self._rng)
        self._sense = _sense_channels(self._chain.busy, cfg, self._rng)
        self._t += 1
        done = self._t >= cfg.max_steps
        info = {
            "pu_busy": pu_busy.copy(),
            "successes": successes,
            "collisions": collisions,
            "pu_clashes": pu_clashes,
            "t": self._t,
        }
        return self._obs(), rewards, done, info


__all__ = [
    "DSAConfig",
    "DSAEnv",
    "DSAWorld",
    "encode_sensing",
    "n_states",
    "n_actions",
]
