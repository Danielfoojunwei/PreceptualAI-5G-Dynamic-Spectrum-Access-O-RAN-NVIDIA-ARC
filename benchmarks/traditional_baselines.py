"""
Traditional Scheduling and Spectrum Access Baselines for DSA Benchmarking.

Implements 14 classical methods spanning fixed allocation (TDMA, FDMA),
heuristic scheduling (round-robin, greedy, proportional fair), bandit
algorithms (UCB, Thompson Sampling, Boltzmann), model-based (Whittle Index),
and standard-inspired approaches (WiFi 7 MLO, OFDMA, carrier aggregation).

These baselines provide canonical reference points for evaluating learned
spectrum agents. All conform to the same select_action() interface used
by the RL benchmark harness.

Important: Results are empirical — if a traditional method outperforms
SAC-LTC on any metric, that is reported honestly.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import numpy as np


# ======================================================================
# Abstract base class
# ======================================================================

class TraditionalBaseline(ABC):
    """
    Base class for all non-learning baselines.

    Contract:
    - select_action(state, deterministic=True) -> int
    - reset_episode()  — called at the start of each eval episode
    - save(path) / load(path)  — no-ops (no learned weights)
    - replay_buffer = None  — compatibility with benchmark harness
    """

    def __init__(self, num_channels: int, num_features: int = 3, **kwargs):
        self.num_channels = num_channels
        self.num_features = num_features
        self.replay_buffer = None
        self.learning_starts = 0  # Compatibility: never blocks evaluation

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select a channel given the observation state."""
        ...

    def reset_episode(self) -> None:
        """Reset any per-episode internal state. Override if needed."""
        pass

    def update(self) -> Dict[str, float]:
        """No-op: traditional baselines don't train."""
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def _parse_observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract per-channel [SNR, interference, occupancy] from the latest timestep.

        The DSA env stacks features as:
            [snr_0, interf_0, occ_0, snr_1, interf_1, occ_1, ...]
        so reshaping to (num_channels, num_features) gives columns [SNR, interf, occ].

        Args:
            state: shape (sequence_length, num_channels * num_features)

        Returns:
            snr, interference, occupancy — each shape (num_channels,)
        """
        latest = state[-1]  # most recent timestep
        features = latest.reshape(self.num_channels, self.num_features)
        snr = features[:, 0]
        interference = features[:, 1]
        occupancy = features[:, 2]
        return snr, interference, occupancy

    def _parse_full_history(self, state: np.ndarray) -> np.ndarray:
        """
        Reshape full history into (seq_len, num_channels, num_features).

        Returns:
            Array of shape (T, num_channels, num_features)
        """
        T = state.shape[0]
        return state.reshape(T, self.num_channels, self.num_features)


# ======================================================================
# 1. Random Baseline
# ======================================================================

class RandomBaseline(TraditionalBaseline):
    """Uniform random channel selection. Theoretical lower bound."""

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        return np.random.randint(0, self.num_channels)


# ======================================================================
# 2. Round-Robin Baseline
# ======================================================================

class RoundRobinBaseline(TraditionalBaseline):
    """Cycle through channels sequentially. No adaptation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx = 0

    def reset_episode(self) -> None:
        self._idx = 0

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        action = self._idx % self.num_channels
        self._idx += 1
        return action


# ======================================================================
# 3. Greedy Max-SINR Baseline
# ======================================================================

class GreedyMaxSINRBaseline(TraditionalBaseline):
    """Pick the channel with the highest observed SINR. Stateless."""

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        snr, interference, _ = self._parse_observation(state)
        sinr = snr / (interference + 1e-8)
        return int(np.argmax(sinr))


# ======================================================================
# 4. Epsilon-Greedy Baseline
# ======================================================================

class EpsilonGreedyBaseline(TraditionalBaseline):
    """
    With probability epsilon, explore randomly.
    Otherwise, exploit by picking max-SINR channel.
    When deterministic=True, always exploits.
    """

    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        snr, interference, _ = self._parse_observation(state)
        sinr = snr / (interference + 1e-8)

        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_channels)
        return int(np.argmax(sinr))


# ======================================================================
# 5. TDMA-style Baseline
# ======================================================================

class TDMABaseline(TraditionalBaseline):
    """
    Fixed time-slot allocation: channel = step_count % num_channels.

    Models Time Division Multiple Access where each user gets a
    predetermined slot regardless of channel conditions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step = 0

    def reset_episode(self) -> None:
        self._step = 0

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        action = self._step % self.num_channels
        self._step += 1
        return action


# ======================================================================
# 6. FDMA-style Baseline
# ======================================================================

class FDMABaseline(TraditionalBaseline):
    """
    Fixed frequency assignment: always use the same channel.

    Models Frequency Division Multiple Access where each user is
    permanently assigned a dedicated frequency band.

    Expected success rate equals steady-state free probability:
        p10 / (p01 + p10) = 0.5 / (0.3 + 0.5) = 62.5%
    """

    def __init__(self, fixed_channel: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.fixed_channel = fixed_channel

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        return self.fixed_channel


# ======================================================================
# 7. Proportional Fair Baseline
# ======================================================================

class ProportionalFairBaseline(TraditionalBaseline):
    """
    Proportional Fair scheduler: picks the channel with the best
    instantaneous-to-average throughput ratio.

    Widely used in LTE/5G downlink scheduling (3GPP TS 36.213).
    Tracks exponentially weighted average throughput per channel.
    """

    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self._avg_throughput = None
        self._last_action: Optional[int] = None

    def reset_episode(self) -> None:
        self._avg_throughput = np.ones(self.num_channels) * 0.5
        self._last_action = None

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        if self._avg_throughput is None:
            self.reset_episode()

        snr, interference, occupancy = self._parse_observation(state)

        # Update average throughput from previous action's outcome
        if self._last_action is not None:
            occ = occupancy[self._last_action]
            success = 1.0 if occ < 0.5 else 0.0  # threshold noisy occupancy
            ch = self._last_action
            self._avg_throughput[ch] = (
                (1 - self.alpha) * self._avg_throughput[ch] + self.alpha * success
            )

        # Instantaneous quality estimate from SNR
        instantaneous = snr / (interference + 1e-8)
        # Proportional fair metric
        pf_metric = instantaneous / (self._avg_throughput + 1e-8)

        action = int(np.argmax(pf_metric))
        self._last_action = action
        return action


# ======================================================================
# 8. Thompson Sampling Baseline
# ======================================================================

class ThompsonSamplingBaseline(TraditionalBaseline):
    """
    Bayesian bandit: maintains Beta(alpha, beta) posterior per channel
    for the probability of being free. Samples from each posterior
    and picks the channel with the highest sample.

    Updates beliefs using thresholded occupancy from observations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._alpha_params = None
        self._beta_params = None
        self._last_action: Optional[int] = None

    def reset_episode(self) -> None:
        self._alpha_params = np.ones(self.num_channels)  # uniform prior
        self._beta_params = np.ones(self.num_channels)
        self._last_action = None

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        if self._alpha_params is None:
            self.reset_episode()

        _, _, occupancy = self._parse_observation(state)

        # Update posterior from previous action's outcome
        if self._last_action is not None:
            ch = self._last_action
            is_free = 1.0 if occupancy[ch] < 0.5 else 0.0
            self._alpha_params[ch] += is_free
            self._beta_params[ch] += (1.0 - is_free)

        if deterministic:
            # Use posterior mean instead of sampling
            scores = self._alpha_params / (self._alpha_params + self._beta_params)
        else:
            # Sample from Beta posteriors
            scores = np.array([
                np.random.beta(self._alpha_params[i], self._beta_params[i])
                for i in range(self.num_channels)
            ])

        action = int(np.argmax(scores))
        self._last_action = action
        return action


# ======================================================================
# 9. Upper Confidence Bound (UCB) Baseline
# ======================================================================

class UCBBaseline(TraditionalBaseline):
    """
    UCB1 algorithm: balances exploration and exploitation using
    confidence intervals.

    Score = mean_reward + c * sqrt(ln(total_pulls) / pulls_per_arm)

    Classic bandit algorithm (Auer et al., 2002).
    """

    def __init__(self, c: float = 1.414, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self._counts = None
        self._values = None
        self._total = 0
        self._last_action: Optional[int] = None

    def reset_episode(self) -> None:
        self._counts = np.zeros(self.num_channels)
        self._values = np.zeros(self.num_channels)
        self._total = 0
        self._last_action = None

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        if self._counts is None:
            self.reset_episode()

        _, _, occupancy = self._parse_observation(state)

        # Update from previous action
        if self._last_action is not None:
            ch = self._last_action
            reward = 1.0 if occupancy[ch] < 0.5 else 0.0
            self._counts[ch] += 1
            # Incremental mean update
            self._values[ch] += (reward - self._values[ch]) / self._counts[ch]

        self._total += 1

        # If any channel untried, try it
        untried = np.where(self._counts == 0)[0]
        if len(untried) > 0:
            action = int(untried[0])
        else:
            # UCB1 score
            exploration = self.c * np.sqrt(np.log(self._total) / (self._counts + 1e-8))
            ucb_scores = self._values + exploration
            action = int(np.argmax(ucb_scores))

        self._last_action = action
        return action


# ======================================================================
# 10. Whittle Index Baseline
# ======================================================================

class WhittleIndexBaseline(TraditionalBaseline):
    """
    Whittle Index policy for restless multi-armed bandits.

    For two-state Markov channels (ON/OFF), computes the one-step-ahead
    probability of each channel being free and selects the channel with
    the highest index.

    This is the theoretically optimal policy for independent Markov
    channels when the indexability condition holds (Weber & Weiss, 1990).
    """

    def __init__(self, pu_on_prob: float = 0.3, pu_off_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.p01 = pu_on_prob   # P(OFF -> ON)
        self.p10 = pu_off_prob  # P(ON -> OFF)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        _, _, occupancy = self._parse_observation(state)

        # Threshold noisy occupancy to get believed state
        believed_occupied = (occupancy > 0.5).astype(float)

        # Whittle index: one-step-ahead probability of being free
        # If currently free (occ=0): P(stays free) = 1 - p01
        # If currently occupied (occ=1): P(becomes free) = p10
        whittle = (1.0 - believed_occupied) * (1.0 - self.p01) + believed_occupied * self.p10

        return int(np.argmax(whittle))


# ======================================================================
# 11. Boltzmann Exploration Baseline
# ======================================================================

class BoltzmannBaseline(TraditionalBaseline):
    """
    Softmax action selection over running Q-value estimates.

    P(channel) = exp(Q(channel) / temperature) / sum(exp(Q / temperature))

    Higher temperature -> more exploration. deterministic=True uses argmax.
    """

    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self._q_values = None
        self._counts = None
        self._last_action: Optional[int] = None

    def reset_episode(self) -> None:
        self._q_values = np.zeros(self.num_channels)
        self._counts = np.zeros(self.num_channels)
        self._last_action = None

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        if self._q_values is None:
            self.reset_episode()

        _, _, occupancy = self._parse_observation(state)

        # Update Q from previous action
        if self._last_action is not None:
            ch = self._last_action
            reward = 1.0 if occupancy[ch] < 0.5 else -1.0
            self._counts[ch] += 1
            self._q_values[ch] += (reward - self._q_values[ch]) / self._counts[ch]

        if deterministic:
            action = int(np.argmax(self._q_values))
        else:
            # Boltzmann / softmax
            logits = self._q_values / (self.temperature + 1e-8)
            logits -= logits.max()  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum()
            action = int(np.random.choice(self.num_channels, p=probs))

        self._last_action = action
        return action


# ======================================================================
# 12. WiFi 7 MLO-style Baseline
# ======================================================================

class WiFi7MLOBaseline(TraditionalBaseline):
    """
    WiFi 7 Multi-Link Operation (MLO) heuristic.

    Monitors all channels (links) using the full observation history
    and selects the channel with the best composite score:
        score = w1*mean_snr - w2*latest_interf - w3*occ_stability - w4*latest_occ

    Models 802.11be multi-link decision logic: monitor all bands
    (2.4/5/6 GHz), estimate quality, and steer traffic to the best
    available link.

    Weights are domain-informed per IEEE 802.11be specification:
    - w1=1.0: favor high SNR
    - w2=0.5: penalize interference
    - w3=0.3: favor stable (predictable) channels
    - w4=2.0: strongly penalize currently-occupied channels
    """

    def __init__(
        self,
        w_snr: float = 1.0,
        w_interf: float = 0.5,
        w_stability: float = 0.3,
        w_occupancy: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_snr = w_snr
        self.w_interf = w_interf
        self.w_stability = w_stability
        self.w_occupancy = w_occupancy

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        history = self._parse_full_history(state)  # (T, C, F)

        # Mean SNR over history
        mean_snr = history[:, :, 0].mean(axis=0)  # (C,)

        # Latest interference
        latest_interf = history[-1, :, 1]  # (C,)

        # Occupancy stability: std of occupancy over time (lower = more stable)
        occ_std = history[:, :, 2].std(axis=0)  # (C,)

        # Latest occupancy
        latest_occ = history[-1, :, 2]  # (C,)

        score = (
            self.w_snr * mean_snr
            - self.w_interf * latest_interf
            - self.w_stability * occ_std
            - self.w_occupancy * latest_occ
        )

        return int(np.argmax(score))


# ======================================================================
# 13. OFDMA-style Baseline
# ======================================================================

class OFDMABaseline(TraditionalBaseline):
    """
    OFDMA-style subcarrier quality estimation.

    Treats each channel as a subcarrier group. Computes average quality
    metric across the observation window (mean SNR - mean interference)
    per channel, then selects the best.

    Models the 3GPP OFDMA scheduling approach where CQI (Channel Quality
    Indicator) is averaged and the best subband is allocated.
    """

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        history = self._parse_full_history(state)  # (T, C, F)

        # Average SNR across history
        avg_snr = history[:, :, 0].mean(axis=0)  # (C,)

        # Average interference across history
        avg_interf = history[:, :, 1].mean(axis=0)  # (C,)

        # Average occupancy across history (lower = more available)
        avg_occ = history[:, :, 2].mean(axis=0)  # (C,)

        # CQI-inspired metric: quality weighted by availability
        quality = (avg_snr - avg_interf) * (1.0 - avg_occ)

        return int(np.argmax(quality))


# ======================================================================
# 14. Carrier Aggregation Baseline
# ======================================================================

class CarrierAggregationBaseline(TraditionalBaseline):
    """
    Carrier Aggregation heuristic adapted for single-channel selection.

    In real CA (3GPP Release 10+), multiple component carriers are bonded.
    Here, we identify the top-K candidate channels by instantaneous SNR,
    then select the one with the lowest observed occupancy from recent
    history — modeling the CA primary component carrier selection logic.
    """

    def __init__(self, k_candidates: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k_candidates = min(k_candidates, kwargs.get("num_channels", 10))

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        history = self._parse_full_history(state)  # (T, C, F)

        # Latest SNR for candidate selection
        latest_snr = history[-1, :, 0]  # (C,)

        # Top-K channels by SNR
        k = min(self.k_candidates, self.num_channels)
        top_k_indices = np.argsort(latest_snr)[-k:]

        # Among top-K, pick the one with lowest mean occupancy
        avg_occ = history[:, :, 2].mean(axis=0)  # (C,)
        best_among_top_k = top_k_indices[np.argmin(avg_occ[top_k_indices])]

        return int(best_among_top_k)


# ======================================================================
# Baseline Registry
# ======================================================================

TRADITIONAL_BASELINES: Dict[str, Tuple[Type[TraditionalBaseline], dict]] = {
    "random":               (RandomBaseline, {}),
    "round_robin":          (RoundRobinBaseline, {}),
    "greedy_sinr":          (GreedyMaxSINRBaseline, {}),
    "epsilon_greedy":       (EpsilonGreedyBaseline, {"epsilon": 0.1}),
    "tdma":                 (TDMABaseline, {}),
    "fdma":                 (FDMABaseline, {"fixed_channel": 0}),
    "proportional_fair":    (ProportionalFairBaseline, {}),
    "thompson_sampling":    (ThompsonSamplingBaseline, {}),
    "ucb":                  (UCBBaseline, {"c": 1.414}),
    "whittle_index":        (WhittleIndexBaseline, {"pu_on_prob": 0.3, "pu_off_prob": 0.5}),
    "boltzmann":            (BoltzmannBaseline, {"temperature": 0.5}),
    "wifi7_mlo":            (WiFi7MLOBaseline, {}),
    "ofdma":                (OFDMABaseline, {}),
    "carrier_aggregation":  (CarrierAggregationBaseline, {"k_candidates": 3}),
}


def make_traditional_agent(
    agent_type: str,
    num_channels: int,
    num_features: int = 3,
    **override_params,
) -> TraditionalBaseline:
    """Instantiate a traditional baseline by name."""
    if agent_type not in TRADITIONAL_BASELINES:
        raise ValueError(
            f"Unknown traditional baseline: {agent_type}. "
            f"Available: {list(TRADITIONAL_BASELINES.keys())}"
        )
    cls, default_params = TRADITIONAL_BASELINES[agent_type]
    params = {**default_params, **override_params}
    return cls(num_channels=num_channels, num_features=num_features, **params)
