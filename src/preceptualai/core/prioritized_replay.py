"""
Prioritized Experience Replay (PER) for UHCI.

Implements Schaul et al. 2015 — transitions with higher TD-error are
sampled more frequently, focusing learning on surprising or rare events.

Critical for UHCI because:
  - NTN handover events are rare (LEO pass every ~90 min) but high-value
  - Cross-provider interference spikes are infrequent but policy-critical
  - Rain fade events on satellite links are episodic and seasonal
  - ISAC sensing targets appear stochastically

Without PER, uniform sampling wastes 90%+ of updates on common,
already-learned terrestrial transitions.

References:
  Schaul et al., "Prioritized Experience Replay", ICLR 2016.
  Horgan et al., "Distributed Prioritized Experience Replay", ICLR 2018.
"""

from typing import Dict, Tuple

import numpy as np
import torch


class SumTree:
    """
    Binary sum-tree for O(log N) priority-based sampling.

    Each leaf holds a priority value. Internal nodes hold the sum
    of their children. Sampling proportional to priority is O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float):
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Sample a leaf proportional to priority.

        Returns: (tree_idx, priority, data_idx)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    @property
    def min_priority(self) -> float:
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 0.0
        return float(self.tree[leaf_start:leaf_end].min())


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for the UHCI agent.

    Uses a sum-tree for O(log N) proportional priority sampling
    and importance-sampling weights for unbiased gradient updates.

    Args:
        capacity:  maximum number of transitions
        obs_dim:   observation dimensionality
        alpha:     priority exponent (0 = uniform, 1 = full prioritization)
        beta_start: initial importance-sampling exponent
        beta_end:  final IS exponent (annealed linearly)
        beta_frames: number of frames to anneal beta over
        device:    torch device
    """

    def __init__(
        self,
        capacity:    int,
        obs_dim:     int,
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_end:    float = 1.0,
        beta_frames: int   = 100_000,
        device:      torch.device = torch.device("cpu"),
    ):
        self.capacity    = capacity
        self.obs_dim     = obs_dim
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.beta_frames = beta_frames
        self.device      = device

        self.tree = SumTree(capacity)
        self._frame = 0
        self._max_priority = 1.0

        # Storage arrays
        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (linearly annealed)."""
        frac = min(1.0, self._frame / max(self.beta_frames, 1))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        """Add a single transition with max priority."""
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)

        # New transitions get max priority so they're sampled at least once
        self.tree.add(self._max_priority ** self.alpha)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push a batch of transitions."""
        if isinstance(states, torch.Tensor):
            states      = states.cpu().numpy()
            actions     = actions.cpu().numpy()
            rewards     = rewards.cpu().numpy()
            next_states = next_states.cpu().numpy()
            dones       = dones.cpu().numpy()

        B = len(states)
        for i in range(B):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a prioritized batch with importance-sampling weights.

        Returns dict with keys:
            states, actions, rewards, next_states, dones, weights, tree_indices
        """
        self._frame += 1
        beta = self.beta

        indices = []
        priorities = []
        tree_indices = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(s)
            # Clamp data_idx to valid range
            data_idx = max(0, min(data_idx, self._size - 1))
            indices.append(data_idx)
            priorities.append(priority)
            tree_indices.append(tree_idx)

        indices = np.array(indices)
        priorities = np.array(priorities, dtype=np.float64)

        # Importance-sampling weights
        total = self.tree.total()
        min_prob = self.tree.min_priority / max(total, 1e-8)
        min_prob = max(min_prob, 1e-8)

        probs = priorities / max(total, 1e-8)
        probs = np.clip(probs, 1e-8, None)

        weights = (self._size * probs) ** (-beta)
        max_weight = (self._size * min_prob) ** (-beta)
        weights = weights / max(max_weight, 1e-8)  # normalise

        return {
            "states":       torch.from_numpy(self.states[indices]).to(self.device),
            "actions":      torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards":      torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_states":  torch.from_numpy(self.next_states[indices]).to(self.device),
            "dones":        torch.from_numpy(self.dones[indices]).to(self.device),
            "weights":      torch.from_numpy(weights.astype(np.float32)).to(self.device),
            "tree_indices":  tree_indices,
        }

    def update_priorities(self, tree_indices, td_errors: torch.Tensor):
        """
        Update priorities for sampled transitions based on TD-error.

        Args:
            tree_indices: list of sum-tree indices from sample()
            td_errors:    (B,) absolute TD-errors
        """
        priorities = (td_errors.abs().cpu().numpy() + 1e-6) ** self.alpha
        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, float(priority))
            self._max_priority = max(self._max_priority, float(priority))

    def __len__(self) -> int:
        return self._size
