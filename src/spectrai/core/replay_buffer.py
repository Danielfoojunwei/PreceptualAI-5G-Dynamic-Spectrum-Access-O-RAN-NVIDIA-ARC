"""Fixed-size ring buffer for off-policy experience replay."""

from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size ring buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int, state_shape: Tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0

        # Pre-allocate numpy arrays for speed
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.from_numpy(self.states[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }

    def __len__(self):
        return self.size
