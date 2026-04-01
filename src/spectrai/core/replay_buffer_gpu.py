"""
GPU-resident replay buffer with pinned memory transfers.

Optimizations:
  1. Stores all data directly on GPU — no CPU->GPU transfer at sample time
  2. Uses pinned memory for fast async pushes from CPU environments
  3. Vectorized batch push for environment vectorization
  4. Pre-computed random indices on GPU (avoids CPU->GPU index transfer)
"""

from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBufferGPU:
    """
    GPU-resident replay buffer.

    All tensors live on GPU. Pushes use pinned memory + async transfer.
    Sampling is pure GPU operations (no CPU roundtrip).
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        device: torch.device,
        prefetch_batches: int = 2,
    ):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0
        self.prefetch_batches = prefetch_batches

        # GPU-resident storage
        self.states = torch.zeros((capacity, *state_shape), device=device, dtype=torch.float32)
        self.actions = torch.zeros(capacity, device=device, dtype=torch.long)
        self.rewards = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *state_shape), device=device, dtype=torch.float32)
        self.dones = torch.zeros(capacity, device=device, dtype=torch.float32)

        # Pinned memory staging buffers for async CPU->GPU transfers
        self._pin_state = torch.zeros((1, *state_shape), dtype=torch.float32).pin_memory()
        self._pin_next = torch.zeros((1, *state_shape), dtype=torch.float32).pin_memory()

        # CUDA stream for async data transfers
        self._transfer_stream = torch.cuda.Stream(device=device)

    def push(self, state, action, reward, next_state, done):
        """Push a single transition using async pinned memory transfer."""
        idx = self.idx

        # Stage in pinned memory
        if isinstance(state, np.ndarray):
            self._pin_state[0].copy_(torch.from_numpy(state))
            self._pin_next[0].copy_(torch.from_numpy(next_state))
        else:
            self._pin_state[0].copy_(state)
            self._pin_next[0].copy_(next_state)

        # Async transfer to GPU
        with torch.cuda.stream(self._transfer_stream):
            self.states[idx].copy_(self._pin_state[0], non_blocking=True)
            self.next_states[idx].copy_(self._pin_next[0], non_blocking=True)
            self.actions[idx] = int(action)
            self.rewards[idx] = float(reward)
            self.dones[idx] = float(done)

        self.idx = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push a batch of transitions (for vectorized environments)."""
        batch_size = len(states)

        if isinstance(states, np.ndarray):
            states_t = torch.from_numpy(states).to(self.device, non_blocking=True)
            next_t = torch.from_numpy(next_states).to(self.device, non_blocking=True)
            actions_t = torch.from_numpy(actions).to(self.device, non_blocking=True)
            rewards_t = torch.from_numpy(rewards).to(self.device, non_blocking=True)
            dones_t = torch.from_numpy(dones).to(self.device, non_blocking=True)
        else:
            states_t = states.to(self.device, non_blocking=True)
            next_t = next_states.to(self.device, non_blocking=True)
            actions_t = actions.to(self.device, non_blocking=True)
            rewards_t = rewards.to(self.device, non_blocking=True)
            dones_t = dones.to(self.device, non_blocking=True)

        # Handle wraparound
        end_idx = self.idx + batch_size
        if end_idx <= self.capacity:
            self.states[self.idx:end_idx] = states_t
            self.next_states[self.idx:end_idx] = next_t
            self.actions[self.idx:end_idx] = actions_t
            self.rewards[self.idx:end_idx] = rewards_t
            self.dones[self.idx:end_idx] = dones_t
        else:
            # Split across boundary
            first = self.capacity - self.idx
            self.states[self.idx:] = states_t[:first]
            self.next_states[self.idx:] = next_t[:first]
            self.actions[self.idx:] = actions_t[:first]
            self.rewards[self.idx:] = rewards_t[:first]
            self.dones[self.idx:] = dones_t[:first]

            remaining = batch_size - first
            self.states[:remaining] = states_t[first:]
            self.next_states[:remaining] = next_t[first:]
            self.actions[:remaining] = actions_t[first:]
            self.rewards[:remaining] = rewards_t[first:]
            self.dones[:remaining] = dones_t[first:]

        self.idx = end_idx % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch entirely on GPU (no CPU roundtrip)."""
        # Generate random indices on GPU
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size
