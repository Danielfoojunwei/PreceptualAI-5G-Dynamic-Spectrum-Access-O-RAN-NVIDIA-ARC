"""
GPU-vectorized Dynamic Spectrum Access environment.

Runs N_ENVS independent environment instances in parallel on the GPU.
All Markov dynamics, feature generation, reward computation, and
history management happen as batched tensor operations — zero Python loops.

This is the biggest single optimization: with 512+ parallel environments,
the replay buffer fills 512x faster and GPU utilization stays near 100%.
"""

import torch
import torch.nn.functional as F


class VectorizedDSAEnv:
    """
    Batched DSA environment running entirely on GPU.

    All operations are vectorized across num_envs parallel instances.
    No Gymnasium dependency — pure PyTorch tensors.
    """

    def __init__(
        self,
        num_envs: int = 256,
        num_channels: int = 10,
        num_features: int = 3,
        sequence_length: int = 16,
        max_steps: int = 200,
        pu_on_prob: float = 0.3,
        pu_off_prob: float = 0.5,
        snr_free: float = 1.0,
        snr_occupied: float = 0.2,
        interference_free: float = 0.1,
        interference_occupied: float = 0.8,
        noise_std: float = 0.05,
        obs_noise_std: float = 0.1,
        reward_success: float = 1.0,
        reward_collision: float = -1.0,
        reward_switch: float = -0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_envs = num_envs
        self.num_channels = num_channels
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.device = device

        # Markov transition probabilities
        self.pu_on_prob = pu_on_prob
        self.pu_off_prob = pu_off_prob

        # Feature generation params
        self.snr_free = snr_free
        self.snr_occupied = snr_occupied
        self.interference_free = interference_free
        self.interference_occupied = interference_occupied
        self.noise_std = noise_std
        self.obs_noise_std = obs_noise_std

        # Reward params
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        self.reward_switch = reward_switch

        # State tensors (all on GPU)
        obs_dim = num_channels * num_features
        self.channel_states = torch.zeros(num_envs, num_channels, device=device)
        self.history = torch.zeros(num_envs, sequence_length, obs_dim, device=device)
        self.prev_actions = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Reset environments. If mask is provided, only reset those.

        Args:
            mask: (num_envs,) bool tensor. If None, reset all.

        Returns:
            observations: (num_envs, sequence_length, obs_dim)
        """
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        n_reset = mask.sum().item()
        if n_reset == 0:
            return self._get_observation()

        # Reset channel states (random init)
        self.channel_states[mask] = (
            torch.rand(n_reset, self.num_channels, device=self.device) < 0.5
        ).float()

        # Reset step counts and previous actions
        self.step_counts[mask] = 0
        self.prev_actions[mask] = -1

        # Fill history buffer
        obs_dim = self.num_channels * self.num_features
        self.history[mask] = torch.zeros(n_reset, self.sequence_length, obs_dim, device=self.device)

        for t in range(self.sequence_length):
            self._step_channels(mask)
            features = self._build_features(mask)
            self.history[mask, t] = features

        return self._get_observation()

    def step(self, actions: torch.Tensor):
        """
        Vectorized step for all environments.

        Args:
            actions: (num_envs,) int tensor — channel selections

        Returns:
            observations: (num_envs, sequence_length, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,) bool
            infos: dict of (num_envs,) tensors
        """
        self.step_counts += 1

        # Advance Markov dynamics for all envs
        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_channels(all_mask)

        # Compute rewards (vectorized)
        # Gather occupancy at selected channels
        action_occupancy = self.channel_states.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)  # (num_envs,)

        rewards = torch.where(
            action_occupancy > 0.5,
            torch.tensor(self.reward_collision, device=self.device),
            torch.tensor(self.reward_success, device=self.device),
        )

        # Switching penalty
        switched = (self.prev_actions >= 0) & (actions != self.prev_actions)
        rewards[switched] += self.reward_switch

        self.prev_actions = actions

        # Update history (roll + append)
        features = self._build_features(all_mask)
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = features

        # Done detection
        dones = self.step_counts >= self.max_steps

        # Observation
        obs = self._get_observation()

        # Collisions and successes
        collisions = action_occupancy > 0.5
        successes = ~collisions

        infos = {
            "collision": collisions,
            "success": successes,
        }

        return obs, rewards, dones, infos

    def auto_reset(self, dones: torch.Tensor) -> torch.Tensor:
        """Reset done environments and return new observations."""
        if dones.any():
            self.reset(mask=dones)
        return self._get_observation()

    def _step_channels(self, mask: torch.Tensor):
        """Advance Markov dynamics for masked environments (vectorized)."""
        n = mask.sum().item()
        if n == 0:
            return

        rand = torch.rand(n, self.num_channels, device=self.device)
        current = self.channel_states[mask]

        # OFF -> ON transition
        turn_on = (current < 0.5) & (rand < self.pu_on_prob)
        # ON -> OFF transition
        turn_off = (current > 0.5) & (rand < self.pu_off_prob)

        new_states = current.clone()
        new_states[turn_on] = 1.0
        new_states[turn_off] = 0.0

        self.channel_states[mask] = new_states

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Build per-channel features for masked envs (vectorized)."""
        n = mask.sum().item()
        occupancy = self.channel_states[mask]  # (n, C)

        # SNR
        snr = torch.where(
            occupancy < 0.5,
            torch.tensor(self.snr_free, device=self.device),
            torch.tensor(self.snr_occupied, device=self.device),
        )
        snr = snr + torch.randn_like(snr) * self.noise_std

        # Interference
        interference = torch.where(
            occupancy > 0.5,
            torch.tensor(self.interference_occupied, device=self.device),
            torch.tensor(self.interference_free, device=self.device),
        )
        interference = interference + torch.randn_like(interference) * self.noise_std

        # Stack features: [snr, interference, occupancy] per channel
        features = torch.stack([snr, interference, occupancy], dim=-1)  # (n, C, F)
        return features.reshape(n, -1)  # (n, C*F)

    def _get_observation(self) -> torch.Tensor:
        """Return observations with optional noise."""
        if self.obs_noise_std > 0:
            return self.history + torch.randn_like(self.history) * self.obs_noise_std
        return self.history.clone()
