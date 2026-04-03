"""
GPU-vectorized DSA environment using NVIDIA Warp JIT kernels.

Compiles channel dynamics and feature generation into fused CUDA kernels
via NVIDIA Warp, eliminating per-op kernel launch overhead from PyTorch.

Requires: warp-lang (pip install warp-lang)
Falls back to VectorizedDSAEnv if Warp is not installed.
"""

import torch
import numpy as np
from typing import Tuple, Dict

try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False

if _HAS_WARP:
    @wp.kernel
    def step_channels_kernel(
        states: wp.array2d(dtype=wp.float32),
        rand_vals: wp.array2d(dtype=wp.float32),
        on_prob: wp.float32,
        off_prob: wp.float32,
    ):
        """Advance Markov channel dynamics for all envs * channels."""
        env_id, ch_id = wp.tid()
        current = states[env_id, ch_id]
        r = rand_vals[env_id, ch_id]
        if current < 0.5:
            if r < on_prob:
                states[env_id, ch_id] = 1.0
        else:
            if r < off_prob:
                states[env_id, ch_id] = 0.0

    @wp.kernel
    def build_features_kernel(
        states: wp.array2d(dtype=wp.float32),
        features_out: wp.array2d(dtype=wp.float32),
        noise: wp.array2d(dtype=wp.float32),
        snr_free: wp.float32,
        snr_occupied: wp.float32,
        intf_free: wp.float32,
        intf_occupied: wp.float32,
        num_features: wp.int32,
    ):
        """Build [SNR, interference, occupancy] features for all envs * channels."""
        env_id, ch_id = wp.tid()
        occ = states[env_id, ch_id]
        base_idx = ch_id * num_features

        if occ < 0.5:
            features_out[env_id, base_idx + 0] = snr_free + noise[env_id, ch_id]
            features_out[env_id, base_idx + 1] = intf_free + noise[env_id, ch_id]
        else:
            features_out[env_id, base_idx + 0] = snr_occupied + noise[env_id, ch_id]
            features_out[env_id, base_idx + 1] = intf_occupied + noise[env_id, ch_id]
        features_out[env_id, base_idx + 2] = occ

    @wp.kernel
    def compute_rewards_kernel(
        states: wp.array2d(dtype=wp.float32),
        actions: wp.array(dtype=wp.int32),
        prev_actions: wp.array(dtype=wp.int32),
        rewards_out: wp.array(dtype=wp.float32),
        collisions_out: wp.array(dtype=wp.int32),
        reward_success: wp.float32,
        reward_collision: wp.float32,
        reward_switch: wp.float32,
    ):
        """Compute rewards for all environments."""
        env_id = wp.tid()
        action = actions[env_id]
        occ = states[env_id, action]

        if occ > 0.5:
            rewards_out[env_id] = reward_collision
            collisions_out[env_id] = 1
        else:
            rewards_out[env_id] = reward_success
            collisions_out[env_id] = 0

        prev = prev_actions[env_id]
        if prev >= 0 and action != prev:
            rewards_out[env_id] = rewards_out[env_id] + reward_switch


class WarpVectorizedDSAEnv:
    """
    Vectorized DSA environment with NVIDIA Warp JIT CUDA kernels.

    Same interface as VectorizedDSAEnv but uses Warp for fused
    channel dynamics, feature generation, and reward computation.

    Requires: pip install warp-lang
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
        if not _HAS_WARP:
            raise ImportError("NVIDIA Warp required. Install with: pip install warp-lang")

        wp.init()

        self.num_envs = num_envs
        self.num_channels = num_channels
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.device = device
        self.obs_noise_std = obs_noise_std

        # Scalar params
        self.pu_on_prob = pu_on_prob
        self.pu_off_prob = pu_off_prob
        self.snr_free = snr_free
        self.snr_occupied = snr_occupied
        self.intf_free = interference_free
        self.intf_occupied = interference_occupied
        self.noise_std = noise_std
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        self.reward_switch = reward_switch

        # Warp arrays (GPU)
        wp_device = "cuda:0"
        self._states = wp.zeros((num_envs, num_channels), dtype=wp.float32, device=wp_device)
        obs_dim = num_channels * num_features
        self._features = wp.zeros((num_envs, obs_dim), dtype=wp.float32, device=wp_device)
        self._rand = wp.zeros((num_envs, num_channels), dtype=wp.float32, device=wp_device)
        self._noise = wp.zeros((num_envs, num_channels), dtype=wp.float32, device=wp_device)
        self._rewards = wp.zeros(num_envs, dtype=wp.float32, device=wp_device)
        self._collisions = wp.zeros(num_envs, dtype=wp.int32, device=wp_device)

        # PyTorch tensors for history and actions (interop via DLPack)
        self.history = torch.zeros(num_envs, sequence_length, obs_dim, device=device)
        self.prev_actions = torch.full((num_envs,), -1, dtype=torch.int32, device=device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        n = mask.sum().item()
        if n == 0:
            return self._get_obs()

        # Reset via PyTorch
        states_torch = wp.to_torch(self._states)
        states_torch[mask] = (torch.rand(n, self.num_channels, device=self.device) < 0.5).float()

        self.step_counts[mask] = 0
        self.prev_actions[mask] = -1

        obs_dim = self.num_channels * self.num_features
        self.history[mask] = 0.0

        for t in range(self.sequence_length):
            self._warp_step_channels()
            self._warp_build_features()
            features_torch = wp.to_torch(self._features)
            self.history[mask, t] = features_torch[mask]

        return self._get_obs()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self.step_counts += 1

        # Step channels via Warp kernel
        self._warp_step_channels()

        # Compute rewards via Warp kernel
        actions_wp = wp.from_torch(actions.int(), dtype=wp.int32)
        prev_wp = wp.from_torch(self.prev_actions, dtype=wp.int32)

        wp.launch(
            compute_rewards_kernel,
            dim=self.num_envs,
            inputs=[
                self._states, actions_wp, prev_wp, self._rewards, self._collisions,
                self.reward_success, self.reward_collision, self.reward_switch,
            ],
        )

        self.prev_actions = actions.int()

        # Build features via Warp kernel
        self._warp_build_features()

        # Update history
        features_torch = wp.to_torch(self._features)
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1] = features_torch

        dones = self.step_counts >= self.max_steps

        obs = self._get_obs()
        rewards = wp.to_torch(self._rewards)
        collisions = wp.to_torch(self._collisions)

        infos = {
            "collision": collisions > 0,
            "success": collisions == 0,
        }

        return obs, rewards, dones, infos

    def auto_reset(self, dones: torch.Tensor) -> torch.Tensor:
        if dones.any():
            self.reset(mask=dones)
        return self._get_obs()

    def _warp_step_channels(self):
        """Launch Warp kernel for Markov channel dynamics."""
        rand_torch = torch.rand(self.num_envs, self.num_channels, device=self.device)
        self._rand = wp.from_torch(rand_torch, dtype=wp.float32)

        wp.launch(
            step_channels_kernel,
            dim=(self.num_envs, self.num_channels),
            inputs=[self._states, self._rand, self.pu_on_prob, self.pu_off_prob],
        )

    def _warp_build_features(self):
        """Launch Warp kernel for feature generation."""
        noise_torch = torch.randn(self.num_envs, self.num_channels, device=self.device) * self.noise_std
        self._noise = wp.from_torch(noise_torch, dtype=wp.float32)

        wp.launch(
            build_features_kernel,
            dim=(self.num_envs, self.num_channels),
            inputs=[
                self._states, self._features, self._noise,
                self.snr_free, self.snr_occupied,
                self.intf_free, self.intf_occupied,
                self.num_features,
            ],
        )

    def _get_obs(self) -> torch.Tensor:
        if self.obs_noise_std > 0:
            return self.history + torch.randn_like(self.history) * self.obs_noise_std
        return self.history.clone()
