"""Tests for the simulated DSA environment."""

import numpy as np
import pytest

from spectrai.env.sim import SimulatedDSAEnv


class TestResetShape:
    """reset() should return an observation of the expected shape."""

    def test_reset_shape(self, sim_env, small_env_config):
        obs, info = sim_env.reset(seed=42)

        seq_len = small_env_config["sequence_length"]
        num_channels = small_env_config["num_channels"]
        num_features = small_env_config["num_features"]
        expected_shape = (seq_len, num_channels * num_features)

        assert obs.shape == expected_shape
        assert obs.dtype == np.float32


class TestStepRewardRange:
    """Reward should be bounded within the expected range."""

    def test_step_reward_range(self, sim_env):
        sim_env.reset(seed=0)

        rewards = []
        for _ in range(100):
            action = sim_env.action_space.sample()
            _, reward, terminated, truncated, _ = sim_env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                sim_env.reset()

        for r in rewards:
            # +1.0 success, -1.0 collision, -0.1 switch penalty possible
            # Worst case: -1.0 - 0.1 = -1.1
            assert -1.1 <= r <= 1.0, f"Reward {r} out of expected range [-1.1, 1.0]"


class TestEpisodeTruncation:
    """Episode should be truncated after max_steps."""

    def test_episode_truncation(self):
        max_steps = 25
        env = SimulatedDSAEnv(num_channels=4, max_steps=max_steps)
        env.reset(seed=1)

        steps = 0
        truncated = False
        for _ in range(max_steps + 10):
            _, _, terminated, truncated, info = env.step(0)
            steps += 1
            if terminated or truncated:
                break

        assert truncated is True
        assert steps == max_steps


class TestConfigurableChannels:
    """Environment should work with different num_channels values."""

    @pytest.mark.parametrize("num_channels", [2, 5, 10, 20])
    def test_configurable_channels(self, num_channels):
        env = SimulatedDSAEnv(
            num_channels=num_channels,
            sequence_length=4,
            num_features=3,
            max_steps=10,
        )
        obs, _ = env.reset(seed=7)

        assert obs.shape == (4, num_channels * 3)
        assert env.action_space.n == num_channels

        # Take a step in each channel
        for ch in range(num_channels):
            _, _, terminated, truncated, info = env.step(ch)
            assert "success" in info
            if terminated or truncated:
                env.reset()
