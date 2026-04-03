"""Tests for the SACLTCAgent end-to-end."""

import os
import tempfile

import numpy as np
import torch


class TestAgentInit:
    """Agent should initialise all sub-networks without error."""

    def test_agent_init(self, small_agent):
        assert small_agent.actor is not None
        assert small_agent.critic1 is not None
        assert small_agent.critic2 is not None
        assert small_agent.target_critic1 is not None
        assert small_agent.target_critic2 is not None
        assert small_agent.train_step_count == 0


class TestSelectAction:
    """select_action should return a valid action index."""

    def test_select_action(self, small_agent, small_env_config):
        num_channels = small_env_config["num_channels"]
        num_features = small_env_config["num_features"]
        seq_len = small_env_config["sequence_length"]
        input_dim = num_channels * num_features

        state = np.random.randn(seq_len, input_dim).astype(np.float32)
        action = small_agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < num_channels


class TestUpdateEmptyBuffer:
    """update() with an empty buffer should return an empty dict (no crash)."""

    def test_update_empty_buffer(self, small_agent):
        result = small_agent.update()
        assert result == {}


class TestSaveLoadRoundtrip:
    """Saving and loading should preserve network weights exactly."""

    def test_save_load_roundtrip(self, small_agent, small_env_config):
        num_channels = small_env_config["num_channels"]
        num_features = small_env_config["num_features"]
        seq_len = small_env_config["sequence_length"]
        input_dim = num_channels * num_features

        state = np.random.randn(seq_len, input_dim).astype(np.float32)

        # Get action before save
        torch.manual_seed(0)
        action_before = small_agent.select_action(state, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.pt")
            small_agent.save(path)

            assert os.path.isfile(path)

            # Create a fresh agent with same architecture
            from preceptualai.agent.sac_ltc import SACLTCAgent

            fresh = SACLTCAgent(
                state_shape=(seq_len, input_dim),
                num_actions=num_channels,
                input_dim=input_dim,
                device=small_agent.device,
                hidden_dim=32,
                latent_dim=32,
                num_layers=1,
                buffer_size=100,
                batch_size=8,
                learning_starts=10,
            )
            fresh.load(path)

            torch.manual_seed(0)
            action_after = fresh.select_action(state, deterministic=True)

            assert action_before == action_after

            # Verify weight equality
            for p1, p2 in zip(
                small_agent.actor.parameters(), fresh.actor.parameters()
            ):
                assert torch.equal(p1, p2)
