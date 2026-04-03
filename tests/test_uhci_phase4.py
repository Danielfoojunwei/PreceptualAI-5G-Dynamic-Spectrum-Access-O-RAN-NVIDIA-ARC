"""
Tests for UHCI Phase 4 — Prioritized Replay + Curriculum Learning.

Validates:
  - SumTree proportional sampling correctness
  - PrioritizedReplayBuffer sampling and priority updates
  - PER integration with UniversalSpectrumAgent
  - CurriculumScheduler stage progression
  - End-to-end curriculum + PER training loop
"""

import math

import pytest
import torch

from preceptualai.env.provider_registry import ProviderType
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityConfig,
    UnifiedConnectivityEnv,
)
from preceptualai.core.universal_spectrum_agent import (
    UniversalAgentConfig,
    UniversalSpectrumAgent,
)
from preceptualai.core.prioritized_replay import (
    SumTree,
    PrioritizedReplayBuffer,
)
from preceptualai.core.curriculum import (
    CurriculumConfig,
    CurriculumScheduler,
    CurriculumStage,
)


# ─────────────────────────────────────────────────────────────────────────────
# SumTree
# ─────────────────────────────────────────────────────────────────────────────

class TestSumTree:

    def test_total_after_adds(self):
        tree = SumTree(capacity=8)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert abs(tree.total() - 6.0) < 1e-6

    def test_sample_returns_valid_index(self):
        tree = SumTree(capacity=8)
        for i in range(5):
            tree.add(float(i + 1))
        for _ in range(20):
            _, _, data_idx = tree.get(tree.total() * 0.5)
            assert 0 <= data_idx < 5

    def test_high_priority_sampled_more(self):
        tree = SumTree(capacity=100)
        # Add 99 low-priority items and 1 high-priority
        for _ in range(99):
            tree.add(0.01)
        tree.add(100.0)  # the last item has huge priority

        # Sample 1000 times — high-priority item should appear frequently
        high_count = 0
        for _ in range(1000):
            import random
            s = random.uniform(0, tree.total())
            _, _, data_idx = tree.get(s)
            if data_idx == 99:  # the high-priority item
                high_count += 1

        assert high_count > 500, \
            f"High-priority item only sampled {high_count}/1000 times"

    def test_update_changes_total(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(1.0)
        old_total = tree.total()
        tree.update(tree.capacity - 1, 10.0)  # update first leaf
        assert tree.total() > old_total


# ─────────────────────────────────────────────────────────────────────────────
# PrioritizedReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestPrioritizedReplayBuffer:

    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=10)
        import numpy as np
        for _ in range(20):
            buf.push(np.zeros(10), 0, 0.0, np.zeros(10), 0.0)
        assert len(buf) == 20

    def test_sample_returns_correct_keys(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=10)
        import numpy as np
        for _ in range(30):
            buf.push(np.random.randn(10), 0, 0.5, np.random.randn(10), 0.0)
        batch = buf.sample(8)
        assert "states" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_states" in batch
        assert "dones" in batch
        assert "weights" in batch
        assert "tree_indices" in batch

    def test_sample_shapes(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=16)
        import numpy as np
        for _ in range(50):
            buf.push(np.random.randn(16), 3, 1.0, np.random.randn(16), 0.0)
        batch = buf.sample(12)
        assert batch["states"].shape == (12, 16)
        assert batch["actions"].shape == (12,)
        assert batch["rewards"].shape == (12,)
        assert batch["weights"].shape == (12,)

    def test_is_weights_positive(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=8)
        import numpy as np
        for _ in range(50):
            buf.push(np.random.randn(8), 0, 0.0, np.random.randn(8), 0.0)
        batch = buf.sample(16)
        assert (batch["weights"] > 0).all(), "IS weights must be positive"
        assert torch.isfinite(batch["weights"]).all()

    def test_priority_update(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=8, alpha=1.0)
        import numpy as np
        for _ in range(30):
            buf.push(np.random.randn(8), 0, 0.0, np.random.randn(8), 0.0)

        batch = buf.sample(8)
        td_errors = torch.ones(8) * 10.0  # high TD error
        buf.update_priorities(batch["tree_indices"], td_errors)

        # After update, max priority should increase
        assert buf._max_priority > 1.0

    def test_push_batch(self):
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=8)
        states = torch.randn(10, 8)
        actions = torch.randint(0, 4, (10,))
        rewards = torch.randn(10)
        next_states = torch.randn(10, 8)
        dones = torch.zeros(10)
        buf.push_batch(states, actions, rewards, next_states, dones)
        assert len(buf) == 10

    def test_beta_annealing(self):
        buf = PrioritizedReplayBuffer(
            capacity=100, obs_dim=8,
            beta_start=0.4, beta_end=1.0, beta_frames=100,
        )
        import numpy as np
        for _ in range(50):
            buf.push(np.random.randn(8), 0, 0.0, np.random.randn(8), 0.0)

        beta_early = buf.beta
        for _ in range(50):
            buf.sample(4)
        beta_mid = buf.beta

        for _ in range(200):
            buf.sample(4)
        beta_late = buf.beta

        assert beta_late >= beta_mid >= beta_early, \
            f"Beta should anneal upward: {beta_early} → {beta_mid} → {beta_late}"


# ─────────────────────────────────────────────────────────────────────────────
# PER + Universal Agent Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPERAgent:

    def _make_agent(self):
        providers = [ProviderType.FR1, ProviderType.FR3]
        num_actions = len(providers) * 3
        cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=32,
            gnn_latent_dim=32, gnn_output_dim=64, temporal_hidden=32,
            latent_dim=64, num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=32, buffer_capacity=2000, batch_size=16,
            use_per=True, per_alpha=0.6, per_beta_start=0.4,
            per_beta_frames=1000, device="cpu",
        )
        return UniversalSpectrumAgent(cfg, num_actions), num_actions

    def test_per_agent_trains(self):
        agent, num_actions = self._make_agent()
        for _ in range(50):
            agent.push(
                torch.randn(4, 32), torch.randint(0, num_actions, (4,)),
                torch.randn(4), torch.randn(4, 32), torch.zeros(4),
            )
        result = agent.update()
        assert result is not None
        for k, v in result.items():
            assert math.isfinite(v), f"Non-finite: {k}={v}"

    def test_per_agent_multiple_updates(self):
        agent, num_actions = self._make_agent()
        for _ in range(200):
            agent.push(
                torch.randn(4, 32), torch.randint(0, num_actions, (4,)),
                torch.randn(4), torch.randn(4, 32), torch.zeros(4),
            )
        for i in range(20):
            result = agent.update()
            assert result is not None
            for k, v in result.items():
                assert math.isfinite(v), f"Non-finite at update {i}: {k}={v}"


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestCurriculumScheduler:

    def test_initial_stage(self):
        sched = CurriculumScheduler()
        assert sched.stage_index == 0
        assert sched.current_stage.name == "basic_dsa"
        assert len(sched.current_providers) == 2

    def test_stage_progression(self):
        config = CurriculumConfig(stages=[
            CurriculumStage("s1", [ProviderType.FR1], min_reward=-0.5, min_steps=10, window_size=5),
            CurriculumStage("s2", [ProviderType.FR1, ProviderType.FR3], min_reward=-999, min_steps=0),
        ])
        sched = CurriculumScheduler(config)
        assert sched.stage_index == 0

        # Feed enough good rewards to advance
        advanced = False
        for _ in range(20):
            advanced = sched.step(0.0)  # above threshold of -0.5
            if advanced:
                break

        assert advanced, "Curriculum should have advanced after sustained good reward"
        assert sched.stage_index == 1
        assert sched.current_stage.name == "s2"

    def test_no_advance_before_min_steps(self):
        config = CurriculumConfig(stages=[
            CurriculumStage("s1", [ProviderType.FR1], min_reward=-0.5, min_steps=100, window_size=5),
            CurriculumStage("s2", [ProviderType.FR1, ProviderType.FR3], min_reward=-999, min_steps=0),
        ])
        sched = CurriculumScheduler(config)

        # Even with great reward, shouldn't advance before min_steps
        for _ in range(50):
            sched.step(10.0)
        assert sched.stage_index == 0, "Should not advance before min_steps"

    def test_final_stage_no_advance(self):
        config = CurriculumConfig(stages=[
            CurriculumStage("only", list(ProviderType), min_reward=-999, min_steps=0),
        ])
        sched = CurriculumScheduler(config)
        assert sched.is_final_stage
        for _ in range(100):
            assert not sched.step(100.0)

    def test_get_stats(self):
        sched = CurriculumScheduler()
        for _ in range(5):
            sched.step(0.0)
        stats = sched.get_stats()
        assert "curriculum/stage" in stats
        assert "curriculum/stage_name" in stats
        assert "curriculum/num_providers" in stats

    def test_force_stage(self):
        sched = CurriculumScheduler()
        sched.force_stage(2)
        assert sched.stage_index == 2
        assert sched.current_stage.name == "full_ntn_sensing"

    def test_four_stage_full_progression(self):
        """Walk through all 4 default curriculum stages."""
        config = CurriculumConfig(stages=[
            CurriculumStage("s1", [ProviderType.FR1], min_reward=-1.0, min_steps=5, window_size=3),
            CurriculumStage("s2", [ProviderType.FR1, ProviderType.LEO], min_reward=-1.0, min_steps=5, window_size=3),
            CurriculumStage("s3", [ProviderType.FR1, ProviderType.LEO, ProviderType.GEO], min_reward=-1.0, min_steps=5, window_size=3),
            CurriculumStage("s4", list(ProviderType), min_reward=-999, min_steps=0),
        ])
        sched = CurriculumScheduler(config)

        stages_visited = [0]
        for _ in range(100):
            if sched.step(0.0):
                stages_visited.append(sched.stage_index)
            if sched.is_final_stage:
                break

        assert sched.is_final_stage, "Should reach final stage"
        assert stages_visited == [0, 1, 2, 3], f"Expected [0,1,2,3], got {stages_visited}"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: Curriculum + PER + UHCI
# ─────────────────────────────────────────────────────────────────────────────

class TestCurriculumPEREndToEnd:

    def test_curriculum_with_per_training(self):
        """
        Full end-to-end: curriculum schedules provider growth,
        PER prioritizes rare NTN events, agent trains throughout.
        """
        config = CurriculumConfig(stages=[
            CurriculumStage("basic", [ProviderType.FR1, ProviderType.FR3],
                            min_reward=-5.0, min_steps=20, window_size=10),
            CurriculumStage("ntn", [ProviderType.FR1, ProviderType.FR3,
                                     ProviderType.LEO, ProviderType.ISAC],
                            min_reward=-999, min_steps=0),
        ])
        sched = CurriculumScheduler(config)

        def make_env_and_agent(providers):
            env_cfg = UnifiedConnectivityConfig(
                num_envs=4, device="cpu",
                provider_types=providers, channels_per_provider=3,
            )
            env = UnifiedConnectivityEnv(env_cfg)
            agent_cfg = UniversalAgentConfig(
                provider_types=providers, raw_feature_dim=env.obs_dim,
                gnn_latent_dim=32, gnn_output_dim=64, temporal_hidden=32,
                latent_dim=64, num_gnn_layers=1, temporal_layers=1,
                kan_hidden_dim=32, buffer_capacity=5000, batch_size=16,
                use_per=True, per_alpha=0.6, per_beta_start=0.4,
                per_beta_frames=500, device="cpu",
            )
            agent = UniversalSpectrumAgent(agent_cfg, env.action_dim)
            return env, agent

        env, agent = make_env_and_agent(sched.current_providers)
        obs = env.reset()
        stage_transitions = 0
        updates = 0

        for step in range(80):
            actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(actions)
            agent.push(obs, actions, rewards, next_obs, dones)

            result = agent.update()
            if result is not None:
                updates += 1
                for k, v in result.items():
                    assert math.isfinite(v), f"Non-finite {k} at step {step}"

            # Curriculum check
            advanced = sched.step(rewards.mean().item())
            if advanced:
                stage_transitions += 1
                # Rebuild env and agent for new provider set
                env, agent = make_env_and_agent(sched.current_providers)
                obs = env.reset()
            else:
                obs = next_obs

        assert updates > 0, "No agent updates occurred"
        # May or may not have advanced — both are valid outcomes
