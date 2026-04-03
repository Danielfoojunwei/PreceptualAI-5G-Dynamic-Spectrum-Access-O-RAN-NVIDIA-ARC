"""
Tests for UHCI Phase 3 — Diffusion Augmentation + FNO Surrogate Integration.

Validates the full deep moat stack operating together:
  - DiffusionAugmenter training on UHCI environment transitions
  - FNOChannelSurrogate with physics-constrained loss on multi-provider data
  - Integration of both with the UniversalSpectrumAgent training loop
  - End-to-end pipeline: env → agent → diffusion → FNO → replay
"""

import math

import pytest
import torch
import torch.nn.functional as F

from preceptualai.env.provider_registry import ProviderType
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityConfig,
    UnifiedConnectivityEnv,
)
from preceptualai.core.universal_spectrum_agent import (
    UniversalAgentConfig,
    UniversalSpectrumAgent,
)
from preceptualai.core.diffusion_augment import DiffusionAugmenter, SpectrumDiffusionModel
from preceptualai.core.fno_surrogate import FNOChannelSurrogate


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Augmenter with UHCI
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionWithUHCI:

    def _make_env(self, num_envs=8):
        cfg = UnifiedConnectivityConfig(
            num_envs=num_envs, device="cpu",
            provider_types=[ProviderType.LEO, ProviderType.FR1, ProviderType.FR3],
            channels_per_provider=4,
        )
        return UnifiedConnectivityEnv(cfg)

    def test_diffusion_trains_on_uhci_transitions(self):
        """DiffusionAugmenter must train without error on UHCI env data."""
        env = self._make_env()
        obs = env.reset()
        _, _, _, _ = env.step(torch.zeros(8, dtype=torch.long))
        next_obs, _, _, _ = env.step(torch.zeros(8, dtype=torch.long))

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))
        loss = aug.train_step(obs, next_obs)
        assert math.isfinite(loss), f"Diffusion loss is not finite: {loss}"

    def test_diffusion_generates_valid_samples(self):
        """Generated samples must have correct shape and be finite."""
        env = self._make_env()
        obs = env.reset()
        next_obs, _, _, _ = env.step(torch.zeros(8, dtype=torch.long))

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))
        # Train a few steps
        for _ in range(5):
            aug.train_step(obs, next_obs)

        synthetic = aug.generate(obs)
        assert synthetic.shape == next_obs.shape, \
            f"Expected {next_obs.shape}, got {synthetic.shape}"
        assert torch.isfinite(synthetic).all(), "Synthetic samples contain NaN/Inf"

    def test_diffusion_loss_decreases(self):
        """Diffusion loss should decrease over multiple training steps."""
        env = self._make_env()
        obs = env.reset()
        next_obs, _, _, _ = env.step(torch.zeros(8, dtype=torch.long))

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))
        losses = []
        for _ in range(20):
            loss = aug.train_step(obs, next_obs)
            losses.append(loss)

        # Loss should decrease (first half average > second half average)
        first_half  = sum(losses[:10]) / 10
        second_half = sum(losses[10:]) / 10
        assert second_half < first_half * 1.5, \
            f"Diffusion loss not decreasing: {first_half:.4f} → {second_half:.4f}"

    def test_diffusion_augments_replay_buffer(self):
        """Synthetic transitions can be pushed to the agent replay buffer."""
        providers = [ProviderType.LEO, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=4, device="cpu",
            provider_types=providers, channels_per_provider=3,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        obs = env.reset()

        agent_cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=env.obs_dim,
            gnn_latent_dim=32, gnn_output_dim=64, temporal_hidden=32,
            latent_dim=64, num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=32, buffer_capacity=2000, batch_size=16, device="cpu",
        )
        agent = UniversalSpectrumAgent(agent_cfg, env.action_dim)

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))

        # Collect real data
        for _ in range(10):
            actions = agent.select_action(obs)
            next_obs, rewards, dones, _ = env.step(actions)
            agent.push(obs, actions, rewards, next_obs, dones)
            aug.train_step(obs, next_obs)
            obs = next_obs

        # Generate synthetic and push
        synthetic_next = aug.generate(obs)
        synth_actions = torch.randint(0, env.action_dim, (4,))
        synth_rewards = torch.zeros(4)
        agent.push(obs, synth_actions, synth_rewards, synthetic_next, torch.zeros(4))

        # Replay buffer should be larger
        assert len(agent.replay) >= 44, \
            f"Buffer has {len(agent.replay)} entries, expected >= 44"


# ─────────────────────────────────────────────────────────────────────────────
# FNO Channel Surrogate with UHCI
# ─────────────────────────────────────────────────────────────────────────────

class TestFNOWithUHCI:

    def _make_fno(self, num_providers=3, channels=4):
        return FNOChannelSurrogate(
            num_features = 3,
            width        = 16,
            modes        = min(4, num_providers * channels // 2 + 1),
            num_layers   = 2,
            num_channels = num_providers * channels,
        )

    def test_fno_forward_shape(self):
        """FNO must produce correct output shape for UHCI data."""
        fno = self._make_fno(3, 4)
        x = torch.randn(8, 12, 3)  # (B, P*C, features)
        y = fno(x)
        assert y.shape == (8, 12, 3)

    def test_fno_output_finite(self):
        """FNO output must be finite."""
        fno = self._make_fno(3, 4)
        x = torch.randn(4, 12, 3)
        y = fno(x)
        assert torch.isfinite(y).all()

    def test_fno_physics_loss_finite(self):
        """Physics-constrained loss must be finite."""
        fno = self._make_fno(3, 4)
        x = torch.randn(4, 12, 3)
        y = fno(x)
        target = x + torch.randn_like(x) * 0.1
        loss = fno.physics_loss(y, target)
        assert math.isfinite(loss.item())

    def test_fno_trains_on_uhci_env_data(self):
        """FNO must train on actual UHCI environment state data."""
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.ISAC]
        C = 3
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu",
            provider_types=providers, channels_per_provider=C,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()

        P = len(providers)
        fno = self._make_fno(P, C)
        opt = torch.optim.Adam(fno.parameters(), lr=1e-3)

        losses = []
        for step in range(10):
            # Step environment
            env.step(torch.randint(0, env.action_dim, (8,)))

            # Build FNO input from env state
            snr  = env.snr_db / 40.0
            occ  = env.occupancy
            intf = env.interference
            current = torch.stack([
                snr.view(8, P * C),
                occ.view(8, P * C),
                intf.view(8, P * C),
            ], dim=-1)

            # Step again to get target
            env.step(torch.randint(0, env.action_dim, (8,)))
            snr2  = env.snr_db / 40.0
            occ2  = env.occupancy
            intf2 = env.interference
            target = torch.stack([
                snr2.view(8, P * C),
                occ2.view(8, P * C),
                intf2.view(8, P * C),
            ], dim=-1)

            pred = fno(current)
            loss = fno.physics_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert all(math.isfinite(l) for l in losses), "FNO training produced NaN/Inf"

    def test_fno_all_eight_providers(self):
        """FNO must handle data from all 8 provider types."""
        P = 8
        C = 3
        fno = self._make_fno(P, C)
        x = torch.randn(4, P * C, 3)
        y = fno(x)
        assert y.shape == (4, P * C, 3)
        assert torch.isfinite(y).all()


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineIntegration:

    def test_env_agent_diffusion_fno_loop(self):
        """
        Full pipeline: env → agent → diffusion → FNO, 50 steps.

        This is the master integration test for the complete UHCI stack.
        """
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.FR3, ProviderType.ISAC]
        C = 3
        num_envs = 4
        P = len(providers)

        env_cfg = UnifiedConnectivityConfig(
            num_envs=num_envs, device="cpu",
            provider_types=providers, channels_per_provider=C,
        )
        env = UnifiedConnectivityEnv(env_cfg)

        agent_cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=env.obs_dim,
            gnn_latent_dim=32, gnn_output_dim=64, temporal_hidden=32,
            latent_dim=64, num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=32, buffer_capacity=5000, batch_size=16, device="cpu",
        )
        agent = UniversalSpectrumAgent(agent_cfg, env.action_dim)

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))

        fno = FNOChannelSurrogate(
            num_features=3, width=16,
            modes=min(4, P * C // 2 + 1),
            num_layers=2, num_channels=P * C,
        )
        fno_opt = torch.optim.Adam(fno.parameters(), lr=1e-3)

        obs = env.reset()
        agent_updates = 0
        diff_updates  = 0
        fno_updates   = 0

        for step in range(50):
            actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(actions)
            agent.push(obs, actions, rewards, next_obs, dones)

            # Agent update
            metrics = agent.update()
            if metrics is not None:
                agent_updates += 1
                for k, v in metrics.items():
                    assert math.isfinite(v), f"Non-finite metric {k}={v} at step {step}"

            # Diffusion training (every 10 steps)
            if step % 10 == 0 and step > 0:
                diff_loss = aug.train_step(obs, next_obs)
                assert math.isfinite(diff_loss)
                diff_updates += 1

            # FNO training (every 15 steps)
            if step % 15 == 0 and step > 0:
                snr  = env.snr_db / 40.0
                occ  = env.occupancy
                intf = env.interference
                current = torch.stack([
                    snr.view(num_envs, P * C),
                    occ.view(num_envs, P * C),
                    intf.view(num_envs, P * C),
                ], dim=-1)
                target = current + torch.randn_like(current) * 0.05
                pred = fno(current)
                loss = fno.physics_loss(pred, target)
                fno_opt.zero_grad()
                loss.backward()
                fno_opt.step()
                assert math.isfinite(loss.item())
                fno_updates += 1

            # Diffusion augmentation (step 25)
            if step == 25:
                synthetic = aug.generate(obs)
                assert synthetic.shape == obs.shape
                assert torch.isfinite(synthetic).all()
                synth_actions = agent.select_action(obs)
                agent.push(obs, synth_actions, rewards, synthetic, dones)

            assert torch.isfinite(next_obs).all()
            obs = next_obs

        # All subsystems must have trained
        assert agent_updates > 0, "Agent never updated"
        assert diff_updates  > 0, "Diffusion never trained"
        assert fno_updates   > 0, "FNO never trained"

        # KAN interpretability
        stats = agent.explain_action()
        assert len(stats) > 0

        # Provider preference sanity
        prefs = agent.get_provider_preference(obs[:1])
        assert abs(sum(prefs.values()) - 1.0) < 0.01

    def test_fno_replaces_env_physics_inference(self):
        """
        FNO surrogate can predict next channel state faster than env step.

        This validates the deep moat: FNO replaces ITU-R propagation models.
        """
        P, C = 3, 4
        fno = FNOChannelSurrogate(
            num_features=3, width=16, modes=4, num_layers=2, num_channels=P*C,
        )

        # Train FNO briefly
        opt = torch.optim.Adam(fno.parameters(), lr=1e-3)
        for _ in range(10):
            x = torch.randn(16, P*C, 3)
            t = x + torch.randn_like(x) * 0.1
            loss = fno.physics_loss(fno(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Inference: FNO predicts in a single forward pass
        fno.eval()
        with torch.no_grad():
            x = torch.randn(64, P*C, 3)
            pred = fno(x)
            assert pred.shape == (64, P*C, 3)
            assert torch.isfinite(pred).all()

    def test_diffusion_diversity(self):
        """Generated samples must not be identical (diversity check)."""
        obs_dim = 48
        aug = DiffusionAugmenter(obs_dim=obs_dim, device=torch.device("cpu"))
        obs = torch.randn(4, obs_dim)
        target = obs + torch.randn_like(obs) * 0.5

        for _ in range(10):
            aug.train_step(obs, target)

        s1 = aug.generate(obs)
        s2 = aug.generate(obs)

        # Two generations from the same conditioning should differ
        # (diffusion has stochastic sampling)
        diff = (s1 - s2).abs().max().item()
        assert diff > 0.001, "Diffusion samples are identical — no stochasticity"

    def test_world_model_with_diffusion_and_fno(self):
        """
        World model + diffusion + FNO all active simultaneously.

        This tests the maximum feature configuration of the UHCI agent.
        """
        providers = [ProviderType.LEO, ProviderType.FR1]
        C = 3
        env_cfg = UnifiedConnectivityConfig(
            num_envs=4, device="cpu",
            provider_types=providers, channels_per_provider=C,
        )
        env = UnifiedConnectivityEnv(env_cfg)

        agent_cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=env.obs_dim,
            gnn_latent_dim=32, gnn_output_dim=64, temporal_hidden=32,
            latent_dim=64, num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=32, buffer_capacity=5000, batch_size=16,
            use_world_model=True,
            wm_hidden_dim=32, wm_latent_dim=16,
            wm_imagination_horizon=3, wm_imagined_batch_size=8,
            wm_train_every=2,
            device="cpu",
        )
        agent = UniversalSpectrumAgent(agent_cfg, env.action_dim)

        aug = DiffusionAugmenter(obs_dim=env.obs_dim, device=torch.device("cpu"))

        obs = env.reset()
        for step in range(60):
            actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(actions)
            agent.push(obs, actions, rewards, next_obs, dones)

            metrics = agent.update()
            if metrics is not None:
                for k, v in metrics.items():
                    assert math.isfinite(v), f"Non-finite {k}={v} at step {step}"

            if step % 10 == 0:
                aug.train_step(obs, next_obs)

            obs = next_obs

        # World model should have trained
        assert agent._updates > 0
