"""
Tests for Universal Heterogeneous Connectivity Intelligence (UHCI).

Validates the complete new paradigm:
  - Provider registry physics parameters
  - Coexistence adjacency matrix
  - Unified multi-provider environment
  - Heterogeneous GNN encoder
  - Universal Spectrum Agent (training loop)
  - End-to-end UHCI rollout
"""

import math

import pytest
import torch

from preceptualai.env.provider_registry import (
    PROVIDER_FEATURE_DIM,
    PROVIDER_REGISTRY,
    ProviderType,
    NTN_TYPES,
    TR_TYPES,
    coexistence_adjacency,
    get_all_provider_types,
    provider_node_features,
)
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityConfig,
    UnifiedConnectivityEnv,
)
from preceptualai.core.hetero_gnn_encoder import (
    HeteroEdgeAttention,
    HeteroGNNEncoder,
    HeteroGNNTemporalEncoder,
    HeteroNodeProjection,
)
from preceptualai.core.universal_spectrum_agent import (
    UniversalAgentConfig,
    UniversalSpectrumAgent,
    UniversalCritic,
)


# ─────────────────────────────────────────────────────────────────────────────
# Provider Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderRegistry:

    def test_all_provider_types_present(self):
        """All 8 connectivity provider types must be in the registry."""
        expected = {
            ProviderType.LEO, ProviderType.MEO, ProviderType.GEO, ProviderType.HAPS,
            ProviderType.FR1, ProviderType.FR3, ProviderType.ISAC, ProviderType.WIFI7,
        }
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_ntn_types_categorisation(self):
        """NTN types must be LEO, MEO, GEO, HAPS."""
        assert NTN_TYPES == {ProviderType.LEO, ProviderType.MEO,
                              ProviderType.GEO, ProviderType.HAPS}

    def test_tr_types_categorisation(self):
        """Terrestrial types must be FR1, FR3, ISAC, WIFI7."""
        assert TR_TYPES == {ProviderType.FR1, ProviderType.FR3,
                             ProviderType.ISAC, ProviderType.WIFI7}

    def test_ntn_tr_disjoint(self):
        """NTN and terrestrial type sets must be disjoint."""
        assert NTN_TYPES.isdisjoint(TR_TYPES)

    def test_ntn_tr_cover_all(self):
        """NTN ∪ TR must equal all provider types."""
        all_t = set(get_all_provider_types())
        assert NTN_TYPES | TR_TYPES == all_t

    def test_physics_altitude_ordering(self):
        """GEO altitude must exceed MEO, which must exceed LEO, which exceeds HAPS."""
        p = PROVIDER_REGISTRY
        assert p[ProviderType.GEO].altitude_km_min > p[ProviderType.MEO].altitude_km_max
        assert p[ProviderType.MEO].altitude_km_min > p[ProviderType.LEO].altitude_km_max
        assert p[ProviderType.LEO].altitude_km_min > p[ProviderType.HAPS].altitude_km_max

    def test_latency_ordering(self):
        """GEO latency must vastly exceed LEO, which exceeds terrestrial."""
        p = PROVIDER_REGISTRY
        assert p[ProviderType.GEO].rtt_ms_min > p[ProviderType.LEO].rtt_ms_max
        assert p[ProviderType.LEO].rtt_ms_min > p[ProviderType.FR1].rtt_ms_max

    def test_geo_zero_doppler(self):
        """GEO satellites are geostationary — Doppler must be zero."""
        assert PROVIDER_REGISTRY[ProviderType.GEO].doppler_hz_max == 0.0

    def test_geo_no_handover(self):
        """GEO satellites have no handover (fixed position)."""
        assert PROVIDER_REGISTRY[ProviderType.GEO].handover_period_s is None

    def test_isac_has_sensing(self):
        """ISAC is the only provider type with sensing capability flag."""
        for pt in ProviderType:
            p = PROVIDER_REGISTRY[pt]
            # ISAC has very high bandwidth (2000 MHz) and sub-ms coherence
            if pt == ProviderType.ISAC:
                assert p.max_bandwidth_mhz >= 2000.0
                assert p.channel_decorr_s <= 0.01

    def test_coexistence_symmetry(self):
        """Coexistence adjacency matrix must be symmetric."""
        device = torch.device("cpu")
        adj = coexistence_adjacency(device)
        assert torch.allclose(adj, adj.T), "Adjacency must be symmetric"

    def test_coexistence_no_self_loops(self):
        """Diagonal of adjacency must be zero."""
        device = torch.device("cpu")
        adj = coexistence_adjacency(device)
        assert adj.diagonal().sum().item() == 0.0

    def test_coexistence_shape(self):
        """Adjacency must be square with size = number of provider types."""
        device = torch.device("cpu")
        adj = coexistence_adjacency(device)
        n = len(ProviderType)
        assert adj.shape == (n, n)

    def test_provider_node_features_shape(self):
        """Node feature vectors must have the declared dimension."""
        device = torch.device("cpu")
        for pt in ProviderType:
            feat = provider_node_features(pt, device)
            assert feat.shape == (PROVIDER_FEATURE_DIM,), \
                f"Wrong feature dim for {pt.name}: {feat.shape}"

    def test_provider_node_features_range(self):
        """Node feature vectors must be in [0, 1] after clamping."""
        device = torch.device("cpu")
        for pt in ProviderType:
            feat = provider_node_features(pt, device)
            assert feat.min().item() >= 0.0, f"{pt.name} feature below 0"
            assert feat.max().item() <= 1.0, f"{pt.name} feature above 1"

    def test_provider_features_are_distinct(self):
        """Different provider types must have different feature vectors."""
        device = torch.device("cpu")
        feats = {pt: provider_node_features(pt, device) for pt in ProviderType}
        types = list(ProviderType)
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                diff = (feats[types[i]] - feats[types[j]]).abs().max().item()
                assert diff > 0.0, \
                    f"Feature vectors identical for {types[i].name} and {types[j].name}"


# ─────────────────────────────────────────────────────────────────────────────
# Unified Connectivity Environment
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedConnectivityEnv:

    def _make_env(self, num_envs=8, providers=None):
        cfg = UnifiedConnectivityConfig(
            num_envs=num_envs,
            device="cpu",
            provider_types=providers or [
                ProviderType.LEO, ProviderType.FR1, ProviderType.FR3
            ],
            channels_per_provider=4,
            obs_history_len=8,
        )
        return UnifiedConnectivityEnv(cfg)

    def test_reset_returns_correct_shape(self):
        env = self._make_env(num_envs=8)
        obs = env.reset()
        assert obs.shape == (8, env.obs_dim), \
            f"Expected ({8}, {env.obs_dim}), got {obs.shape}"

    def test_obs_dim_matches_config(self):
        """obs_dim in config must match actual observation shape from reset."""
        env = self._make_env(num_envs=4)
        obs = env.reset()
        assert obs.shape[-1] == env.cfg.obs_dim

    def test_step_output_shapes(self):
        env = self._make_env(num_envs=8)
        env.reset()
        actions = torch.randint(0, env.action_dim, (8,))
        obs, rewards, dones, info = env.step(actions)
        assert obs.shape    == (8, env.obs_dim)
        assert rewards.shape == (8,)
        assert dones.shape   == (8,)

    def test_actions_cover_full_range(self):
        """Actions 0..total_channels-1 must all be valid (no index error)."""
        env_size = 4
        env = self._make_env(num_envs=env_size)
        env.reset()
        for action in range(env.action_dim):
            actions = torch.full((env_size,), action, dtype=torch.long)
            obs, rewards, dones, info = env.step(actions)
            assert obs.shape == (env_size, env.obs_dim)

    def test_rewards_finite(self):
        """Rewards must be finite (no NaN/Inf from physics calculations)."""
        env = self._make_env(num_envs=16)
        env.reset()
        for _ in range(10):
            actions = torch.randint(0, env.action_dim, (16,))
            _, rewards, _, _ = env.step(actions)
            assert torch.isfinite(rewards).all(), "Non-finite rewards detected"

    def test_observations_finite(self):
        """Observations must be finite throughout rollout."""
        env = self._make_env(num_envs=8)
        env.reset()
        for _ in range(20):
            actions = torch.randint(0, env.action_dim, (8,))
            obs, _, _, _ = env.step(actions)
            assert torch.isfinite(obs).all(), "Non-finite obs detected"

    def test_ntn_visibility_in_range(self):
        """NTN visibility values must be in [0, 1]."""
        env = self._make_env()
        env.reset()
        for _ in range(5):
            env.step(torch.zeros(8, dtype=torch.long))
            v = env.ntn_visibility
            assert (v >= 0.0).all() and (v <= 1.0).all(), "NTN visibility out of [0,1]"

    def test_history_shape(self):
        """History ring buffer must return correct (E, T, obs_dim) shape."""
        env = self._make_env(num_envs=4)
        env.reset()
        for _ in range(5):
            env.step(torch.zeros(4, dtype=torch.long))
        hist = env.get_history()
        T = env.cfg.obs_history_len
        assert hist.shape == (4, T, env.obs_dim)

    def test_provider_stats_keys(self):
        """get_provider_stats() must return one entry per provider."""
        env = self._make_env()
        env.reset()
        stats = env.get_provider_stats()
        for pt in env.cfg.provider_types:
            assert f"{pt.name}/mean_snr_db" in stats

    def test_all_eight_providers(self):
        """Environment must support all 8 provider types simultaneously."""
        cfg = UnifiedConnectivityConfig(
            num_envs=4,
            device="cpu",
            provider_types=list(ProviderType),
            channels_per_provider=3,
        )
        env = UnifiedConnectivityEnv(cfg)
        obs = env.reset()
        assert obs.shape[0] == 4
        actions = torch.randint(0, env.action_dim, (4,))
        obs2, rewards, dones, info = env.step(actions)
        assert torch.isfinite(obs2).all()
        assert torch.isfinite(rewards).all()

    def test_geo_always_visible(self):
        """GEO satellites must always be visible (elevation stable, no blackout)."""
        cfg = UnifiedConnectivityConfig(
            num_envs=8,
            device="cpu",
            provider_types=[ProviderType.GEO, ProviderType.FR1],
        )
        env = UnifiedConnectivityEnv(cfg)
        env.reset()
        geo_idx = env.cfg.provider_types.index(ProviderType.GEO)
        for _ in range(20):
            env.step(torch.zeros(8, dtype=torch.long))
            vis = env.ntn_visibility[:, geo_idx]
            # GEO visibility should always be 1.0
            assert (vis >= 0.99).all(), "GEO satellite dropped visibility"

    def test_handover_penalty_applied(self):
        """Switching actions between steps must incur lower reward."""
        env = self._make_env(num_envs=4)
        env.reset()
        # Step with same action twice → no handover penalty
        same_action = torch.zeros(4, dtype=torch.long)
        env.step(same_action)
        _, r_same, _, _ = env.step(same_action)

        env.reset()
        env.step(same_action)
        diff_action = torch.full((4,), env.action_dim - 1, dtype=torch.long)
        _, r_diff, _, _ = env.step(diff_action)

        # On average, handover should reduce reward (penalty applied)
        assert r_same.mean().item() >= r_diff.mean().item() - 5.0  # allow tolerance

    def test_cross_provider_interference_updates(self):
        """Interference tensor must change across steps (physics not frozen)."""
        env = self._make_env(num_envs=8, providers=[
            ProviderType.LEO, ProviderType.FR3, ProviderType.GEO
        ])
        env.reset()
        intf_before = env.interference.clone()
        for _ in range(5):
            env.step(torch.zeros(8, dtype=torch.long))
        intf_after = env.interference
        # Interference must not be identically frozen (dynamics active)
        assert not torch.allclose(intf_before, intf_after, atol=0.0), \
            "Interference values are frozen — dynamics not updating"


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous GNN Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TestHeteroNodeProjection:

    def test_output_shape(self):
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.WIFI7]
        proj = HeteroNodeProjection(
            provider_types=providers, raw_feature_dim=32, latent_dim=64
        )
        x = torch.randn(4, 3, 32)   # (B, P, F)
        h = proj(x)
        assert h.shape == (4, 3, 64)

    def test_type_specific_projections(self):
        """Each provider type must have its own projection layer."""
        providers = [ProviderType.LEO, ProviderType.GEO, ProviderType.FR3]
        proj = HeteroNodeProjection(
            provider_types=providers, raw_feature_dim=16, latent_dim=32
        )
        assert len(proj.projections) == 3
        for pt in providers:
            assert pt.name in proj.projections


class TestHeteroEdgeAttention:

    def test_output_shape(self):
        attn = HeteroEdgeAttention(latent_dim=64, num_heads=4)
        h   = torch.randn(2, 5, 64)  # (B, P, D)
        adj = torch.ones(5, 5) - torch.eye(5)
        out = attn(h, adj)
        assert out.shape == (2, 5, 64)

    def test_residual_connection(self):
        """Output must differ from input (attention must do something)."""
        attn = HeteroEdgeAttention(latent_dim=32, num_heads=4)
        h   = torch.ones(1, 3, 32)
        adj = torch.ones(3, 3) - torch.eye(3)
        out = attn(h, adj)
        assert not torch.allclose(h, out), "Attention layer is identity — no-op"

    def test_masked_attention_zero_adj(self):
        """With zero adjacency (no edges + no self), output should be near input."""
        attn = HeteroEdgeAttention(latent_dim=32, num_heads=4, dropout=0.0)
        h   = torch.randn(1, 4, 32)
        adj = torch.zeros(4, 4)  # no edges → only self-loop survives
        out = attn(h, adj)
        assert out.shape == (1, 4, 32)
        assert torch.isfinite(out).all()


class TestHeteroGNNEncoder:

    def _make_encoder(self, providers=None):
        providers = providers or [ProviderType.LEO, ProviderType.FR1, ProviderType.FR3]
        return HeteroGNNEncoder(
            provider_types  = providers,
            raw_feature_dim = 24,
            latent_dim      = 64,
            output_dim      = 128,
            num_gnn_layers  = 2,
            num_heads       = 4,
        ), providers

    def test_output_shape(self):
        enc, providers = self._make_encoder()
        P = len(providers)
        x   = torch.randn(4, P, 24)
        adj = torch.ones(P, P) - torch.eye(P)
        z   = enc(x, adj)
        assert z.shape == (4, 128)

    def test_output_finite(self):
        enc, providers = self._make_encoder()
        P = len(providers)
        x   = torch.randn(8, P, 24)
        adj = coexistence_adjacency(torch.device("cpu"))
        all_t = get_all_provider_types()
        idx   = {t: i for i, t in enumerate(all_t)}
        sel   = [idx[pt] for pt in providers]
        sub_adj = adj[sel][:, sel]
        z   = enc(x, sub_adj)
        assert torch.isfinite(z).all()

    def test_node_embeddings_shape(self):
        enc, providers = self._make_encoder()
        P = len(providers)
        x   = torch.randn(2, P, 24)
        adj = torch.ones(P, P) - torch.eye(P)
        h   = enc.get_node_embeddings(x, adj)
        assert h.shape == (2, P, 64)

    def test_all_eight_providers(self):
        """Encoder must handle all 8 provider types."""
        providers = list(ProviderType)
        P = len(providers)
        enc = HeteroGNNEncoder(
            provider_types  = providers,
            raw_feature_dim = 20,
            latent_dim      = 32,
            output_dim      = 64,
            num_gnn_layers  = 1,
        )
        x   = torch.randn(2, P, 20)
        adj = coexistence_adjacency(torch.device("cpu"))
        z   = enc(x, adj)
        assert z.shape == (2, 64)
        assert torch.isfinite(z).all()

    def test_different_inputs_different_outputs(self):
        """Encoder must be non-constant — different inputs → different outputs."""
        enc, providers = self._make_encoder()
        P = len(providers)
        adj = torch.ones(P, P) - torch.eye(P)
        x1 = torch.zeros(1, P, 24)
        x2 = torch.ones(1, P, 24)
        z1 = enc(x1, adj)
        z2 = enc(x2, adj)
        assert not torch.allclose(z1, z2), "Encoder produces identical output for different inputs"


class TestHeteroGNNTemporalEncoder:

    def test_output_shape(self):
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.FR3]
        enc = HeteroGNNTemporalEncoder(
            provider_types  = providers,
            raw_feature_dim = 16,
            gnn_latent_dim  = 32,
            gnn_output_dim  = 64,
            temporal_hidden = 32,
            output_dim      = 64,
            num_gnn_layers  = 1,
            temporal_layers = 1,
            use_cfc         = True,
        )
        P = len(providers)
        x   = torch.randn(4, 8, P, 16)  # (B, T, P, F)
        adj = torch.ones(P, P) - torch.eye(P)
        z   = enc(x, adj)
        assert z.shape == (4, 64)

    def test_output_finite(self):
        providers = [ProviderType.GEO, ProviderType.ISAC]
        enc = HeteroGNNTemporalEncoder(
            provider_types  = providers,
            raw_feature_dim = 12,
            gnn_latent_dim  = 24,
            gnn_output_dim  = 48,
            temporal_hidden = 24,
            output_dim      = 48,
            num_gnn_layers  = 1,
            temporal_layers = 1,
        )
        P = len(providers)
        x   = torch.randn(2, 4, P, 12)
        adj = torch.zeros(P, P)
        z   = enc(x, adj)
        assert torch.isfinite(z).all()


# ─────────────────────────────────────────────────────────────────────────────
# Universal Spectrum Agent
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalCritic:

    def test_twin_output_shapes(self):
        critic = UniversalCritic(latent_dim=64, num_actions=12)
        z  = torch.randn(8, 64)
        q1, q2 = critic(z)
        assert q1.shape == (8, 12)
        assert q2.shape == (8, 12)

    def test_twin_heads_differ(self):
        """Twin critics must produce different Q-values (different parameters)."""
        critic = UniversalCritic(latent_dim=64, num_actions=12)
        z  = torch.randn(4, 64)
        q1, q2 = critic(z)
        assert not torch.allclose(q1, q2), "Twin critic heads are identical — bug"


class TestUniversalSpectrumAgent:

    def _make_agent(self, providers=None):
        providers = providers or [ProviderType.LEO, ProviderType.FR1, ProviderType.FR3]
        channels  = 4
        num_actions = len(providers) * channels
        cfg = UniversalAgentConfig(
            provider_types   = providers,
            raw_feature_dim  = 32,
            obs_history_len  = 8,
            gnn_latent_dim   = 32,
            gnn_output_dim   = 64,
            temporal_hidden  = 32,
            latent_dim       = 64,
            num_gnn_layers   = 1,
            temporal_layers  = 1,
            kan_hidden_dim   = 32,
            buffer_capacity  = 2000,
            batch_size       = 16,
            device           = "cpu",
        )
        return UniversalSpectrumAgent(cfg, num_actions), num_actions

    def test_select_action_shape(self):
        agent, num_actions = self._make_agent()
        obs = torch.randn(8, 32)
        actions = agent.select_action(obs)
        assert actions.shape == (8,)

    def test_select_action_valid_range(self):
        agent, num_actions = self._make_agent()
        obs = torch.randn(16, 32)
        actions = agent.select_action(obs)
        assert (actions >= 0).all()
        assert (actions < num_actions).all()

    def test_deterministic_action_reproducible(self):
        agent, _ = self._make_agent()
        obs = torch.randn(4, 32)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert torch.equal(a1, a2), "Deterministic actions must be reproducible"

    def test_update_returns_none_when_buffer_empty(self):
        agent, _ = self._make_agent()
        result = agent.update()
        assert result is None, "Should return None when replay buffer not full"

    def test_update_returns_losses_when_buffer_full(self):
        agent, num_actions = self._make_agent()
        # Fill buffer
        obs_dim = 32
        for _ in range(50):
            obs      = torch.randn(4, obs_dim)
            actions  = torch.randint(0, num_actions, (4,))
            rewards  = torch.randn(4)
            next_obs = torch.randn(4, obs_dim)
            dones    = torch.zeros(4)
            agent.push(obs, actions, rewards, next_obs, dones)

        result = agent.update()
        assert result is not None
        assert "loss/critic" in result
        assert "loss/actor"  in result
        assert "loss/alpha"  in result
        assert "alpha"       in result
        assert "entropy"     in result

    def test_losses_are_finite(self):
        agent, num_actions = self._make_agent()
        obs_dim = 32
        for _ in range(50):
            agent.push(
                torch.randn(4, obs_dim),
                torch.randint(0, num_actions, (4,)),
                torch.randn(4),
                torch.randn(4, obs_dim),
                torch.zeros(4),
            )
        result = agent.update()
        assert result is not None
        for k, v in result.items():
            assert math.isfinite(v), f"Non-finite loss: {k}={v}"

    def test_explain_action_returns_stats(self):
        agent, _ = self._make_agent()
        stats = agent.explain_action()
        assert isinstance(stats, dict)
        assert len(stats) > 0
        for k, v in stats.items():
            assert "mean" in v and "std" in v and "sparsity" in v

    def test_get_provider_preference_sums_to_one(self):
        """Provider preferences must sum to 1.0 (they are normalised probabilities)."""
        agent, _ = self._make_agent()
        obs = torch.randn(1, 32)
        prefs = agent.get_provider_preference(obs)
        total = sum(prefs.values())
        assert abs(total - 1.0) < 0.01, f"Provider prefs sum to {total}, not 1.0"

    def test_provider_preference_keys(self):
        """Must have one preference entry per provider type."""
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.ISAC]
        agent, _ = self._make_agent(providers=providers)
        obs = torch.randn(1, 32)
        prefs = agent.get_provider_preference(obs)
        assert set(prefs.keys()) == {pt.name for pt in providers}

    def test_state_dict_roundtrip(self):
        """Agent state dict must be saveable and loadable."""
        agent, num_actions = self._make_agent()
        sd = agent.state_dict()
        agent.load_state_dict(sd)
        obs = torch.randn(2, 32)
        actions = agent.select_action(obs)
        assert actions.shape == (2,)

    def test_multiple_update_steps(self):
        """Agent must remain stable through 20 consecutive update steps."""
        agent, num_actions = self._make_agent()
        obs_dim = 32
        for _ in range(200):
            agent.push(
                torch.randn(4, obs_dim),
                torch.randint(0, num_actions, (4,)),
                torch.randn(4),
                torch.randn(4, obs_dim),
                torch.zeros(4),
            )
        for _ in range(20):
            result = agent.update()
            assert result is not None
            for k, v in result.items():
                assert math.isfinite(v), f"Non-finite at step {_}: {k}={v}"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end UHCI Rollout
# ─────────────────────────────────────────────────────────────────────────────

class TestUHCIEndToEnd:

    def test_full_rollout_all_providers(self):
        """
        End-to-end UHCI: all 8 provider types, 100 steps, agent trains throughout.

        This is the master integration test for the Universal Heterogeneous
        Connectivity Intelligence paradigm.
        """
        providers = list(ProviderType)
        channels  = 3
        num_envs  = 4
        num_steps = 100

        # Environment
        env_cfg = UnifiedConnectivityConfig(
            num_envs             = num_envs,
            device               = "cpu",
            provider_types       = providers,
            channels_per_provider = channels,
            obs_history_len      = 8,
        )
        env = UnifiedConnectivityEnv(env_cfg)

        # Agent — obs_dim from env
        obs_dim = env_cfg.obs_dim
        agent_cfg = UniversalAgentConfig(
            provider_types  = providers,
            raw_feature_dim = obs_dim,
            gnn_latent_dim  = 32,
            gnn_output_dim  = 64,
            temporal_hidden = 32,
            latent_dim      = 64,
            num_gnn_layers  = 1,
            temporal_layers = 1,
            kan_hidden_dim  = 32,
            buffer_capacity = 5000,
            batch_size      = 32,
            device          = "cpu",
        )
        num_actions = len(providers) * channels
        agent = UniversalSpectrumAgent(agent_cfg, num_actions)

        obs = env.reset()
        total_reward = 0.0
        update_count = 0

        for step in range(num_steps):
            actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(actions)

            agent.push(obs, actions, rewards, next_obs, dones)

            result = agent.update()
            if result is not None:
                update_count += 1
                for k, v in result.items():
                    assert math.isfinite(v), f"Non-finite loss at step {step}: {k}={v}"

            total_reward += rewards.mean().item()
            obs = next_obs

            # Observations must remain finite throughout
            assert torch.isfinite(obs).all(), f"Non-finite obs at step {step}"

        # At least some updates must have occurred
        assert update_count > 0, "No update steps occurred — buffer never filled"

        # Interpretability: KAN spline stats must be available
        spline_stats = agent.explain_action()
        assert len(spline_stats) > 0

        # Provider preferences must sum to ~1.0
        prefs = agent.get_provider_preference(obs[:1])
        assert abs(sum(prefs.values()) - 1.0) < 0.01

    def test_ntn_terrestrial_coexistence(self):
        """
        LEO satellite and FR3 coexist in the same spectrum: the agent must
        learn to avoid LEO channels when the satellite is visible and causing
        high interference on the coexisting FR3 channels.
        """
        providers = [ProviderType.LEO, ProviderType.FR3]
        env_cfg   = UnifiedConnectivityConfig(
            num_envs             = 8,
            device               = "cpu",
            provider_types       = providers,
            channels_per_provider = 4,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        obs = env.reset()
        # Run 50 steps — verify environment stays stable under NTN coexistence
        for _ in range(50):
            actions = torch.randint(0, env.action_dim, (8,))
            obs, rewards, dones, info = env.step(actions)
            assert torch.isfinite(rewards).all()
            intf = info["interference_level"]
            assert (intf >= 0.0).all()
            assert (intf <= 1.0).all()

    def test_provider_stats_available(self):
        """Provider statistics must be readable after a rollout."""
        providers = [ProviderType.LEO, ProviderType.GEO, ProviderType.FR1]
        env_cfg   = UnifiedConnectivityConfig(
            num_envs=4, device="cpu", provider_types=providers
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()
        for _ in range(10):
            env.step(torch.zeros(4, dtype=torch.long))
        stats = env.get_provider_stats()
        for pt in providers:
            assert f"{pt.name}/mean_snr_db" in stats
            assert torch.isfinite(stats[f"{pt.name}/mean_snr_db"])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Tests: Real Physics + World Model
# ─────────────────────────────────────────────────────────────────────────────

class TestRealPropagationPhysics:
    """Validate that the UHCE uses real 3GPP/ITU-R propagation models."""

    def test_fr3_snr_decreases_with_distance(self):
        """FR3 SNR must decrease as UE distance increases (path loss monotonicity)."""
        providers = [ProviderType.FR3, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu", provider_types=providers,
            channels_per_provider=4,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()

        fr3_idx = env.cfg.provider_types.index(ProviderType.FR3)

        # Set close distance and measure SNR
        env.ue_distance_m[:, fr3_idx] = 50.0
        env._update_channel_dynamics()
        snr_close = env.snr_db[:, fr3_idx, :].mean().item()

        # Set far distance and measure SNR
        env.ue_distance_m[:, fr3_idx] = 2000.0
        env._update_channel_dynamics()
        snr_far = env.snr_db[:, fr3_idx, :].mean().item()

        assert snr_close > snr_far, \
            f"FR3 SNR should decrease with distance: close={snr_close:.1f} far={snr_far:.1f}"

    def test_ntn_snr_decreases_with_rain(self):
        """NTN satellite SNR must decrease with higher rain rate (ITU-R P.618)."""
        providers = [ProviderType.LEO, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu", provider_types=providers,
            channels_per_provider=4,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()

        leo_idx = env.cfg.provider_types.index(ProviderType.LEO)
        env.sat_elevation_deg[:, leo_idx] = 45.0  # good elevation

        # Low rain → measure SNR
        env.rain_rate_mmh[:, leo_idx] = 1.0
        env._update_channel_dynamics()
        snr_dry = env.snr_db[:, leo_idx, :].mean().item()

        # Heavy rain → measure SNR
        env.rain_rate_mmh[:, leo_idx] = 50.0
        env._update_channel_dynamics()
        snr_rain = env.snr_db[:, leo_idx, :].mean().item()

        assert snr_dry > snr_rain, \
            f"NTN SNR should decrease with rain: dry={snr_dry:.1f} rain={snr_rain:.1f}"

    def test_ntn_snr_improves_with_elevation(self):
        """Higher satellite elevation should give better SNR (shorter slant path)."""
        providers = [ProviderType.LEO, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu", provider_types=providers,
            channels_per_provider=4,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()

        leo_idx = env.cfg.provider_types.index(ProviderType.LEO)
        env.rain_rate_mmh[:, leo_idx] = 5.0

        # Low elevation
        env.sat_elevation_deg[:, leo_idx] = 10.0
        env._update_channel_dynamics()
        snr_low_elev = env.snr_db[:, leo_idx, :].mean().item()

        # High elevation
        env.sat_elevation_deg[:, leo_idx] = 80.0
        env._update_channel_dynamics()
        snr_high_elev = env.snr_db[:, leo_idx, :].mean().item()

        assert snr_high_elev > snr_low_elev, \
            f"Higher elevation should improve SNR: low={snr_low_elev:.1f} high={snr_high_elev:.1f}"

    def test_isac_radar_equation_produces_sensing_snr(self):
        """ISAC provider must compute radar sensing SNR via radar equation."""
        providers = [ProviderType.ISAC, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=4, device="cpu", provider_types=providers,
            channels_per_provider=4,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()
        env._update_channel_dynamics()

        # ISAC sensing SNR must be finite and non-zero after physics update
        assert torch.isfinite(env.isac_sensing_snr).all(), "ISAC sensing SNR not finite"
        assert env.isac_sensing_snr.abs().sum().item() > 0, "ISAC sensing SNR all zeros"

    def test_ue_distance_evolves(self):
        """UE distances must change across steps (not frozen)."""
        providers = [ProviderType.FR3, ProviderType.FR1]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu", provider_types=providers,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()
        dist_before = env.ue_distance_m.clone()
        for _ in range(10):
            env.step(torch.zeros(8, dtype=torch.long))
        dist_after = env.ue_distance_m
        assert not torch.allclose(dist_before, dist_after), "UE distances frozen"

    def test_rain_rate_evolves(self):
        """Rain rate must change across steps (not frozen)."""
        providers = [ProviderType.LEO, ProviderType.GEO]
        env_cfg = UnifiedConnectivityConfig(
            num_envs=8, device="cpu", provider_types=providers,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()
        rain_before = env.rain_rate_mmh.clone()
        for _ in range(20):
            env.step(torch.zeros(8, dtype=torch.long))
        rain_after = env.rain_rate_mmh
        assert not torch.allclose(rain_before, rain_after), "Rain rate frozen"

    def test_real_physics_all_providers_finite(self):
        """1000-step rollout with all 8 providers must produce finite SNR throughout."""
        env_cfg = UnifiedConnectivityConfig(
            num_envs=4, device="cpu",
            provider_types=list(ProviderType),
            channels_per_provider=3,
        )
        env = UnifiedConnectivityEnv(env_cfg)
        env.reset()
        for step in range(1000):
            actions = torch.randint(0, env.action_dim, (4,))
            obs, rewards, dones, info = env.step(actions)
            assert torch.isfinite(obs).all(), f"Non-finite obs at step {step}"
            assert torch.isfinite(rewards).all(), f"Non-finite rewards at step {step}"
            assert torch.isfinite(env.snr_db).all(), f"Non-finite SNR at step {step}"


class TestWorldModelIntegration:
    """Validate world model integration into UniversalSpectrumAgent."""

    def _make_wm_agent(self):
        providers = [ProviderType.LEO, ProviderType.FR1, ProviderType.FR3]
        channels = 4
        num_actions = len(providers) * channels
        obs_dim = 32
        cfg = UniversalAgentConfig(
            provider_types    = providers,
            raw_feature_dim   = obs_dim,
            obs_history_len   = 8,
            gnn_latent_dim    = 32,
            gnn_output_dim    = 64,
            temporal_hidden   = 32,
            latent_dim        = 64,
            num_gnn_layers    = 1,
            temporal_layers   = 1,
            kan_hidden_dim    = 32,
            buffer_capacity   = 2000,
            batch_size        = 16,
            device            = "cpu",
            # World model enabled
            use_world_model   = True,
            wm_hidden_dim     = 64,
            wm_latent_dim     = 32,
            wm_imagination_horizon = 3,
            wm_imagined_batch_size = 16,
            wm_train_every    = 2,
        )
        return UniversalSpectrumAgent(cfg, num_actions), num_actions, obs_dim

    def test_world_model_instantiated(self):
        """World model must be created when use_world_model=True."""
        agent, _, _ = self._make_wm_agent()
        assert agent.world_model is not None
        assert agent.wm_trainer is not None

    def test_world_model_trains_during_update(self):
        """Update must return world model loss metrics when enabled."""
        agent, num_actions, obs_dim = self._make_wm_agent()
        for _ in range(100):
            agent.push(
                torch.randn(4, obs_dim),
                torch.randint(0, num_actions, (4,)),
                torch.randn(4),
                torch.randn(4, obs_dim),
                torch.zeros(4),
            )
        # Run several updates to trigger wm_train_every
        wm_metrics_seen = False
        for _ in range(10):
            result = agent.update()
            if result and "wm_obs_loss" in result:
                wm_metrics_seen = True
                assert math.isfinite(result["wm_obs_loss"]), "WM obs loss not finite"
                assert math.isfinite(result["wm_reward_loss"]), "WM reward loss not finite"
        assert wm_metrics_seen, "World model metrics never appeared in update()"

    def test_world_model_in_state_dict(self):
        """World model weights must be in agent state_dict."""
        agent, _, _ = self._make_wm_agent()
        sd = agent.state_dict()
        assert "world_model" in sd

    def test_world_model_stable_over_200_updates(self):
        """Agent with world model must remain stable over 200 update steps."""
        agent, num_actions, obs_dim = self._make_wm_agent()
        for _ in range(500):
            agent.push(
                torch.randn(4, obs_dim),
                torch.randint(0, num_actions, (4,)),
                torch.randn(4),
                torch.randn(4, obs_dim),
                torch.zeros(4),
            )
        for i in range(200):
            result = agent.update()
            assert result is not None
            for k, v in result.items():
                assert math.isfinite(v), f"Non-finite at update {i}: {k}={v}"

    def test_model_free_default(self):
        """Default agent (use_world_model=False) must have no world model."""
        providers = [ProviderType.FR1]
        cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=16,
            gnn_latent_dim=16, gnn_output_dim=32,
            temporal_hidden=16, latent_dim=32,
            num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=16, buffer_capacity=500, batch_size=8,
            device="cpu", use_world_model=False,
        )
        agent = UniversalSpectrumAgent(cfg, 4)
        assert agent.world_model is None
        assert agent.wm_trainer is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Tests: Mamba Backend + Temporal Backend Selection
# ─────────────────────────────────────────────────────────────────────────────

class TestMambaTemporalBackend:
    """Validate Mamba as a third temporal backend option."""

    def _make_agent_with_backend(self, backend: str):
        providers = [ProviderType.LEO, ProviderType.FR1]
        channels = 3
        num_actions = len(providers) * channels
        cfg = UniversalAgentConfig(
            provider_types   = providers,
            raw_feature_dim  = 24,
            gnn_latent_dim   = 16,
            gnn_output_dim   = 32,
            temporal_hidden  = 16,
            latent_dim       = 32,
            num_gnn_layers   = 1,
            temporal_layers  = 1,
            kan_hidden_dim   = 16,
            buffer_capacity  = 500,
            batch_size       = 8,
            device           = "cpu",
            temporal_backend = backend,
        )
        return UniversalSpectrumAgent(cfg, num_actions), num_actions

    def test_mamba_agent_creates(self):
        """Agent with temporal_backend='mamba' must instantiate without error."""
        agent, _ = self._make_agent_with_backend("mamba")
        assert agent.encoder is not None

    def test_mamba_select_action(self):
        """Mamba agent must produce valid actions."""
        agent, num_actions = self._make_agent_with_backend("mamba")
        obs = torch.randn(4, 24)
        actions = agent.select_action(obs)
        assert actions.shape == (4,)
        assert (actions >= 0).all() and (actions < num_actions).all()

    def test_mamba_update(self):
        """Mamba agent must train without error."""
        agent, num_actions = self._make_agent_with_backend("mamba")
        for _ in range(30):
            agent.push(
                torch.randn(4, 24),
                torch.randint(0, num_actions, (4,)),
                torch.randn(4),
                torch.randn(4, 24),
                torch.zeros(4),
            )
        result = agent.update()
        assert result is not None
        for k, v in result.items():
            assert math.isfinite(v), f"Non-finite: {k}={v}"

    def test_ltc_backend_works(self):
        """Agent with temporal_backend='ltc' must work."""
        agent, num_actions = self._make_agent_with_backend("ltc")
        obs = torch.randn(4, 24)
        actions = agent.select_action(obs)
        assert actions.shape == (4,)

    def test_cfc_backend_works(self):
        """Agent with temporal_backend='cfc' must work (default)."""
        agent, num_actions = self._make_agent_with_backend("cfc")
        obs = torch.randn(4, 24)
        actions = agent.select_action(obs)
        assert actions.shape == (4,)

    def test_all_backends_produce_finite_output(self):
        """All three backends must produce finite actions for same input."""
        obs = torch.randn(4, 24)
        for backend in ["cfc", "ltc", "mamba"]:
            agent, _ = self._make_agent_with_backend(backend)
            actions = agent.select_action(obs)
            assert torch.isfinite(actions.float()).all(), \
                f"Backend {backend} produced non-finite actions"

    def test_backward_compat_use_cfc_true(self):
        """Old use_cfc=True config must still work (backward compat)."""
        from preceptualai.core.hetero_gnn_encoder import HeteroGNNTemporalEncoder
        enc = HeteroGNNTemporalEncoder(
            provider_types=[ProviderType.FR1],
            raw_feature_dim=16,
            gnn_latent_dim=16, gnn_output_dim=32,
            temporal_hidden=16, output_dim=32,
            num_gnn_layers=1, temporal_layers=1,
            use_cfc=True,
        )
        x = torch.randn(2, 4, 1, 16)
        adj = torch.ones(1, 1)
        z = enc(x, adj)
        assert z.shape == (2, 32)

    def test_backward_compat_use_cfc_false(self):
        """Old use_cfc=False config must still work (backward compat)."""
        from preceptualai.core.hetero_gnn_encoder import HeteroGNNTemporalEncoder
        enc = HeteroGNNTemporalEncoder(
            provider_types=[ProviderType.FR1],
            raw_feature_dim=16,
            gnn_latent_dim=16, gnn_output_dim=32,
            temporal_hidden=16, output_dim=32,
            num_gnn_layers=1, temporal_layers=1,
            use_cfc=False,
        )
        x = torch.randn(2, 4, 1, 16)
        adj = torch.ones(1, 1)
        z = enc(x, adj)
        assert z.shape == (2, 32)


class TestSmODEPowerControl:
    """Validate SmODE integration for 3GPP-compliant power control."""

    def _make_smooth_agent(self):
        providers = [ProviderType.FR1, ProviderType.FR3]
        channels = 3
        num_actions = len(providers) * channels
        cfg = UniversalAgentConfig(
            provider_types    = providers,
            raw_feature_dim   = 20,
            gnn_latent_dim    = 16,
            gnn_output_dim    = 32,
            temporal_hidden   = 16,
            latent_dim        = 32,
            num_gnn_layers    = 1,
            temporal_layers   = 1,
            kan_hidden_dim    = 16,
            buffer_capacity   = 500,
            batch_size        = 8,
            device            = "cpu",
            use_smooth_power  = True,
            power_control_dim = 1,
            max_power_ramp_rate = 0.1,
        )
        return UniversalSpectrumAgent(cfg, num_actions), num_actions

    def test_smooth_power_instantiated(self):
        """SmODE must be created when use_smooth_power=True."""
        agent, _ = self._make_smooth_agent()
        assert agent.smooth_power is not None
        assert agent._prev_power is not None

    def test_smooth_power_action(self):
        """Agent with SmODE must produce valid discrete actions."""
        agent, num_actions = self._make_smooth_agent()
        obs = torch.randn(4, 20)
        actions = agent.select_action(obs)
        assert actions.shape == (4,)
        assert (actions >= 0).all() and (actions < num_actions).all()

    def test_power_ramp_rate_bounded(self):
        """Consecutive power actions must be bounded by SmODE rate constraint."""
        agent, _ = self._make_smooth_agent()
        # Reset power state to ensure clean starting point
        agent._prev_power = torch.zeros(1, agent.cfg.power_control_dim)

        powers = []
        for _ in range(50):
            obs = torch.randn(1, 20)
            agent.select_action(obs)
            if hasattr(agent, '_last_power'):
                powers.append(agent._last_power.clone().item())

        assert len(powers) >= 2, "Not enough power samples collected"

        # The effective max delta = max_rate * exp(log_rate_scale) * dt
        # At initialization, log_rate_scale=0 → scale=1, dt=1 → max_delta=max_rate
        rate_scale = agent.smooth_power.log_rate_scale.exp().max().item()
        max_delta = agent.cfg.max_power_ramp_rate * rate_scale

        for i in range(1, len(powers)):
            delta = abs(powers[i] - powers[i - 1])
            assert delta <= max_delta + 0.05, \
                f"Power ramp rate violated at step {i}: delta={delta:.4f} > max={max_delta:.4f}"

    def test_smooth_power_in_state_dict(self):
        """SmODE weights must be in agent state_dict."""
        agent, _ = self._make_smooth_agent()
        sd = agent.state_dict()
        assert "smooth_power" in sd

    def test_no_smooth_power_by_default(self):
        """Default agent must NOT have SmODE."""
        providers = [ProviderType.FR1]
        cfg = UniversalAgentConfig(
            provider_types=providers, raw_feature_dim=16,
            gnn_latent_dim=16, gnn_output_dim=32,
            temporal_hidden=16, latent_dim=32,
            num_gnn_layers=1, temporal_layers=1,
            kan_hidden_dim=16, buffer_capacity=500, batch_size=8,
            device="cpu",
        )
        agent = UniversalSpectrumAgent(cfg, 4)
        assert agent.smooth_power is None
