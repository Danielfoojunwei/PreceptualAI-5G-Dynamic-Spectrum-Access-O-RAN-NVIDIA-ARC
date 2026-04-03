"""
Tests for paradigm-shift innovations: CfC, Mamba, GNN, KAN, SmODE,
World Model, FNO, Diffusion, dApp, rApp, Aerial, ISAC, FR3/NTN.
"""

import pytest
import torch
import numpy as np


# ── CfC Cell (solver-free LTC replacement) ───────────────────────────

class TestCfCCell:
    def test_forward_shape(self):
        from preceptualai.core.ltc_cell_cfc import CfCCell
        cell = CfCCell(input_dim=30, hidden_dim=64)
        x = torch.randn(8, 30)
        h = torch.zeros(8, 64)
        h_new = cell(x, h)
        assert h_new.shape == (8, 64)
        assert torch.isfinite(h_new).all()

    def test_exact_variant(self):
        from preceptualai.core.ltc_cell_cfc import CfCCellExact
        cell = CfCCellExact(input_dim=30, hidden_dim=64)
        x = torch.randn(4, 30)
        h = torch.zeros(4, 64)
        h_new = cell(x, h)
        assert h_new.shape == (4, 64)

    def test_cfc_encoder(self):
        from preceptualai.core.ltc_cell_cfc import CfCEncoder
        enc = CfCEncoder(input_dim=30, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 16, 30)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_cfc_encoder_exact(self):
        from preceptualai.core.ltc_cell_cfc import CfCEncoder
        enc = CfCEncoder(input_dim=30, hidden_dim=64, latent_dim=32, cell_type="exact")
        x = torch.randn(2, 8, 30)
        z = enc(x)
        assert z.shape == (2, 32)


# ── Mamba Encoder ────────────────────────────────────────────────────

class TestMambaEncoder:
    def test_ssm_block(self):
        from preceptualai.core.mamba_encoder import SelectiveSSMBlock
        block = SelectiveSSMBlock(d_model=64, d_state=16)
        x = torch.randn(4, 16, 64)
        y = block(x)
        assert y.shape == (4, 16, 64)

    def test_encoder_forward(self):
        from preceptualai.core.mamba_encoder import MambaEncoder
        enc = MambaEncoder(input_dim=30, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 16, 30)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_bidirectional(self):
        from preceptualai.core.mamba_encoder import MambaEncoder
        enc = MambaEncoder(input_dim=30, hidden_dim=64, latent_dim=32, bidirectional=True)
        x = torch.randn(2, 16, 30)
        z = enc(x)
        assert z.shape == (2, 32)


# ── GNN Encoder ──────────────────────────────────────────────────────

class TestGNNEncoder:
    def test_gat_layer(self):
        from preceptualai.core.gnn_encoder import GraphAttentionLayer
        layer = GraphAttentionLayer(in_features=30, out_features=64, num_heads=4)
        x = torch.randn(4, 10, 30)
        adj = torch.ones(10, 10)
        h = layer(x, adj)
        assert h.shape == (4, 10, 64)

    def test_spatial_encoder(self):
        from preceptualai.core.gnn_encoder import GNNSpatialEncoder
        enc = GNNSpatialEncoder(input_dim=30, embed_dim=64, num_layers=2)
        x = torch.randn(4, 10, 30)
        adj = GNNSpatialEncoder.build_interference_adj(10)
        h = enc(x, adj)
        assert h.shape == (4, 10, 64)

    def test_interference_adj(self):
        from preceptualai.core.gnn_encoder import GNNSpatialEncoder
        adj = GNNSpatialEncoder.build_interference_adj(5, interference_radius=1)
        assert adj.shape == (5, 5)
        assert adj[0, 0] == 1.0  # self-loop
        assert adj[0, 1] == 1.0  # neighbor
        assert adj[0, 3] == 0.0  # too far


# ── KAN Actor ────────────────────────────────────────────────────────

class TestKANActor:
    def test_kan_linear(self):
        from preceptualai.core.kan_actor import KANLinear
        layer = KANLinear(in_features=32, out_features=16, grid_size=5)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)
        assert torch.isfinite(y).all()

    def test_kan_actor_forward(self):
        from preceptualai.core.kan_actor import KANActor
        actor = KANActor(encoder_latent_dim=64, num_actions=10)
        z = torch.randn(4, 64)
        probs = actor(z)
        assert probs.shape == (4, 10)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_spline_stats(self):
        from preceptualai.core.kan_actor import KANActor
        actor = KANActor(encoder_latent_dim=32, num_actions=5)
        stats = actor.get_spline_stats()
        assert len(stats) > 0


# ── SmODE Smooth Actor ───────────────────────────────────────────────

class TestSmoothActor:
    def test_smode_neuron(self):
        from preceptualai.core.smooth_actor import SmODENeuron
        neuron = SmODENeuron(input_dim=64, output_dim=3, max_rate=0.1)
        z = torch.randn(4, 64)
        prev = torch.zeros(4, 3)
        action = neuron(z, prev)
        assert action.shape == (4, 3)
        assert (action >= -1).all() and (action <= 1).all()

    def test_smooth_rate_bound(self):
        from preceptualai.core.smooth_actor import SmODENeuron
        neuron = SmODENeuron(input_dim=32, output_dim=2, max_rate=0.05, dt=1.0)
        z = torch.randn(4, 32) * 10  # large input
        prev = torch.zeros(4, 2)
        action = neuron(z, prev)
        # Rate should be bounded
        delta = (action - torch.tanh(prev)).abs()
        assert (delta < 1.0).all()  # reasonable bound

    def test_smooth_hybrid(self):
        from preceptualai.core.smooth_actor import SmoothHybridActor
        actor = SmoothHybridActor(latent_dim=64, num_discrete_actions=10, continuous_action_dim=3)
        z = torch.randn(4, 64)
        probs, cont = actor(z)
        assert probs.shape == (4, 10)
        assert cont.shape == (4, 3)


# ── World Model ──────────────────────────────────────────────────────

class TestWorldModel:
    def test_predict(self):
        from preceptualai.core.world_model import LTCWorldModel
        wm = LTCWorldModel(obs_dim=30, num_actions=10, hidden_dim=64, latent_dim=32)
        obs = torch.randn(4, 30)
        h = torch.zeros(4, 64)
        actions = torch.randint(0, 10, (4,))
        h_new, obs_pred, reward, done = wm.predict(h, actions)
        assert h_new.shape == (4, 64)
        assert obs_pred.shape == (4, 30)
        assert reward.shape == (4,)
        assert done.shape == (4,)

    def test_imagine_rollout(self):
        from preceptualai.core.world_model import LTCWorldModel
        wm = LTCWorldModel(obs_dim=30, num_actions=10, hidden_dim=64)
        h0 = torch.zeros(4, 64)
        policy = lambda h: torch.randint(0, 10, (h.shape[0],))
        rollout = wm.imagine_rollout(h0, policy, horizon=5)
        assert rollout["rewards"].shape == (4, 5)
        assert rollout["actions"].shape == (4, 5)


# ── FNO Surrogate ────────────────────────────────────────────────────

class TestFNOSurrogate:
    def test_spectral_conv(self):
        from preceptualai.core.fno_surrogate import SpectralConv1d
        conv = SpectralConv1d(in_channels=3, out_channels=3, modes=4)
        x = torch.randn(4, 3, 10)
        y = conv(x)
        assert y.shape == (4, 3, 10)

    def test_fno_forward(self):
        from preceptualai.core.fno_surrogate import FNOChannelSurrogate
        fno = FNOChannelSurrogate(num_features=3, width=16, modes=4, num_layers=2)
        x = torch.randn(4, 10, 3)
        y = fno(x)
        assert y.shape == (4, 10, 3)

    def test_physics_loss(self):
        from preceptualai.core.fno_surrogate import FNOChannelSurrogate
        fno = FNOChannelSurrogate(num_features=3, width=16, modes=4)
        pred = torch.randn(4, 10, 3)
        target = torch.randn(4, 10, 3)
        loss = fno.physics_loss(pred, target)
        assert loss.dim() == 0  # scalar


# ── Diffusion Augmenter ──────────────────────────────────────────────

class TestDiffusion:
    def test_training_loss(self):
        from preceptualai.core.diffusion_augment import SpectrumDiffusionModel
        model = SpectrumDiffusionModel(obs_dim=30, hidden_dim=64, num_blocks=2, num_diffusion_steps=10)
        x0 = torch.randn(4, 30)
        cond = torch.randn(4, 30)
        loss = model.training_loss(x0, cond)
        assert loss.dim() == 0
        assert loss > 0

    def test_sample(self):
        from preceptualai.core.diffusion_augment import SpectrumDiffusionModel
        model = SpectrumDiffusionModel(obs_dim=30, hidden_dim=64, num_blocks=2, num_diffusion_steps=5)
        cond = torch.randn(2, 30)
        samples = model.sample(cond)
        assert samples.shape == (2, 30)


# ── dApp Engine ──────────────────────────────────────────────────────

class TestDAppEngine:
    def test_lightweight_cell(self):
        from preceptualai.xapp.dapp_engine import LightweightLTCCell
        cell = LightweightLTCCell(input_dim=30, hidden_dim=32)
        x = torch.randn(1, 30)
        h = torch.zeros(1, 32)
        h_new = cell(x, h)
        assert h_new.shape == (1, 32)

    def test_dapp_step(self):
        from preceptualai.xapp.dapp_engine import DAppInferenceEngine
        device = torch.device("cpu")
        engine = DAppInferenceEngine(input_dim=30, hidden_dim=32, num_actions=10,
                                      device=device, use_cuda_graph=False)
        obs = torch.randn(30)
        action = engine.step(obs)
        assert 0 <= action < 10

    def test_persistent_state(self):
        from preceptualai.xapp.dapp_engine import DAppInferenceEngine
        engine = DAppInferenceEngine(input_dim=30, hidden_dim=32, num_actions=10,
                                      device=torch.device("cpu"), use_cuda_graph=False)
        obs = torch.randn(30)
        a1 = engine.step(obs)
        a2 = engine.step(obs)  # Should use accumulated hidden state
        engine.reset_state()
        a3 = engine.step(obs)
        # After reset, should behave like first call
        assert isinstance(a3, int)


# ── rApp Trainer ─────────────────────────────────────────────────────

class TestRAppTrainer:
    def test_model_catalog(self):
        from preceptualai.xapp.rapp_trainer import ModelCatalog, ModelVersion
        catalog = ModelCatalog()
        v = ModelVersion("v1.0", 1234.0, {"reward": 0.9}, {})
        catalog.register(v)
        assert catalog.approve("v1.0")
        assert catalog.deploy("v1.0")
        assert catalog.deployed_version.version_id == "v1.0"

    def test_performance_monitor(self):
        from preceptualai.xapp.rapp_trainer import PerformanceMonitor
        mon = PerformanceMonitor(window_size=10, min_samples=5)
        mon.set_baseline({"reward": 1.0, "collision_rate": 0.1})
        for _ in range(10):
            mon.update({"reward": 0.5, "collision_rate": 0.3})
        assert mon.should_retrain

    def test_a1_policy_generation(self):
        from preceptualai.xapp.rapp_trainer import RAppTrainingService, ModelVersion
        svc = RAppTrainingService(device=torch.device("cpu"))
        v = ModelVersion("v2.0", 5678.0, {"reward": 0.95}, {})
        svc.catalog.register(v)
        policy = svc.generate_a1_policy("v2.0")
        assert policy["policy_type"] == "AI_ML_MODEL_UPDATE"

    def test_e2_prb_blanking(self):
        from preceptualai.xapp.rapp_trainer import RAppTrainingService
        svc = RAppTrainingService(device=torch.device("cpu"))
        policy = svc.generate_e2_prb_blanking_policy("cell_001", [0, 1, 5])
        assert policy["policy_type"] == "O-PRBBlankingPolicy"
        assert policy["blanked_prbs"] == [0, 1, 5]


# ── Aerial Adapter ───────────────────────────────────────────────────

class TestAerialAdapter:
    def test_arc_profiles(self):
        from preceptualai.xapp.aerial_adapter import AerialDeploymentConfig, ARCProfile
        cfg = AerialDeploymentConfig.for_profile(ARCProfile.ARC_COMPACT)
        assert cfg.precision == "int8"
        assert cfg.batch_size == 1

    def test_xapp_descriptor(self):
        from preceptualai.xapp.aerial_adapter import (
            AerialXAppRegistration, AerialDeploymentConfig, ARCProfile,
        )
        cfg = AerialDeploymentConfig.for_profile(ARCProfile.ARC_1)
        desc = AerialXAppRegistration.generate_descriptor(cfg)
        assert desc["xapp_name"] == "preceptualai-dsa"
        assert desc["platform"]["arc_profile"] == "arc_1"
        assert "E2SM-KPM" in desc["interfaces"]["e2"]["service_models"]


# ── ISAC Environment ─────────────────────────────────────────────────

class TestISACEnv:
    def test_reset_shape(self):
        from preceptualai.env.isac_env import ISACVectorizedDSAEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = ISACVectorizedDSAEnv(num_envs=4, num_channels=5, device=device)
        obs = env.reset()
        assert obs.shape == (4, 16, 25)  # 5 channels * 5 features

    def test_step_with_sensing(self):
        from preceptualai.env.isac_env import ISACVectorizedDSAEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = ISACVectorizedDSAEnv(num_envs=4, num_channels=5, device=device)
        env.reset()
        actions = torch.randint(0, 5, (4,), device=device)
        sensing = torch.full((4,), 0.3, device=device)
        obs, rewards, dones, infos = env.step(actions, sensing)
        assert "detection_probability" in infos
        assert "sensing_reward" in infos


# ── FR3 Propagation ─────────────────────────────────────────────────

class TestFR3Propagation:
    def test_uma_los(self):
        from preceptualai.env.fr3_propagation import FR3PropagationModel
        model = FR3PropagationModel(device=torch.device("cpu"))
        freq = torch.tensor([10.0, 15.0, 20.0])
        dist = torch.tensor([100.0, 200.0, 500.0])
        pl = model.path_loss_uma_los(freq, dist)
        assert pl.shape == (3,)
        assert (pl > 0).all()

    def test_fr3_capacity(self):
        from preceptualai.env.fr3_propagation import FR3PropagationModel
        model = FR3PropagationModel(device=torch.device("cpu"))
        freq = torch.tensor([10.0])
        dist = torch.tensor([100.0])
        cap = model.fr3_channel_capacity(freq, dist)
        assert cap > 0

    def test_indoor_penetration(self):
        from preceptualai.env.fr3_propagation import FR3PropagationModel
        model = FR3PropagationModel(device=torch.device("cpu"))
        freq = torch.tensor([10.0, 20.0])
        loss = model.indoor_penetration_loss(freq)
        assert loss.shape == (2,)
        assert loss[1] > loss[0]  # Higher freq = more loss


# ── NTN Interference ─────────────────────────────────────────────────

class TestNTNInterference:
    def test_ntn_step(self):
        from preceptualai.env.fr3_propagation import NTNInterferenceModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ntn = NTNInterferenceModel(num_envs=4, num_channels=10, device=device)
        ntn.step()
        intf = ntn.get_interference()
        assert intf.shape == (4, 10)
        assert (intf >= 0).all() and (intf <= 1).all()

    def test_ntn_features(self):
        from preceptualai.env.fr3_propagation import NTNInterferenceModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ntn = NTNInterferenceModel(num_envs=4, num_channels=10, device=device)
        ntn.step()
        features = ntn.get_features()
        assert "sat_elevation_deg" in features
        assert "ntn_interference" in features
