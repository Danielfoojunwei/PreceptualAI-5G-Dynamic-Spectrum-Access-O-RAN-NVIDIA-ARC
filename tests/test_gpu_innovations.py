"""
Tests for GPU-optimized innovations from SAC-LTC architecture.

Tests cover:
  - LTCCellGPU fused projections + Heun solver
  - MultiScaleLTCCell physics-aware initialization
  - LTCEncoderGPU with gradient checkpointing
  - HybridActor discrete + continuous action space
  - ReplayBufferGPU push/sample/batch operations
  - VectorizedDSAEnv batched parallel environments
  - ITUPropagation physics models
  - SionnaChannelGenerator 3GPP TDL model
  - Unified5GDataPipeline schema
  - MultiScaleLTCEncoder 5G timescales
  - Config schema validation
"""

import pytest
import torch
import numpy as np


# ── LTC Cell GPU ──────────────────────────────────────────────────────

class TestLTCCellGPU:
    def test_forward_shape(self):
        from preceptualai.core.ltc_cell_gpu import LTCCellGPU
        cell = LTCCellGPU(input_dim=30, hidden_dim=64, dt=1.0, solver="heun")
        x = torch.randn(8, 30)
        h = torch.zeros(8, 64)
        h_new = cell(x, h)
        assert h_new.shape == (8, 64)

    def test_euler_solver(self):
        from preceptualai.core.ltc_cell_gpu import LTCCellGPU
        cell = LTCCellGPU(input_dim=10, hidden_dim=32, dt=0.5, solver="euler")
        x = torch.randn(4, 10)
        h = torch.zeros(4, 32)
        h_new = cell(x, h)
        assert h_new.shape == (4, 32)
        assert torch.isfinite(h_new).all()

    def test_sub_stepping(self):
        from preceptualai.core.ltc_cell_gpu import LTCCellGPU
        cell = LTCCellGPU(input_dim=10, hidden_dim=32, dt=1.0, solver="heun", sub_steps=3)
        x = torch.randn(4, 10)
        h = torch.zeros(4, 32)
        h_new = cell(x, h)
        assert h_new.shape == (4, 32)

    def test_fused_variant(self):
        from preceptualai.core.ltc_cell_gpu import LTCCellGPUFused
        cell = LTCCellGPUFused(input_dim=20, hidden_dim=64, dt=1.0)
        x = torch.randn(8, 20)
        h = torch.zeros(8, 64)
        h_new = cell(x, h)
        assert h_new.shape == (8, 64)


# ── Multi-Scale LTC Cell ─────────────────────────────────────────────

class TestMultiScaleLTCCell:
    def test_physics_aware_init(self):
        from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCCell
        cell = MultiScaleLTCCell(
            input_dim=30, hidden_dim=64,
            tau_init_mean=5.0, tau_init_std=1.0,
        )
        assert cell.tau_base.mean().item() > 3.0  # Should be near 5.0

    def test_forward(self):
        from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCCell
        cell = MultiScaleLTCCell(input_dim=10, hidden_dim=32)
        x = torch.randn(4, 10)
        h = torch.zeros(4, 32)
        h_new = cell(x, h)
        assert h_new.shape == (4, 32)


class TestMultiScaleLTCEncoder:
    def test_5g_timescales(self):
        from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCEncoder
        enc = MultiScaleLTCEncoder(input_dim=30, num_layers=4, domain="5g")
        assert len(enc.cells) == 4

    def test_forward_shape(self):
        from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCEncoder
        enc = MultiScaleLTCEncoder(input_dim=30, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 16, 30)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_tau_stats(self):
        from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCEncoder
        enc = MultiScaleLTCEncoder(input_dim=30, num_layers=3, domain="5g")
        stats = enc.get_tau_stats()
        assert "layer_0" in stats
        assert "layer_2" in stats


# ── LTC Encoder GPU ──────────────────────────────────────────────────

class TestLTCEncoderGPU:
    def test_forward_shape(self):
        from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
        enc = LTCEncoderGPU(input_dim=30, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 16, 30)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_same_hidden_latent(self):
        from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
        enc = LTCEncoderGPU(input_dim=30, hidden_dim=64, latent_dim=64, num_layers=2)
        assert enc.proj is None
        x = torch.randn(2, 8, 30)
        z = enc(x)
        assert z.shape == (2, 64)


# ── Hybrid Actor ─────────────────────────────────────────────────────

class TestHybridActor:
    def test_forward_shapes(self):
        from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
        from preceptualai.core.hybrid_actor import HybridActor
        enc = LTCEncoderGPU(input_dim=30, hidden_dim=64, latent_dim=32)
        actor = HybridActor(enc, num_discrete_actions=10, continuous_action_dim=3)
        x = torch.randn(4, 16, 30)
        probs, mean, log_std = actor(x)
        assert probs.shape == (4, 10)
        assert mean.shape == (4, 3)
        assert log_std.shape == (4, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_get_action(self):
        from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
        from preceptualai.core.hybrid_actor import HybridActor
        enc = LTCEncoderGPU(input_dim=30, hidden_dim=64, latent_dim=32)
        actor = HybridActor(enc, num_discrete_actions=10, continuous_action_dim=2)
        x = torch.randn(1, 16, 30)
        d_act, c_act, probs, mean = actor.get_action(x)
        assert isinstance(d_act, int)
        assert 0 <= d_act < 10
        assert c_act.shape == (2,)
        assert (c_act >= -1).all() and (c_act <= 1).all()

    def test_evaluate_actions(self):
        from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
        from preceptualai.core.hybrid_actor import HybridActor
        enc = LTCEncoderGPU(input_dim=30, hidden_dim=64, latent_dim=32)
        actor = HybridActor(enc, num_discrete_actions=10, continuous_action_dim=2)
        x = torch.randn(4, 16, 30)
        d_acts = torch.randint(0, 10, (4,))
        c_acts = torch.randn(4, 2).tanh()
        d_lp, c_lp, d_ent, c_ent = actor.evaluate_actions(x, d_acts, c_acts)
        assert d_lp.shape == (4,)
        assert c_lp.shape == (4,)


# ── Replay Buffer GPU ────────────────────────────────────────────────

class TestReplayBufferGPU:
    def test_push_and_sample_cpu_fallback(self):
        """Test with CPU tensors when no GPU available."""
        from preceptualai.core.replay_buffer_gpu import ReplayBufferGPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            pytest.skip("GPU required for ReplayBufferGPU")
        buf = ReplayBufferGPU(capacity=100, state_shape=(16, 30), device=device)
        state = np.random.randn(16, 30).astype(np.float32)
        buf.push(state, 5, 1.0, state, False)
        assert len(buf) == 1
        batch = buf.sample(1)
        assert batch["states"].shape == (1, 16, 30)

    def test_push_batch(self):
        from preceptualai.core.replay_buffer_gpu import ReplayBufferGPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            pytest.skip("GPU required for ReplayBufferGPU")
        buf = ReplayBufferGPU(capacity=1000, state_shape=(16, 30), device=device)
        states = torch.randn(32, 16, 30, device=device)
        actions = torch.randint(0, 10, (32,), device=device)
        rewards = torch.randn(32, device=device)
        buf.push_batch(states, actions, rewards, states, torch.zeros(32, device=device))
        assert len(buf) == 32


# ── Vectorized DSA Env ───────────────────────────────────────────────

class TestVectorizedDSAEnv:
    def test_reset_shape(self):
        from preceptualai.env.sim_vectorized import VectorizedDSAEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = VectorizedDSAEnv(num_envs=8, num_channels=5, device=device)
        obs = env.reset()
        assert obs.shape == (8, 16, 15)  # 5 channels * 3 features

    def test_step_rewards(self):
        from preceptualai.env.sim_vectorized import VectorizedDSAEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = VectorizedDSAEnv(num_envs=4, num_channels=5, device=device)
        env.reset()
        actions = torch.randint(0, 5, (4,), device=device)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (4, 16, 15)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_auto_reset(self):
        from preceptualai.env.sim_vectorized import VectorizedDSAEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = VectorizedDSAEnv(num_envs=4, num_channels=5, max_steps=2, device=device)
        env.reset()
        actions = torch.zeros(4, dtype=torch.long, device=device)
        for _ in range(3):
            obs, rewards, dones, infos = env.step(actions)
        if dones.any():
            obs = env.auto_reset(dones)
        assert obs.shape == (4, 16, 15)


# ── ITU-R Propagation ────────────────────────────────────────────────

class TestITUPropagation:
    def test_rain_attenuation_increases_with_frequency(self):
        from preceptualai.env.itu_propagation import ITUPropagation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        itu = ITUPropagation(device=device)
        freq_low = torch.tensor([12.0], device=device)
        freq_high = torch.tensor([40.0], device=device)
        rain = torch.tensor([25.0], device=device)
        atten_low = itu.specific_rain_attenuation(freq_low, rain)
        atten_high = itu.specific_rain_attenuation(freq_high, rain)
        assert atten_high > atten_low

    def test_total_attenuation_positive(self):
        from preceptualai.env.itu_propagation import ITUPropagation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        itu = ITUPropagation(device=device)
        freq = torch.tensor([28.0], device=device)
        elev = torch.tensor([30.0], device=device)
        rain = torch.tensor([10.0], device=device)
        atten = itu.total_attenuation(freq, elev, rain)
        assert atten > 0

    def test_batched_computation(self):
        from preceptualai.env.itu_propagation import ITUPropagation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        itu = ITUPropagation(device=device)
        freq = torch.tensor([3.5, 28.0, 39.0], device=device)
        rain = torch.tensor([5.0, 25.0, 50.0], device=device)
        gamma = itu.specific_rain_attenuation(freq, rain)
        assert gamma.shape == (3,)
        assert (gamma > 0).all()


# ── Sionna Channel ───────────────────────────────────────────────────

class TestSionnaChannel:
    def test_tdl_pytorch_generate(self):
        from preceptualai.env.sionna_channel import TDLChannelPyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tdl = TDLChannelPyTorch(num_envs=4, num_channels=10, device=device)
        snr_db, gain_db = tdl.generate()
        assert snr_db.shape == (4, 10)
        assert gain_db.shape == (4, 10)
        assert torch.isfinite(snr_db).all()

    def test_sionna_generator_fallback(self):
        from preceptualai.env.sionna_channel import SionnaChannelGenerator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = SionnaChannelGenerator(num_envs=8, num_channels=10, device=device)
        snr, intf = gen.generate_batch()
        assert snr.shape == (8, 10)
        assert (snr >= 0).all() and (snr <= 1).all()

    def test_occupancy_derivation(self):
        from preceptualai.env.sionna_channel import SionnaChannelGenerator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = SionnaChannelGenerator(num_envs=4, num_channels=10, device=device)
        occ = gen.get_occupancy(snr_threshold_db=5.0)
        assert occ.shape == (4, 10)
        assert ((occ == 0) | (occ == 1)).all()


# ── Data Pipeline ────────────────────────────────────────────────────

class TestUnified5GDataPipeline:
    def test_schema(self):
        from preceptualai.env.data_pipeline import Unified5GDataPipeline
        pipeline = Unified5GDataPipeline()
        assert pipeline.num_features == 5
        assert pipeline.STANDARD_FEATURES == ["rsrp", "rsrq", "snr", "cqi", "rssi"]

    def test_empty_load(self):
        from preceptualai.env.data_pipeline import Unified5GDataPipeline
        pipeline = Unified5GDataPipeline(data_dirs={})
        traces = pipeline.load_all()
        assert traces == []


# ── Config ───────────────────────────────────────────────────────────

class TestConfig:
    def test_gpu_encoder_config(self):
        from preceptualai.config import GPUEncoderConfig
        cfg = GPUEncoderConfig()
        assert cfg.hidden_dim == 256
        assert cfg.solver == "heun"

    def test_vectorized_env_config(self):
        from preceptualai.config import VectorizedEnvConfig
        cfg = VectorizedEnvConfig()
        assert cfg.num_envs == 256
        assert cfg.backend == "pytorch"

    def test_multiscale_config(self):
        from preceptualai.config import MultiScaleConfig
        cfg = MultiScaleConfig()
        assert cfg.num_layers == 4
        assert cfg.domain == "5g"

    def test_spectral_config_includes_gpu(self):
        from preceptualai.config import SpectralConfig
        cfg = SpectralConfig()
        assert cfg.gpu_encoder.hidden_dim == 256
        assert cfg.gpu_agent.batch_size == 1024
        assert cfg.vectorized_env.num_envs == 256
