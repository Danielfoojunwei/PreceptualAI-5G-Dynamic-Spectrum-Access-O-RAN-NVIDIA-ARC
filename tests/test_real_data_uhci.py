"""
Tests for real dataset integration into the UHCI unified environment.

Validates that UCC MISL and Colosseum traces drive terrestrial provider
SNR values instead of synthetic noise.
"""

import math
import os

import pytest
import torch

from preceptualai.env.provider_registry import ProviderType
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityConfig,
    UnifiedConnectivityEnv,
)


# Paths to real datasets
UCC_MISL_PATH = "data/ucc_misl/5Gdataset/extracted/5G-production-dataset"
COLOSSEUM_PATH = "data/colosseum/colosseum-oran-commag-dataset"


def _data_available():
    return os.path.exists(UCC_MISL_PATH) and os.path.exists(COLOSSEUM_PATH)


@pytest.mark.skipif(not _data_available(), reason="Real datasets not on disk")
class TestRealDataUHCI:

    def _make_env(self, providers=None, num_envs=4):
        providers = providers or [ProviderType.FR1, ProviderType.FR3, ProviderType.LEO]
        cfg = UnifiedConnectivityConfig(
            num_envs=num_envs,
            device="cpu",
            provider_types=providers,
            channels_per_provider=4,
            real_data_dirs={
                "ucc_misl": UCC_MISL_PATH,
                "colosseum": COLOSSEUM_PATH,
            },
            max_traces_per_source=20,  # small for fast tests
        )
        return UnifiedConnectivityEnv(cfg)

    def test_traces_loaded(self):
        """Real traces must be loaded into the environment."""
        env = self._make_env()
        assert env._real_traces is not None, "No traces loaded"
        assert env._real_traces.shape[0] > 100, "Too few trace rows"
        assert env._real_traces.shape[1] == 5, "Expected 5 features"

    def test_terrestrial_providers_use_traces(self):
        """FR1 must be in the trace-using provider list."""
        env = self._make_env()
        fr1_idx = env.cfg.provider_types.index(ProviderType.FR1)
        assert fr1_idx in env._terrestrial_trace_providers

    def test_step_with_real_traces(self):
        """Environment must step without errors when using real traces."""
        env = self._make_env(num_envs=4)
        obs = env.reset()
        assert torch.isfinite(obs).all()

        for _ in range(20):
            actions = torch.randint(0, env.action_dim, (4,))
            obs, rewards, dones, info = env.step(actions)
            assert torch.isfinite(obs).all(), "Non-finite obs with real traces"
            assert torch.isfinite(rewards).all(), "Non-finite rewards with real traces"

    def test_snr_from_traces_differs_from_default(self):
        """
        SNR values driven by real traces must differ from the default
        synthetic initialization.
        """
        env = self._make_env(num_envs=8)
        env.reset()

        # Step a few times to let traces drive the SNR
        for _ in range(10):
            env.step(torch.zeros(8, dtype=torch.long))

        fr1_idx = env.cfg.provider_types.index(ProviderType.FR1)
        fr1_snr = env.snr_db[:, fr1_idx, :]  # (E, C)

        # Real traces should produce varied SNR (not all identical)
        snr_std = fr1_snr.std().item()
        assert snr_std > 0.5, f"FR1 SNR too uniform ({snr_std:.3f}) — traces not active?"

    def test_trace_ptrs_advance(self):
        """Trace pointers must advance each step (not frozen)."""
        env = self._make_env(num_envs=2)
        env.reset()

        fr1_idx = env.cfg.provider_types.index(ProviderType.FR1)
        ptr_before = env._trace_ptr[0, fr1_idx].item()

        for _ in range(5):
            env.step(torch.zeros(2, dtype=torch.long))

        ptr_after = env._trace_ptr[0, fr1_idx].item()
        assert ptr_after != ptr_before, "Trace pointer not advancing"

    def test_reset_reassigns_traces(self):
        """Reset must reassign random traces to each env."""
        env = self._make_env(num_envs=4)
        env.reset()

        fr1_idx = env.cfg.provider_types.index(ProviderType.FR1)
        assignments_1 = env._trace_assignments[:, fr1_idx].clone()

        # Reset multiple times — at least one should differ (stochastic)
        any_different = False
        for _ in range(10):
            env.reset()
            assignments_2 = env._trace_assignments[:, fr1_idx]
            if not torch.equal(assignments_1, assignments_2):
                any_different = True
                break

        assert any_different, "Trace assignments never changed across resets"

    def test_full_rollout_with_real_data(self):
        """100-step rollout with real traces, all metrics finite."""
        env = self._make_env(
            providers=[ProviderType.FR1, ProviderType.FR3, ProviderType.LEO, ProviderType.ISAC],
            num_envs=4,
        )
        obs = env.reset()

        for step in range(100):
            actions = torch.randint(0, env.action_dim, (4,))
            obs, rewards, dones, info = env.step(actions)
            assert torch.isfinite(obs).all(), f"Non-finite obs at step {step}"
            assert torch.isfinite(rewards).all(), f"Non-finite reward at step {step}"

    def test_provider_stats_with_real_data(self):
        """Provider stats must be available and finite with real traces."""
        env = self._make_env()
        env.reset()
        for _ in range(10):
            env.step(torch.zeros(4, dtype=torch.long))
        stats = env.get_provider_stats()
        for k, v in stats.items():
            assert torch.isfinite(v), f"Non-finite stat: {k}"


@pytest.mark.skipif(not _data_available(), reason="Real datasets not on disk")
class TestRealDataPipeline:

    def test_pipeline_loads_both_sources(self):
        """Pipeline must load traces from both UCC MISL and Colosseum."""
        from preceptualai.env.data_pipeline import Unified5GDataPipeline

        pipeline = Unified5GDataPipeline(
            data_dirs={
                "ucc_misl": UCC_MISL_PATH,
                "colosseum": COLOSSEUM_PATH,
            },
            max_traces_per_source=10,
        )
        traces = pipeline.load_all()
        assert len(traces) > 0, "No traces loaded"
        assert all(t.shape[1] == 5 for t in traces), "Wrong feature count"

    def test_normalization_stats_computed(self):
        """Z-score normalization stats must be computed."""
        from preceptualai.env.data_pipeline import Unified5GDataPipeline

        pipeline = Unified5GDataPipeline(
            data_dirs={"ucc_misl": UCC_MISL_PATH},
            max_traces_per_source=5,
        )
        pipeline.load_all()
        stats = pipeline.normalization_stats
        assert "snr" in stats
        assert "rsrp" in stats
        assert stats["snr"]["std"] > 0
