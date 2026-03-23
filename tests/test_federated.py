"""
Tests for the SpectrAI federated learning module.

Covers the core FL mechanics without requiring a real 5G dataset or
long training runs.  All tests use synthetic state_dicts that mirror
the actual LTC-based SAC architecture.
"""

import copy
from collections import OrderedDict

import pytest
import torch

from spectrai.agent.sac_ltc import SACLTCAgent
from spectrai.federated.aggregator import (
    HybridFederatedAggregator,
    _is_tau_weight,
    _partition_state_dict,
)
from spectrai.federated.client import FederatedSACLTCClient
from spectrai.federated.config import FLConfig

# ======================================================================
# Fixtures
# ======================================================================


def _make_fake_state_dict(seed: int = 0, hidden: int = 8) -> OrderedDict:
    """Create a synthetic state_dict mimicking an LTCActor/Critic.

    Contains both structural (W_h, W_x, head, layer_norm) and tau
    (W_tau, tau_base) parameters, as in the real LTCEncoder.
    """
    torch.manual_seed(seed)
    sd = OrderedDict()
    # Encoder layer 0
    sd["encoder.cells.0.W_h.weight"] = torch.randn(hidden, hidden)
    sd["encoder.cells.0.W_h.bias"] = torch.randn(hidden)
    sd["encoder.cells.0.W_x.weight"] = torch.randn(hidden, 5)
    sd["encoder.cells.0.W_x.bias"] = torch.randn(hidden)
    sd["encoder.cells.0.W_tau.weight"] = torch.randn(hidden, 5)
    sd["encoder.cells.0.W_tau.bias"] = torch.randn(hidden)
    sd["encoder.cells.0.tau_base"] = torch.ones(hidden) + seed * 0.1
    # Layer norm
    sd["encoder.layer_norms.0.weight"] = torch.ones(hidden)
    sd["encoder.layer_norms.0.bias"] = torch.zeros(hidden)
    # Head
    sd["head.weight"] = torch.randn(5, hidden)
    sd["head.bias"] = torch.randn(5)
    return sd


def _make_fake_weights(seed: int = 0, hidden: int = 8) -> dict:
    """Create a full set of fake weights for actor + critics."""
    return {
        "actor": _make_fake_state_dict(seed, hidden),
        "critic1": _make_fake_state_dict(seed + 100, hidden),
        "critic2": _make_fake_state_dict(seed + 200, hidden),
    }


@pytest.fixture
def fl_config() -> FLConfig:
    return FLConfig(
        tau_mix_ratio=0.3,
        num_rounds=5,
        local_steps_per_round=100,
        min_devices_per_round=1,
        aggregation_method="hybrid_ltc",
    )


@pytest.fixture
def agent() -> SACLTCAgent:
    """Create a small SAC-LTC agent for testing."""
    device = torch.device("cpu")
    return SACLTCAgent(
        state_shape=(16, 25),
        num_actions=5,
        input_dim=25,
        device=device,
        hidden_dim=16,
        latent_dim=16,
        num_layers=2,
        dt=1.0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=1000,
        batch_size=32,
        learning_starts=50,
    )


# ======================================================================
# test_tau_identification
# ======================================================================


class TestTauIdentification:
    """Verify _is_tau_weight correctly identifies W_tau and tau_base keys."""

    def test_w_tau_weight(self):
        assert _is_tau_weight("encoder.cells.0.W_tau.weight") is True

    def test_w_tau_bias(self):
        assert _is_tau_weight("encoder.cells.0.W_tau.bias") is True

    def test_tau_base(self):
        assert _is_tau_weight("encoder.cells.0.tau_base") is True

    def test_tau_base_nested(self):
        assert _is_tau_weight("encoder.cells.1.tau_base") is True

    def test_w_h_not_tau(self):
        assert _is_tau_weight("encoder.cells.0.W_h.weight") is False

    def test_w_x_not_tau(self):
        assert _is_tau_weight("encoder.cells.0.W_x.bias") is False

    def test_head_not_tau(self):
        assert _is_tau_weight("head.weight") is False

    def test_layer_norm_not_tau(self):
        assert _is_tau_weight("encoder.layer_norms.0.weight") is False

    def test_proj_not_tau(self):
        assert _is_tau_weight("encoder.proj.weight") is False


# ======================================================================
# test_hybrid_aggregation
# ======================================================================


class TestHybridAggregation:
    """Test the hybrid LTC-aware aggregation logic."""

    def test_structural_weights_averaged(self, fl_config):
        """Structural weights should be the weighted average across devices."""
        agg = HybridFederatedAggregator(fl_config)

        # 3 devices with different seeds → different weights
        weights = [_make_fake_weights(seed=i) for i in range(3)]
        samples = [100, 200, 300]

        for i in range(3):
            agg.receive_update(f"dev_{i}", weights[i], samples[i])

        agg.aggregate()
        global_model = agg.get_global_model()

        # Verify structural key is FedAvg'd
        total = sum(samples)
        key = "encoder.cells.0.W_h.weight"
        expected = sum(
            weights[i]["actor"][key] * (samples[i] / total) for i in range(3)
        )
        actual = global_model["actor"][key]
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"Structural weight {key} not correctly averaged"
        )

    def test_tau_weights_in_global_are_averaged(self, fl_config):
        """Global model tau weights should also be fully averaged."""
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(3)]
        samples = [100, 100, 100]  # Equal weights for simplicity

        for i in range(3):
            agg.receive_update(f"dev_{i}", weights[i], samples[i])

        agg.aggregate()
        global_model = agg.get_global_model()

        key = "encoder.cells.0.tau_base"
        expected = sum(weights[i]["actor"][key] for i in range(3)) / 3
        actual = global_model["actor"][key]
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_personalized_tau_is_mixed(self, fl_config):
        """Personalized model should mix global and local tau."""
        fl_config.tau_mix_ratio = 0.4
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(3)]
        samples = [100, 100, 100]

        for i in range(3):
            agg.receive_update(f"dev_{i}", weights[i], samples[i])

        agg.aggregate()

        # Get personalized model for device 0
        personalized = agg.get_personalized_model("dev_0")

        key = "encoder.cells.0.tau_base"
        global_tau = sum(weights[i]["actor"][key] for i in range(3)) / 3
        local_tau = weights[0]["actor"][key]
        expected_mixed = 0.4 * global_tau + 0.6 * local_tau

        actual = personalized["actor"][key]
        assert torch.allclose(actual, expected_mixed, atol=1e-5), (
            "Personalized tau should be tau_mix_ratio * global + (1-ratio) * local"
        )

    def test_structural_same_in_personalized(self, fl_config):
        """Structural weights in personalized model should match global."""
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        global_model = agg.get_global_model()
        personalized = agg.get_personalized_model("dev_0")

        key = "encoder.cells.0.W_h.weight"
        assert torch.allclose(personalized["actor"][key], global_model["actor"][key])

    def test_min_devices_enforcement(self, fl_config):
        """Aggregation should fail if too few devices report."""
        fl_config.min_devices_per_round = 3
        agg = HybridFederatedAggregator(fl_config)

        agg.receive_update("dev_0", _make_fake_weights(0), 100)
        agg.receive_update("dev_1", _make_fake_weights(1), 100)

        with pytest.raises(RuntimeError, match="Only 2 device"):
            agg.aggregate()

    def test_round_buffer_cleared_after_aggregation(self, fl_config):
        """Round buffer should be empty after aggregation."""
        agg = HybridFederatedAggregator(fl_config)
        agg.receive_update("dev_0", _make_fake_weights(0), 100)
        agg.receive_update("dev_1", _make_fake_weights(1), 100)
        agg.aggregate()
        assert agg.num_pending_updates == 0

    def test_fedavg_mode(self):
        """FedAvg mode should average all parameters equally."""
        config = FLConfig(
            aggregation_method="fedavg",
            min_devices_per_round=1,
        )
        agg = HybridFederatedAggregator(config)

        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        global_model = agg.get_global_model()

        # Both structural and tau should be simple averages
        for key in ["encoder.cells.0.W_h.weight", "encoder.cells.0.tau_base"]:
            expected = (weights[0]["actor"][key] + weights[1]["actor"][key]) / 2
            actual = global_model["actor"][key]
            assert torch.allclose(actual, expected, atol=1e-5)


# ======================================================================
# test_client_weight_extraction
# ======================================================================


class TestClientWeightExtraction:
    """Test weight extraction from a real SACLTCAgent."""

    def test_extraction_structure(self, agent, fl_config):
        """Extracted weights should contain actor, critic1, critic2."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)
        weights = client.get_local_weights()

        assert "actor" in weights
        assert "critic1" in weights
        assert "critic2" in weights
        assert "target_critic1" not in weights
        assert "target_critic2" not in weights

    def test_extracted_keys_match_model(self, agent, fl_config):
        """Extracted keys should match the model's state_dict keys."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)
        weights = client.get_local_weights()

        actor_keys = set(agent.actor.state_dict().keys())
        assert set(weights["actor"].keys()) == actor_keys

    def test_extracted_weights_on_cpu(self, agent, fl_config):
        """Extracted weights should always be on CPU."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)
        weights = client.get_local_weights()

        for comp_name, sd in weights.items():
            for key, tensor in sd.items():
                assert tensor.device == torch.device("cpu"), (
                    f"{comp_name}.{key} should be on CPU"
                )

    def test_extracted_weights_contain_tau(self, agent, fl_config):
        """Extracted weights should contain W_tau and tau_base parameters."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)
        weights = client.get_local_weights()

        tau_keys = [k for k in weights["actor"] if _is_tau_weight(k)]
        assert len(tau_keys) > 0, "Should contain W_tau and tau_base keys"

        has_w_tau = any("W_tau" in k for k in tau_keys)
        has_tau_base = any("tau_base" in k for k in tau_keys)
        assert has_w_tau, "Should contain W_tau keys"
        assert has_tau_base, "Should contain tau_base keys"


# ======================================================================
# test_global_model_application
# ======================================================================


class TestGlobalModelApplication:
    """Test that applying global weights updates actor/critic but not targets."""

    def test_actor_critic_updated(self, agent, fl_config):
        """Actor and critic weights should change after applying global model."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)

        # Capture before state
        actor_before = copy.deepcopy(agent.actor.state_dict())

        # Create different weights (from another agent)
        other_agent = SACLTCAgent(
            state_shape=(16, 25), num_actions=5, input_dim=25,
            device=torch.device("cpu"), hidden_dim=16, latent_dim=16,
        )
        other_client = FederatedSACLTCClient(other_agent, "other", fl_config)
        other_weights = other_client.get_local_weights()

        # Apply
        client.apply_global_weights(other_weights)

        # Verify actor changed
        actor_after = agent.actor.state_dict()
        any_changed = False
        for key in actor_before:
            if not torch.equal(actor_before[key], actor_after[key]):
                any_changed = True
                break
        assert any_changed, "Actor weights should have changed"

    def test_targets_unchanged(self, agent, fl_config):
        """Target critics should NOT change when global weights are applied."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)

        # Capture target state
        target1_before = copy.deepcopy(agent.target_critic1.state_dict())
        target2_before = copy.deepcopy(agent.target_critic2.state_dict())

        # Apply different weights
        other_agent = SACLTCAgent(
            state_shape=(16, 25), num_actions=5, input_dim=25,
            device=torch.device("cpu"), hidden_dim=16, latent_dim=16,
        )
        other_client = FederatedSACLTCClient(other_agent, "other", fl_config)
        client.apply_global_weights(other_client.get_local_weights())

        # Verify targets unchanged
        for key in target1_before:
            assert torch.equal(
                target1_before[key], agent.target_critic1.state_dict()[key]
            ), f"target_critic1.{key} should not change"
        for key in target2_before:
            assert torch.equal(
                target2_before[key], agent.target_critic2.state_dict()[key]
            ), f"target_critic2.{key} should not change"

    def test_log_alpha_unchanged(self, agent, fl_config):
        """log_alpha should NOT change when global weights are applied."""
        client = FederatedSACLTCClient(agent, "test_dev", fl_config)
        alpha_before = agent.log_alpha.item()

        other_agent = SACLTCAgent(
            state_shape=(16, 25), num_actions=5, input_dim=25,
            device=torch.device("cpu"), hidden_dim=16, latent_dim=16,
        )
        other_client = FederatedSACLTCClient(other_agent, "other", fl_config)
        client.apply_global_weights(other_client.get_local_weights())

        assert agent.log_alpha.item() == alpha_before


# ======================================================================
# test_cold_start
# ======================================================================


class TestColdStart:
    """Test that a new device can use the global model immediately."""

    def test_cold_start_weights_match_global(self, fl_config):
        """Cold-start device should receive the fully averaged global model."""
        agg = HybridFederatedAggregator(fl_config)

        # Two devices contribute
        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        # New device (never seen) requests personalized model
        personalized = agg.get_personalized_model("new_device")
        global_model = agg.get_global_model()

        # Should be identical to global (no local history for mixing)
        for comp in ["actor", "critic1", "critic2"]:
            for key in global_model[comp]:
                assert torch.allclose(
                    personalized[comp][key], global_model[comp][key]
                ), f"Cold-start {comp}.{key} should match global"

    def test_cold_start_different_from_known_device(self, fl_config):
        """Cold-start model should differ from a known device's personalized model."""
        fl_config.tau_mix_ratio = 0.3
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        cold = agg.get_personalized_model("new_device")
        known = agg.get_personalized_model("dev_0")

        # Structural should be the same
        key_structural = "encoder.cells.0.W_h.weight"
        assert torch.allclose(cold["actor"][key_structural], known["actor"][key_structural])

        # Tau should differ (unless tau_mix_ratio is 1.0)
        key_tau = "encoder.cells.0.tau_base"
        assert not torch.equal(cold["actor"][key_tau], known["actor"][key_tau]), (
            "Cold-start tau should differ from known device's personalized tau"
        )


# ======================================================================
# test_personalized_model
# ======================================================================


class TestPersonalizedModel:
    """Test personalized model construction for known devices."""

    def test_different_devices_get_different_tau(self, fl_config):
        """Different devices should get different personalized tau weights."""
        fl_config.tau_mix_ratio = 0.3
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(3)]
        for i in range(3):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        p0 = agg.get_personalized_model("dev_0")
        p1 = agg.get_personalized_model("dev_1")

        key = "encoder.cells.0.tau_base"
        assert not torch.equal(p0["actor"][key], p1["actor"][key]), (
            "Different devices should have different personalized tau"
        )

    def test_tau_mix_ratio_zero_keeps_local(self, fl_config):
        """With tau_mix_ratio=0, personalized tau should equal device's local tau."""
        fl_config.tau_mix_ratio = 0.0
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        personalized = agg.get_personalized_model("dev_0")
        key = "encoder.cells.0.tau_base"

        # Should match device 0's local tau exactly
        assert torch.allclose(
            personalized["actor"][key],
            weights[0]["actor"][key],
            atol=1e-5,
        )

    def test_tau_mix_ratio_one_gives_global(self, fl_config):
        """With tau_mix_ratio=1, personalized tau should equal global average."""
        fl_config.tau_mix_ratio = 1.0
        agg = HybridFederatedAggregator(fl_config)

        weights = [_make_fake_weights(seed=i) for i in range(2)]
        for i in range(2):
            agg.receive_update(f"dev_{i}", weights[i], 100)
        agg.aggregate()

        personalized = agg.get_personalized_model("dev_0")
        global_model = agg.get_global_model()

        key = "encoder.cells.0.tau_base"
        assert torch.allclose(
            personalized["actor"][key],
            global_model["actor"][key],
            atol=1e-5,
        )

    def test_no_aggregation_raises(self, fl_config):
        """Requesting a model before any aggregation should raise."""
        agg = HybridFederatedAggregator(fl_config)
        with pytest.raises(RuntimeError, match="No aggregation"):
            agg.get_personalized_model("any_device")

    def test_registered_devices(self, fl_config):
        """registered_devices should list all devices that have contributed."""
        agg = HybridFederatedAggregator(fl_config)

        for i in range(3):
            agg.receive_update(f"dev_{i}", _make_fake_weights(i), 100)
        agg.aggregate()

        devices = set(agg.registered_devices)
        assert devices == {"dev_0", "dev_1", "dev_2"}


# ======================================================================
# test_partition_state_dict
# ======================================================================


class TestPartitionStateDict:
    """Test the state_dict partitioning helper."""

    def test_partition_separates_correctly(self):
        sd = _make_fake_state_dict()
        structural, tau = _partition_state_dict(sd)

        # Tau keys
        expected_tau_keys = {
            "encoder.cells.0.W_tau.weight",
            "encoder.cells.0.W_tau.bias",
            "encoder.cells.0.tau_base",
        }
        assert set(tau.keys()) == expected_tau_keys

        # Structural keys = everything else
        expected_structural_keys = set(sd.keys()) - expected_tau_keys
        assert set(structural.keys()) == expected_structural_keys

    def test_partition_preserves_all_keys(self):
        sd = _make_fake_state_dict()
        structural, tau = _partition_state_dict(sd)
        assert set(structural.keys()) | set(tau.keys()) == set(sd.keys())
        assert set(structural.keys()) & set(tau.keys()) == set()


# ======================================================================
# FLConfig validation
# ======================================================================


class TestFLConfig:
    """Test FLConfig pydantic validation."""

    def test_default_config(self):
        config = FLConfig()
        assert config.tau_mix_ratio == 0.3
        assert config.aggregation_method == "hybrid_ltc"

    def test_invalid_tau_mix_ratio(self):
        with pytest.raises(Exception):
            FLConfig(tau_mix_ratio=1.5)

    def test_invalid_aggregation_method(self):
        with pytest.raises(Exception):
            FLConfig(aggregation_method="invalid_method")

    def test_fedprox_config(self):
        config = FLConfig(aggregation_method="fedprox", fedprox_mu=0.05)
        assert config.fedprox_mu == 0.05
