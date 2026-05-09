"""FedAvg / FedProx aggregation + sparsifier tests."""

import pytest
import torch

from horizon_ric.federated import (
    DEFAULT_FEDPROX_MU,
    ClientUpdate,
    FedAvg,
    FedProx,
    aggregate_fedavg,
    default_aggregator,
    quantize_int8,
    sign_sgd_compress,
    top_k_sparsify,
)


def _client_update(client_id: str, w: float, n_samples: int) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        state_dict={
            "linear.weight": torch.tensor([[w]], dtype=torch.float32),
            "linear.bias": torch.tensor([w], dtype=torch.float32),
        },
        sample_count=n_samples,
    )


class TestFedAvg:
    def test_two_equal_clients(self):
        a = _client_update("a", 1.0, 100)
        b = _client_update("b", 3.0, 100)
        result = aggregate_fedavg([a, b])
        assert torch.allclose(
            result["linear.weight"], torch.tensor([[2.0]])
        )

    def test_sample_weighted(self):
        a = _client_update("a", 1.0, 100)
        b = _client_update("b", 5.0, 900)
        # 0.1 * 1.0 + 0.9 * 5.0 = 4.6
        result = aggregate_fedavg([a, b])
        assert abs(result["linear.weight"].item() - 4.6) < 1e-6

    def test_object_form_matches_function(self):
        clients = [
            _client_update("a", 1.0, 100),
            _client_update("b", 2.0, 200),
        ]
        agg = FedAvg()
        f_result = aggregate_fedavg(clients)
        o_result = agg.aggregate(clients)
        assert torch.allclose(f_result["linear.weight"], o_result["linear.weight"])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            aggregate_fedavg([])

    def test_rejects_zero_samples(self):
        bad = _client_update("a", 1.0, 0)
        with pytest.raises(ValueError):
            aggregate_fedavg([bad])

    def test_rejects_mismatched_keys(self):
        a = _client_update("a", 1.0, 100)
        b = ClientUpdate(
            "b", {"other.weight": torch.tensor([[1.0]])}, 100,
        )
        with pytest.raises(ValueError):
            aggregate_fedavg([a, b])

    def test_rejects_mismatched_shapes(self):
        a = _client_update("a", 1.0, 100)
        b = ClientUpdate(
            "b",
            {
                "linear.weight": torch.tensor([[[1.0]]]),
                "linear.bias": torch.tensor([1.0]),
            },
            100,
        )
        with pytest.raises(ValueError):
            aggregate_fedavg([a, b])


class TestFedProx:
    def test_aggregation_matches_fedavg(self):
        clients = [_client_update("a", 1.0, 100), _client_update("b", 5.0, 900)]
        prox = FedProx(mu=0.01)
        avg = FedAvg()
        prox_out = prox.aggregate(clients)
        avg_out = avg.aggregate(clients)
        assert torch.allclose(prox_out["linear.weight"], avg_out["linear.weight"])

    def test_negative_mu_rejected(self):
        with pytest.raises(ValueError):
            FedProx(mu=-1.0)


class TestDefaultAggregator:
    """Locks the system-wide default to FedProx (closes Devil-C #37).

    FedAvg under heterogeneous client distributions has provable drift
    (SCAFFOLD ICML 2020); FedProx with μ=0.01 is the recommended
    default for the heterogeneous edge regime UHCI deploys in.
    """

    def test_default_is_fedprox(self):
        agg = default_aggregator()
        assert isinstance(agg, FedProx), (
            "system default MUST be FedProx (Devil-C #37); got "
            f"{type(agg).__name__}"
        )

    def test_default_mu_is_pinned(self):
        agg = default_aggregator()
        assert agg.mu == DEFAULT_FEDPROX_MU == 0.01, (
            "default μ must be 0.01 per Li-2020 §5.2; "
            f"got {agg.mu}"
        )

    def test_default_aggregates_correctly(self):
        # Smoke test: the FedProx default produces the same numerical
        # result as raw FedAvg (server-side aggregation is identical) so
        # switching the default does NOT regress correctness.
        clients = [
            _client_update("a", 1.0, 100),
            _client_update("b", 5.0, 900),
        ]
        prox = default_aggregator().aggregate(clients)
        avg = FedAvg().aggregate(clients)
        assert torch.allclose(prox["linear.weight"], avg["linear.weight"])


class TestSparsifiers:
    def test_top_k_keeps_top_one_percent(self):
        delta = {"w": torch.arange(100.0)}
        out = top_k_sparsify(delta, sparsity=0.99)
        n_nonzero = (out["w"] != 0).sum().item()
        assert n_nonzero == 1
        # Only the largest (99) should survive
        assert out["w"][99].item() == 99.0

    def test_top_k_zero_sparsity_keeps_all(self):
        delta = {"w": torch.arange(10.0)}
        out = top_k_sparsify(delta, sparsity=0.0)
        assert torch.allclose(out["w"], delta["w"])

    def test_top_k_invalid_sparsity(self):
        with pytest.raises(ValueError):
            top_k_sparsify({"w": torch.zeros(5)}, sparsity=1.0)

    def test_sign_sgd_preserves_sign(self):
        delta = {"w": torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])}
        out = sign_sgd_compress(delta)
        # Signs preserved
        assert out["w"][0].item() < 0
        assert out["w"][4].item() > 0
        # All non-zero magnitudes equal (the per-tensor mean abs)
        nonzero = out["w"][out["w"] != 0]
        assert torch.allclose(nonzero.abs(), nonzero.abs()[0:1].expand_as(nonzero.abs()))

    def test_quantize_int8_round_trip(self):
        delta = {"w": torch.randn(64) * 5.0}
        q, scales = quantize_int8(delta)
        assert q["w"].dtype == torch.int8
        recon = q["w"].to(torch.float32) * scales["w"]
        # int8 quantization: max error = scale (≈ absmax/127); test < 2× scale
        max_err = (recon - delta["w"]).abs().max().item()
        assert max_err < 2 * scales["w"]
