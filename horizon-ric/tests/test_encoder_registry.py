"""Encoder registry — pluggable temporal encoders.

Verifies the registry pattern (mirror of `io.registry`) and that all
built-in encoders honour the (B, T, D) → (B, T, D) shape contract. The
swap is a single config-flip; no caller code changes.
"""

from __future__ import annotations

import pytest
import torch

from horizon_ric.core.encoder_registry import (
    EncoderRegistry,
    TemporalEncoder,
    get_encoder,
    list_encoders,
    register_encoder,
)


def test_builtins_are_registered():
    """The four built-ins should be present after import."""
    names = list_encoders()
    for required in ("identity", "cfc", "liquid_s4", "latent_ode"):
        assert required in names, f"missing built-in encoder {required!r}"


@pytest.mark.parametrize("name", ["identity", "cfc", "liquid_s4", "latent_ode"])
def test_each_builtin_builds_and_forwards(name):
    """Every built-in must build and forward an (B,T,D) tensor."""
    d_model = 16
    encoder = get_encoder(name, d_model=d_model)
    assert isinstance(encoder, TemporalEncoder)
    x = torch.randn(2, 5, d_model)
    y = encoder(x)
    assert y.shape == (2, 5, d_model), (
        f"encoder {name!r} broke (B,T,D) contract: in={tuple(x.shape)} "
        f"out={tuple(y.shape)}"
    )
    assert torch.isfinite(y).all(), f"encoder {name!r} produced NaN/Inf"


def test_swap_is_config_flip():
    """Train-time swap from cfc -> liquid_s4 -> identity must be one
    string change, no callsite branching."""
    d_model = 8
    inputs = torch.randn(1, 4, d_model)

    def run(name: str) -> torch.Tensor:
        return get_encoder(name, d_model=d_model)(inputs)

    out_cfc = run("cfc")
    out_l4 = run("liquid_s4")
    out_id = run("identity")
    assert out_cfc.shape == out_l4.shape == out_id.shape == inputs.shape
    # Identity is exact; the others should differ from input.
    assert torch.allclose(out_id, inputs)
    assert not torch.allclose(out_cfc, inputs)
    assert not torch.allclose(out_l4, inputs)


def test_register_custom_encoder_picked_up():
    """A user-defined encoder registered at runtime must be retrievable."""

    class MyEncoder(TemporalEncoder):
        def __init__(self, d_model: int = 4, **_):
            super().__init__()
            self.d_model = d_model
            self.scale = torch.nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            return x * self.scale

    reg = EncoderRegistry()
    reg.register("my_scaler", MyEncoder)
    e = reg.get("my_scaler", d_model=4)
    x = torch.randn(2, 3, 4)
    y = e(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x)  # init scale=1
    assert "my_scaler" in reg.list_encoders()


def test_unknown_encoder_raises():
    """Asking for an unregistered encoder must error loudly with the
    list of valid names — silent fallbacks are forbidden."""
    with pytest.raises(KeyError) as exc:
        get_encoder("__never_registered__")
    assert "registered:" in str(exc.value)


def test_global_register_function():
    """The module-level `register_encoder` must hit the singleton."""

    class Tiny(TemporalEncoder):
        def __init__(self, d_model: int = 2, **_):
            super().__init__()
            self.d_model = d_model

        def forward(self, x):
            return x

    register_encoder("__test_tiny__", Tiny)
    assert "__test_tiny__" in list_encoders()
    e = get_encoder("__test_tiny__", d_model=2)
    out = e(torch.zeros(1, 1, 2))
    assert out.shape == (1, 1, 2)
