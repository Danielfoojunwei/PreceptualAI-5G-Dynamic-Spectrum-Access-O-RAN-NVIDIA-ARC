"""Tests for per-site LoRA adapters (`horizon_ric.continual.lora_adapter`)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from horizon_ric.continual.lora_adapter import (
    LoRAAdapter,
    apply_lora,
    get_lora_state_dict,
    load_lora_state_dict,
)


class _ToyMLP(nn.Module):
    def __init__(self, d_in: int = 64, d_hidden: int = 128, d_out: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_lora_param_count_drops_to_r_times_d_plus_k():
    """Wrapping a (out=128, in=64) Linear at rank=8 must give 8*(128+64)=1536
    LoRA params, vs 128*64=8192 dense params. Two orders smaller is the
    whole point of LoRA.
    """
    base = nn.Linear(64, 128, bias=True)
    dense_count = 64 * 128  # 8192
    adapter = LoRAAdapter(base, rank=8)
    assert adapter.lora_param_count == 8 * (64 + 128) == 1536
    assert adapter.lora_param_count < dense_count // 5


def test_zero_init_lora_is_identity_passthrough():
    """B is zero-initialised so a freshly wrapped LoRA must produce the same
    output as the underlying base linear (delta path == 0)."""
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    adapter = LoRAAdapter(base, rank=4)
    x = torch.randn(7, 16)
    with torch.no_grad():
        y_base = base(x)
        y_adapter = adapter(x)
    assert torch.allclose(y_base, y_adapter, atol=1e-6)


def test_lora_gradients_flow_into_A_and_B_only():
    """Grads must flow into lora_A / lora_B; the frozen base weight must
    NOT receive a gradient."""
    torch.manual_seed(1)
    base = nn.Linear(8, 12)
    adapter = LoRAAdapter(base, rank=3)
    x = torch.randn(4, 8, requires_grad=False)
    target = torch.randn(4, 12)
    y = adapter(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert adapter.lora_A.grad is not None
    assert adapter.lora_B.grad is not None
    assert torch.any(adapter.lora_A.grad != 0) or torch.any(adapter.lora_B.grad != 0)
    # Base weight is frozen → grad must be None (requires_grad=False).
    assert adapter.base.weight.grad is None


def test_state_dict_round_trip_only_carries_lora_params():
    """get_lora_state_dict → load_lora_state_dict must restore identical
    LoRA tensors and the round-trip must contain no base-weight tensors."""
    torch.manual_seed(2)
    model = _ToyMLP(64, 32, 16)
    wrapped = apply_lora(model, ["fc1", "fc2"], rank=4)
    assert wrapped == ["fc1", "fc2"]

    # Train one fake step so A/B are not zero/random-only.
    x = torch.randn(2, 64)
    y = model(x).sum()
    y.backward()
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, LoRAAdapter):
                m.lora_A += 0.1 * torch.randn_like(m.lora_A)
                m.lora_B += 0.1 * torch.randn_like(m.lora_B)

    sd = get_lora_state_dict(model)
    # Only LoRA tensors are persisted; no base weights.
    assert set(sd) == {"fc1.lora_A", "fc1.lora_B", "fc2.lora_A", "fc2.lora_B"}
    assert all("base" not in k for k in sd)

    # Build a fresh model with the same base weights and zero LoRA, then
    # load — outputs should match the trained one. After apply_lora,
    # model.fc1 is a LoRAAdapter; the original Linear lives at .base.
    model2 = _ToyMLP(64, 32, 16)
    model2.fc1.weight.data.copy_(model.fc1.base.weight.data)
    model2.fc1.bias.data.copy_(model.fc1.base.bias.data)
    model2.fc2.weight.data.copy_(model.fc2.base.weight.data)
    model2.fc2.bias.data.copy_(model.fc2.base.bias.data)
    apply_lora(model2, ["fc1", "fc2"], rank=4)
    load_lora_state_dict(model2, sd)

    inp = torch.randn(3, 64)
    with torch.no_grad():
        out1 = model(inp)
        out2 = model2(inp)
    assert torch.allclose(out1, out2, atol=1e-6)

    # Mismatched shape must raise.
    bad = dict(sd)
    bad["fc1.lora_A"] = torch.zeros(99, 99)
    with pytest.raises(KeyError):
        load_lora_state_dict(model2, bad)


def test_multiple_loras_with_different_ranks_coexist():
    """fc1 wrapped at rank=4 and fc2 wrapped at rank=8 must coexist; their
    state dicts must carry the expected shapes."""
    torch.manual_seed(3)
    model = _ToyMLP(32, 64, 16)
    apply_lora(model, ["fc1"], rank=4)
    apply_lora(model, ["fc2"], rank=8)

    # Both adapters present.
    adapters = {n: m for n, m in model.named_modules() if isinstance(m, LoRAAdapter)}
    assert set(adapters) == {"fc1", "fc2"}
    assert adapters["fc1"].rank == 4
    assert adapters["fc2"].rank == 8

    sd = get_lora_state_dict(model)
    assert sd["fc1.lora_A"].shape == (4, 32)
    assert sd["fc1.lora_B"].shape == (64, 4)
    assert sd["fc2.lora_A"].shape == (8, 64)
    assert sd["fc2.lora_B"].shape == (16, 8)

    # End-to-end forward still runs.
    out = model(torch.randn(2, 32))
    assert out.shape == (2, 16)
