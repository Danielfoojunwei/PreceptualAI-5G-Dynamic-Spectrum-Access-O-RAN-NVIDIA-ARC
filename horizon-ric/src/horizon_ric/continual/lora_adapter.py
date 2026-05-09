"""Per-site LoRA (Low-Rank Adaptation) adapters.

Federated rApps want to ship a single, shared base model and let each
site train a tiny per-site delta. LoRA factorises the weight delta
``ΔW`` of an `nn.Linear` into a low-rank product:

    W' = W + (B @ A) * (alpha / r)

with ``A ∈ R^{r×in}``, ``B ∈ R^{out×r}``, rank ``r ≪ min(in,out)``.
The base linear is **frozen**; only ``A`` and ``B`` are trained per
site. Persisting just `(A, B)` per site costs ``r*(in+out)`` params
instead of ``in*out`` — typically two-orders-of-magnitude smaller.

`B` is zero-initialised so a fresh LoRA wrap is a no-op (W' == W) and
training is purely additive.

Public API
----------
* ``LoRAAdapter`` — `nn.Module` subclass that wraps an `nn.Linear`.
* ``apply_lora(model, target_module_names, rank=8, alpha=None)`` —
  in-place wrap matching `nn.Linear` submodules.
* ``get_lora_state_dict(model)`` / ``load_lora_state_dict(model, sd)``
  — round-trip only the LoRA params, keep the base model shared.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-rank delta wrapper around a single ``nn.Linear``.

    Parameters
    ----------
    base
        The frozen base ``nn.Linear``; its weight is registered as a
        non-trainable buffer-like parameter (``requires_grad=False``).
    rank
        Rank of the low-rank delta. Must satisfy
        ``1 <= rank <= min(in_features, out_features)``.
    alpha
        Scale factor; the effective delta is ``(B @ A) * (alpha/rank)``.
        Defaults to ``rank`` (i.e. unit scale).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAAdapter expects nn.Linear, got {type(base).__name__}")
        in_f = base.in_features
        out_f = base.out_features
        if rank < 1 or rank > min(in_f, out_f):
            raise ValueError(
                f"rank={rank} out of range [1, {min(in_f, out_f)}] for "
                f"linear of shape ({out_f}, {in_f})"
            )
        self.in_features = in_f
        self.out_features = out_f
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        # Freeze the base linear in-place.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        # LoRA parameters (trainable). Use float32 to keep numerics
        # stable across mixed-precision pipelines; user can `.to(dtype)`
        # afterwards.
        # A: (rank, in_features) - small Kaiming-style init.
        # B: (out_features, rank) - zero init so initial delta == 0.
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )

    @property
    def lora_param_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def base_param_count(self) -> int:
        # Just the dense weight count (matches d*k from the docstring).
        return self.in_features * self.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen).
        out = self.base(x)
        # Low-rank delta path: x @ A^T @ B^T  (shapes: (..., in) -> (..., r) -> (..., out))
        delta = x @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling


def _resolve_parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Return ``(parent_module, leaf_name)`` for a dotted attribute path."""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    target_module_names: Iterable[str],
    rank: int = 8,
    alpha: float | None = None,
) -> list[str]:
    """Wrap matching ``nn.Linear`` submodules with :class:`LoRAAdapter`.

    Targets are matched by exact dotted name first; if no exact match
    is found for a given target, the function falls back to wrapping
    every ``nn.Linear`` whose dotted name **ends with** the target
    string (so callers can pass e.g. ``"q_proj"`` to wrap every
    attention q-projection).

    Returns the list of dotted names that were wrapped, so callers can
    log the change.
    """
    targets = list(target_module_names)
    if not targets:
        return []

    # Snapshot dotted names so we don't iterate while mutating.
    all_linears: list[tuple[str, nn.Linear]] = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    wrapped: list[str] = []
    seen: set[str] = set()

    name_set = {n for n, _ in all_linears}

    for target in targets:
        # Prefer exact match.
        candidates: list[str]
        if target in name_set:
            candidates = [target]
        else:
            candidates = [n for n, _ in all_linears if n == target or n.endswith("." + target)]
        for cand in candidates:
            if cand in seen:
                continue
            parent, leaf = _resolve_parent(model, cand)
            base = getattr(parent, leaf)
            if not isinstance(base, nn.Linear):
                continue
            adapter = LoRAAdapter(base, rank=rank, alpha=alpha)
            setattr(parent, leaf, adapter)
            wrapped.append(cand)
            seen.add(cand)
    return wrapped


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA parameter tensors, keyed by dotted name."""
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAAdapter):
            out[f"{name}.lora_A"] = mod.lora_A.detach().clone()
            out[f"{name}.lora_B"] = mod.lora_B.detach().clone()
    return out


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA tensors back into matching adapters in ``model``.

    Raises ``KeyError`` on shape or name mismatch so silent corruption
    is impossible.
    """
    by_name: dict[str, LoRAAdapter] = {
        n: m for n, m in model.named_modules() if isinstance(m, LoRAAdapter)
    }
    expected = set()
    for n in by_name:
        expected.add(f"{n}.lora_A")
        expected.add(f"{n}.lora_B")
    missing = expected - set(state_dict.keys())
    if missing:
        raise KeyError(f"missing LoRA tensors in state_dict: {sorted(missing)}")

    with torch.no_grad():
        for name, mod in by_name.items():
            a = state_dict[f"{name}.lora_A"]
            b = state_dict[f"{name}.lora_B"]
            if a.shape != mod.lora_A.shape:
                raise KeyError(
                    f"shape mismatch for {name}.lora_A: got {tuple(a.shape)} "
                    f"expected {tuple(mod.lora_A.shape)}"
                )
            if b.shape != mod.lora_B.shape:
                raise KeyError(
                    f"shape mismatch for {name}.lora_B: got {tuple(b.shape)} "
                    f"expected {tuple(mod.lora_B.shape)}"
                )
            mod.lora_A.copy_(a)
            mod.lora_B.copy_(b)


__all__ = [
    "LoRAAdapter",
    "apply_lora",
    "get_lora_state_dict",
    "load_lora_state_dict",
]
