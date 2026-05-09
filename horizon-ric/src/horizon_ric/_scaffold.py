"""Honest-scaffold warning + ``from_pretrained`` helper.

Several modules in this repo (``encoder/graph_jepa``,
``encoder/perceiver_fusion``, ``encoder/entity_tokenizer``,
``core/cfc_core``, ``core/liquid_s4``, ``core/latent_ode``,
``core/latent_dynamics``, ``core/physics_residual``, ``heads/sla_risk``)
are real PyTorch architectures with real ``nn.Module`` weights and a
real training script under ``scripts/train_*.py``. The trained
checkpoints exist under ``checkpoints/*.pt`` on the dev box but are
**gitignored** by `.gitignore` (`*.pt` excluded), so:

  * On the dev box   — modules can be loaded via ``load_state_dict``.
  * On a fresh clone — ``checkpoints/*.pt`` are absent; instantiating
                       a bare module produces **random projections**.

This file provides one shared mechanism so every architecture surfaces
the same honest signal:

  1. ``UntrainedScaffoldWarning`` is emitted at instantiation time when
     no checkpoint has been loaded. It is a real Python warning, so
     downstream code can promote it to an error via ``warnings.simplefilter``.
  2. ``from_pretrained(path)`` is a classmethod helper that loads the
     state dict, marks the module as trained, and returns it.
  3. ``is_trained()`` returns the marker so audit code can refuse to
     emit a counterfactual envelope unless the upstream encoder is
     known-trained.

This module is **not** a fake/stub — it is the honest disclosure layer
that every untrained PyTorch architecture in the repo wires through.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn


class UntrainedScaffoldWarning(UserWarning):
    """Emitted when a real ``nn.Module`` is instantiated without a
    checkpoint. Downstream consumers SHOULD load via
    ``Module.from_pretrained(path)`` before use; failure to do so means
    the forward pass produces a random projection of the input.
    """


def warn_untrained(module_name: str, suggested_path: str | None = None) -> None:
    """Emit ``UntrainedScaffoldWarning`` for a module instantiated bare.

    ``suggested_path`` is the canonical checkpoint path (under
    ``checkpoints/``) that ``from_pretrained()`` would load.
    """
    suggestion = (
        f" Load with `{module_name}.from_pretrained({suggested_path!r})` "
        f"if {suggested_path} exists on disk."
    ) if suggested_path else (
        " A trained checkpoint is required for production use."
    )
    warnings.warn(
        f"{module_name} instantiated WITHOUT a loaded checkpoint. "
        f"Forward pass will produce random projections of the input.{suggestion}",
        UntrainedScaffoldWarning,
        stacklevel=3,
    )


class TrainedMarkerMixin:
    """Mixin that adds ``is_trained()`` + ``from_pretrained(path)`` to any
    ``nn.Module`` so the honest-scaffold contract is uniform across the
    architecture catalogue.
    """

    _is_trained: bool = False

    def is_trained(self) -> bool:
        return getattr(self, "_is_trained", False)

    def mark_trained(self) -> None:
        """Used by ``from_pretrained()`` AND by training scripts after the
        final epoch; flips the marker so audit consumers stop seeing the
        scaffold warning.
        """
        object.__setattr__(self, "_is_trained", True)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **module_kwargs):
        """Instantiate the module, ``load_state_dict()`` from disk, mark trained.

        Raises ``FileNotFoundError`` if the checkpoint is missing — the
        bare scaffold path is never silently used here.
        """
        import torch

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint {path} not found; cannot instantiate "
                f"{cls.__name__} via from_pretrained(). The bare "
                f"scaffold (random weights) is available via the "
                f"normal constructor but emits UntrainedScaffoldWarning."
            )
        # Suppress the bare-init warning since we're about to load weights.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UntrainedScaffoldWarning)
            module = cls(**module_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Accept either a raw state_dict or a {"state_dict": ..., ...} wrapper.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        module.load_state_dict(state)  # type: ignore[attr-defined]
        module.mark_trained()
        return module


__all__ = [
    "TrainedMarkerMixin",
    "UntrainedScaffoldWarning",
    "warn_untrained",
]
