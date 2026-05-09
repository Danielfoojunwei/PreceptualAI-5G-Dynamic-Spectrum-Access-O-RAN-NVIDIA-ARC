"""ConstraintLayer — abstract base for hard regulatory/operational constraints.

A ConstraintLayer is non-trainable. It computes feasibility and projects
infeasible actions onto the feasible set. Hard constraints (regulatory PFD
limits, ITU spectral masks, edge GPU capacity ceilings) must be enforced
mathematically — soft penalties in reward are insufficient for regulatory
compliance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ConstraintViolation:
    """One specific constraint violation reported by check_feasibility."""

    constraint_id: str  # e.g. "gso_pfd_arc_140W"
    severity: str  # "hard" or "soft"
    margin_dB: float | None  # how far over (negative if compliant)
    message: str  # human-readable explanation


class ConstraintLayer(ABC):
    """Hard-constraint enforcement contract.

    Used in two places:
      1. During training: computes Lagrangian violation signals for the policy.
      2. During inference: projects any infeasible action onto the feasible
         set before A1 emission. Any production action MUST pass the layer.
    """

    @abstractmethod
    def check_feasibility(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> list[ConstraintViolation]:
        """Return list of violations for a candidate action.

        Args:
            action: structured action dict from DomainAdapter.decode_action().
            context: runtime context (ephemeris, neighbor states, GSO catalog).

        Returns:
            List of violations (empty list = feasible).
        """

    @abstractmethod
    def project(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[dict[str, Any], list[ConstraintViolation]]:
        """Project an action onto the feasible set.

        Args:
            action: candidate action.
            context: runtime context.

        Returns:
            (feasible_action, applied_corrections):
                feasible_action: action guaranteed to satisfy all hard constraints.
                applied_corrections: list of corrections applied during projection.
        """

    @abstractmethod
    def lagrangian_violation(
        self, action: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        """Differentiable scalar Lagrangian violation, for training use.

        Args:
            action: (B, action_dim) continuous action tensor.
            context: runtime context.

        Returns:
            (B,) tensor of constraint-violation magnitudes (≥0).
            Used as the constraint signal in CSAC.
        """

    @abstractmethod
    def hard_constraint_ids(self) -> list[str]:
        """List of constraint IDs that are HARD (regulatory)."""
