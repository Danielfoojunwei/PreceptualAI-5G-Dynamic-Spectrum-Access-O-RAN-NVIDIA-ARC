"""DomainAdapter — abstract base for vertical adapters.

Every vertical (ntn_airan, power_grid, fleet, ...) implements this contract.
The platform core consumes only this interface; verticals are swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class DomainAdapter(ABC):
    """Contract a vertical adapter must implement.

    The platform core (world model + controller + federated + canary)
    depends only on this interface. Vertical-specific logic (NTN encoder,
    GSO constraints, beam-pointing actions) lives in the concrete subclass.
    """

    name: str  # e.g. "ntn_airan"
    version: str  # e.g. "0.1.0"

    @abstractmethod
    def encode(self, raw_telemetry: dict[str, Any]) -> torch.Tensor:
        """Encode raw vertical-specific telemetry into the platform's latent space.

        Args:
            raw_telemetry: dict of vertical-specific tensors / metadata.
                For NTN: {"telemetry": (T, F), "ephemeris": (T, E),
                          "paa_h_matrix": (T, N_elem, F_subc) complex}

        Returns:
            z: (B, latent_dim) tensor in the platform's shared latent space.
        """

    @abstractmethod
    def decode_action(self, action_logits: torch.Tensor) -> dict[str, Any]:
        """Decode platform-emitted action logits into vertical-specific actions.

        Args:
            action_logits: (B, action_dim) raw network outputs.

        Returns:
            structured action dict, e.g. for NTN:
                {"beam_id": int, "tx_power_dBm": float, "freq_bin": int,
                 "null_directions": list[tuple[float, float]],
                 "paa_weights": complex tensor}
        """

    @abstractmethod
    def constraints(self) -> "ConstraintLayer":  # noqa: F821
        """Return the hard-constraint layer for this vertical.

        Returns:
            ConstraintLayer instance that projects infeasible actions onto
            the feasible set BEFORE A1 emission.
        """

    @abstractmethod
    def metrics(self) -> "MetricSuite":  # noqa: F821
        """Return the metric suite (Section-7-style) for this vertical.

        Returns:
            MetricSuite that computes the per-vertical KPIs the
            PolicyValidator gates promotion on.
        """

    @abstractmethod
    def scenario_embedding(self, scenario_id: str) -> torch.Tensor:
        """Return a learned embedding for the named scenario.

        Used by HyperRouter to specialize the policy per scenario class.

        Args:
            scenario_id: e.g. "maritime", "disaster_response", "rural",
                         "ai_ran_edge".

        Returns:
            (scenario_emb_dim,) embedding tensor.
        """

    def supported_scenarios(self) -> list[str]:
        """List the scenario IDs this vertical understands. Default: empty.

        Subclasses override to declare their scenarios.
        """
        return []
