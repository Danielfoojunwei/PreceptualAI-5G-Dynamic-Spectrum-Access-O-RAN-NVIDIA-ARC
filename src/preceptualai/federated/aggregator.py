"""
Hybrid Federated Aggregator for PreceptualAI.

Implements the novel LTC-aware aggregation strategy that partitions model
parameters into two groups:

  1. **Structural weights** (W_h, W_x, heads, layer norms, projections):
     Aggregated via standard FedAvg — weighted average proportional to each
     device's sample count.

  2. **Time-constant weights** (W_tau, tau_base):
     Partially personalized per device via a configurable mixing ratio
     ``tau_mix_ratio``.  The returned tau weights for each device are:

         mixed_tau = tau_mix_ratio * global_avg + (1 - tau_mix_ratio) * device_local

     This preserves device-specific temporal dynamics (adaptation to local
     radio conditions) while still benefiting from the global consensus.

Rationale:
    In LTC networks the time constants tau(x) govern *how quickly* the hidden
    state tracks input changes.  Different 5G environments (urban driving vs
    static indoor) exhibit vastly different channel dynamics.  Full FedAvg
    would wash out these differences.  The hybrid scheme retains local temporal
    personality while still sharing structural knowledge (feature extraction,
    value estimation) across the fleet.
"""

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from preceptualai.federated.config import FLConfig


def _is_tau_weight(key: str) -> bool:
    """Return True if ``key`` names a time-constant parameter.

    Time-constant parameters are identified by containing ``W_tau`` or
    ``tau_base`` anywhere in the fully-qualified parameter name.  All other
    parameters are treated as structural.
    """
    return "W_tau" in key or "tau_base" in key


def _partition_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a state_dict into structural and tau subsets."""
    structural: Dict[str, torch.Tensor] = {}
    tau: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if _is_tau_weight(key):
            tau[key] = value
        else:
            structural[key] = value
    return structural, tau


class DeviceUpdate:
    """Stores a single device's contribution for one FL round."""

    __slots__ = ("device_id", "weights", "num_samples")

    def __init__(
        self,
        device_id: str,
        weights: Dict[str, Dict[str, torch.Tensor]],
        num_samples: int,
    ):
        self.device_id = device_id
        self.weights = weights  # {"actor": sd, "critic1": sd, "critic2": sd}
        self.num_samples = num_samples


class HybridFederatedAggregator:
    """
    Federated aggregator with LTC-aware hybrid weight merging.

    Typical lifecycle:

    .. code-block:: python

        agg = HybridFederatedAggregator(config)
        for fl_round in range(config.num_rounds):
            for device in fleet:
                agg.receive_update(device.id, device.get_weights(), device.n)
            agg.aggregate()
            for device in fleet:
                w = agg.get_personalized_model(device.id)
                device.apply_weights(w)
    """

    def __init__(self, config: FLConfig):
        self.config = config
        self.tau_mix_ratio = config.tau_mix_ratio

        # Per-round accumulator: list of DeviceUpdate
        self._round_updates: List[DeviceUpdate] = []

        # Latest global model (fully averaged, used for cold-start)
        self._global_weights: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        # Per-device tau weights (latest local values before mixing)
        self._device_tau_weights: Dict[
            str, Dict[str, Dict[str, torch.Tensor]]
        ] = {}  # device_id -> {"actor": {tau_keys}, "critic1": {...}, ...}

        # Global structural + global tau averages (split)
        self._global_structural: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._global_tau: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        self._round_number = 0

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    def receive_update(
        self,
        device_id: str,
        local_weights: Dict[str, Dict[str, torch.Tensor]],
        num_samples: int,
    ) -> None:
        """Buffer a device's local model weights for the current round.

        Args:
            device_id: Unique identifier for the reporting device.
            local_weights: Dict mapping component name (``"actor"``,
                ``"critic1"``, ``"critic2"``) to that component's
                ``state_dict``.
            num_samples: Number of environment samples the device collected
                during its local training phase.  Used as the weighting
                factor in FedAvg.
        """
        # Store latest local tau weights for this device (before aggregation)
        device_tau: Dict[str, Dict[str, torch.Tensor]] = {}
        for component, sd in local_weights.items():
            _, tau = _partition_state_dict(sd)
            device_tau[component] = {k: v.clone() for k, v in tau.items()}
        self._device_tau_weights[device_id] = device_tau

        self._round_updates.append(
            DeviceUpdate(
                device_id=device_id,
                weights={
                    comp: {k: v.clone() for k, v in sd.items()}
                    for comp, sd in local_weights.items()
                },
                num_samples=num_samples,
            )
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, Any]:
        """Run one round of aggregation over buffered device updates.

        Returns:
            A metrics dict with round number and device count.

        Raises:
            RuntimeError: If fewer than ``min_devices_per_round`` updates
                have been received.
        """
        n_updates = len(self._round_updates)
        if n_updates < self.config.min_devices_per_round:
            raise RuntimeError(
                f"Only {n_updates} device(s) reported, but "
                f"min_devices_per_round={self.config.min_devices_per_round}"
            )

        method = self.config.aggregation_method
        if method == "hybrid_ltc":
            self._aggregate_hybrid()
        elif method in ("fedavg", "fedprox"):
            self._aggregate_fedavg()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        self._round_number += 1
        metrics = {
            "round": self._round_number,
            "num_devices": n_updates,
            "method": method,
        }

        # Clear round buffer
        self._round_updates.clear()
        return metrics

    # ------------------------------------------------------------------
    # Hybrid aggregation (novel)
    # ------------------------------------------------------------------

    def _aggregate_hybrid(self) -> None:
        """Hybrid LTC-aware aggregation.

        Structural keys get standard FedAvg.  Tau keys also get FedAvg'd
        to produce a *global* tau average, but the final per-device model
        mixes global and local tau via ``tau_mix_ratio``.
        """
        total_samples = sum(u.num_samples for u in self._round_updates)
        if total_samples == 0:
            total_samples = len(self._round_updates)  # uniform fallback

        # Determine component names from first update
        components = list(self._round_updates[0].weights.keys())

        global_structural: Dict[str, Dict[str, torch.Tensor]] = {}
        global_tau: Dict[str, Dict[str, torch.Tensor]] = {}
        global_full: Dict[str, Dict[str, torch.Tensor]] = {}

        for comp in components:
            # Weighted average for structural keys
            structural_avg: Dict[str, torch.Tensor] = OrderedDict()
            tau_avg: Dict[str, torch.Tensor] = OrderedDict()

            for update in self._round_updates:
                weight = update.num_samples / total_samples
                sd = update.weights[comp]
                structural, tau = _partition_state_dict(sd)

                for k, v in structural.items():
                    if k not in structural_avg:
                        structural_avg[k] = v.float() * weight
                    else:
                        structural_avg[k] = structural_avg[k] + v.float() * weight

                for k, v in tau.items():
                    if k not in tau_avg:
                        tau_avg[k] = v.float() * weight
                    else:
                        tau_avg[k] = tau_avg[k] + v.float() * weight

            global_structural[comp] = structural_avg
            global_tau[comp] = tau_avg

            # Full global model (for cold-start): structural + tau_avg
            merged = OrderedDict()
            merged.update(structural_avg)
            merged.update(tau_avg)
            global_full[comp] = merged

        self._global_structural = global_structural
        self._global_tau = global_tau
        self._global_weights = global_full

    # ------------------------------------------------------------------
    # Standard FedAvg / FedProx aggregation
    # ------------------------------------------------------------------

    def _aggregate_fedavg(self) -> None:
        """Standard FedAvg: weighted average of all parameters."""
        total_samples = sum(u.num_samples for u in self._round_updates)
        if total_samples == 0:
            total_samples = len(self._round_updates)

        components = list(self._round_updates[0].weights.keys())
        global_weights: Dict[str, Dict[str, torch.Tensor]] = {}

        for comp in components:
            avg: Dict[str, torch.Tensor] = OrderedDict()
            for update in self._round_updates:
                weight = update.num_samples / total_samples
                sd = update.weights[comp]
                for k, v in sd.items():
                    if k not in avg:
                        avg[k] = v.float() * weight
                    else:
                        avg[k] = avg[k] + v.float() * weight
            global_weights[comp] = avg

        self._global_weights = global_weights
        # Also maintain structural/tau split for consistency
        self._global_structural = {}
        self._global_tau = {}
        for comp, sd in global_weights.items():
            s, t = _partition_state_dict(sd)
            self._global_structural[comp] = s
            self._global_tau[comp] = t

    # ------------------------------------------------------------------
    # Model retrieval
    # ------------------------------------------------------------------

    def get_personalized_model(
        self, device_id: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a model with global structural weights and mixed tau.

        For a known device (one that participated in at least one round),
        the tau weights are:

            mixed = tau_mix_ratio * global_tau + (1 - tau_mix_ratio) * local_tau

        For an unknown device (cold-start), this is equivalent to
        ``get_global_model()`` — the fully averaged model.

        Args:
            device_id: The device to build the personalized model for.

        Returns:
            Dict mapping component names to merged state_dicts.
        """
        if self._global_structural is None or self._global_tau is None:
            raise RuntimeError("No aggregation has been performed yet.")

        # Cold-start: device has no local tau history
        if device_id not in self._device_tau_weights:
            return self.get_global_model()

        local_tau = self._device_tau_weights[device_id]
        result: Dict[str, Dict[str, torch.Tensor]] = {}
        ratio = self.tau_mix_ratio

        for comp in self._global_structural:
            merged = OrderedDict()
            # Structural weights: take global
            merged.update(self._global_structural[comp])

            # Tau weights: mix global and local
            global_tau_comp = self._global_tau[comp]
            local_tau_comp = local_tau.get(comp, {})

            for k in global_tau_comp:
                g = global_tau_comp[k]
                if k in local_tau_comp:
                    local_val = local_tau_comp[k]
                    merged[k] = ratio * g + (1.0 - ratio) * local_val
                else:
                    merged[k] = g  # fallback to global if key missing locally

            result[comp] = merged

        return result

    def get_global_model(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return the fully averaged global model (cold-start friendly).

        All parameters (structural and tau) are the weighted average across
        all devices.  Suitable for new devices that have no local history.

        Returns:
            Dict mapping component names to complete state_dicts.
        """
        if self._global_weights is None:
            raise RuntimeError("No aggregation has been performed yet.")
        return copy.deepcopy(self._global_weights)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def round_number(self) -> int:
        """Number of completed aggregation rounds."""
        return self._round_number

    @property
    def registered_devices(self) -> List[str]:
        """Device IDs that have contributed at least one update."""
        return list(self._device_tau_weights.keys())

    def get_device_tau(
        self, device_id: str
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Return the stored local tau weights for a specific device.

        Returns None if the device has never participated.
        """
        return self._device_tau_weights.get(device_id)

    @property
    def num_pending_updates(self) -> int:
        """Number of updates received for the current (not yet aggregated) round."""
        return len(self._round_updates)
