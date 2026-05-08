"""Federated learning — weights-only FedAvg / FedProx.

The hard contract per project memory (`project_uhci_constraints.md`):
**no raw KPMs ever leave the device**. Every aggregator here transports
only model weight deltas + signed validation reports — never telemetry.

Module map:
    aggregator    FedAvg + FedProx aggregation primitives (pure Python).
    sparsifier    Top-k + signSGD compression to fit cellular backhaul.
    flwr_glue     Optional binding to Flower (`flwr`) when present;
                  exposes our aggregator as a Flower Strategy.

Public API:

    >>> from horizon_ric.federated import FedAvg, FedProx, top_k_sparsify
    >>> agg = FedAvg()
    >>> global_state = agg.aggregate(client_updates, client_weights)
"""

from horizon_ric.federated.aggregator import (
    DEFAULT_FEDPROX_MU,
    ClientUpdate,
    FedAvg,
    FedProx,
    aggregate_fedavg,
    aggregate_fedprox,
    default_aggregator,
)
from horizon_ric.federated.secure_aggregation import (
    SecureFedAvg,
    ShamirSecretSharing,
)
from horizon_ric.federated.sparsifier import (
    quantize_int8,
    sign_sgd_compress,
    top_k_sparsify,
)

__all__ = [
    "ClientUpdate",
    "DEFAULT_FEDPROX_MU",
    "FedAvg",
    "FedProx",
    "SecureFedAvg",
    "ShamirSecretSharing",
    "aggregate_fedavg",
    "aggregate_fedprox",
    "default_aggregator",
    "quantize_int8",
    "sign_sgd_compress",
    "top_k_sparsify",
]
