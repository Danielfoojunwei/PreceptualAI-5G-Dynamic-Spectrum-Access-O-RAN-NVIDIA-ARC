"""Spectrum layer — federated Dynamic Spectrum Access (DSA) on licensed bands.

The real bridge from the robust/secure federated aggregators to an RL spectrum-
decision agent (the NTU FCP problem class: federated deep-RL for dynamic
spectrum access; privacy-preserving power control for LEO-satellite IoT; secure
DSA via swarm learning / unlearning):

* :mod:`~horizon_ric.spectrum.dsa_env`     — a real DSA MDP (N SUs, K channels,
  stochastic PU occupancy, collisions), numpy-only.
* :mod:`~horizon_ric.spectrum.federated_q` — torch-free federated tabular
  Q-learning whose server aggregation calls
  :mod:`horizon_ric.federated.robust` / :mod:`horizon_ric.federated.secure`.
* :mod:`~horizon_ric.spectrum.pipeline`    — one decision end-to-end: policy →
  Shield → hash-chained evidence record + SafetyCertificate.
"""

from horizon_ric.spectrum.dsa_env import (
    DSAConfig,
    DSAEnv,
    DSAWorld,
    encode_sensing,
    n_actions,
    n_states,
)
from horizon_ric.spectrum.federated_q import (
    FederatedDSAResult,
    QLearnConfig,
    craft_poison_q,
    evaluate_policy,
    federated_dsa_round,
    train_local_q,
)
from horizon_ric.spectrum.pipeline import (
    ChannelPlan,
    DSADecisionConfig,
    DSADecisionResult,
    action_from_policy,
    decide_and_record,
)

__all__ = [
    "DSAConfig",
    "DSAEnv",
    "DSAWorld",
    "encode_sensing",
    "n_actions",
    "n_states",
    "FederatedDSAResult",
    "QLearnConfig",
    "craft_poison_q",
    "evaluate_policy",
    "federated_dsa_round",
    "train_local_q",
    "ChannelPlan",
    "DSADecisionConfig",
    "DSADecisionResult",
    "action_from_policy",
    "decide_and_record",
]
