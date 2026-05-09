"""PreceptualAI — NTN-Aware AI-RAN Resource-Orchestration rApp Suite.

What ships today (v0.1.0):
  * R1 + A1 adapters and rApp lifecycle (`horizon_ric.rapp`).
  * Edge inference agent (`horizon_ric.agent`).
  * Hard-constraint layer with GSO PFD floor (`horizon_ric.policy`).
  * SLA risk head with two-hot symlog regression (`horizon_ric.heads`).
  * Compositional-WM physics modules — ITU-R P.525/P.676/P.838 + 3GPP
    TR 38.901 / ITU-R F.1336 (`horizon_ric.planner.physics`).
  * Tamper-evident evidence store (`horizon_ric.evidence`).
  * Maritime synthetic scenario for offline training
    (`horizon_ric.scenarios`).
  * Health + Prometheus endpoints (`horizon_ric.rapp.health`).

What is roadmapped but NOT yet implemented (do NOT depend on these):
  * Encoder (`horizon_ric.encoder`)            — Phase 1
  * Federated coordinator (`horizon_ric.federated`) — Phase 2
  * Cross-operator trading (`horizon_ric.trading`)  — Phase 3
  * O1 NETCONF adapter                                — Phase 1.5

Each of these submodules raises a clear `NotImplementedError` at import
of any name; the package never silently no-ops.
"""

from horizon_ric.evidence import (
    DecisionRecord,
    JsonlEvidenceStore,
    SqliteEvidenceStore,
    generate_human_explanation,
)
from horizon_ric.rapp import HorizonRAppLifecycle, RAppState

__version__ = "0.1.0"

__all__ = [
    "DecisionRecord",
    "HorizonRAppLifecycle",
    "JsonlEvidenceStore",
    "RAppState",
    "SqliteEvidenceStore",
    "__version__",
    "generate_human_explanation",
]
