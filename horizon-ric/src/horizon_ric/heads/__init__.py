"""Risk heads — multi-head model on shared latent (PRD §2.3, improvement §1.1).

Each head consumes the JEPA encoder's z_resource latent and emits a
calibrated risk prediction at multiple horizons.

Heads:
    sla_risk:   SLA breach probability at 30s/1min/5min   (SHIPS today)
    beam_risk:  beam degradation / handover failure       (Phase 2)
    gateway_risk: gateway congestion / unavailability     (Phase 2)
    compute_risk: edge GPU/CPU overload                   (Phase 2)

All heads use the two-hot symlog regression pattern (from DreamerV3 / parent
project's M13) for robust multi-OOM-scale targets — see `_two_hot.py`.
"""

from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead

__all__ = ["SLARiskConfig", "SLARiskHead"]
