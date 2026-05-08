"""Core neural primitives: Liquid CfC, physics-residual, timing budgets.

These are the small, composable PyTorch building blocks used by the
encoder, the world model, and the planner. Each module:

  * runs in single-digit ms on Jetson Orin Nano (the project's edge target),
  * is quantization-aware (CfC + Liquid-S4 are int8-friendly),
  * has a documented citation back to the source paper.

Reserved for v0.2:
    cfc_core            Liquid CfC cell (Hasani 2022 Nature MI)
    physics_residual    Compositional ŷ = f_phys + g_θ wrapper
    timing_budgets      L1/L2/A1 inference admit-control gates

NOT YET implemented:
    liquid_s4           Hasani 2023 ICLR Liquid-S4 backbone
    latent_ode          Rubanova 2019 Latent-ODE for irregular series
    multi_rate_fusion   SeFT triplets + CRU + Liquid-S4
"""

from horizon_ric.core.cfc_core import CfCCell, CfCConfig
from horizon_ric.core.latent_dynamics import LatentDynamics, LatentDynamicsConfig
from horizon_ric.core.latent_ode import (
    LatentODEConfig,
    LatentODEPosterior,
    ODESolverHead,
    reparameterize,
)
from horizon_ric.core.liquid_s4 import LiquidS4, LiquidS4Config
from horizon_ric.core.physics_residual import PhysicsResidualHead
from horizon_ric.core.timing_budgets import (
    a1_admit,
    edge_compute_lower_bound_us,
    end_to_end_latency_ms,
    inference_budget_ms,
    l1_admit,
    l2_inference_budget_us,
    memory_bound_latency_ms,
)

__all__ = [
    "CfCCell",
    "CfCConfig",
    "LatentDynamics",
    "LatentDynamicsConfig",
    "LatentODEConfig",
    "LatentODEPosterior",
    "LiquidS4",
    "LiquidS4Config",
    "ODESolverHead",
    "PhysicsResidualHead",
    "a1_admit",
    "edge_compute_lower_bound_us",
    "end_to_end_latency_ms",
    "inference_budget_ms",
    "l1_admit",
    "l2_inference_budget_us",
    "memory_bound_latency_ms",
    "reparameterize",
]
