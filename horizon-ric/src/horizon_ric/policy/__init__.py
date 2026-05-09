"""Policy & planner layer.

Three sub-modules:

    constraints       Differentiable hard-constraint projection layer
                      (ITU-R EPFD, ITU spectral masks, edge GPU ceiling).
    td_mpc_planner    TD-MPC2 MPPI planner over latent dynamics.
    diffusion_tail    Latent-space DDPM for tail-risk rollouts.

Wire: planner → constraint-layer projection → A1Adapter.emit_policy.
"""

from horizon_ric.policy.constraints import (
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)
from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
    candidates_from_planner_elite,
)
from horizon_ric.policy.diffusion_tail import (
    DiffusionConfig,
    DiffusionTailSampler,
)
from horizon_ric.policy.emit_guards import (
    EmitTrace,
    GuardFailure,
    run_guard_chain,
)
from horizon_ric.policy.li_constraint import (
    LIConstraint,
    LIJurisdictionRule,
)
from horizon_ric.policy.td_mpc_planner import (
    PlanResult,
    TDMPCConfig,
    TDMPCPlanner,
)

__all__ = [
    "AlternativeCandidate",
    "DiffusionConfig",
    "DiffusionTailSampler",
    "EmitTrace",
    "GSOArcEntry",
    "GuardFailure",
    "PreceptualAIConstraintConfig",
    "PreceptualAIConstraintLayer",
    "LIConstraint",
    "LIJurisdictionRule",
    "PlanResult",
    "TDMPCConfig",
    "TDMPCPlanner",
    "build_rejected_alternatives",
    "candidates_from_planner_elite",
    "run_guard_chain",
]
