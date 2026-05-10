"""Constraint-aware MPPI / CEM planner — "TD-MPC2-style" historically; honest disclosure below.

> **Honest scope (v3 audit pass).** The class name and the historical
> filename reference Hansen 2024 TD-MPC2, but the **runtime planner that
> ships in production is the trained-value-head-OFF variant** — i.e.
> CEM (Williams 2017 / de Boer 2005) with softmax-elite refit over the
> learned latent dynamics, plus pluggable Lagrangian constraint
> penalties. The training script ``scripts/train_tdmpc_planner.py``
> attempted to fit a TD-bootstrap value head; it diverged in repeated
> runs (the model-card notes are verbatim) and was shipped disabled
> because trained MPPI lost to random MPPI on the validation set. The
> Hansen 2024 paper requires the TD-bootstrap to claim the "TD-MPC2"
> result; without it, this file is honestly characterised as **CEM with
> MPPI-elite refit and constraint costs**. The constraint-cost
> machinery (PFD / EPFD / spectrum-mask / LI penalties pluggable into
> the rollout score) IS engineering-grade and works against any
> underlying dynamics model.

MPPI (Model Predictive Path Integral) over the learned latent dynamics:
    1. Sample N candidate action sequences from a Gaussian proposal.
    2. Roll each sequence through the world model.
    3. Score each rollout: cumulative reward minus constraint penalties.
    4. Update the proposal toward the high-scoring tail (exponential
       importance weighting), iterate K times.
    5. Return the mean of the final proposal as the recommended action
       sequence — or `nominal_action()` to lock in the first step.

Constraint penalties are pluggable callables; the rApp wires
`policy.constraints.PreceptualAIConstraintLayer.lagrangian_violation` into
this list so EPFD / spectral-mask / GPU-capacity violations carry cost
weight in the rollout score.

References:
    Hansen et al. *Temporal Difference Learning for Model Predictive
        Control* (TD-MPC), ICML 2022 — arXiv:2203.04955.
    Hansen, Su, Wang *TD-MPC2: Scalable, Robust World Models for
        Continuous Control*, ICLR 2024 — arXiv:2310.16828.
    Williams et al. *Model Predictive Path Integral Control*, JGCD 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


class _DynamicsLike(Protocol):
    """Anything with a `rollout(z0, actions) -> trajectory` method."""

    def rollout(
        self, z0: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass
class TDMPCConfig:
    horizon: int = 12
    """Plan H steps ahead — keep below the latent dynamics' training horizon."""
    n_samples: int = 256
    n_iterations: int = 6
    elite_frac: float = 0.1
    """Fraction of top-scoring rollouts used to refit the proposal."""
    temperature: float = 1.0
    """Softmax temperature for the importance weights."""
    init_std: float = 1.0
    min_std: float = 0.05
    constraint_weight: float = 100.0
    """Lagrangian multiplier on summed constraint violations."""
    random_seed: int | None = None
    """Default RNG seed for `plan()`. None = non-reproducible (legacy);
    pin to an int for auditable / counterfactual-reproducible plans
    (Devil A/C #9, #26)."""


@dataclass
class PlanResult:
    actions: torch.Tensor          # (H, d_action) — recommended sequence
    expected_score: float
    elite_score: float
    elite_violation: float
    random_seed: int | None = None
    """Seed used to produce this plan. Pinned by `plan(..., random_seed=...)`
    or inherited from `TDMPCConfig.random_seed`. None marks the legacy
    non-reproducible path (Devil A/C #9, #26)."""
    elite_actions: torch.Tensor | None = None
    """Final-iteration elite action sequences (E, H, A) — surfaced so the
    counterfactual builder can construct deterministic alternatives
    pinned to the same seed as `actions`."""
    elite_rewards: torch.Tensor | None = None
    """Per-elite cumulative reward (E,)."""
    elite_violations: torch.Tensor | None = None
    """Per-elite cumulative constraint violation (E,)."""


class TDMPCPlanner:
    """MPPI planner over a learned latent dynamics + value model.

    When ``value_head`` and/or ``policy_prior`` are passed in, the planner
    behaves as the trained TD-MPC2 sample/score loop:
        * ``policy_prior(z0)`` returns ``(mean (H,A), log_std (H,A))`` and
          replaces the default ``N(0, init_std)`` initialisation of the MPPI
          proposal distribution.
        * ``value_head(z_final, last_action)`` adds a long-horizon Q estimate
          to the per-rollout score (weight ``value_weight``), so the planner
          sees terminal value, not only the immediate reward.
    Both knobs are optional; when neither is provided the planner is the
    "scaffold (random)" baseline.
    """

    def __init__(
        self,
        dynamics: _DynamicsLike,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        constraint_fns: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
        config: TDMPCConfig | None = None,
        action_dim: int | None = None,
        device: torch.device | str = "cpu",
        value_head: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        policy_prior: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        value_weight: float = 0.5,
    ):
        """Args:
            dynamics: any object with `rollout(z0, actions) → traj`.
            reward_fn: maps (latent_traj (B,H,Z), action_traj (B,H,A)) →
                       per-step reward (B,H).
            constraint_fns: list of callables (latent_traj, action_traj) →
                       per-step violation magnitude (B,H), each ≥ 0.
            config: TDMPCConfig.
            action_dim: integer dimensionality of one action vector. Required
                       since this planner is generic over the dynamics' action
                       space.
            device: torch device for sampling.
            value_head: optional callable Q(z_final, a_last) → (B,) scalar.
                       Added to the rollout score with weight ``value_weight``.
            policy_prior: optional callable π(z0) → (mean(H,A), log_std(H,A))
                       used to initialise the MPPI proposal distribution.
            value_weight: scalar multiplier on the value-head contribution.
        """
        if action_dim is None or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.constraint_fns = list(constraint_fns or [])
        self.cfg = config or TDMPCConfig()
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.value_head = value_head
        self.policy_prior = policy_prior
        self.value_weight = float(value_weight)

    def plan(
        self, z0: torch.Tensor, random_seed: int | None = None,
    ) -> PlanResult:
        """Plan from a single starting latent z0 of shape (d_latent,).

        ``random_seed`` (Devil A/C #9, #26) — when not None, pins the
        planner's RNG so the returned ``actions`` and elite set are a
        pure function of (z0, dynamics, reward_fn, constraint_fns,
        config, seed). The same seed → byte-identical PlanResult.
        Falls back to ``config.random_seed`` if both are None the path
        remains non-reproducible (legacy behaviour).
        """
        if z0.ndim != 1:
            raise ValueError(f"z0 must be 1-D (d_latent,), got {tuple(z0.shape)}")
        H = self.cfg.horizon
        A = self.action_dim
        device = self.device

        seed = random_seed if random_seed is not None else self.cfg.random_seed
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            generator = None

        if self.policy_prior is not None:
            with torch.no_grad():
                prior_mean, prior_log_std = self.policy_prior(z0.unsqueeze(0))
            # Accept either (H, A) or (1, H, A) depending on the prior.
            if prior_mean.ndim == 3:
                prior_mean = prior_mean[0]
                prior_log_std = prior_log_std[0]
            if prior_mean.shape != (H, A):
                # Broadcast a single-step prior across H steps.
                prior_mean = prior_mean.reshape(-1, A)
                prior_log_std = prior_log_std.reshape(-1, A)
                if prior_mean.shape[0] == 1:
                    prior_mean = prior_mean.expand(H, A).contiguous()
                    prior_log_std = prior_log_std.expand(H, A).contiguous()
            mean = prior_mean.to(device)
            std = prior_log_std.exp().to(device).clamp_min(self.cfg.min_std)
        else:
            mean = torch.zeros(H, A, device=device)
            std = torch.full_like(mean, self.cfg.init_std)

        elite_scores = None  # populated each iteration
        elite_idx = None
        violations = None
        for _ in range(self.cfg.n_iterations):
            if generator is not None:
                noise = torch.randn(
                    self.cfg.n_samples, H, A,
                    device=device, generator=generator,
                )
            else:
                noise = torch.randn(self.cfg.n_samples, H, A, device=device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise   # (N, H, A)
            actions = actions.clamp(-3.0, 3.0)  # generic bound; rApp clips later

            z0_batch = z0.unsqueeze(0).expand(self.cfg.n_samples, -1)
            traj = self.dynamics.rollout(z0_batch, actions)          # (N, H, Z)

            rewards = self.reward_fn(traj, actions)                  # (N, H)
            scores = rewards.sum(dim=-1)                             # (N,)
            if self.value_head is not None:
                with torch.no_grad():
                    q = self.value_head(traj[:, -1, :], actions[:, -1, :])
                if q.ndim > 1:
                    q = q.squeeze(-1)
                scores = scores + self.value_weight * q
            violations = torch.zeros_like(scores)
            for fn in self.constraint_fns:
                v = fn(traj, actions)                                # (N, H)
                violations = violations + v.sum(dim=-1)
            scores = scores - self.cfg.constraint_weight * violations

            # MPPI importance weighting → exponential weights of the elite set.
            n_elite = max(int(round(self.cfg.elite_frac * self.cfg.n_samples)), 1)
            elite_scores, elite_idx = torch.topk(scores, n_elite)
            elite_actions = actions[elite_idx]                       # (E, H, A)

            elite_norm = (elite_scores - elite_scores.max()) / max(
                self.cfg.temperature, 1e-6
            )
            weights = torch.softmax(elite_norm, dim=0)               # (E,)
            mean = (weights[:, None, None] * elite_actions).sum(dim=0)
            # Variance from elite spread, lower-bounded.
            var = (
                weights[:, None, None] * (elite_actions - mean.unsqueeze(0)) ** 2
            ).sum(dim=0)
            std = var.clamp_min(self.cfg.min_std ** 2).sqrt()

        # Re-score the final mean to report.
        traj_mean = self.dynamics.rollout(
            z0.unsqueeze(0), mean.unsqueeze(0)
        )[0]
        reward_sum = self.reward_fn(
            traj_mean.unsqueeze(0), mean.unsqueeze(0)
        ).sum().item()
        violation_sum = 0.0
        for fn in self.constraint_fns:
            violation_sum += float(fn(
                traj_mean.unsqueeze(0), mean.unsqueeze(0),
            ).sum().item())

        return PlanResult(
            actions=mean,
            expected_score=reward_sum,
            elite_score=float(elite_scores.mean().item()),
            elite_violation=float(violations[elite_idx].mean().item()),
            random_seed=int(seed) if seed is not None else None,
            elite_actions=elite_actions.detach(),
            elite_rewards=elite_scores.detach(),
            elite_violations=violations[elite_idx].detach(),
        )

    def nominal_action(
        self, z0: torch.Tensor, random_seed: int | None = None,
    ) -> torch.Tensor:
        """Return only the first action of the planned sequence (the
        action the rApp will actually emit this slot)."""
        return self.plan(z0, random_seed=random_seed).actions[0]


__all__ = ["PlanResult", "TDMPCConfig", "TDMPCPlanner"]
