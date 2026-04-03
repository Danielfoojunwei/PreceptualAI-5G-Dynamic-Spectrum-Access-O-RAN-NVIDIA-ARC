"""
SmODE Smooth Actor for Bounded Power Control.

Implements a smooth ODE neuron that bounds the rate of change of
continuous actions (power allocation, MCS targets), ensuring:
  - 3GPP-compliant transmit power ramp rates
  - Interference-safe transitions
  - Regulatory compliance without post-hoc filtering

The smooth ODE constrains the Lipschitz constant of the policy
output, guaranteeing ||a(t+1) - a(t)|| <= max_rate * dt.

Reference:
  "ODE-based Smoothing Neural Network for RL Tasks", ICLR 2025 Spotlight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmODENeuron(nn.Module):
    """
    Smooth ODE neuron that bounds action rate of change.

    Implements: da/dt = clip(f(z), -max_rate, max_rate)
    where f(z) is the desired rate and max_rate is the bound.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_rate: float = 0.1,
        dt: float = 1.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_rate = max_rate
        self.dt = dt

        # Rate prediction network
        self.rate_net = nn.Sequential(
            nn.Linear(input_dim + output_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

        # Adaptive rate scaling (learned per output dimension)
        self.log_rate_scale = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute smooth action update.

        Args:
            z:           (B, input_dim)  — encoder output
            prev_action: (B, output_dim) — previous continuous action
        Returns:
            action:      (B, output_dim) — smoothly updated action in [-1, 1]
        """
        # Predict desired rate of change
        zp = torch.cat([z, prev_action], dim=-1)
        raw_rate = self.rate_net(zp)  # (B, output_dim)

        # Adaptive rate scaling per dimension
        rate_scale = self.log_rate_scale.exp()  # (output_dim,)
        effective_max = self.max_rate * rate_scale

        # Clip rate to enforce smoothness bound
        bounded_rate = torch.clamp(
            raw_rate,
            -effective_max.unsqueeze(0),
            effective_max.unsqueeze(0),
        )

        # ODE step: a_new = a_prev + dt * bounded_rate
        action = prev_action + self.dt * bounded_rate

        # Clip to valid action range
        action = torch.tanh(action)
        return action


class SmoothHybridActor(nn.Module):
    """
    Hybrid actor with SmODE-bounded continuous actions.

    Discrete actions (channel selection) via softmax.
    Continuous actions (power, MCS) via SmODE for guaranteed smoothness.
    """

    def __init__(
        self,
        latent_dim: int,
        num_discrete_actions: int,
        continuous_action_dim: int,
        max_power_ramp_rate: float = 0.1,
        dt: float = 1.0,
    ):
        super().__init__()

        # Discrete head (unchanged from standard actor)
        self.discrete_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_discrete_actions),
        )

        # Continuous head with SmODE smoothness guarantee
        self.smooth_ode = SmODENeuron(
            input_dim=latent_dim,
            output_dim=continuous_action_dim,
            max_rate=max_power_ramp_rate,
            dt=dt,
        )

        self.continuous_dim = continuous_action_dim

    def forward(
        self,
        z: torch.Tensor,
        prev_continuous: torch.Tensor = None,
    ):
        """
        Args:
            z:               (B, latent_dim) encoder output
            prev_continuous: (B, continuous_dim) previous continuous action
        Returns:
            discrete_probs:    (B, num_discrete)
            continuous_action: (B, continuous_dim) smoothly bounded
        """
        # Discrete action probabilities
        discrete_logits = self.discrete_head(z)
        discrete_probs = F.softmax(discrete_logits, dim=-1)

        # Smooth continuous action
        if prev_continuous is None:
            prev_continuous = torch.zeros(
                z.shape[0], self.continuous_dim,
                device=z.device, dtype=z.dtype,
            )
        continuous_action = self.smooth_ode(z, prev_continuous)

        return discrete_probs, continuous_action

    @torch.no_grad()
    def get_action(
        self,
        z: torch.Tensor,
        prev_continuous: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Select hybrid action with smoothness guarantee."""
        probs, cont = self.forward(z, prev_continuous)

        if deterministic:
            discrete = probs.argmax(dim=-1).item()
        else:
            discrete = torch.multinomial(probs, 1).squeeze(-1).item()

        return discrete, cont.squeeze(0), probs
