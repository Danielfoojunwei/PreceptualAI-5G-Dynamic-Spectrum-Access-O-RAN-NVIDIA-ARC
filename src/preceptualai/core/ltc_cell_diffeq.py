"""
LTC Cell with torchdiffeq GPU-accelerated ODE solvers.

Uses torchdiffeq for:
  - Adaptive solvers (dopri5, adaptive_heun) for accuracy
  - Adjoint method for O(1) memory backpropagation
  - Fixed-step solvers (euler, midpoint, rk4) for benchmarking

Reference: Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint, odeint_adjoint
    _HAS_TORCHDIFFEQ = True
except ImportError:
    _HAS_TORCHDIFFEQ = False


class LTCODEFunc(nn.Module):
    """ODE right-hand side for the LTC cell: dh/dt = (1/tau) * (-h + f(x, h))."""

    def __init__(self, W_h: nn.Linear, x_proj: torch.Tensor, tau: torch.Tensor):
        super().__init__()
        self.W_h = W_h
        self.x_proj = x_proj
        self.tau = tau

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        f = torch.tanh(self.W_h(h) + self.x_proj)
        return (1.0 / self.tau) * (-h + f)


class LTCCellDiffeq(nn.Module):
    """
    LTC cell using torchdiffeq for ODE integration.

    Supports adaptive (dopri5) and fixed-step (euler, rk4) solvers,
    with optional adjoint method for O(1) memory backpropagation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 1.0,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        if not _HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq is required for LTCCellDiffeq. "
                "Install with: pip install torchdiffeq"
            )

        self.hidden_dim = hidden_dim
        self.dt = dt
        self.solver = solver
        self.use_adjoint = use_adjoint
        self.rtol = rtol
        self.atol = atol

        # Fused input projection: [W_x_out, W_tau_out]
        self.W_x_fused = nn.Linear(input_dim, hidden_dim * 2)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))

        self._integrate = odeint_adjoint if use_adjoint else odeint
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x_fused.weight)
        nn.init.zeros_(self.W_x_fused.bias)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_h.bias)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Single ODE step using torchdiffeq.

        Args:
            x_t: (B, input_dim)
            h:   (B, hidden_dim)
        Returns:
            h':  (B, hidden_dim)
        """
        # Fused input projection
        x_fused = self.W_x_fused(x_t)
        x_proj, tau_proj = x_fused.chunk(2, dim=-1)
        tau = self.tau_base + F.softplus(tau_proj)

        # Define ODE function for this step
        ode_func = LTCODEFunc(self.W_h, x_proj, tau)

        # Integrate from t=0 to t=dt
        t_span = torch.tensor([0.0, self.dt], device=h.device, dtype=h.dtype)

        h_new = self._integrate(
            ode_func, h, t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )[-1]  # Take final state

        return h_new
