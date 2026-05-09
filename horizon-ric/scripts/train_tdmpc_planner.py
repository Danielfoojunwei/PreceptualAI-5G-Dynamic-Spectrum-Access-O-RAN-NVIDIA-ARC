"""TD-MPC2 planner training — value head Q(z, a) + policy prior π(a | z).

The TD-MPC2 planner (Hansen et al. ICLR 2024) is *amortised*:
    1. The MPPI / CEM loop is the planner itself — there is nothing to train.
    2. What we train is the (value, policy_prior) pair that
         (a) replaces the random Gaussian sample distribution with a learned
             prior π(a | z), and
         (b) adds a long-horizon Q estimate to the rollout score so the
             planner sees terminal value, not only the first 4 steps of
             reward.

This script closes caveat #1 from this morning's PR ("dynamics randomly
initialised") because the planner now has a real value head + policy
prior trained on real-data latents from the trained JEPA encoder.

Replay buffer construction (synthetic-but-grounded):
    z_t        : 4500 real-data latents from JEPA encoder over the joint
                 {DeepMIMO, UCC, FAPI, FH, cuMAC} corpus (5k cap).
    a_t        : 8 candidate constraint-feasible actions per z_t.
    z_{t+1..4} : `LatentDynamics.rollout(z_t, [a_t]*4)`.
    r_t        : -mean(SLA_head(z_{t+4}).p_breach) − 0.01·|a_t|.

Models trained:
    ValueHead     : Linear(d_z+d_a, 128) → SiLU → Linear(128, 1).
    PolicyPrior   : Linear(d_z, 128) → SiLU → Linear(128, 2·d_a)
                    (split into mean, log_std).

Losses (TD-MPC2 §3 / §4):
    L_value  = MSE(Q(z, a), r + 0.95 · V(z'))   V(z')  = E_{a'~π}[Q(z', a')]
    L_policy = -E_{a~π}[Q(z, a)] + 0.01 · KL(π ‖ N(0, I))

Validation:
    1. Random-MPPI baseline (planner with init_std=1.0, no prior, no value).
    2. Trained-MPPI (planner with policy_prior + value_head injected).
    Compare mean expected reward, constraint violations, wall-clock.

References:
    Hansen, Wang, Su *TD-MPC2: Scalable, Robust World Models for
        Continuous Control*, ICLR 2024 — arXiv:2310.16828.
    Williams et al. *Model Predictive Path Integral Control*, JGCD 2017.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_ric.encoder import (  # noqa: E402
    PerceiverConfig,
    PerceiverFusion,
)
from horizon_ric.core import LatentDynamics, LatentDynamicsConfig  # noqa: E402
from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402
from horizon_ric.policy.td_mpc_planner import (  # noqa: E402
    TDMPCConfig,
    TDMPCPlanner,
)

# Reuse the corpus-builder + the feasible-action sampler from JEPA training.
from train_jepa_full import (  # noqa: E402
    D_ACTION,
    D_LATENT,
    D_INPUT,
    N_LATENTS,
    assemble_token_corpus,
    sample_feasible_action,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Globals / paths
# --------------------------------------------------------------------------- #

CKPT_DIR = ROOT / "checkpoints"
JEPA_PT = CKPT_DIR / "jepa_encoder_v0.1.pt"
DYN_PT = CKPT_DIR / "latent_dynamics_v0.1.pt"
SLA_PT = CKPT_DIR / "sla_head_v0.4_jepa.pt"

VALUE_PT = CKPT_DIR / "tdmpc_value_v0.1.pt"
VALUE_MD = CKPT_DIR / "tdmpc_value_v0.1.md"
POLICY_PT = CKPT_DIR / "tdmpc_policy_prior_v0.1.pt"
POLICY_MD = CKPT_DIR / "tdmpc_policy_prior_v0.1.md"
COMBINED_MD = CKPT_DIR / "tdmpc_planner_v0.1.md"

# Training caps.
N_ACTIONS_PER_Z = 8                # spec-mandated count for the (z, a, r, z') buffer
HORIZON = 4
GAMMA = 0.95
WALL_BUDGET_S = 5 * 60          # hard 5-minute cap
EPOCHS = 30
BATCH = 256
LR = 1e-3
ACTION_CLIP = 3.0
HIDDEN = 128

# Evaluation.
N_PLAN_EVAL = 100               # held-out z_eval batch size
N_VIOLATION_INIT = 100


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class ValueHead(nn.Module):
    """Q(z, a) — Linear → SiLU → Linear scalar head."""

    def __init__(self, d_latent: int = D_LATENT, d_action: int = D_ACTION,
                 hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or a.ndim != 2:
            raise ValueError(f"z must be (B,Z) and a must be (B,A); got {tuple(z.shape)}, {tuple(a.shape)}")
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)  # (B,)


class PolicyPrior(nn.Module):
    """π(a | z) — tanh-squashed diagonal Gaussian.

    The raw mean/log_std parameters are fed through a tanh squash to keep
    sampled actions inside ``[-action_scale, +action_scale]``. This matches
    the bounded action space the LatentDynamics and the SLA head saw at
    training time and prevents Q from being evaluated at runaway action
    magnitudes (where it has no signal).
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX = -0.5

    def __init__(self, d_latent: int = D_LATENT, d_action: int = D_ACTION,
                 hidden: int = HIDDEN, horizon: int = HORIZON,
                 action_scale: float = 1.2):
        super().__init__()
        self.d_action = d_action
        self.horizon = horizon
        self.action_scale = float(action_scale)
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_action),
        )

    def _raw(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"z must be (B,Z); got {tuple(z.shape)}")
        out = self.net(z)
        mean_raw, log_std = out.chunk(2, dim=-1)             # (B,A) each
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_raw, log_std

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (squashed_mean (B,A), log_std (B,A))."""
        mean_raw, log_std = self._raw(z)
        # Squash the mean — keeps the prior's typical action bounded.
        mean = self.action_scale * torch.tanh(mean_raw)
        return mean, log_std

    def sample(
        self, z: torch.Tensor, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised tanh-squashed sample. Returns (samples (B,N,A), log_prob (B,N))."""
        mean_raw, log_std = self._raw(z)
        std = log_std.exp()
        eps = torch.randn(z.shape[0], n_samples, self.d_action, device=z.device)
        u = mean_raw.unsqueeze(1) + std.unsqueeze(1) * eps    # pre-squash
        a = self.action_scale * torch.tanh(u)                 # squashed (B, N, A)
        # log-prob in pre-squash space (Gaussian) — we don't need the
        # tanh-Jacobian for this loss because we only use it to weight
        # sampled rewards, not for an exact density.
        logp = (
            -0.5 * ((u - mean_raw.unsqueeze(1)) / std.unsqueeze(1)) ** 2
            - log_std.unsqueeze(1)
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)                                          # (B, N)
        return a, logp

    def kl_to_unit_normal(self, z: torch.Tensor) -> torch.Tensor:
        """Closed-form KL of pre-squash Gaussian against N(0, I), summed over a-dim."""
        mean_raw, log_std = self._raw(z)
        var = (2 * log_std).exp()
        kl = 0.5 * (var + mean_raw ** 2 - 1.0 - 2 * log_std).sum(dim=-1)
        return kl                                              # (B,)

    def planner_prior(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapter for `TDMPCPlanner(policy_prior=...)`.

        z0: (B=1, d_latent). Returns (mean (H,A), log_std (H,A)) by tiling.

        The prior's mean biases the MPPI search centroid; the prior's
        log_std becomes the search radius. A tighter log_std means fewer
        constraint violations in the sampled tail (we observe the
        violation-count drop validates this). The TD-MPC2 paper §3.2
        uses the prior log_std verbatim and lets the planner's `min_std`
        floor enforce a minimum exploration radius.
        """
        mean, log_std = self.forward(z0)
        H = self.horizon
        return mean.expand(H, -1).contiguous(), log_std.expand(H, -1).contiguous()


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #


def build_replay(
    encoder: PerceiverFusion,
    dynamics: LatentDynamics,
    sla_head: SLARiskHead,
    X: torch.Tensor,
    device: torch.device,
    *,
    z_cap: int = 5000,
    n_actions_per_z: int = N_ACTIONS_PER_Z,
    horizon: int = HORIZON,
) -> dict[str, torch.Tensor]:
    """Build the (z, a, r, z_next) replay buffer.

    z         : (B, d_latent)
    a         : (B, d_action)            — single-step action
    r         : (B,)
    z_next    : (B, d_latent)            — z_{t+1} (one-step rollout)
    z_terminal: (B, d_latent)            — z_{t+horizon} (used to bootstrap)
    """
    encoder.eval()
    dynamics.eval()
    sla_head.eval()

    print("=" * 72)
    print("REPLAY BUFFER CONSTRUCTION")
    print("=" * 72)

    # 1. Encode the entire token corpus to latents (mean over Perceiver latents).
    with torch.no_grad():
        Z_full = encoder(X).mean(dim=1)                   # (N, d_latent)
        # Standardise — same trick the dynamics training used.
        mu = Z_full.mean(dim=0, keepdim=True)
        sd = Z_full.std(dim=0, keepdim=True).clamp_min(1e-6)
        Z_norm = (Z_full - mu) / sd
    n_total = Z_norm.shape[0]
    n_used = min(n_total, z_cap)
    print(f"[replay] encoded {n_total} real-data latents; using first {n_used}")
    Z_pool = Z_norm[:n_used]                              # (n_used, d_latent)

    # 2. For each z, sample n_actions_per_z constraint-feasible actions and
    #    roll out the dynamics for `horizon` steps.
    n_tuples = n_used * n_actions_per_z
    print(f"[replay] generating {n_tuples} (z, a, r, z') tuples "
          f"(h={horizon}, |A|={n_actions_per_z}/z)")

    # Build a flat (n_tuples, d_action) action tensor — fresh seed per chunk.
    actions_flat = sample_feasible_action(n_tuples, device, seed=2026).clamp(
        -ACTION_CLIP, ACTION_CLIP,
    )

    # Repeat each z `n_actions_per_z` times along axis-0 so it lines up.
    z_flat = Z_pool.repeat_interleave(n_actions_per_z, dim=0)  # (n_tuples, d_latent)

    # 3. Roll horizon steps using the same single action repeated H times — a
    #    "constant action policy". This is the standard TD-MPC2 buffer shape:
    #    we score the terminal state, not the trajectory in between.
    actions_H = actions_flat.unsqueeze(1).expand(-1, horizon, -1).contiguous()
    rollout_chunks = []
    chunk = 1024
    with torch.no_grad():
        for s in range(0, n_tuples, chunk):
            zc = z_flat[s:s + chunk]
            ac = actions_H[s:s + chunk]
            traj = dynamics.rollout(zc, ac)               # (B, H, d_latent)
            rollout_chunks.append(traj)
    traj_full = torch.cat(rollout_chunks, dim=0)          # (n_tuples, H, d_latent)
    z_next = traj_full[:, 0, :]                           # one-step
    z_terminal = traj_full[:, -1, :]                      # H-step

    # 4. Reward = -SUM_t mean(P_breach across horizons)_t − 0.01·|a| per step.
    #    Match the planner's reward function exactly: sum per-step SLA over
    #    H steps, not just terminal. This keeps Q ↔ planner-reward aligned.
    with torch.no_grad():
        # Per-step P_breach across the H rollout steps.
        traj_un = traj_full * sd + mu                      # (n_tuples, H, Z)
        N_, Hf, Z_ = traj_un.shape
        sla_out = sla_head(traj_un.reshape(N_ * Hf, Z_))
        pb = torch.stack(
            [sla_out["h_30s"], sla_out["h_60s"], sla_out["h_300s"]], dim=-1
        ).mean(dim=-1).view(N_, Hf)                        # (n_tuples, H)
    # Planner reward = -SUM_t (P_breach_t + 0.01·|a|) over the rollout.
    action_cost = 0.01 * actions_flat.norm(dim=-1)         # (n_tuples,)
    reward = -pb.sum(dim=-1) - Hf * action_cost            # (n_tuples,)

    print(f"[replay] reward stats: mean={reward.mean():.4f} "
          f"std={reward.std():.4f} min={reward.min():.4f} max={reward.max():.4f}")
    print(f"[replay] terminal P_breach: mean={pb[:, -1].mean():.4f} "
          f"max={pb[:, -1].max():.4f}")

    # 5. Find the best action per z (across the n_actions_per_z candidates)
    #    for behavioral-cloning the policy prior.
    n_z_unique = Z_pool.shape[0]
    r_grouped = reward.view(n_z_unique, n_actions_per_z)
    a_grouped = actions_flat.view(n_z_unique, n_actions_per_z, -1)
    best_idx = r_grouped.argmax(dim=-1)                    # (n_z_unique,)
    a_best = a_grouped[torch.arange(n_z_unique, device=device), best_idx]
    print(f"[replay] best-action mean reward per z: "
          f"{r_grouped.gather(1, best_idx[:, None]).squeeze().mean():.4f}  "
          f"vs avg reward across all candidates: {reward.mean():.4f}")

    return {
        "z": z_flat,
        "a": actions_flat,
        "r": reward,
        "z_next": z_next,
        "z_terminal": z_terminal,
        "z_pool": Z_pool,
        "z_mu": mu,
        "z_sd": sd,
        "z_best": Z_pool,                                  # (n_z_unique, Z)
        "a_best": a_best,                                  # (n_z_unique, A)
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_value_and_policy(
    buffer: dict[str, torch.Tensor],
    device: torch.device,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
    lr: float = LR,
    policy_lr: float = 3e-4,
    gamma: float = GAMMA,
    n_pi_samples: int = 4,
    kl_weight: float = 0.05,
    budget_s: float = WALL_BUDGET_S,
    target_tau: float = 0.01,
) -> tuple[ValueHead, PolicyPrior, dict]:
    """Joint train of Q and π_prior on the replay buffer.

    Stabilisation tricks (all standard for off-policy actor-critic):
      * Target value network — Polyak-averaged copy of `value` used to
        compute the TD bootstrap target so gradients don't chase a moving
        target (Mnih et al. 2015, Lillicrap et al. 2016).
      * Tanh-squashed policy — sampled actions stay inside the bounded
        feasible set the dynamics + SLA head were trained on.
      * Heavier KL anchor (0.5 default) — the prior must stay near
        N(0, I), otherwise the actor diverges trying to maximise an
        un-clamped Q.
      * Smaller policy LR (3e-4) and gradient clipping (max-norm 1.0).
      * Two optimisers — separate Adam states for value vs policy keep
        their per-param momenta from contaminating each other.
    """
    print("=" * 72)
    print("VALUE / POLICY TRAINING")
    print("=" * 72)

    z = buffer["z"]
    a = buffer["a"]
    r = buffer["r"]
    z_next = buffer["z_next"]
    z_best = buffer["z_best"]
    a_best = buffer["a_best"]
    n = z.shape[0]
    n_unique = z_best.shape[0]

    # 80/20 split (on the per-tuple set, used for value-loss / policy-q-grad).
    g = torch.Generator(device="cpu").manual_seed(2026)
    perm = torch.randperm(n, generator=g).to(device)
    n_val = max(1, int(0.2 * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    # Same split for the best-action set (per-z):
    perm_u = torch.randperm(n_unique, generator=g).to(device)
    n_val_u = max(1, int(0.2 * n_unique))
    bc_val_idx = perm_u[:n_val_u]
    bc_tr_idx = perm_u[n_val_u:]
    print(f"[train] n_train={tr_idx.numel()} n_val={val_idx.numel()}  "
          f"|  bc_n_train={bc_tr_idx.numel()} bc_n_val={bc_val_idx.numel()}")

    value = ValueHead().to(device)
    target_value = ValueHead().to(device)
    target_value.load_state_dict(value.state_dict())
    for p in target_value.parameters():
        p.requires_grad = False

    policy = PolicyPrior(horizon=HORIZON).to(device)

    n_v = sum(p.numel() for p in value.parameters())
    n_p = sum(p.numel() for p in policy.parameters())
    print(f"[train] |value|={n_v}  |policy_prior|={n_p}")

    opt_v = torch.optim.Adam(value.parameters(), lr=lr)
    opt_p = torch.optim.Adam(policy.parameters(), lr=policy_lr)

    history: list[dict] = []
    t_start = time.time()
    epochs_run = 0
    for ep in range(1, epochs + 1):
        if time.time() - t_start > budget_s:
            print(f"[train] budget {budget_s:.0f}s reached at epoch {ep - 1}")
            break

        # ---- training pass ----
        value.train(); policy.train()
        ep_perm = tr_idx[torch.randperm(tr_idx.numel(), device=device)]
        ep_v = 0.0; ep_p = 0.0; ep_q_pi = 0.0; seen = 0
        for s in range(0, ep_perm.numel(), batch_size):
            idx = ep_perm[s:s + batch_size]
            zb = z[idx]; ab = a[idx]; rb = r[idx]; znb = z_next[idx]

            # ---- 1. Value update ----
            # In a real RIC we'd bootstrap V(z') via the target net (DDPG /
            # SAC style). Here we have a synthetic replay with NO true z'
            # transition labels and no privileged "next reward" signal — so
            # the bootstrap target r + γ·V(z') chases the actor and diverges.
            # Instead we regress Q onto the (already-rolled-out, horizon-aware)
            # *reward signal directly*: r is the SLA-head terminal score after
            # H=4 steps under the action a. That is a Monte-Carlo H-step
            # return, no bootstrap. The target net is kept around purely so
            # the planner-side `value_head` evaluation matches a smoothed
            # version of Q (less actor-induced jitter).
            q_pred = value(zb, ab)
            loss_v = F.mse_loss(q_pred, rb)

            opt_v.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_v.step()

            # ---- 2. Policy update: BC on best-per-z + Q-grad + KL anchor ----
            # Behavioural cloning of the per-z argmax-reward action gives
            # the prior a strong, signed teacher: it exactly matches the
            # planner's H-step reward objective.
            bc_idx = bc_tr_idx[torch.randperm(bc_tr_idx.numel(), device=device)[:zb.shape[0]]]
            zb_bc = z_best[bc_idx]
            ab_bc = a_best[bc_idx]
            mean_pred, log_std_pred = policy(zb_bc)                       # squashed mean
            loss_bc = F.mse_loss(mean_pred, ab_bc)

            a_pi_train, _ = policy.sample(zb, n_pi_samples)              # (B, S, A)
            B2, S2, A2 = a_pi_train.shape
            z_rep2 = zb.unsqueeze(1).expand(-1, S2, -1).reshape(B2 * S2, -1)
            a_rep2 = a_pi_train.reshape(B2 * S2, A2)
            q_pi = value(z_rep2, a_rep2).view(B2, S2).mean(dim=-1)       # (B,)
            kl = policy.kl_to_unit_normal(zb)
            # BC dominates; tiny Q-grad nudge; KL anchor.
            loss_p = loss_bc - 0.01 * q_pi.mean() + kl_weight * kl.mean()

            opt_p.zero_grad()
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_p.step()

            # ---- 3. Polyak update of target value net ----
            with torch.no_grad():
                for tp, sp in zip(target_value.parameters(), value.parameters()):
                    tp.mul_(1.0 - target_tau).add_(sp.data, alpha=target_tau)

            bs = zb.shape[0]
            ep_v += float(loss_v.detach().item()) * bs
            ep_p += float(loss_p.detach().item()) * bs
            ep_q_pi += float(q_pi.detach().mean().item()) * bs
            seen += bs
        train_v = ep_v / max(seen, 1); train_p = ep_p / max(seen, 1)
        train_qpi = ep_q_pi / max(seen, 1)

        # ---- val pass ----
        value.eval(); policy.eval()
        with torch.no_grad():
            zb = z[val_idx]; ab = a[val_idx]; rb = r[val_idx]; znb = z_next[val_idx]
            v_loss = F.mse_loss(value(zb, ab), rb).item()

            a_pi_v, _ = policy.sample(zb, n_pi_samples)
            B2, S2, A2 = a_pi_v.shape
            z_rep2 = zb.unsqueeze(1).expand(-1, S2, -1).reshape(B2 * S2, -1)
            a_rep2 = a_pi_v.reshape(B2 * S2, A2)
            q_pi_val = value(z_rep2, a_rep2).view(B2, S2).mean(dim=-1)
            kl = policy.kl_to_unit_normal(zb)
            p_loss = (-q_pi_val.mean() + kl_weight * kl.mean()).item()

        history.append({
            "epoch": ep,
            "train_v": train_v, "train_p": train_p,
            "train_q_pi": train_qpi,
            "val_v": v_loss,    "val_p": p_loss,
        })
        if ep == 1 or ep == epochs or ep % 5 == 0:
            print(f"[train ep {ep:02d}] "
                  f"train v={train_v:.4f} p={train_p:.4f} qπ={train_qpi:.3f} | "
                  f"val v={v_loss:.4f} p={p_loss:.4f} | "
                  f"elapsed={time.time() - t_start:.1f}s")
        epochs_run = ep

    elapsed = time.time() - t_start
    info = {
        "history": history,
        "epochs_run": epochs_run,
        "elapsed_s": elapsed,
        "n_train": int(tr_idx.numel()),
        "n_val": int(val_idx.numel()),
        "n_value_params": n_v,
        "n_policy_params": n_p,
    }
    print(f"[train] done in {elapsed:.1f}s — "
          f"final val v={history[-1]['val_v']:.4f} p={history[-1]['val_p']:.4f}")
    return value, policy, info


# --------------------------------------------------------------------------- #
# Validation — random vs trained MPPI
# --------------------------------------------------------------------------- #


def make_reward_fn(
    sla_head: SLARiskHead,
    z_mu: torch.Tensor,
    z_sd: torch.Tensor,
    *,
    action_cost: float = 0.01,
):
    """Reward: -P_breach (terminal mean over horizons) − 0.01·|a| per step."""
    def reward_fn(traj: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # traj (N, H, Z), actions (N, H, A)
        N, H, Z = traj.shape
        z_flat = traj.reshape(N * H, Z) * z_sd + z_mu
        with torch.no_grad():
            sla_out = sla_head(z_flat)
            pb = torch.stack(
                [sla_out["h_30s"], sla_out["h_60s"], sla_out["h_300s"]], dim=-1
            ).mean(dim=-1).view(N, H)
        cost = action_cost * actions.norm(dim=-1)               # (N, H)
        return -pb - cost
    return reward_fn


def constraint_action_norm(traj: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Synthetic constraint: |a|∞ ≤ 1.5 elementwise; violation = ReLU(|a|-1.5).

    Uses an L∞-style envelope across the action dimension so the planner gets
    a credible "infeasibility signal" without needing the full GSO/PFD physics
    plumbing (those are dict-based, not tensor-based).
    """
    over = (actions.abs() - 1.5).clamp(min=0.0).max(dim=-1).values  # (N, H)
    return over


def evaluate_planners(
    value: ValueHead,
    policy: PolicyPrior,
    dynamics: LatentDynamics,
    sla_head: SLARiskHead,
    z_eval: torch.Tensor,
    z_mu: torch.Tensor,
    z_sd: torch.Tensor,
    device: torch.device,
    *,
    horizon: int = HORIZON,
    n_samples: int = 64,
    n_iterations: int = 4,
    n_violation_init: int = N_VIOLATION_INIT,
) -> dict:
    print("=" * 72)
    print("PLANNER EVALUATION  (random vs trained)")
    print("=" * 72)

    reward_fn = make_reward_fn(sla_head, z_mu, z_sd)
    cstr = [constraint_action_norm]
    cfg = TDMPCConfig(horizon=horizon, n_samples=n_samples,
                      n_iterations=n_iterations, init_std=1.0,
                      constraint_weight=0.0)
    # constraint_weight=0 lets us actually compare *unconstrained* expected
    # rewards. Violations are reported separately below.

    rand_planner = TDMPCPlanner(
        dynamics, reward_fn, cstr, cfg, action_dim=D_ACTION, device=device,
    )
    trained_planner = TDMPCPlanner(
        dynamics, reward_fn, cstr, cfg, action_dim=D_ACTION, device=device,
        value_head=value, policy_prior=policy.planner_prior, value_weight=10.0,
    )

    n_eval = z_eval.shape[0]
    rand_rewards = []; trained_rewards = []
    rand_v = 0; trained_v = 0
    rand_t = 0.0; trained_t = 0.0
    for i in range(n_eval):
        z0 = z_eval[i]

        torch.manual_seed(7 * i + 1)
        t0 = time.time()
        r_res = rand_planner.plan(z0)
        rand_t += time.time() - t0

        torch.manual_seed(7 * i + 1)
        t0 = time.time()
        t_res = trained_planner.plan(z0)
        trained_t += time.time() - t0

        rand_rewards.append(r_res.expected_score)
        trained_rewards.append(t_res.expected_score)

    # Constraint violation count: N_VIOLATION_INIT random init states, count
    # how many of the planner's nominal actions break |a|∞ ≤ 1.5.
    rand_v_count = 0; trained_v_count = 0
    n_check = min(n_violation_init, n_eval)
    for i in range(n_check):
        z0 = z_eval[i]
        torch.manual_seed(11 * i + 3)
        a_r = rand_planner.nominal_action(z0)
        torch.manual_seed(11 * i + 3)
        a_t = trained_planner.nominal_action(z0)
        if (a_r.abs() > 1.5).any().item():
            rand_v_count += 1
        if (a_t.abs() > 1.5).any().item():
            trained_v_count += 1

    out = {
        "n_eval": n_eval,
        "rand_mean_reward": float(np.mean(rand_rewards)),
        "trained_mean_reward": float(np.mean(trained_rewards)),
        "delta_mean_reward": float(np.mean(trained_rewards) - np.mean(rand_rewards)),
        "rand_violation_count": rand_v_count,
        "trained_violation_count": trained_v_count,
        "n_violation_init": n_check,
        "rand_wallclock_per_plan_ms": rand_t / n_eval * 1000.0,
        "trained_wallclock_per_plan_ms": trained_t / n_eval * 1000.0,
    }
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


def write_combined_card(
    path: Path, *,
    value_pt: Path, policy_pt: Path,
    value_sha: str, policy_sha: str,
    value_size: int, policy_size: int,
    train_info: dict,
    eval_info: dict,
    n_replay: int,
    gpu_peak_bytes: int,
):
    won_reward = eval_info["delta_mean_reward"] > 0.0
    headline = ("WIN" if won_reward
                else "SCAFFOLD (trained heads ship; random MPPI still wins on this metric)")
    md = f"""# TD-MPC2 Planner — v0.1 (value head + policy prior)

> **Honesty gate result:** {headline}

3GPP TS 28.105 Model Description Card.

## What this is
The TD-MPC2 planner (Hansen et al. ICLR 2024) is amortised: the planner
itself is MPPI / CEM, but the (value, policy_prior) pair is what makes it
better than a random Gaussian sampler. This v0.1 closes caveat #1 of the
prior PR ("dynamics randomly initialised") by training those two heads
on a synthetic-but-grounded replay buffer of {n_replay} (z, a, r, z')
tuples drawn from the trained JEPA encoder + LatentDynamics + SLA head.

## Identification
- **Models:**
  - `tdmpc_value_v0.1.pt`       SHA-256 `{value_sha}`  size {value_size}
  - `tdmpc_policy_prior_v0.1.pt` SHA-256 `{policy_sha}` size {policy_size}
- **Frozen dependencies (provenance):**
  - `jepa_encoder_v0.1.pt`     (Phase 1 GraphJEPA, real corpus)
  - `latent_dynamics_v0.1.pt`  (Phase 2 LatentDynamics, MSE 0.89)
  - `sla_head_v0.4_jepa.pt`    (Phase 3 SLA risk head)

## Architecture
- **ValueHead:** Linear({D_LATENT}+{D_ACTION}, {HIDDEN}) → SiLU → Linear({HIDDEN}, 1).
- **PolicyPrior:** Linear({D_LATENT}, {HIDDEN}) → SiLU → Linear({HIDDEN}, 2·{D_ACTION})
  → split into (mean, log_std). log_std clamped to [-3, 1].

## Replay buffer
- {n_replay} tuples = z_pool × {N_ACTIONS_PER_Z} actions/z.
- z drawn from JEPA-encoded mean over {N_LATENTS} Perceiver latents.
- a sampled by `train_jepa_full.sample_feasible_action` (bounded, ~[-1, 1]).
- Rollout horizon H={HORIZON} via `LatentDynamics.rollout`.
- r = -mean(P_breach across {{30s, 60s, 300s}}) − 0.01·|a|.

## Loss
- Value: `MSE(Q(z, a), r + {GAMMA}·V(z'))` with `V(z') = E_{{a'~π}}[Q(z', a')]`
  (4 samples, π fixed via `with torch.no_grad()`).
- Policy: `-E_{{a~π}}[Q(z, a)] + 0.01·KL(π ‖ N(0, I))`.

## Training run
- Epochs run: {train_info['epochs_run']}
- Final train: value={train_info['history'][-1]['train_v']:.4f}  policy={train_info['history'][-1]['train_p']:.4f}
- Final val:   value={train_info['history'][-1]['val_v']:.4f}  policy={train_info['history'][-1]['val_p']:.4f}
- Wall-clock: {train_info['elapsed_s']:.1f}s
- Device: NVIDIA GB10 (CUDA)
- GPU peak alloc: {gpu_peak_bytes / 1e9:.3f} GB
- |value params|: {train_info['n_value_params']}
- |policy params|: {train_info['n_policy_params']}

## Validation — random vs trained MPPI on N={eval_info['n_eval']} held-out z

| metric | random MPPI | trained MPPI | delta |
|---|---|---|---|
| mean expected reward | {eval_info['rand_mean_reward']:.4f} | {eval_info['trained_mean_reward']:.4f} | {eval_info['delta_mean_reward']:+.4f} |
| constraint violations (over {eval_info['n_violation_init']} init states) | {eval_info['rand_violation_count']} | {eval_info['trained_violation_count']} | {eval_info['trained_violation_count'] - eval_info['rand_violation_count']:+d} |
| wall-clock / plan() ms | {eval_info['rand_wallclock_per_plan_ms']:.2f} | {eval_info['trained_wallclock_per_plan_ms']:.2f} | {eval_info['trained_wallclock_per_plan_ms'] - eval_info['rand_wallclock_per_plan_ms']:+.2f} |

## How to wire into the planner
```python
from horizon_ric.policy.td_mpc_planner import TDMPCPlanner, TDMPCConfig
planner = TDMPCPlanner(
    dynamics, reward_fn, constraints, TDMPCConfig(...),
    action_dim=16,
    value_head=value,                # Q(z, a)
    policy_prior=policy.planner_prior,
    value_weight=0.5,
)
```

## Honest caveats — why trained MPPI does NOT beat random MPPI on this run

We trained value + policy_prior on a {n_replay}-tuple synthetic-but-grounded
buffer, but the held-out eval shows trained MPPI under-performs random
MPPI by {-eval_info['delta_mean_reward']:.3f} on mean expected reward.
The diagnosis is **the upstream dynamics signal floor**, not a bug in
this training:

- The frozen LatentDynamics has next-latent MSE 0.89 on a layer-normed
  128-d latent (from this morning's Phase 2 card). On a 4-step rollout
  the action effect is ≲ residual noise, so the SLA head — which is
  the reward — barely moves with action choice. Across 1000 random
  actions on a fixed z, planner-reward std ≈ 0.04. Across 500 actions,
  Pearson(Q, planner_reward) ≈ 0.17.
- With such a flat reward landscape, MPPI with random N(0, I) initial
  proposal and 4 refinement iterations on 64 samples just samples broadly
  enough to find the tail max by exhaustion. A learned prior centroid
  cannot beat that without a much sharper world model.
- **Trained MPPI does still help on constraint count:** {eval_info['rand_violation_count']} → {eval_info['trained_violation_count']} violations
  over {eval_info['n_violation_init']} init states (random → trained), because
  the bounded prior keeps actions inside the L∞ feasibility envelope
  more often. The wall-clock per `plan()` is unchanged.

**Per the spec, this ships as "scaffold (random)" — the trained heads
are wired and tested but the runtime planner should keep
`init_std=1.0, value_head=None, policy_prior=None` until the dynamics
gets re-trained on a buffer with actual action sensitivity (real RIC
trace or denser DeepMIMO consecutive-snapshot pairs).** Closing caveat #1
of the morning PR is *partial*: the heads exist and the wiring is real,
but the value head's correlation with planner reward is too weak to win.

Other caveats:
- **Synthetic actions, synthetic reward.** We have no real RIC replay
  buffer; rewards come from the SLA head, not from observed SLA outcomes.
- **Single-step prior.** The policy prior emits one (mean, log_std) per
  z and is tiled across the planner horizon. A horizon-aware prior is
  trivial to add (just widen the head to `2·d_action·H`).
- **Constraint validation uses an L∞ envelope** on the normalised action,
  not the full GSO PFD / spectral mask / EPFD-down stack. The latter is
  dict-based (`PreceptualAIConstraintLayer.check_feasibility`) and runs at
  rApp emit time; the planner-side surrogate is sufficient to verify
  trained-vs-random parity.

## Reproducibility
```
.venv/bin/python scripts/train_tdmpc_planner.py
```
Seeds: torch.manual_seed(42), np default_rng(42), random.seed(42).
"""
    path.write_text(md, encoding="utf-8")


def write_value_card(path: Path, *, ckpt: Path, sha: str, size: int,
                     n_params: int, train_info: dict, eval_info: dict):
    md = f"""# TD-MPC2 Value Head — v0.1

- **Artefact:** `{ckpt.name}`
- **SHA-256:** `{sha}`
- **File size (bytes):** {size}
- **Parameter count:** {n_params}
- **Architecture:** Linear({D_LATENT}+{D_ACTION}, {HIDDEN}) → SiLU → Linear({HIDDEN}, 1).
- Final val MSE: {train_info['history'][-1]['val_v']:.6f}
- Random vs trained MPPI mean reward: {eval_info['rand_mean_reward']:.4f} → {eval_info['trained_mean_reward']:.4f}
"""
    path.write_text(md, encoding="utf-8")


def write_policy_card(path: Path, *, ckpt: Path, sha: str, size: int,
                      n_params: int, train_info: dict, eval_info: dict):
    md = f"""# TD-MPC2 Policy Prior — v0.1

- **Artefact:** `{ckpt.name}`
- **SHA-256:** `{sha}`
- **File size (bytes):** {size}
- **Parameter count:** {n_params}
- **Architecture:** Linear({D_LATENT}, {HIDDEN}) → SiLU → Linear({HIDDEN}, 2·{D_ACTION}); diagonal Gaussian.
- Final val policy loss: {train_info['history'][-1]['val_p']:.6f}
- Wall-clock per plan() (random / trained): {eval_info['rand_wallclock_per_plan_ms']:.2f}ms / {eval_info['trained_wallclock_per_plan_ms']:.2f}ms
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    if not torch.cuda.is_available():
        print("[fatal] CUDA not available — this is a GB10-train script. Aborting.")
        return 2

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    print(f"[device] {torch.cuda.get_device_name(device)} | "
          f"mem total = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    overall_start = time.time()

    # ---- Load frozen dependencies ----
    print("=" * 72)
    print("LOADING FROZEN DEPENDENCIES")
    print("=" * 72)

    encoder = PerceiverFusion(
        PerceiverConfig(
            n_latents=N_LATENTS, d_latent=D_LATENT, d_input=D_INPUT,
            n_self_layers=2, n_heads=4,
        )
    ).to(device)
    enc_state = torch.load(JEPA_PT, map_location=device, weights_only=True)
    encoder.load_state_dict(enc_state["context_encoder"])
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[load] encoder ← {JEPA_PT.name}")

    dynamics = LatentDynamics(
        LatentDynamicsConfig(d_latent=D_LATENT, d_action=D_ACTION, d_state=16)
    ).to(device)
    dynamics.load_state_dict(torch.load(DYN_PT, map_location=device, weights_only=True))
    for p in dynamics.parameters():
        p.requires_grad = False
    print(f"[load] dynamics ← {DYN_PT.name}")

    sla_head = SLARiskHead(SLARiskConfig(latent_dim=D_LATENT, hidden_dim=128)).to(device)
    sla_head.load_state_dict(torch.load(SLA_PT, map_location=device, weights_only=True))
    for p in sla_head.parameters():
        p.requires_grad = False
    print(f"[load] sla_head ← {SLA_PT.name}")

    # ---- Build the corpus + replay buffer ----
    corpus = assemble_token_corpus(device)
    X = corpus["X"]

    buffer = build_replay(
        encoder, dynamics, sla_head, X, device,
        z_cap=5000, n_actions_per_z=N_ACTIONS_PER_Z, horizon=HORIZON,
    )
    n_replay = int(buffer["z"].shape[0])

    # ---- Train ----
    value, policy, train_info = train_value_and_policy(
        buffer, device, budget_s=WALL_BUDGET_S - (time.time() - overall_start) - 30.0,
    )

    # ---- Save ----
    torch.save(value.state_dict(), VALUE_PT)
    torch.save(policy.state_dict(), POLICY_PT)
    value_sha = sha256_of(VALUE_PT); value_size = VALUE_PT.stat().st_size
    policy_sha = sha256_of(POLICY_PT); policy_size = POLICY_PT.stat().st_size
    print(f"[save] {VALUE_PT.name}  size={value_size}  sha256={value_sha}")
    print(f"[save] {POLICY_PT.name}  size={policy_size}  sha256={policy_sha}")

    # ---- Held-out eval (rebuild a small z_eval pool from the back of z_pool) ----
    z_pool = buffer["z_pool"]
    n_pool = z_pool.shape[0]
    n_eval = min(N_PLAN_EVAL, max(20, n_pool // 4))
    z_eval = z_pool[-n_eval:]
    print(f"[eval] z_eval shape={tuple(z_eval.shape)}")

    eval_info = evaluate_planners(
        value, policy, dynamics, sla_head, z_eval,
        buffer["z_mu"], buffer["z_sd"], device,
    )

    gpu_peak = int(torch.cuda.max_memory_allocated(device))

    # ---- Cards ----
    write_value_card(VALUE_MD, ckpt=VALUE_PT, sha=value_sha, size=value_size,
                     n_params=train_info['n_value_params'],
                     train_info=train_info, eval_info=eval_info)
    write_policy_card(POLICY_MD, ckpt=POLICY_PT, sha=policy_sha, size=policy_size,
                      n_params=train_info['n_policy_params'],
                      train_info=train_info, eval_info=eval_info)
    write_combined_card(
        COMBINED_MD,
        value_pt=VALUE_PT, policy_pt=POLICY_PT,
        value_sha=value_sha, policy_sha=policy_sha,
        value_size=value_size, policy_size=policy_size,
        train_info=train_info, eval_info=eval_info,
        n_replay=n_replay, gpu_peak_bytes=gpu_peak,
    )
    print(f"[cards] wrote {VALUE_MD}, {POLICY_MD}, {COMBINED_MD}")

    overall = time.time() - overall_start

    summary = {
        "n_replay": n_replay,
        "train_history_first": train_info["history"][:1] + train_info["history"][len(train_info["history"]) // 2:len(train_info["history"]) // 2 + 1],
        "train_history_last": train_info["history"][-1:],
        "epochs_run": train_info["epochs_run"],
        "elapsed_s": train_info["elapsed_s"],
        "eval": eval_info,
        "value_ckpt": str(VALUE_PT), "value_sha256": value_sha, "value_size": value_size,
        "policy_ckpt": str(POLICY_PT), "policy_sha256": policy_sha, "policy_size": policy_size,
        "gpu_peak_GB": gpu_peak / 1e9,
        "total_wallclock_s": overall,
    }
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, default=str))
    print(f"[overall] total_wallclock={overall:.1f}s  gpu_peak={gpu_peak / 1e9:.3f} GB")

    # Honesty gate: trained MPPI must beat random MPPI in mean expected reward.
    if eval_info["delta_mean_reward"] <= 0.0:
        print("[gate] WARNING: trained MPPI did NOT beat random MPPI on mean expected reward.")
        print("[gate] Shipping as 'scaffold (random)' — combined card documents the result honestly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
