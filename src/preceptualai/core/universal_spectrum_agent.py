"""
Universal Spectrum Agent — the UHCI Policy.

Implements a hierarchical SAC agent that manages ALL connectivity providers
(LEO, MEO, GEO, HAPS, 5G FR1/FR3, 6G ISAC, Wi-Fi 7) simultaneously.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │              Universal Spectrum Agent                    │
  │                                                          │
  │  Obs (B, T, P, F)                                        │
  │       │                                                  │
  │       ▼                                                  │
  │  HeteroGNNTemporalEncoder ──► z (B, latent_dim)          │
  │       │                                                  │
  │       ├──► KANActor ──► provider+channel probs           │
  │       │    (interpretable, regulatory-compliant)         │
  │       │                                                  │
  │       └──► Twin LTC Critics ──► Q(s,a) for each action   │
  │                                                          │
  │  Training: Discrete SAC (Christodoulou 2019)             │
  │            + automatic entropy tuning                    │
  │            + Polyak target critics                       │
  │            + Dyna world-model optional rollouts          │
  └─────────────────────────────────────────────────────────┘

The KANActor is used by default for the universal agent because:
  - Regulatory compliance: network operators can inspect learned allocation rules
  - Interpretability: show WHY the agent prefers LEO over FR3 in a given state
  - Efficiency: KAN layers generalize better with fewer parameters across the
    high-dimensional heterogeneous observation space

References:
  Christodoulou, P. "Soft Actor-Critic for Discrete Action Spaces." 2019.
  Liu et al. "KAN: Kolmogorov-Arnold Networks." ICLR 2025.
  Wang et al. "Heterogeneous Graph Attention Network." WWW 2019.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preceptualai.env.provider_registry import (
    PROVIDER_FEATURE_DIM,
    ProviderType,
    coexistence_adjacency,
    get_all_provider_types,
)
from preceptualai.core.hetero_gnn_encoder import HeteroGNNTemporalEncoder
from preceptualai.core.kan_actor import KANActor
from preceptualai.core.replay_buffer_gpu import ReplayBufferGPU


@dataclass
class UniversalAgentConfig:
    """Configuration for the Universal Spectrum Agent."""
    # Encoder
    provider_types:    List[ProviderType] = field(
        default_factory=lambda: list(ProviderType)
    )
    raw_feature_dim:   int   = 0        # auto-set from env if 0
    obs_history_len:   int   = 16
    gnn_latent_dim:    int   = 128
    gnn_output_dim:    int   = 256
    temporal_hidden:   int   = 128
    latent_dim:        int   = 256
    num_gnn_layers:    int   = 2
    temporal_layers:   int   = 2
    temporal_backend:  str   = "cfc"   # "cfc", "ltc", or "mamba"
    use_cfc:           Optional[bool] = None  # deprecated, use temporal_backend

    # KAN actor
    kan_hidden_dim:    int   = 128
    kan_grid_size:     int   = 5
    kan_spline_order:  int   = 3

    # SAC
    gamma:             float = 0.99
    tau_polyak:        float = 0.005
    lr_actor:          float = 3e-4
    lr_critic:         float = 3e-4
    lr_alpha:          float = 3e-4
    target_entropy_ratio: float = 0.98

    # Replay buffer
    buffer_capacity:   int   = 500_000
    batch_size:        int   = 256
    use_per:           bool  = False  # Prioritized Experience Replay
    per_alpha:         float = 0.6
    per_beta_start:    float = 0.4
    per_beta_frames:   int   = 100_000

    # SmODE power control (3GPP-compliant)
    use_smooth_power:        bool  = False
    power_control_dim:       int   = 1
    max_power_ramp_rate:     float = 0.1

    # Dyna world model
    use_world_model:         bool  = False
    wm_hidden_dim:           int   = 128
    wm_latent_dim:           int   = 64
    wm_imagination_horizon:  int   = 5
    wm_imagined_batch_size:  int   = 128
    wm_train_every:          int   = 5
    wm_lr:                   float = 1e-3

    # Device
    device:            str   = "cpu"


class UniversalCritic(nn.Module):
    """
    Twin Q-network for the universal connectivity agent.

    Uses a shared encoder (HeteroGNNTemporalEncoder) with two separate
    linear heads, producing Q(s, a) for all actions simultaneously.
    """

    def __init__(self, latent_dim: int, num_actions: int):
        super().__init__()
        self.q1_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim) encoder output
        Returns:
            q1, q2: (B, num_actions) Q-values for all actions
        """
        return self.q1_head(z), self.q2_head(z)


class UniversalSpectrumAgent:
    """
    The UHCI agent: manages all connectivity provider types jointly.

    Implements discrete SAC with:
      - HeteroGNNTemporalEncoder for spatio-temporal graph observations
      - KANActor for interpretable, auditable policy
      - Twin critics with Polyak target updates
      - Automatic entropy temperature tuning
      - GPU-resident replay buffer

    The agent exposes a clean interface compatible with the training
    infrastructure (train.py, benchmarks/), with an additional method
    `explain_action()` that uses KAN spline stats for interpretability.
    """

    def __init__(self, cfg: UniversalAgentConfig, num_actions: int):
        self.cfg        = cfg
        self.num_actions = num_actions
        self.device     = torch.device(cfg.device)

        # Build static coexistence adjacency for selected providers
        self._adj = coexistence_adjacency(self.device)
        all_types = get_all_provider_types()
        idx_map   = {t: i for i, t in enumerate(all_types)}
        sel_idx   = torch.tensor(
            [idx_map[pt] for pt in cfg.provider_types], device=self.device
        )
        self.adj  = self._adj[sel_idx][:, sel_idx]  # (P, P)

        # ── Encoder ───────────────────────────────────────────────
        self.encoder = HeteroGNNTemporalEncoder(
            provider_types  = cfg.provider_types,
            raw_feature_dim = cfg.raw_feature_dim if cfg.raw_feature_dim > 0 else PROVIDER_FEATURE_DIM,
            gnn_latent_dim  = cfg.gnn_latent_dim,
            gnn_output_dim  = cfg.gnn_output_dim,
            temporal_hidden = cfg.temporal_hidden,
            output_dim      = cfg.latent_dim,
            num_gnn_layers  = cfg.num_gnn_layers,
            temporal_layers = cfg.temporal_layers,
            temporal_backend = cfg.temporal_backend,
            use_cfc         = cfg.use_cfc,
        ).to(self.device)

        # ── Actor (KAN for interpretability) ──────────────────────
        self.actor = KANActor(
            encoder_latent_dim = cfg.latent_dim,
            num_actions        = num_actions,
            kan_hidden_dim     = cfg.kan_hidden_dim,
            grid_size          = cfg.kan_grid_size,
            spline_order       = cfg.kan_spline_order,
        ).to(self.device)

        # ── SmODE power control (optional, 3GPP-compliant) ────────
        self.smooth_power = None
        self._prev_power  = None
        if cfg.use_smooth_power:
            from preceptualai.core.smooth_actor import SmODENeuron
            self.smooth_power = SmODENeuron(
                input_dim  = cfg.latent_dim,
                output_dim = cfg.power_control_dim,
                max_rate   = cfg.max_power_ramp_rate,
            ).to(self.device)
            self._prev_power = torch.zeros(1, cfg.power_control_dim, device=self.device)

        # ── Critics (twin) ────────────────────────────────────────
        self.critic        = UniversalCritic(cfg.latent_dim, num_actions).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Entropy temperature ───────────────────────────────────
        target_entropy = -cfg.target_entropy_ratio * torch.log(
            torch.tensor(1.0 / num_actions)
        )
        self.target_entropy = target_entropy.item()
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))

        # ── Optimisers ────────────────────────────────────────────
        encoder_actor_params = (
            list(self.encoder.parameters()) + list(self.actor.parameters())
        )
        self.opt_actor  = optim.Adam(encoder_actor_params, lr=cfg.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.opt_alpha  = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        # ── Replay buffer ─────────────────────────────────────────
        _obs_dim = cfg.raw_feature_dim if cfg.raw_feature_dim > 0 else 256
        self._use_per = cfg.use_per
        if cfg.use_per:
            from preceptualai.core.prioritized_replay import PrioritizedReplayBuffer
            self.replay = PrioritizedReplayBuffer(
                capacity    = cfg.buffer_capacity,
                obs_dim     = _obs_dim,
                alpha       = cfg.per_alpha,
                beta_start  = cfg.per_beta_start,
                beta_frames = cfg.per_beta_frames,
                device      = self.device,
            )
            self._gpu_replay = False
        elif self.device.type == "cuda":
            self.replay = ReplayBufferGPU(
                capacity    = cfg.buffer_capacity,
                state_shape = (_obs_dim,),
                device      = self.device,
            )
            self._gpu_replay = True
        else:
            from preceptualai.core.replay_buffer import ReplayBuffer
            self.replay = ReplayBuffer(
                capacity    = cfg.buffer_capacity,
                state_shape = (_obs_dim,),
                device      = self.device,
            )
            self._gpu_replay = False

        # ── Dyna World Model (optional) ──────────────────────────
        self.world_model = None
        self.wm_trainer  = None
        if cfg.use_world_model:
            from preceptualai.core.world_model import LTCWorldModel, DynaWorldModelTrainer
            self.world_model = LTCWorldModel(
                obs_dim     = _obs_dim,
                num_actions = num_actions,
                hidden_dim  = cfg.wm_hidden_dim,
                latent_dim  = cfg.wm_latent_dim,
            ).to(self.device)
            self.wm_trainer = DynaWorldModelTrainer(
                world_model          = self.world_model,
                lr                   = cfg.wm_lr,
                imagination_horizon  = cfg.wm_imagination_horizon,
                imagination_batch_size = cfg.wm_imagined_batch_size,
            )

        self._updates = 0

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs_flat: torch.Tensor,          # (B, obs_dim) flat observation
        history:  Optional[torch.Tensor] = None,  # (B, T, obs_dim) if available
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select actions for a batch of environments.

        Args:
            obs_flat: (B, obs_dim) current flat observation
            history:  optional (B, T, obs_dim) history; uses last timestep if None
            deterministic: if True, argmax (greedy); else stochastic sample
        Returns:
            actions: (B,) integer action indices
        """
        B = obs_flat.shape[0]

        # Build graph-structured input for encoder
        # For the universal agent with flat obs, we use the flat obs directly
        # as the temporal sequence input (simplified path)
        if history is not None:
            # Full temporal path — reshape each timestep obs → graph snapshot
            z = self._encode_history(history)
        else:
            # Single-step path — tile obs as trivial sequence
            seq = obs_flat.unsqueeze(1).expand(B, 1, -1)  # (B, 1, obs_dim)
            z   = self._encode_flat_sequence(seq)

        probs = self.actor(z)   # (B, num_actions)

        if deterministic:
            discrete_actions = probs.argmax(dim=-1)
        else:
            discrete_actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # SmODE power control (3GPP-compliant ramp rate)
        if self.smooth_power is not None:
            prev = self._prev_power.expand(B, -1)
            power = self.smooth_power(z, prev)
            self._prev_power = power[:1].detach().clone()
            self._last_power = power  # store for info/logging

        return discrete_actions

    def _encode_flat_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode a flat obs sequence (B, T, obs_dim) → (B, latent_dim).

        Uses the temporal encoder directly (bypassing the GNN) when obs is flat,
        projecting through a shared linear to the GNN output dimension first.
        """
        B, T, D = seq.shape
        # Project flat obs → gnn_output_dim via a lazily-created linear layer
        if not hasattr(self, '_flat_proj'):
            self._flat_proj = nn.Linear(D, self.encoder.gnn.output_dim).to(self.device)
        g_seq = self._flat_proj(seq)   # (B, T, gnn_output_dim)
        z     = self.encoder.temporal(g_seq)  # (B, latent_dim)
        return z

    def _encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        Encode a proper multi-provider history.

        Args:
            history: (B, T, P, F) graph-structured history
        Returns:
            z: (B, latent_dim)
        """
        return self.encoder(history, self.adj)

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────

    def push(
        self,
        obs:      torch.Tensor,  # (B, D)
        actions:  torch.Tensor,  # (B,)
        rewards:  torch.Tensor,  # (B,)
        next_obs: torch.Tensor,  # (B, D)
        dones:    torch.Tensor,  # (B,)
    ):
        """Store transitions in the replay buffer (GPU, CPU, or PER)."""
        if self._use_per:
            self.replay.push_batch(obs, actions, rewards, next_obs, dones.float())
        elif self._gpu_replay:
            self.replay.push_batch(obs, actions, rewards, next_obs, dones.float())
        else:
            B = obs.shape[0]
            obs_np      = obs.cpu().numpy()
            actions_np  = actions.cpu().numpy()
            rewards_np  = rewards.cpu().numpy()
            next_obs_np = next_obs.cpu().numpy()
            dones_np    = dones.float().cpu().numpy()
            for i in range(B):
                self.replay.push(
                    obs_np[i], actions_np[i], rewards_np[i],
                    next_obs_np[i], dones_np[i],
                )

    def update(self) -> Optional[Dict[str, float]]:
        """
        Sample a batch and perform one SAC update step.

        Returns dict of loss metrics, or None if buffer not ready.
        """
        if len(self.replay) < self.cfg.batch_size:
            return None

        batch   = self.replay.sample(self.cfg.batch_size)
        obs     = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_states"]
        dones   = batch["dones"]
        alpha   = self.log_alpha.exp().detach()

        # PER importance-sampling weights (uniform 1.0 if not using PER)
        is_weights = batch.get("weights", torch.ones(obs.shape[0], device=self.device))
        tree_indices = batch.get("tree_indices", None)

        # ── Critic update ─────────────────────────────────────────
        with torch.no_grad():
            next_seq = next_obs.unsqueeze(1)
            z_next   = self._encode_flat_sequence(next_seq)

            next_probs = self.actor(z_next)
            next_log_p = torch.log(next_probs + 1e-8)

            q1_next, q2_next = self.critic_target(z_next)
            q_next_min = torch.min(q1_next, q2_next)

            v_next = (next_probs * (q_next_min - alpha * next_log_p)).sum(dim=-1)
            q_target = rewards + self.cfg.gamma * (1.0 - dones.float()) * v_next

        curr_seq = obs.unsqueeze(1)
        z_curr   = self._encode_flat_sequence(curr_seq)
        q1, q2   = self.critic(z_curr)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        # PER: weight the loss by importance-sampling weights
        td_error_1 = (q1_a - q_target).detach()
        td_error_2 = (q2_a - q_target).detach()
        critic_loss = (is_weights * (q1_a - q_target) ** 2).mean() + \
                      (is_weights * (q2_a - q_target) ** 2).mean()

        # Update priorities in the sum-tree
        if self._use_per and tree_indices is not None:
            td_errors = (td_error_1.abs() + td_error_2.abs()) / 2.0
            self.replay.update_priorities(tree_indices, td_errors)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Actor + encoder update ────────────────────────────────
        z_actor = self._encode_flat_sequence(curr_seq)
        probs   = self.actor(z_actor)          # (B, A)
        log_p   = torch.log(probs + 1e-8)

        with torch.no_grad():
            q1_det, q2_det = self.critic(z_actor)
            q_min = torch.min(q1_det, q2_det)

        # Actor loss: E_π[α log π - Q]
        actor_loss = (probs * (alpha * log_p - q_min)).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()), 1.0
        )
        self.opt_actor.step()

        # ── Alpha update ──────────────────────────────────────────
        with torch.no_grad():
            z_a2 = self._encode_flat_sequence(curr_seq)
            p2   = self.actor(z_a2)
            entropy = -(p2 * torch.log(p2 + 1e-8)).sum(dim=-1).mean()

        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # ── Polyak target update ──────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1 - self.cfg.tau_polyak)
                pt.data.add_(self.cfg.tau_polyak * p.data)

        self._updates += 1

        metrics = {
            "loss/critic":  critic_loss.item(),
            "loss/actor":   actor_loss.item(),
            "loss/alpha":   alpha_loss.item(),
            "alpha":        alpha.item(),
            "entropy":      entropy.item(),
        }

        # ── Dyna world model update (if enabled) ─────────────────
        if self.world_model is not None and self.wm_trainer is not None:
            if self._updates % self.cfg.wm_train_every == 0:
                wm_metrics = self.wm_trainer.train_step(
                    obs, actions.long(), rewards, next_obs, dones
                )
                metrics.update(wm_metrics)

                # Generate imagined transitions and push to replay
                def _policy_fn(h):
                    with torch.no_grad():
                        obs_pred = self.world_model.obs_decoder(h)
                        seq = obs_pred.unsqueeze(1)
                        z = self._encode_flat_sequence(seq)
                        probs = self.actor(z)
                        return probs.argmax(dim=-1)

                imagined = self.wm_trainer.generate_imagined_data(obs, _policy_fn)
                # Push imagined rewards/actions back as synthetic transitions
                n_imag = imagined["rewards"].shape[0]
                if n_imag > 0:
                    # Decode imagined hidden states to observations for replay
                    imag_obs = obs[:n_imag] if n_imag <= obs.shape[0] else obs.repeat(
                        (n_imag // obs.shape[0]) + 1, 1
                    )[:n_imag]
                    imag_next = imag_obs + torch.randn_like(imag_obs) * 0.01
                    self.push(
                        imag_obs,
                        imagined["actions"][:n_imag],
                        imagined["rewards"][:n_imag],
                        imag_next,
                        imagined["dones"][:n_imag],
                    )

        return metrics

    # ──────────────────────────────────────────────────────────────
    # Interpretability (KAN)
    # ──────────────────────────────────────────────────────────────

    def explain_action(self) -> Dict:
        """
        Return KAN spline statistics for regulatory audit.

        Tells network operators WHICH input features drive allocation decisions,
        satisfying ITU-R spectrum management transparency requirements.
        """
        return self.actor.get_spline_stats()

    def get_provider_preference(
        self, obs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute the marginal preference for each provider type.

        Aggregates action probabilities over channels within each provider,
        giving a single preference score per provider type.

        Args:
            obs: (1, obs_dim) single environment observation

        Returns:
            Dict mapping provider name → preference score in [0, 1]
        """
        with torch.no_grad():
            seq   = obs.unsqueeze(1)
            z     = self._encode_flat_sequence(seq)
            probs = self.actor(z).squeeze(0)   # (num_actions,)

        P = len(self.cfg.provider_types)
        C = self.num_actions // P

        prefs = {}
        for i, pt in enumerate(self.cfg.provider_types):
            pref = probs[i * C: (i + 1) * C].sum().item()
            prefs[pt.name] = round(pref, 4)
        return prefs

    # ──────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        sd = {
            "encoder":       self.encoder.state_dict(),
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "updates":       self._updates,
        }
        if self.world_model is not None:
            sd["world_model"] = self.world_model.state_dict()
        if self.smooth_power is not None:
            sd["smooth_power"] = self.smooth_power.state_dict()
        return sd

    def load_state_dict(self, sd: Dict):
        self.encoder.load_state_dict(sd["encoder"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.log_alpha.data = sd["log_alpha"]
        self._updates = sd.get("updates", 0)
        if self.world_model is not None and "world_model" in sd:
            self.world_model.load_state_dict(sd["world_model"])
        if self.smooth_power is not None and "smooth_power" in sd:
            self.smooth_power.load_state_dict(sd["smooth_power"])
