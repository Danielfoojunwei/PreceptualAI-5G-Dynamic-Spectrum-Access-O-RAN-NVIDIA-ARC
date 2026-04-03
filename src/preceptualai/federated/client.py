"""
Federated SAC-LTC Client.

Wraps a ``SACLTCAgent`` for participation in the PreceptualAI federated learning
loop.  Handles weight extraction, global weight application, local training,
and evaluation.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch

from preceptualai.agent.sac_ltc import SACLTCAgent
from preceptualai.federated.config import FLConfig


class FederatedSACLTCClient:
    """
    Client wrapper for federated SAC-LTC training.

    Each physical edge device instantiates one ``FederatedSACLTCClient``
    around its local ``SACLTCAgent``.  The client exposes methods to:

    * Extract local model weights (actor + critics) for upload.
    * Apply globally aggregated weights (without touching targets or alpha).
    * Run a local training loop for a configurable number of steps.
    * Evaluate the current policy over a number of episodes.
    """

    def __init__(
        self,
        agent: SACLTCAgent,
        device_id: str,
        config: FLConfig,
    ):
        self.agent = agent
        self.device_id = device_id
        self.config = config

        # Snapshot of global weights received at start of local round,
        # used for FedProx proximal term.
        self._global_snapshot: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    def get_local_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract state_dicts for the trainable model components.

        Returns:
            A dict with keys ``"actor"``, ``"critic1"``, ``"critic2"``,
            each mapping to the component's ``state_dict()`` (detached,
            moved to CPU).

        Note:
            Target critics and ``log_alpha`` are intentionally excluded.
            Targets are Polyak-updated locally and should not participate
            in federated aggregation.  ``log_alpha`` is device-specific
            entropy tuning.
        """
        return {
            "actor": _cpu_state_dict(self.agent.actor),
            "critic1": _cpu_state_dict(self.agent.critic1),
            "critic2": _cpu_state_dict(self.agent.critic2),
        }

    # ------------------------------------------------------------------
    # Weight application
    # ------------------------------------------------------------------

    def apply_global_weights(
        self, weights: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """Load globally aggregated weights into actor and critics.

        **Does NOT touch:**
          - ``target_critic1`` / ``target_critic2`` (Polyak-updated locally)
          - ``log_alpha`` (device-specific entropy coefficient)

        Args:
            weights: Dict mapping ``"actor"``, ``"critic1"``, ``"critic2"``
                to the corresponding aggregated state_dicts.
        """
        device = self.agent.device

        if "actor" in weights:
            self.agent.actor.load_state_dict(
                _to_device(weights["actor"], device)
            )
        if "critic1" in weights:
            self.agent.critic1.load_state_dict(
                _to_device(weights["critic1"], device)
            )
        if "critic2" in weights:
            self.agent.critic2.load_state_dict(
                _to_device(weights["critic2"], device)
            )

        # Keep a snapshot for FedProx
        if self.config.aggregation_method == "fedprox":
            self._global_snapshot = {
                comp: {k: v.clone().to(device) for k, v in sd.items()}
                for comp, sd in weights.items()
            }

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train_local(
        self, env: Any, num_steps: int
    ) -> Dict[str, float]:
        """Run a local training loop for ``num_steps`` environment steps.

        The agent interacts with ``env`` collecting transitions and
        performing SAC updates at every step (once the replay buffer is
        sufficiently populated).

        If the aggregation method is ``"fedprox"``, a proximal term
        is added to each critic update to penalise divergence from the
        global model.

        Args:
            env: A gymnasium-style environment with ``reset()`` and
                ``step()`` methods.
            num_steps: Number of environment steps to run.

        Returns:
            Metrics dict with aggregated training statistics:
            ``mean_reward``, ``total_steps``, ``mean_critic_loss``,
            ``mean_actor_loss``.
        """
        rewards = []
        critic_losses = []
        actor_losses = []

        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(num_steps):
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            self.agent.replay_buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward

            # Apply FedProx proximal gradient correction BEFORE optimizer
            # steps so the regularization term affects the update direction.
            if (
                self.config.aggregation_method == "fedprox"
                and self._global_snapshot is not None
            ):
                self._apply_fedprox_correction()

            # SAC update
            update_info = self.agent.update()

            if update_info:
                if "critic1_loss" in update_info:
                    critic_losses.append(
                        (update_info["critic1_loss"] + update_info["critic2_loss"]) / 2
                    )
                if "actor_loss" in update_info:
                    actor_losses.append(update_info["actor_loss"])

            if terminated or truncated:
                rewards.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()

        # Final partial episode
        if episode_reward != 0.0:
            rewards.append(episode_reward)

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "total_steps": num_steps,
            "num_episodes": len(rewards),
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        }

    def _apply_fedprox_correction(self) -> None:
        """Apply FedProx proximal gradient correction to trainable params.

        Adds ``mu * (param - global_param)`` to each parameter's gradient
        for actor and both critics.
        """
        mu = self.config.fedprox_mu
        if mu <= 0 or self._global_snapshot is None:
            return

        pairs = [
            ("actor", self.agent.actor),
            ("critic1", self.agent.critic1),
            ("critic2", self.agent.critic2),
        ]
        for comp_name, module in pairs:
            if comp_name not in self._global_snapshot:
                continue
            global_sd = self._global_snapshot[comp_name]
            for name, param in module.named_parameters():
                if param.grad is not None and name in global_sd:
                    param.grad.data.add_(
                        mu * (param.data - global_sd[name])
                    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, env: Any, num_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate the current policy deterministically.

        Runs ``num_episodes`` rollouts using the greedy (deterministic)
        policy and reports aggregate statistics.

        Args:
            env: A gymnasium-style environment.
            num_episodes: Number of evaluation episodes.

        Returns:
            Dict with ``success_rate``, ``collision_rate``, ``mean_reward``,
            ``std_reward``, and ``mean_episode_length``.
        """
        episode_rewards = []
        episode_lengths = []
        total_successes = 0
        total_collisions = 0
        total_steps = 0

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_len += 1
                total_steps += 1

                if info.get("success", False):
                    total_successes += 1
                if info.get("collision", False):
                    total_collisions += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

        return {
            "success_rate": total_successes / max(total_steps, 1),
            "collision_rate": total_collisions / max(total_steps, 1),
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a detached, CPU copy of a module's state_dict."""
    return OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in module.state_dict().items()
    )


def _to_device(
    state_dict: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move every tensor in a state_dict to ``device``."""
    return OrderedDict((k, v.to(device)) for k, v in state_dict.items())
