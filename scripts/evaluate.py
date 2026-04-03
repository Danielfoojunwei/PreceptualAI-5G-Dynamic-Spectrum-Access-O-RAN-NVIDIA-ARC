"""
Evaluation for trained PreceptualAI SAC-LTC agents.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoint_final.pt
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preceptualai.agent.sac_ltc import SACLTCAgent
from preceptualai.env.sim import SimulatedDSAEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PreceptualAI SAC-LTC agent")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--curves", type=str, default=None,
                   help="Path to training_curves.json (auto-detected if omitted)")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="results")

    # Environment / model shape (must match training config)
    p.add_argument("--num_channels", type=int, default=10)
    p.add_argument("--sequence_length", type=int, default=16)
    p.add_argument("--num_features", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)

    return p.parse_args()


def run_evaluation(env, agent, num_episodes: int):
    """Run evaluation episodes and collect detailed metrics."""
    episode_rewards = []
    episode_successes = []
    episode_collisions = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_success = 0
        ep_collision = 0
        ep_len = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_success += int(info.get("success", False))
            ep_collision += int(info.get("collision", False))
            ep_len += 1
            state = next_state

        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        episode_collisions.append(ep_collision)
        episode_lengths.append(ep_len)

    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    episode_successes = np.array(episode_successes)
    episode_collisions = np.array(episode_collisions)

    total_steps = episode_lengths.sum()
    success_rate = episode_successes.sum() / total_steps
    collision_rate = episode_collisions.sum() / total_steps

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": float(episode_rewards.mean()),
        "std_reward": float(episode_rewards.std()),
        "median_reward": float(np.median(episode_rewards)),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "mean_episode_length": float(episode_lengths.mean()),
    }


def plot_eval_metrics(metrics: dict, output_path: str):
    """Bar chart of success rate vs collision rate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(metrics["episode_rewards"], bins=20, edgecolor="black", alpha=0.75)
    ax.axvline(metrics["mean_reward"], color="red", linestyle="--",
               label=f"Mean = {metrics['mean_reward']:.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("PreceptualAI Evaluation Reward Distribution")
    ax.legend()

    ax = axes[1]
    labels = ["Success Rate", "Collision Rate"]
    values = [metrics["success_rate"] * 100, metrics["collision_rate"] * 100]
    colours = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, values, color=colours, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_title("DSA Performance Metrics")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Evaluation metrics plot saved → {output_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    env = SimulatedDSAEnv(
        num_channels=args.num_channels,
        sequence_length=args.sequence_length,
        num_features=args.num_features,
        max_steps=args.max_episode_steps,
    )
    input_dim = args.num_channels * args.num_features
    state_shape = (args.sequence_length, input_dim)

    agent = SACLTCAgent(
        state_shape=state_shape,
        num_actions=args.num_channels,
        input_dim=input_dim,
        device=device,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    print(f"Running {args.num_episodes} evaluation episodes ...")
    metrics = run_evaluation(env, agent, args.num_episodes)

    print("\n" + "=" * 50)
    print("PRECEPTUALAI EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Episodes          : {args.num_episodes}")
    print(f"  Mean Reward       : {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Median Reward     : {metrics['median_reward']:.2f}")
    print(f"  Success Rate      : {metrics['success_rate']:.2%}")
    print(f"  Collision Rate    : {metrics['collision_rate']:.2%}")
    print(f"  Mean Ep. Length   : {metrics['mean_episode_length']:.1f}")
    print("=" * 50)

    save_metrics = {k: v for k, v in metrics.items() if k != "episode_rewards"}
    save_metrics["episode_rewards"] = metrics["episode_rewards"].tolist()
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    eval_plot_path = os.path.join(args.output_dir, "eval_metrics.png")
    plot_eval_metrics(metrics, eval_plot_path)


if __name__ == "__main__":
    main()
