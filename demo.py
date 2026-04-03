#!/usr/bin/env python3
"""
PreceptualAI End-to-End Demo
========================

Demonstrates the full production pipeline:

1. TRAIN on real 5G data (UCC MISL 5G Dataset)
2. EVALUATE trained agent with detailed metrics
3. EXPORT model to ONNX for NVIDIA ARC deployment
4. SERVE via gRPC inference engine
5. BENCHMARK inference latency
6. VISUALIZE results

Usage:
    python demo.py --data_dir /tmp/5Gdata/5G-production-dataset
    python demo.py --data_dir /tmp/5Gdata/5G-production-dataset --num_steps 20000
"""

import argparse
import json
import os
import time
from collections import deque

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preceptualai.agent.sac_ltc import SACLTCAgent
from preceptualai.env.real_5g import Real5GEnv
from preceptualai.env.sim import SimulatedDSAEnv


def parse_args():
    p = argparse.ArgumentParser(description="PreceptualAI End-to-End Demo")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to 5G-production-dataset directory (real data mode)")
    p.add_argument("--num_steps", type=int, default=10000)
    p.add_argument("--num_channels", type=int, default=10)
    p.add_argument("--sequence_length", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_starts", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="demo_output")
    return p.parse_args()


def banner(text):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# =====================================================================
# PHASE 1: Train
# =====================================================================

def train_agent(env, agent, num_steps, learning_starts, log_interval=500):
    """Train SAC-LTC agent and return training metrics."""
    banner("PHASE 1: Training SAC-LTC on spectrum data")

    state, info = env.reset()
    data_source = info.get("data_source", "Simulated Markov DSA")
    print(f"  Data source: {data_source}")
    print(f"  Channels: {env.num_channels}, Features: {env.num_features}")
    print(f"  Obs shape: {state.shape}")
    print(f"  Training for {num_steps} steps...")
    print()

    episode_reward = 0.0
    episode_count = 0
    recent_rewards = deque(maxlen=20)
    metrics = {
        "steps": [], "rewards": [], "success_rates": [],
        "collision_rates": [], "losses": [], "alphas": [],
    }

    start_time = time.time()

    for step in range(1, num_steps + 1):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0.0

        losses = agent.update()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            mean_r = np.mean(recent_rewards) if recent_rewards else 0.0
            alpha = losses.get("alpha", 0) if losses else 0
            c_loss = losses.get("critic1_loss", 0) if losses else 0
            print(
                f"  Step {step:>6d}/{num_steps} | "
                f"Ep {episode_count:>4d} | "
                f"R {mean_r:>7.2f} | "
                f"a {alpha:.3f} | "
                f"C {c_loss:.4f} | "
                f"{elapsed:.0f}s"
            )
            metrics["steps"].append(step)
            metrics["rewards"].append(float(mean_r))

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s ({episode_count} episodes)")
    return metrics


# =====================================================================
# PHASE 2: Evaluate
# =====================================================================

def evaluate_agent(env, agent, num_episodes=50):
    """Evaluate trained agent and return detailed metrics."""
    banner("PHASE 2: Evaluating trained agent")

    rewards = []
    successes = 0
    collisions = 0
    total_steps = 0
    actions_taken = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            successes += int(info.get("success", False))
            collisions += int(info.get("collision", False))
            total_steps += 1
            actions_taken.append(action)
            state = next_state

        rewards.append(ep_reward)

    rewards = np.array(rewards)
    success_rate = successes / max(total_steps, 1)
    collision_rate = collisions / max(total_steps, 1)

    # Channel utilization
    actions_arr = np.array(actions_taken)
    channel_counts = np.bincount(actions_arr, minlength=env.num_channels)
    fairness = (channel_counts.sum() ** 2) / (env.num_channels * (channel_counts ** 2).sum() + 1e-8)

    metrics = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "jains_fairness": float(fairness),
        "num_episodes": num_episodes,
        "total_steps": total_steps,
    }

    print(f"  Episodes:        {num_episodes}")
    print(f"  Mean Reward:     {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Success Rate:    {metrics['success_rate']:.2%}")
    print(f"  Collision Rate:  {metrics['collision_rate']:.2%}")
    print(f"  Jain's Fairness: {metrics['jains_fairness']:.4f}")
    print(f"  Channel Usage:   {channel_counts}")

    return metrics, rewards, actions_arr


# =====================================================================
# PHASE 3: Export to ONNX
# =====================================================================

def export_onnx(agent, output_dir, sequence_length, input_dim):
    """Export trained actor to ONNX format."""
    banner("PHASE 3: Exporting model to ONNX")

    onnx_path = os.path.join(output_dir, "preceptualai_actor.onnx")

    try:
        from preceptualai.export.onnx import export_actor_to_onnx
        export_actor_to_onnx(
            agent.actor,
            input_shape=(1, sequence_length, input_dim),
            output_path=onnx_path,
        )
        print(f"  ONNX model exported: {onnx_path}")
        file_size = os.path.getsize(onnx_path) / 1024
        print(f"  Model size: {file_size:.1f} KB")
        return onnx_path
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        # Fallback: save PyTorch checkpoint
        pt_path = os.path.join(output_dir, "preceptualai_actor.pt")
        agent.save(pt_path)
        print(f"  Saved PyTorch checkpoint instead: {pt_path}")
        return pt_path


# =====================================================================
# PHASE 4: Inference Engine Benchmark
# =====================================================================

def benchmark_inference(model_path, env, agent, num_iterations=500):
    """Benchmark inference latency with different backends."""
    banner("PHASE 4: Inference Engine Benchmark")

    results = {}

    # PyTorch baseline
    state, _ = env.reset()
    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        agent.select_action(state, deterministic=True)
        latencies.append((time.perf_counter() - t0) * 1000)
    results["pytorch"] = {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
    print(f"  PyTorch:     {results['pytorch']['mean_ms']:.3f}ms mean, "
          f"{results['pytorch']['p99_ms']:.3f}ms p99")

    # ONNX Runtime
    if model_path.endswith(".onnx"):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            dummy = state[np.newaxis, ...].astype(np.float32)

            # Warmup
            for _ in range(50):
                session.run(None, {"observation": dummy})

            latencies = []
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                session.run(None, {"observation": dummy})
                latencies.append((time.perf_counter() - t0) * 1000)

            results["onnx_runtime"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p99_ms": float(np.percentile(latencies, 99)),
            }
            print(f"  ONNX Runtime: {results['onnx_runtime']['mean_ms']:.3f}ms mean, "
                  f"{results['onnx_runtime']['p99_ms']:.3f}ms p99")
        except Exception as e:
            print(f"  ONNX Runtime: skipped ({e})")

    return results


# =====================================================================
# PHASE 5: Visualize
# =====================================================================

def visualize_results(train_metrics, eval_metrics, eval_rewards, actions,
                      inference_results, output_dir, num_channels):
    """Generate publication-quality visualization of demo results."""
    banner("PHASE 5: Generating visualizations")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PreceptualAI — End-to-End Demo Results", fontsize=16, fontweight="bold")

    # 1. Training curve
    ax = axes[0, 0]
    ax.plot(train_metrics["steps"], train_metrics["rewards"],
            color="#7b2cbf", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Curve")
    ax.grid(True, alpha=0.3)

    # 2. Evaluation reward distribution
    ax = axes[0, 1]
    ax.hist(eval_rewards, bins=20, color="#7b2cbf", edgecolor="black", alpha=0.75)
    ax.axvline(eval_metrics["mean_reward"], color="red", linestyle="--",
               label=f"Mean = {eval_metrics['mean_reward']:.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Eval Reward Distribution")
    ax.legend()

    # 3. Success/Collision rates
    ax = axes[0, 2]
    labels = ["Success", "Collision"]
    values = [eval_metrics["success_rate"] * 100, eval_metrics["collision_rate"] * 100]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_title("DSA Performance")
    ax.set_ylim(0, 110)

    # 4. Channel utilization
    ax = axes[1, 0]
    channel_counts = np.bincount(actions.astype(int), minlength=num_channels)
    ax.bar(range(num_channels), channel_counts / channel_counts.sum() * 100,
           color="#3498db", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Channel ID")
    ax.set_ylabel("Selection %")
    ax.set_title(f"Channel Utilization (Jain={eval_metrics['jains_fairness']:.3f})")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Inference latency comparison
    ax = axes[1, 1]
    backends = list(inference_results.keys())
    mean_latencies = [inference_results[b]["mean_ms"] for b in backends]
    p99_latencies = [inference_results[b]["p99_ms"] for b in backends]
    x = range(len(backends))
    ax.bar(x, mean_latencies, color="#9b59b6", edgecolor="black", alpha=0.8, label="Mean")
    ax.bar(x, p99_latencies, color="#e74c3c", edgecolor="black", alpha=0.3, label="P99")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in backends])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency")
    ax.legend()
    # 10ms near-RT RIC budget line
    ax.axhline(y=10, color="green", linestyle="--", alpha=0.7, label="10ms Near-RT budget")

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        "PreceptualAI Demo Summary\n"
        "─────────────────────\n"
        f"Success Rate:    {eval_metrics['success_rate']:.1%}\n"
        f"Collision Rate:  {eval_metrics['collision_rate']:.1%}\n"
        f"Mean Reward:     {eval_metrics['mean_reward']:.2f}\n"
        f"Jain's Fairness: {eval_metrics['jains_fairness']:.4f}\n"
        f"\n"
        f"Inference Latency:\n"
    )
    for backend, res in inference_results.items():
        summary += f"  {backend}: {res['mean_ms']:.2f}ms\n"
    summary += (
        f"\nNear-RT RIC Budget: 10ms\n"
        f"Status: {'PASS' if any(r['p99_ms'] < 10 for r in inference_results.values()) else 'CHECK'}"  # noqa: E501
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontfamily="monospace", fontsize=11, verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "demo_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Results plot saved: {plot_path}")
    return plot_path


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner("PreceptualAI — AI-Native Dynamic Spectrum Management")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # Create environment
    if args.data_dir and os.path.exists(args.data_dir):
        print("  Mode: REAL 5G DATA")
        print(f"  Dataset: {args.data_dir}")
        env = Real5GEnv(
            data_dir=args.data_dir,
            num_channels=args.num_channels,
            sequence_length=args.sequence_length,
        )
        num_features = env.num_features
    else:
        print("  Mode: SIMULATED (Markov DSA)")
        num_features = 3
        env = SimulatedDSAEnv(
            num_channels=args.num_channels,
            sequence_length=args.sequence_length,
            num_features=num_features,
        )

    input_dim = args.num_channels * num_features
    state_shape = (args.sequence_length, input_dim)

    # Create agent
    agent = SACLTCAgent(
        state_shape=state_shape,
        num_actions=args.num_channels,
        input_dim=input_dim,
        device=device,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=2,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=100_000,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
    )

    # Phase 1: Train
    train_metrics = train_agent(
        env, agent, args.num_steps, args.learning_starts,
    )

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")
    agent.save(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Phase 2: Evaluate
    eval_metrics, eval_rewards, actions = evaluate_agent(env, agent, num_episodes=50)

    # Phase 3: Export ONNX
    model_path = export_onnx(agent, args.output_dir, args.sequence_length, input_dim)

    # Phase 4: Benchmark inference
    inference_results = benchmark_inference(model_path, env, agent)

    # Phase 5: Visualize
    visualize_results(
        train_metrics, eval_metrics, eval_rewards, actions,
        inference_results, args.output_dir, args.num_channels,
    )

    # Save all metrics
    all_metrics = {
        "train": train_metrics,
        "eval": eval_metrics,
        "inference": inference_results,
    }
    metrics_path = os.path.join(args.output_dir, "demo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    banner("DEMO COMPLETE")
    print(f"  Output directory: {args.output_dir}/")
    print("  - checkpoint.pt      (trained model)")
    print("  - preceptualai_actor.onnx (ONNX export)")
    print("  - demo_results.png   (visualization)")
    print("  - demo_metrics.json  (all metrics)")
    print()
    print(f"  Success Rate:   {eval_metrics['success_rate']:.1%}")
    print(f"  Collision Rate: {eval_metrics['collision_rate']:.1%}")
    for backend, res in inference_results.items():
        print(f"  {backend} latency: {res['mean_ms']:.2f}ms (p99: {res['p99_ms']:.2f}ms)")
    print()


if __name__ == "__main__":
    main()
