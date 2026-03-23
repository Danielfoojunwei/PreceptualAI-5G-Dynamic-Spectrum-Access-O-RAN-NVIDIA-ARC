"""
Federated Learning Flywheel Demo for SpectrAI.

Proves three key properties of the hybrid LTC-aware federated aggregation:

  1. **Convergence speed-up** — FL devices converge faster than independent
     training because structural knowledge is shared.
  2. **Cold-start elimination** — A brand-new device can download the global
     model and perform well immediately, without any local training.
  3. **Flywheel effect** — Adding new devices to the federation improves
     performance for *existing* devices, not just the newcomers.

Runs entirely in-process (no gRPC required).  Uses the Real5GEnv backed by
the UCC MISL 5G dataset with traces split non-IID across simulated devices.

Usage:
    python -m spectrai.federated.demo \\
        --data_dir data/5g_dataset \\
        --num_rounds 10 \\
        --local_steps 500 \\
        --output_dir demo_output
"""

import argparse
import glob
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from spectrai.agent.sac_ltc import SACLTCAgent
from spectrai.env.real_5g import Real5GEnv
from spectrai.federated.aggregator import HybridFederatedAggregator
from spectrai.federated.client import FederatedSACLTCClient
from spectrai.federated.config import FLConfig

# ======================================================================
# Trace splitting (non-IID)
# ======================================================================


def discover_trace_groups(data_dir: str) -> Dict[str, List[str]]:
    """Group CSV trace files by their parent directory path pattern.

    The UCC MISL 5G dataset is structured as:
        data_dir/<App>/<Mobility>/<run>.csv
    e.g.  Netflix/Static/run1.csv, Amazon_Prime/Driving/run3.csv

    Returns:
        Dict mapping group key (e.g. "Netflix/Static") to list of CSV paths.
    """
    csv_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    )
    csv_files = [f for f in csv_files if "MACOSX" not in f]

    groups: Dict[str, List[str]] = defaultdict(list)
    for fpath in csv_files:
        rel = os.path.relpath(fpath, data_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            group_key = os.path.join(*parts[:-1])
        else:
            group_key = "default"
        groups[group_key].append(fpath)

    return dict(groups)


def split_traces_to_devices(
    data_dir: str, num_devices: int, holdout_device: bool = False
) -> Tuple[List[List[str]], Optional[List[str]]]:
    """Split traces non-IID across devices.

    Each device gets one or more full trace groups (e.g. all Netflix/Static
    traces go to one device).  This creates realistic heterogeneity — each
    device sees a different mix of application types and mobility patterns.

    Args:
        data_dir: Root directory of the 5G dataset.
        num_devices: Number of FL devices to create.
        holdout_device: If True, reserve one trace group as a holdout for
            cold-start testing (returned separately).

    Returns:
        Tuple of (device_traces, holdout_traces).
        device_traces: List of length ``num_devices``, each element a list
            of CSV file paths assigned to that device.
        holdout_traces: List of CSV paths held out, or None.
    """
    groups = discover_trace_groups(data_dir)
    group_keys = sorted(groups.keys())

    if not group_keys:
        raise FileNotFoundError(f"No trace groups found in {data_dir}")

    holdout_traces: Optional[List[str]] = None
    if holdout_device and len(group_keys) > num_devices:
        holdout_key = group_keys.pop()
        holdout_traces = groups[holdout_key]

    # Round-robin assignment of groups to devices
    device_traces: List[List[str]] = [[] for _ in range(num_devices)]
    for i, key in enumerate(group_keys):
        device_idx = i % num_devices
        device_traces[device_idx].extend(groups[key])

    # Ensure every device has at least some traces
    # If a device ended up empty, give it copies from the largest device
    for i in range(num_devices):
        if not device_traces[i]:
            largest = max(device_traces, key=len)
            # Split the largest group
            half = len(largest) // 2
            if half > 0:
                device_traces[i] = largest[half:]
                del largest[half:]

    return device_traces, holdout_traces


# ======================================================================
# Environment factory
# ======================================================================


def make_env(
    data_dir: str,
    trace_files: Optional[List[str]] = None,
    num_channels: int = 5,
    sequence_length: int = 16,
    max_steps: int = 200,
) -> Real5GEnv:
    """Create a Real5GEnv, optionally restricted to specific trace files.

    If ``trace_files`` is given, a temporary symlink directory is created
    so that Real5GEnv only loads the specified traces.
    """
    if trace_files is not None and len(trace_files) > 0:
        # Create a temp directory with symlinks to the selected traces
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="spectrai_fl_")
        for idx, src in enumerate(trace_files):
            # Preserve directory structure for grouping
            rel = os.path.basename(src)
            dst = os.path.join(tmp_dir, f"trace_{idx}_{rel}")
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                # Fallback: copy if symlinks not supported
                import shutil
                shutil.copy2(src, dst)
        data_dir = tmp_dir

    return Real5GEnv(
        data_dir=data_dir,
        num_channels=num_channels,
        sequence_length=sequence_length,
        max_steps=max_steps,
    )


# ======================================================================
# Agent factory
# ======================================================================


def make_agent(
    num_channels: int = 5,
    num_features: int = 5,
    sequence_length: int = 16,
    hidden_dim: int = 32,
    latent_dim: int = 32,
    device: torch.device = torch.device("cpu"),
    learning_starts: int = 64,
    batch_size: int = 64,
    buffer_size: int = 50_000,
) -> SACLTCAgent:
    """Create a small SAC-LTC agent suitable for the demo."""
    obs_dim = num_channels * num_features
    state_shape = (sequence_length, obs_dim)
    return SACLTCAgent(
        state_shape=state_shape,
        num_actions=num_channels,
        input_dim=obs_dim,
        device=device,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=2,
        dt=1.0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
    )


# ======================================================================
# Experiment helpers
# ======================================================================


def train_independent(
    agents: List[SACLTCAgent],
    envs: List[Real5GEnv],
    total_steps: int,
    checkpoint_interval: int,
) -> List[List[float]]:
    """Train agents independently (no FL) and record checkpointed rewards.

    Returns:
        List of per-agent reward curves (list of mean rewards at checkpoints).
    """
    n = len(agents)
    reward_curves: List[List[float]] = [[] for _ in range(n)]

    for i in range(n):
        state, _ = envs[i].reset()
        episode_reward = 0.0
        recent_rewards: List[float] = []

        for step in range(total_steps):
            action = agents[i].select_action(state)
            next_state, reward, terminated, truncated, info = envs[i].step(action)
            agents[i].replay_buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward

            agents[i].update()

            if terminated or truncated:
                recent_rewards.append(episode_reward)
                episode_reward = 0.0
                state, _ = envs[i].reset()

            if (step + 1) % checkpoint_interval == 0:
                if recent_rewards:
                    reward_curves[i].append(float(np.mean(recent_rewards[-20:])))
                else:
                    reward_curves[i].append(0.0)

        print(f"  [Independent] Device {i}: final reward "
              f"{reward_curves[i][-1]:.2f}" if reward_curves[i] else "N/A")

    return reward_curves


def train_federated(
    clients: List[FederatedSACLTCClient],
    envs: List[Real5GEnv],
    aggregator: HybridFederatedAggregator,
    num_rounds: int,
    local_steps: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Run the FL training loop and record per-device metrics.

    Returns:
        (reward_curves, success_curves): per-device lists of per-round metrics.
    """
    n = len(clients)
    reward_curves: List[List[float]] = [[] for _ in range(n)]
    success_curves: List[List[float]] = [[] for _ in range(n)]

    for fl_round in range(num_rounds):
        print(f"  [FL] Round {fl_round + 1}/{num_rounds}")

        # Local training
        for i, (client, env) in enumerate(zip(clients, envs)):
            metrics = client.train_local(env, local_steps)
            reward_curves[i].append(metrics["mean_reward"])

            # Quick evaluation
            eval_metrics = client.evaluate(env, num_episodes=3)
            success_curves[i].append(eval_metrics["success_rate"])

            # Upload
            aggregator.receive_update(
                client.device_id,
                client.get_local_weights(),
                num_samples=local_steps,
            )

            print(f"    Device {client.device_id}: reward={metrics['mean_reward']:.2f}, "
                  f"success={eval_metrics['success_rate']:.3f}")

        # Aggregate
        agg_metrics = aggregator.aggregate()
        print(f"    Aggregated ({agg_metrics['method']}, "
              f"{agg_metrics['num_devices']} devices)")

        # Download personalized models
        for client in clients:
            personalized = aggregator.get_personalized_model(client.device_id)
            client.apply_global_weights(personalized)

    return reward_curves, success_curves


# ======================================================================
# Experiment 1: Convergence speed-up
# ======================================================================


def experiment_convergence(
    data_dir: str,
    device_traces: List[List[str]],
    num_rounds: int,
    local_steps: int,
    num_channels: int,
    torch_device: torch.device,
) -> Dict[str, Any]:
    """Compare FL vs independent training convergence."""
    print("\n=== Experiment 1: Convergence Speed-up ===")
    n_devices = len(device_traces)
    total_steps = num_rounds * local_steps

    # --- Independent baseline ---
    print("Training independent agents...")
    ind_agents = [make_agent(num_channels=num_channels, device=torch_device)
                  for _ in range(n_devices)]
    ind_envs = [make_env(data_dir, traces, num_channels=num_channels)
                for traces in device_traces]
    ind_curves = train_independent(
        ind_agents, ind_envs, total_steps,
        checkpoint_interval=local_steps,
    )

    # --- Federated ---
    print("Training federated agents...")
    fl_config = FLConfig(  # type: ignore[call-arg]
        num_rounds=num_rounds,
        local_steps_per_round=local_steps,
        min_devices_per_round=1,
        aggregation_method="hybrid_ltc",
        tau_mix_ratio=0.3,
    )
    fl_agents = [make_agent(num_channels=num_channels, device=torch_device)
                 for _ in range(n_devices)]
    fl_envs = [make_env(data_dir, traces, num_channels=num_channels)
               for traces in device_traces]
    fl_clients = [
        FederatedSACLTCClient(agent, f"device_{i}", fl_config)
        for i, agent in enumerate(fl_agents)
    ]
    aggregator = HybridFederatedAggregator(fl_config)

    fl_curves, fl_success = train_federated(
        fl_clients, fl_envs, aggregator, num_rounds, local_steps
    )

    return {
        "ind_curves": ind_curves,
        "fl_curves": fl_curves,
        "fl_success": fl_success,
        "fl_clients": fl_clients,
        "aggregator": aggregator,
        "fl_agents": fl_agents,
        "fl_envs": fl_envs,
    }


# ======================================================================
# Experiment 2: Cold-start elimination
# ======================================================================


def experiment_cold_start(
    data_dir: str,
    holdout_traces: List[str],
    aggregator: HybridFederatedAggregator,
    num_channels: int,
    torch_device: torch.device,
    num_eval_episodes: int = 5,
) -> Dict[str, Any]:
    """Test cold-start device performance with global model vs random init."""
    print("\n=== Experiment 2: Cold-start Elimination ===")

    fl_config = FLConfig(aggregation_method="hybrid_ltc", tau_mix_ratio=0.3)  # type: ignore[call-arg]

    # Device 6 with holdout traces
    env = make_env(data_dir, holdout_traces, num_channels=num_channels)
    agent_cold = make_agent(num_channels=num_channels, device=torch_device)
    client_cold = FederatedSACLTCClient(agent_cold, "device_cold", fl_config)

    # Evaluate with random initialization (baseline)
    random_metrics = client_cold.evaluate(env, num_episodes=num_eval_episodes)
    print(f"  Random init: success={random_metrics['success_rate']:.3f}, "
          f"reward={random_metrics['mean_reward']:.2f}")

    # Download global model (cold-start — device has no history)
    global_model = aggregator.get_global_model()
    client_cold.apply_global_weights(global_model)

    # Evaluate immediately (no local training)
    global_metrics = client_cold.evaluate(env, num_episodes=num_eval_episodes)
    print(f"  Global model: success={global_metrics['success_rate']:.3f}, "
          f"reward={global_metrics['mean_reward']:.2f}")

    return {
        "random_metrics": random_metrics,
        "global_metrics": global_metrics,
    }


# ======================================================================
# Experiment 3: Flywheel effect
# ======================================================================


def experiment_flywheel(
    data_dir: str,
    device_traces: List[List[str]],
    num_rounds_phase1: int,
    num_rounds_phase2: int,
    local_steps: int,
    num_channels: int,
    torch_device: torch.device,
) -> Dict[str, Any]:
    """Show that adding devices improves existing devices."""
    print("\n=== Experiment 3: Flywheel Effect ===")

    fl_config = FLConfig(  # type: ignore[call-arg]
        num_rounds=num_rounds_phase1 + num_rounds_phase2,
        local_steps_per_round=local_steps,
        min_devices_per_round=1,
        aggregation_method="hybrid_ltc",
        tau_mix_ratio=0.3,
    )

    # Phase 1: 3 devices
    n_phase1 = min(3, len(device_traces))
    n_total = len(device_traces)

    agents = [make_agent(num_channels=num_channels, device=torch_device)
              for _ in range(n_total)]
    envs = [make_env(data_dir, traces, num_channels=num_channels)
            for traces in device_traces]
    clients = [
        FederatedSACLTCClient(agent, f"fly_{i}", fl_config)
        for i, agent in enumerate(agents)
    ]
    aggregator = HybridFederatedAggregator(fl_config)

    print(f"  Phase 1: {n_phase1} devices for {num_rounds_phase1} rounds")
    phase1_clients = clients[:n_phase1]
    phase1_envs = envs[:n_phase1]

    p1_rewards, p1_success = train_federated(
        phase1_clients, phase1_envs, aggregator,
        num_rounds_phase1, local_steps,
    )

    # Record devices 1-3 success rate at end of phase 1
    phase1_final_success = [
        clients[i].evaluate(envs[i], num_episodes=3)["success_rate"]
        for i in range(n_phase1)
    ]
    print(f"  Phase 1 final success (devices 0-{n_phase1-1}): "
          f"{[f'{s:.3f}' for s in phase1_final_success]}")

    # Phase 2: add devices 4-5 (copy global model for warm start)
    print(f"  Phase 2: adding devices {n_phase1}-{n_total-1}")
    for i in range(n_phase1, n_total):
        global_model = aggregator.get_global_model()
        clients[i].apply_global_weights(global_model)

    all_clients = clients[:n_total]
    all_envs = envs[:n_total]

    p2_rewards, p2_success = train_federated(
        all_clients, all_envs, aggregator,
        num_rounds_phase2, local_steps,
    )

    # Record devices 1-3 success rate at end of phase 2
    phase2_final_success = [
        clients[i].evaluate(envs[i], num_episodes=3)["success_rate"]
        for i in range(n_phase1)
    ]
    print(f"  Phase 2 final success (devices 0-{n_phase1-1}): "
          f"{[f'{s:.3f}' for s in phase2_final_success]}")

    return {
        "n_phase1": n_phase1,
        "phase1_success": p1_success,
        "phase2_success": p2_success,
        "phase1_final": phase1_final_success,
        "phase2_final": phase2_final_success,
        "p1_rewards": p1_rewards,
        "p2_rewards": p2_rewards,
    }


# ======================================================================
# Tau divergence analysis
# ======================================================================


def extract_tau_divergence(
    aggregator: HybridFederatedAggregator,
) -> Dict[str, np.ndarray]:
    """Extract per-device tau_base values to show personalization divergence."""
    device_taus: Dict[str, np.ndarray] = {}

    for device_id in aggregator.registered_devices:
        tau_weights = aggregator.get_device_tau(device_id)
        if tau_weights is None:
            continue

        # Collect all tau_base parameters across components
        all_tau_base: List[np.ndarray] = []
        for comp_name, comp_tau in tau_weights.items():
            for key, tensor in comp_tau.items():
                if "tau_base" in key:
                    all_tau_base.append(tensor.detach().cpu().numpy().flatten())

        if all_tau_base:
            device_taus[device_id] = np.concatenate(all_tau_base)

    return device_taus


# ======================================================================
# Plotting
# ======================================================================


def plot_results(
    exp1: Dict[str, Any],
    exp2: Dict[str, Any],
    exp3: Dict[str, Any],
    aggregator: HybridFederatedAggregator,
    output_path: str,
) -> None:
    """Generate the 6-panel summary figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel 1: FL vs Independent convergence ----
    ax1 = fig.add_subplot(gs[0, 0])
    ind_curves = np.array(exp1["ind_curves"])
    fl_curves = np.array(exp1["fl_curves"])

    n_points_ind = ind_curves.shape[1] if ind_curves.ndim == 2 and ind_curves.shape[1] > 0 else 0
    n_points_fl = fl_curves.shape[1] if fl_curves.ndim == 2 and fl_curves.shape[1] > 0 else 0

    if n_points_ind > 0:
        x_ind = np.arange(1, n_points_ind + 1)
        mean_ind = ind_curves.mean(axis=0)
        std_ind = ind_curves.std(axis=0)
        ax1.plot(x_ind, mean_ind, "b-", label="Independent", linewidth=2)
        ax1.fill_between(x_ind, mean_ind - std_ind, mean_ind + std_ind, alpha=0.2, color="blue")

    if n_points_fl > 0:
        x_fl = np.arange(1, n_points_fl + 1)
        mean_fl = fl_curves.mean(axis=0)
        std_fl = fl_curves.std(axis=0)
        ax1.plot(x_fl, mean_fl, "r-", label="Federated (hybrid)", linewidth=2)
        ax1.fill_between(x_fl, mean_fl - std_fl, mean_fl + std_fl, alpha=0.2, color="red")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("1. FL vs Independent Convergence")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Per-device FL reward ----
    ax2 = fig.add_subplot(gs[0, 1])
    if n_points_fl > 0:
        for i in range(fl_curves.shape[0]):
            ax2.plot(x_fl, fl_curves[i], marker="o", markersize=3,
                     label=f"Device {i}", linewidth=1.5)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("2. Per-Device FL Reward")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Cold-start ----
    ax3 = fig.add_subplot(gs[0, 2])
    random_sr = exp2["random_metrics"]["success_rate"]
    global_sr = exp2["global_metrics"]["success_rate"]
    _random_rw = exp2["random_metrics"]["mean_reward"]
    _global_rw = exp2["global_metrics"]["mean_reward"]

    x_pos = [0, 1]
    bars_sr = ax3.bar(x_pos, [random_sr, global_sr], width=0.4, color=["gray", "green"],
                      alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(["Random Init", "Global Model"])
    ax3.set_ylabel("Success Rate")
    ax3.set_title("3. Cold-Start Device 6")
    ax3.set_ylim(0, 1.0)

    # Add value labels
    for bar, val in zip(bars_sr, [random_sr, global_sr]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax3.grid(True, alpha=0.3, axis="y")

    # ---- Panel 4: Flywheel ----
    ax4 = fig.add_subplot(gs[1, 0])
    n_p1 = exp3["n_phase1"]
    p1_final = exp3["phase1_final"]
    p2_final = exp3["phase2_final"]

    x_devices = np.arange(n_p1)
    width = 0.35
    ax4.bar(x_devices - width / 2, p1_final, width, label="Before (3 devices)",
            color="steelblue", alpha=0.8)
    ax4.bar(x_devices + width / 2, p2_final, width, label="After (5 devices)",
            color="coral", alpha=0.8)
    ax4.set_xlabel("Original Device")
    ax4.set_ylabel("Success Rate")
    ax4.set_title("4. Flywheel: Adding Devices Helps Everyone")
    ax4.set_xticks(x_devices)
    ax4.set_xticklabels([f"Dev {i}" for i in range(n_p1)])
    ax4.legend()
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis="y")

    # ---- Panel 5: Tau divergence ----
    ax5 = fig.add_subplot(gs[1, 1])
    device_taus = extract_tau_divergence(aggregator)

    if device_taus:
        sorted_ids = sorted(device_taus.keys())
        # Show first N tau_base values per device as a grouped bar / line
        max_show = min(16, min(len(v) for v in device_taus.values()))
        for did in sorted_ids:
            vals = device_taus[did][:max_show]
            ax5.plot(range(max_show), vals, marker=".", markersize=4,
                     label=did, linewidth=1.2)
        ax5.set_xlabel("tau_base index")
        ax5.set_ylabel("Value")
        ax5.set_title("5. Tau Divergence (Personalization)")
        ax5.legend(fontsize=7, loc="upper right")
    else:
        ax5.text(0.5, 0.5, "No tau data available", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=12)
        ax5.set_title("5. Tau Divergence (Personalization)")
    ax5.grid(True, alpha=0.3)

    # ---- Panel 6: Summary text ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Compute summary metrics
    fl_final_mean = float(fl_curves.mean(axis=0)[-1]) if n_points_fl > 0 else 0.0
    ind_final_mean = float(ind_curves.mean(axis=0)[-1]) if n_points_ind > 0 else 0.0
    speedup = (fl_final_mean - ind_final_mean) / max(abs(ind_final_mean), 1e-6) * 100

    cold_start_improvement = (global_sr - random_sr) / max(random_sr, 1e-6) * 100

    flywheel_improvement = (
        (np.mean(p2_final) - np.mean(p1_final))
        / max(np.mean(p1_final), 1e-6) * 100
    )

    # Tau spread
    if device_taus:
        tau_values = list(device_taus.values())
        if len(tau_values) >= 2:
            # Mean L2 distance between device tau vectors
            dists = []
            for i in range(len(tau_values)):
                for j in range(i + 1, len(tau_values)):
                    min_len = min(len(tau_values[i]), len(tau_values[j]))
                    d = np.linalg.norm(tau_values[i][:min_len] - tau_values[j][:min_len])
                    dists.append(d)
            tau_spread = float(np.mean(dists))
        else:
            tau_spread = 0.0
    else:
        tau_spread = 0.0

    summary = (
        "SpectrAI Federated Learning — Results Summary\n"
        "=" * 46 + "\n\n"
        f"Aggregation method: hybrid_ltc (tau_mix=0.3)\n"
        f"Number of FL devices: {fl_curves.shape[0] if fl_curves.ndim == 2 else 'N/A'}\n\n"
        f"1. CONVERGENCE\n"
        f"   FL final reward:  {fl_final_mean:+.2f}\n"
        f"   Ind final reward: {ind_final_mean:+.2f}\n"
        f"   Improvement:      {speedup:+.1f}%\n\n"
        f"2. COLD-START\n"
        f"   Random init:      {random_sr:.3f} success\n"
        f"   Global model:     {global_sr:.3f} success\n"
        f"   Improvement:      {cold_start_improvement:+.1f}%\n\n"
        f"3. FLYWHEEL\n"
        f"   Before (3 dev):   {np.mean(p1_final):.3f} success\n"
        f"   After  (5 dev):   {np.mean(p2_final):.3f} success\n"
        f"   Improvement:      {flywheel_improvement:+.1f}%\n\n"
        f"4. PERSONALIZATION\n"
        f"   Mean tau spread:  {tau_spread:.4f}\n"
        f"   (Higher = more device-specific adaptation)"
    )

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("SpectrAI Federated Learning: Hybrid LTC-Aware Aggregation",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpectrAI Federated Learning Flywheel Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of UCC MISL 5G dataset (contains CSV traces).",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=10,
        help="Number of FL communication rounds per experiment.",
    )
    parser.add_argument(
        "--local_steps", type=int, default=500,
        help="Local training steps per device per round.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="demo_output",
        help="Directory for output figure and logs.",
    )
    parser.add_argument(
        "--num_channels", type=int, default=5,
        help="Number of radio channels in the environment.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device (cpu or cuda).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch_device = torch.device(args.device)

    print("=" * 60)
    print("  SpectrAI Federated Learning — Flywheel Demo")
    print("=" * 60)
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Rounds:       {args.num_rounds}")
    print(f"  Local steps:  {args.local_steps}")
    print(f"  Channels:     {args.num_channels}")
    print(f"  Device:       {args.device}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60)

    start_time = time.time()

    # Split traces non-IID across 5 devices + 1 holdout
    device_traces, holdout_traces = split_traces_to_devices(
        args.data_dir, num_devices=5, holdout_device=True
    )

    print("\nTrace distribution:")
    for i, traces in enumerate(device_traces):
        print(f"  Device {i}: {len(traces)} traces")
    if holdout_traces:
        print(f"  Holdout (cold-start): {len(holdout_traces)} traces")

    # Experiment 1: Convergence
    exp1 = experiment_convergence(
        data_dir=args.data_dir,
        device_traces=device_traces,
        num_rounds=args.num_rounds,
        local_steps=args.local_steps,
        num_channels=args.num_channels,
        torch_device=torch_device,
    )

    # Experiment 2: Cold-start
    if holdout_traces:
        exp2 = experiment_cold_start(
            data_dir=args.data_dir,
            holdout_traces=holdout_traces,
            aggregator=exp1["aggregator"],
            num_channels=args.num_channels,
            torch_device=torch_device,
        )
    else:
        # Fallback: use first device's traces if no holdout available
        exp2 = experiment_cold_start(
            data_dir=args.data_dir,
            holdout_traces=device_traces[0],
            aggregator=exp1["aggregator"],
            num_channels=args.num_channels,
            torch_device=torch_device,
        )

    # Experiment 3: Flywheel
    num_rounds_p1 = max(1, args.num_rounds // 2)
    num_rounds_p2 = max(1, args.num_rounds - num_rounds_p1)

    exp3 = experiment_flywheel(
        data_dir=args.data_dir,
        device_traces=device_traces,
        num_rounds_phase1=num_rounds_p1,
        num_rounds_phase2=num_rounds_p2,
        local_steps=args.local_steps,
        num_channels=args.num_channels,
        torch_device=torch_device,
    )

    # Plot
    output_path = os.path.join(args.output_dir, "fl_flywheel_results.png")
    plot_results(exp1, exp2, exp3, exp1["aggregator"], output_path)

    elapsed = time.time() - start_time
    print(f"\nDemo completed in {elapsed:.1f}s")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
