#!/usr/bin/env python
"""
Enhanced SAC-LTC vs Traditional Methods: Three Environment Comparison.

Runs the comparison across three environments of increasing difficulty:

  Env 1: Original DSAEnv (occupancy leak, homogeneous channels)
  Env 2: RealisticDSAEnv (no occupancy, noisy, non-stationary, correlated PUs)
  Env 3: HeterogeneousDSAEnv (multi-provider: LEO + 5G FR1 + WiFi 7)

With longer training (configurable, default 10k steps) to give SAC-LTC
a fair chance to converge.

Results are empirical: if traditional methods still win, that is reported.

Usage:
    python run_enhanced_comparison.py --quick    # 1 seed, 2000 steps
    python run_enhanced_comparison.py            # 3 seeds, 10000 steps
    python run_enhanced_comparison.py --full     # 3 seeds, 50000 steps
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import evaluate_agent, train_off_policy
from dsa_env import DSAEnv
from realistic_envs import RealisticDSAEnv, HeterogeneousDSAEnv
from traditional_baselines import TRADITIONAL_BASELINES, TraditionalBaseline


# ======================================================================
# Agent factory (supports all three envs)
# ======================================================================

def make_agent_for_env(agent_type, agent_params, env_cfg, device):
    """Create an agent compatible with the given env config."""
    num_channels = env_cfg["num_channels"]
    num_features = env_cfg["num_features"]
    seq_len = env_cfg["sequence_length"]
    input_dim = num_channels * num_features
    state_shape = (seq_len, input_dim)

    # Traditional baselines
    if agent_type in TRADITIONAL_BASELINES:
        cls, default_params = TRADITIONAL_BASELINES[agent_type]
        merged = {**default_params, **agent_params}
        return cls(num_channels=num_channels, num_features=num_features, **merged)

    # SAC-LTC
    if agent_type == "sac_ltc":
        from sac_ltc_agent import SACLTCAgent
        return SACLTCAgent(
            state_shape=state_shape,
            num_actions=num_channels,
            input_dim=input_dim,
            device=device,
            **agent_params,
        )

    # SAC-LSTM (for reference)
    if agent_type == "sac_lstm":
        from sac_lstm_agent import SACLSTMAgent
        return SACLSTMAgent(
            state_shape=state_shape,
            num_actions=num_channels,
            input_dim=input_dim,
            device=device,
            **agent_params,
        )

    raise ValueError(f"Unknown agent type: {agent_type}")


def make_env(env_type, env_cfg):
    """Create an environment by type."""
    if env_type == "original":
        return DSAEnv(
            num_channels=env_cfg["num_channels"],
            sequence_length=env_cfg["sequence_length"],
            num_features=env_cfg["num_features"],
            pu_on_prob=env_cfg.get("pu_on_prob", 0.3),
            pu_off_prob=env_cfg.get("pu_off_prob", 0.5),
            noise_std=env_cfg.get("noise_std", 0.1),
            max_steps=env_cfg.get("max_steps", 200),
        )
    elif env_type == "realistic":
        return RealisticDSAEnv(
            num_channels=env_cfg["num_channels"],
            sequence_length=env_cfg["sequence_length"],
            num_features=env_cfg["num_features"],
            pu_on_prob=env_cfg.get("pu_on_prob", 0.3),
            pu_off_prob=env_cfg.get("pu_off_prob", 0.5),
            noise_std=env_cfg.get("noise_std", 0.25),
            max_steps=env_cfg.get("max_steps", 200),
            nonstationarity=env_cfg.get("nonstationarity", 0.05),
            correlation=env_cfg.get("correlation", 0.3),
        )
    elif env_type == "heterogeneous":
        return HeterogeneousDSAEnv(
            sequence_length=env_cfg["sequence_length"],
            max_steps=env_cfg.get("max_steps", 200),
        )
    else:
        raise ValueError(f"Unknown env type: {env_type}")


# ======================================================================
# Environment configurations
# ======================================================================

ENV_CONFIGS = {
    "original": {
        "label": "Original DSA (occupancy leak, homogeneous)",
        "num_channels": 10,
        "sequence_length": 16,
        "num_features": 3,
        "pu_on_prob": 0.3,
        "pu_off_prob": 0.5,
        "noise_std": 0.1,
        "max_steps": 200,
    },
    "realistic": {
        "label": "Realistic DSA (no occupancy, noisy, non-stationary)",
        "num_channels": 10,
        "sequence_length": 16,
        "num_features": 2,  # SNR + interference only
        "pu_on_prob": 0.3,
        "pu_off_prob": 0.5,
        "noise_std": 0.25,
        "max_steps": 200,
        "nonstationarity": 0.05,
        "correlation": 0.3,
    },
    "heterogeneous": {
        "label": "Heterogeneous (LEO + 5G FR1 + WiFi 7, no occupancy)",
        "num_channels": 12,   # 4 LEO + 4 FR1 + 4 WiFi7
        "sequence_length": 16,
        "num_features": 3,    # SNR + interference + provider_id
        "max_steps": 200,
    },
}

# Traditional baselines to evaluate
TRADITIONAL_AGENTS = [
    "random", "round_robin", "greedy_sinr", "epsilon_greedy",
    "tdma", "fdma", "proportional_fair", "thompson_sampling",
    "ucb", "whittle_index", "boltzmann",
    "wifi7_mlo", "ofdma", "carrier_aggregation",
]

RL_PARAMS = {
    "hidden_dim": 64,
    "latent_dim": 64,
    "num_layers": 2,
    "dt": 1.0,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 100000,
    "batch_size": 128,
    "learning_starts": 500,
}

RL_LSTM_PARAMS = {
    "hidden_dim": 64,
    "latent_dim": 64,
    "num_layers": 2,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 100000,
    "batch_size": 128,
    "learning_starts": 500,
}


def welch_t_test(a, b):
    a, b = np.array(a, float), np.array(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se < 1e-12:
        return 0.0, 1.0
    t = (ma - mb) / se
    from math import erfc, sqrt
    return float(t), float(erfc(abs(t) / sqrt(2)))


# ======================================================================
# Run one environment comparison
# ======================================================================

def run_env_comparison(env_type, env_cfg, seeds, training_steps, eval_episodes,
                       output_dir, device):
    """Run all agents on one environment type."""
    env_label = env_cfg["label"]
    env_dir = os.path.join(output_dir, env_type)
    os.makedirs(env_dir, exist_ok=True)

    print(f"\n{'='*80}", flush=True)
    print(f"  ENV: {env_label}", flush=True)
    print(f"  Channels={env_cfg['num_channels']}  Features={env_cfg['num_features']}  Seeds={len(seeds)}", flush=True)
    print(f"{'='*80}", flush=True)

    all_results = {}

    # ── Traditional baselines (no training) ──
    print(f"\n  Traditional baselines:", flush=True)
    for agent_name in TRADITIONAL_AGENTS:
        all_results[agent_name] = defaultdict(list)
        t0 = time.time()

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = make_agent_for_env(agent_name, {}, env_cfg, device)
            eval_env = make_env(env_type, env_cfg)
            metrics = evaluate_agent(agent, agent_name, eval_env, eval_episodes)
            metrics.pop("episode_rewards", None)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[agent_name][k].append(v)

        sr = np.mean(all_results[agent_name]["success_rate"])
        elapsed = time.time() - t0
        print(f"    {agent_name:25s} Success={sr:.2%}  ({elapsed:.1f}s)", flush=True)

    # ── RL agents (train + evaluate) ──
    for rl_name, rl_params in [("sac_ltc", RL_PARAMS), ("sac_lstm", RL_LSTM_PARAMS)]:
        all_results[rl_name] = defaultdict(list)
        print(f"\n  {rl_name} (training {training_steps} steps):", flush=True)

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            run_dir = os.path.join(env_dir, f"{rl_name}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            agent = make_agent_for_env(rl_name, rl_params, env_cfg, device)
            train_env = make_env(env_type, env_cfg)

            t0 = time.time()
            train_off_policy(
                agent, train_env,
                total_steps=training_steps,
                log_interval=max(training_steps // 5, 100),
                eval_interval=max(training_steps // 2, 500),
                eval_episodes=10,
                run_dir=run_dir,
            )
            train_time = time.time() - t0

            eval_env = make_env(env_type, env_cfg)
            metrics = evaluate_agent(agent, rl_name, eval_env, eval_episodes)
            metrics.pop("episode_rewards", None)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[rl_name][k].append(v)

            sr = metrics["success_rate"]
            print(f"    seed{seed}: Success={sr:.2%}  Train={train_time:.0f}s", flush=True)

    # ── Aggregate ──
    summary = {}
    for agent_name, metric_lists in all_results.items():
        summary[agent_name] = {}
        for k, vals in metric_lists.items():
            arr = np.array(vals)
            summary[agent_name][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in vals],
            }

    # Save
    with open(os.path.join(env_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary, all_results


# ======================================================================
# Print comparison table
# ======================================================================

def print_comparison(env_type, env_label, summary, all_results, seeds):
    METRICS = [
        ("success_rate", "Success%", True),
        ("collision_rate", "Collisn%", False),
        ("spectral_efficiency", "Spec.Eff", True),
        ("mean_reward", "Reward", True),
        ("mean_inference_ms", "Infer(ms)", False),
    ]

    print(f"\n{'='*100}", flush=True)
    print(f"  {env_label}  ({len(seeds)} seeds)", flush=True)
    print(f"{'='*100}", flush=True)

    header = f"{'Agent':25s} {'Cat':12s}"
    for _, label, _ in METRICS:
        header += f" {label:>14s}"
    print(header, flush=True)
    print("-" * 100, flush=True)

    # Sort by success rate descending
    sorted_agents = sorted(
        summary.keys(),
        key=lambda a: summary[a].get("success_rate", {}).get("mean", 0),
        reverse=True,
    )

    # Find best per metric
    best = {}
    for metric, _, higher_better in METRICS:
        vals = [(summary[a].get(metric, {}).get("mean", 0), a) for a in summary]
        if vals:
            best[metric] = max(vals, key=lambda x: x[0])[1] if higher_better else min(vals, key=lambda x: x[0])[1]

    ltc_results = all_results.get("sac_ltc", {})

    for agent_name in sorted_agents:
        cat = "rl" if agent_name in ("sac_ltc", "sac_lstm") else "traditional"
        row = f"{agent_name:25s} {cat:12s}"

        for metric, _, _ in METRICS:
            m = summary[agent_name].get(metric, {})
            mean = m.get("mean", 0)
            std = m.get("std", 0)
            marker = "*" if best.get(metric) == agent_name else " "

            if metric in ("success_rate", "collision_rate"):
                row += f" {mean:.2%}±{std:.2%}{marker}"
            elif metric == "mean_inference_ms":
                row += f" {mean:>8.3f}±{std:.3f}{marker}"
            else:
                row += f" {mean:>8.3f}±{std:.3f}{marker}"

        # Delta vs SAC-LTC
        if agent_name != "sac_ltc" and "sac_ltc" in summary:
            ltc_sr = summary["sac_ltc"]["success_rate"]["mean"]
            this_sr = summary[agent_name]["success_rate"]["mean"]
            delta = (ltc_sr - this_sr) * 100  # pp
            ltc_vals = ltc_results.get("success_rate", [])
            this_vals = all_results.get(agent_name, {}).get("success_rate", [])
            if ltc_vals and this_vals:
                _, p = welch_t_test(ltc_vals, this_vals)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            else:
                sig = ""
            row += f"  LTC {delta:+.1f}pp{sig}"

        print(row, flush=True)

    print("-" * 100, flush=True)
    print("  * = best for metric  ** = p<0.01  * = p<0.05 (Welch's t-test)", flush=True)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced 3-Environment Comparison: SAC-LTC vs Traditional Methods"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 1 seed, 2000 steps, 10 eval episodes")
    parser.add_argument("--full", action="store_true",
                        help="Full: 3 seeds, 50000 steps, 50 eval episodes")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="enhanced_comparison_results")
    parser.add_argument("--envs", nargs="*", default=None,
                        help="Which envs to run (original, realistic, heterogeneous)")
    args = parser.parse_args()

    if args.quick:
        seeds, steps, eval_eps = [1], 2000, 10
    elif args.full:
        seeds, steps, eval_eps = list(range(1, 4)), 50000, 50
    else:
        seeds = list(range(1, args.seeds + 1))
        steps = args.steps
        eval_eps = args.eval_episodes

    device = torch.device("cpu")
    env_types = args.envs or ["original", "realistic", "heterogeneous"]

    print(f"Enhanced Comparison Benchmark", flush=True)
    print(f"  Seeds: {seeds}  Steps: {steps}  Eval: {eval_eps} episodes", flush=True)
    print(f"  Environments: {env_types}", flush=True)

    all_summaries = {}

    for env_type in env_types:
        env_cfg = ENV_CONFIGS[env_type]
        summary, raw_results = run_env_comparison(
            env_type, env_cfg, seeds, steps, eval_eps,
            args.output_dir, device,
        )
        all_summaries[env_type] = summary
        print_comparison(env_type, env_cfg["label"], summary, raw_results, seeds)

    # Save combined results
    combined_path = os.path.join(args.output_dir, "combined_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results saved -> {combined_path}", flush=True)

    # Print cross-environment summary
    if "sac_ltc" in all_summaries.get(env_types[0], {}):
        print(f"\n{'='*80}", flush=True)
        print(f"  CROSS-ENVIRONMENT SUMMARY: SAC-LTC Success Rate", flush=True)
        print(f"{'='*80}", flush=True)
        for env_type in env_types:
            s = all_summaries.get(env_type, {})
            if "sac_ltc" in s:
                ltc = s["sac_ltc"]["success_rate"]["mean"]
                # Best traditional
                best_trad_name = max(
                    [a for a in s if a not in ("sac_ltc", "sac_lstm")],
                    key=lambda a: s[a].get("success_rate", {}).get("mean", 0),
                )
                best_trad = s[best_trad_name]["success_rate"]["mean"]
                delta = (ltc - best_trad) * 100
                direction = "ahead" if delta > 0 else "behind"
                print(
                    f"  {ENV_CONFIGS[env_type]['label'][:50]:50s}  "
                    f"LTC={ltc:.2%}  Best trad={best_trad_name}({best_trad:.2%})  "
                    f"LTC {abs(delta):.1f}pp {direction}",
                    flush=True,
                )
        print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
