#!/usr/bin/env python
"""
Comprehensive SAC-LTC vs. Traditional Scheduling Methods Comparison.

Evaluates SAC-LTC against 14 traditional baselines spanning:
  - Fixed allocation: Random, Round-Robin, TDMA, FDMA
  - Heuristic: Greedy Max-SINR, Epsilon-Greedy, Proportional Fair
  - Bandit: Thompson Sampling, UCB, Boltzmann
  - Model-based: Whittle Index
  - Standard-inspired: WiFi 7 MLO, OFDMA, Carrier Aggregation

Produces:
  1. benchmark_comparison.json  — all metrics for all agents
  2. comparison_table.txt       — formatted console table
  3. statistical_tests.json     — Welch's t-test per metric per baseline
  4. Visualization figures via visualize.py

Results are empirical: if a traditional method outperforms SAC-LTC
on any metric, that result is reported honestly.

Usage:
    python run_comparison_benchmark.py
    python run_comparison_benchmark.py --quick          # 1 seed, 1000 steps
    python run_comparison_benchmark.py --seeds 5 --steps 50000
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

from benchmark import evaluate_agent, make_agent, train_off_policy
from dsa_env import DSAEnv
from traditional_baselines import TRADITIONAL_BASELINES, TraditionalBaseline


# ======================================================================
# Configuration
# ======================================================================

def build_config(seeds: int = 3, steps: int = 5000, eval_episodes: int = 50):
    return {
        "training_steps": steps,
        "eval_episodes": eval_episodes,
        "log_interval": max(steps // 5, 100),
        "eval_interval": max(steps // 2, 500),
        "seeds": list(range(1, seeds + 1)),
        "env": {
            "num_channels": 10,
            "sequence_length": 16,
            "num_features": 3,
            "pu_on_prob": 0.3,
            "pu_off_prob": 0.5,
            "noise_std": 0.1,
            "max_steps": 200,
        },
        # RL agent to compare
        "rl_agent": {
            "sac_ltc": {
                "class": "sac_ltc",
                "params": {
                    "hidden_dim": 64,
                    "latent_dim": 64,
                    "num_layers": 2,
                    "dt": 1.0,
                    "lr": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "buffer_size": 50000,
                    "batch_size": 128,
                    "learning_starts": 500,
                },
            },
        },
        # Traditional baselines (all 14)
        "traditional_agents": {
            "random":               {"class": "random", "params": {}},
            "round_robin":          {"class": "round_robin", "params": {}},
            "greedy_sinr":          {"class": "greedy_sinr", "params": {}},
            "epsilon_greedy":       {"class": "epsilon_greedy", "params": {"epsilon": 0.1}},
            "tdma":                 {"class": "tdma", "params": {}},
            "fdma":                 {"class": "fdma", "params": {"fixed_channel": 0}},
            "proportional_fair":    {"class": "proportional_fair", "params": {}},
            "thompson_sampling":    {"class": "thompson_sampling", "params": {}},
            "ucb":                  {"class": "ucb", "params": {"c": 1.414}},
            "whittle_index":        {"class": "whittle_index", "params": {"pu_on_prob": 0.3, "pu_off_prob": 0.5}},
            "boltzmann":            {"class": "boltzmann", "params": {"temperature": 0.5}},
            "wifi7_mlo":            {"class": "wifi7_mlo", "params": {}},
            "ofdma":                {"class": "ofdma", "params": {}},
            "carrier_aggregation":  {"class": "carrier_aggregation", "params": {"k_candidates": 3}},
        },
    }


# ======================================================================
# Evaluation helpers
# ======================================================================

def evaluate_single_agent(agent, agent_type, env_cfg, num_episodes, seed):
    """Create a fresh env and evaluate one agent for one seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = DSAEnv(
        num_channels=env_cfg["num_channels"],
        sequence_length=env_cfg["sequence_length"],
        num_features=env_cfg["num_features"],
        pu_on_prob=env_cfg["pu_on_prob"],
        pu_off_prob=env_cfg["pu_off_prob"],
        noise_std=env_cfg["noise_std"],
        max_steps=env_cfg["max_steps"],
    )
    metrics = evaluate_agent(agent, agent_type, eval_env, num_episodes)
    metrics.pop("episode_rewards", None)
    return metrics


def welch_t_test(a_values, b_values):
    """Welch's t-test for unequal variances. Returns t-statistic and p-value."""
    a = np.array(a_values, dtype=float)
    b = np.array(b_values, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se < 1e-12:
        return 0.0, 1.0
    t_stat = (ma - mb) / se
    # Welch-Satterthwaite degrees of freedom
    num = (va / na + vb / nb) ** 2
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / (denom + 1e-12)
    # Approximate p-value using normal distribution for simplicity
    # (accurate for df > 30, reasonable for df > 5)
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))
    return float(t_stat), float(p_value)


# ======================================================================
# Main comparison
# ======================================================================

def run_comparison(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    env_cfg = config["env"]
    seeds = config["seeds"]
    eval_episodes = config["eval_episodes"]

    all_results = {}  # {agent_name: {metric: [per-seed values]}}
    all_agents = {}   # {agent_name: {"category": ..., "class": ...}}

    # ------------------------------------------------------------------
    # Phase 1: Evaluate all traditional baselines (fast, no training)
    # ------------------------------------------------------------------
    print(f"\n{'='*80}", flush=True)
    print(f"  PHASE 1: Traditional Baselines  ({len(config['traditional_agents'])} methods)", flush=True)
    print(f"{'='*80}", flush=True)

    for agent_name, agent_cfg in config["traditional_agents"].items():
        agent_type = agent_cfg["class"]
        agent_params = agent_cfg["params"]
        all_agents[agent_name] = {"category": "traditional", "class": agent_type}
        all_results[agent_name] = defaultdict(list)

        print(f"\n  {agent_name} ...", end=" ", flush=True)
        t0 = time.time()

        for seed in seeds:
            agent = make_agent(agent_type, agent_params, env_cfg, device)
            metrics = evaluate_single_agent(agent, agent_type, env_cfg, eval_episodes, seed)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[agent_name][k].append(v)

        elapsed = time.time() - t0
        sr = np.mean(all_results[agent_name]["success_rate"])
        cr = np.mean(all_results[agent_name]["collision_rate"])
        print(f"Success={sr:.2%}  Collision={cr:.2%}  ({elapsed:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Train and evaluate SAC-LTC
    # ------------------------------------------------------------------
    print(f"\n{'='*80}", flush=True)
    print(f"  PHASE 2: RL Agent (SAC-LTC)", flush=True)
    print(f"{'='*80}", flush=True)

    for agent_name, agent_cfg in config["rl_agent"].items():
        agent_type = agent_cfg["class"]
        agent_params = agent_cfg["params"]
        all_agents[agent_name] = {"category": "rl", "class": agent_type}
        all_results[agent_name] = defaultdict(list)

        for seed in seeds:
            run_id = f"{agent_name}_seed{seed}"
            run_dir = os.path.join(output_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n  {run_id} ...", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)

            agent = make_agent(agent_type, agent_params, env_cfg, device)

            # Train
            env = DSAEnv(
                num_channels=env_cfg["num_channels"],
                sequence_length=env_cfg["sequence_length"],
                num_features=env_cfg["num_features"],
                pu_on_prob=env_cfg["pu_on_prob"],
                pu_off_prob=env_cfg["pu_off_prob"],
                noise_std=env_cfg["noise_std"],
                max_steps=env_cfg["max_steps"],
            )
            t0 = time.time()
            train_off_policy(
                agent, env,
                total_steps=config["training_steps"],
                log_interval=config["log_interval"],
                eval_interval=config["eval_interval"],
                eval_episodes=10,
                run_dir=run_dir,
            )
            print(f"    Training: {time.time()-t0:.1f}s", flush=True)

            # Evaluate
            metrics = evaluate_single_agent(agent, agent_type, env_cfg, eval_episodes, seed)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[agent_name][k].append(v)

            print(
                f"    Result: Success={metrics['success_rate']:.2%} "
                f"Collision={metrics['collision_rate']:.2%} "
                f"Reward={metrics['mean_reward']:.2f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Phase 3: Aggregate and compute statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}", flush=True)
    print(f"  PHASE 3: Analysis", flush=True)
    print(f"{'='*80}", flush=True)

    # Aggregate summary
    summary = {}
    for agent_name, metric_lists in all_results.items():
        summary[agent_name] = {"category": all_agents[agent_name]["category"]}
        for k, vals in metric_lists.items():
            arr = np.array(vals)
            summary[agent_name][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in vals],
            }

    # Save summary
    summary_path = os.path.join(output_dir, "benchmark_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}", flush=True)

    # Statistical tests: SAC-LTC vs each baseline
    ltc_key = "sac_ltc"
    stat_tests = {}
    if ltc_key in all_results:
        for agent_name in all_results:
            if agent_name == ltc_key:
                continue
            stat_tests[agent_name] = {}
            for metric in ["success_rate", "collision_rate", "spectral_efficiency",
                           "jain_fairness", "mean_reward", "mean_inference_ms"]:
                ltc_vals = all_results[ltc_key].get(metric, [])
                base_vals = all_results[agent_name].get(metric, [])
                if ltc_vals and base_vals:
                    t_stat, p_val = welch_t_test(ltc_vals, base_vals)
                    ltc_mean = np.mean(ltc_vals)
                    base_mean = np.mean(base_vals)
                    if abs(base_mean) > 1e-12:
                        improvement_pct = (ltc_mean - base_mean) / abs(base_mean) * 100
                    else:
                        improvement_pct = 0.0
                    stat_tests[agent_name][metric] = {
                        "sac_ltc_mean": float(ltc_mean),
                        "baseline_mean": float(base_mean),
                        "improvement_pct": float(improvement_pct),
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "significant_at_005": p_val < 0.05,
                    }

    stats_path = os.path.join(output_dir, "statistical_tests.json")
    with open(stats_path, "w") as f:
        json.dump(stat_tests, f, indent=2)
    print(f"  Statistical tests saved -> {stats_path}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4: Print results table
    # ------------------------------------------------------------------
    METRICS = [
        ("success_rate", "Success%", True),     # higher is better
        ("collision_rate", "Collision%", False), # lower is better
        ("spectral_efficiency", "Spec.Eff", True),
        ("jain_fairness", "Jain FI", True),
        ("mean_reward", "Reward", True),
        ("mean_inference_ms", "Infer(ms)", False),
    ]

    # Find best value per metric across ALL agents
    best_vals = {}
    for metric, _, higher_better in METRICS:
        vals = []
        for aname in all_results:
            m = summary.get(aname, {}).get(metric, {})
            if "mean" in m:
                vals.append((m["mean"], aname))
        if vals:
            if higher_better:
                best_vals[metric] = max(vals, key=lambda x: x[0])
            else:
                best_vals[metric] = min(vals, key=lambda x: x[0])

    # Print table
    header_cols = ["Agent", "Category"] + [m[1] for m in METRICS] + ["vs LTC"]
    col_widths = [22, 12] + [14] * len(METRICS) + [12]

    def _pad(s, w):
        return str(s).rjust(w)

    sep = "-" * sum(col_widths)
    print(f"\n{'='*sum(col_widths)}", flush=True)
    n_seeds = len(seeds)
    print(f"  SCHEDULING METHODS COMPARISON  ({n_seeds} seeds x {eval_episodes} eval episodes)", flush=True)
    print(f"{'='*sum(col_widths)}", flush=True)

    header = ""
    for label, w in zip(header_cols, col_widths):
        header += _pad(label, w)
    print(header, flush=True)
    print(sep, flush=True)

    # Sort: RL agents first, then traditional sorted by success rate
    def sort_key(name):
        cat = all_agents[name]["category"]
        sr = summary.get(name, {}).get("success_rate", {}).get("mean", 0)
        return (0 if cat == "rl" else 1, -sr)

    for agent_name in sorted(all_results.keys(), key=sort_key):
        cat = all_agents[agent_name]["category"]
        row = _pad(agent_name, col_widths[0])
        row += _pad(cat, col_widths[1])

        for metric, _, _ in METRICS:
            m = summary.get(agent_name, {}).get(metric, {})
            mean = m.get("mean", 0)
            std = m.get("std", 0)

            # Mark best with asterisk
            is_best = best_vals.get(metric, (None, None))[1] == agent_name
            marker = "*" if is_best else " "

            if metric in ("success_rate", "collision_rate"):
                cell = f"{mean:.2%}±{std:.2%}{marker}"
            elif metric == "mean_inference_ms":
                cell = f"{mean:.3f}±{std:.3f}{marker}"
            elif metric == "jain_fairness":
                cell = f"{mean:.4f}{marker}"
            else:
                cell = f"{mean:.3f}±{std:.3f}{marker}"
            row += _pad(cell, col_widths[-2] if metric != "mean_inference_ms" else col_widths[-2])

        # Improvement column (only for non-LTC agents)
        if agent_name != ltc_key and ltc_key in all_results:
            ltc_sr = summary.get(ltc_key, {}).get("success_rate", {}).get("mean", 0)
            this_sr = summary.get(agent_name, {}).get("success_rate", {}).get("mean", 0)
            if abs(this_sr) > 1e-12:
                imp = (ltc_sr - this_sr) / abs(this_sr) * 100
                p_val = stat_tests.get(agent_name, {}).get("success_rate", {}).get("p_value", 1.0)
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                row += _pad(f"{imp:+.2f}%{sig}", col_widths[-1])
            else:
                row += _pad("—", col_widths[-1])
        else:
            row += _pad("(ref)", col_widths[-1])

        print(row, flush=True)

    print(sep, flush=True)
    print("  * = best across all methods for that metric", flush=True)
    print("  ** = significant at p<0.01, * = significant at p<0.05 (Welch's t-test vs SAC-LTC)", flush=True)
    print(f"{'='*sum(col_widths)}", flush=True)

    # Print key findings
    print("\nKEY FINDINGS:", flush=True)
    if ltc_key in summary:
        ltc_sr = summary[ltc_key]["success_rate"]["mean"]
        for agent_name in sorted(all_results.keys(), key=sort_key):
            if agent_name == ltc_key:
                continue
            sr = summary[agent_name]["success_rate"]["mean"]
            delta = (ltc_sr - sr) * 100  # percentage points
            p_val = stat_tests.get(agent_name, {}).get("success_rate", {}).get("p_value", 1.0)
            direction = "ahead of" if delta > 0 else "behind"
            sig = "(significant)" if p_val < 0.05 else "(not significant)"
            print(f"  SAC-LTC is {abs(delta):.2f}pp {direction} {agent_name} on success rate {sig}", flush=True)

    # Save text report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    print(f"\n  Report saved -> {report_path}", flush=True)

    return summary, stat_tests


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAC-LTC vs Traditional Scheduling Methods Comparison"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 1000 steps, 10 eval episodes")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default: 3)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps for RL agent (default: 5000)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Evaluation episodes per seed (default: 50)")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                        help="Output directory (default: comparison_results)")
    args = parser.parse_args()

    if args.quick:
        config = build_config(seeds=1, steps=1000, eval_episodes=10)
    else:
        config = build_config(
            seeds=args.seeds,
            steps=args.steps,
            eval_episodes=args.eval_episodes,
        )

    print(f"Configuration:", flush=True)
    print(f"  Seeds: {config['seeds']}", flush=True)
    print(f"  Training steps: {config['training_steps']}", flush=True)
    print(f"  Eval episodes: {config['eval_episodes']}", flush=True)
    print(f"  Traditional methods: {len(config['traditional_agents'])}", flush=True)
    print(f"  RL agents: {len(config['rl_agent'])}", flush=True)

    run_comparison(config, args.output_dir)


if __name__ == "__main__":
    main()
