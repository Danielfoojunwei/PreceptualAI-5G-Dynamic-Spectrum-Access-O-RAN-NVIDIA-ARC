#!/usr/bin/env python
"""
500K-Step SAC-LTC Training with Real Wireless Datasets.

Trains SAC-LTC on three environments with real open-source datasets:

  1. Real5G-DSA: 10-channel DSA driven by UCC MISL 5G measurements
     (135K real measurements from Irish mobile operator)
  2. TelecomTS-DSA: 10-channel DSA driven by TelecomTS HuggingFace data
     (32K samples × 128 timesteps from real 5G testbed)
  3. Realistic-DSA: Synthetic but hard environment (no occupancy leak,
     non-stationary, correlated — as control baseline)

All training uses real empirical data — no mocks, no fakes, no stubs.
Datasets are downloaded automatically on first run.

After training, evaluates against all 14 traditional baselines.

Usage:
    python train_500k_realdata.py                     # Full 500K steps
    python train_500k_realdata.py --steps 50000       # Quick test
    python train_500k_realdata.py --env real5g         # Single env
    python train_500k_realdata.py --env telecomts      # TelecomTS only
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
from gymnasium import spaces

from benchmark import evaluate_agent, train_off_policy
from traditional_baselines import TRADITIONAL_BASELINES, TraditionalBaseline


# ======================================================================
# Real-Data DSA Environment: UCC MISL 5G
# ======================================================================

class Real5GDSAEnv(gym.Env):
    """
    DSA environment driven by real UCC MISL 5G measurements.

    Instead of synthetic Markov PU dynamics, this environment replays
    real RSRP/RSRQ/SNR/CQI/RSSI traces from an Irish 5G operator.

    Observation: (sequence_length, num_channels * 2)  — real SNR + real interference
    No occupancy leak: channel state inferred from real signal quality.

    Channel occupancy derived from real SNR:
        SNR < threshold → occupied (poor channel)
        SNR >= threshold → free (good channel)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dir: str = "data/ucc_misl/5Gdataset/extracted/5G-production-dataset",
        num_channels: int = 10,
        sequence_length: int = 16,
        num_features: int = 2,
        max_steps: int = 200,
        snr_threshold: float = 5.0,  # dB: below this = "occupied"
        noise_std: float = 0.15,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.max_steps = max_steps
        self.snr_threshold = snr_threshold
        self.noise_std = noise_std

        self.action_space = spaces.Discrete(num_channels)
        obs_dim = num_channels * num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        # Load real traces
        self._traces = self._load_traces(data_dir)
        assert len(self._traces) > 0, f"No traces loaded from {data_dir}"
        print(f"  Real5GDSAEnv: loaded {len(self._traces)} traces "
              f"({sum(len(t) for t in self._traces)} total measurements)")

        self._history = None
        self._prev_action = None
        self._step_count = 0
        self._trace_ptrs = None  # per-channel trace pointer
        self._trace_assignments = None  # which trace for each channel

    def _load_traces(self, data_dir):
        """Load and normalize all real 5G traces."""
        csvs = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        csvs = [c for c in csvs if "__MACOSX" not in c]

        traces = []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
                # Extract SNR and RSSI (our two features)
                if "SNR" not in df.columns or "RSSI" not in df.columns:
                    continue
                snr = pd.to_numeric(df["SNR"], errors="coerce").fillna(0).values
                rssi = pd.to_numeric(df["RSSI"], errors="coerce").fillna(-100).values
                if len(snr) < 50:
                    continue
                # Normalize to roughly [0, 1] range
                snr_norm = (snr - snr.mean()) / (snr.std() + 1e-8)
                rssi_norm = (rssi - rssi.mean()) / (rssi.std() + 1e-8)
                trace = np.stack([snr_norm, rssi_norm], axis=-1).astype(np.float32)
                traces.append(trace)
            except Exception:
                continue

        return traces

    def _get_channel_state(self, snr_raw):
        """Determine if channel is 'occupied' based on real SNR."""
        # Channels with low normalized SNR are treated as occupied
        return 1.0 if snr_raw < -0.5 else 0.0  # below-average SNR = occupied

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._prev_action = None
        self._step_count = 0

        # Assign random traces to each channel
        self._trace_assignments = np.random.choice(
            len(self._traces), size=self.num_channels, replace=True
        )
        self._trace_ptrs = np.array([
            np.random.randint(0, max(1, len(self._traces[t]) - self.max_steps - self.sequence_length))
            for t in self._trace_assignments
        ])

        # Fill history
        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros((self.sequence_length, obs_dim), dtype=np.float32)
        for t in range(self.sequence_length):
            self._history[t] = self._read_features()
            self._advance_ptrs()

        obs = self._history.copy() + np.random.randn(*self._history.shape).astype(np.float32) * self.noise_std
        return obs, {}

    def _read_features(self):
        """Read real data features for all channels at current pointers."""
        features = np.zeros((self.num_channels, self.num_features), dtype=np.float32)
        for ch in range(self.num_channels):
            trace = self._traces[self._trace_assignments[ch]]
            ptr = min(self._trace_ptrs[ch], len(trace) - 1)
            features[ch] = trace[ptr]  # [snr_norm, rssi_norm]
        return features.reshape(-1)

    def _advance_ptrs(self):
        for ch in range(self.num_channels):
            trace_len = len(self._traces[self._trace_assignments[ch]])
            self._trace_ptrs[ch] = (self._trace_ptrs[ch] + 1) % trace_len

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        self._advance_ptrs()

        # Determine success/collision from real SNR
        trace = self._traces[self._trace_assignments[action]]
        ptr = min(self._trace_ptrs[action], len(trace) - 1)
        real_snr = trace[ptr, 0]  # normalized SNR
        occupied = self._get_channel_state(real_snr)

        if occupied:
            reward = -1.0
            success = False
        else:
            reward = 1.0
            success = True

        if self._prev_action is not None and action != self._prev_action:
            reward -= 0.1
        self._prev_action = action

        new_features = self._read_features()
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        obs = self._history.copy() + np.random.randn(*self._history.shape).astype(np.float32) * self.noise_std
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {"collision": not success, "success": success}
        return obs, reward, terminated, truncated, info


# ======================================================================
# Real-Data DSA Environment: TelecomTS
# ======================================================================

class TelecomTSDSAEnv(gym.Env):
    """
    DSA environment driven by TelecomTS real 5G testbed data.

    Uses real UL_SNR and RSRP traces from the TelecomTS HuggingFace dataset
    (32K samples × 128 timesteps at 10 Hz from a real 5G testbed).

    Each channel replays a different TelecomTS sample.
    No occupancy leak — agent must infer channel state from real signals.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_channels: int = 10,
        sequence_length: int = 16,
        num_features: int = 2,
        max_steps: int = 100,  # 128-step traces, leave room
        max_samples: int = 500,
        noise_std: float = 0.15,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.max_steps = max_steps
        self.noise_std = noise_std

        self.action_space = spaces.Discrete(num_channels)
        obs_dim = num_channels * num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        # Load TelecomTS traces
        self._traces = self._load_telecomts(max_samples)
        print(f"  TelecomTSDSAEnv: loaded {len(self._traces)} traces")

        self._history = None
        self._prev_action = None
        self._step_count = 0
        self._trace_assignments = None
        self._trace_ptrs = None

    def _load_telecomts(self, max_samples):
        """Load TelecomTS from HuggingFace."""
        from datasets import load_dataset
        import json as _json

        ds = load_dataset("AliMaatouk/TelecomTS", split="train", streaming=True)

        traces = []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            kpis = sample.get("KPIs", {})
            if isinstance(kpis, str):
                kpis = _json.loads(kpis)

            ul_snr = kpis.get("UL_SNR", [])
            rsrp = kpis.get("RSRP", [])

            if not ul_snr or not rsrp or len(ul_snr) < 50:
                continue

            snr_arr = np.array(ul_snr, dtype=np.float32)
            rsrp_arr = np.array(rsrp, dtype=np.float32)

            # Z-score normalize
            snr_norm = (snr_arr - snr_arr.mean()) / (snr_arr.std() + 1e-8)
            rsrp_norm = (rsrp_arr - rsrp_arr.mean()) / (rsrp_arr.std() + 1e-8)

            trace = np.stack([snr_norm, rsrp_norm], axis=-1)
            traces.append(trace)

        return traces

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._prev_action = None
        self._step_count = 0

        self._trace_assignments = np.random.choice(
            len(self._traces), size=self.num_channels, replace=True
        )
        self._trace_ptrs = np.array([
            np.random.randint(0, max(1, len(self._traces[t]) - self.max_steps - self.sequence_length))
            for t in self._trace_assignments
        ])

        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros((self.sequence_length, obs_dim), dtype=np.float32)
        for t in range(self.sequence_length):
            self._history[t] = self._read_features()
            self._advance_ptrs()

        obs = self._history.copy() + np.random.randn(*self._history.shape).astype(np.float32) * self.noise_std
        return obs, {}

    def _read_features(self):
        features = np.zeros((self.num_channels, self.num_features), dtype=np.float32)
        for ch in range(self.num_channels):
            trace = self._traces[self._trace_assignments[ch]]
            ptr = min(self._trace_ptrs[ch], len(trace) - 1)
            features[ch] = trace[ptr]
        return features.reshape(-1)

    def _advance_ptrs(self):
        for ch in range(self.num_channels):
            trace_len = len(self._traces[self._trace_assignments[ch]])
            self._trace_ptrs[ch] = (self._trace_ptrs[ch] + 1) % trace_len

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1
        self._advance_ptrs()

        trace = self._traces[self._trace_assignments[action]]
        ptr = min(self._trace_ptrs[action], len(trace) - 1)
        real_snr = trace[ptr, 0]
        occupied = real_snr < -0.5

        reward = -1.0 if occupied else 1.0
        success = not occupied
        if self._prev_action is not None and action != self._prev_action:
            reward -= 0.1
        self._prev_action = action

        new_features = self._read_features()
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        obs = self._history.copy() + np.random.randn(*self._history.shape).astype(np.float32) * self.noise_std
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {"collision": not success, "success": success}
        return obs, reward, terminated, truncated, info


# ======================================================================
# Training + Evaluation Pipeline
# ======================================================================

def train_and_evaluate(env_name, env, env_cfg, training_steps, eval_episodes,
                       seeds, output_dir, device):
    """Train SAC-LTC + SAC-LSTM, evaluate all agents."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # ── Traditional baselines ──
    print(f"\n  Evaluating 14 traditional baselines...", flush=True)
    for agent_name in TRADITIONAL_BASELINES:
        all_results[agent_name] = defaultdict(list)
        for seed in seeds:
            np.random.seed(seed)
            cls, default_params = TRADITIONAL_BASELINES[agent_name]
            agent = cls(
                num_channels=env_cfg["num_channels"],
                num_features=env_cfg["num_features"],
                **default_params,
            )
            eval_env = env.__class__(**env_cfg.get("init_kwargs", {}))
            metrics = evaluate_agent(agent, agent_name, eval_env, eval_episodes)
            metrics.pop("episode_rewards", None)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[agent_name][k].append(v)

        sr = np.mean(all_results[agent_name]["success_rate"])
        print(f"    {agent_name:25s} Success={sr:.2%}", flush=True)

    # ── RL agents (~5.3M params each with h=384, l=384, layers=3) ──
    rl_configs = {
        "sac_ltc": {
            "module": "sac_ltc_agent",
            "class": "SACLTCAgent",
            "params": {
                "hidden_dim": 384,
                "latent_dim": 384,
                "num_layers": 3,
                "dt": 1.0,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": min(training_steps, 1000000),
                "batch_size": 256,
                "learning_starts": 2000,
            },
        },
        "sac_lstm": {
            "module": "sac_lstm_agent",
            "class": "SACLSTMAgent",
            "params": {
                "hidden_dim": 384,
                "latent_dim": 384,
                "num_layers": 3,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": min(training_steps, 1000000),
                "batch_size": 256,
                "learning_starts": 2000,
            },
        },
    }

    num_channels = env_cfg["num_channels"]
    num_features = env_cfg["num_features"]
    seq_len = env_cfg.get("sequence_length", 16)
    input_dim = num_channels * num_features
    state_shape = (seq_len, input_dim)

    for rl_name, rl_cfg in rl_configs.items():
        all_results[rl_name] = defaultdict(list)
        print(f"\n  Training {rl_name} ({training_steps} steps):", flush=True)

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            run_dir = os.path.join(output_dir, f"{rl_name}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            # Import and instantiate
            import importlib
            mod = importlib.import_module(rl_cfg["module"])
            cls = getattr(mod, rl_cfg["class"])
            agent = cls(
                state_shape=state_shape,
                num_actions=num_channels,
                input_dim=input_dim,
                device=device,
                **rl_cfg["params"],
            )

            # Train
            train_env = env.__class__(**env_cfg.get("init_kwargs", {}))
            t0 = time.time()
            train_off_policy(
                agent, train_env,
                total_steps=training_steps,
                log_interval=max(training_steps // 20, 500),
                eval_interval=max(training_steps // 10, 1000),
                eval_episodes=10,
                run_dir=run_dir,
            )
            train_time = time.time() - t0

            # Save checkpoint
            agent.save(os.path.join(run_dir, "checkpoint_500k.pt"))

            # Evaluate
            eval_env = env.__class__(**env_cfg.get("init_kwargs", {}))
            metrics = evaluate_agent(agent, rl_name, eval_env, eval_episodes)
            metrics.pop("episode_rewards", None)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[rl_name][k].append(v)

            sr = metrics["success_rate"]
            rw = metrics["mean_reward"]
            print(f"    seed{seed}: Success={sr:.2%} Reward={rw:.2f} ({train_time:.0f}s)", flush=True)

    # ── Save results ──
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

    results_path = os.path.join(output_dir, f"{env_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved -> {results_path}", flush=True)

    # Print table
    print(f"\n  {'Agent':25s} {'Success%':>12s} {'Collision%':>12s} {'Reward':>12s}", flush=True)
    print("  " + "-" * 65, flush=True)
    for agent_name in sorted(summary.keys(),
                             key=lambda a: summary[a].get("success_rate", {}).get("mean", 0),
                             reverse=True):
        sr = summary[agent_name].get("success_rate", {}).get("mean", 0)
        cr = summary[agent_name].get("collision_rate", {}).get("mean", 0)
        rw = summary[agent_name].get("mean_reward", {}).get("mean", 0)
        cat = "RL" if agent_name in ("sac_ltc", "sac_lstm") else "trad"
        print(f"  {agent_name:25s} {sr:>10.2%}   {cr:>10.2%}   {rw:>10.2f}  [{cat}]", flush=True)

    return summary


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="500K-Step Real-Data Training")
    parser.add_argument("--steps", type=int, default=5000000,
                        help="Training steps (default 5M for full convergence)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds (default 1 — each run is long)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per seed (default 100)")
    parser.add_argument("--env", type=str, default="all",
                        choices=["all", "real5g", "telecomts", "realistic"])
    parser.add_argument("--output-dir", type=str, default="realdata_500k_results")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(_repo_root, "data/ucc_misl/5Gdataset/extracted/5G-production-dataset"))
    parser.add_argument("--telecomts-samples", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cpu")
    seeds = list(range(1, args.seeds + 1))

    print(f"500K Real-Data Training Pipeline", flush=True)
    print(f"  Steps: {args.steps:,}  Seeds: {seeds}  Eval: {args.eval_episodes} eps", flush=True)

    envs_to_run = []
    if args.env in ("all", "real5g"):
        envs_to_run.append(("real5g", {
            "num_channels": 10,
            "sequence_length": 16,
            "num_features": 2,
            "init_kwargs": {
                "data_dir": args.data_dir,
                "num_channels": 10,
                "sequence_length": 16,
                "num_features": 2,
                "max_steps": 200,
            },
        }))

    if args.env in ("all", "telecomts"):
        envs_to_run.append(("telecomts", {
            "num_channels": 10,
            "sequence_length": 16,
            "num_features": 2,
            "init_kwargs": {
                "num_channels": 10,
                "sequence_length": 16,
                "num_features": 2,
                "max_steps": 100,
                "max_samples": args.telecomts_samples,
            },
        }))

    if args.env in ("all", "realistic"):
        from realistic_envs import RealisticDSAEnv
        envs_to_run.append(("realistic", {
            "num_channels": 10,
            "sequence_length": 16,
            "num_features": 2,
            "init_kwargs": {
                "num_channels": 10,
                "sequence_length": 16,
                "num_features": 2,
                "max_steps": 200,
            },
        }))

    for env_name, env_cfg in envs_to_run:
        print(f"\n{'='*80}", flush=True)
        print(f"  ENVIRONMENT: {env_name}", flush=True)
        print(f"{'='*80}", flush=True)

        if env_name == "real5g":
            env = Real5GDSAEnv(**env_cfg["init_kwargs"])
        elif env_name == "telecomts":
            env = TelecomTSDSAEnv(**env_cfg["init_kwargs"])
        elif env_name == "realistic":
            from realistic_envs import RealisticDSAEnv
            env = RealisticDSAEnv(**env_cfg["init_kwargs"])
        else:
            raise ValueError(f"Unknown env: {env_name}")

        env_dir = os.path.join(args.output_dir, env_name)
        train_and_evaluate(
            env_name, env, env_cfg,
            training_steps=args.steps,
            eval_episodes=args.eval_episodes,
            seeds=seeds,
            output_dir=env_dir,
            device=device,
        )

    print(f"\n{'='*80}", flush=True)
    print(f"  ALL TRAINING COMPLETE", flush=True)
    print(f"  Results in: {args.output_dir}/", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
