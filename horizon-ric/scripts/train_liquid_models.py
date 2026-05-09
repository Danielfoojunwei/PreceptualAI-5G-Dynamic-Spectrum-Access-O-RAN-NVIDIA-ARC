"""Train CfCCell and LiquidS4 next-step predictors on real 5G traces.

Pipeline:
  1. Walk every CSV in the UCC MISL 5G production dataset.
  2. Extract numeric KPI columns (RSRP, RSRQ, SNR, CQI, RSSI, DL/UL bitrate, ...).
  3. Build sliding windows (window=64, stride=32) per trace.
  4. Z-score normalise per feature using train-set stats.
  5. Train both models for next-step prediction; checkpoint with norm stats.

Run on CUDA only (halt otherwise).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/danielfoojunwei/Preceptualv1/horizon-ric")
sys.path.insert(0, str(REPO / "src"))

from horizon_ric.core.cfc_core import CfCCell, CfCConfig  # noqa: E402
from horizon_ric.core.liquid_s4 import LiquidS4, LiquidS4Config  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    "/home/danielfoojunwei/Preceptualv1/data/ucc_misl/5Gdataset/extracted/"
    "5G-production-dataset"
)
CKPT_DIR = REPO / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)

WINDOW = 64
STRIDE = 32
SEED = 0

# Numeric KPI columns we care about. Many are stored as strings with '-' for NA.
KPI_COLUMNS = [
    "RSRP", "RSRQ", "SNR", "CQI", "RSSI",
    "DL_bitrate", "UL_bitrate", "Speed",
    "NRxRSRP", "NRxRSRQ",
]

DEVICE = "cuda" if torch.cuda.is_available() else None
if DEVICE is None:
    print("FATAL: CUDA not available; halting per task constraints.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def _to_float(series: pd.Series) -> pd.Series:
    """Convert a column with '-' / blank entries to float, NaN otherwise."""
    return pd.to_numeric(series.astype(str).str.strip().replace({"-": np.nan, "": np.nan}),
                         errors="coerce")


def load_traces(root: Path) -> list[tuple[str, np.ndarray]]:
    """Return list of (trace_id, array[T, F]) for every CSV under root.

    Skips macOS __MACOSX shadow files. Drops columns that are all-NaN. Forward-
    fills then back-fills remaining NaNs within a trace; drops traces shorter
    than WINDOW+1 samples.
    """
    traces: list[tuple[str, np.ndarray]] = []
    csvs = sorted(p for p in root.rglob("*.csv") if "__MACOSX" not in p.parts)
    for path in csvs:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue
        cols_present = [c for c in KPI_COLUMNS if c in df.columns]
        if not cols_present:
            continue
        sub = pd.DataFrame({c: _to_float(df[c]) for c in cols_present})
        # Drop columns all NaN for this trace.
        sub = sub.dropna(axis=1, how="all")
        if sub.shape[1] < 4:
            continue
        # Fill within trace.
        sub = sub.ffill().bfill()
        if sub.isna().any().any():
            continue
        if len(sub) <= WINDOW + 1:
            continue
        traces.append((str(path.relative_to(root)), sub.values.astype(np.float32),
                       list(sub.columns)))
    return traces


def harmonise_features(traces):
    """Keep only features present in every trace; align column order."""
    common = None
    for _id, _arr, cols in traces:
        common = set(cols) if common is None else common & set(cols)
    feats = [c for c in KPI_COLUMNS if c in common]
    out = []
    for tid, arr, cols in traces:
        idx = [cols.index(c) for c in feats]
        out.append((tid, arr[:, idx]))
    return out, feats


def build_windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = (len(arr) - window) // stride + 1
    if n <= 0:
        return np.empty((0, window, arr.shape[1]), dtype=arr.dtype)
    out = np.stack([arr[i * stride : i * stride + window] for i in range(n)], axis=0)
    return out


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class CfCNextStep(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.cell = CfCCell(CfCConfig(input_dim=n_features, hidden_dim=hidden_dim))
        self.head = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Predict x[:, t+1] from hidden after seeing x[:, :t+1].
        B, T, _ = x.shape
        h = self.cell.init_hidden(B, device=x.device)
        preds = []
        for t in range(T - 1):
            h = self.cell(x[:, t], h, dt=1.0)
            preds.append(self.head(h))
        return torch.stack(preds, dim=1)  # (B, T-1, F)


class LiquidS4NextStep(nn.Module):
    def __init__(self, n_features: int, d_state: int = 16, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(n_features, n_features)  # identity-ish init
        nn.init.eye_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.backbone = LiquidS4(
            LiquidS4Config(d_model=n_features, d_state=d_state, n_layers=n_layers)
        )
        self.head = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Use full sequence; predict x[:, t+1] from y[:, t].
        y = self.backbone(self.input_proj(x))  # (B, T, F)
        return self.head(y[:, :-1])             # (B, T-1, F)


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def mse_per_feature(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=(0, 1))


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def evaluate(model, loader, device) -> tuple[float, np.ndarray, float]:
    model.eval()
    se_total = 0.0
    n_total = 0
    per_feat_se = None
    per_feat_n = 0
    all_pred = []
    all_targ = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            target = xb[:, 1:]
            pred = model(xb)
            err = (pred - target) ** 2
            se_total += err.sum().item()
            n_total += err.numel()
            pf = err.mean(dim=(0, 1)).cpu().numpy()
            per_feat_se = pf if per_feat_se is None else per_feat_se + pf
            per_feat_n += 1
            all_pred.append(pred.cpu().numpy())
            all_targ.append(target.cpu().numpy())
    mse = se_total / max(n_total, 1)
    per_feat = per_feat_se / max(per_feat_n, 1)
    pred_arr = np.concatenate(all_pred, axis=0).reshape(-1, all_pred[0].shape[-1])
    targ_arr = np.concatenate(all_targ, axis=0).reshape(-1, all_targ[0].shape[-1])
    r2 = r2_score(pred_arr, targ_arr)
    return mse, per_feat, r2


def rollout_mse(model, val_windows: torch.Tensor, horizon: int, device) -> float:
    """Multi-step rollout: feed predictions back as input for `horizon` steps.

    Seeds with the first WINDOW-horizon real samples, then auto-regresses.
    Compares to the observed last `horizon` samples.
    """
    model.eval()
    seeds = val_windows[:, : WINDOW - horizon].to(device)  # (B, S, F)
    targets = val_windows[:, WINDOW - horizon : WINDOW].to(device)  # (B, h, F)
    se = 0.0
    n = 0
    with torch.no_grad():
        # Build context dynamically and forecast one step at a time.
        ctx = seeds.clone()
        for h in range(horizon):
            pred = model(ctx)  # (B, T-1, F)
            next_step = pred[:, -1:, :]  # (B, 1, F)
            ctx = torch.cat([ctx, next_step], dim=1)
            se += ((next_step.squeeze(1) - targets[:, h]) ** 2).sum().item()
            n += next_step.numel()
    return se / max(n, 1)


def train_model(model, train_loader, val_loader, *, max_epochs: int,
                max_seconds: float, device: str, label: str) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    best_val = float("inf")
    history = []
    for epoch in range(max_epochs):
        model.train()
        ep_se = 0.0
        ep_n = 0
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            target = xb[:, 1:]
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_se += loss.item() * target.numel()
            ep_n += target.numel()
        train_mse = ep_se / max(ep_n, 1)
        val_mse, _, val_r2 = evaluate(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "elapsed_s": elapsed,
        })
        print(f"[{label}] epoch {epoch+1:02d} train_mse={train_mse:.4f} "
              f"val_mse={val_mse:.4f} val_r2={val_r2:.3f} "
              f"elapsed={elapsed:.1f}s")
        if elapsed >= max_seconds:
            print(f"[{label}] hit time budget {max_seconds:.0f}s; stopping early")
            break
    return history


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    overall_t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Loading traces from {DATA_ROOT}")

    raw = load_traces(DATA_ROOT)
    print(f"  parsed {len(raw)} traces with at least 4 KPI columns")
    if not raw:
        sys.exit("No usable traces.")

    traces, feats = harmonise_features(raw)
    print(f"  common features ({len(feats)}): {feats}")
    print(f"  retained {len(traces)} traces after column harmonisation")

    # Train/val split by trace.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(traces))
    n_train = int(0.8 * len(traces))
    train_traces = [traces[i] for i in idx[:n_train]]
    val_traces = [traces[i] for i in idx[n_train:]]

    # Build windows.
    train_windows = np.concatenate(
        [build_windows(arr, WINDOW, STRIDE) for _, arr in train_traces], axis=0
    )
    val_windows = np.concatenate(
        [build_windows(arr, WINDOW, STRIDE) for _, arr in val_traces], axis=0
    )
    print(f"  train windows: {train_windows.shape}")
    print(f"  val   windows: {val_windows.shape}")
    n_features = train_windows.shape[-1]

    # Per-feature z-score on train set only.
    flat = train_windows.reshape(-1, n_features)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    train_norm = ((train_windows - mean) / std).astype(np.float32)
    val_norm = ((val_windows - mean) / std).astype(np.float32)
    # Clip extreme outliers so a single rogue sample doesn't dominate MSE.
    train_norm = np.clip(train_norm, -8.0, 8.0)
    val_norm = np.clip(val_norm, -8.0, 8.0)

    train_t = torch.from_numpy(train_norm)
    val_t = torch.from_numpy(val_norm)

    print(f"  feature means: {np.round(mean, 2).tolist()}")
    print(f"  feature stds : {np.round(std, 2).tolist()}")
    n_windows_total = len(train_windows) + len(val_windows)
    n_traces_total = len(traces)

    norm_stats = {
        "features": feats,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window": WINDOW,
        "stride": STRIDE,
    }

    results: dict = {
        "n_traces": n_traces_total,
        "n_windows": int(n_windows_total),
        "features": feats,
    }

    # =====================================================================
    # Phase A — CfCCell next-step
    # =====================================================================
    print("\n=== Phase A: CfCCell next-step ===")
    cfc_train_loader = DataLoader(TensorDataset(train_t), batch_size=128,
                                  shuffle=True, drop_last=False, num_workers=0)
    cfc_val_loader = DataLoader(TensorDataset(val_t), batch_size=128,
                                shuffle=False, num_workers=0)

    cfc_model = CfCNextStep(n_features=n_features, hidden_dim=64).to(DEVICE)
    cfc_random = copy.deepcopy(cfc_model)  # snapshot pre-training

    # Random-init baseline metrics
    rnd_val_mse, rnd_per_feat, rnd_r2 = evaluate(cfc_random, cfc_val_loader, DEVICE)
    rnd_rollout = rollout_mse(cfc_random, val_t, horizon=8, device=DEVICE)
    print(f"[cfc-random] val_mse={rnd_val_mse:.4f} r2={rnd_r2:.3f} "
          f"rollout8_mse={rnd_rollout:.4f}")

    cfc_history = train_model(
        cfc_model, cfc_train_loader, cfc_val_loader,
        max_epochs=20, max_seconds=180.0, device=DEVICE, label="cfc",
    )
    cfc_train_mse = cfc_history[-1]["train_mse"]
    cfc_val_mse, cfc_per_feat, cfc_r2 = evaluate(cfc_model, cfc_val_loader, DEVICE)
    cfc_rollout = rollout_mse(cfc_model, val_t, horizon=8, device=DEVICE)

    cfc_beats_random = cfc_val_mse < rnd_val_mse
    cfc_ckpt = CKPT_DIR / "cfc_cell_v0.1.pt"
    torch.save({
        "state_dict": cfc_model.state_dict(),
        "config": {"n_features": n_features, "hidden_dim": 64,
                    "input_dim": n_features},
        "norm_stats": norm_stats,
        "metrics": {
            "train_mse": float(cfc_train_mse),
            "val_mse": float(cfc_val_mse),
            "val_r2": float(cfc_r2),
            "rollout8_mse": float(cfc_rollout),
            "random_val_mse": float(rnd_val_mse),
            "random_rollout8_mse": float(rnd_rollout),
        },
        "history": cfc_history,
        "beats_random": bool(cfc_beats_random),
    }, cfc_ckpt)

    cfc_card = CKPT_DIR / "cfc_cell_v0.1.md"
    cfc_card.write_text(_format_card(
        title="CfCCell next-step v0.1",
        n_features=n_features, feats=feats,
        n_traces=n_traces_total, n_windows=n_windows_total,
        train_mse=cfc_train_mse, val_mse=cfc_val_mse, val_r2=cfc_r2,
        rollout=cfc_rollout, rnd_val_mse=rnd_val_mse,
        rnd_rollout=rnd_rollout, beats_random=cfc_beats_random,
        norm_stats=norm_stats,
    ))

    results["cfc"] = {
        "train_mse": float(cfc_train_mse),
        "val_mse": float(cfc_val_mse),
        "val_r2": float(cfc_r2),
        "rollout8_mse": float(cfc_rollout),
        "random_val_mse": float(rnd_val_mse),
        "random_rollout8_mse": float(rnd_rollout),
        "beats_random": bool(cfc_beats_random),
        "ckpt": str(cfc_ckpt),
        "ckpt_sha256": sha256(cfc_ckpt),
        "per_feature_val_mse": {f: float(v) for f, v in zip(feats, cfc_per_feat)},
    }

    # =====================================================================
    # Phase B — LiquidS4 next-step
    # =====================================================================
    print("\n=== Phase B: LiquidS4 next-step ===")
    s4_train_loader = DataLoader(TensorDataset(train_t), batch_size=64,
                                 shuffle=True, drop_last=False, num_workers=0)
    s4_val_loader = DataLoader(TensorDataset(val_t), batch_size=64,
                               shuffle=False, num_workers=0)

    s4_model = LiquidS4NextStep(n_features=n_features, d_state=16,
                                n_layers=4).to(DEVICE)
    s4_random = copy.deepcopy(s4_model)

    rnd_val_mse_s4, rnd_per_feat_s4, rnd_r2_s4 = evaluate(s4_random, s4_val_loader, DEVICE)
    rnd_rollout_s4 = rollout_mse(s4_random, val_t, horizon=8, device=DEVICE)
    print(f"[s4-random] val_mse={rnd_val_mse_s4:.4f} r2={rnd_r2_s4:.3f} "
          f"rollout8_mse={rnd_rollout_s4:.4f}")

    s4_history = train_model(
        s4_model, s4_train_loader, s4_val_loader,
        max_epochs=20, max_seconds=300.0, device=DEVICE, label="s4",
    )
    s4_train_mse = s4_history[-1]["train_mse"]
    s4_val_mse, s4_per_feat, s4_r2 = evaluate(s4_model, s4_val_loader, DEVICE)
    s4_rollout = rollout_mse(s4_model, val_t, horizon=8, device=DEVICE)

    s4_beats_random = s4_val_mse < rnd_val_mse_s4
    s4_ckpt = CKPT_DIR / "liquid_s4_v0.1.pt"
    torch.save({
        "state_dict": s4_model.state_dict(),
        "config": {"d_model": n_features, "d_state": 16, "n_layers": 4},
        "norm_stats": norm_stats,
        "metrics": {
            "train_mse": float(s4_train_mse),
            "val_mse": float(s4_val_mse),
            "val_r2": float(s4_r2),
            "rollout8_mse": float(s4_rollout),
            "random_val_mse": float(rnd_val_mse_s4),
            "random_rollout8_mse": float(rnd_rollout_s4),
        },
        "history": s4_history,
        "beats_random": bool(s4_beats_random),
    }, s4_ckpt)

    s4_card = CKPT_DIR / "liquid_s4_v0.1.md"
    s4_card.write_text(_format_card(
        title="LiquidS4 next-step v0.1",
        n_features=n_features, feats=feats,
        n_traces=n_traces_total, n_windows=n_windows_total,
        train_mse=s4_train_mse, val_mse=s4_val_mse, val_r2=s4_r2,
        rollout=s4_rollout, rnd_val_mse=rnd_val_mse_s4,
        rnd_rollout=rnd_rollout_s4, beats_random=s4_beats_random,
        norm_stats=norm_stats,
    ))

    results["liquid_s4"] = {
        "train_mse": float(s4_train_mse),
        "val_mse": float(s4_val_mse),
        "val_r2": float(s4_r2),
        "rollout8_mse": float(s4_rollout),
        "random_val_mse": float(rnd_val_mse_s4),
        "random_rollout8_mse": float(rnd_rollout_s4),
        "beats_random": bool(s4_beats_random),
        "ckpt": str(s4_ckpt),
        "ckpt_sha256": sha256(s4_ckpt),
        "per_feature_val_mse": {f: float(v) for f, v in zip(feats, s4_per_feat)},
    }

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    wall = time.time() - overall_t0
    results["gpu_peak_mb"] = peak_mb
    results["wall_clock_s"] = wall

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))


def _format_card(*, title, n_features, feats, n_traces, n_windows,
                  train_mse, val_mse, val_r2, rollout,
                  rnd_val_mse, rnd_rollout, beats_random, norm_stats):
    bar = "trained beats random" if beats_random else (
        "WARNING: trained does NOT beat random init -- ship with caveat"
    )
    return f"""# {title}

Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}
Dataset: UCC MISL 5G production dataset ({n_traces} traces, {n_windows} windows)
Features ({n_features}): {', '.join(feats)}
Window: {norm_stats['window']}, stride: {norm_stats['stride']}

## Metrics
- train_mse (z-scored): {train_mse:.4f}
- val_mse   (z-scored): {val_mse:.4f}
- val R^2: {val_r2:.4f}
- 8-step rollout MSE (z-scored): {rollout:.4f}

## Random-init baseline
- random val_mse: {rnd_val_mse:.4f}
- random 8-step rollout MSE: {rnd_rollout:.4f}
- Verdict: **{bar}**

## Normalisation stats (z-score)
- features: {feats}
- mean: {norm_stats['mean']}
- std:  {norm_stats['std']}

Inference: load `state_dict` from the .pt sibling, apply `(x - mean) / std`
per feature, run forward, invert with `pred * std + mean`.
"""


if __name__ == "__main__":
    main()
