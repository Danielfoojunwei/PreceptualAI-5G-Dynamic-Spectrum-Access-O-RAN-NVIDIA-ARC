"""Train the first real SLA risk head checkpoint (PreceptualAI v0.1).

Trains :class:`SLARiskHead` against the maritime synthetic generator and
ships a checkpoint + 3GPP TS 28.105-flavoured model card. The team's
``E2E_DEBUG.md`` flagged that an untrained head emits ~0.5 noise — this
script exists to fix gap #2.

Usage:
    .venv/bin/python scripts/train_sla_head.py [--n-windows 2000]

Outputs:
    checkpoints/sla_head_v0.1.pt
    checkpoints/sla_head_v0.1.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make the package importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402
from horizon_ric.scenarios.maritime import (  # noqa: E402
    MaritimeScenarioConfig,
    MaritimeSyntheticGenerator,
    MaritimeWindow,
)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

LATENT_DIM = 64
HORIZON_STEPS = {
    "h_30s": 6,    # ~6 timesteps at 5s cadence
    "h_60s": 12,   # ~12 timesteps
    "h_300s": 60,  # full window
}


def _hash_projection(values: list[float], out_dim: int, seed_salt: bytes) -> np.ndarray:
    """Deterministic hash-based random projection of a small feature vector.

    Uses SHA-256 over the bytes of (values, seed_salt) to seed a NumPy RNG,
    then draws ``out_dim`` projection coefficients in [-1, 1] and dot-products
    with a tiled feature vector. Pure function of inputs — same window
    aggregates always produce the same noise dims.
    """
    h = hashlib.sha256()
    for v in values:
        h.update(np.float64(v).tobytes())
    h.update(seed_salt)
    seed = int.from_bytes(h.digest()[:8], "little") & 0x7FFF_FFFF
    rng = np.random.default_rng(seed)
    # Draw a (out_dim, len(values)) matrix; project the aggregates through it.
    W = rng.uniform(-1.0, 1.0, size=(out_dim, len(values)))
    feats = np.asarray(values, dtype=np.float64)
    proj = W @ feats
    # Bound the magnitude so concatenation with [0,1]-scale aggregates is sane.
    return np.tanh(proj).astype(np.float32)


def window_to_latent(window: MaritimeWindow) -> torch.Tensor:
    """Build a 64-d feature vector from a maritime window.

    First 5 dims = real aggregates; remaining 59 = deterministic hash projection
    so the head has *some* signal beyond the 5 raw aggregates without being a
    trivial noise vector. The hash is keyed on the aggregates, so it carries
    no extra information — but it is consistent per window (no leakage).
    """
    mean_prb = float(window.cell_prb_utilization.mean())
    max_gpu = float(window.edge_gpu_load.max())
    max_ntn = float(window.ntn_beam_capacity_used.max())
    total_ship_demand = float(window.ship_uplink_demand_mbps.sum())
    mean_rain = float(window.weather_rain_mm_per_hr.mean())

    aggregates = [mean_prb, max_gpu, max_ntn, total_ship_demand / 1000.0, mean_rain / 30.0]
    noise = _hash_projection(aggregates, LATENT_DIM - len(aggregates), b"sla_head_v0.1")
    vec = np.concatenate([np.asarray(aggregates, dtype=np.float32), noise])
    assert vec.shape == (LATENT_DIM,)
    return torch.from_numpy(vec)


def window_to_targets(window: MaritimeWindow) -> dict[str, float]:
    """Empirical SLA breach rate over each forecast horizon.

    For training simplicity we use the *full* window's breach rate trimmed to
    the horizon length — i.e. we treat the window as the look-ahead view.
    """
    breaches = window.sla_breaches.astype(np.float32)
    out: dict[str, float] = {}
    for key, n_steps in HORIZON_STEPS.items():
        head = breaches[: min(n_steps, len(breaches))]
        out[key] = float(head.mean()) if head.size else 0.0
    return out


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def build_corpus(n_windows: int, rng_seed: int = 0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generate ``n_windows`` (latent, target) pairs."""
    cfg = MaritimeScenarioConfig(rng_seed=rng_seed)
    gen = MaritimeSyntheticGenerator(cfg)

    latents: list[torch.Tensor] = []
    targets: dict[str, list[float]] = {k: [] for k in HORIZON_STEPS}
    for _ in range(n_windows):
        w = gen.sample()
        latents.append(window_to_latent(w))
        t = window_to_targets(w)
        for k in HORIZON_STEPS:
            targets[k].append(t[k])

    Z = torch.stack(latents, dim=0)
    Y = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
    return Z, Y


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def split_train_val(
    Z: torch.Tensor,
    Y: dict[str, torch.Tensor],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    n = Z.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    Y_train = {k: v[train_idx] for k, v in Y.items()}
    Y_val = {k: v[val_idx] for k, v in Y.items()}
    return Z[train_idx], Y_train, Z[val_idx], Y_val


def epoch_loss(
    head: SLARiskHead,
    Z: torch.Tensor,
    Y: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
) -> float:
    """Run one pass; if optimizer is None, no grad (val pass)."""
    is_train = optimizer is not None
    head.train(mode=is_train)
    n = Z.shape[0]
    perm = torch.randperm(n) if is_train else torch.arange(n)
    total = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        zb = Z[idx]
        yb = {k: v[idx] for k, v in Y.items()}
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            loss = head.loss(zb, yb)
            if torch.isnan(loss).any():
                raise RuntimeError("NaN loss encountered — aborting training.")
            if is_train:
                loss.backward()
                optimizer.step()
        total += float(loss.detach().item()) * zb.shape[0]
        seen += zb.shape[0]
    return total / max(seen, 1)


def train(
    head: SLARiskHead,
    Z_train: torch.Tensor,
    Y_train: dict[str, torch.Tensor],
    Z_val: torch.Tensor,
    Y_val: dict[str, torch.Tensor],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
) -> dict:
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history: list[dict] = []
    best_val = math.inf
    best_state: dict | None = None
    bad_epochs = 0
    initial_val = epoch_loss(head, Z_val, Y_val, optimizer=None, batch_size=batch_size)
    print(f"[init] val_loss={initial_val:.6f}")

    for ep in range(1, epochs + 1):
        train_l = epoch_loss(head, Z_train, Y_train, optimizer, batch_size)
        val_l = epoch_loss(head, Z_val, Y_val, optimizer=None, batch_size=batch_size)
        history.append({"epoch": ep, "train_loss": train_l, "val_loss": val_l})
        improved = val_l < best_val - 1e-6
        if improved:
            best_val = val_l
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"[ep {ep:02d}] train={train_l:.6f} val={val_l:.6f} "
            f"{'*' if improved else ''}best={best_val:.6f} bad={bad_epochs}"
        )
        if bad_epochs >= patience:
            print(f"[early-stop] no val improvement for {patience} epochs.")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return {
        "history": history,
        "initial_val_loss": initial_val,
        "best_val_loss": best_val,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Evaluation + checkpoint
# --------------------------------------------------------------------------- #


def evaluate(
    head: SLARiskHead,
    Z_val: torch.Tensor,
    Y_val: dict[str, torch.Tensor],
) -> dict:
    head.eval()
    with torch.no_grad():
        preds = head(Z_val)
    # Binary observed labels via 0.5 threshold on the empirical breach rate.
    observed = {k: (v >= 0.5).float() for k, v in Y_val.items()}
    ece = head.calibration_summary(preds, observed)
    brier = head.brier_score(preds, observed)
    p_min = {k: float(p.min()) for k, p in preds.items()}
    p_max = {k: float(p.max()) for k, p in preds.items()}
    p_mean = {k: float(p.mean()) for k, p in preds.items()}
    return {
        "ece": ece,
        "brier": brier,
        "p_min": p_min,
        "p_max": p_max,
        "p_mean": p_mean,
    }


def write_model_card(
    path: Path,
    *,
    n_train: int,
    n_val: int,
    rng_seed: int,
    final_train_loss: float,
    final_val_loss: float,
    best_val_loss: float,
    initial_val_loss: float,
    ece: dict[str, float],
    brier: dict[str, float],
    p_min: dict[str, float],
    p_max: dict[str, float],
    p_mean: dict[str, float],
    param_count: int,
    ckpt_path: Path,
    ckpt_size_bytes: int,
    ckpt_sha256: str,
    epochs_run: int,
    horizon_steps: dict[str, int],
) -> None:
    """3GPP TS 28.105-flavoured model card."""
    horizons_md = "\n".join(
        f"| {k} | {ece.get(k, float('nan')):.4f} | {brier.get(k, float('nan')):.4f} | "
        f"{p_min.get(k, float('nan')):.3f} | {p_mean.get(k, float('nan')):.3f} | "
        f"{p_max.get(k, float('nan')):.3f} |"
        for k in horizon_steps
    )
    md = f"""# SLA Risk Head — v0.1

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model name:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.1
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha256}`
- **File size (bytes):** {ckpt_size_bytes}
- **Parameter count:** {param_count}

## Intended Use
Multi-horizon SLA breach probability forecasting (30 s / 60 s / 300 s) from
the shared PreceptualAI `z_resource` latent. Phase-1 quality: trained
exclusively on synthetic maritime telemetry. NOT for production policy
deployment without subsequent fine-tuning on real telemetry.

## Training Data
- **Source:** `MaritimeSyntheticGenerator(rng_seed={rng_seed})` from
  `horizon_ric.scenarios.maritime`.
- **Synthetic-only:** True (per `MaritimeScenarioConfig.synthetic_only`).
- **Train samples:** {n_train}
- **Val samples:** {n_val}
- **Window length:** 300 s @ 5 s cadence (T=60).
- **Latent construction:** 5 hand-engineered window aggregates
  (mean PRB, max GPU, max NTN, total ship demand, mean rain) + 59
  deterministic SHA-256-seeded random projection dims (caller-side feature
  expansion to fill the 64-d slot).

## Targets
Empirical SLA breach rate over the next N timesteps per horizon:
- `h_30s`: first {horizon_steps['h_30s']} steps
- `h_60s`: first {horizon_steps['h_60s']} steps
- `h_300s`: full window ({horizon_steps['h_300s']} steps)

## Training Configuration
- Optimizer: Adam(lr=1e-3)
- Batch size: 64
- Max epochs: 30
- Early stopping: patience=5 on val loss
- Epochs actually run: {epochs_run}
- Loss: two-hot CE in logit space (DreamerV3 / `_two_hot.py`)

## Performance
- Initial val loss (random init): {initial_val_loss:.6f}
- Final train loss: {final_train_loss:.6f}
- Final val loss: {final_val_loss:.6f}
- Best val loss (loaded into checkpoint): {best_val_loss:.6f}

| horizon | ECE | Brier | P(min) | P(mean) | P(max) |
|---|---|---|---|---|---|
{horizons_md}

ECE computed per Naeini et al. 2015, 10 equal-width bins on [0, 1].
Observed labels are binary (`target_p >= 0.5`) at val time.

## Limitations & Honest Caveats
- Trained on a single rng_seed of synthetic telemetry — no domain shift,
  no real radio impairments, no real workload diversity.
- 59 of the 64 input dims are a hash-derived projection of the same 5
  aggregates; they carry no extra information. A future revision should
  consume the actual `z_resource` latent from the JEPA encoder.
- Binary thresholding at 0.5 of the empirical rate is convenient, not
  rigorous: expect calibration to drift on real data.
- Class imbalance not corrected; if storms are rare the high-P regime is
  under-represented.

## Reproducibility
```
.venv/bin/python scripts/train_sla_head.py --n-windows {n_train + n_val}
```
Seeds: torch.manual_seed(42), numpy default_rng(42), generator rng_seed={rng_seed}.

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- 3GPP TS 28.554 §6.x (E2E KPIs targeted by SLA breach definition).
- Brier score: Brier 1950. ECE: Naeini, Cooper, Hauskrecht 2015.
"""
    path.write_text(md, encoding="utf-8")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints"))
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"[setup] n_windows={args.n_windows} seed={args.rng_seed}")

    t0 = time.time()
    Z, Y = build_corpus(args.n_windows, rng_seed=args.rng_seed)
    print(f"[corpus] Z={tuple(Z.shape)} built in {time.time() - t0:.1f}s")
    for k, v in Y.items():
        print(
            f"[corpus] {k}: mean={float(v.mean()):.4f} "
            f"min={float(v.min()):.4f} max={float(v.max()):.4f} "
            f"frac>=0.5={float((v >= 0.5).float().mean()):.4f}"
        )

    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z, Y, val_frac=0.2, seed=42)
    print(f"[split] train={Z_tr.shape[0]} val={Z_val.shape[0]}")

    head = SLARiskHead(SLARiskConfig(latent_dim=LATENT_DIM, hidden_dim=64))
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params}")

    train_info = train(
        head,
        Z_tr,
        Y_tr,
        Z_val,
        Y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    eval_info = evaluate(head, Z_val, Y_val)
    print("[eval] ECE:", eval_info["ece"])
    print("[eval] Brier:", eval_info["brier"])
    print("[eval] P min:", eval_info["p_min"])
    print("[eval] P mean:", eval_info["p_mean"])
    print("[eval] P max:", eval_info["p_max"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sla_head_v0.1.pt"
    card_path = out_dir / "sla_head_v0.1.md"

    torch.save(head.state_dict(), ckpt_path)
    sha = sha256_of(ckpt_path)
    size = ckpt_path.stat().st_size
    print(f"[save] ckpt={ckpt_path} sha256={sha} size={size}")

    write_model_card(
        card_path,
        n_train=Z_tr.shape[0],
        n_val=Z_val.shape[0],
        rng_seed=args.rng_seed,
        final_train_loss=train_info["final_train_loss"],
        final_val_loss=train_info["final_val_loss"],
        best_val_loss=train_info["best_val_loss"],
        initial_val_loss=train_info["initial_val_loss"],
        ece=eval_info["ece"],
        brier=eval_info["brier"],
        p_min=eval_info["p_min"],
        p_max=eval_info["p_max"],
        p_mean=eval_info["p_mean"],
        param_count=n_params,
        ckpt_path=ckpt_path,
        ckpt_size_bytes=size,
        ckpt_sha256=sha,
        epochs_run=len(train_info["history"]),
        horizon_steps=HORIZON_STEPS,
    )
    print(f"[save] card={card_path}")

    summary = {
        "final_train_loss": train_info["final_train_loss"],
        "final_val_loss": train_info["final_val_loss"],
        "initial_val_loss": train_info["initial_val_loss"],
        "best_val_loss": train_info["best_val_loss"],
        "ece": eval_info["ece"],
        "brier": eval_info["brier"],
        "p_min": eval_info["p_min"],
        "p_max": eval_info["p_max"],
        "p_mean": eval_info["p_mean"],
        "ckpt": str(ckpt_path),
        "card": str(card_path),
        "sha256": sha,
        "size_bytes": size,
        "param_count": n_params,
    }
    print("[summary]", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
