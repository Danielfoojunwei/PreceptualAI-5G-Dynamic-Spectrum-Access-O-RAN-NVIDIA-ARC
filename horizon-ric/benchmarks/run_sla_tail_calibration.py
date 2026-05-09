"""SLA tail-risk calibration benchmark — Row 11 of GAPS_TO_PILOT.md.

Trains the production ``SLARiskHead`` (two-hot symlog, multi-horizon) on a
synthetic but **non-trivial** dataset and emits a regulator-readable
reliability diagram + numerical artefacts.

Outputs:
    benchmarks/sla_tail_calibration.json — ECE, Brier, bootstrap 95 % CI, bin
        edges, predicted probability per bin, empirical breach frequency per
        bin, sample counts.
    benchmarks/sla_tail_calibration.png  — reliability diagram (diagonal +
        per-bin points + 95 % CI ribbon).

Reference:
    Naeini, Cooper & Hauskrecht 2015 — Obtaining Well Calibrated Probabilities.
    McMahan 1989 — A statistical model for the verification of probability
        forecasts. (Brier-score decomposition, telco-friendly.)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CFG = {
    "n_train": 5_000,
    "n_test": 1_000,
    "state_dim": 32,            # synthetic state vector dim
    "latent_dim": 64,           # compact latent (production head is generic)
    "hidden_dim": 64,
    "num_bins": 51,
    "horizon_keys": ("h_30s", "h_60s", "h_300s"),
    # Different horizons see different P(breach) regimes — longer horizons see
    # higher breach rates (more time for SLA violation to occur).
    "horizon_base_rate": {"h_30s": 0.10, "h_60s": 0.20, "h_300s": 0.35},
    "epochs": 100,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-2,        # heavy WD — prevents memorising Bernoulli labels
    "label_smoothing": 0.02,     # mild smoothing for calibration
    "n_calibration_bins": 10,
    "n_bootstrap": 1_000,
    "ci_alpha": 0.05,            # 95 % CI
    "seed": 20260507,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Synthetic generator: realistic-ish state -> P(breach) -> Bernoulli outcome.
# --------------------------------------------------------------------------- #


def _generate_dataset(
    n: int,
    state_dim: int,
    base_rates: dict[str, float],
    rng: np.random.Generator,
    proj: np.ndarray,
    risk_mean: float | None = None,
    risk_std: float | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    float,
    float,
]:
    """Make (states, true_breach_probs, sampled_outcomes, risk_mean, risk_std).

    The true probability is a smooth non-linear function of a fixed projection
    of the state — the same ``proj`` is used for train and test so the head
    can actually generalise. Risk standardisation stats are also passed in
    from the train pass so the test set sits on the same logit scale.
    """
    states = rng.standard_normal(size=(n, state_dim)).astype(np.float32)
    z = states @ proj  # (n, 4)
    risk = 0.7 * z[:, 0] + 0.4 * np.tanh(z[:, 1] * z[:, 2]) - 0.3 * z[:, 3] ** 2
    if risk_mean is None:
        risk_mean = float(risk.mean())
        risk_std = float(risk.std() + 1e-6)
    risk = (risk - risk_mean) / risk_std

    true_probs: dict[str, torch.Tensor] = {}
    outcomes: dict[str, torch.Tensor] = {}
    for key, base in base_rates.items():
        scale = {"h_30s": 1.0, "h_60s": 1.3, "h_300s": 1.7}[key]
        logit_base = float(np.log(base / (1 - base)))
        logit = scale * risk + logit_base
        p = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, p).astype(np.float32)
        true_probs[key] = torch.from_numpy(p.astype(np.float32))
        outcomes[key] = torch.from_numpy(y)
    return (
        torch.from_numpy(states),
        true_probs,
        outcomes,
        float(risk_mean),
        float(risk_std),
    )


# --------------------------------------------------------------------------- #
# Tiny encoder: state_dim -> latent_dim, joined to the production head.
# --------------------------------------------------------------------------- #


class _StateEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int) -> None:
        super().__init__()
        # Compact encoder — calibration is the goal, not capacity. With
        # 5 000 samples × {0,1} labels a wide encoder memorises the Bernoulli
        # draw; we keep capacity moderate and rely on weight-decay + dropout.
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Calibration metrics with bootstrap CI.
# --------------------------------------------------------------------------- #


def _ece_brier(p: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    brier = float(np.mean((p - y) ** 2))
    return float(ece), brier


def _reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred, bin_emp, bin_n = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.any():
            bin_pred.append(float(p[mask].mean()))
            bin_emp.append(float(y[mask].mean()))
            bin_n.append(int(mask.sum()))
        else:
            bin_pred.append(float((lo + hi) / 2))
            bin_emp.append(float("nan"))
            bin_n.append(0)
    return edges.tolist(), bin_pred, bin_emp, bin_n


def _bootstrap_ci(
    p: np.ndarray, y: np.ndarray, n_bins: int, n_boot: int, alpha: float, rng: np.random.Generator
) -> tuple[tuple[float, float], np.ndarray]:
    n = len(p)
    eces = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        eces[b], _ = _ece_brier(p[idx], y[idx], n_bins)
    lo = float(np.quantile(eces, alpha / 2))
    hi = float(np.quantile(eces, 1 - alpha / 2))
    return (lo, hi), eces


# --------------------------------------------------------------------------- #
# Train / evaluate.
# --------------------------------------------------------------------------- #


def _train(
    encoder: nn.Module,
    head: SLARiskHead,
    states: torch.Tensor,
    targets: dict[str, torch.Tensor],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-3,
    label_smoothing: float = 0.0,
) -> list[float]:
    """Train via BCE on the head's sigmoid output.

    The head's two-hot CE ``loss`` expects target *probabilities*; we have
    binary outcomes (Bernoulli draws). Training BCE directly on the head's
    forward sigmoid is the standard well-calibrated objective and keeps the
    two-hot bin grid as the parameterisation of the logit decoder.
    """
    encoder.train(); head.train()
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    n = states.shape[0]
    history: list[float] = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            s = states[idx].to(DEVICE)
            tgt = {k: v[idx].to(DEVICE) for k, v in targets.items()}
            z = encoder(s)
            preds = head(z)  # dict[hk -> (B,) prob in (0,1)]
            loss = z.new_zeros(())
            eps = 1e-6
            for k, p in preds.items():
                p_clamped = p.clamp(eps, 1.0 - eps)
                y = tgt[k]
                if label_smoothing > 0.0:
                    y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                loss = loss - (y * torch.log(p_clamped)
                               + (1.0 - y) * torch.log(1.0 - p_clamped)).mean()
            loss = loss / max(len(preds), 1)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
        if ep == 0 or (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss={history[-1]:.4f}")
    return history


@torch.no_grad()
def _predict(
    encoder: nn.Module, head: SLARiskHead, states: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor]:
    encoder.eval(); head.eval()
    chunks: dict[str, list[torch.Tensor]] = {}
    for i in range(0, states.shape[0], batch_size):
        s = states[i : i + batch_size].to(DEVICE)
        out = head(encoder(s))
        for k, v in out.items():
            chunks.setdefault(k, []).append(v.detach().cpu())
    return {k: torch.cat(vs, dim=0) for k, vs in chunks.items()}


# --------------------------------------------------------------------------- #
# Plot.
# --------------------------------------------------------------------------- #


def _plot(payload: dict, out_path: Path) -> None:
    horizons = list(payload["per_horizon"].keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.8), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, hk in zip(axes, horizons):
        h = payload["per_horizon"][hk]
        bin_pred = np.array(h["bin_pred"])
        bin_emp = np.array(h["bin_emp"])
        bin_n = np.array(h["bin_n"])
        # Per-bin Wilson 95 % CI on the empirical fraction.
        z = 1.96
        lo = np.empty_like(bin_emp); hi = np.empty_like(bin_emp)
        for i, (e, n) in enumerate(zip(bin_emp, bin_n)):
            if n == 0 or np.isnan(e):
                lo[i] = np.nan; hi[i] = np.nan; continue
            denom = 1 + z * z / n
            centre = (e + z * z / (2 * n)) / denom
            half = z * np.sqrt(e * (1 - e) / n + z * z / (4 * n * n)) / denom
            lo[i] = max(0.0, centre - half); hi[i] = min(1.0, centre + half)

        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
        valid = ~np.isnan(bin_emp)
        ax.fill_between(bin_pred[valid], lo[valid], hi[valid], alpha=0.25, color="C0",
                        label="95% Wilson CI")
        ax.plot(bin_pred[valid], bin_emp[valid], "o-", color="C0", lw=2, label="empirical")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted P(breach)")
        ax.set_title(
            f"{hk}\nECE = {h['ece']:.4f}  "
            f"[{h['ece_ci'][0]:.4f}, {h['ece_ci'][1]:.4f}]\n"
            f"Brier = {h['brier']:.4f}    n = {h['n_test']}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("empirical breach frequency")
    fig.suptitle("SLA breach calibration — multi-horizon two-hot symlog head", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def main() -> None:
    rng = np.random.default_rng(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    print(f"[device] {DEVICE}")
    print("[data] generating synthetic SLA-breach dataset ...")
    # Shared, fixed projection so train and test share the state→risk map.
    proj = rng.standard_normal(size=(CFG["state_dim"], 4)).astype(np.float32) / np.sqrt(
        CFG["state_dim"]
    )
    states_train, _, y_train, rmu, rsd = _generate_dataset(
        CFG["n_train"], CFG["state_dim"], CFG["horizon_base_rate"], rng, proj
    )
    states_test, true_probs_test, y_test, _, _ = _generate_dataset(
        CFG["n_test"], CFG["state_dim"], CFG["horizon_base_rate"], rng, proj,
        risk_mean=rmu, risk_std=rsd,
    )
    print(f"[data] train={len(states_train)}  test={len(states_test)}")

    head_cfg = SLARiskConfig(
        latent_dim=CFG["latent_dim"],
        hidden_dim=CFG["hidden_dim"],
        num_bins=CFG["num_bins"],
        horizons_seconds=(30, 60, 300),
    )
    encoder = _StateEncoder(CFG["state_dim"], CFG["latent_dim"]).to(DEVICE)
    head = SLARiskHead(head_cfg).to(DEVICE)

    n_params_head = sum(p.numel() for p in head.parameters())
    n_params_enc = sum(p.numel() for p in encoder.parameters())
    print(f"[model] encoder={n_params_enc:,}  head={n_params_head:,}")

    print(f"[train] {CFG['epochs']} epochs ...")
    history = _train(
        encoder, head, states_train, y_train,
        epochs=CFG["epochs"], batch_size=CFG["batch_size"],
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
        label_smoothing=CFG["label_smoothing"],
    )

    print("[eval] computing predictions on held-out test set ...")
    preds = _predict(encoder, head, states_test, batch_size=CFG["batch_size"])

    per_horizon: dict[str, dict] = {}
    boot_rng = np.random.default_rng(CFG["seed"] + 1)
    for hk in CFG["horizon_keys"]:
        p = preds[hk].numpy().astype(np.float64)
        y = y_test[hk].numpy().astype(np.float64)
        ece, brier = _ece_brier(p, y, CFG["n_calibration_bins"])
        edges, bin_pred, bin_emp, bin_n = _reliability_curve(p, y, CFG["n_calibration_bins"])
        ci, _ = _bootstrap_ci(p, y, CFG["n_calibration_bins"],
                              CFG["n_bootstrap"], CFG["ci_alpha"], boot_rng)
        per_horizon[hk] = {
            "ece": ece, "ece_ci": list(ci),
            "brier": brier, "n_test": int(len(p)),
            "bin_edges": edges, "bin_pred": bin_pred,
            "bin_emp": bin_emp, "bin_n": bin_n,
        }
        print(
            f"  {hk}: ECE={ece:.4f} CI95=[{ci[0]:.4f}, {ci[1]:.4f}]  "
            f"Brier={brier:.4f}  n={len(p)}"
        )

    payload = {
        "config": CFG,
        "training": {"first_loss": history[0], "final_loss": history[-1],
                     "loss_history": history},
        "per_horizon": per_horizon,
        "calibration_target_ece": 0.10,
        "device": str(DEVICE),
    }
    json_path = HERE / "sla_tail_calibration.json"
    png_path = HERE / "sla_tail_calibration.png"
    json_path.write_text(json.dumps(payload, indent=2))
    _plot(payload, png_path)

    print(f"[out] {json_path}")
    print(f"[out] {png_path}")


if __name__ == "__main__":
    main()
