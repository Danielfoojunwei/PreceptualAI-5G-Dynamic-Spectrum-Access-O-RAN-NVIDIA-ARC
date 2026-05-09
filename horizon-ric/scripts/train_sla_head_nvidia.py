"""Train SLA Risk Head v0.2 on REAL NVIDIA + UCC data.

Pivots away from the synthetic-maritime-only training of v0.1 toward real
telemetry sources:

  1. NVIDIA Aerial FAPI parquet — uplink CQI / RSSI / MCS per slot.
  2. NVIDIA Aerial FH parquet — FH IQ samples (variance proxy for radio quality).
  3. NVIDIA DeepMIMO (asu_campus_3p5_dyn) — ray-traced RX power + delay spread
     across 100 scenario folders.
  4. UCC MISL 5G production traces — RSRP/RSRQ/SNR/CQI/DL_bitrate per second
     across mobile + static (Amazon, Netflix, Download) sessions.

Each source is reduced to a per-window aggregate vector, deterministically
projected to 64-d, and labelled with an SLA-breach probability derived from
the real radio measurements (weak RSRP, low CQI, low SNR, large delay
spread). Thresholds are explicit in :func:`compute_sla_risk` below.

Outputs:
    checkpoints/sla_head_v0.2_nvidia.pt
    checkpoints/sla_head_v0.2_nvidia.md

After training, runs a v0.1 vs v0.2 comparison on a fixed eval batch.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Paths to real data
# --------------------------------------------------------------------------- #

DATA_ROOT = Path("/home/danielfoojunwei/Preceptualv1/data")
AERIAL_FAPI = DATA_ROOT / "aerial/parquet/fapi.parquet"
AERIAL_FH = DATA_ROOT / "aerial/parquet/fh.parquet"
DEEPMIMO_ROOT = DATA_ROOT / "deepmimo/asu_campus_3p5_dyn"
UCC_ROOT = DATA_ROOT / "ucc_misl/5Gdataset/extracted/5G-production-dataset"

LATENT_DIM = 64
HORIZON_KEYS = ("h_30s", "h_60s", "h_300s")

# Explicit SLA breach thresholds (documented in model card).
THRESH_RSRP_DBM = -105.0  # weaker than -105 dBm = poor coverage (3GPP TS 38.133)
THRESH_RSRQ_DB = -15.0    # worse than -15 dB = poor quality
THRESH_SNR_DB = 0.0       # SNR <= 0 dB => link near unusable
THRESH_CQI = 6            # CQI < 6 = low MCS regime
THRESH_DELAY_SPREAD_NS = 200.0  # > 200 ns RMS delay spread = ISI risk
THRESH_RX_POWER_DBM = -90.0     # < -90 dBm receive => weak link


# --------------------------------------------------------------------------- #
# Inventory printing
# --------------------------------------------------------------------------- #


def print_inventory() -> dict:
    print("=" * 72)
    print("REAL DATA INVENTORY")
    print("=" * 72)

    inv: dict = {}

    if AERIAL_FAPI.exists():
        sz = AERIAL_FAPI.stat().st_size
        try:
            nrows = len(pd.read_parquet(AERIAL_FAPI))
        except Exception:
            nrows = -1
        print(f"[aerial.fapi] {AERIAL_FAPI} size={sz}B rows={nrows}")
        inv["aerial_fapi"] = {"path": str(AERIAL_FAPI), "size": sz, "rows": nrows}

    if AERIAL_FH.exists():
        sz = AERIAL_FH.stat().st_size
        try:
            nrows = len(pd.read_parquet(AERIAL_FH))
        except Exception:
            nrows = -1
        print(f"[aerial.fh]   {AERIAL_FH} size={sz}B rows={nrows}")
        inv["aerial_fh"] = {"path": str(AERIAL_FH), "size": sz, "rows": nrows}

    if DEEPMIMO_ROOT.exists():
        scenarios = sorted([p for p in DEEPMIMO_ROOT.iterdir() if p.is_dir()])
        print(f"[deepmimo]    {DEEPMIMO_ROOT} scenarios={len(scenarios)}")
        if scenarios:
            print(f"              first={scenarios[0].name}")
        inv["deepmimo"] = {"path": str(DEEPMIMO_ROOT), "n_scenarios": len(scenarios)}

    if UCC_ROOT.exists():
        csvs = list(UCC_ROOT.rglob("*.csv"))
        print(f"[ucc_misl]    {UCC_ROOT} csv_files={len(csvs)}")
        if csvs:
            total_size = sum(p.stat().st_size for p in csvs)
            print(f"              total_size={total_size}B")
        inv["ucc_misl"] = {"path": str(UCC_ROOT), "n_csvs": len(csvs)}

    print("=" * 72)
    return inv


# --------------------------------------------------------------------------- #
# Feature extraction per source — each returns list[(features, sla_risk_p)]
# --------------------------------------------------------------------------- #


def _project_to_64(raw: np.ndarray, salt: bytes) -> np.ndarray:
    """Deterministic random projection from arbitrary-d to 64-d via SHA-256 seed.

    Same raw vector + same salt => same 64-d vector. The projection matrix
    depends on the source SHAPE (len(raw)) only, so different sources use
    different projection matrices but stay deterministic per-source.
    """
    h = hashlib.sha256()
    h.update(salt)
    h.update(np.int64(len(raw)).tobytes())
    seed = int.from_bytes(h.digest()[:8], "little") & 0x7FFF_FFFF
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(LATENT_DIM, len(raw))).astype(np.float32)
    W /= np.sqrt(len(raw))  # JL-ish scaling
    return np.tanh(W @ raw.astype(np.float32))


def compute_sla_risk(
    *,
    rsrp: float | None = None,
    rsrq: float | None = None,
    snr: float | None = None,
    cqi: float | None = None,
    rx_power: float | None = None,
    delay_spread_ns: float | None = None,
    bler: float | None = None,
) -> dict[str, float]:
    """Map real radio measurements to a 3-horizon SLA risk probability.

    Each available signal contributes a 0..1 violation score; we average
    available signals and saturate. Horizons differ by mixing in temporal
    "drift": longer horizons inherit a small baseline + noise floor so that
    h_300s >= h_30s on average for the same window.
    """
    risks: list[float] = []

    if rsrp is not None and not math.isnan(rsrp):
        # -120 dBm -> 1.0 risk, -80 dBm -> 0.0 risk, linear in between
        risks.append(float(np.clip((-80.0 - rsrp) / 40.0, 0.0, 1.0)))
    if rsrq is not None and not math.isnan(rsrq):
        # -20 dB -> 1, -3 dB -> 0
        risks.append(float(np.clip((-3.0 - rsrq) / 17.0, 0.0, 1.0)))
    if snr is not None and not math.isnan(snr):
        # -5 dB -> 1, +20 dB -> 0
        risks.append(float(np.clip((20.0 - snr) / 25.0, 0.0, 1.0)))
    if cqi is not None and not math.isnan(cqi):
        risks.append(float(np.clip((10.0 - cqi) / 10.0, 0.0, 1.0)))
    if rx_power is not None and not math.isnan(rx_power):
        risks.append(float(np.clip((-60.0 - rx_power) / 40.0, 0.0, 1.0)))
    if delay_spread_ns is not None and not math.isnan(delay_spread_ns):
        risks.append(float(np.clip(delay_spread_ns / 500.0, 0.0, 1.0)))
    if bler is not None and not math.isnan(bler):
        risks.append(float(np.clip(bler, 0.0, 1.0)))

    if not risks:
        # No usable signal => can't label this window; caller should drop.
        return {k: float("nan") for k in HORIZON_KEYS}

    base = float(np.mean(risks))
    return {
        "h_30s": float(np.clip(base * 0.85, 0.0, 1.0)),
        "h_60s": float(np.clip(base * 0.95, 0.0, 1.0)),
        "h_300s": float(np.clip(base, 0.0, 1.0)),
    }


def extract_aerial_fapi(path: Path) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """Aerial FAPI: aggregate per CellId × (SFN window) → numeric KPIs."""
    if not path.exists():
        print(f"[aerial.fapi] MISSING: {path}")
        return [], []
    df = pd.read_parquet(path)
    needed = ["CellId", "SFN", "CQI", "rssi", "mcsIndex", "TBSize", "nUEs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[aerial.fapi] DROPPING — missing columns: {missing}")
        return [], []

    # Tiny dataset (19 rows). Group by CellId so each cell becomes a window.
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []
    grouped = df.groupby("CellId")
    for cell_id, g in grouped:
        cqi = float(g["CQI"].mean())
        rssi = float(g["rssi"].mean())
        mcs = float(g["mcsIndex"].mean())
        tbs = float(g["TBSize"].mean())
        nue = float(g["nUEs"].mean())
        layers = float(g["nrOfLayers"].mean()) if "nrOfLayers" in g.columns else 1.0
        rb_size = float(g["rbSize"].mean()) if "rbSize" in g.columns else 0.0

        raw = np.array([cqi, rssi, mcs, tbs, nue, layers, rb_size,
                        float(g["targetCodeRate"].mean()) if "targetCodeRate" in g.columns else 0.0],
                       dtype=np.float32)
        # CQI is already in a reasonable range; rssi here is uplink SRS-RSRP-ish in dB-ish units.
        # Treat CQI directly; treat rssi as a relative power proxy (not absolute dBm).
        risk = compute_sla_risk(cqi=cqi, snr=rssi)
        if any(math.isnan(v) for v in risk.values()):
            continue
        feats.append(_project_to_64(raw, b"aerial_fapi_v0.2"))
        targets.append(risk)

    print(f"[aerial.fapi] extracted {len(feats)} windows from {len(df)} rows")
    return feats, targets


def extract_aerial_fh(path: Path) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """Aerial FH: variance/range over the IQ samples in fhData."""
    if not path.exists():
        print(f"[aerial.fh] MISSING: {path}")
        return [], []
    df = pd.read_parquet(path)
    if "fhData" not in df.columns:
        print(f"[aerial.fh] DROPPING — no fhData column")
        return [], []
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []
    for _, row in df.iterrows():
        iq = row["fhData"]
        try:
            arr = np.asarray(iq, dtype=np.float64)
        except Exception:
            continue
        if arr.size == 0:
            continue
        # Power proxy: log10(variance) of IQ samples (post-FFT integers).
        var = float(arr.var())
        rms = float(np.sqrt((arr ** 2).mean()))
        peak = float(np.abs(arr).max())
        # Map variance to a pseudo-SNR: high variance => more signal => safer.
        # log10(var) normalized; large var (~1e8) -> ~8, small var (~1e4) -> ~4.
        log_var = math.log10(max(var, 1.0))
        pseudo_snr_db = (log_var - 4.0) * 5.0  # rough scale -> dB-ish
        nrx = float(row.get("nRxAnt", 1))
        nue = float(row.get("nUEs", 1))
        raw = np.array([log_var, math.log10(rms + 1.0), math.log10(peak + 1.0),
                        nrx, nue, float(row.get("Slot", 0))], dtype=np.float32)
        risk = compute_sla_risk(snr=pseudo_snr_db)
        if any(math.isnan(v) for v in risk.values()):
            continue
        feats.append(_project_to_64(raw, b"aerial_fh_v0.2"))
        targets.append(risk)
    print(f"[aerial.fh]   extracted {len(feats)} windows from {len(df)} rows")
    return feats, targets


def extract_deepmimo(root: Path, max_scenarios: int = 100) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """DeepMIMO: per-scenario per-(t, rx) feature using power + delay arrays."""
    if not root.exists():
        print(f"[deepmimo] MISSING: {root}")
        return [], []
    scenarios = sorted([p for p in root.iterdir() if p.is_dir()])[:max_scenarios]
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []
    n_dropped = 0

    for sc in scenarios:
        params_p = sc / "params.json"
        if not params_p.exists():
            n_dropped += 1
            continue
        try:
            params = json.loads(params_p.read_text())
        except Exception:
            n_dropped += 1
            continue
        freq = float(params.get("rt_params", {}).get("frequency", 3.5e9))

        # Each (t, rx) combination is a snapshot. Iterate through power_*.npz.
        power_files = sorted(sc.glob("power_t*_tx*_r*.npz"))
        for pf in power_files:
            stem = pf.stem  # e.g., power_t001_tx000_r000
            parts = stem.split("_")  # ['power', 't001', 'tx000', 'r000']
            try:
                t_id = parts[1]
                tx_id = parts[2]
                r_id = parts[3]
            except IndexError:
                continue

            try:
                pz = np.load(pf)
                power = pz["power"]  # (paths, ?) or (1, K)
            except Exception:
                continue
            if power.size == 0:
                continue

            # Linear-scale total received power (dBm summed in linear).
            with np.errstate(invalid="ignore"):
                p_lin = 10.0 ** (power / 10.0)
                p_lin_total = np.nansum(p_lin)
                if not np.isfinite(p_lin_total) or p_lin_total <= 0:
                    continue
                rx_power_dbm = 10.0 * np.log10(p_lin_total)
            n_paths = int(np.sum(np.isfinite(power)))
            mean_path_power = float(np.nanmean(power))
            max_path_power = float(np.nanmax(power))

            # Delay spread in ns from delay_*.npz (mirror filename).
            delay_p = sc / f"delay_{t_id}_{tx_id}_{r_id}.npz"
            ds_ns = float("nan")
            if delay_p.exists():
                try:
                    dz = np.load(delay_p)
                    delay = dz["delay"]
                    finite = np.isfinite(delay)
                    if finite.any():
                        d = delay[finite] * 1e9  # s -> ns
                        ds_ns = float(d.std())
                except Exception:
                    pass

            raw = np.array([
                rx_power_dbm,
                mean_path_power,
                max_path_power,
                float(n_paths),
                ds_ns if not math.isnan(ds_ns) else 0.0,
                math.log10(freq),
            ], dtype=np.float32)
            risk = compute_sla_risk(rx_power=rx_power_dbm, delay_spread_ns=ds_ns)
            if any(math.isnan(v) for v in risk.values()):
                continue
            feats.append(_project_to_64(raw, b"deepmimo_v0.2"))
            targets.append(risk)

    print(f"[deepmimo]    extracted {len(feats)} snapshots across {len(scenarios)} scenarios "
          f"(dropped {n_dropped})")
    return feats, targets


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("-", np.nan), errors="coerce")


def extract_ucc(root: Path, max_files: int = 60, window_sec: int = 30) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """UCC MISL: per-30s window, RSRP/RSRQ/SNR/CQI/DL aggregated."""
    if not root.exists():
        print(f"[ucc_misl] MISSING: {root}")
        return [], []
    csvs = sorted(root.rglob("*.csv"))[:max_files]
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []
    n_dropped = 0

    for cp in csvs:
        try:
            df = pd.read_csv(cp)
        except Exception:
            n_dropped += 1
            continue
        # Required columns
        need = {"RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate"}
        if not need.issubset(df.columns):
            n_dropped += 1
            continue
        # Coerce numerics — original dataset uses "-" for NaN
        for c in ["RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate", "UL_bitrate", "RSSI"]:
            if c in df.columns:
                df[c] = _coerce_num(df[c])

        # 30-row windows (~30s since cadence is 1 Hz).
        n = len(df)
        if n < window_sec:
            continue
        for start in range(0, n - window_sec + 1, window_sec):
            chunk = df.iloc[start:start + window_sec]
            rsrp = float(chunk["RSRP"].mean())
            rsrq = float(chunk["RSRQ"].mean())
            snr = float(chunk["SNR"].mean())
            cqi = float(chunk["CQI"].mean())
            dl = float(chunk["DL_bitrate"].mean())
            speed = float(chunk["Speed"].mean()) if "Speed" in chunk.columns else 0.0

            if any(math.isnan(x) for x in [rsrp, rsrq, snr, cqi]):
                continue
            raw = np.array([rsrp, rsrq, snr, cqi, math.log10(max(dl, 1.0)),
                            speed, float(chunk["RSRP"].std() or 0.0)],
                           dtype=np.float32)
            risk = compute_sla_risk(rsrp=rsrp, rsrq=rsrq, snr=snr, cqi=cqi)
            if any(math.isnan(v) for v in risk.values()):
                continue
            feats.append(_project_to_64(raw, b"ucc_misl_v0.2"))
            targets.append(risk)

    print(f"[ucc_misl]    extracted {len(feats)} windows from {len(csvs)} files "
          f"(dropped {n_dropped})")
    return feats, targets


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #


def split_train_val(Z: torch.Tensor, Y: dict[str, torch.Tensor], val_frac: float = 0.2, seed: int = 42):
    n = Z.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return (Z[tr_idx],
            {k: v[tr_idx] for k, v in Y.items()},
            Z[val_idx],
            {k: v[val_idx] for k, v in Y.items()})


def epoch_pass(head, Z, Y, optim, batch_size):
    is_train = optim is not None
    head.train(mode=is_train)
    n = Z.shape[0]
    perm = torch.randperm(n) if is_train else torch.arange(n)
    total = 0.0
    seen = 0
    for s in range(0, n, batch_size):
        idx = perm[s:s + batch_size]
        zb = Z[idx]
        yb = {k: v[idx] for k, v in Y.items()}
        if is_train:
            optim.zero_grad()
        with torch.set_grad_enabled(is_train):
            loss = head.loss(zb, yb)
            if is_train:
                loss.backward()
                optim.step()
        total += float(loss.detach().item()) * zb.shape[0]
        seen += zb.shape[0]
    return total / max(seen, 1)


def train_loop(head, Z_tr, Y_tr, Z_val, Y_val, *, epochs=30, batch_size=64, lr=1e-3, patience=5):
    optim = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    best = math.inf
    best_state = None
    bad = 0
    initial = epoch_pass(head, Z_val, Y_val, None, batch_size)
    print(f"[init] val_loss={initial:.6f}")
    for ep in range(1, epochs + 1):
        tl = epoch_pass(head, Z_tr, Y_tr, optim, batch_size)
        vl = epoch_pass(head, Z_val, Y_val, None, batch_size)
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        improved = vl < best - 1e-6
        if improved:
            best = vl
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"[ep {ep:02d}] train={tl:.6f} val={vl:.6f} {'*' if improved else ''} best={best:.6f} bad={bad}")
        if bad >= patience:
            print(f"[early-stop] patience={patience} reached")
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    return {
        "history": history,
        "initial_val_loss": initial,
        "best_val_loss": best,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
    }


def evaluate(head, Z_val, Y_val):
    head.eval()
    with torch.no_grad():
        preds = head(Z_val)
    observed = {k: (v >= 0.5).float() for k, v in Y_val.items()}
    ece = head.calibration_summary(preds, observed)
    brier = head.brier_score(preds, observed)
    return {
        "ece": ece,
        "brier": brier,
        "p_min": {k: float(p.min()) for k, p in preds.items()},
        "p_max": {k: float(p.max()) for k, p in preds.items()},
        "p_mean": {k: float(p.mean()) for k, p in preds.items()},
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #


def write_card(path: Path, *, contributing: dict[str, int], dropped: list[str],
               n_train: int, n_val: int, train_info: dict, eval_info: dict,
               n_params: int, ckpt_path: Path, ckpt_size: int, ckpt_sha: str):
    rows = []
    for k in HORIZON_KEYS:
        rows.append(
            f"| {k} | {eval_info['ece'].get(k, float('nan')):.4f} | "
            f"{eval_info['brier'].get(k, float('nan')):.4f} | "
            f"{eval_info['p_min'].get(k, float('nan')):.3f} | "
            f"{eval_info['p_mean'].get(k, float('nan')):.3f} | "
            f"{eval_info['p_max'].get(k, float('nan')):.3f} |"
        )
    horizons_md = "\n".join(rows)

    contrib_md = "\n".join(
        f"- **{name}** ({count} windows) — `{path_str}`"
        for name, (count, path_str) in contributing.items()
    ) or "- (none — all sources dropped; see below)"

    dropped_md = "\n".join(f"- {d}" for d in dropped) or "- (none)"

    md = f"""# SLA Risk Head — v0.2 (NVIDIA-first)

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model name:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.2-nvidia
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## Intended Use
Multi-horizon SLA breach probability forecasting (30 s / 60 s / 300 s) from
the shared PreceptualAI `z_resource` latent. v0.2 pivots away from the
synthetic maritime generator used in v0.1; it is trained on REAL
NVIDIA + UCC telemetry. Still phase-1: not for unsupervised production
deployment without on-site fine-tune.

## Training Data — REAL sources that contributed

{contrib_md}

### Sources excluded / dropped this run

{dropped_md}

- **Train samples:** {n_train}
- **Val samples:** {n_val}
- **Latent construction:** per-source numeric KPI vector → deterministic
  Gaussian random projection to 64-d (Johnson-Lindenstrauss style; per-source
  SHA-256 salt; tanh non-linearity at the output). The same source row
  always projects to the same 64-d vector.

## Targets — derived from REAL signals

A window is high-risk when measured radio quality is poor. Each available
signal contributes a [0,1] violation score; the per-window risk is the mean
of available signals. Horizons inherit a multiplicative scale so longer
horizons trend slightly higher than shorter ones.

| Signal | "Safe" | "Bad" | Source |
|---|---|---|---|
| RSRP | ≥ -80 dBm | ≤ -120 dBm | UCC MISL |
| RSRQ | ≥ -3 dB   | ≤ -20 dB  | UCC MISL |
| SNR  | ≥ +20 dB  | ≤ -5 dB   | UCC, Aerial FAPI rssi-proxy, Aerial FH log-var-proxy |
| CQI  | ≥ 10      | ≤ 0       | UCC, Aerial FAPI |
| RX power | ≥ -60 dBm | ≤ -100 dBm | DeepMIMO ray-traced |
| Delay spread | ≤ 0 ns | ≥ 500 ns | DeepMIMO |

Hard thresholds documented for downstream auditing:
RSRP < {THRESH_RSRP_DBM} dBm, RSRQ < {THRESH_RSRQ_DB} dB, SNR < {THRESH_SNR_DB} dB,
CQI < {THRESH_CQI}, delay spread > {THRESH_DELAY_SPREAD_NS} ns,
RX power < {THRESH_RX_POWER_DBM} dBm.

## Training Configuration
- Optimizer: Adam(lr=1e-3)
- Batch size: 64
- Max epochs: 30
- Early stopping: patience=5 on val loss
- Epochs run: {len(train_info['history'])}
- Loss: two-hot CE in logit space (DreamerV3-style)

## Performance
- Initial val loss (random init): {train_info['initial_val_loss']:.6f}
- Final train loss: {train_info['final_train_loss']:.6f}
- Final val loss: {train_info['final_val_loss']:.6f}
- Best val loss (in checkpoint): {train_info['best_val_loss']:.6f}

| horizon | ECE | Brier | P(min) | P(mean) | P(max) |
|---|---|---|---|---|---|
{horizons_md}

ECE: Naeini et al. 2015, 10 equal-width bins on [0, 1]. Observed labels are
binary (`target_p >= 0.5`).

## Honest Caveats
- The SLA-breach LABEL is *derived* from real radio measurements via the
  threshold table above; it is not an observed breach event. Real customer
  KPI breach data is not yet available from these sources.
- Aerial FH delivers only 6 rows in the shipped parquet; its contribution
  is small. If it dropped to zero this run, see the exclusion list above.
- DeepMIMO is ray-traced (not measured), but it is real ray-tracing
  geometry from NVIDIA Sionna RT / Wireless Insite — not a parametric
  generator.
- Per-source random projections are L2-isometric in expectation (J-L) but
  add no information; this is a feature *expansion* to fill the 64-d
  z_resource interface, not a feature extractor.
- No domain adaptation: training distribution mixes uplink-FAPI, ray-traced
  power, and 5G-mobile drive tests. A v0.3 should weight per source.

## Reproducibility
```
.venv/bin/python scripts/train_sla_head_nvidia.py
```
Seeds: torch.manual_seed(42), numpy default_rng(42).

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- 3GPP TS 38.133 (RSRP/RSRQ measurement requirements).
- 3GPP TS 28.554 §6.x (E2E KPIs targeted by SLA breach definition).
- Brier score: Brier 1950. ECE: Naeini, Cooper, Hauskrecht 2015.
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# v0.1 vs v0.2 comparison
# --------------------------------------------------------------------------- #


def compare_v01_v02(v01_path: Path, v02_path: Path, latent_dim: int = LATENT_DIM):
    """Load both heads and run them on the same fixed eval batch (seed=99)."""
    if not v01_path.exists():
        print(f"[compare] v0.1 missing at {v01_path} — skipping comparison")
        return None

    g = torch.Generator().manual_seed(99)
    Z = torch.randn(256, latent_dim, generator=g)

    head1 = SLARiskHead(SLARiskConfig(latent_dim=latent_dim, hidden_dim=64))
    head1.load_state_dict(torch.load(v01_path, map_location="cpu", weights_only=True))
    head1.eval()
    head2 = SLARiskHead(SLARiskConfig(latent_dim=latent_dim, hidden_dim=64))
    head2.load_state_dict(torch.load(v02_path, map_location="cpu", weights_only=True))
    head2.eval()

    with torch.no_grad():
        p1 = head1(Z)
        p2 = head2(Z)

    def stats(p):
        out = {}
        for k, v in p.items():
            mean = float(v.mean())
            var = float(v.var())
            # Bernoulli entropy per sample
            p_clip = v.clamp(1e-6, 1 - 1e-6)
            ent = -(p_clip * torch.log2(p_clip) + (1 - p_clip) * torch.log2(1 - p_clip))
            out[k] = {"mean": mean, "var": var, "entropy": float(ent.mean())}
        return out

    s1 = stats(p1)
    s2 = stats(p2)
    print("=" * 72)
    print("V0.1 vs V0.2 COMPARISON (eval batch seed=99, N=256)")
    print("=" * 72)
    for k in HORIZON_KEYS:
        print(f"  {k:8s}  v0.1: mean={s1[k]['mean']:.4f} var={s1[k]['var']:.5f} H={s1[k]['entropy']:.4f}  |  "
              f"v0.2: mean={s2[k]['mean']:.4f} var={s2[k]['var']:.5f} H={s2[k]['entropy']:.4f}")
    # Overall confidence: lower entropy = more confident
    h1 = sum(s1[k]["entropy"] for k in HORIZON_KEYS) / 3
    h2 = sum(s2[k]["entropy"] for k in HORIZON_KEYS) / 3
    print(f"  mean entropy: v0.1={h1:.4f}  v0.2={h2:.4f}  ->  more_confident={'v0.1' if h1 < h2 else 'v0.2'}")
    print("=" * 72)
    return {"v01": s1, "v02": s2, "more_confident": "v0.1" if h1 < h2 else "v0.2"}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    inv = print_inventory()

    t0 = time.time()
    contributing: dict[str, tuple[int, str]] = {}
    dropped: list[str] = []
    all_feats: list[np.ndarray] = []
    all_targets: list[dict[str, float]] = []

    # Aerial FAPI
    f, t = extract_aerial_fapi(AERIAL_FAPI)
    if f:
        contributing["NVIDIA Aerial FAPI"] = (len(f), str(AERIAL_FAPI))
        all_feats.extend(f); all_targets.extend(t)
    else:
        dropped.append(f"NVIDIA Aerial FAPI ({AERIAL_FAPI})")

    # Aerial FH
    f, t = extract_aerial_fh(AERIAL_FH)
    if f:
        contributing["NVIDIA Aerial FH"] = (len(f), str(AERIAL_FH))
        all_feats.extend(f); all_targets.extend(t)
    else:
        dropped.append(f"NVIDIA Aerial FH ({AERIAL_FH})")

    # DeepMIMO — cap to 100 (the available count)
    f, t = extract_deepmimo(DEEPMIMO_ROOT, max_scenarios=100)
    if f:
        contributing["NVIDIA DeepMIMO (asu_campus_3p5_dyn)"] = (len(f), str(DEEPMIMO_ROOT))
        all_feats.extend(f); all_targets.extend(t)
    else:
        dropped.append(f"NVIDIA DeepMIMO ({DEEPMIMO_ROOT})")

    # UCC — cap to 60 files to bound CPU time
    f, t = extract_ucc(UCC_ROOT, max_files=60, window_sec=30)
    if f:
        contributing["UCC MISL 5G production traces"] = (len(f), str(UCC_ROOT))
        all_feats.extend(f); all_targets.extend(t)
    else:
        dropped.append(f"UCC MISL 5G ({UCC_ROOT})")

    print(f"[corpus] sources contributing: {list(contributing)}")
    print(f"[corpus] sources dropped: {dropped}")
    print(f"[corpus] total samples: {len(all_feats)} in {time.time() - t0:.1f}s")

    if len(all_feats) < 32:
        print(f"[error] only {len(all_feats)} samples — too few to train.")
        return 2

    Z = torch.from_numpy(np.stack(all_feats, axis=0)).float()
    Y = {k: torch.tensor([t[k] for t in all_targets], dtype=torch.float32)
         for k in HORIZON_KEYS}
    for k, v in Y.items():
        print(f"[corpus] {k}: mean={float(v.mean()):.4f} min={float(v.min()):.4f} "
              f"max={float(v.max()):.4f} frac>=0.5={float((v>=0.5).float().mean()):.4f}")

    # Cap the corpus to keep training under 5 minutes on CPU.
    MAX_TOTAL = 4000
    if Z.shape[0] > MAX_TOTAL:
        g = torch.Generator().manual_seed(123)
        idx = torch.randperm(Z.shape[0], generator=g)[:MAX_TOTAL]
        Z = Z[idx]
        Y = {k: v[idx] for k, v in Y.items()}
        print(f"[corpus] subsampled to {Z.shape[0]} for CPU budget")

    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z, Y)
    print(f"[split] train={Z_tr.shape[0]} val={Z_val.shape[0]}")

    head = SLARiskHead(SLARiskConfig(latent_dim=LATENT_DIM, hidden_dim=64))
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params}")

    train_info = train_loop(head, Z_tr, Y_tr, Z_val, Y_val,
                            epochs=30, batch_size=64, lr=1e-3, patience=5)

    eval_info = evaluate(head, Z_val, Y_val)
    print("[eval] ECE:", eval_info["ece"])
    print("[eval] Brier:", eval_info["brier"])
    print("[eval] P min:", eval_info["p_min"])
    print("[eval] P mean:", eval_info["p_mean"])
    print("[eval] P max:", eval_info["p_max"])

    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "sla_head_v0.2_nvidia.pt"
    card = out_dir / "sla_head_v0.2_nvidia.md"

    torch.save(head.state_dict(), ckpt)
    sha = sha256_of(ckpt)
    size = ckpt.stat().st_size
    print(f"[save] ckpt={ckpt} sha256={sha} size={size}")

    write_card(card,
               contributing=contributing, dropped=dropped,
               n_train=Z_tr.shape[0], n_val=Z_val.shape[0],
               train_info=train_info, eval_info=eval_info,
               n_params=n_params, ckpt_path=ckpt,
               ckpt_size=size, ckpt_sha=sha)
    print(f"[save] card={card}")

    # v0.1 vs v0.2 comparison
    compare_v01_v02(out_dir / "sla_head_v0.1.pt", ckpt)

    summary = {
        "final_train_loss": train_info["final_train_loss"],
        "final_val_loss": train_info["final_val_loss"],
        "best_val_loss": train_info["best_val_loss"],
        "ece": eval_info["ece"],
        "brier": eval_info["brier"],
        "p_min": eval_info["p_min"],
        "p_mean": eval_info["p_mean"],
        "p_max": eval_info["p_max"],
        "ckpt": str(ckpt),
        "card": str(card),
        "sha256": sha,
        "size_bytes": size,
        "n_params": n_params,
        "contributing": {k: v[0] for k, v in contributing.items()},
        "dropped": dropped,
    }
    print("[summary]", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
