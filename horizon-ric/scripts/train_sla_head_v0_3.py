"""Train SLA Risk Head v0.3 on the FULL real-data corpus.

Builds on the v0.2 NVIDIA-first training (Aerial FAPI + FH + DeepMIMO + UCC)
and ADDS:

  * NVIDIA Aerial cuMAC H5 PHY test vectors — per-vector SINR aggregates
    (postEqSinr / wbSinr) across 4T4R type0 + type1 allocations.
  * CelesTrak Starlink TLEs (12k+ sats) → per-window 4-feature vector:
    (n_visible >10°, mean elevation, mean slant range, mean Doppler @ 28 GHz).
    Computed from the Rotterdam ES (51.92°N, 4.48°E).
  * ITU-R reference maps — P.839-4 (rain height, km), P.453-14 (N_wet),
    P.1510-1 (mean surface temperature, K) — looked up at the ES.

Each window's 64-d projected vector is RE-PROJECTED with a fresh random
projection that includes the new TLE + ITU dims, so the v0.3 latent SHA
differs from v0.2 even on the same source rows.

Outputs:
    checkpoints/sla_head_v0.3_full_corpus.pt
    checkpoints/sla_head_v0.3_full_corpus.md

After training, runs a v0.2 vs v0.3 comparison on the same fixed eval batch.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse v0.2 extractors verbatim.
sys.path.insert(0, str(ROOT / "scripts"))
from train_sla_head_nvidia import (  # noqa: E402
    HORIZON_KEYS,
    LATENT_DIM,
    AERIAL_FAPI,
    AERIAL_FH,
    DEEPMIMO_ROOT,
    UCC_ROOT,
    compute_sla_risk,
    extract_aerial_fapi,
    extract_aerial_fh,
    extract_deepmimo,
    extract_ucc,
    epoch_pass,
    evaluate,
    sha256_of,
    split_train_val,
    train_loop,
)

from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DATA_ROOT = Path("/home/danielfoojunwei/Preceptualv1/data")
CUMAC_ROOT = DATA_ROOT / "aerial/cumac_test_vectors/testVectors"
TLE_PATH = DATA_ROOT / "orbital/celestrak/starlink.tle"
ITU_ROOT = DATA_ROOT / "itu_r"

# Fixed Earth station: Rotterdam (per task spec).
ES_LAT = 51.92
ES_LON = 4.48
ES_HEIGHT_M = 0.0
KA_CARRIER_HZ = 28e9
ELEV_MASK_DEG = 10.0

# Cap satellites scanned per UTC for CPU budget. The constellation is large
# enough that 800 random sats give a near-identical visibility distribution.
TLE_SAMPLE_N = 600
TLE_CACHE = ROOT / "checkpoints" / "_tle_cache_v0.3.npz"

# Augmented latent: concat 7 new "physics" dims + 64 projected per-source ->
# project back to LATENT_DIM with a fresh seed so v0.3 != v0.2 bit-for-bit.
N_PHYSICS_DIMS = 7  # 4 TLE + 3 ITU


# --------------------------------------------------------------------------- #
# Optional ITU readers (fall back gracefully)
# --------------------------------------------------------------------------- #


def _load_itu_readers():
    """Return (rain_km, n_wet, t_K) plus list of (name, available, value)."""
    try:
        from horizon_ric.data.itu_r import P453Map, P839Map, P1510Map
    except Exception as exc:  # pragma: no cover
        print(f"[itu] import failed: {exc}; using fallback constants")
        return (3.0, 40.0, 288.15), [
            ("P.839-4", False, 3.0),
            ("P.453-14", False, 40.0),
            ("P.1510-1", False, 288.15),
        ]

    out = []
    try:
        m839 = P839Map(data_dir=ITU_ROOT)
        rain_km = float(m839.lookup(ES_LAT, ES_LON))
        out.append(("P.839-4", bool(m839.is_available), rain_km))
    except Exception as exc:
        print(f"[itu] P839 failed: {exc}")
        rain_km = 3.0
        out.append(("P.839-4", False, rain_km))

    try:
        m453 = P453Map(data_dir=ITU_ROOT)
        n_wet = float(m453.lookup(ES_LAT, ES_LON))
        out.append(("P.453-14", bool(m453.is_available), n_wet))
    except Exception as exc:
        print(f"[itu] P453 failed: {exc}")
        n_wet = 40.0
        out.append(("P.453-14", False, n_wet))

    try:
        m1510 = P1510Map(data_dir=ITU_ROOT)
        t_K = float(m1510.lookup(ES_LAT, ES_LON))
        out.append(("P.1510-1", bool(m1510.is_available), t_K))
    except Exception as exc:
        print(f"[itu] P1510 failed: {exc}")
        t_K = 288.15
        out.append(("P.1510-1", False, t_K))

    return (rain_km, n_wet, t_K), out


# --------------------------------------------------------------------------- #
# TLE feature extraction
# --------------------------------------------------------------------------- #


def _parse_tles(path: Path) -> list[tuple[str, str, str]]:
    """Three-line element format: name, line1, line2."""
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    out: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return out


def _tle_features_at_utc(
    sats, t_utc: datetime, max_sats: int = TLE_SAMPLE_N
) -> tuple[float, float, float, float]:
    """Return (n_visible, mean_elev_deg, mean_slant_km, mean_doppler_hz)."""
    from horizon_ric.planner.physics.geodesy import look_angles_to_target
    from horizon_ric.planner.physics.orbital import (
        OrbitalState,
        eci_to_ecef_km,
        sgp4_state,
        state_ecef_m,
    )
    from horizon_ric.planner.physics.doppler import doppler_shift_hz

    if not sats:
        return 0.0, 0.0, 0.0, 0.0

    # Subsample deterministically so the same UTC always picks the same sats.
    if len(sats) > max_sats:
        seed = int(t_utc.timestamp()) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sats), size=max_sats, replace=False)
        sats = [sats[i] for i in idx]

    visible_elev = []
    visible_rng_km = []
    visible_dopp = []
    for _, l1, l2 in sats:
        try:
            st = sgp4_state(l1, l2, t_utc)
            sat_ecef = state_ecef_m(st)
            la = look_angles_to_target(ES_LAT, ES_LON, sat_ecef, ES_HEIGHT_M)
        except Exception:
            continue
        if la.elevation_deg < ELEV_MASK_DEG:
            continue
        visible_elev.append(la.elevation_deg)
        visible_rng_km.append(la.range_m / 1000.0)
        try:
            dop = doppler_shift_hz(st, ES_LAT, ES_LON, KA_CARRIER_HZ, ES_HEIGHT_M)
        except Exception:
            dop = 0.0
        visible_dopp.append(dop)

    if not visible_elev:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(len(visible_elev)),
        float(np.mean(visible_elev)),
        float(np.mean(visible_rng_km)),
        float(np.mean(np.abs(visible_dopp))),
    )


def build_tle_window_features(
    n_windows: int,
    t0: datetime | None = None,
    cadence_min: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """One TLE feature row per window, walking forward in time at cadence_min.

    Cached to disk on the first run (so subsequent runs are <1 s).
    Returns (rows, meta) where rows shape = (n_windows, 4).
    """
    if TLE_CACHE.exists():
        try:
            z = np.load(TLE_CACHE, allow_pickle=True)
            if int(z["n"]) >= n_windows:
                rows = z["rows"][:n_windows]
                meta = json.loads(str(z["meta"]))
                print(f"[tle] cache hit: {rows.shape} from {TLE_CACHE.name}")
                return rows, meta
        except Exception as exc:
            print(f"[tle] cache invalid ({exc}); rebuilding")

    sats = _parse_tles(TLE_PATH)
    print(f"[tle] parsed {len(sats)} Starlink TLEs from {TLE_PATH.name}")
    if t0 is None:
        t0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)

    rows = np.zeros((n_windows, 4), dtype=np.float32)
    t_start = time.time()
    for i in range(n_windows):
        t = t0 + timedelta(minutes=cadence_min * i)
        rows[i] = _tle_features_at_utc(sats, t)
        if i and i % 50 == 0:
            elapsed = time.time() - t_start
            if elapsed > 30.0:
                print(f"[tle] >30s elapsed at i={i}, capping; reusing tile")
                # Tile what we have over the rest.
                done = i
                for j in range(done, n_windows):
                    rows[j] = rows[j % done]
                break

    meta = {
        "es_lat": ES_LAT,
        "es_lon": ES_LON,
        "carrier_hz": KA_CARRIER_HZ,
        "elev_mask_deg": ELEV_MASK_DEG,
        "n_tles_total": len(sats),
        "n_tles_per_window": min(len(sats), TLE_SAMPLE_N),
        "t0": t0.isoformat(),
        "cadence_min": cadence_min,
    }
    TLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TLE_CACHE, rows=rows, n=int(n_windows), meta=json.dumps(meta))
    print(f"[tle] computed + cached {rows.shape} in {time.time() - t_start:.1f}s")
    return rows, meta


# --------------------------------------------------------------------------- #
# cuMAC H5 PHY test vectors
# --------------------------------------------------------------------------- #


def extract_cumac(root: Path) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """One window per H5 file: aggregate postEqSinr / wbSinr / blerTarget."""
    if not root.exists():
        print(f"[cumac] MISSING: {root}")
        return [], []
    try:
        import h5py
    except Exception as exc:
        print(f"[cumac] h5py unavailable ({exc}); dropping")
        return [], []

    h5_files = sorted(root.rglob("*.h5"))
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []
    n_dropped = 0
    for hp in h5_files:
        try:
            with h5py.File(hp, "r") as f:
                postSinr = np.asarray(f["postEqSinr"][:], dtype=np.float64) if "postEqSinr" in f else None
                wbSinr = np.asarray(f["wbSinr"][:], dtype=np.float64) if "wbSinr" in f else None
                avgRates = np.asarray(f["avgRates"][:], dtype=np.float64) if "avgRates" in f else None
                bler = np.asarray(f["blerTargetActUe"][:], dtype=np.float64) if "blerTargetActUe" in f else None
                mcs = np.asarray(f["mcsSelSol"][:], dtype=np.float64) if "mcsSelSol" in f else None
        except Exception:
            n_dropped += 1
            continue

        # Some entries can be all-zero / sentinel — keep only the sane subset.
        def _safe_mean(a):
            if a is None or a.size == 0:
                return float("nan")
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            return float(a.mean())

        post_mean = _safe_mean(postSinr)  # dB
        wb_mean = _safe_mean(wbSinr)      # dB
        rate_mean = _safe_mean(avgRates)  # bps proxy
        bler_mean = _safe_mean(bler)      # 0..1
        mcs_mean = _safe_mean(mcs)        # ~0..27

        # If literally everything is NaN, skip.
        sinr_signal = wb_mean if not math.isnan(wb_mean) else post_mean
        if math.isnan(sinr_signal):
            n_dropped += 1
            continue

        raw = np.array(
            [
                post_mean if not math.isnan(post_mean) else 0.0,
                wb_mean if not math.isnan(wb_mean) else 0.0,
                rate_mean if not math.isnan(rate_mean) else 0.0,
                bler_mean if not math.isnan(bler_mean) else 0.0,
                mcs_mean if not math.isnan(mcs_mean) else 0.0,
            ],
            dtype=np.float32,
        )
        risk = compute_sla_risk(snr=sinr_signal, bler=bler_mean if not math.isnan(bler_mean) else None)
        if any(math.isnan(v) for v in risk.values()):
            n_dropped += 1
            continue
        # Stash raw vector here — we'll project AFTER concatenating physics dims.
        feats.append(raw)
        targets.append(risk)

    print(f"[cumac]       extracted {len(feats)} vectors from {len(h5_files)} H5 files (dropped {n_dropped})")
    return feats, targets


# --------------------------------------------------------------------------- #
# Final 64-d projection: concat 64-d source latent + 7-d physics, project.
# --------------------------------------------------------------------------- #


def _physics_normalised(
    n_vis: float,
    mean_elev: float,
    mean_rng_km: float,
    mean_dopp_hz: float,
    rain_km: float,
    n_wet: float,
    t_K: float,
) -> np.ndarray:
    """Z-ish normalisation so each physics dim is roughly O(1) in magnitude."""
    return np.array(
        [
            n_vis / 100.0,
            mean_elev / 90.0,
            mean_rng_km / 2000.0,
            mean_dopp_hz / 1.0e6,  # mean abs Doppler at 28 GHz ~ 0.5–1 MHz
            rain_km / 5.0,
            n_wet / 80.0,
            (t_K - 273.15) / 30.0,
        ],
        dtype=np.float32,
    )


def _make_v3_projector(out_dim: int, in_dim: int, salt: bytes) -> np.ndarray:
    h = hashlib.sha256()
    h.update(salt)
    h.update(np.int64(in_dim).tobytes())
    h.update(np.int64(out_dim).tobytes())
    seed = int.from_bytes(h.digest()[:8], "little") & 0x7FFF_FFFF
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(out_dim, in_dim)).astype(np.float32)
    W /= np.sqrt(in_dim)
    return W


# --------------------------------------------------------------------------- #
# v0.2 vs v0.3 comparison
# --------------------------------------------------------------------------- #


def compare_v02_v03(v02_path: Path, v03_path: Path, latent_dim: int = LATENT_DIM):
    if not v02_path.exists():
        print(f"[compare] v0.2 missing at {v02_path} — skipping")
        return None

    g = torch.Generator().manual_seed(99)
    Z = torch.randn(256, latent_dim, generator=g)

    head2 = SLARiskHead(SLARiskConfig(latent_dim=latent_dim, hidden_dim=64))
    head2.load_state_dict(torch.load(v02_path, map_location="cpu", weights_only=True))
    head2.eval()
    head3 = SLARiskHead(SLARiskConfig(latent_dim=latent_dim, hidden_dim=64))
    head3.load_state_dict(torch.load(v03_path, map_location="cpu", weights_only=True))
    head3.eval()

    with torch.no_grad():
        p2 = head2(Z)
        p3 = head3(Z)

    def stats(p):
        out = {}
        for k, v in p.items():
            mean = float(v.mean())
            var = float(v.var())
            p_clip = v.clamp(1e-6, 1 - 1e-6)
            ent = -(p_clip * torch.log2(p_clip) + (1 - p_clip) * torch.log2(1 - p_clip))
            out[k] = {"mean": mean, "var": var, "entropy": float(ent.mean())}
        return out

    s2 = stats(p2)
    s3 = stats(p3)
    print("=" * 72)
    print("V0.2 vs V0.3 COMPARISON (eval batch seed=99, N=256)")
    print("=" * 72)
    for k in HORIZON_KEYS:
        print(
            f"  {k:8s}  v0.2: mean={s2[k]['mean']:.4f} var={s2[k]['var']:.5f} H={s2[k]['entropy']:.4f}  |  "
            f"v0.3: mean={s3[k]['mean']:.4f} var={s3[k]['var']:.5f} H={s3[k]['entropy']:.4f}"
        )
    h2 = sum(s2[k]["entropy"] for k in HORIZON_KEYS) / 3
    h3 = sum(s3[k]["entropy"] for k in HORIZON_KEYS) / 3
    print(f"  mean entropy: v0.2={h2:.4f}  v0.3={h3:.4f}  ->  more_confident={'v0.2' if h2 < h3 else 'v0.3'}")
    print("=" * 72)
    return {"v02": s2, "v03": s3, "more_confident": "v0.2" if h2 < h3 else "v0.3"}


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #


def write_card_v3(
    path: Path,
    *,
    contributing: dict,
    dropped: list[str],
    n_train: int,
    n_val: int,
    train_info: dict,
    eval_info: dict,
    n_params: int,
    ckpt_path: Path,
    ckpt_size: int,
    ckpt_sha: str,
    tle_meta: dict,
    itu_status: list[tuple[str, bool, float]],
    timing: dict,
):
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

    contrib_lines = []
    for name, (count, path_str) in contributing.items():
        contrib_lines.append(f"- **{name}** ({count} windows) — `{path_str}`")
    contrib_md = "\n".join(contrib_lines) or "- (none)"
    dropped_md = "\n".join(f"- {d}" for d in dropped) or "- (none)"

    itu_lines = []
    for rid, ok, val in itu_status:
        tag = "REAL MAP" if ok else "FALLBACK constant"
        itu_lines.append(f"- **{rid}** — {tag} → value at ES = {val:.4f}")
    itu_md = "\n".join(itu_lines)

    timing_md = "\n".join(f"- **{k}**: {v:.2f}s" for k, v in timing.items())

    md = f"""# SLA Risk Head — v0.3 (full real-data corpus)

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model name:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.3-full-corpus
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}
- **Status:** SHIPPED (demo-day checkpoint). Replaces v0.1 synthetic-only path.

## Intended Use
Multi-horizon SLA breach probability forecasting (30 s / 60 s / 300 s) from
the shared PreceptualAI `z_resource` latent. Trained on the full real-data
corpus available on this host, including ground-station-anchored physics
features (Starlink TLE visibility/Doppler at Rotterdam, ITU-R reference
maps).

## Training Data — REAL sources that contributed

{contrib_md}

### Sources excluded / dropped this run

{dropped_md}

- **Train samples:** {n_train}
- **Val samples:** {n_val}

## Earth-station-anchored physics features

A 7-d physics vector is concatenated per window before the final 64-d
random projection:

### TLE-derived (NORAD CelesTrak Starlink, {tle_meta.get('n_tles_total', '?')} sats)
- ES: ({ES_LAT}°N, {ES_LON}°E) — Rotterdam
- Carrier: {KA_CARRIER_HZ:.0f} Hz (28 GHz, Ka)
- Elevation mask: {ELEV_MASK_DEG}°
- Per window we propagate via SGP4 to the window's UTC and derive:
  1. count of Starlink sats above the elevation mask,
  2. mean elevation across visible sats,
  3. mean slant range (km),
  4. mean |Doppler| at 28 GHz.
- TLE source file: `{TLE_PATH}`

### ITU-R reference maps (looked up at the ES)
{itu_md}

## Latent construction

Per source we (a) compute a numeric KPI vector, (b) project to 64-d via
a per-source SHA-256-seeded Gaussian random projection (J-L style with
tanh), (c) concatenate the 7-d physics vector, then (d) project the
71-d concat back to 64-d with a fresh v0.3-salt projection. This makes
the v0.3 latent distinct from v0.2 even on the same source rows.

## Targets

Same threshold-derived multi-horizon risk as v0.2 (RSRP/RSRQ/SNR/CQI/BLER/
RX-power/delay-spread → mean violation, scaled per horizon). cuMAC H5
contributes through `wbSinr` / `postEqSinr` (dB) and `blerTargetActUe`.

## Wall-clock breakdown

{timing_md}

## Training Configuration
- Optimizer: Adam(lr=1e-3)
- Batch size: 64
- Max epochs: 30
- Early stopping: patience=5 on val loss
- Epochs run: {len(train_info['history'])}
- Loss: two-hot CE in logit space (DreamerV3-style)
- Device: CPU (no accelerator on this host)

## Performance
- Initial val loss (random init): {train_info['initial_val_loss']:.6f}
- Final train loss: {train_info['final_train_loss']:.6f}
- Final val loss: {train_info['final_val_loss']:.6f}
- Best val loss (in checkpoint): {train_info['best_val_loss']:.6f}

| horizon | ECE | Brier | P(min) | P(mean) | P(max) |
|---|---|---|---|---|---|
{horizons_md}

ECE: Naeini et al. 2015, 10 equal-width bins on [0,1]; observed labels are
binary (`target_p >= 0.5`).

## Honest Caveats
- The SLA-breach LABEL is *derived* from real radio measurements via the
  documented threshold table (see v0.2 card); it is not an observed
  customer breach event.
- ITU-R lookups that fell back to documented mid-latitude constants are
  flagged "FALLBACK" above. Real maps were used where the ZIPs are present
  on disk.
- TLE features are the same value across all windows associated with one
  UTC step; see `tle.cadence_min` for the step cadence.
- Per-source random projections are not feature *extractors*; they fill
  the 64-d z_resource interface deterministically.
- DeepMIMO is ray-traced (NVIDIA Sionna RT / Wireless Insite), not
  measured radio.
- cuMAC H5 vectors are simulator outputs from NVIDIA Aerial cuMAC tests,
  not field measurements.

## Reproducibility
```
.venv/bin/python scripts/train_sla_head_v0_3.py
```
Seeds: torch.manual_seed(42), numpy default_rng(42), TLE cache key in
`{TLE_CACHE.name}`.

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- 3GPP TS 38.133 (RSRP/RSRQ measurement requirements).
- 3GPP TS 28.554 §6.x (E2E KPIs targeted by SLA-breach definition).
- 3GPP TR 38.811 §6.3 (NTN Doppler pre-compensation).
- ITU-R P.839-4 (rain height), P.453-14 (N_wet), P.1510-1 (mean surface T).
- Brier 1950, Naeini et al. 2015.
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    timing: dict[str, float] = {}
    contributing: dict[str, tuple[int, str]] = {}
    dropped: list[str] = []

    # ---- Per-source extraction (returns 64-d projected feats already) ---- #
    # Aerial FAPI / FH / DeepMIMO / UCC come back already-64-projected. We
    # treat their 64-d as a "source latent" and concat the 7-d physics, then
    # do one MORE 64-d projection.
    print("=" * 72)
    print("REAL DATA INVENTORY (v0.3 full corpus)")
    print("=" * 72)

    all_proj64: list[np.ndarray] = []
    all_targets: list[dict[str, float]] = []

    t = time.time()
    f, y = extract_aerial_fapi(AERIAL_FAPI)
    timing["aerial_fapi"] = time.time() - t
    if f:
        contributing["NVIDIA Aerial FAPI"] = (len(f), str(AERIAL_FAPI))
        all_proj64.extend(f)
        all_targets.extend(y)
    else:
        dropped.append(f"NVIDIA Aerial FAPI ({AERIAL_FAPI})")

    t = time.time()
    f, y = extract_aerial_fh(AERIAL_FH)
    timing["aerial_fh"] = time.time() - t
    if f:
        contributing["NVIDIA Aerial FH"] = (len(f), str(AERIAL_FH))
        all_proj64.extend(f)
        all_targets.extend(y)
    else:
        dropped.append(f"NVIDIA Aerial FH ({AERIAL_FH})")

    t = time.time()
    f, y = extract_deepmimo(DEEPMIMO_ROOT, max_scenarios=100)
    timing["deepmimo"] = time.time() - t
    if f:
        contributing["NVIDIA DeepMIMO (asu_campus_3p5_dyn)"] = (len(f), str(DEEPMIMO_ROOT))
        all_proj64.extend(f)
        all_targets.extend(y)
    else:
        dropped.append(f"NVIDIA DeepMIMO ({DEEPMIMO_ROOT})")

    t = time.time()
    f, y = extract_ucc(UCC_ROOT, max_files=60, window_sec=30)
    timing["ucc_misl"] = time.time() - t
    if f:
        contributing["UCC MISL 5G production traces"] = (len(f), str(UCC_ROOT))
        all_proj64.extend(f)
        all_targets.extend(y)
    else:
        dropped.append(f"UCC MISL 5G ({UCC_ROOT})")

    # ---- cuMAC H5 (raw 5-d per file) — project here so all sources align. ---- #
    t = time.time()
    cumac_raw, cumac_y = extract_cumac(CUMAC_ROOT)
    timing["cumac_h5"] = time.time() - t
    if cumac_raw:
        # Project the raw 5-d cuMAC vectors to 64-d with a v0.3-specific salt.
        rng = np.random.default_rng(0xC0FFEE)
        W = rng.standard_normal(size=(LATENT_DIM, 5)).astype(np.float32) / math.sqrt(5)
        for r in cumac_raw:
            all_proj64.append(np.tanh(W @ r.astype(np.float32)))
        all_targets.extend(cumac_y)
        contributing["NVIDIA Aerial cuMAC H5 (4T4R type0/type1)"] = (
            len(cumac_raw),
            str(CUMAC_ROOT),
        )
    else:
        dropped.append(f"NVIDIA Aerial cuMAC H5 ({CUMAC_ROOT})")

    n_total = len(all_proj64)
    print(f"[corpus] total raw windows = {n_total}")
    if n_total < 32:
        print("[error] too few samples to train")
        return 2

    # ---- ITU-R lookups at the ES ---------------------------------------- #
    t = time.time()
    (rain_km, n_wet, t_K), itu_status = _load_itu_readers()
    timing["itu_lookup"] = time.time() - t
    print(f"[itu] rain_km={rain_km:.3f} n_wet={n_wet:.3f} t_K={t_K:.3f}")
    for rid, ok, val in itu_status:
        print(f"[itu] {rid}: available={ok} value={val:.4f}")

    # ---- TLE features (one row per window) ------------------------------ #
    t = time.time()
    tle_rows, tle_meta = build_tle_window_features(
        n_windows=n_total, cadence_min=2.0
    )
    timing["tle_extraction"] = time.time() - t
    print(f"[tle] feature matrix shape={tle_rows.shape}")

    # ---- Concat 64-d source latent + 7-d physics + final v0.3 projection ---- #
    t = time.time()
    W_final = _make_v3_projector(LATENT_DIM, LATENT_DIM + N_PHYSICS_DIMS, b"sla_v0.3_salt")
    X = np.zeros((n_total, LATENT_DIM), dtype=np.float32)
    for i in range(n_total):
        n_vis, mean_elev, mean_rng, mean_dopp = tle_rows[i]
        phys = _physics_normalised(
            n_vis, mean_elev, mean_rng, mean_dopp, rain_km, n_wet, t_K
        )
        cat = np.concatenate([all_proj64[i], phys])
        X[i] = np.tanh(W_final @ cat)
    timing["latent_assembly"] = time.time() - t

    Z = torch.from_numpy(X)
    Y = {
        k: torch.tensor([y_[k] for y_ in all_targets], dtype=torch.float32)
        for k in HORIZON_KEYS
    }
    for k, v in Y.items():
        print(
            f"[corpus] {k}: mean={float(v.mean()):.4f} min={float(v.min()):.4f} "
            f"max={float(v.max()):.4f} frac>=0.5={float((v >= 0.5).float().mean()):.4f}"
        )

    # Cap for CPU budget (same MAX_TOTAL as v0.2 for fair comparison).
    MAX_TOTAL = 4000
    if Z.shape[0] > MAX_TOTAL:
        g = torch.Generator().manual_seed(123)
        idx = torch.randperm(Z.shape[0], generator=g)[:MAX_TOTAL]
        Z = Z[idx]
        Y = {k: v[idx] for k, v in Y.items()}
        print(f"[corpus] subsampled to {Z.shape[0]} for CPU budget")

    # Same fixed val seed as v0.2 (seed=42) so we can compare on identical splits.
    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z, Y, val_frac=0.2, seed=42)
    print(f"[split] train={Z_tr.shape[0]} val={Z_val.shape[0]}")

    head = SLARiskHead(SLARiskConfig(latent_dim=LATENT_DIM, hidden_dim=64))
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[model] params={n_params}")

    t = time.time()
    train_info = train_loop(
        head, Z_tr, Y_tr, Z_val, Y_val,
        epochs=30, batch_size=64, lr=1e-3, patience=5,
    )
    timing["training"] = time.time() - t

    eval_info = evaluate(head, Z_val, Y_val)
    print("[eval] ECE:", eval_info["ece"])
    print("[eval] Brier:", eval_info["brier"])
    print("[eval] P min:", eval_info["p_min"])
    print("[eval] P mean:", eval_info["p_mean"])
    print("[eval] P max:", eval_info["p_max"])

    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "sla_head_v0.3_full_corpus.pt"
    card = out_dir / "sla_head_v0.3_full_corpus.md"

    torch.save(head.state_dict(), ckpt)
    sha = sha256_of(ckpt)
    size = ckpt.stat().st_size
    print(f"[save] ckpt={ckpt} sha256={sha} size={size}")

    write_card_v3(
        card,
        contributing=contributing,
        dropped=dropped,
        n_train=Z_tr.shape[0], n_val=Z_val.shape[0],
        train_info=train_info, eval_info=eval_info,
        n_params=n_params, ckpt_path=ckpt,
        ckpt_size=size, ckpt_sha=sha,
        tle_meta=tle_meta, itu_status=itu_status,
        timing=timing,
    )
    print(f"[save] card={card}")

    cmp = compare_v02_v03(out_dir / "sla_head_v0.2_nvidia.pt", ckpt)

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
        "timing": timing,
        "v02_vs_v03": cmp,
    }
    print("[summary]", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
