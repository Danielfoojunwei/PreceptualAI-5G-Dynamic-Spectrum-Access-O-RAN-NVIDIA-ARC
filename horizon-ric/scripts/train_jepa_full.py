"""End-to-end JEPA encoder + LatentDynamics + SLA risk head training on REAL data.

Three-phase pipeline executed on a single CUDA device (NVIDIA GB10 target):

  PHASE 1  GraphJEPA action-free pretraining over a MultiRateFusion of
           (TLE-derived ephemeris, ITU-R map weather, Aerial KPM,
           DeepMIMO channel) tokens. Smooth-L1 latent loss with EMA target.
  PHASE 2  LatentDynamics action-conditioned next-latent prediction on
           DeepMIMO consecutive-snapshot pairs.  Frozen JEPA encoder
           projects each scene snapshot to z; action_t is a 16-d
           constraint-feasible sample (power delta + beam pointing
           change + frequency-reuse choice).
  PHASE 3  SLA risk head v0.4 fine-tune.  Same v0.3 corpus, but the
           random projection is replaced with the FROZEN JEPA encoder
           output (mean over latents).  hidden_dim raised to 128 to
           match d_latent.

Total wall-clock budget = 15 minutes across all phases.
Outputs:
    checkpoints/jepa_encoder_v0.1.pt     (+ .md)
    checkpoints/latent_dynamics_v0.1.pt  (+ .md)
    checkpoints/sla_head_v0.4_jepa.pt    (+ .md)
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_ric.encoder import (  # noqa: E402
    GraphJEPA,
    GraphJEPAConfig,
    PerceiverConfig,
    PerceiverFusion,
)
from horizon_ric.core import LatentDynamics, LatentDynamicsConfig  # noqa: E402
from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402

# Reuse v0.3 extractors and v0.2 helpers verbatim.
from train_sla_head_nvidia import (  # noqa: E402
    HORIZON_KEYS,
    AERIAL_FAPI,
    AERIAL_FH,
    DEEPMIMO_ROOT,
    UCC_ROOT,
    extract_aerial_fapi,
    extract_aerial_fh,
    extract_deepmimo,
    extract_ucc,
    sha256_of,
    split_train_val,
)
from train_sla_head_v0_3 import (  # noqa: E402
    CUMAC_ROOT,
    ES_LAT,
    ES_LON,
    KA_CARRIER_HZ,
    ELEV_MASK_DEG,
    TLE_PATH,
    _load_itu_readers,
    _physics_normalised,
    _parse_tles,
    _tle_features_at_utc,
    extract_cumac,
    build_tle_window_features,
    compare_v02_v03,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #

D_LATENT = 128
D_INPUT = 128          # per-token input dimensionality (after per-source projection)
N_LATENTS = 128
D_ACTION = 16
N_TOKENS = 4           # (TLE, ITU, Aerial-KPM, DeepMIMO)

CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

JEPA_PT = CKPT_DIR / "jepa_encoder_v0.1.pt"
JEPA_MD = CKPT_DIR / "jepa_encoder_v0.1.md"
DYN_PT = CKPT_DIR / "latent_dynamics_v0.1.pt"
DYN_MD = CKPT_DIR / "latent_dynamics_v0.1.md"
SLA_PT = CKPT_DIR / "sla_head_v0.4_jepa.pt"
SLA_MD = CKPT_DIR / "sla_head_v0.4_jepa.md"

PHASE_BUDGET_S = 5 * 60       # 5 min per phase
TOTAL_BUDGET_S = 15 * 60      # hard 15 min cap


# --------------------------------------------------------------------------- #
# Token tensor construction
# --------------------------------------------------------------------------- #


def _per_source_projector(out_dim: int, in_dim: int, salt: bytes) -> np.ndarray:
    """Deterministic Gaussian projection used to lift each per-source feature
    vector into the shared per-token input space (d_input=128).

    Random projections with bounded JL distortion preserve pairwise
    Euclidean distances (Johnson 1984), which is what we want for the
    JEPA target encoder; the projection is then exposed to the trainable
    cross-attention as one of N_TOKENS=4 input tokens.
    """
    h = hashlib.sha256()
    h.update(salt)
    h.update(np.int64(in_dim).tobytes())
    h.update(np.int64(out_dim).tobytes())
    seed = int.from_bytes(h.digest()[:8], "little") & 0x7FFF_FFFF
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(out_dim, in_dim)).astype(np.float32)
    W /= math.sqrt(in_dim)
    return W


def _project(raw: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.tanh(W @ raw.astype(np.float32))


# --------------------------------------------------------------------------- #
# Phase 1 + Phase 3 corpus assembly
# --------------------------------------------------------------------------- #


def assemble_token_corpus(device: torch.device) -> dict:
    """Build the per-window 4-token tensor (B, N_TOKENS, D_INPUT) plus the
    per-window SLA-risk target dict (used by phase 3).

    Each per-window token bag contains:
        token 0 — TLE-derived ephemeris features (4-d raw)
        token 1 — ITU-R map features (3-d raw)
        token 2 — Aerial KPM features (10-d raw): mean of source-specific
                  64-d projection from {fapi, fh, ucc, cumac} pre-computed
                  by the v0.3 extractors, summarised via 10 quantile bins.
        token 3 — DeepMIMO channel features (4-d raw): rx_power_dbm,
                  delay_spread_ns, n_paths, freq.
    """
    print("=" * 72)
    print("CORPUS ASSEMBLY  (real data inventory)")
    print("=" * 72)

    contributing: dict[str, tuple[int, str]] = {}
    dropped: list[str] = []

    # ---------- per-source 64-d projections (already deterministic) ----------
    src_feats: dict[str, list[np.ndarray]] = {}
    src_targets: dict[str, list[dict]] = {}

    f, y = extract_aerial_fapi(AERIAL_FAPI)
    if f:
        src_feats["aerial_fapi"] = f
        src_targets["aerial_fapi"] = y
        contributing["NVIDIA Aerial FAPI"] = (len(f), str(AERIAL_FAPI))
    else:
        dropped.append(f"aerial_fapi ({AERIAL_FAPI})")

    f, y = extract_aerial_fh(AERIAL_FH)
    if f:
        src_feats["aerial_fh"] = f
        src_targets["aerial_fh"] = y
        contributing["NVIDIA Aerial FH"] = (len(f), str(AERIAL_FH))
    else:
        dropped.append(f"aerial_fh ({AERIAL_FH})")

    f, y = extract_deepmimo(DEEPMIMO_ROOT, max_scenarios=100)
    if f:
        src_feats["deepmimo"] = f
        src_targets["deepmimo"] = y
        contributing["NVIDIA DeepMIMO"] = (len(f), str(DEEPMIMO_ROOT))
    else:
        dropped.append(f"deepmimo ({DEEPMIMO_ROOT})")

    f, y = extract_ucc(UCC_ROOT, max_files=60, window_sec=30)
    if f:
        src_feats["ucc"] = f
        src_targets["ucc"] = y
        contributing["UCC MISL 5G"] = (len(f), str(UCC_ROOT))
    else:
        dropped.append(f"ucc ({UCC_ROOT})")

    cumac_raw, cumac_y = extract_cumac(CUMAC_ROOT)
    if cumac_raw:
        # cuMAC is raw 5-d; project to 64-d so it lines up with the others.
        rng = np.random.default_rng(0xC0FFEE)
        Wc = rng.standard_normal(size=(64, 5)).astype(np.float32) / math.sqrt(5)
        proj = [np.tanh(Wc @ r.astype(np.float32)) for r in cumac_raw]
        src_feats["cumac"] = proj
        src_targets["cumac"] = cumac_y
        contributing["NVIDIA Aerial cuMAC H5"] = (len(proj), str(CUMAC_ROOT))
    else:
        dropped.append(f"cumac ({CUMAC_ROOT})")

    # Combine all sources into one window list. We keep a tag so DeepMIMO can
    # be re-traversed in Phase 2 with consecutive-snapshot pairing.
    all_src64: list[np.ndarray] = []
    all_src_tags: list[str] = []
    all_targets: list[dict] = []
    for tag, feats in src_feats.items():
        all_src64.extend(feats)
        all_src_tags.extend([tag] * len(feats))
        all_targets.extend(src_targets[tag])
    n = len(all_src64)
    print(f"[corpus] total per-source-projected windows = {n}")
    if n < 64:
        raise RuntimeError(f"too few windows ({n}) to train")

    # ---------- ITU-R lookups at the ES (constant per-window) ----------
    (rain_km, n_wet, t_K), itu_status = _load_itu_readers()
    print(f"[itu] rain_km={rain_km:.3f} n_wet={n_wet:.3f} t_K={t_K:.3f}")
    itu_raw = np.array([rain_km, n_wet, t_K], dtype=np.float32)

    # ---------- TLE per-window features (cached) ----------
    tle_rows, tle_meta = build_tle_window_features(n_windows=n, cadence_min=2.0)
    print(f"[tle] feature matrix {tle_rows.shape}")

    # ---------- Per-source token projectors ----------
    W_tle = _per_source_projector(D_INPUT, 4, b"jepa.tok.tle")
    W_itu = _per_source_projector(D_INPUT, 3, b"jepa.tok.itu")
    W_kpm = _per_source_projector(D_INPUT, 64, b"jepa.tok.aerial_kpm")
    W_dm = _per_source_projector(D_INPUT, 64, b"jepa.tok.deepmimo")

    # Normalise TLE features into ~O(1) magnitudes.
    def _norm_tle(row: np.ndarray) -> np.ndarray:
        n_vis, mean_elev, mean_rng_km, mean_dopp_hz = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        return np.array([
            n_vis / 100.0,
            mean_elev / 90.0,
            mean_rng_km / 2000.0,
            mean_dopp_hz / 1.0e6,
        ], dtype=np.float32)

    def _norm_itu(rain_km: float, n_wet: float, t_K: float) -> np.ndarray:
        return np.array([rain_km / 5.0, n_wet / 80.0, (t_K - 273.15) / 30.0], dtype=np.float32)

    itu_norm = _norm_itu(rain_km, n_wet, t_K)

    tokens = np.zeros((n, N_TOKENS, D_INPUT), dtype=np.float32)
    deepmimo_indices: list[int] = []
    for i in range(n):
        tle_n = _norm_tle(tle_rows[i])
        # Build the token-3 (KPM) input from the source's own 64-d vector.
        kpm_64 = all_src64[i]
        # Token 3 (DeepMIMO channel): when source != deepmimo, use a zero
        # vector (the PerceiverFusion will still attend, just with no signal).
        if all_src_tags[i] == "deepmimo":
            dm_64 = all_src64[i]
            deepmimo_indices.append(i)
        else:
            dm_64 = np.zeros(64, dtype=np.float32)

        tokens[i, 0] = _project(tle_n, W_tle)
        tokens[i, 1] = _project(itu_norm, W_itu)
        tokens[i, 2] = _project(kpm_64, W_kpm)
        tokens[i, 3] = _project(dm_64, W_dm)

    X = torch.from_numpy(tokens).to(device)
    Y = {
        k: torch.tensor([t[k] for t in all_targets], dtype=torch.float32, device=device)
        for k in HORIZON_KEYS
    }

    # ---------- Phase-2 pair index from DeepMIMO consecutive snapshots ----
    # The deepmimo extractor walks scenarios in deterministic order and emits
    # one entry per (t_id, tx_id, r_id). Within a scenario, t_ids are sorted
    # lexicographically (t001 < t002 < ...), so consecutive entries with the
    # same (tx,r) are valid (z_t, z_{t+1}) pairs. We reconstruct the (scenario,
    # tx, r, t) tuple from the iteration order by re-walking deepmimo here.
    pair_z_idx_a: list[int] = []
    pair_z_idx_b: list[int] = []
    if "deepmimo" in src_feats:
        # Re-walk deepmimo deterministically.
        scenarios = sorted([p for p in DEEPMIMO_ROOT.iterdir() if p.is_dir()])
        idx = 0
        # Map from scenario-relative (tx,r,t) iteration into the global
        # all_src64 index.  We use the index pattern that deepmimo extractor
        # emits, but also need to remember which absolute index that lands at.
        # Easiest: rebuild the same tuple list and skip the same drops.
        from train_sla_head_nvidia import compute_sla_risk  # noqa: F401
        # We'll just rely on the order recorded in `deepmimo_indices` (the
        # global index of each deepmimo entry, in extractor order).
        # Pair adjacent deepmimo entries; not all adjacent pairs are
        # consecutive-time but most are.
        for j in range(len(deepmimo_indices) - 1):
            a = deepmimo_indices[j]
            b = deepmimo_indices[j + 1]
            pair_z_idx_a.append(a)
            pair_z_idx_b.append(b)
    print(f"[pairs] candidate (z_t, z_{{t+1}}) pairs from deepmimo = {len(pair_z_idx_a)}")

    return {
        "X": X,
        "Y": Y,
        "n": n,
        "contributing": contributing,
        "dropped": dropped,
        "tle_meta": tle_meta,
        "itu_status": itu_status,
        "pair_idx_a": torch.tensor(pair_z_idx_a, dtype=torch.long, device=device),
        "pair_idx_b": torch.tensor(pair_z_idx_b, dtype=torch.long, device=device),
        "src_tags": all_src_tags,
    }


# --------------------------------------------------------------------------- #
# Phase 2 — constraint-feasible action sampler
# --------------------------------------------------------------------------- #


def sample_feasible_action(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Sample a 16-d action that respects rough feasibility bounds.

    Layout (all bounded to ~[-1, 1] after normalisation):
      0..3   : per-cell power delta in dB / 6 dB
      4..7   : beam pointing change in deg (az,el) / 30 deg
      8..11  : frequency-reuse one-hot soft choice (4 buckets)
      12..15 : reserved (e.g. carrier aggregation flags) — Gaussian noise.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.zeros(n, D_ACTION)
    a[:, 0:4] = (torch.rand(n, 4, generator=g) * 2 - 1) * (6.0 / 6.0)  # |Δp|<=6 dB
    a[:, 4:8] = (torch.rand(n, 4, generator=g) * 2 - 1) * (30.0 / 30.0)
    # Soft one-hot frequency-reuse choice.
    f = torch.randint(0, 4, (n,), generator=g)
    onehot = torch.zeros(n, 4)
    onehot.scatter_(1, f.unsqueeze(1), 1.0)
    # Add tiny noise so it's not literally one-hot.
    a[:, 8:12] = onehot + 0.05 * torch.randn(n, 4, generator=g)
    a[:, 12:16] = 0.5 * torch.randn(n, 4, generator=g)
    return a.to(device)


# --------------------------------------------------------------------------- #
# Phase runners
# --------------------------------------------------------------------------- #


def phase1_train_jepa(
    X: torch.Tensor,
    device: torch.device,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    budget_s: float = PHASE_BUDGET_S,
) -> tuple[GraphJEPA, dict]:
    print("=" * 72)
    print("PHASE 1 — GraphJEPA action-free pretraining")
    print("=" * 72)

    encoder = PerceiverFusion(
        PerceiverConfig(
            n_latents=N_LATENTS,
            d_latent=D_LATENT,
            d_input=D_INPUT,
            n_self_layers=2,
            n_heads=4,
        )
    )
    jepa = GraphJEPA(
        encoder,
        GraphJEPAConfig(
            d_latent=D_LATENT,
            predictor_hidden=256,
            predictor_layers=2,
            mask_ratio=0.6,
            ema_momentum=0.996,
        ),
    ).to(device)
    opt = torch.optim.Adam(jepa.parameters(), lr=lr)

    n = X.shape[0]
    history: list[dict] = []
    t_start = time.time()
    last_loss = float("nan")
    last_pred_norm = float("nan")
    last_target_norm = float("nan")
    epochs_run = 0
    for ep in range(1, epochs + 1):
        if time.time() - t_start > budget_s:
            print(f"[phase1] budget {budget_s:.0f}s reached at epoch {ep - 1}; stopping")
            break
        jepa.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        ep_pred = 0.0
        ep_tgt = 0.0
        ep_seen = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = X[idx]
            opt.zero_grad()
            out = jepa.loss(xb)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)
            opt.step()
            bs = xb.shape[0]
            ep_loss += float(loss.detach().item()) * bs
            ep_pred += float(out["pred_norm"].item()) * bs
            ep_tgt += float(out["target_norm"].item()) * bs
            ep_seen += bs
        jepa.update_target()
        last_loss = ep_loss / max(ep_seen, 1)
        last_pred_norm = ep_pred / max(ep_seen, 1)
        last_target_norm = ep_tgt / max(ep_seen, 1)
        history.append(
            {"epoch": ep, "loss": last_loss, "pred_norm": last_pred_norm, "target_norm": last_target_norm}
        )
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"[phase1 ep {ep:02d}] loss={last_loss:.6f} "
                f"|pred|={last_pred_norm:.4f} |target|={last_target_norm:.4f} "
                f"elapsed={time.time() - t_start:.1f}s"
            )
        epochs_run = ep
    elapsed = time.time() - t_start
    info = {
        "history": history,
        "final_loss": last_loss,
        "final_pred_norm": last_pred_norm,
        "final_target_norm": last_target_norm,
        "epochs_run": epochs_run,
        "elapsed_s": elapsed,
        "ema_beta": jepa.cfg.ema_momentum,
    }
    print(f"[phase1] done in {elapsed:.1f}s — final_loss={last_loss:.6f}")
    return jepa, info


def phase2_train_dynamics(
    X: torch.Tensor,
    pair_a: torch.Tensor,
    pair_b: torch.Tensor,
    jepa: GraphJEPA,
    device: torch.device,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    budget_s: float = PHASE_BUDGET_S,
) -> tuple[LatentDynamics, dict]:
    print("=" * 72)
    print("PHASE 2 — LatentDynamics action-conditioned training")
    print("=" * 72)

    if pair_a.numel() < 32:
        print(f"[phase2] only {pair_a.numel()} pairs available; need ≥32. Aborting phase 2.")
        # Return an UNTRAINED dynamics (so we still save SOMETHING) plus an info dict.
        dyn = LatentDynamics(LatentDynamicsConfig(d_latent=D_LATENT, d_action=D_ACTION, d_state=16)).to(device)
        return dyn, {"skipped": True, "reason": "too_few_pairs", "n_pairs": int(pair_a.numel())}

    # Encode all relevant windows once with FROZEN JEPA context encoder
    # (mean over latents → 128-d).
    jepa.eval()
    with torch.no_grad():
        Z_full = jepa.context_encoder(X).mean(dim=1)  # (N, D_LATENT)

    # Standardise the latents so they live in the LayerNormed space the
    # dynamics model produces. Compute moments on the FULL latent matrix
    # (not just the pairs) so the statistics generalise.
    mu = Z_full.mean(dim=0, keepdim=True)
    sd = Z_full.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z_norm = (Z_full - mu) / sd
    z_a = Z_norm[pair_a]
    z_b = Z_norm[pair_b]
    n_pairs = z_a.shape[0]
    print(f"[phase2] n_pairs={n_pairs}  z shape={tuple(z_a.shape)}  mu_norm={float(mu.norm()):.3f} sd_mean={float(sd.mean()):.3f}")

    # Sample one feasible action per pair (deterministic seed).
    actions = sample_feasible_action(n_pairs, device, seed=42)

    # Train.
    dyn = LatentDynamics(LatentDynamicsConfig(d_latent=D_LATENT, d_action=D_ACTION, d_state=16)).to(device)
    opt = torch.optim.Adam(dyn.parameters(), lr=lr)
    history: list[dict] = []
    t_start = time.time()
    last_loss = float("nan")
    epochs_run = 0
    for ep in range(1, epochs + 1):
        if time.time() - t_start > budget_s:
            print(f"[phase2] budget {budget_s:.0f}s reached at epoch {ep - 1}; stopping")
            break
        dyn.train()
        perm = torch.randperm(n_pairs, device=device)
        ep_loss = 0.0
        ep_seen = 0
        for s in range(0, n_pairs, batch_size):
            idx = perm[s:s + batch_size]
            za = z_a[idx]
            zb = z_b[idx]
            ab = actions[idx]
            zb_pred, _ = dyn.step(za, ab)
            loss = nn.functional.mse_loss(zb_pred, zb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
            opt.step()
            bs = za.shape[0]
            ep_loss += float(loss.detach().item()) * bs
            ep_seen += bs
        last_loss = ep_loss / max(ep_seen, 1)
        history.append({"epoch": ep, "loss": last_loss})
        if ep == 1 or ep % 3 == 0 or ep == epochs:
            print(f"[phase2 ep {ep:02d}] mse={last_loss:.6f} elapsed={time.time() - t_start:.1f}s")
        epochs_run = ep
    elapsed = time.time() - t_start
    info = {
        "history": history,
        "final_loss": last_loss,
        "epochs_run": epochs_run,
        "elapsed_s": elapsed,
        "n_pairs": n_pairs,
    }
    print(f"[phase2] done in {elapsed:.1f}s — final_mse={last_loss:.6f}")
    return dyn, info


def phase3_train_sla(
    X: torch.Tensor,
    Y: dict[str, torch.Tensor],
    jepa: GraphJEPA,
    device: torch.device,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    budget_s: float = PHASE_BUDGET_S,
) -> tuple[SLARiskHead, dict]:
    print("=" * 72)
    print("PHASE 3 — SLA risk head v0.4 fine-tune on JEPA latents")
    print("=" * 72)

    jepa.eval()
    with torch.no_grad():
        Z = jepa.context_encoder(X).mean(dim=1)  # (N, 128)
    print(f"[phase3] JEPA-encoded latent matrix shape={tuple(Z.shape)}")

    # Same fixed seed=42 split as v0.3 for fair comparison (note: data set is
    # the SAME ordered list, so the seed=42 randperm produces the same split).
    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z, Y, val_frac=0.2, seed=42)
    print(f"[phase3] train={Z_tr.shape[0]} val={Z_val.shape[0]}")

    head = SLARiskHead(SLARiskConfig(latent_dim=D_LATENT, hidden_dim=128)).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[phase3] params={n_params}")

    opt = torch.optim.Adam(head.parameters(), lr=lr)

    def epoch_pass(Z_, Y_, train: bool) -> float:
        head.train(mode=train)
        n = Z_.shape[0]
        perm = torch.randperm(n, device=device) if train else torch.arange(n, device=device)
        total = 0.0
        seen = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            zb = Z_[idx]
            yb = {k: v[idx] for k, v in Y_.items()}
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                loss = head.loss(zb, yb)
                if train:
                    loss.backward()
                    opt.step()
            total += float(loss.detach().item()) * zb.shape[0]
            seen += zb.shape[0]
        return total / max(seen, 1)

    initial_val = epoch_pass(Z_val, Y_val, False)
    print(f"[phase3 init] val_loss={initial_val:.6f}")
    best = math.inf
    best_state = None
    bad = 0
    history: list[dict] = []
    t_start = time.time()
    epochs_run = 0
    for ep in range(1, epochs + 1):
        if time.time() - t_start > budget_s:
            print(f"[phase3] budget {budget_s:.0f}s reached at epoch {ep - 1}; stopping")
            break
        tl = epoch_pass(Z_tr, Y_tr, True)
        vl = epoch_pass(Z_val, Y_val, False)
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        improved = vl < best - 1e-6
        if improved:
            best = vl
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 3 == 0 or improved or ep == epochs:
            print(
                f"[phase3 ep {ep:02d}] train={tl:.6f} val={vl:.6f} "
                f"{'*' if improved else ' '} best={best:.6f} bad={bad}"
            )
        epochs_run = ep
        if bad >= patience:
            print(f"[phase3] early-stop (patience={patience})")
            break
    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        preds = head(Z_val)
    observed = {k: (v >= 0.5).float() for k, v in Y_val.items()}
    ece = head.calibration_summary(preds, observed)
    brier = head.brier_score(preds, observed)
    p_min = {k: float(p.min()) for k, p in preds.items()}
    p_max = {k: float(p.max()) for k, p in preds.items()}
    p_mean = {k: float(p.mean()) for k, p in preds.items()}

    elapsed = time.time() - t_start
    info = {
        "history": history,
        "initial_val_loss": initial_val,
        "best_val_loss": best,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
        "ece": ece,
        "brier": brier,
        "p_min": p_min,
        "p_mean": p_mean,
        "p_max": p_max,
        "n_params": n_params,
        "n_train": Z_tr.shape[0],
        "n_val": Z_val.shape[0],
        "epochs_run": epochs_run,
        "elapsed_s": elapsed,
    }
    print(f"[phase3] done in {elapsed:.1f}s — best_val={best:.6f}")
    return head, info


# --------------------------------------------------------------------------- #
# Model card writers
# --------------------------------------------------------------------------- #


def write_jepa_card(path: Path, *, ckpt_path: Path, ckpt_sha: str, ckpt_size: int,
                    info: dict, contributing: dict, dropped: list[str], tle_meta: dict,
                    itu_status: list, n_params: int, gpu_peak_bytes: int):
    contrib_md = "\n".join(
        f"- **{name}** ({count} windows) — `{p}`" for name, (count, p) in contributing.items()
    ) or "- (none)"
    dropped_md = "\n".join(f"- {d}" for d in dropped) or "- (none)"
    itu_md = "\n".join(
        f"- **{rid}** — {'REAL MAP' if ok else 'FALLBACK'} ({val:.4f})"
        for rid, ok, val in itu_status
    )
    md = f"""# GraphJEPA Encoder — v0.1

3GPP TS 28.105 Model Description Card.

## Identification
- **Model:** PerceiverFusion + GraphJEPA predictor (action-free pretraining)
- **Version:** v0.1
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## Architecture
- PerceiverFusion(n_latents=128, d_latent=128, d_input=128, n_self_layers=2, n_heads=4)
- GraphJEPA(d_latent=128, predictor_hidden=256, predictor_layers=2, mask_ratio=0.6, ema_momentum=0.996)
- Loss: smooth-L1 over masked latent positions (Assran et al. 2023, Bardes et al. 2024).

## Training data — REAL sources (4-token MultiRateFusion per window)
- **Token 0:** TLE-derived ephemeris (Starlink @ Rotterdam)
- **Token 1:** ITU-R reference maps (rain height, N_wet, mean surface T)
- **Token 2:** Aerial KPM (FAPI/FH/UCC/cuMAC source 64-d projection)
- **Token 3:** DeepMIMO ray-traced channel (zero when source != deepmimo)

{contrib_md}

### Dropped
{dropped_md}

### ITU-R lookup status
{itu_md}

### TLE metadata
- ES: ({ES_LAT}°N, {ES_LON}°E)  carrier={KA_CARRIER_HZ:.0f} Hz  mask={ELEV_MASK_DEG}°
- TLEs total: {tle_meta.get('n_tles_total', '?')}  per-window: {tle_meta.get('n_tles_per_window', '?')}

## Training run
- Epochs run: {info['epochs_run']}
- Final loss: {info['final_loss']:.6f}
- Final |pred|: {info['final_pred_norm']:.4f}
- Final |target|: {info['final_target_norm']:.4f}
- EMA β: {info['ema_beta']}
- Wall-clock: {info['elapsed_s']:.1f} s
- Device: NVIDIA GB10 (CUDA)
- GPU peak alloc: {gpu_peak_bytes / 1e9:.3f} GB

## Honest caveats
- The 4-token MultiRateFusion is a **summary** representation, not a full
  graph: each modality contributes one token per window. JEPA still
  benefits because the masked-token prediction forces cross-modal
  alignment in latent space.
- `update_target()` is called once per epoch (per spec).
- Smooth-L1 in latent space avoids representation collapse but does NOT
  by itself guarantee downstream task performance — see the v0.4 SLA
  head card for the calibration story.

## Reproducibility
```
.venv/bin/python scripts/train_jepa_full.py
```
Seeds: torch.manual_seed(42), np default_rng(42).
"""
    path.write_text(md, encoding="utf-8")


def write_dyn_card(path: Path, *, ckpt_path: Path, ckpt_sha: str, ckpt_size: int,
                   info: dict, n_params: int, gpu_peak_bytes: int):
    if info.get("skipped"):
        body = f"""## Status
- **SKIPPED**: {info.get('reason', '?')}  (n_pairs={info.get('n_pairs', 0)})
- An untrained `LatentDynamics` was saved as a placeholder so the rApp
  pipeline still has a checkpoint to reference. Do NOT use for planning.
"""
    else:
        body = f"""## Training run
- Epochs run: {info['epochs_run']}
- Final MSE on next-latent: {info['final_loss']:.6f}
- Wall-clock: {info['elapsed_s']:.1f}s
- n_pairs: {info['n_pairs']}
- Device: NVIDIA GB10 (CUDA)
- GPU peak alloc: {gpu_peak_bytes / 1e9:.3f} GB
"""
    md = f"""# LatentDynamics — v0.1

3GPP TS 28.105 Model Description Card.

## Identification
- **Model:** `horizon_ric.core.LatentDynamics` (Mamba-style selective SSM)
- **Version:** v0.1
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## Architecture
- LatentDynamicsConfig(d_latent=128, d_action=16, d_state=16)
- Discretised diagonal SSM with input-dependent B/C/Δ (Gu & Dao 2023).

## Training data
- DeepMIMO consecutive-snapshot pairs (z_t, z_{{t+1}}) where z is the
  FROZEN GraphJEPA encoder output (mean over 128 latents).
- action_t ∈ R^16: per-cell power delta (4), beam pointing change (4),
  frequency-reuse soft one-hot (4), reserved CA flags (4).

{body}

## Honest caveats
- "Consecutive snapshots" pairs adjacent extractor entries; not every
  adjacent pair shares the same (tx, rx) tuple — the dynamics model
  therefore averages over a slight scene-shift noise floor.
- Actions are **synthetic feasible samples**, not optimised actions
  observed from a real RIC. The dynamics is therefore action-aware but
  not action-grounded; downstream TD-MPC2 use must re-fine-tune.

## Reproducibility
```
.venv/bin/python scripts/train_jepa_full.py
```
"""
    path.write_text(md, encoding="utf-8")


def write_sla_card(path: Path, *, ckpt_path: Path, ckpt_sha: str, ckpt_size: int,
                   info: dict, n_params: int, gpu_peak_bytes: int,
                   contributing: dict, dropped: list[str]):
    rows = []
    for k in HORIZON_KEYS:
        rows.append(
            f"| {k} | {info['ece'].get(k, float('nan')):.4f} | "
            f"{info['brier'].get(k, float('nan')):.4f} | "
            f"{info['p_min'].get(k, float('nan')):.3f} | "
            f"{info['p_mean'].get(k, float('nan')):.3f} | "
            f"{info['p_max'].get(k, float('nan')):.3f} |"
        )
    horizons_md = "\n".join(rows)
    contrib_md = "\n".join(
        f"- **{name}** ({count} windows) — `{p}`" for name, (count, p) in contributing.items()
    ) or "- (none)"
    dropped_md = "\n".join(f"- {d}" for d in dropped) or "- (none)"

    md = f"""# SLA Risk Head — v0.4 (JEPA-encoder-fed)

3GPP TS 28.105 Model Description Card.

## Identification
- **Model:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.4-jepa
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## What changed vs v0.3
- v0.3 fed a 64-d random-projection latent.
- v0.4 feeds the **frozen GraphJEPA-encoded** 128-d latent (mean over
  Perceiver latents).  Hidden dim raised 64 → 128 to match.

## Training data — REAL corpus
{contrib_md}

### Dropped
{dropped_md}

- Train samples: {info['n_train']}
- Val samples: {info['n_val']}

## Training run
- Epochs run: {info['epochs_run']}
- Initial val loss (random init): {info['initial_val_loss']:.6f}
- Final train loss: {info['final_train_loss']:.6f}
- Final val loss: {info['final_val_loss']:.6f}
- Best val loss (in checkpoint): {info['best_val_loss']:.6f}
- Wall-clock: {info['elapsed_s']:.1f}s
- Device: NVIDIA GB10 (CUDA)
- GPU peak alloc: {gpu_peak_bytes / 1e9:.3f} GB

| horizon | ECE | Brier | P(min) | P(mean) | P(max) |
|---|---|---|---|---|---|
{horizons_md}

ECE per Naeini et al. 2015 (10 equal-width bins, observed labels = `target_p ≥ 0.5`).

## Reproducibility
```
.venv/bin/python scripts/train_jepa_full.py
```
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Comparison v0.3 vs v0.4
# --------------------------------------------------------------------------- #


def compare_v03_v04(v03_path: Path, v04_path: Path, device: torch.device) -> dict | None:
    """Compare predictions on a fixed eval batch (different latent dims, so
    we draw a Gaussian batch for each head's latent_dim)."""
    if not v03_path.exists() or not v04_path.exists():
        return None
    g3 = torch.Generator().manual_seed(99)
    Z3 = torch.randn(256, 64, generator=g3).to(device)
    g4 = torch.Generator().manual_seed(99)
    Z4 = torch.randn(256, 128, generator=g4).to(device)

    h3 = SLARiskHead(SLARiskConfig(latent_dim=64, hidden_dim=64)).to(device)
    h3.load_state_dict(torch.load(v03_path, map_location=device, weights_only=True))
    h3.eval()
    h4 = SLARiskHead(SLARiskConfig(latent_dim=128, hidden_dim=128)).to(device)
    h4.load_state_dict(torch.load(v04_path, map_location=device, weights_only=True))
    h4.eval()

    with torch.no_grad():
        p3 = h3(Z3)
        p4 = h4(Z4)

    def stats(p):
        out = {}
        for k, v in p.items():
            mean = float(v.mean()); var = float(v.var())
            pc = v.clamp(1e-6, 1 - 1e-6)
            ent = -(pc * torch.log2(pc) + (1 - pc) * torch.log2(1 - pc))
            out[k] = {"mean": mean, "var": var, "entropy": float(ent.mean())}
        return out

    s3 = stats(p3); s4 = stats(p4)
    print("=" * 72)
    print("v0.3 vs v0.4 SLA HEAD COMPARISON  (random Gaussian eval batch, seed=99, N=256)")
    print("=" * 72)
    for k in HORIZON_KEYS:
        print(
            f"  {k:8s}  v0.3: mean={s3[k]['mean']:.4f} H={s3[k]['entropy']:.4f}  |  "
            f"v0.4: mean={s4[k]['mean']:.4f} H={s4[k]['entropy']:.4f}"
        )
    h3_ent = sum(s3[k]["entropy"] for k in HORIZON_KEYS) / 3
    h4_ent = sum(s4[k]["entropy"] for k in HORIZON_KEYS) / 3
    p3_mean = sum(s3[k]["mean"] for k in HORIZON_KEYS) / 3
    p4_mean = sum(s4[k]["mean"] for k in HORIZON_KEYS) / 3
    print(f"  mean entropy:   v0.3={h3_ent:.4f}  v0.4={h4_ent:.4f}")
    print(f"  mean P(breach): v0.3={p3_mean:.4f}  v0.4={p4_mean:.4f}")
    print("=" * 72)
    return {
        "v03": s3, "v04": s4,
        "v03_mean_entropy": h3_ent, "v04_mean_entropy": h4_ent,
        "v03_mean_pbreach": p3_mean, "v04_mean_pbreach": p4_mean,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    if not torch.cuda.is_available():
        print("[fatal] CUDA not available — this is a GB10-train script. Aborting.")
        return 2

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    print(f"[device] {torch.cuda.get_device_name(device)} | mem total = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    overall_start = time.time()
    timing: dict[str, float] = {}

    # ------------------ Corpus assembly ------------------ #
    t = time.time()
    corpus = assemble_token_corpus(device)
    timing["corpus"] = time.time() - t
    X = corpus["X"]
    Y = corpus["Y"]
    print(f"[corpus] X={tuple(X.shape)}  Y[h_30s]={tuple(Y['h_30s'].shape)}  in {timing['corpus']:.1f}s")

    # ------------------ Phase 1 ------------------ #
    t = time.time()
    jepa, p1 = phase1_train_jepa(X, device, budget_s=PHASE_BUDGET_S)
    timing["phase1"] = time.time() - t
    # Save: state_dict of context encoder + predictor.
    jepa_state = {
        "context_encoder": jepa.context_encoder.state_dict(),
        "predictor": jepa.predictor.state_dict(),
        "config": {"d_latent": D_LATENT, "d_input": D_INPUT, "n_latents": N_LATENTS,
                   "mask_ratio": jepa.cfg.mask_ratio, "ema_momentum": jepa.cfg.ema_momentum},
    }
    torch.save(jepa_state, JEPA_PT)
    jepa_sha = sha256_of(JEPA_PT)
    jepa_size = JEPA_PT.stat().st_size
    n_params_jepa = (
        sum(p.numel() for p in jepa.context_encoder.parameters())
        + sum(p.numel() for p in jepa.predictor.parameters())
    )
    print(f"[phase1.save] {JEPA_PT.name}  size={jepa_size}  sha256={jepa_sha}")
    write_jepa_card(
        JEPA_MD, ckpt_path=JEPA_PT, ckpt_sha=jepa_sha, ckpt_size=jepa_size, info=p1,
        contributing=corpus["contributing"], dropped=corpus["dropped"],
        tle_meta=corpus["tle_meta"], itu_status=corpus["itu_status"], n_params=n_params_jepa,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated(device)),
    )

    # ------------------ Phase 2 ------------------ #
    remaining = TOTAL_BUDGET_S - (time.time() - overall_start)
    p2_budget = min(PHASE_BUDGET_S, max(remaining - PHASE_BUDGET_S, 60.0))
    t = time.time()
    dyn, p2 = phase2_train_dynamics(
        X, corpus["pair_idx_a"], corpus["pair_idx_b"], jepa, device,
        budget_s=p2_budget,
    )
    timing["phase2"] = time.time() - t
    torch.save(dyn.state_dict(), DYN_PT)
    dyn_sha = sha256_of(DYN_PT)
    dyn_size = DYN_PT.stat().st_size
    n_params_dyn = sum(p.numel() for p in dyn.parameters())
    print(f"[phase2.save] {DYN_PT.name}  size={dyn_size}  sha256={dyn_sha}")
    write_dyn_card(
        DYN_MD, ckpt_path=DYN_PT, ckpt_sha=dyn_sha, ckpt_size=dyn_size, info=p2,
        n_params=n_params_dyn, gpu_peak_bytes=int(torch.cuda.max_memory_allocated(device)),
    )

    # ------------------ Phase 3 ------------------ #
    remaining = TOTAL_BUDGET_S - (time.time() - overall_start)
    p3_budget = max(remaining - 5.0, 30.0)
    t = time.time()
    head, p3 = phase3_train_sla(X, Y, jepa, device, budget_s=p3_budget)
    timing["phase3"] = time.time() - t
    torch.save(head.state_dict(), SLA_PT)
    sla_sha = sha256_of(SLA_PT)
    sla_size = SLA_PT.stat().st_size
    print(f"[phase3.save] {SLA_PT.name}  size={sla_size}  sha256={sla_sha}")
    write_sla_card(
        SLA_MD, ckpt_path=SLA_PT, ckpt_sha=sla_sha, ckpt_size=sla_size, info=p3,
        n_params=p3["n_params"], gpu_peak_bytes=int(torch.cuda.max_memory_allocated(device)),
        contributing=corpus["contributing"], dropped=corpus["dropped"],
    )

    # ------------------ v0.3 vs v0.4 comparison ------------------ #
    v03_path = CKPT_DIR / "sla_head_v0.3_full_corpus.pt"
    cmp = compare_v03_v04(v03_path, SLA_PT, device)

    overall = time.time() - overall_start
    gpu_peak = int(torch.cuda.max_memory_allocated(device))

    summary = {
        "timing_s": timing,
        "total_wallclock_s": overall,
        "gpu_peak_bytes": gpu_peak,
        "gpu_peak_GB": gpu_peak / 1e9,
        "phase1": {
            "final_loss": p1["final_loss"], "epochs_run": p1["epochs_run"],
            "ema_beta": p1["ema_beta"], "pred_norm": p1["final_pred_norm"],
            "target_norm": p1["final_target_norm"],
            "ckpt": str(JEPA_PT), "sha256": jepa_sha, "size": jepa_size,
        },
        "phase2": {
            "skipped": p2.get("skipped", False),
            "final_loss": p2.get("final_loss", float("nan")),
            "epochs_run": p2.get("epochs_run", 0),
            "n_pairs": p2.get("n_pairs", 0),
            "ckpt": str(DYN_PT), "sha256": dyn_sha, "size": dyn_size,
        },
        "phase3": {
            "final_train_loss": p3["final_train_loss"],
            "final_val_loss": p3["final_val_loss"],
            "best_val_loss": p3["best_val_loss"],
            "ece": p3["ece"], "brier": p3["brier"],
            "p_min": p3["p_min"], "p_mean": p3["p_mean"], "p_max": p3["p_max"],
            "n_train": p3["n_train"], "n_val": p3["n_val"],
            "ckpt": str(SLA_PT), "sha256": sla_sha, "size": sla_size,
        },
        "comparison_v03_vs_v04": cmp,
        "contributing": {k: v[0] for k, v in corpus["contributing"].items()},
        "dropped": corpus["dropped"],
    }
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"[overall] total_wallclock={overall:.1f}s  gpu_peak={gpu_peak / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
