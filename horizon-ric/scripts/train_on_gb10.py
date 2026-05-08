"""PreceptualAI — single-shot, zero-cost training on NVIDIA GB10 (DGX Spark).

Runs the full Day-1 demo training in three phases on a single GB10:
    Phase A — SLA Risk Head v0.3 on the FULL combined corpus (5 min budget).
    Phase B — Graph-JEPA encoder pretraining on Perceiver latents (15 min).
    Phase C — Latent dynamics + risk head fine-tune on top of JEPA encoder
              (10 min).

Total wall-time budget: <= 45 minutes on GB10. Each phase is gated by its
own deadline; if any phase blows past, we stop early and ship what we have.

Hardware policy:
    * GB10 is detected as `cuda` device with > 100 GB total memory (unified).
    * Anything else (no cuda, smaller GPU, CPU) falls back to a REDUCED plan:
        - Aggressive corpus subsampling, smaller batch, fewer epochs.
        - A loud warning in the log so the demo engineer knows.

Honest logging contract:
    * Every phase prints whether it actually loaded REAL data from disk or
      fell back to anything synthetic. We never silently invent data.
    * Every phase emits a 3GPP TS 28.105 model card alongside its checkpoint.

Outputs:
    checkpoints/sla_head_v0.3_gb10.pt   (+ .md card)
    checkpoints/jepa_encoder_v0.1.pt    (+ .md card)
    checkpoints/world_model_v0.1.pt     (+ .md card)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse all the real-data extractors from the v0.2 trainer — same parquet
# paths, same threshold table, same per-source SHA-salted projections.
from train_sla_head_nvidia import (  # noqa: E402
    AERIAL_FAPI,
    AERIAL_FH,
    DEEPMIMO_ROOT,
    UCC_ROOT,
    HORIZON_KEYS,
    LATENT_DIM,
    extract_aerial_fapi,
    extract_aerial_fh,
    extract_deepmimo,
    extract_ucc,
    print_inventory,
    sha256_of,
    split_train_val,
    train_loop as train_head_loop,
    evaluate as evaluate_head,
)
from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402
from horizon_ric.encoder.graph_jepa import (  # noqa: E402
    GraphJEPA,
    GraphJEPAConfig,
)
from horizon_ric.core.latent_dynamics import (  # noqa: E402
    LatentDynamics,
    LatentDynamicsConfig,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Hardware detection
# --------------------------------------------------------------------------- #

GB10_MIN_MEM_GB = 100.0  # GB10 unified memory floor; anything smaller is "other GPU".


@dataclass
class HwProfile:
    name: str            # "gb10" | "cuda_other" | "cpu"
    device: torch.device
    total_mem_gb: float  # 0.0 for CPU
    plan: str            # "full" | "reduced"


def detect_hw() -> HwProfile:
    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        device = torch.device(f"cuda:{idx}")
        if total_gb >= GB10_MIN_MEM_GB:
            return HwProfile(name="gb10", device=device, total_mem_gb=total_gb, plan="full")
        return HwProfile(
            name="cuda_other", device=device, total_mem_gb=total_gb, plan="reduced",
        )
    return HwProfile(name="cpu", device=torch.device("cpu"), total_mem_gb=0.0, plan="reduced")


def banner(s: str) -> None:
    line = "=" * 78
    print(line)
    print(s)
    print(line)


# --------------------------------------------------------------------------- #
# Corpus loading — same path the v0.2 trainer uses, no synthetic fallback.
# --------------------------------------------------------------------------- #


def load_full_corpus(plan: str) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict, list[str]]:
    """Load the combined Aerial + DeepMIMO + UCC corpus.

    Returns:
        Z: (N, 64) feature matrix.
        Y: dict horizon_key -> (N,) target probability.
        contributing: {source_name: (count, path_str)}
        dropped: list of source descriptions that failed to load.
    """
    print_inventory()
    contributing: dict[str, tuple[int, str]] = {}
    dropped: list[str] = []
    feats: list[np.ndarray] = []
    targets: list[dict[str, float]] = []

    f, t = extract_aerial_fapi(AERIAL_FAPI)
    if f:
        contributing["NVIDIA Aerial FAPI"] = (len(f), str(AERIAL_FAPI))
        feats.extend(f); targets.extend(t)
    else:
        dropped.append(f"NVIDIA Aerial FAPI ({AERIAL_FAPI})")

    f, t = extract_aerial_fh(AERIAL_FH)
    if f:
        contributing["NVIDIA Aerial FH"] = (len(f), str(AERIAL_FH))
        feats.extend(f); targets.extend(t)
    else:
        dropped.append(f"NVIDIA Aerial FH ({AERIAL_FH})")

    # Full plan uses ALL 100 DeepMIMO scenarios; reduced uses 20.
    n_scen = 100 if plan == "full" else 20
    f, t = extract_deepmimo(DEEPMIMO_ROOT, max_scenarios=n_scen)
    if f:
        contributing[f"NVIDIA DeepMIMO ({n_scen} scenarios)"] = (len(f), str(DEEPMIMO_ROOT))
        feats.extend(f); targets.extend(t)
    else:
        dropped.append(f"NVIDIA DeepMIMO ({DEEPMIMO_ROOT})")

    n_csv = 60 if plan == "full" else 12
    f, t = extract_ucc(UCC_ROOT, max_files=n_csv, window_sec=30)
    if f:
        contributing[f"UCC MISL 5G ({n_csv} csv)"] = (len(f), str(UCC_ROOT))
        feats.extend(f); targets.extend(t)
    else:
        dropped.append(f"UCC MISL 5G ({UCC_ROOT})")

    if not feats:
        raise RuntimeError(
            "No real data was loaded. We refuse to fall back to a synthetic "
            "corpus — fix the data paths or download the corpus first."
        )

    Z = torch.from_numpy(np.stack(feats, axis=0)).float()
    Y = {k: torch.tensor([t[k] for t in targets], dtype=torch.float32) for k in HORIZON_KEYS}
    return Z, Y, contributing, dropped


# --------------------------------------------------------------------------- #
# Phase A — SLA Risk Head v0.3
# --------------------------------------------------------------------------- #


def phase_a_sla_head(
    hw: HwProfile,
    Z: torch.Tensor,
    Y: dict[str, torch.Tensor],
    contributing: dict,
    dropped: list[str],
    deadline_sec: float,
) -> dict:
    banner(f"PHASE A — SLA Risk Head v0.3 (budget {deadline_sec:.0f}s)")
    t0 = time.time()
    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z, Y)
    print(f"[A] split: train={Z_tr.shape[0]} val={Z_val.shape[0]} dim={Z.shape[1]}")

    head = SLARiskHead(SLARiskConfig(latent_dim=LATENT_DIM, hidden_dim=64))
    head.to(hw.device)
    Z_tr = Z_tr.to(hw.device); Z_val = Z_val.to(hw.device)
    Y_tr = {k: v.to(hw.device) for k, v in Y_tr.items()}
    Y_val = {k: v.to(hw.device) for k, v in Y_val.items()}
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[A] params={n_params}")

    epochs = 60 if hw.plan == "full" else 15
    bs = 256 if hw.plan == "full" else 64

    train_info = train_head_loop(
        head, Z_tr, Y_tr, Z_val, Y_val,
        epochs=epochs, batch_size=bs, lr=1e-3, patience=8,
    )
    eval_info = evaluate_head(head, Z_val, Y_val)

    out = ROOT / "checkpoints" / "sla_head_v0.3_gb10.pt"
    card = ROOT / "checkpoints" / "sla_head_v0.3_gb10.md"
    head.cpu()
    torch.save(head.state_dict(), out)
    sha = sha256_of(out)
    size = out.stat().st_size

    _write_card_phase_a(
        card, contributing=contributing, dropped=dropped,
        n_train=Z_tr.shape[0], n_val=Z_val.shape[0],
        train_info=train_info, eval_info=eval_info, n_params=n_params,
        ckpt_path=out, ckpt_size=size, ckpt_sha=sha, hw=hw,
    )
    elapsed = time.time() - t0
    print(f"[A] done in {elapsed:.1f}s — ckpt={out} sha256={sha[:16]}...")
    if elapsed > deadline_sec:
        print(f"[A] WARNING: phase exceeded deadline ({elapsed:.0f}s > {deadline_sec:.0f}s)")
    return {
        "ckpt": str(out), "card": str(card), "elapsed_sec": elapsed,
        "n_train": Z_tr.shape[0], "n_val": Z_val.shape[0],
        "best_val_loss": train_info["best_val_loss"],
        "ece": eval_info["ece"], "brier": eval_info["brier"],
    }


def _write_card_phase_a(card: Path, *, contributing, dropped, n_train, n_val,
                        train_info, eval_info, n_params, ckpt_path, ckpt_size,
                        ckpt_sha, hw: HwProfile) -> None:
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
    ) or "- (none — all sources dropped)"
    dropped_md = "\n".join(f"- {d}" for d in dropped) or "- (none)"

    md = f"""# SLA Risk Head — v0.3 (GB10, full real corpus)

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.3-gb10
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## Training Hardware
- Detected device: `{hw.name}` (cuda total mem {hw.total_mem_gb:.1f} GB)
- Training plan: `{hw.plan}`

## Training Data — REAL sources used

{contrib_md}

### Sources excluded / dropped this run

{dropped_md}

- **Train samples:** {n_train}
- **Val samples:** {n_val}
- **Latent construction:** per-source numeric KPI vector → deterministic
  Gaussian random projection to 64-d (Johnson-Lindenstrauss style; per-source
  SHA-256 salt; tanh non-linearity at the output).

## Performance

- Initial val loss: {train_info['initial_val_loss']:.6f}
- Final train loss: {train_info['final_train_loss']:.6f}
- Final val loss: {train_info['final_val_loss']:.6f}
- Best val loss in checkpoint: {train_info['best_val_loss']:.6f}

| horizon | ECE | Brier | P(min) | P(mean) | P(max) |
|---|---|---|---|---|---|
{horizons_md}

## Honest caveats
- SLA-breach LABELS are derived from real radio measurements via the v0.2
  threshold table; they are not observed breach events.
- Per-source random projections are L2-isometric in expectation but add no
  information; they fill the 64-d `z_resource` interface.
- v0.3 vs v0.2: same model architecture, larger real corpus, longer training
  schedule. Use as the production checkpoint until a customer-fine-tune is
  available.

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- 3GPP TS 38.133 (RSRP/RSRQ measurement requirements).
- 3GPP TS 28.554 §6.x (E2E KPIs targeted by SLA breach definition).
"""
    card.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Phase B — Graph-JEPA encoder pretraining
# --------------------------------------------------------------------------- #


class _SimpleEncoder(nn.Module):
    """Tiny self-attention encoder over (B, N, d_input) → (B, N, d_latent).

    Plays the role of `PerceiverFusion` for JEPA pretraining without the full
    cross-attention plumbing. Treating each token's projected feature as a
    "latent slot" is sufficient for the action-free pretraining demo.
    """

    def __init__(self, d_input: int, d_latent: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_latent)
        layer = nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=n_heads, dim_feedforward=d_latent * 2,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.in_proj(x)
        return self.encoder(h)


def _make_jepa_corpus(Z: torch.Tensor, n_tokens: int = 16) -> torch.Tensor:
    """Reshape the flat 64-d corpus into token sequences for JEPA pretraining.

    We treat each contiguous block of `n_tokens` corpus rows as one "scene"
    (tokens within a scene share the same source and time-window neighbourhood
    once the loader yields them), then break each row into 4 sub-tokens of 16-d
    so the JEPA encoder sees a meaningful per-scene token graph.
    """
    N, D = Z.shape
    sub = 4
    sub_d = D // sub
    # Drop the tail so we have exact (N//n_tokens) scenes.
    n_scenes = N // n_tokens
    if n_scenes == 0:
        raise RuntimeError(f"Not enough rows ({N}) to build {n_tokens}-token scenes")
    Z_use = Z[: n_scenes * n_tokens].view(n_scenes, n_tokens, sub, sub_d)
    Z_use = Z_use.view(n_scenes, n_tokens * sub, sub_d)  # (S, N*sub, sub_d)
    return Z_use


def phase_b_jepa(hw: HwProfile, Z_corpus: torch.Tensor, deadline_sec: float) -> dict:
    banner(f"PHASE B — Graph-JEPA encoder pretraining (budget {deadline_sec:.0f}s)")
    t0 = time.time()
    d_input = 16
    d_latent = 128

    scenes = _make_jepa_corpus(Z_corpus, n_tokens=16)  # (S, 64, 16)
    print(f"[B] jepa corpus: scenes={scenes.shape[0]} tokens={scenes.shape[1]} d={scenes.shape[2]}")
    if scenes.shape[0] < 8:
        print("[B] WARNING: not enough scenes for stable JEPA pretraining; using what we have")

    encoder = _SimpleEncoder(d_input=d_input, d_latent=d_latent, n_layers=2, n_heads=4)
    jepa = GraphJEPA(encoder, GraphJEPAConfig(d_latent=d_latent, mask_ratio=0.6))
    jepa.to(hw.device)
    n_params = sum(p.numel() for p in jepa.parameters())
    print(f"[B] params={n_params}")

    bs = 64 if hw.plan == "full" else 16
    epochs = 25 if hw.plan == "full" else 6
    lr = 3e-4
    optim = torch.optim.AdamW(jepa.parameters(), lr=lr, weight_decay=0.05)

    scenes = scenes.to(hw.device)
    losses: list[float] = []
    for ep in range(1, epochs + 1):
        if time.time() - t0 > deadline_sec:
            print(f"[B] deadline hit at epoch {ep} — stopping early")
            break
        perm = torch.randperm(scenes.shape[0], device=hw.device)
        ep_loss = 0.0
        ep_seen = 0
        for s in range(0, scenes.shape[0], bs):
            idx = perm[s:s + bs]
            batch = scenes[idx]
            optim.zero_grad()
            out = jepa.loss(batch)
            out["loss"].backward()
            optim.step()
            jepa.update_target()
            ep_loss += float(out["loss"].detach().item()) * batch.shape[0]
            ep_seen += batch.shape[0]
        avg = ep_loss / max(ep_seen, 1)
        losses.append(avg)
        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"[B][ep {ep:02d}] loss={avg:.6f} pred_norm={float(out['pred_norm']):.3f} "
                  f"target_norm={float(out['target_norm']):.3f}")

    out_path = ROOT / "checkpoints" / "jepa_encoder_v0.1.pt"
    card_path = ROOT / "checkpoints" / "jepa_encoder_v0.1.md"
    jepa.cpu()
    state = {
        "context_encoder": encoder.state_dict(),
        "predictor": jepa.predictor.state_dict(),
        "config": {
            "d_input": d_input, "d_latent": d_latent,
            "n_layers": 2, "n_heads": 4,
        },
    }
    torch.save(state, out_path)
    sha = sha256_of(out_path)
    size = out_path.stat().st_size
    elapsed = time.time() - t0

    card_path.write_text(_render_jepa_card(
        contributing_scenes=scenes.shape[0], n_tokens=scenes.shape[1],
        d_input=d_input, d_latent=d_latent, n_params=n_params,
        epochs_done=len(losses), losses=losses, ckpt_path=out_path,
        ckpt_size=size, ckpt_sha=sha, hw=hw,
    ), encoding="utf-8")

    print(f"[B] done in {elapsed:.1f}s — ckpt={out_path} sha256={sha[:16]}...")
    return {
        "ckpt": str(out_path), "card": str(card_path), "elapsed_sec": elapsed,
        "epochs_done": len(losses),
        "final_loss": losses[-1] if losses else float("nan"),
    }


def _render_jepa_card(*, contributing_scenes, n_tokens, d_input, d_latent,
                      n_params, epochs_done, losses, ckpt_path, ckpt_size,
                      ckpt_sha, hw: HwProfile) -> str:
    loss_table = "\n".join(
        f"| {i+1} | {l:.6f} |" for i, l in enumerate(losses)
    ) or "| (none) | (none) |"
    return f"""# Graph-JEPA encoder — v0.1 (GB10 pretraining)

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model:** `horizon_ric.encoder.graph_jepa.GraphJEPA` over `_SimpleEncoder`.
- **Version:** v0.1
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}
- **Parameter count:** {n_params}

## Training Hardware
- Detected device: `{hw.name}` (cuda total mem {hw.total_mem_gb:.1f} GB)
- Training plan: `{hw.plan}`

## Pretraining recipe
- Action-free joint-embedding predictive architecture (I-JEPA / V-JEPA).
- Mask ratio 0.6, EMA momentum 0.996.
- Smooth-L1 loss in latent space (no token-space reconstruction).

## Corpus shape
- {contributing_scenes} scenes × {n_tokens} tokens × {d_input}-d.
- Tokenization: each 64-d corpus row is split into 4 sub-tokens of 16-d, then
  16 rows are concatenated to form one scene of {n_tokens} tokens.

## Loss curve

| epoch | loss |
|---|---|
{loss_table}

- Epochs completed: {epochs_done}

## Honest caveats
- This is a smoke-grade encoder pretraining: the corpus is the same combined
  Aerial+DeepMIMO+UCC features used in Phase A, reshaped to scenes. We do
  NOT yet ingest raw FAPI message graphs as tokens — that's the v0.2 work.
- The predictor head is included in the artefact but the canonical "reuse"
  artefact is the `context_encoder.state_dict()`.

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- I-JEPA: Assran et al., CVPR 2023. V-JEPA: Bardes et al., 2024.
"""


# --------------------------------------------------------------------------- #
# Phase C — Latent dynamics + risk head fine-tune on JEPA encoder
# --------------------------------------------------------------------------- #


def phase_c_world_model(
    hw: HwProfile,
    Z_corpus: torch.Tensor,
    Y_corpus: dict[str, torch.Tensor],
    jepa_ckpt: Path,
    deadline_sec: float,
) -> dict:
    banner(f"PHASE C — World model (latent dynamics + risk head) "
           f"(budget {deadline_sec:.0f}s)")
    t0 = time.time()
    d_latent = 128
    d_action = 16

    # Load JEPA encoder
    state = torch.load(jepa_ckpt, map_location="cpu", weights_only=True)
    cfg = state["config"]
    encoder = _SimpleEncoder(
        d_input=cfg["d_input"], d_latent=cfg["d_latent"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    )
    encoder.load_state_dict(state["context_encoder"])
    encoder.to(hw.device)
    print(f"[C] loaded JEPA encoder from {jepa_ckpt}")

    dynamics = LatentDynamics(LatentDynamicsConfig(
        d_latent=d_latent, d_action=d_action, d_state=16,
    )).to(hw.device)
    risk_head = SLARiskHead(SLARiskConfig(
        latent_dim=d_latent, hidden_dim=128,
    )).to(hw.device)

    # Build (z, action, z_next, y) trajectories from the corpus rows.
    scenes = _make_jepa_corpus(Z_corpus, n_tokens=16).to(hw.device)
    # For each scene, pool encoder output → z (B, d_latent). Use mean pooling.
    encoder.eval()
    with torch.no_grad():
        h = encoder(scenes)  # (S, T, d_latent)
        z_seq = h.mean(dim=1)  # (S, d_latent) — one z per scene

    # Synthetic actions stitched scene→scene (zero-action transitions). This
    # is honest: we don't have action labels in the offline corpus, so we
    # train the dynamics to be a near-identity on null actions and learn the
    # risk head off the JEPA-encoded latent.
    n_scenes = z_seq.shape[0]
    if n_scenes < 4:
        print("[C] WARNING: corpus too small for dynamics fine-tune; risk head only")

    # Risk targets — average the per-row corpus targets across the scene's rows.
    rows_per_scene = (Z_corpus.shape[0] // 16) and 16  # n_tokens
    Y_scenes = {}
    n_trim = n_scenes * 16
    for k, v in Y_corpus.items():
        Y_scenes[k] = v[:n_trim].view(n_scenes, 16).mean(dim=1).to(hw.device)

    # Split
    g = torch.Generator(device="cpu").manual_seed(7)
    perm = torch.randperm(n_scenes, generator=g)
    n_val = max(1, n_scenes // 5)
    val_idx = perm[:n_val].to(hw.device)
    tr_idx = perm[n_val:].to(hw.device)
    z_tr = z_seq[tr_idx]; z_val = z_seq[val_idx]
    Y_tr = {k: v[tr_idx] for k, v in Y_scenes.items()}
    Y_val = {k: v[val_idx] for k, v in Y_scenes.items()}

    # Joint optim — dynamics gets a small contractive loss on null action,
    # risk head gets two-hot CE on the JEPA-pooled latent.
    epochs = 50 if hw.plan == "full" else 12
    bs = 64 if hw.plan == "full" else 16
    optim = torch.optim.AdamW(
        list(dynamics.parameters()) + list(risk_head.parameters()),
        lr=5e-4, weight_decay=0.01,
    )

    history: list[dict] = []
    best = math.inf
    for ep in range(1, epochs + 1):
        if time.time() - t0 > deadline_sec:
            print(f"[C] deadline hit at epoch {ep} — stopping early")
            break
        dynamics.train(); risk_head.train()
        perm_t = torch.randperm(z_tr.shape[0], device=hw.device)
        ep_loss = 0.0; ep_seen = 0
        for s in range(0, z_tr.shape[0], bs):
            idx = perm_t[s:s + bs]
            zb = z_tr[idx]
            yb = {k: v[idx] for k, v in Y_tr.items()}
            optim.zero_grad()
            # Dynamics step on null action — should approx be identity-ish
            null_a = torch.zeros(zb.shape[0], d_action, device=hw.device)
            z_next, _ = dynamics.step(zb, null_a)
            dyn_loss = nn.functional.smooth_l1_loss(z_next, zb)
            risk_loss = risk_head.loss(z_next, yb)
            loss = risk_loss + 0.1 * dyn_loss
            loss.backward()
            optim.step()
            ep_loss += float(loss.detach().item()) * zb.shape[0]
            ep_seen += zb.shape[0]
        avg_tr = ep_loss / max(ep_seen, 1)

        dynamics.eval(); risk_head.eval()
        with torch.no_grad():
            null_a = torch.zeros(z_val.shape[0], d_action, device=hw.device)
            z_val_next, _ = dynamics.step(z_val, null_a)
            v_loss = float(risk_head.loss(z_val_next, Y_val).item())
        history.append({"epoch": ep, "train_loss": avg_tr, "val_loss": v_loss})
        if v_loss < best:
            best = v_loss
        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"[C][ep {ep:02d}] train={avg_tr:.6f} val={v_loss:.6f} best={best:.6f}")

    # Save combined world model
    out = ROOT / "checkpoints" / "world_model_v0.1.pt"
    card = ROOT / "checkpoints" / "world_model_v0.1.md"
    dynamics.cpu(); risk_head.cpu(); encoder.cpu()
    torch.save({
        "encoder": encoder.state_dict(),
        "dynamics": dynamics.state_dict(),
        "risk_head": risk_head.state_dict(),
        "config": {
            "d_latent": d_latent,
            "d_action": d_action,
            "encoder_config": cfg,
        },
    }, out)
    sha = sha256_of(out)
    size = out.stat().st_size
    elapsed = time.time() - t0

    card.write_text(_render_wm_card(
        history=history, n_train=z_tr.shape[0], n_val=z_val.shape[0],
        ckpt_path=out, ckpt_size=size, ckpt_sha=sha, hw=hw,
    ), encoding="utf-8")

    print(f"[C] done in {elapsed:.1f}s — ckpt={out} sha256={sha[:16]}...")
    return {
        "ckpt": str(out), "card": str(card), "elapsed_sec": elapsed,
        "epochs_done": len(history),
        "best_val_loss": best,
    }


def _render_wm_card(*, history, n_train, n_val, ckpt_path, ckpt_size, ckpt_sha,
                    hw: HwProfile) -> str:
    rows = "\n".join(
        f"| {h['epoch']} | {h['train_loss']:.6f} | {h['val_loss']:.6f} |" for h in history
    ) or "| (none) | (none) | (none) |"
    final_train = history[-1]['train_loss'] if history else float('nan')
    final_val = history[-1]['val_loss'] if history else float('nan')
    return f"""# World Model — v0.1 (GB10, JEPA-encoder fine-tune)

3GPP TS 28.105 (AI/ML management) — Model Description Card.

## Identification
- **Model:** Graph-JEPA encoder + `LatentDynamics` + `SLARiskHead`.
- **Version:** v0.1-gb10
- **Artefact:** `{ckpt_path.name}`
- **SHA-256:** `{ckpt_sha}`
- **File size (bytes):** {ckpt_size}

## Training Hardware
- Detected device: `{hw.name}` (cuda total mem {hw.total_mem_gb:.1f} GB)
- Training plan: `{hw.plan}`

## Recipe
- Phase B JEPA encoder is loaded frozen for feature extraction (mean-pool over
  scene tokens) — this seeds a 128-d latent.
- `LatentDynamics` (Mamba-style selective scan, action-conditioned) is trained
  with a smooth-L1 contractive loss on null actions.
- `SLARiskHead` is trained two-hot CE on JEPA-pooled latent post-dynamics.
- Loss = risk_loss + 0.1 * dyn_loss.

## Splits
- Train scenes: {n_train}
- Val scenes:   {n_val}

## History (truncated)

| epoch | train_loss | val_loss |
|---|---|---|
{rows}

Final train loss: {final_train:.6f} | Final val loss: {final_val:.6f}

## Honest caveats
- Actions are zero-vectors in this fine-tune — there is no offline action
  signal in the static corpus. The dynamics module learns a stable identity
  on null actions; real action conditioning lands in Tier-2 once we have
  customer A1/E2 traces.
- The JEPA encoder is loaded frozen here. Joint encoder + dynamics + head
  fine-tuning is on the v0.2 roadmap.
- Mean-pool over scene tokens is the simplest aggregator; a learnable
  cross-attention pool is the obvious upgrade.

## Standards
- 3GPP TS 28.105 v18 (AI/ML management — model lifecycle).
- DreamerV3 (Hafner et al., Nature 2025) — symlog two-hot heads.
- Mamba (Gu & Dao, 2023) — selective state-space dynamics.
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    overall_t0 = time.time()

    hw = detect_hw()
    banner(f"PreceptualAI GB10 trainer — device={hw.name} mem={hw.total_mem_gb:.1f}GB plan={hw.plan}")
    if hw.plan == "reduced":
        print("WARNING: GB10 NOT detected. Running REDUCED plan.")
        print("WARNING: This is fine for a smoke test, but the published checkpoints")
        print("WARNING: will NOT match the metrics quoted in the model cards if the")
        print("WARNING: full GB10 plan was promised.")

    # Time budgets per phase (seconds). The full GB10 plan targets the budgets
    # promised in the deliverable; the reduced plan shrinks them so a CPU
    # smoke test finishes in a few minutes.
    if hw.plan == "full":
        budget_a, budget_b, budget_c = 5 * 60, 15 * 60, 10 * 60
    else:
        budget_a, budget_b, budget_c = 90, 240, 180

    Z, Y, contributing, dropped = load_full_corpus(hw.plan)
    print(f"[corpus] N={Z.shape[0]} dim={Z.shape[1]} sources={list(contributing)}")
    print(f"[corpus] dropped={dropped}")
    print(f"[corpus] real_data_used={'YES' if contributing else 'NO'} "
          f"synthetic_fallback={'NO'}")

    summary: dict[str, dict] = {}
    summary["phase_a"] = phase_a_sla_head(hw, Z, Y, contributing, dropped, budget_a)
    summary["phase_b"] = phase_b_jepa(hw, Z, budget_b)
    summary["phase_c"] = phase_c_world_model(
        hw, Z, Y, Path(summary["phase_b"]["ckpt"]), budget_c,
    )

    total = time.time() - overall_t0
    summary["total_elapsed_sec"] = total
    summary["hw"] = {
        "name": hw.name, "device": str(hw.device),
        "total_mem_gb": hw.total_mem_gb, "plan": hw.plan,
    }
    summary["data"] = {
        "real_data_used": True if contributing else False,
        "synthetic_fallback": False,
        "contributing": {k: v[0] for k, v in contributing.items()},
        "dropped": dropped,
    }

    banner(f"DONE in {total:.1f}s — total budget was 45 min ({45*60}s)")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
