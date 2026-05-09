"""SLA-head edge distillation — KL(student || teacher) on the v0.4 val set.

Distills the v0.4 teacher (latent_dim=128, hidden_dim=128) down to a
small student (latent_dim=32, hidden_dim=32) and writes
``checkpoints/sla_head_v0.4_jepa_edge.pt``. The student MUST:

  * be ≤ 30% of the teacher's on-disk size,
  * predict within 0.05 MAE of the teacher on the SAME val set (split
    seed=42, val_frac=0.2 — identical to v0.4 training).

We re-use the same assembly pipeline (`scripts/train_jepa_full.py`) so
the val data really is the v0.4 val set, not a synthetic stand-in.

Usage::

    .venv/bin/python scripts/distill_for_edge.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_ric.encoder import GraphJEPA, GraphJEPAConfig, PerceiverConfig, PerceiverFusion  # noqa: E402
from horizon_ric.heads import SLARiskConfig, SLARiskHead  # noqa: E402

# Pull the same corpus + split logic the v0.4 training used.
from train_sla_head_nvidia import split_train_val  # noqa: E402
from train_jepa_full import (  # noqa: E402
    D_INPUT,
    D_LATENT,
    N_LATENTS,
    N_TOKENS,
    assemble_token_corpus,
)


CKPT_DIR = ROOT / "checkpoints"
TEACHER_PT = CKPT_DIR / "sla_head_v0.4_jepa.pt"
JEPA_PT = CKPT_DIR / "jepa_encoder_v0.1.pt"
STUDENT_PT = CKPT_DIR / "sla_head_v0.4_jepa_edge.pt"
STUDENT_MD = CKPT_DIR / "sla_head_v0.4_jepa_edge.md"

EDGE_LATENT_DIM = 32
EDGE_HIDDEN_DIM = 32

EPOCHS = 60
BATCH = 64
LR = 1e-3
TEMPERATURE = 1.0   # KL temperature; teacher logits are already in logit space


def _build_jepa(device: torch.device) -> nn.Module:
    """Reconstruct the GraphJEPA encoder used at v0.4 train time."""
    fusion_cfg = PerceiverConfig(
        n_latents=N_LATENTS,
        d_latent=D_LATENT,
        d_input=D_INPUT,
        n_self_layers=2,
        n_heads=4,
    )
    fusion = PerceiverFusion(fusion_cfg)
    jepa = GraphJEPA(
        context_encoder=fusion,
        config=GraphJEPAConfig(
            d_latent=D_LATENT,
            predictor_hidden=256,
            predictor_layers=2,
            mask_ratio=0.6,
            ema_momentum=0.996,
        ),
    ).to(device)
    state = torch.load(JEPA_PT, map_location=device, weights_only=False)
    jepa.context_encoder.load_state_dict(state["context_encoder"])
    if "predictor" in state:
        jepa.predictor.load_state_dict(state["predictor"])
    jepa.eval()
    return jepa


def _project_to_edge(z_full: torch.Tensor, edge_dim: int) -> torch.Tensor:
    """Truncate the JEPA latent to the student's lower dimensionality.

    The student is configured with ``latent_dim=edge_dim`` so its first
    Linear expects (B, edge_dim). We pick a deterministic linear
    projection (sliced average pooling along the latent axis) — this is
    cheap, runs at O(D), and avoids extra trainable parameters.
    """
    D = z_full.shape[-1]
    assert D >= edge_dim, f"need D ≥ edge_dim, got D={D}, edge={edge_dim}"
    factor = D // edge_dim
    cut = factor * edge_dim
    return z_full[..., :cut].view(*z_full.shape[:-1], edge_dim, factor).mean(dim=-1)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[distill] device={device}")
    if not TEACHER_PT.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {TEACHER_PT}")

    # ── 1. Rebuild teacher and load weights ───────────────────────────
    teacher = SLARiskHead(SLARiskConfig(latent_dim=128, hidden_dim=128)).to(device)
    state = torch.load(TEACHER_PT, map_location=device, weights_only=False)
    teacher.load_state_dict(state)
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_size = TEACHER_PT.stat().st_size
    print(f"[distill] teacher: {teacher_params} params, {teacher_size} bytes")

    # ── 2. Re-assemble the exact same corpus the v0.4 training used ───
    print("[distill] assembling real corpus (this can take a minute) ...")
    corpus = assemble_token_corpus(device)
    X_tokens = corpus["X"]            # (N, N_TOKENS, D_INPUT)
    Y = corpus["Y"]                   # dict of (N,)

    # ── 3. JEPA-encode -> shared 128-d latent ─────────────────────────
    jepa = _build_jepa(device)
    with torch.no_grad():
        Z_full = jepa.context_encoder(X_tokens).mean(dim=1)
    print(f"[distill] Z_full {tuple(Z_full.shape)}")

    # Same split as v0.4 (seed=42, val_frac=0.2).
    Z_tr, Y_tr, Z_val, Y_val = split_train_val(Z_full, Y, val_frac=0.2, seed=42)
    print(f"[distill] train={Z_tr.shape[0]} val={Z_val.shape[0]}")

    # ── 4. Build student ──────────────────────────────────────────────
    student = SLARiskHead(
        SLARiskConfig(latent_dim=EDGE_LATENT_DIM, hidden_dim=EDGE_HIDDEN_DIM)
    ).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"[distill] student: {student_params} params")

    # ── 5. Distill via KL(student || teacher) per horizon ─────────────
    Z_tr_edge = _project_to_edge(Z_tr, EDGE_LATENT_DIM)
    Z_val_edge = _project_to_edge(Z_val, EDGE_LATENT_DIM)

    opt = torch.optim.Adam(student.parameters(), lr=LR)
    teacher_logits_tr = {
        h_key: teacher.predict_logits(Z_tr, h_key).detach()
        for h_key in teacher.heads
    }
    teacher_logits_val = {
        h_key: teacher.predict_logits(Z_val, h_key).detach()
        for h_key in teacher.heads
    }

    history = []
    best_kl = math.inf
    best_state = None
    bad = 0
    patience = 8
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        student.train()
        n = Z_tr_edge.shape[0]
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            zb = Z_tr_edge[idx]
            opt.zero_grad()
            loss = zb.new_zeros(())
            for h_key in student.heads:
                t_logits = teacher_logits_tr[h_key][idx] / TEMPERATURE
                s_logits = student.predict_logits(zb, h_key) / TEMPERATURE
                # KL(teacher || student) on per-bin distributions.
                t_logp = F.log_softmax(t_logits, dim=-1)
                s_logp = F.log_softmax(s_logits, dim=-1)
                t_p = t_logp.exp()
                kl = (t_p * (t_logp - s_logp)).sum(dim=-1).mean()
                loss = loss + kl
            loss = loss / len(student.heads)
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * zb.shape[0]
            seen += zb.shape[0]
        train_kl = total / max(seen, 1)

        # Val KL.
        student.eval()
        with torch.no_grad():
            val_kl = 0.0
            for h_key in student.heads:
                t_logits = teacher_logits_val[h_key] / TEMPERATURE
                s_logits = student.predict_logits(Z_val_edge, h_key) / TEMPERATURE
                t_logp = F.log_softmax(t_logits, dim=-1)
                s_logp = F.log_softmax(s_logits, dim=-1)
                t_p = t_logp.exp()
                val_kl += float(
                    (t_p * (t_logp - s_logp)).sum(dim=-1).mean().item()
                )
            val_kl /= len(student.heads)

        history.append({"epoch": ep, "train_kl": train_kl, "val_kl": val_kl})
        improved = val_kl < best_kl - 1e-6
        if improved:
            best_kl = val_kl
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or improved:
            print(
                f"[distill ep {ep:02d}] train_kl={train_kl:.4f} val_kl={val_kl:.4f} "
                f"{'*' if improved else ' '} best={best_kl:.4f}"
            )
        if bad >= patience:
            print(f"[distill] early-stop (patience={patience})")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    # ── 6. MAE between student and teacher on val ────────────────────
    student.eval()
    with torch.no_grad():
        teacher_preds = teacher(Z_val)
        student_preds = student(Z_val_edge)
    mae_per_h = {
        h_key: float((student_preds[h_key] - teacher_preds[h_key]).abs().mean())
        for h_key in teacher_preds
    }
    mae_max = max(mae_per_h.values())
    mae_mean = sum(mae_per_h.values()) / len(mae_per_h)
    print(f"[distill] MAE per horizon: {mae_per_h}")
    print(f"[distill] MAE max={mae_max:.4f} mean={mae_mean:.4f}")

    # ── 7. Save and report sizes ─────────────────────────────────────
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(student.state_dict(), STUDENT_PT)
    student_size = STUDENT_PT.stat().st_size
    ratio = student_size / teacher_size
    print(
        f"[distill] sizes: teacher={teacher_size} bytes, student={student_size} bytes "
        f"({ratio*100:.1f}% of teacher)"
    )
    sha = hashlib.sha256(STUDENT_PT.read_bytes()).hexdigest()

    md = f"""# SLA Risk Head — v0.4 EDGE-distilled

3GPP TS 28.105 Model Description Card (distilled student).

## Identification
- **Name:** `horizon_ric.heads.SLARiskHead` (edge-distilled)
- **Model:** `horizon_ric.heads.SLARiskHead`
- **Version:** v0.4-jepa-edge
- **Artefact:** `sla_head_v0.4_jepa_edge.pt`
- **SHA-256:** `{sha}`
- **File size (bytes):** {student_size}
- **Parameter count:** {student_params}
- **Teacher size (bytes):** {teacher_size}
- **Compression ratio:** {ratio*100:.1f}% of teacher

## Training Data
- **Source:** distilled from teacher checkpoint
  `checkpoints/sla_head_v0.4_jepa.pt` over a Gaussian latent batch of
  shape (N, 128) (see `scripts/distill_for_edge.py` for the exact
  sampling code). The teacher itself was trained on the real corpus
  documented in `checkpoints/sla_head_v0.4_jepa.md` (`## Training data
  — REAL corpus`).
- **Train samples:** see `scripts/distill_for_edge.py` for exact count

## Distillation
- Loss: KL(teacher || student) per horizon, mean over horizons.
- Temperature: {TEMPERATURE}
- Train epochs: {len(history)}  (patience={patience})

## Metrics
- Best val KL: {best_kl:.4f}
- MAE vs teacher (val): per-horizon {mae_per_h}; max={mae_max:.4f}, mean={mae_mean:.4f}

## Edge architecture
- latent_dim={EDGE_LATENT_DIM}  (vs teacher 128)
- hidden_dim={EDGE_HIDDEN_DIM}  (vs teacher 128)
- num_bins=51 (unchanged)

## Reproducibility
```
.venv/bin/python scripts/distill_for_edge.py
```
"""
    STUDENT_MD.write_text(md, encoding="utf-8")

    # ── 8. Validate constraints ──────────────────────────────────────
    fail = False
    if ratio > 0.30:
        print(f"FAIL: student is {ratio*100:.1f}% > 30% of teacher size")
        fail = True
    if mae_max > 0.05:
        print(f"FAIL: MAE max {mae_max:.4f} > 0.05")
        fail = True

    summary = {
        "teacher_size_bytes": teacher_size,
        "student_size_bytes": student_size,
        "ratio": ratio,
        "mae_per_horizon": mae_per_h,
        "mae_max": mae_max,
        "mae_mean": mae_mean,
        "best_val_kl": best_kl,
        "epochs_run": len(history),
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(summary, indent=2))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
