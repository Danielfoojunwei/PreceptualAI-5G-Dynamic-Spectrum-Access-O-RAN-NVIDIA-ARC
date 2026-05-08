"""Real entity-count scaling benchmark.

Walks the full encoder front-door — SpatialPrior → EntityTokenizer →
PerceiverFusion — at N ∈ {1, 10, 100, 1000, 5000} entities and prints the
wall-clock for each stage. Plots the curve to ``benchmarks/scaling.png``.

The Perceiver's fixed-latent design is the architectural bet that makes
PreceptualAI scalable to constellation-sized resource graphs: even at 5000
entities the cross-attention writes into the SAME ``(L, d)`` latent.
SpatialPrior is O(N²) by definition (pairwise look-angles); we keep it in
the benchmark so the curve shows the real cost ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from horizon_ric.encoder import (
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
    PerceiverConfig,
    PerceiverFusion,
    SpatialPrior,
)
from horizon_ric.policy import TDMPCConfig


REPO = Path(__file__).resolve().parent.parent


# Latitudes/longitudes that span the globe so the spatial-prior LOS check
# returns a non-trivial mix.
def _make_entities(n: int, *, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    entities: list[dict] = []
    for _ in range(n):
        entities.append(
            {
                "placement": "ground",
                "lat_deg": rng.uniform(-60.0, 60.0),
                "lon_deg": rng.uniform(-180.0, 180.0),
                "height_m": rng.uniform(0.0, 100.0),
            }
        )
    return entities


def _make_assets(n: int, *, attr_dim: int = 8, seed: int = 11) -> list[AssetAttributes]:
    rng = random.Random(seed)
    out: list[AssetAttributes] = []
    for _ in range(n):
        out.append(
            AssetAttributes(
                entity_type=EntityType.SAT_NGSO,
                attributes=torch.randn(attr_dim),
                xyz_m=(
                    rng.uniform(-7e6, 7e6),
                    rng.uniform(-7e6, 7e6),
                    rng.uniform(-7e6, 7e6),
                ),
            )
        )
    return out


def _ms_since(t0_ns: int) -> float:
    return (time.perf_counter_ns() - t0_ns) / 1e6


def _run_one(n: int, *, run_spatial: bool = True) -> dict:
    """Time one (N) point on the scaling curve.

    SpatialPrior is O(N²) in entities — we still run it up to N=1000 and
    skip past that to keep the benchmark < 1 minute. The Perceiver is run
    every time because it is the load-bearing scalability claim.
    """
    out: dict = {"n": n}

    # ── 1. SpatialPrior (O(N²)) ─────────────────────────────────────────
    if run_spatial:
        sp = SpatialPrior()
        entities = _make_entities(n)
        t0 = time.perf_counter_ns()
        sp_out = sp.build(entities)
        out["spatial_prior_ms"] = _ms_since(t0)
        # Sanity: shapes match.
        assert sp_out.los_mask.shape == (n, n), sp_out.los_mask.shape

    # ── 2. EntityTokenizer (O(N) at fixed d_model) ──────────────────────
    cfg_tok = EntityTokenizerConfig(d_model=64)
    tok = EntityTokenizer(cfg_tok).eval()
    assets = _make_assets(n, attr_dim=cfg_tok.attr_dims[EntityType.SAT_NGSO])
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        tokens = tok(assets)
    out["entity_tokenizer_ms"] = _ms_since(t0)
    assert tokens.shape == (n, cfg_tok.d_model)

    # ── 3. PerceiverFusion (cross-attention into FIXED latent) ─────────
    perc_cfg = PerceiverConfig(
        n_latents=64,
        d_latent=64,
        d_input=cfg_tok.d_model,
        n_self_layers=2,
        n_heads=4,
    )
    perc = PerceiverFusion(perc_cfg).eval()
    inputs = tokens.unsqueeze(0)  # (B=1, N, d)
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        z = perc(inputs)
    out["perceiver_ms"] = _ms_since(t0)
    assert z.shape == (1, perc_cfg.n_latents, perc_cfg.d_latent)

    out["total_ms"] = sum(
        v for k, v in out.items() if k.endswith("_ms")
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-png", default=str(REPO / "benchmarks" / "scaling.png"))
    parser.add_argument("--out-json", default=str(REPO / "benchmarks" / "scaling.json"))
    parser.add_argument(
        "--ns", type=int, nargs="+",
        default=[1, 10, 100, 1000, 5000],
    )
    parser.add_argument(
        "--max-spatial-n", type=int, default=1000,
        help="Skip the O(N²) SpatialPrior above this N (default 1000)",
    )
    args = parser.parse_args(argv)

    # Smoke test that TDMPC config sizes are also configurable.
    for h, samples in [(4, 32), (12, 256), (24, 512)]:
        _ = TDMPCConfig(horizon=h, n_samples=samples)

    print("=" * 72)
    print(f"Scaling benchmark — N ∈ {args.ns}")
    print("=" * 72)

    results: list[dict] = []
    for n in args.ns:
        run_sp = n <= args.max_spatial_n
        r = _run_one(n, run_spatial=run_sp)
        if not run_sp:
            r["spatial_prior_ms"] = float("nan")
        results.append(r)
        sp_str = (
            f"{r['spatial_prior_ms']:8.1f}ms" if run_sp else "    n/a   "
        )
        print(
            f"N={n:>5d}: SpatialPrior={sp_str}  "
            f"EntityTok={r['entity_tokenizer_ms']:7.1f}ms  "
            f"Perceiver={r['perceiver_ms']:6.1f}ms  "
            f"total={r['total_ms']:7.1f}ms"
        )

    # JSON dump (machine-readable).
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {args.out_json}")

    # ── Plot ────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping plot — matplotlib failed to load: {exc})")
        return 0

    ns = [r["n"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        ns,
        [r["spatial_prior_ms"] for r in results],
        "o-",
        label="SpatialPrior (O(N²))",
        color="tab:red",
    )
    ax.plot(
        ns,
        [r["entity_tokenizer_ms"] for r in results],
        "s-",
        label="EntityTokenizer",
        color="tab:orange",
    )
    ax.plot(
        ns,
        [r["perceiver_ms"] for r in results],
        "D-",
        label="PerceiverFusion (fixed latent)",
        color="tab:blue",
    )
    ax.plot(
        ns,
        [r["total_ms"] for r in results],
        "k--",
        label="total",
        alpha=0.7,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (entities)")
    ax.set_ylabel("wall-clock (ms)")
    ax.set_title("PreceptualAI encoder scaling — real production code path")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)
    print(f"plot    -> {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
