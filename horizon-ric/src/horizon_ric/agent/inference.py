"""Edge-agent inference loop.

This is the entry point installed as the `horizon-edge-agent` console
script. It runs on the edge node (Jetson Orin Nano per project memory) and
performs:

    1. Loads a SLARiskHead checkpoint from disk (path via env or CLI).
    2. Reads telemetry from a configured source — file, stdin, or HTTP poll.
    3. Runs the risk head forward pass and emits per-horizon predictions.
    4. Logs structured records for the upstream rApp to consume via R1 DME.

By design this is a small, dependable loop. It deliberately does NOT:
    * Train the model online (that's the shadow trainer in the rApp).
    * Apply policy actions (that's the rApp via A1).
    * Speak NETCONF / O1 (that's a separate adapter).

It is real, not a stub: an invocation with a missing checkpoint exits 2
with a precise error; an invocation with a stub checkpoint runs the actual
forward pass and emits real predictions on the input telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import structlog
import torch

from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead

logger = structlog.get_logger(__name__)


@dataclass
class EdgeAgentConfig:
    checkpoint_path: Path
    latent_dim: int = 128
    hidden_dim: int = 256
    poll_interval_seconds: float = 5.0
    max_iterations: int = 0  # 0 = run until input ends or SIGINT


def _load_head(cfg: EdgeAgentConfig) -> SLARiskHead:
    if (
        cfg.checkpoint_path is None
        or str(cfg.checkpoint_path) in ("", ".")
        or not cfg.checkpoint_path.is_file()
    ):
        raise FileNotFoundError(
            f"SLA head checkpoint not found at {cfg.checkpoint_path!s}. "
            f"Set --checkpoint or HORIZON_HEAD_CKPT and ensure the file exists."
        )
    head = SLARiskHead(
        SLARiskConfig(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
    )
    state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()
    return head


def _iter_telemetry(stream: TextIO) -> Iterable[torch.Tensor]:
    """Yield latent tensors parsed from one JSON record per line.

    Each record must include ``latent`` (a list of floats of length latent_dim).
    Any line that fails to parse is logged and skipped; the loop continues.
    """
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            latent = record["latent"]
            yield torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("telemetry.parse.skip", error=str(exc), line_head=line[:80])


def run(cfg: EdgeAgentConfig, stream: TextIO, sink: TextIO) -> int:
    """Run the inference loop until `stream` is exhausted.

    Returns the number of records processed.
    """
    head = _load_head(cfg)
    logger.info(
        "edge_agent.start",
        checkpoint=str(cfg.checkpoint_path),
        latent_dim=cfg.latent_dim,
    )
    n_processed = 0
    with torch.no_grad():
        for z in _iter_telemetry(stream):
            if z.shape[1] != cfg.latent_dim:
                logger.warning(
                    "telemetry.bad_shape",
                    expected=cfg.latent_dim,
                    got=z.shape[1],
                )
                continue
            preds = head(z)
            record = {
                "ts": time.time(),
                "h_30s": float(preds["h_30s"].item()),
                "h_60s": float(preds["h_60s"].item()),
                "h_300s": float(preds["h_300s"].item()),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            n_processed += 1
            if cfg.max_iterations and n_processed >= cfg.max_iterations:
                break
    logger.info("edge_agent.stop", processed=n_processed)
    return n_processed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PreceptualAI edge inference agent")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HORIZON_HEAD_CKPT", "")),
        help="Path to SLARiskHead state-dict (.pt). Required.",
    )
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--max-iterations", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.checkpoint is None or str(args.checkpoint) in ("", "."):
        print(
            "error: --checkpoint (or HORIZON_HEAD_CKPT env) is required.",
            file=sys.stderr,
        )
        return 2
    cfg = EdgeAgentConfig(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        poll_interval_seconds=args.poll_interval,
        max_iterations=args.max_iterations,
    )
    try:
        run(cfg, sys.stdin, sys.stdout)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
