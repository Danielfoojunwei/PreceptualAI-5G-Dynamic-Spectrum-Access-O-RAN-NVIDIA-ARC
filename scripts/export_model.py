"""
Model export CLI for PreceptualAI.

Loads a trained SAC-LTC checkpoint, exports the actor to ONNX,
optionally compiles a TensorRT engine, and runs validation
plus latency benchmarking.

Usage:
    python scripts/export_model.py --checkpoint models/sac_ltc.pt \
        --output models/actor.onnx
    python scripts/export_model.py --checkpoint models/sac_ltc.pt \
        --output models/actor.onnx --tensorrt
    python scripts/export_model.py --checkpoint models/sac_ltc.pt \
        --output models/actor.onnx --tensorrt --precision int8
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preceptualai.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAC-LTC actor to ONNX / TensorRT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the SAC-LTC checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the ONNX model (.onnx).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=10,
        help="Number of channels (determines input_dim).",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=3,
        help="Features per channel.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Observation sequence length.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LTC hidden dimension.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="LTC latent dimension.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LTC layers.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Also compile a TensorRT engine.",
    )
    parser.add_argument(
        "--trt-output",
        type=str,
        default=None,
        help="TensorRT engine output path (default: same as --output with .trt).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TensorRT precision.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for TensorRT optimisation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run latency benchmark after export.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=1000,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip numerical validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dim = args.num_channels * args.num_features
    num_actions = args.num_channels

    # ------------------------------------------------------------------
    # 1. Load checkpoint and reconstruct agent
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)

    from preceptualai.agent.sac_ltc import SACLTCAgent

    device = torch.device("cpu")
    state_shape = (args.sequence_length, input_dim)

    agent = SACLTCAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        input_dim=input_dim,
        device=device,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
    )
    agent.load(args.checkpoint)
    logger.info("Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # 2. Export to ONNX
    # ------------------------------------------------------------------
    logger.info("Exporting to ONNX: %s", args.output)

    from preceptualai.export.onnx_export import export_actor_onnx

    onnx_path = export_actor_onnx(
        agent=agent,
        output_path=args.output,
        sequence_length=args.sequence_length,
        input_dim=input_dim,
        opset_version=args.opset,
        validate=not args.no_validate,
    )
    logger.info("ONNX export complete: %s", onnx_path)

    # ------------------------------------------------------------------
    # 3. Benchmark ONNX
    # ------------------------------------------------------------------
    if args.benchmark:
        logger.info("Benchmarking ONNX Runtime ...")
        from preceptualai.export.onnx_export import benchmark_onnx

        try:
            latency = benchmark_onnx(
                onnx_path,
                sequence_length=args.sequence_length,
                input_dim=input_dim,
                num_iterations=args.benchmark_iters,
            )
            logger.info("ONNX latency: %.3f ms/inference", latency)
        except ImportError:
            logger.warning("onnxruntime not installed — skipping ONNX benchmark.")

    # ------------------------------------------------------------------
    # 4. TensorRT compilation (optional)
    # ------------------------------------------------------------------
    if args.tensorrt:
        trt_output = args.trt_output
        if trt_output is None:
            trt_output = args.output.rsplit(".", 1)[0] + ".trt"

        logger.info(
            "Compiling TensorRT engine: %s (precision=%s)",
            trt_output,
            args.precision,
        )

        from preceptualai.export.tensorrt_export import compile_tensorrt

        try:
            trt_path = compile_tensorrt(
                onnx_path=onnx_path,
                output_path=trt_output,
                precision=args.precision,
                max_batch_size=args.max_batch_size,
            )
            logger.info("TensorRT engine saved: %s", trt_path)

            if args.benchmark:
                logger.info("Benchmarking TensorRT ...")
                from preceptualai.export.tensorrt_export import benchmark_tensorrt

                trt_latency = benchmark_tensorrt(
                    trt_path,
                    sequence_length=args.sequence_length,
                    input_dim=input_dim,
                    batch_size=1,
                    num_iterations=args.benchmark_iters,
                )
                logger.info("TensorRT latency: %.3f ms/inference", trt_latency)

        except Exception as exc:
            logger.error("TensorRT compilation failed: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    logger.info("--- Export Summary ---")
    logger.info("  ONNX model : %s", onnx_path)
    if args.tensorrt:
        logger.info("  TRT engine : %s", trt_output)
    logger.info("  Input shape: (%d, %d, %d)", 1, args.sequence_length, input_dim)
    logger.info("  Num actions: %d", num_actions)
    logger.info("Done.")


if __name__ == "__main__":
    main()
