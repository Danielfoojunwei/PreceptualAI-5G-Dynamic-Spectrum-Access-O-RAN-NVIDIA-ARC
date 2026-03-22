"""
ONNX export for SAC-LTC actor networks.

Exports the trained LTC actor to ONNX format with dynamic batch axes
and validates numerical equivalence against the PyTorch model.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class OnnxExportError(Exception):
    """Raised when ONNX export or validation fails."""


def _build_dummy_input(
    batch_size: int,
    sequence_length: int,
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a representative dummy input for tracing."""
    return torch.randn(batch_size, sequence_length, input_dim, device=device)


def _trace_actor(actor: torch.nn.Module) -> torch.jit.ScriptModule:
    """
    Script the actor module so the LTC sequential loop is captured.

    ``torch.jit.script`` handles Python control flow (for-loops over
    the sequence dimension in LTCEncoder) correctly, whereas
    ``torch.jit.trace`` would unroll a single-length loop.
    """
    try:
        scripted = torch.jit.script(actor)
        return scripted
    except Exception as exc:
        logger.warning(
            "torch.jit.script failed (%s); falling back to trace-based export.",
            exc,
        )
        return None


def export_actor_onnx(
    agent: object,
    output_path: Union[str, Path],
    sequence_length: int = 16,
    input_dim: int = 30,
    opset_version: int = 17,
    batch_size: int = 1,
    validate: bool = True,
    atol: float = 1e-5,
) -> str:
    """
    Export the actor network from a trained SAC-LTC agent to ONNX.

    Args:
        agent: A ``SACLTCAgent`` instance (or any object with an ``.actor`` attribute).
        output_path: Destination ``.onnx`` file path.
        sequence_length: Temporal window length (T).
        input_dim: Feature dimension (num_channels * num_features).
        opset_version: ONNX opset version (>= 14 required for LayerNorm).
        batch_size: Batch size used during tracing (dynamic axes still apply).
        validate: If True, load the exported model and validate outputs.
        atol: Absolute tolerance for validation.

    Returns:
        The resolved output path as a string.

    Raises:
        OnnxExportError: On export failure or validation mismatch.
    """
    import onnx

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Extract actor and move to CPU for export
    actor: torch.nn.Module = getattr(agent, "actor", agent)
    actor_cpu = actor.cpu().eval()
    device = torch.device("cpu")

    dummy = _build_dummy_input(batch_size, sequence_length, input_dim, device)

    # Compute reference output
    with torch.no_grad():
        ref_output = actor_cpu(dummy)

    dynamic_axes = {
        "observation": {0: "batch_size"},
        "action_probs": {0: "batch_size"},
    }

    try:
        torch.onnx.export(
            actor_cpu,
            (dummy,),
            output_path,
            input_names=["observation"],
            output_names=["action_probs"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    except Exception as exc:
        raise OnnxExportError(f"ONNX export failed: {exc}") from exc

    # Validate the exported model structure
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    logger.info("ONNX model saved to %s", output_path)

    # Numerical validation
    if validate:
        _validate_onnx_output(
            output_path, dummy.numpy(), ref_output.numpy(), atol
        )

    return output_path


def _validate_onnx_output(
    onnx_path: str,
    input_array: np.ndarray,
    expected_output: np.ndarray,
    atol: float,
) -> None:
    """
    Load the ONNX model with ONNX Runtime and check numerical equivalence.

    Raises:
        OnnxExportError: If outputs diverge beyond ``atol``.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("onnxruntime not installed — skipping validation.")
        return

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {"observation": input_array.astype(np.float32)}
    ort_outputs = sess.run(None, ort_inputs)
    actual = ort_outputs[0]

    max_diff = float(np.max(np.abs(actual - expected_output)))
    if max_diff > atol:
        raise OnnxExportError(
            f"ONNX validation failed: max abs diff = {max_diff:.8f} "
            f"(tolerance = {atol})"
        )
    logger.info(
        "ONNX validation passed (max diff = %.2e, tolerance = %.2e)",
        max_diff,
        atol,
    )


def benchmark_onnx(
    onnx_path: Union[str, Path],
    sequence_length: int = 16,
    input_dim: int = 30,
    num_iterations: int = 1000,
    batch_size: int = 1,
) -> float:
    """
    Benchmark ONNX Runtime inference latency.

    Returns:
        Mean latency in milliseconds.
    """
    import time

    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    dummy = np.random.randn(batch_size, sequence_length, input_dim).astype(np.float32)

    # Warmup
    for _ in range(50):
        sess.run(None, {"observation": dummy})

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iterations):
        sess.run(None, {"observation": dummy})
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / num_iterations) * 1000.0
    logger.info(
        "ONNX Runtime benchmark: %.3f ms/inference (%d iterations)",
        mean_ms,
        num_iterations,
    )
    return mean_ms
