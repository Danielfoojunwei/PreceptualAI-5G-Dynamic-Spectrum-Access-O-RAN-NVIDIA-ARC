"""
TensorRT compilation for ONNX models.

Compiles an ONNX model into a TensorRT engine for maximum inference
throughput on NVIDIA GPUs.  Supports FP32, FP16, and INT8 precision.

TensorRT is an optional dependency — the module fails gracefully
with a clear error message if ``tensorrt`` is not installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TensorRT import
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt

    _HAS_TRT = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    trt = None  # type: ignore[assignment]
    _HAS_TRT = False
    TRT_LOGGER = None


class TensorRTCompileError(Exception):
    """Raised when TensorRT compilation fails."""


def _check_tensorrt() -> None:
    if not _HAS_TRT:
        raise TensorRTCompileError(
            "TensorRT Python bindings are not installed. "
            "Install with: pip install tensorrt"
        )


def compile_tensorrt(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_gb: float = 1.0,
    min_batch_size: int = 1,
    opt_batch_size: Optional[int] = None,
    calibrator: object = None,
) -> str:
    """
    Compile an ONNX model to a TensorRT engine.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_path: Path for the output ``.engine`` / ``.trt`` file.
        precision: ``"fp32"``, ``"fp16"``, or ``"int8"``.
        max_batch_size: Maximum batch size for the optimisation profile.
        workspace_gb: GPU workspace in GiB for the builder.
        min_batch_size: Minimum batch size for the optimisation profile.
        opt_batch_size: Optimal batch size (defaults to ``max_batch_size``).
        calibrator: INT8 calibration data provider (required for int8).

    Returns:
        The resolved output path as a string.

    Raises:
        TensorRTCompileError: On missing dependency or build failure.
    """
    _check_tensorrt()

    onnx_path = str(onnx_path)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if opt_batch_size is None:
        opt_batch_size = max_batch_size

    precision = precision.lower().strip()
    if precision not in ("fp32", "fp16", "int8"):
        raise TensorRTCompileError(
            f"Unsupported precision '{precision}'. Choose fp32, fp16, or int8."
        )

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise TensorRTCompileError(
            "Failed to parse ONNX model:\n" + "\n".join(errors)
        )

    logger.info(
        "Parsed ONNX model: %d inputs, %d outputs, %d layers",
        network.num_inputs,
        network.num_outputs,
        network.num_layers,
    )

    # Builder configuration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    # Precision flags
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("GPU does not have native FP16 — performance may be suboptimal.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("GPU does not have native INT8 — performance may be suboptimal.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for non-quantised layers
        if calibrator is not None:
            config.int8_calibrator = calibrator
        else:
            logger.warning(
                "No INT8 calibrator provided — using per-tensor dynamic range. "
                "For best accuracy, supply a calibration dataset."
            )

    # Optimisation profiles for dynamic batch
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = inp.shape  # e.g. (-1, seq_len, input_dim)

        min_shape = list(shape)
        opt_shape = list(shape)
        max_shape = list(shape)

        # Replace dynamic dimension (typically batch = -1)
        for d in range(len(shape)):
            if shape[d] == -1:
                min_shape[d] = min_batch_size
                opt_shape[d] = opt_batch_size
                max_shape[d] = max_batch_size

        profile.set_shape(
            name,
            tuple(min_shape),
            tuple(opt_shape),
            tuple(max_shape),
        )
        logger.info(
            "Input '%s': min=%s opt=%s max=%s",
            name,
            min_shape,
            opt_shape,
            max_shape,
        )

    config.add_optimization_profile(profile)

    # Build engine
    logger.info("Building TensorRT engine (precision=%s) ...", precision)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise TensorRTCompileError("TensorRT engine build failed (returned None).")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = len(serialized_engine) / (1 << 20)
    logger.info(
        "TensorRT engine saved to %s (%.1f MiB, precision=%s)",
        output_path,
        engine_size_mb,
        precision,
    )

    return output_path


def load_engine(engine_path: Union[str, Path]) -> object:
    """
    Deserialise a TensorRT engine from disk.

    Returns:
        A ``trt.ICudaEngine`` instance.
    """
    _check_tensorrt()

    runtime = trt.Runtime(TRT_LOGGER)
    with open(str(engine_path), "rb") as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise TensorRTCompileError(
            f"Failed to deserialise TensorRT engine from {engine_path}"
        )
    return engine


def benchmark_tensorrt(
    engine_path: Union[str, Path],
    sequence_length: int = 16,
    input_dim: int = 30,
    batch_size: int = 1,
    num_iterations: int = 1000,
    warmup: int = 50,
) -> float:
    """
    Benchmark TensorRT inference latency using CUDA events.

    Returns:
        Mean latency in milliseconds.
    """
    _check_tensorrt()

    import numpy as np

    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # noqa: F401
    except ImportError:
        raise TensorRTCompileError(
            "pycuda is required for TensorRT benchmarking. "
            "Install with: pip install pycuda"
        )

    engine = load_engine(engine_path)
    context = engine.create_execution_context()  # type: ignore[attr-defined]

    # Allocate host and device buffers
    input_shape = (batch_size, sequence_length, input_dim)
    h_input = np.random.randn(*input_shape).astype(np.float32)
    output_size = engine.get_tensor_shape("action_probs")[-1]  # type: ignore[attr-defined]
    h_output = np.empty((batch_size, output_size), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    context.set_input_shape("observation", input_shape)

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.set_tensor_address("observation", int(d_input))
        context.set_tensor_address("action_probs", int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Timed runs
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record(stream)
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.set_tensor_address("observation", int(d_input))
        context.set_tensor_address("action_probs", int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    end_event.record(stream)
    stream.synchronize()

    elapsed_ms = start_event.time_till(end_event)
    mean_ms = elapsed_ms / num_iterations

    logger.info(
        "TensorRT benchmark: %.3f ms/inference (%d iterations, precision from engine)",
        mean_ms,
        num_iterations,
    )
    return float(mean_ms)
