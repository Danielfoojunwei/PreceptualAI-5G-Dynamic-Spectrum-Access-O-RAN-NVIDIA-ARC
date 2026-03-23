"""
Async gRPC inference server for the SpectrAI xApp.

Implements the ``SpectrAI`` service defined in ``proto/spectrai.proto``.
Loads an :class:`InferenceEngine` on startup and exposes health-check
and Prometheus metrics endpoints alongside the prediction RPCs.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import AsyncIterator, Optional

import grpc
import grpc.aio
import numpy as np

from spectrai.xapp.config import SpectralConfig
from spectrai.xapp.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import generated protobuf stubs.  If they have not been
# compiled yet we define a minimal shim so the module can still be imported
# (e.g. for type-checking or documentation builds).
# ---------------------------------------------------------------------------
try:
    from spectrai.proto import spectrai_pb2, spectrai_pb2_grpc

    _HAS_PROTO = True
except ImportError:
    _HAS_PROTO = False
    spectrai_pb2 = None  # type: ignore[assignment]
    spectrai_pb2_grpc = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Service implementation
# ---------------------------------------------------------------------------

class SpectrAIServicer:
    """
    gRPC servicer implementing the SpectrAI proto service.

    All RPC methods are async-compatible (``grpc.aio``).
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: SpectralConfig,
    ):
        self._engine = engine
        self._config = config
        self._start_time = time.monotonic()

        # Counters
        self._predictions_total: int = 0
        self._collisions_total: int = 0
        self._successes_total: int = 0

    # ------------------------------------------------------------------
    # Predict (unary)
    # ------------------------------------------------------------------

    async def Predict(self, request: object, context: grpc.aio.ServicerContext) -> object:
        """Single-shot prediction."""
        seq_len = request.sequence_length or self._config.environment.sequence_length  # type: ignore[attr-defined]
        input_dim = request.input_dim or self._config.input_dim  # type: ignore[attr-defined]

        obs = np.array(request.observation, dtype=np.float32).reshape(  # type: ignore[attr-defined]
            seq_len, input_dim
        )

        action = self._engine.predict(obs)
        latency = self._engine.last_latency_ms
        self._predictions_total += 1

        # Retrieve full probability distribution
        obs_batch = obs[np.newaxis, ...]
        probs = self._engine._infer(obs_batch)[0].tolist()

        return spectrai_pb2.PredictResponse(  # type: ignore[attr-defined]
            action=action,
            action_probs=probs,
            latency_ms=latency,
            request_id=request.request_id,  # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # PredictStream (bidirectional streaming)
    # ------------------------------------------------------------------

    async def PredictStream(
        self,
        request_iterator: AsyncIterator,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator:
        """Bidirectional streaming prediction."""
        async for request in request_iterator:
            seq_len = request.sequence_length or self._config.environment.sequence_length
            input_dim = request.input_dim or self._config.input_dim

            obs = np.array(request.observation, dtype=np.float32).reshape(
                seq_len, input_dim
            )

            action = self._engine.predict(obs)
            latency = self._engine.last_latency_ms
            self._predictions_total += 1

            obs_batch = obs[np.newaxis, ...]
            probs = self._engine._infer(obs_batch)[0].tolist()

            yield spectrai_pb2.PredictResponse(  # type: ignore[attr-defined]
                action=action,
                action_probs=probs,
                latency_ms=latency,
                request_id=request.request_id,  # type: ignore[attr-defined]
            )

    # ------------------------------------------------------------------
    # GetHealth
    # ------------------------------------------------------------------

    async def GetHealth(self, request: object, context: grpc.aio.ServicerContext) -> object:
        """Return current serving status."""
        uptime = time.monotonic() - self._start_time
        status = spectrai_pb2.HealthStatus.ServingStatus.SERVING  # type: ignore[attr-defined]

        return spectrai_pb2.HealthStatus(  # type: ignore[attr-defined]
            status=status,
            model_path=str(self._config.inference.model_path),
            backend=self._engine.backend or "unknown",
            version="0.1.0",
            uptime_seconds=uptime,
        )

    # ------------------------------------------------------------------
    # GetMetrics
    # ------------------------------------------------------------------

    async def GetMetrics(self, request: object, context: grpc.aio.ServicerContext) -> object:
        """Return current metrics snapshot."""
        latency_mean = self._engine.latency_ms

        # Compute percentiles from the engine's latency deque
        latencies = list(self._engine._latencies)
        if latencies:
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
            p99 = sorted_lat[p99_idx]
        else:
            p50 = 0.0
            p99 = 0.0

        # Prometheus text exposition
        prom_text = ""
        try:
            from prometheus_client import generate_latest

            prom_text = generate_latest().decode("utf-8")
        except ImportError:
            pass

        return spectrai_pb2.MetricsResponse(  # type: ignore[attr-defined]
            predictions_total=self._predictions_total,
            collisions_total=self._collisions_total,
            successes_total=self._successes_total,
            inference_latency_mean_ms=latency_mean,
            inference_latency_p50_ms=p50,
            inference_latency_p99_ms=p99,
            model_version="0.1.0",
            channels_monitored=self._config.environment.num_channels,
            prometheus_text=prom_text,
        )

    # ------------------------------------------------------------------
    # Counter updates (called externally by xApp loop)
    # ------------------------------------------------------------------

    def record_collision(self) -> None:
        self._collisions_total += 1

    def record_success(self) -> None:
        self._successes_total += 1


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

async def serve(config: SpectralConfig) -> None:
    """
    Start the SpectrAI gRPC server.

    Blocks until a termination signal (SIGINT / SIGTERM) is received.
    """
    if not _HAS_PROTO:
        raise RuntimeError(
            "Protobuf stubs not found. Generate them with:\n"
            "  python -m grpc_tools.protoc -Iproto "
            "--python_out=src/spectrai/proto "
            "--grpc_python_out=src/spectrai/proto "
            "proto/spectrai.proto"
        )

    # Load inference engine
    logger.info(
        "Loading inference engine: path=%s, device=%s",
        config.inference.model_path,
        config.inference.device,
    )
    engine = InferenceEngine(
        model_path=config.inference.model_path,
        device=config.inference.device,
        sequence_length=config.environment.sequence_length,
        input_dim=config.input_dim,
        num_actions=config.num_actions,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
    )
    logger.info("Inference engine loaded (backend=%s)", engine.backend)

    # Create servicer and server
    servicer = SpectrAIServicer(engine=engine, config=config)

    server = grpc.aio.server()
    spectrai_pb2_grpc.add_SpectrAIServicer_to_server(servicer, server)

    listen_addr = f"{config.grpc.host}:{config.grpc.port}"
    server.add_insecure_port(listen_addr)

    # Optional gRPC reflection
    if config.grpc.reflection:
        try:
            from grpc_reflection.v1alpha import reflection as grpc_reflection

            service_names = (
                spectrai_pb2.DESCRIPTOR.services_by_name["SpectrAI"].full_name,
                grpc_reflection.SERVICE_NAME,
            )
            grpc_reflection.enable_server_reflection(service_names, server)
            logger.info("gRPC reflection enabled")
        except ImportError:
            logger.debug("grpc-reflection not installed — skipping.")

    # Start Prometheus metrics server
    try:
        from prometheus_client import start_http_server as start_prom_server

        start_prom_server(config.monitoring.prometheus_port)
        logger.info(
            "Prometheus metrics server on port %d", config.monitoring.prometheus_port
        )
    except ImportError:
        logger.debug("prometheus_client not installed — metrics endpoint disabled.")

    await server.start()
    logger.info("SpectrAI gRPC server listening on %s", listen_addr)

    # Graceful shutdown on signal
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await shutdown_event.wait()
    logger.info("Shutting down gRPC server ...")
    await server.stop(grace=5)
    logger.info("Server stopped.")


def run_server(config_path: Optional[str] = None) -> None:
    """
    Entry point for the gRPC server.

    Loads configuration from YAML (or uses defaults) and starts the
    async event loop.
    """
    from spectrai.monitoring.logging import configure_logging

    if config_path:
        config = SpectralConfig.from_yaml(config_path)
    else:
        config = SpectralConfig()

    configure_logging(
        level=config.monitoring.log_level,
        fmt=config.monitoring.log_format,
        log_file=config.monitoring.log_file,
    )

    asyncio.run(serve(config))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpectrAI gRPC Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to spectrai config YAML file.",
    )
    args = parser.parse_args()
    run_server(config_path=args.config)
