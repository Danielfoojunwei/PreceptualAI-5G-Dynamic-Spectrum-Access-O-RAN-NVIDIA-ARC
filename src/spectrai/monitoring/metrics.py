"""
Prometheus metrics for SpectrAI.

Defines counters, histograms, and gauges used across the xApp
inference loop, gRPC server, and monitoring dashboards.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Default registry (singleton metrics)
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry(auto_describe=True)

# Counters
predictions_total = Counter(
    "spectrai_predictions_total",
    "Total number of inference predictions made.",
    labelnames=["backend"],
    registry=REGISTRY,
)

collisions_total = Counter(
    "spectrai_collisions_total",
    "Total number of PU collisions detected.",
    registry=REGISTRY,
)

successes_total = Counter(
    "spectrai_successes_total",
    "Total number of successful transmissions.",
    registry=REGISTRY,
)

actions_total = Counter(
    "spectrai_actions_total",
    "Total actions taken per channel.",
    labelnames=["channel"],
    registry=REGISTRY,
)

channel_switches_total = Counter(
    "spectrai_channel_switches_total",
    "Total number of channel switches performed.",
    registry=REGISTRY,
)

# Histograms
inference_latency_seconds = Histogram(
    "spectrai_inference_latency_seconds",
    "Inference latency in seconds.",
    labelnames=["backend"],
    buckets=(
        0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    ),
    registry=REGISTRY,
)

kpm_indication_latency_seconds = Histogram(
    "spectrai_kpm_indication_latency_seconds",
    "Latency of KPM indication processing in seconds.",
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    ),
    registry=REGISTRY,
)

rc_control_latency_seconds = Histogram(
    "spectrai_rc_control_latency_seconds",
    "Latency of E2SM-RC control request round-trip in seconds.",
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0,
    ),
    registry=REGISTRY,
)

# Gauges
model_version = Gauge(
    "spectrai_model_version_info",
    "Currently loaded model version (encoded as float for Prometheus).",
    registry=REGISTRY,
)

channels_monitored = Gauge(
    "spectrai_channels_monitored",
    "Number of radio channels currently being monitored.",
    registry=REGISTRY,
)

current_channel = Gauge(
    "spectrai_current_channel",
    "Channel index currently selected by the agent.",
    registry=REGISTRY,
)

episode_reward = Gauge(
    "spectrai_episode_reward",
    "Cumulative reward in the current episode.",
    registry=REGISTRY,
)

collision_rate = Gauge(
    "spectrai_collision_rate",
    "Rolling collision rate (collisions / total predictions).",
    registry=REGISTRY,
)

inference_backend = Info(
    "spectrai_inference_backend",
    "Information about the active inference backend.",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def record_prediction(backend: str, latency_s: float) -> None:
    """Record a single prediction event."""
    predictions_total.labels(backend=backend).inc()
    inference_latency_seconds.labels(backend=backend).observe(latency_s)


def record_collision() -> None:
    """Record a PU collision event."""
    collisions_total.inc()


def record_success() -> None:
    """Record a successful transmission event."""
    successes_total.inc()


def record_action(channel: int) -> None:
    """Record the selected channel action."""
    actions_total.labels(channel=str(channel)).inc()
    current_channel.set(channel)


def record_channel_switch() -> None:
    """Record a channel switch event."""
    channel_switches_total.inc()


def update_collision_rate() -> None:
    """Recompute the rolling collision rate gauge."""
    total_preds = 0.0
    total_colls = 0.0
    # Sum across all label combinations for predictions
    for sample in predictions_total.collect()[0].samples:
        total_preds += sample.value
    for sample in collisions_total.collect()[0].samples:
        total_colls += sample.value
    if total_preds > 0:
        collision_rate.set(total_colls / total_preds)


def set_model_info(version: str, backend: str, path: str) -> None:
    """Update model metadata gauges."""
    # Encode version as float (e.g. "0.1.0" -> 0.1)
    try:
        parts = version.split(".")
        version_float = float(f"{parts[0]}.{parts[1]}")
    except (IndexError, ValueError):
        version_float = 0.0
    model_version.set(version_float)
    inference_backend.info(
        {"version": version, "backend": backend, "path": path}
    )


def get_metrics_text() -> str:
    """Generate Prometheus text exposition from the custom registry."""
    return str(generate_latest(REGISTRY).decode("utf-8"))
