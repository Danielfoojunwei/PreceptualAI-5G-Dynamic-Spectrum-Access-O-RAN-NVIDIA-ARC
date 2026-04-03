"""
Pydantic configuration schema for the PreceptualAI xApp.

Provides a single validated configuration object that covers all
subsystems: environment, model, inference, xApp, and monitoring.
Supports loading from YAML files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class EnvironmentConfig(BaseModel):
    """Spectrum environment configuration."""

    type: Literal["simulated", "oran", "aodt"] = Field(
        "simulated", description="Environment backend."
    )
    num_channels: int = Field(10, ge=1)
    num_features: int = Field(3, ge=1)
    sequence_length: int = Field(16, ge=1)
    max_steps: int = Field(200, ge=1)
    reward_success: float = 1.0
    reward_collision: float = -1.0
    reward_switch: float = -0.1

    # Simulated-only
    pu_on_prob: float = Field(0.3, ge=0.0, le=1.0)
    pu_off_prob: float = Field(0.5, ge=0.0, le=1.0)
    observation_noise_std: float = 0.1
    snr_free: float = 1.0
    snr_occupied: float = 0.2
    interference_occupied: float = 0.8
    interference_free: float = 0.1
    feature_noise_std: float = 0.05

    # AODT-only
    channel_model: str = "UMa"
    num_prbs: int = 52
    gpu_idx: int = 0


class ModelConfig(BaseModel):
    """LTC model architecture configuration."""

    hidden_dim: int = Field(128, ge=1)
    latent_dim: int = Field(128, ge=1)
    num_layers: int = Field(2, ge=1)
    dt: float = Field(1.0, gt=0.0)


class TrainingConfig(BaseModel):
    """SAC training hyperparameters."""

    lr: float = Field(3e-4, gt=0.0)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    tau: float = Field(0.005, ge=0.0, le=1.0)
    batch_size: int = Field(256, ge=1)
    buffer_size: int = Field(1_000_000, ge=1)
    learning_starts: int = Field(1000, ge=0)
    target_entropy_ratio: float = Field(0.5, gt=0.0, le=1.0)
    total_steps: int = Field(100_000, ge=1)
    eval_interval: int = Field(5000, ge=1)
    save_interval: int = Field(10_000, ge=1)
    device: str = "cuda"


class InferenceConfig(BaseModel):
    """Inference engine configuration."""

    model_path: str = Field(
        "models/actor.onnx", description="Path to the model file."
    )
    backend: Literal["auto", "tensorrt", "onnx", "pytorch"] = Field(
        "auto",
        description="Preferred inference backend. 'auto' tries TRT -> ONNX -> PyTorch.",
    )
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    max_batch_size: int = Field(1, ge=1)
    device: str = "cuda"


class XAppConfig(BaseModel):
    """O-RAN xApp configuration."""

    rmr_port: int = Field(4560, ge=1, le=65535)
    e2_node_id: str = "gnb_001"
    ran_func_id: int = 2
    kpm_report_period_ms: int = Field(100, ge=1)
    kpm_metrics: List[str] = Field(
        default_factory=lambda: [
            "DRB.UEThpDl",
            "RRU.PrbUsedDl",
            "L1M.RS-SINR",
            "RRU.PrbAvailDl",
        ]
    )
    sinr_occupied_threshold: float = 5.0
    prb_occupied_threshold: float = 0.7
    control_request_timeout_s: float = 2.0


class GrpcConfig(BaseModel):
    """gRPC server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(50051, ge=1, le=65535)
    max_workers: int = Field(4, ge=1)
    reflection: bool = True


class MonitoringConfig(BaseModel):
    """Prometheus and logging configuration."""

    prometheus_port: int = Field(9090, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    log_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

class SpectralConfig(BaseModel):
    """
    Top-level configuration for the PreceptualAI system.

    All subsystem configurations are nested under their respective keys.
    Supports construction from a YAML file via ``SpectralConfig.from_yaml()``.
    """

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)  # type: ignore[arg-type]
    model: ModelConfig = Field(default_factory=ModelConfig)  # type: ignore[arg-type]
    training: TrainingConfig = Field(default_factory=TrainingConfig)  # type: ignore[arg-type]
    inference: InferenceConfig = Field(default_factory=InferenceConfig)  # type: ignore[arg-type]
    xapp: XAppConfig = Field(default_factory=XAppConfig)  # type: ignore[arg-type]
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)  # type: ignore[arg-type]
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)  # type: ignore[arg-type]

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        """Total observation feature dimension per timestep."""
        return self.environment.num_channels * self.environment.num_features

    @property
    def num_actions(self) -> int:
        """Number of discrete actions (= number of channels)."""
        return self.environment.num_channels

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> SpectralConfig:
        """
        Load configuration from a YAML file.

        Environment variables in the form ``${VAR}`` or ``${VAR:default}``
        are expanded.
        """
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        raw = _expand_env_vars(raw)
        data = yaml.safe_load(raw)
        if data is None:
            data = {}
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_env_vars(text: str) -> str:
    """
    Expand ``${VAR}`` and ``${VAR:default}`` patterns in *text*.

    If ``VAR`` is not set and no default is given, the placeholder is left
    unchanged so that Pydantic's own validation can surface the error.
    """
    import re

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        if ":" in var:
            name, default = var.split(":", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, text)
