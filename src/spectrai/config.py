"""
SpectrAI configuration schema using Pydantic.

Provides validated, typed configuration for all system components:
environment, LTC encoder, SAC agent, export, and deployment.
"""


import yaml
from pydantic import BaseModel, Field, field_validator


class EnvironmentConfig(BaseModel):
    """DSA environment parameters."""

    num_channels: int = Field(10, ge=1, le=256, description="Number of radio channels")
    sequence_length: int = Field(16, ge=1, le=1024, description="Observation window length")
    num_features: int = Field(3, ge=1, le=32, description="Features per channel")
    pu_on_prob: float = Field(0.3, ge=0.0, le=1.0)
    pu_off_prob: float = Field(0.5, ge=0.0, le=1.0)
    noise_std: float = Field(0.1, ge=0.0)
    max_steps: int = Field(200, ge=1)
    switch_penalty: float = Field(0.1, ge=0.0)


class EncoderConfig(BaseModel):
    """LTC encoder hyperparameters."""

    hidden_dim: int = Field(128, ge=1)
    latent_dim: int = Field(128, ge=1)
    num_layers: int = Field(2, ge=1, le=16)
    dt: float = Field(1.0, gt=0.0)


class AgentConfig(BaseModel):
    """SAC agent hyperparameters."""

    lr: float = Field(3e-4, gt=0.0)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    tau: float = Field(0.005, ge=0.0, le=1.0)
    buffer_size: int = Field(1_000_000, ge=1)
    batch_size: int = Field(256, ge=1)
    learning_starts: int = Field(1000, ge=0)
    target_entropy_ratio: float = Field(0.5, gt=0.0, le=1.0)


class GPUEncoderConfig(BaseModel):
    """GPU-optimized LTC encoder hyperparameters."""

    hidden_dim: int = Field(256, ge=1)
    latent_dim: int = Field(256, ge=1)
    num_layers: int = Field(3, ge=1, le=16)
    dt: float = Field(1.0, gt=0.0)
    solver: str = Field("heun", description="ODE solver: 'euler' or 'heun'")
    sub_steps: int = Field(1, ge=1)
    use_gradient_checkpointing: bool = False


class GPUAgentConfig(BaseModel):
    """GPU-optimized SAC agent hyperparameters."""

    lr: float = Field(3e-4, gt=0.0)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    tau: float = Field(0.005, ge=0.0, le=1.0)
    buffer_size: int = Field(2_000_000, ge=1)
    batch_size: int = Field(1024, ge=1)
    learning_starts: int = Field(2000, ge=0)
    target_entropy_ratio: float = Field(0.5, gt=0.0, le=1.0)
    amp_dtype: str = Field("bfloat16", description="AMP dtype: 'bfloat16', 'float16', or 'none'")
    grad_clip: float = Field(1.0, ge=0.0)


class VectorizedEnvConfig(BaseModel):
    """GPU-vectorized environment parameters."""

    num_envs: int = Field(256, ge=1, le=4096)
    num_channels: int = Field(10, ge=1, le=256)
    num_features: int = Field(3, ge=1, le=32)
    sequence_length: int = Field(16, ge=1, le=1024)
    max_steps: int = Field(200, ge=1)
    pu_on_prob: float = Field(0.3, ge=0.0, le=1.0)
    pu_off_prob: float = Field(0.5, ge=0.0, le=1.0)
    backend: str = Field("pytorch", description="Backend: 'pytorch', 'sionna', or 'warp'")
    channel_model: str = Field("TDL-A", description="3GPP channel model for Sionna backend")
    carrier_freq_hz: float = Field(3.5e9, gt=0.0)


class MultiScaleConfig(BaseModel):
    """Multi-scale LTC encoder with physics-aware timescales."""

    num_layers: int = Field(4, ge=1, le=8)
    hidden_dim: int = Field(128, ge=1)
    latent_dim: int = Field(128, ge=1)
    domain: str = Field("5g", description="Domain: '5g' or 'satellite'")
    solver: str = Field("heun", description="ODE solver: 'euler' or 'heun'")


class ExportConfig(BaseModel):
    """ONNX / TensorRT export settings."""

    onnx_opset: int = Field(17, ge=11)
    onnx_path: str = "models/spectrai_actor.onnx"
    enable_tensorrt: bool = False


class SpectralConfig(BaseModel):
    """
    Top-level SpectrAI configuration.

    Aggregates all sub-configs and supports loading from YAML.
    """

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)  # type: ignore[arg-type]
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)  # type: ignore[arg-type]
    agent: AgentConfig = Field(default_factory=AgentConfig)  # type: ignore[arg-type]
    export: ExportConfig = Field(default_factory=ExportConfig)  # type: ignore[arg-type]

    # GPU-optimized configurations
    gpu_encoder: GPUEncoderConfig = Field(default_factory=GPUEncoderConfig)  # type: ignore[arg-type]
    gpu_agent: GPUAgentConfig = Field(default_factory=GPUAgentConfig)  # type: ignore[arg-type]
    vectorized_env: VectorizedEnvConfig = Field(default_factory=VectorizedEnvConfig)  # type: ignore[arg-type]
    multiscale: MultiScaleConfig = Field(default_factory=MultiScaleConfig)  # type: ignore[arg-type]

    device: str = Field("auto", description="Device: 'cpu', 'cuda', or 'auto'")
    seed: int = Field(42, ge=0)
    num_steps: int = Field(50_000, ge=1)
    output_dir: str = "results"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "auto"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SpectralConfig":
        """Parse configuration from a YAML string."""
        data = yaml.safe_load(yaml_str)
        if data is None:
            data = {}
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "SpectralConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            return cls.from_yaml(f.read())
