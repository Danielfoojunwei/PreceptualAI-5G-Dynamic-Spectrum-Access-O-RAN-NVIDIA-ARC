"""
Federated Learning configuration for SpectrAI.

Defines the FLConfig pydantic model controlling all aspects of the
hybrid LTC-aware federated aggregation process.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FLConfig(BaseModel):
    """
    Configuration for SpectrAI federated learning.

    The hybrid LTC aggregation separates model parameters into two groups:
      - Structural weights (W_h, W_x, heads, layer norms, projections):
        fully averaged across devices via standard FedAvg.
      - Time-constant weights (W_tau, tau_base): partially personalized per
        device via a configurable mixing ratio.

    Attributes:
        tau_mix_ratio: Mixing ratio for time-constant weights.
            0.0 = fully local (device keeps its own tau weights entirely).
            1.0 = fully global (standard FedAvg for tau weights too).
            Default 0.3 keeps 70% local personality, 30% global consensus.
        num_rounds: Total number of federated communication rounds.
        local_steps_per_round: Number of local SGD / environment steps each
            device performs between communication rounds.
        min_devices_per_round: Minimum devices that must report before
            aggregation proceeds. If fewer report, the round is skipped.
        aggregation_method: Aggregation strategy.
            "hybrid_ltc" — the novel structural/tau split (default).
            "fedavg" — standard weighted averaging of all parameters.
            "fedprox" — FedAvg with a proximal regularisation term.
        fedprox_mu: Proximal term coefficient for FedProx. Only used when
            aggregation_method is "fedprox".
        device_type: Compute device for aggregation tensors.
        aggregator_host: Host address for the gRPC aggregation server.
        aggregator_port: Port for the gRPC aggregation server.
    """

    tau_mix_ratio: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Tau mixing ratio: 0=fully local, 1=fully global",
    )
    num_rounds: int = Field(10, ge=1, description="Number of FL communication rounds")
    local_steps_per_round: int = Field(
        500, ge=1, description="Local training steps between rounds"
    )
    min_devices_per_round: int = Field(
        2,
        ge=1,
        description="Minimum devices required to proceed with aggregation",
    )
    aggregation_method: Literal["hybrid_ltc", "fedavg", "fedprox"] = Field(
        "hybrid_ltc",
        description="Aggregation strategy",
    )
    fedprox_mu: float = Field(
        0.01,
        ge=0.0,
        description="Proximal regularisation coefficient for FedProx",
    )
    device_type: str = Field("cpu", description="Compute device (cpu or cuda)")
    aggregator_host: str = Field("0.0.0.0", description="gRPC server bind address")
    aggregator_port: int = Field(
        50051, ge=1, le=65535, description="gRPC server port"
    )

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "auto"}
        if v not in allowed:
            raise ValueError(f"device_type must be one of {allowed}, got '{v}'")
        return v
