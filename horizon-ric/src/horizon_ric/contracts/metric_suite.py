"""MetricSuite — abstract base for vertical-specific metric sets.

Each vertical defines its own MetricSuite (PreceptualAI NTN+AI-RAN: 12 metrics
including SLA, beam, gateway, compute, GSO compliance, etc.).

PolicyValidator gates candidate-promotion against the suite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricDirection(str, Enum):
    """Whether higher or lower values are better."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    HARD_FLOOR = "hard_floor"  # must be == 0 (e.g. constraint violations)


@dataclass
class MetricDef:
    """Definition of a single metric."""

    name: str
    direction: MetricDirection
    unit: str  # e.g. "dB", "Mbps", "ms", "%", "count"
    description: str
    spec_reference: str | None = None  # e.g. "3GPP TS 28.554 §6.2"


@dataclass
class MetricResult:
    """Computed value for a single metric."""

    name: str
    value: float
    direction: MetricDirection
    horizon: str | None = None  # e.g. "30s", "1min", "5min"
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricSuite(ABC):
    """Contract for a vertical-specific metric set."""

    @abstractmethod
    def metric_definitions(self) -> list[MetricDef]:
        """List of metric definitions in this suite."""

    @abstractmethod
    def compute(
        self,
        predicted: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
    ) -> list[MetricResult]:
        """Compute metric values from predictions (and optional ground truth).

        Args:
            predicted: dict containing the model's predictions for the slice.
            ground_truth: optional dict with observed outcomes (for offline eval).

        Returns:
            List of MetricResult, one per metric.
        """

    def hard_floor_metrics(self) -> list[str]:
        """Return names of metrics that must be at hard floor (== 0).

        These are non-negotiable; a candidate fails promotion if any
        hard-floor metric is non-zero.
        """
        return [
            d.name for d in self.metric_definitions()
            if d.direction == MetricDirection.HARD_FLOOR
        ]

    def section_7_dimension(self) -> int:
        """Total number of metrics in the suite."""
        return len(self.metric_definitions())
