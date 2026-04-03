"""
rApp Training Service for Non-RT RIC.

Runs in the Service Management and Orchestration (SMO) layer
to train/retrain the SAC-LTC model using aggregated data from
multiple near-RT RIC xApps, then pushes updated weights via A1.

Architecture:
  - Collects KPI data from near-RT RICs via R1 interface
  - Trains SAC-LTC agent on aggregated experience
  - Evaluates model against performance thresholds
  - Pushes approved models to xApps via A1 ML policies
  - Monitors for model degradation and triggers retraining

Reference:
  O-RAN WG2 AI/ML Framework, Release L, 2025.
  "Multi-Scale Agentic AI for Autonomous O-RAN", 2025.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ModelVersion:
    """Metadata for a trained model version."""
    version_id: str
    timestamp: float
    metrics: Dict[str, float]
    config: Dict
    status: str = "candidate"  # candidate, approved, deployed, retired
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingJob:
    """Configuration for a training job."""
    job_id: str
    data_sources: List[str]
    hyperparameters: Dict
    target_metrics: Dict[str, float]
    max_training_steps: int = 100_000
    evaluation_interval: int = 1000
    early_stopping_patience: int = 10


class ModelCatalog:
    """
    O-RAN-compliant model catalog with version management.

    Supports the O-RAN AI/ML model lifecycle:
    train -> validate -> register -> deploy -> monitor -> retrain
    """

    def __init__(self, catalog_dir: str = "models/catalog"):
        self.catalog_dir = catalog_dir
        self.versions: Dict[str, ModelVersion] = {}
        self._deployed_version: Optional[str] = None

    def register(self, version: ModelVersion):
        """Register a new model version as candidate."""
        self.versions[version.version_id] = version

    def approve(self, version_id: str) -> bool:
        """Approve a candidate model for deployment."""
        if version_id not in self.versions:
            return False
        version = self.versions[version_id]
        if version.status != "candidate":
            return False
        version.status = "approved"
        return True

    def deploy(self, version_id: str) -> bool:
        """Mark a model as deployed, retiring the previous."""
        if version_id not in self.versions:
            return False
        version = self.versions[version_id]
        if version.status != "approved":
            return False
        # Retire current
        if self._deployed_version:
            self.versions[self._deployed_version].status = "retired"
        version.status = "deployed"
        self._deployed_version = version_id
        return True

    @property
    def deployed_version(self) -> Optional[ModelVersion]:
        if self._deployed_version:
            return self.versions.get(self._deployed_version)
        return None

    def list_versions(self) -> List[ModelVersion]:
        return list(self.versions.values())


class PerformanceMonitor:
    """
    Online model performance monitoring.

    Detects degradation in spectral efficiency, collision rate,
    or interference metrics and triggers retraining.
    """

    def __init__(
        self,
        window_size: int = 100,
        degradation_threshold: float = 0.1,
        min_samples: int = 50,
    ):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.min_samples = min_samples

        self._metrics_buffer: Dict[str, List[float]] = {
            "spectral_efficiency": [],
            "collision_rate": [],
            "switching_rate": [],
            "reward": [],
        }
        self._baseline: Dict[str, float] = {}

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics from model validation."""
        self._baseline = metrics.copy()

    def update(self, metrics: Dict[str, float]):
        """Add new metrics observation."""
        for key, value in metrics.items():
            if key in self._metrics_buffer:
                self._metrics_buffer[key].append(value)
                # Keep window
                if len(self._metrics_buffer[key]) > self.window_size:
                    self._metrics_buffer[key] = self._metrics_buffer[key][-self.window_size:]

    def check_degradation(self) -> Dict[str, bool]:
        """Check if any metric has degraded beyond threshold."""
        results = {}
        for key, buffer in self._metrics_buffer.items():
            if len(buffer) < self.min_samples or key not in self._baseline:
                results[key] = False
                continue

            current = sum(buffer[-self.min_samples:]) / self.min_samples
            baseline = self._baseline[key]

            if key == "collision_rate" or key == "switching_rate":
                # Higher is worse
                degraded = current > baseline * (1.0 + self.degradation_threshold)
            else:
                # Higher is better
                degraded = current < baseline * (1.0 - self.degradation_threshold)

            results[key] = degraded
        return results

    @property
    def should_retrain(self) -> bool:
        """True if any metric has degraded."""
        return any(self.check_degradation().values())


class RAppTrainingService:
    """
    Non-RT RIC rApp for SAC-LTC model training and lifecycle management.

    Provides:
      - Automated training from aggregated xApp data
      - Model validation and approval workflow
      - A1 policy-based model distribution
      - Performance monitoring and retraining triggers
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        catalog_dir: str = "models/catalog",
    ):
        self.device = device
        self.catalog = ModelCatalog(catalog_dir)
        self.monitor = PerformanceMonitor()
        self._training_jobs: Dict[str, TrainingJob] = {}

    def submit_training_job(self, job: TrainingJob) -> str:
        """Submit a new training job."""
        self._training_jobs[job.job_id] = job
        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get training job status."""
        if job_id not in self._training_jobs:
            return None
        job = self._training_jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": "submitted",
            "data_sources": job.data_sources,
            "max_steps": job.max_training_steps,
        }

    def generate_a1_policy(self, version_id: str) -> Dict:
        """
        Generate O-RAN A1 ML policy for model distribution.

        Returns a policy document conforming to A1 interface spec
        for pushing model updates to near-RT RIC xApps.
        """
        version = self.catalog.versions.get(version_id)
        if not version:
            return {}

        return {
            "policy_type": "AI_ML_MODEL_UPDATE",
            "policy_id": f"preceptualai-dsa-{version_id}",
            "scope": {
                "ue_group": "all",
                "cell_group": "all",
            },
            "model_info": {
                "model_id": f"sac-ltc-{version_id}",
                "model_type": "SAC_LTC_ACTOR",
                "version": version_id,
                "checkpoint_path": version.checkpoint_path,
                "metrics": version.metrics,
                "status": version.status,
            },
            "deployment_config": {
                "inference_interval_ms": 10,
                "batch_size": 1,
                "precision": "fp16",
            },
            "timestamp": time.time(),
        }

    def generate_e2_prb_blanking_policy(
        self,
        cell_id: str,
        blanked_prbs: List[int],
        duration_ms: int = 100,
    ) -> Dict:
        """
        Generate O-RAN E2SM PRB Blanking policy.

        Translates SAC-LTC agent's spectrum decisions into
        E2-compliant PRB blanking commands.
        """
        return {
            "policy_type": "O-PRBBlankingPolicy",
            "cell_id": cell_id,
            "blanked_prbs": blanked_prbs,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            "source": "preceptualai-dsa-rapp",
        }
