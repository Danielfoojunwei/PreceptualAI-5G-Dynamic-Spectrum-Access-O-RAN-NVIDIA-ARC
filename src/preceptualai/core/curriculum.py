"""
Curriculum Learning Scheduler for UHCI Training.

Progressively increases the complexity of the multi-provider environment:

  Stage 1: 2 providers  (FR1 + FR3)          — learn basic DSA
  Stage 2: 4 providers  (+ LEO + WIFI7)      — learn NTN + coexistence
  Stage 3: 6 providers  (+ GEO + ISAC)       — learn full NTN + sensing
  Stage 4: 8 providers  (+ MEO + HAPS)       — universal coverage

Progression is based on performance thresholds: the agent must sustain
a minimum reward over a sliding window before advancing.

This prevents the cold-start problem where an 8-provider, 48-action
agent sees almost-zero learning signal (random policy collision rate
≈ 40-60% across 48 channels).

References:
  Bengio et al., "Curriculum Learning", ICML 2009.
  Narvekar et al., "Curriculum Learning for RL: A Survey", JMLR 2020.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from preceptualai.env.provider_registry import ProviderType


@dataclass
class CurriculumStage:
    """A single stage in the curriculum."""
    name:           str
    providers:      List[ProviderType]
    min_reward:     float      # sustained reward to advance
    min_steps:      int        # minimum steps before advancement check
    window_size:    int = 500  # rolling window for reward check


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum scheduler."""
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            name="basic_dsa",
            providers=[ProviderType.FR1, ProviderType.FR3],
            min_reward=-0.2,
            min_steps=2000,
        ),
        CurriculumStage(
            name="ntn_coexist",
            providers=[ProviderType.FR1, ProviderType.FR3,
                       ProviderType.LEO, ProviderType.WIFI7],
            min_reward=-0.3,
            min_steps=3000,
        ),
        CurriculumStage(
            name="full_ntn_sensing",
            providers=[ProviderType.FR1, ProviderType.FR3,
                       ProviderType.LEO, ProviderType.WIFI7,
                       ProviderType.GEO, ProviderType.ISAC],
            min_reward=-0.4,
            min_steps=5000,
        ),
        CurriculumStage(
            name="universal",
            providers=list(ProviderType),
            min_reward=-999.0,  # final stage — never advance
            min_steps=0,
        ),
    ])
    # If True, allow skipping stages when reward exceeds threshold quickly
    allow_skip: bool = False


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive UHCI training.

    Tracks training progress and signals when to advance to the next stage
    (more provider types, larger action space, harder coexistence dynamics).
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self._current_stage = 0
        self._step = 0
        self._stage_step = 0
        self._reward_history: List[float] = []

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self._current_stage]

    @property
    def current_providers(self) -> List[ProviderType]:
        return self.current_stage.providers

    @property
    def stage_index(self) -> int:
        return self._current_stage

    @property
    def num_stages(self) -> int:
        return len(self.config.stages)

    @property
    def is_final_stage(self) -> bool:
        return self._current_stage >= self.num_stages - 1

    def step(self, mean_reward: float) -> bool:
        """
        Record a step's mean reward and check for stage advancement.

        Args:
            mean_reward: mean reward across all environments this step

        Returns:
            True if stage advanced (environment needs rebuilding)
        """
        self._step += 1
        self._stage_step += 1
        self._reward_history.append(mean_reward)

        # Trim history
        window = self.current_stage.window_size
        if len(self._reward_history) > window * 2:
            self._reward_history = self._reward_history[-window:]

        return self._check_advance()

    def _check_advance(self) -> bool:
        """Check if we should advance to the next curriculum stage."""
        if self.is_final_stage:
            return False

        stage = self.current_stage

        # Must have completed minimum steps in current stage
        if self._stage_step < stage.min_steps:
            return False

        # Check sustained reward over window
        window = stage.window_size
        if len(self._reward_history) < window:
            return False

        recent_avg = sum(self._reward_history[-window:]) / window
        if recent_avg >= stage.min_reward:
            self._advance()
            return True

        return False

    def _advance(self):
        """Move to the next curriculum stage."""
        old_name = self.current_stage.name
        self._current_stage += 1
        self._stage_step = 0
        new_name = self.current_stage.name
        print(f"  [Curriculum] Advanced: {old_name} → {new_name} "
              f"({len(self.current_providers)} providers)")

    def force_stage(self, stage_idx: int):
        """Force a specific stage (for testing/debugging)."""
        self._current_stage = min(stage_idx, self.num_stages - 1)
        self._stage_step = 0

    def get_stats(self) -> Dict:
        """Return curriculum statistics for logging."""
        window = self.current_stage.window_size
        recent = self._reward_history[-window:] if self._reward_history else [0.0]
        return {
            "curriculum/stage":        self._current_stage,
            "curriculum/stage_name":   self.current_stage.name,
            "curriculum/num_providers": len(self.current_providers),
            "curriculum/stage_step":   self._stage_step,
            "curriculum/total_step":   self._step,
            "curriculum/recent_reward": sum(recent) / max(len(recent), 1),
            "curriculum/threshold":    self.current_stage.min_reward,
        }
