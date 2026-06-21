"""AI-PHY model card lineage — extends `observability/model_card.py`.

Every AI-PHY model swap (HybridDeepRx version, DPoD coefficients, neural-
RX checkpoint, learned-constellation table) emits a TS 28.105 §7.4 model
card with the 4 mandatory fields PLUS AI-PHY-specific metadata:

  * ``block_replaced``           — what classical L1 block this AI replaces
  * ``applicable_pa_regimes``    — list of (backoff_min_dB, backoff_max_dB)
  * ``applicable_mobility_classes``
  * ``cell_config_tags``         — e.g. ``["FR1-3.5GHz", "100MHz", "MIMO-4x4"]``
  * ``previous_version_sha``     — lineage chain back to predecessor
  * ``training_data_lineage_ref`` — pointer to ``data/lineage.py::DataManifest``

Reference
---------
3GPP TS 28.105 §7.4 — AI/ML inference function description (4 mandatory
fields). See ~/.claude/plans/AUDIT_STANDARDS.md §10.1 P0-2 — the mandatory
fields are at §7.4, NOT §6.4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

PHYBlockKind = Literal[
    "channel_estimation",
    "equalization",
    "symbol_demapping",
    "constellation_mapping",
    "papr_shaping",
    "sic_decoder",
]

MobilityClass = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PARange:
    """One PA-backoff range in dB; min ≤ max, both inclusive."""

    min_db: float
    max_db: float

    def __post_init__(self):
        if self.min_db > self.max_db:
            raise ValueError(
                f"PARange min_db {self.min_db} must be ≤ max_db {self.max_db}"
            )

    def contains(self, backoff_db: float) -> bool:
        return self.min_db <= backoff_db <= self.max_db


@dataclass(frozen=True)
class AIPhyModelCard:
    """TS 28.105 §7.4 model card with AI-PHY extensions.

    Carries the 4 mandatory §7.4 fields:
      * ``inference_type``
      * ``expected_run_time_context``
      * ``last_training_dataset_uri`` + ``last_training_dataset_sha256``
      * ``evaluation_report_uri`` + ``evaluation_report_sha256``

    Plus AI-PHY-specific lineage.
    """

    # ─── TS 28.105 §7.4 mandatory ──────────────────────────────────────
    name: str
    version: str
    sha256: str
    inference_type: str  # "real-time" / "batch" / "stream"
    expected_run_time_context: str
    last_training_dataset_uri: str
    last_training_dataset_sha256: str
    evaluation_report_uri: str
    evaluation_report_sha256: str

    # ─── AI-PHY extension ──────────────────────────────────────────────
    block_replaced: PHYBlockKind
    applicable_pa_regimes: tuple[PARange, ...]
    applicable_mobility_classes: tuple[MobilityClass, ...]
    cell_config_tags: tuple[str, ...] = field(default_factory=tuple)
    previous_version_sha: Optional[str] = None
    training_data_lineage_ref: Optional[str] = None  # SHA from data/lineage.py
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ─── Helpers ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "inference_type": self.inference_type,
            "expected_run_time_context": self.expected_run_time_context,
            "last_training_dataset_uri": self.last_training_dataset_uri,
            "last_training_dataset_sha256": self.last_training_dataset_sha256,
            "evaluation_report_uri": self.evaluation_report_uri,
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "block_replaced": self.block_replaced,
            "applicable_pa_regimes": [
                {"min_db": p.min_db, "max_db": p.max_db}
                for p in self.applicable_pa_regimes
            ],
            "applicable_mobility_classes": list(self.applicable_mobility_classes),
            "cell_config_tags": list(self.cell_config_tags),
            "previous_version_sha": self.previous_version_sha,
            "training_data_lineage_ref": self.training_data_lineage_ref,
            "issued_at": self.issued_at.isoformat(),
        }

    def card_sha256(self) -> str:
        """Stable SHA-256 over canonical-JSON of this card."""
        d = self.to_dict()
        d.pop("issued_at")  # exclude wall-clock so SHAs are content-addressed
        blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def applies_to(
        self,
        *,
        pa_backoff_db: Optional[float] = None,
        mobility: Optional[MobilityClass] = None,
        cell_tag: Optional[str] = None,
    ) -> bool:
        """Check whether this card's applicability covers the given context."""
        if pa_backoff_db is not None:
            if not any(r.contains(pa_backoff_db) for r in self.applicable_pa_regimes):
                return False
        if mobility is not None and mobility not in self.applicable_mobility_classes:
            return False
        if cell_tag is not None and cell_tag not in self.cell_config_tags:
            return False
        return True


# ─── Factory + registry ──────────────────────────────────────────────


_REGISTRY: dict[str, AIPhyModelCard] = {}


def emit_ai_phy_card(card: AIPhyModelCard) -> AIPhyModelCard:
    """Register an AI-PHY card. Indexed by ``card_sha256()`` so the
    lineage chain is content-addressed (previous_version_sha references
    a card's content hash, not its weight-file SHA)."""
    _REGISTRY[card.card_sha256()] = card
    return card


def replay_lineage_chain(card_sha: str) -> list[AIPhyModelCard]:
    """Walk the registry backwards via ``previous_version_sha`` references.

    Returns the chain ending with the card identified by ``card_sha``,
    in chronological order (oldest first).
    """
    if card_sha not in _REGISTRY:
        raise KeyError(f"card SHA {card_sha} not in registry")
    chain: list[AIPhyModelCard] = []
    cur: Optional[AIPhyModelCard] = _REGISTRY[card_sha]
    while cur is not None:
        chain.append(cur)
        if cur.previous_version_sha is None:
            break
        cur = _REGISTRY.get(cur.previous_version_sha)
    chain.reverse()
    return chain


def clear_registry() -> None:
    """Test helper — wipes the in-memory registry."""
    _REGISTRY.clear()


__all__ = [
    "AIPhyModelCard",
    "MobilityClass",
    "PARange",
    "PHYBlockKind",
    "clear_registry",
    "emit_ai_phy_card",
    "replay_lineage_chain",
]
