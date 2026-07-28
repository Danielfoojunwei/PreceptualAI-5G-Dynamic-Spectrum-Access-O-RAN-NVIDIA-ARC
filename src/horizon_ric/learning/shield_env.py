"""Real-data spectrum environment wrapped by the Decision Safety Shield.

This is the shared substrate for the closed-loop study: *does the Shield's
projection operator bias what a learner learns?* It provides

1. :class:`ShieldedSpectrumEnv` — an environment whose reward is computed from
   **measured DeepMIMO ray tracing** (per-receiver, per-subband channel gains).
   A proposed action goes through :func:`default_terrestrial_shield`, which
   *projects* it onto the nearest legal action (band membership + EIRP cap); the
   environment then executes the **projected** action and returns its real
   reward, while recording both the proposed and executed actions to a
   hash-chained evidence store.
2. :func:`load_transitions` — a **verify-gated** replay buffer. It refuses to
   emit training data from a chain whose hash links do not verify, so a tampered
   decision log cannot silently become training data.
3. A **feasibility mask derived from the Shield's own predicate**
   (:meth:`ShieldedSpectrumEnv.is_feasible`,
   :meth:`ShieldedSpectrumEnv.feasible_mask_grid`), so a benchmark never has to
   restate the constraint in its own words. Restating it is exactly how the
   published exploration-truncation rates came to be measured against the wrong
   feasible set: a centre-frequency-in-band test calls the two edge subbands
   legal, while the Shield's occupied-bandwidth-in-band test (TS 38.104) does
   not. Masking is **defence in depth** — :meth:`step` still disposes every
   action through the Shield unconditionally, so a caller that ignores the mask
   is exactly as safe as before.

Why this environment is the honest test case
--------------------------------------------
Reward is monotonically increasing in EIRP, so the *unconstrained* optimum lies
outside the feasible set and the Shield genuinely **binds**: on the committed
4096-receiver build the served fraction is ~0.374 at the 33 dBm cap versus
~0.564 at 46 dBm. Earlier Horizon benchmarks only ever exercised the Shield on
tasks whose optimum was already feasible (channel selection, coverage
regression), where the projection was provably free. Here it is not free, which
is precisely what makes the bias question measurable.

The credit-assignment problem
-----------------------------
The learner proposes ``a``; the environment executes ``Π(a)``. A learner that
attributes the observed reward to ``a`` (rather than to ``Π(a)``) is training on
censored feedback: every proposal above the cap yields the *same* reward, so the
gradient in the infeasible region vanishes and the learned value of illegal
actions is systematically wrong. Each :class:`StepResult` and each evidence
record therefore carries **both** actions plus the projection flags, so a
downstream learner can be built either way and the difference measured.

All physics is real; no synthetic channels. Pure numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.policy.emit_guards import run_guard_chain
from horizon_ric.shield import default_terrestrial_shield

# Defaults shared with the DSA / coverage benchmarks so results are comparable.
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
N_SUBBANDS = 6
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
BANDWIDTH_HZ = 20e6
NOISE_FIGURE_DB = 7.0
SINR_THRESHOLD_DB = 0.0


class EvidenceIntegrityError(RuntimeError):
    """Raised when replay is attempted over a chain that fails verification."""


@dataclass(frozen=True)
class SpectrumAction:
    """A proposed or executed spectrum action."""

    frequency_hz: float
    tx_power_dbm: float

    @property
    def eirp_dbm(self) -> float:
        return self.tx_power_dbm + ANTENNA_GAIN_DBI


@dataclass(frozen=True)
class StepResult:
    """One closed-loop interaction, carrying BOTH actions for credit assignment.

    ``reward`` is the real reward of the **executed** (projected) action — the
    only reward the environment actually realises. ``counterfactual_reward`` is
    what the *proposed* action would have earned had no Shield been present; it
    is an oracle quantity used to measure the safety/utility gap and must never
    be fed to a learner that is meant to be Shield-constrained. It is ``None``
    when the proposal is illegal in a way that has no physical reward (an
    out-of-band frequency has no measured channel).
    """

    proposed: SpectrumAction
    executed: SpectrumAction
    reward: float
    counterfactual_reward: float | None
    projected: bool
    guard_refused: bool
    illegal_without_shield: bool


@dataclass(frozen=True)
class Transition:
    """A replayed transition recovered from the verified evidence chain."""

    decision_id: str
    proposed_frequency_hz: float
    proposed_tx_power_dbm: float
    executed_frequency_hz: float
    executed_tx_power_dbm: float
    reward: float
    projected: bool
    guard_refused: bool

    @property
    def proposed_eirp_dbm(self) -> float:
        return self.proposed_tx_power_dbm + ANTENNA_GAIN_DBI

    @property
    def executed_eirp_dbm(self) -> float:
        return self.executed_tx_power_dbm + ANTENNA_GAIN_DBI


def subband_center_hz(subband: int) -> float:
    """Centre frequency of an in-band subband index (0 .. N_SUBBANDS-1)."""
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    return BAND_LO_HZ + (subband + 0.5) * width


def load_gain_matrix(features: Path) -> np.ndarray:
    """Load the real per-receiver, per-subband channel gains (dB)."""
    gains = []
    for line in features.read_text(encoding="utf-8").splitlines():
        if line.strip():
            gains.append(json.loads(line)["subband_gain_dbw"])
    matrix = np.asarray(gains, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_SUBBANDS:
        raise ValueError(f"expected (n, {N_SUBBANDS}) real gains, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature rows contain non-finite gains")
    return matrix


class ShieldedSpectrumEnv:
    """Real-reward spectrum environment behind the Decision Safety Shield."""

    def __init__(
        self,
        gains: np.ndarray,
        *,
        band_lo_hz: float = BAND_LO_HZ,
        band_hi_hz: float = BAND_HI_HZ,
        max_eirp_dbm: float = MAX_EIRP_DBM,
        bandwidth_hz: float = BANDWIDTH_HZ,
        sinr_threshold_db: float = SINR_THRESHOLD_DB,
    ) -> None:
        self.gains = gains
        self.band_lo_hz = band_lo_hz
        self.band_hi_hz = band_hi_hz
        self.max_eirp_dbm = max_eirp_dbm
        self.bandwidth_hz = bandwidth_hz
        self.sinr_threshold_db = sinr_threshold_db
        self.noise_dbm = (
            -174.0 + 10.0 * np.log10(bandwidth_hz) + NOISE_FIGURE_DB
        )
        self._shield = default_terrestrial_shield(
            band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz, max_eirp_dBm=max_eirp_dbm
        )

    # -- reward -----------------------------------------------------------
    def subband_for(self, frequency_hz: float) -> int | None:
        """In-band subband whose centre is nearest ``frequency_hz``; None if OOB."""
        if frequency_hz < self.band_lo_hz or frequency_hz > self.band_hi_hz:
            return None
        centres = np.array([subband_center_hz(b) for b in range(N_SUBBANDS)])
        return int(np.argmin(np.abs(centres - frequency_hz)))

    def reward(self, action: SpectrumAction) -> float | None:
        """Real served fraction: receivers whose SINR clears the threshold.

        ``SINR_dB = EIRP_dBm + measured_channel_gain_dB - noise_dBm``. Returns
        ``None`` for an out-of-band frequency, which has no measured channel.
        """
        subband = self.subband_for(action.frequency_hz)
        if subband is None:
            return None
        sinr = action.eirp_dbm + self.gains[:, subband] - self.noise_dbm
        return float(np.mean(sinr >= self.sinr_threshold_db))

    def optimal_feasible_reward(self) -> float:
        """Best reward achievable *inside* the feasible set (EIRP at the cap)."""
        best = 0.0
        for b in range(N_SUBBANDS):
            action = SpectrumAction(
                subband_center_hz(b), self.max_eirp_dbm - ANTENNA_GAIN_DBI
            )
            value = self.reward(action)
            if value is not None:
                best = max(best, value)
        return best

    # -- feasibility (the Shield's OWN predicate, promoted for masking) ----
    def action_payload(self, action: SpectrumAction) -> dict[str, Any]:
        """The exact Shield action dict :meth:`step` would build for ``action``.

        Public so that a feasibility mask is computed over *the same payload*
        the closed loop actually disposes — the mask cannot drift from the
        enforcement path because there is only one constructor.
        """
        return {
            "block": "policy_emit",
            "policy_type": "horizon.spectrum.reservation",
            "frequency_hz": action.frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "tx_power_dBm": action.tx_power_dbm,
            "antenna_gain_dBi": ANTENNA_GAIN_DBI,
        }

    def is_feasible(self, action: SpectrumAction) -> bool:
        """``True`` iff the Shield would not have to correct ``action``.

        Delegates to :meth:`horizon_ric.shield.Shield.is_feasible` — the
        analytic constraint the Shield enforces, not a benchmark's private
        restatement of it. Note this is *occupied-bandwidth*-in-band (the
        carrier's ``bandwidth_hz`` must fit inside the licensed channel), which
        is strictly stronger than centre-frequency-in-band: the two edge
        subbands have legal centres but their 20 MHz allocation spills past a
        band edge, so they are **infeasible**.
        """
        return self._shield.is_feasible(self.action_payload(action))

    def infeasibility_reasons(self, action: SpectrumAction) -> list[str]:
        """Invariant ids ``action`` would violate (empty iff feasible)."""
        return [
            check.invariant_id
            for check in self._shield.violations(self.action_payload(action))
        ]

    def feasible_mask(self, actions: Sequence[SpectrumAction]) -> np.ndarray:
        """Boolean mask over ``actions``, ``True`` where the Shield need not correct."""
        return np.asarray([self.is_feasible(a) for a in actions], dtype=bool)

    def feasible_mask_grid(
        self,
        freq_bins_hz: Sequence[float] | np.ndarray,
        eirp_bins_dbm: Sequence[float] | np.ndarray,
        *,
        fast: bool = True,
    ) -> np.ndarray:
        """Feasibility mask over a (frequency x EIRP) product grid.

        Returns shape ``(len(freq_bins_hz), len(eirp_bins_dbm))``. ``.ravel()``
        yields the flat arm order ``arm = f_idx * len(eirp_bins) + e_idx`` used
        by the exploration loop's action space, so
        ``env.feasible_mask_grid(freq_bins, eirp_bins).ravel()[arm]`` is the
        mask for that arm. A 1-D EIRP grid at a fixed centre frequency (the
        credit loop's geometry) is ``feasible_mask_grid([f], eirps)[0]``.

        ``fast=True`` uses a vectorised pure predicate; it is asserted
        equivalent to the full per-action Shield evaluation over the whole grid
        by ``tests/test_shield_feasibility_api.py``. Pass ``fast=False`` to run
        the Shield itself on every cell.
        """
        freqs = np.asarray(freq_bins_hz, dtype=np.float64)
        eirps = np.asarray(eirp_bins_dbm, dtype=np.float64)
        if not fast:
            return np.asarray(
                [
                    [
                        self.is_feasible(SpectrumAction(float(f), float(e) - ANTENNA_GAIN_DBI))
                        for e in eirps
                    ]
                    for f in freqs
                ],
                dtype=bool,
            ).reshape(len(freqs), len(eirps))
        # Vectorised restatement of the three invariants that can bite on a
        # spectrum-reservation payload (numeric sanity, TS 38.104 occupied-
        # bandwidth-in-band, EIRP ceiling). The neural-RX and constellation
        # invariants are n/a for block="policy_emit" with no constellation.
        bw = float(self.bandwidth_hz)
        bw_ok = bool(np.isfinite(bw) and bw > 0.0)
        row_ok = (
            np.isfinite(freqs)
            & (freqs > 0.0)
            & (freqs - bw / 2.0 >= self.band_lo_hz)
            & (freqs + bw / 2.0 <= self.band_hi_hz)
        ) & bw_ok
        # Reproduce the caller's tx = eirp - gain round trip exactly, because
        # MaxEirpInvariant recomputes tx_power_dBm + antenna_gain_dBi.
        eirp_eff = (eirps - ANTENNA_GAIN_DBI) + ANTENNA_GAIN_DBI
        col_ok = np.isfinite(eirp_eff) & (eirp_eff <= self.max_eirp_dbm)
        return row_ok[:, None] & col_ok[None, :]

    # -- closed loop ------------------------------------------------------
    def step(
        self,
        proposed: SpectrumAction,
        *,
        decision_id: str,
        store: JsonlEvidenceStore | None = None,
        rapp_instance_id: str = "shield-feedback",
        policy_label: str = "learner",
    ) -> StepResult:
        """Project ``proposed`` through the Shield, execute it, record it."""
        action_dict: dict[str, Any] = self.action_payload(proposed)
        disposition = self._shield.dispose(action_dict, {}, decision_id=decision_id)
        certificate = disposition.certificate
        guard_fails = run_guard_chain(certificate=certificate, elapsed_ms=1.0)
        safe = disposition.safe_action
        executed = SpectrumAction(
            float(safe["frequency_hz"]), float(safe["tx_power_dBm"])
        )

        realised = self.reward(executed)
        if realised is None:  # pragma: no cover - Shield always returns in-band
            raise RuntimeError("Shield returned an out-of-band action")
        counterfactual = self.reward(proposed)

        projected = (
            abs(executed.frequency_hz - proposed.frequency_hz) > 1.0
            or executed.eirp_dbm < proposed.eirp_dbm - 1e-9
        )
        illegal = (
            proposed.eirp_dbm > self.max_eirp_dbm + 1e-9
            or proposed.frequency_hz < self.band_lo_hz
            or proposed.frequency_hz > self.band_hi_hz
        )

        if store is not None:
            store.append(
                DecisionRecord.new(
                    decision_id=decision_id,
                    rapp_instance_id=rapp_instance_id,
                    state_hash=f"freq-{proposed.frequency_hz:.0f}",
                    chosen_action={
                        "label": policy_label,
                        "policy_type": "horizon.spectrum.reservation",
                        # BOTH actions, so a replayed learner can attribute the
                        # reward to what was executed rather than what was asked.
                        "requested_frequency_hz": proposed.frequency_hz,
                        "requested_tx_power_dBm": proposed.tx_power_dbm,
                        "safe_frequency_hz": executed.frequency_hz,
                        "safe_tx_power_dBm": executed.tx_power_dbm,
                        "reward": realised,
                        "projected": bool(projected),
                        "guard_refused": bool(guard_fails),
                    },
                    predicted_outcome_chosen=PredictedOutcome(
                        sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
                    ),
                    rejected_alternatives=[],
                    model_versions=ModelVersions(
                        encoder="none",
                        risk_heads="shield-feedback",
                        dyna="none",
                        policy=policy_label,
                        constraint_layer="terrestrial-shield",
                        rapp="0.2.0",
                    ),
                )
            )

        return StepResult(
            proposed=proposed,
            executed=executed,
            reward=realised,
            counterfactual_reward=counterfactual,
            projected=bool(projected),
            guard_refused=bool(guard_fails),
            illegal_without_shield=bool(illegal),
        )


def load_transitions(path: Path | str) -> list[Transition]:
    """Replay transitions from a hash-chained evidence store, verify-gated.

    Raises :class:`EvidenceIntegrityError` if the chain does not verify, so a
    tampered decision log can never silently become training data. This is the
    provenance gate every learner in this package inherits.
    """
    store = JsonlEvidenceStore(Path(path))
    broken = store.verify()
    if broken != -1:
        raise EvidenceIntegrityError(
            f"evidence chain failed verification at record index {broken}; "
            "refusing to build training data from a tampered decision log"
        )
    transitions: list[Transition] = []
    for record, _hash in store:
        action = record.chosen_action
        if "requested_frequency_hz" not in action or "reward" not in action:
            continue
        transitions.append(
            Transition(
                decision_id=record.decision_id,
                proposed_frequency_hz=float(action["requested_frequency_hz"]),
                proposed_tx_power_dbm=float(action["requested_tx_power_dBm"]),
                executed_frequency_hz=float(action["safe_frequency_hz"]),
                executed_tx_power_dbm=float(action["safe_tx_power_dBm"]),
                reward=float(action["reward"]),
                projected=bool(action["projected"]),
                guard_refused=bool(action["guard_refused"]),
            )
        )
    return transitions


def summarise(results: Sequence[StepResult] | Iterable[StepResult]) -> dict[str, Any]:
    """Aggregate closed-loop statistics shared by the experiments."""
    items = list(results)
    if not items:
        return {"steps": 0}
    return {
        "steps": len(items),
        "mean_reward": round(float(np.mean([r.reward for r in items])), 6),
        "final_reward": round(float(items[-1].reward), 6),
        "projection_rate": round(float(np.mean([r.projected for r in items])), 6),
        "illegal_proposal_rate": round(
            float(np.mean([r.illegal_without_shield for r in items])), 6
        ),
        "mean_proposed_eirp_dbm": round(
            float(np.mean([r.proposed.eirp_dbm for r in items])), 4
        ),
        "final_proposed_eirp_dbm": round(float(items[-1].proposed.eirp_dbm), 4),
        "mean_executed_eirp_dbm": round(
            float(np.mean([r.executed.eirp_dbm for r in items])), 4
        ),
    }
