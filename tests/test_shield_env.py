"""Shared closed-loop substrate, exercised on REAL DeepMIMO channels.

These tests use the real, licence-gated feature build. That file is not
redistributed in the repo, so the module is skipped when it is absent (the same
convention the Sionna PHY link-level test uses) and is run for real by the
``realdata`` workflow, which rebuilds the DeepMIMO features deterministically
before invoking pytest on this file. No synthetic channels are constructed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.learning import (
    EvidenceIntegrityError,
    ShieldedSpectrumEnv,
    SpectrumAction,
    load_transitions,
)
from horizon_ric.learning.shield_env import (
    ANTENNA_GAIN_DBI,
    BAND_HI_HZ,
    BAND_LO_HZ,
    MAX_EIRP_DBM,
    load_gain_matrix,
    subband_center_hz,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"

pytestmark = pytest.mark.skipif(
    not FEATURES.is_file(),
    reason="licence-gated real DeepMIMO features absent; built by the realdata workflow",
)


@pytest.fixture(scope="module")
def env() -> ShieldedSpectrumEnv:
    return ShieldedSpectrumEnv(load_gain_matrix(FEATURES))


def test_real_gains_load(env: ShieldedSpectrumEnv) -> None:
    assert env.gains.shape[1] == 6
    assert env.gains.shape[0] >= 512
    assert np.all(np.isfinite(env.gains))


def test_reward_is_monotone_in_eirp(env: ShieldedSpectrumEnv) -> None:
    """Real served fraction rises with EIRP, so the optimum is above the cap."""
    freq = subband_center_hz(3)
    rewards = [
        env.reward(SpectrumAction(freq, eirp - ANTENNA_GAIN_DBI))
        for eirp in (20.0, 26.0, 33.0, 40.0, 46.0)
    ]
    assert all(r is not None for r in rewards)
    values = [r for r in rewards if r is not None]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_shield_binds_on_this_task(env: ShieldedSpectrumEnv) -> None:
    """The unconstrained optimum is strictly better than the best legal action.

    This is what makes the task a real test of the projection operator: unlike
    channel selection or coverage regression, the constraint is not free here.
    """
    freq = subband_center_hz(3)
    unconstrained = env.reward(SpectrumAction(freq, 46.0 - ANTENNA_GAIN_DBI))
    assert unconstrained is not None
    assert unconstrained > env.optimal_feasible_reward()


def test_over_eirp_proposal_is_projected_to_the_cap(
    env: ShieldedSpectrumEnv, tmp_path: Path
) -> None:
    store = JsonlEvidenceStore(tmp_path / "chain.jsonl")
    result = env.step(
        SpectrumAction(subband_center_hz(3), 44.0), decision_id="over", store=store
    )
    assert result.proposed.eirp_dbm > MAX_EIRP_DBM
    assert result.executed.eirp_dbm <= MAX_EIRP_DBM + 1e-9
    assert result.projected is True
    assert result.illegal_without_shield is True
    # The realised reward is the capped one; the proposal's true value is higher.
    assert result.counterfactual_reward is not None
    assert result.counterfactual_reward > result.reward


def test_out_of_band_proposal_is_projected_into_band(
    env: ShieldedSpectrumEnv, tmp_path: Path
) -> None:
    store = JsonlEvidenceStore(tmp_path / "chain.jsonl")
    result = env.step(
        SpectrumAction(3.62e9, 26.0), decision_id="oob", store=store
    )
    assert BAND_LO_HZ <= result.executed.frequency_hz <= BAND_HI_HZ
    assert result.projected is True
    assert result.illegal_without_shield is True


def test_legal_proposal_passes_through_untouched(
    env: ShieldedSpectrumEnv, tmp_path: Path
) -> None:
    """When the proposal is already feasible the Shield costs nothing."""
    store = JsonlEvidenceStore(tmp_path / "chain.jsonl")
    proposed = SpectrumAction(subband_center_hz(3), 26.0)
    result = env.step(proposed, decision_id="legal", store=store)
    assert result.projected is False
    assert result.illegal_without_shield is False
    assert result.guard_refused is False
    assert result.executed.eirp_dbm == pytest.approx(proposed.eirp_dbm)
    assert result.counterfactual_reward == pytest.approx(result.reward)


def test_transitions_round_trip_through_the_chain(
    env: ShieldedSpectrumEnv, tmp_path: Path
) -> None:
    path = tmp_path / "chain.jsonl"
    store = JsonlEvidenceStore(path)
    for i, tx in enumerate((26.0, 44.0, 30.0)):
        env.step(SpectrumAction(subband_center_hz(2), tx), decision_id=f"d{i}", store=store)
    assert store.verify() == -1

    transitions = load_transitions(path)
    assert len(transitions) == 3
    # Both actions survive the round trip, which is what lets a replayed learner
    # attribute reward to what was executed rather than what was proposed.
    assert transitions[1].proposed_eirp_dbm == pytest.approx(50.0)
    assert transitions[1].executed_eirp_dbm <= MAX_EIRP_DBM + 1e-9
    assert transitions[1].projected is True


def test_replay_refuses_a_tampered_chain(
    env: ShieldedSpectrumEnv, tmp_path: Path
) -> None:
    """The provenance gate: a tampered decision log cannot become training data."""
    path = tmp_path / "chain.jsonl"
    store = JsonlEvidenceStore(path)
    for i, tx in enumerate((26.0, 44.0, 30.0)):
        env.step(SpectrumAction(subband_center_hz(2), tx), decision_id=f"d{i}", store=store)

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"reward":', '"reward": 0.999, "_t":', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError):
        load_transitions(path)
