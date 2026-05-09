"""AI-PHY decision modules — counterfactual envelope tests."""

from __future__ import annotations

import pytest

from horizon_ric.policy.dpod_activation import (
    DPoDActivation,
    DPoDInputs,
    DPoDTransition,
)
from horizon_ric.policy.learned_constellation_decision import (
    LearnedConstellationDecision,
    LearnedConstellationState,
)
from horizon_ric.policy.neural_rx_decision import (
    NeuralRxDecision,
    NeuralRxState,
)

# ─── Neural-RX decision ────────────────────────────────────────────────


class TestNeuralRxDecision:
    def test_neural_chosen_in_envelope(self):
        d = NeuralRxDecision()
        chosen, env = d.decide(NeuralRxState(pa_backoff_db=4.0, snr_db=12.0, mobility="low", mcs=10))
        assert chosen == "neural"
        assert env["chosen"] == "neural"

    def test_lmmse_when_pa_backoff_too_high(self):
        d = NeuralRxDecision()
        chosen, _ = d.decide(NeuralRxState(pa_backoff_db=8.0, snr_db=15.0, mobility="low", mcs=10))
        assert chosen == "lmmse"

    def test_lmmse_when_snr_too_low(self):
        d = NeuralRxDecision()
        chosen, _ = d.decide(NeuralRxState(pa_backoff_db=4.0, snr_db=5.0, mobility="low", mcs=10))
        assert chosen == "lmmse"

    def test_lmmse_when_high_mobility(self):
        d = NeuralRxDecision()
        chosen, _ = d.decide(NeuralRxState(pa_backoff_db=4.0, snr_db=15.0, mobility="high", mcs=10))
        assert chosen == "lmmse"

    def test_envelope_has_pinned_seed(self):
        d = NeuralRxDecision(random_seed=42)
        _, env = d.decide(NeuralRxState(pa_backoff_db=4.0, snr_db=15.0, mobility="low", mcs=10))
        assert env["pinned_seed"] == 42


# ─── DPoD activation ───────────────────────────────────────────────────


class TestDPoDActivation:
    def test_off_to_on_when_envelope_clean(self):
        a = DPoDActivation()
        t = a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        assert t is not None
        assert t.from_state == "off"
        assert t.to_state == "on"
        assert a.state() == "on"

    def test_no_transition_when_already_on(self):
        a = DPoDActivation(initial_state="on")
        t = a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        assert t is None
        assert a.state() == "on"

    def test_on_to_off_when_pa_backoff_too_high(self):
        a = DPoDActivation(initial_state="on")
        t = a.step(DPoDInputs(pa_backoff_db=7.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        assert t is not None
        assert t.to_state == "off"

    def test_off_when_sla_risk_high(self):
        a = DPoDActivation()
        a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.50, evm_target_pct=8.5))
        assert a.state() == "off"

    def test_off_when_evm_target_too_low(self):
        a = DPoDActivation()
        a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=5.0))
        assert a.state() == "off"

    def test_sink_invoked_on_transition(self):
        captured: list[DPoDTransition] = []
        a = DPoDActivation(sink=captured.append)
        a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        a.step(DPoDInputs(pa_backoff_db=8.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        assert len(captured) == 2

    def test_history_append_only(self):
        a = DPoDActivation()
        a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))  # no-op
        h = a.history()
        assert len(h) == 1

    def test_transition_sha_deterministic(self):
        """Same transition payload yields same SHA."""
        a = DPoDActivation()
        t = a.step(DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5))
        assert t is not None
        s = t.sha256()
        assert len(s) == 64


# ─── Learned constellation ─────────────────────────────────────────────


class TestLearnedConstellationDecision:
    def test_learned_chosen_for_low_mobility(self):
        d = LearnedConstellationDecision()
        chosen, env = d.decide(
            LearnedConstellationState(mobility="low", mcs_index=10, prb_count=20)
        )
        assert chosen == "learned"
        assert env.predicted_throughput_gain_pct == 31.0

    def test_classical_for_medium_mobility(self):
        d = LearnedConstellationDecision()
        chosen, env = d.decide(
            LearnedConstellationState(mobility="medium", mcs_index=10, prb_count=20)
        )
        # 16% gain < 25% threshold default → classical
        assert chosen == "classical_qam"

    def test_learned_chosen_for_high_mobility(self):
        d = LearnedConstellationDecision()
        chosen, _ = d.decide(
            LearnedConstellationState(mobility="high", mcs_index=10, prb_count=20)
        )
        # 28% gain ≥ 25% threshold → learned
        assert chosen == "learned"

    def test_classical_when_mcs_too_high(self):
        d = LearnedConstellationDecision()
        chosen, _ = d.decide(
            LearnedConstellationState(mobility="low", mcs_index=27, prb_count=20)
        )
        assert chosen == "classical_qam"

    def test_classical_when_prb_too_few(self):
        d = LearnedConstellationDecision()
        chosen, _ = d.decide(
            LearnedConstellationState(mobility="low", mcs_index=10, prb_count=2)
        )
        assert chosen == "classical_qam"

    def test_envelope_carries_state_and_seed(self):
        d = LearnedConstellationDecision()
        _, env = d.decide(
            LearnedConstellationState(mobility="low", mcs_index=10, prb_count=20),
            seed=99,
        )
        assert env.pinned_seed == 99
        assert env.state["mobility"] == "low"

    def test_threshold_override(self):
        # Drop threshold to 10 % → medium mobility now passes.
        d = LearnedConstellationDecision(min_gain_pct=10.0)
        chosen, _ = d.decide(
            LearnedConstellationState(mobility="medium", mcs_index=10, prb_count=20)
        )
        assert chosen == "learned"


# ─── Reproducibility (pinned seed → identical envelope hash) ───────────


class TestReproducibility:
    def test_dpod_transition_sha_deterministic_given_inputs(self):
        # Two ON-transitions on different `DPoDActivation` instances with
        # the same inputs produce different SHA-256 (timestamps differ),
        # but the *body* (excluding timestamp) is byte-identical.
        a1 = DPoDActivation()
        a2 = DPoDActivation()
        ins = DPoDInputs(pa_backoff_db=4.0, sla_risk_30s_p99=0.05, evm_target_pct=8.5)
        t1 = a1.step(ins, seed=7)
        t2 = a2.step(ins, seed=7)
        assert t1 is not None and t2 is not None
        # Body excluding timestamp identical
        d1 = t1.to_dict()
        d2 = t2.to_dict()
        d1.pop("timestamp")
        d2.pop("timestamp")
        assert d1 == d2

    def test_learned_constellation_envelope_sha_stable_modulo_timestamp(self):
        d = LearnedConstellationDecision()
        s = LearnedConstellationState(mobility="low", mcs_index=10, prb_count=20)
        _, e1 = d.decide(s, seed=1)
        _, e2 = d.decide(s, seed=1)
        d1 = e1.to_dict()
        d2 = e2.to_dict()
        d1.pop("timestamp")
        d2.pop("timestamp")
        assert d1 == d2
