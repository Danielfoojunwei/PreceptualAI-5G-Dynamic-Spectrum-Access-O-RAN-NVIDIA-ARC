"""Tests for federated DSA DATA-poisoning, backdoor, and free-rider attacks.

These complement ``tests/test_spectrum_dsa.py`` (which covers the crafted-update
``qtable_target`` poison) by exercising the *data-side* attacks in
:mod:`horizon_ric.spectrum.data_poison`:

  * reward/label poisoning degrades the UN-defended global policy (it learns to
    seek PU clashes), and robust aggregation REDUCES — not necessarily eliminates
    — that effect;
  * a single-row backdoor raises trigger-state misbehaviour WITHOUT robust
    aggregation, while staying stealthy on the main task;
  * a free-rider does no work yet stays finite/benign;
  * CRUCIALLY: regardless of poisoning or a live backdoor, the Decision Safety
    Shield emits ZERO illegal (out-of-band / over-EIRP) actions and the evidence
    hash-chain verifies intact.

Sizes are kept small/fast.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.security.tenant import TenantScope
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.spectrum.data_poison import (
    BackdoorSpec,
    backdoor_success_rate,
    default_backdoor,
    federated_dsa_round_data_poison,
    free_rider_update,
    stamp_backdoor,
    train_reward_poisoned_q,
)
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv, n_states
from horizon_ric.spectrum.federated_q import (
    QLearnConfig,
    evaluate_policy,
    federated_dsa_round,
    train_local_q,
)
from horizon_ric.spectrum.pipeline import DSADecisionConfig, decide_and_record

BAND_LO, BAND_HI, MAX_EIRP = 3.40e9, 3.50e9, 33.0


def _cfg() -> DSAConfig:
    return DSAConfig(n_channels=4, n_users=3, max_steps=60)


# ── 1. reward / label poisoning ─────────────────────────────────────────────
def test_reward_poisoning_inverts_local_policy_objective():
    """A reward-poisoned client learns to CLASH with the primary user."""
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=30, epsilon=0.2)
    honest = train_local_q(DSAEnv(cfg=cfg, seed=5), q_cfg, seed=5)
    poisoned = train_reward_poisoned_q(DSAEnv(cfg=cfg, seed=5), q_cfg, seed=5)

    honest_clash = evaluate_policy(honest, dsa_cfg=cfg, n_episodes=8, seed=999)["pu_clash_per_slot"]
    poison_clash = evaluate_policy(poisoned, dsa_cfg=cfg, n_episodes=8, seed=999)["pu_clash_per_slot"]
    # The inverted objective rewards clashing with the PU → far more PU clashes.
    assert poison_clash > honest_clash


def test_reward_poisoning_degrades_undefended_and_robust_agg_reduces_it():
    """Reward poisoning raises the aggregate PU-clash rate under FedAvg; a robust
    aggregator (median) reduces — not necessarily eliminates — that rise."""
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=20, epsilon=0.2)
    n, nm = 12, 4

    clean = federated_dsa_round(
        n_clients=n, n_malicious=0, dsa_cfg=cfg, q_cfg=q_cfg, method="fedavg", seed=1
    )
    fa = federated_dsa_round_data_poison(
        attack="reward_poison", n_clients=n, n_malicious=nm, dsa_cfg=cfg,
        q_cfg=q_cfg, method="fedavg", seed=1,
    )
    md = federated_dsa_round_data_poison(
        attack="reward_poison", n_clients=n, n_malicious=nm, dsa_cfg=cfg,
        q_cfg=q_cfg, method="median", seed=1,
    )

    clean_clash = evaluate_policy(clean.global_q, dsa_cfg=cfg, n_episodes=10, seed=999)["pu_clash_per_slot"]
    fa_clash = evaluate_policy(fa.global_q, dsa_cfg=cfg, n_episodes=10, seed=999)["pu_clash_per_slot"]
    md_clash = evaluate_policy(md.global_q, dsa_cfg=cfg, n_episodes=10, seed=999)["pu_clash_per_slot"]

    # Un-defended FedAvg lets the poison through: PU-clash rises over clean.
    assert fa_clash > clean_clash
    # Robust aggregation REDUCES the attack effect (closer to clean than FedAvg).
    assert md_clash < fa_clash


# ── 2. backdoor / trigger attack ────────────────────────────────────────────
def test_stamp_backdoor_only_touches_the_trigger_row():
    cfg = _cfg()
    q = train_local_q(DSAEnv(cfg=cfg, seed=3), QLearnConfig(episodes=10), seed=3)
    spec = BackdoorSpec(trigger_state=2, target_channel=0, boost=50.0)
    bd = stamp_backdoor(q, spec)
    # Only the trigger row changed; the greedy action there is the target.
    assert int(np.argmax(bd[2])) == 0
    others = [s for s in range(q.shape[0]) if s != 2]
    assert np.allclose(bd[others], q[others])


def test_backdoor_raises_trigger_misbehaviour_without_robust_agg():
    """Without robust aggregation the backdoor fires on the trigger state while
    a robust aggregator (median) with the malicious clients in the minority
    screens it back toward the clean rate."""
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=20, epsilon=0.2)
    n, nm = 12, 4
    spec = default_backdoor(cfg.n_channels, trigger_state=0)

    clean = federated_dsa_round(
        n_clients=n, n_malicious=0, dsa_cfg=cfg, q_cfg=q_cfg, method="fedavg", seed=1
    )
    fa = federated_dsa_round_data_poison(
        attack="backdoor", n_clients=n, n_malicious=nm, dsa_cfg=cfg,
        q_cfg=q_cfg, method="fedavg", backdoor=spec, seed=1,
    )
    md = federated_dsa_round_data_poison(
        attack="backdoor", n_clients=n, n_malicious=nm, dsa_cfg=cfg,
        q_cfg=q_cfg, method="median", backdoor=spec, seed=1,
    )

    sr_clean = backdoor_success_rate(clean.global_q, spec, dsa_cfg=cfg)
    sr_fa = backdoor_success_rate(fa.global_q, spec, dsa_cfg=cfg)
    sr_md = backdoor_success_rate(md.global_q, spec, dsa_cfg=cfg)

    # Backdoor succeeds under un-defended FedAvg.
    assert sr_fa > 0.8
    assert sr_fa > sr_clean + 0.5
    # A robust aggregator with the attacker in the minority reduces it.
    assert sr_md < sr_fa


def test_backdoor_survives_robust_agg_at_breakdown_point():
    """HONEST residual: at the coordinate-median ~50% breakdown point the
    malicious clients control the trigger coordinate and the backdoor survives
    robust aggregation. Robust aggregation BOUNDS, it does not ELIMINATE."""
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=20, epsilon=0.2)
    n, nm = 12, 6  # 50% Byzantine — exactly the median breakdown point
    spec = default_backdoor(cfg.n_channels, trigger_state=0)
    md = federated_dsa_round_data_poison(
        attack="backdoor", n_clients=n, n_malicious=nm, dsa_cfg=cfg,
        q_cfg=q_cfg, method="median", backdoor=spec, seed=1,
    )
    assert backdoor_success_rate(md.global_q, spec, dsa_cfg=cfg) > 0.8


# ── 3. free-rider ────────────────────────────────────────────────────────────
def test_free_rider_returns_stale_model_plus_noise():
    cfg = _cfg()
    shape = (n_states(cfg.n_channels), cfg.n_channels + 1)
    g = np.full(shape, 0.5)
    fr = free_rider_update(g, shape, sigma=1e-3, seed=7)
    assert fr.shape == shape
    assert np.all(np.isfinite(fr))
    # It is the stale model plus a tiny perturbation, not real learning.
    assert np.linalg.norm(fr - g) < 0.1
    # First-round (no global model yet) → noise around zero.
    fr0 = free_rider_update(None, shape, sigma=1e-3, seed=7)
    assert np.linalg.norm(fr0) < 0.5


def test_free_rider_round_is_finite_and_benign():
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=10, epsilon=0.2)
    res = federated_dsa_round_data_poison(
        attack="free_rider", n_clients=10, n_malicious=3, dsa_cfg=cfg,
        q_cfg=q_cfg, method="median", seed=1,
    )
    assert np.all(np.isfinite(res.global_q))


# ── 4. CRUCIAL: Shield legality + audit integrity under every data attack ────
@pytest.mark.parametrize("attack", ["reward_poison", "backdoor", "free_rider"])
def test_shield_emits_zero_illegal_actions_under_data_poisoning(attack):
    """Regardless of the data attack, the Shield emits 0 illegal (out-of-band /
    over-EIRP) actions and the evidence hash-chain verifies intact."""
    cfg = _cfg()
    q_cfg = QLearnConfig(episodes=15, epsilon=0.2)
    spec = default_backdoor(cfg.n_channels, trigger_state=0)
    # Un-defended FedAvg so the attack is maximally LIVE in the global policy.
    res = federated_dsa_round_data_poison(
        attack=attack, n_clients=12, n_malicious=4, dsa_cfg=cfg,
        q_cfg=q_cfg, method="fedavg", backdoor=spec, seed=1,
    )

    dcfg = DSADecisionConfig(band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP)
    tmp = Path(tempfile.mkdtemp()) / "ev.jsonl"
    store = JsonlEvidenceStore(tmp)

    ns = n_states(cfg.n_channels)
    illegal = 0
    with TenantScope("dsa-poison-test"):
        for i in range(ns):
            out = decide_and_record(
                res.global_q, i, cfg=dcfg, method="fedavg", evidence=store,
                decision_id=f"{attack}-{i}", rng_seed=i,
                requested_tx_power_dBm=42.0,  # over-EIRP → Shield must clamp
            )
            a = out.safe_action
            # The certificate is embedded in the audit record.
            assert "safety_certificate" in out.record.chosen_action
            if a.get("emit", True):
                lo = a["frequency_hz"] - a["bandwidth_hz"] / 2.0
                hi = a["frequency_hz"] + a["bandwidth_hz"] / 2.0
                eirp = a["tx_power_dBm"] + a["antenna_gain_dBi"]
                if lo < BAND_LO - 1e-3 or hi > BAND_HI + 1e-3 or eirp > MAX_EIRP + 1e-6:
                    illegal += 1
        chain = store.verify()
        n_rec = len(store)

    assert illegal == 0          # Shield bounds EVERY emitted action to legal RF
    assert chain == -1           # audit hash-chain intact
    assert n_rec == ns


def test_shield_blocks_or_clips_explicit_out_of_band_backdoor():
    """The worst a backdoor could attempt — an explicitly out-of-band, over-EIRP
    carrier straight at the Shield — is blocked or projected back to legal."""
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP
    )
    proposed = {
        "block": "fed_dsa_policy",
        "emit": True,
        "frequency_hz": BAND_HI + 40e6,  # out of band
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 48.0,            # over EIRP ceiling
        "antenna_gain_dBi": 6.0,
    }
    disp = shield.dispose(proposed, decision_id="oob")
    sa = disp.safe_action
    lo = sa["frequency_hz"] - sa["bandwidth_hz"] / 2.0
    hi = sa["frequency_hz"] + sa["bandwidth_hz"] / 2.0
    eirp = sa["tx_power_dBm"] + sa["antenna_gain_dBi"]
    legal = lo >= BAND_LO - 1e-3 and hi <= BAND_HI + 1e-3 and eirp <= MAX_EIRP + 1e-6
    assert disp.certificate.emit_blocked or legal
