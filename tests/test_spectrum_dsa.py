"""Tests for the federated Dynamic Spectrum Access bridge.

Covers: the DSA MDP (determinism, collisions, PU clashes), federated tabular
Q-learning with robust aggregation (the security bridge), the poisoning attack
on the Q-table, and the end-to-end policy → Shield → evidence-chain pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.spectrum import (
    DSAConfig,
    DSAEnv,
    DSAWorld,
    decide_and_record,
    evaluate_policy,
    federated_dsa_round,
    n_actions,
    n_states,
)
from horizon_ric.spectrum.federated_q import QLearnConfig, craft_poison_q, train_local_q


# ── environment ─────────────────────────────────────────────────────────
def test_env_is_deterministic_under_seed():
    cfg = DSAConfig(n_channels=4, n_users=3, max_steps=50)
    e1, e2 = DSAEnv(cfg=cfg, seed=11), DSAEnv(cfg=cfg, seed=11)
    s1, s2 = e1.reset(11), e2.reset(11)
    assert s1 == s2
    for a in [0, 1, 2, 4, 3, 0, 1]:
        o1, r1, d1, i1 = e1.step(a)
        o2, r2, d2, i2 = e2.step(a)
        assert (o1, r1, d1) == (o2, r2, d2)


def test_env_rewards_success_collision_and_pu_clash():
    cfg = DSAConfig(n_channels=4, n_users=3, max_steps=200, sense_error=0.0)
    env = DSAEnv(cfg=cfg, seed=3)
    env.reset(3)
    saw_success = saw_collision_or_clash = False
    for _ in range(200):
        # Always transmit on channel 0 so we provoke both successes and clashes.
        _o, r, done, info = env.step(0)
        if info["success"]:
            saw_success = True
        if info["collision"] or info["pu_clash"]:
            saw_collision_or_clash = True
        if done:
            break
    assert saw_success and saw_collision_or_clash


def test_state_action_space_sizes():
    assert n_states(4) == 16  # 2**4 sensing snapshots
    assert n_actions(4) == 5  # 4 channels + idle


def test_world_resolves_simultaneous_collisions():
    cfg = DSAConfig(n_channels=4, n_users=3, max_steps=10, sense_error=0.0)
    world = DSAWorld(cfg=cfg, seed=2)
    world.reset(2)
    # All three SUs pick channel 0 → at most one can succeed, the rest collide
    # (or all clash if PU busy). Either way ≥ 2 are non-successful.
    _o, rewards, _d, info = world.step(np.array([0, 0, 0]))
    assert info["successes"] <= 1
    assert info["collisions"] + info["pu_clashes"] >= 2


# ── federated Q-learning ────────────────────────────────────────────────
def test_local_q_learns_to_transmit_on_idle_channels():
    cfg = DSAConfig()  # learnable 6-channel / 3-user regime
    env = DSAEnv(cfg=cfg, seed=7)
    q = train_local_q(env, QLearnConfig(episodes=40, epsilon=0.2), seed=7)
    # Evaluate greedily on a fresh single-agent env: a learned policy beats the
    # "always idle" policy (throughput 0).
    eval_env = DSAEnv(cfg=cfg, seed=4242)
    s = eval_env.reset(4242)
    succ = tot = 0
    for _ in range(2000):
        a = int(np.argmax(q[s]))
        s, _r, done, info = eval_env.step(a)
        succ += info["success"]
        tot += 1
        if done:
            s = eval_env.reset(eval_env.seed + 1)
    assert succ / tot > 0.3  # genuinely learns to use idle spectrum


def test_federated_round_shapes_and_krum_selection():
    cfg = DSAConfig()
    res = federated_dsa_round(
        n_clients=8, n_malicious=2, dsa_cfg=cfg,
        q_cfg=QLearnConfig(episodes=8), method="krum", seed=1,
    )
    assert res.global_q.shape == (n_states(cfg.n_channels), n_actions(cfg.n_channels))
    # Krum must select an honest client (the malicious ones are the last 2).
    assert res.selected_index is not None
    assert res.selected_index < res.n_clients - res.n_malicious


def test_poisoning_collapses_fedavg_but_robust_aggregation_recovers():
    cfg = DSAConfig()
    q_cfg = QLearnConfig(episodes=30, epsilon=0.2)

    clean = federated_dsa_round(n_clients=10, n_malicious=0, dsa_cfg=cfg, q_cfg=q_cfg,
                                method="fedavg", seed=1)
    poisoned = federated_dsa_round(n_clients=10, n_malicious=2, dsa_cfg=cfg, q_cfg=q_cfg,
                                   method="fedavg", seed=1)
    defended = federated_dsa_round(n_clients=10, n_malicious=2, dsa_cfg=cfg, q_cfg=q_cfg,
                                   method="median", seed=1)

    thr_clean = evaluate_policy(clean.global_q, dsa_cfg=cfg, n_episodes=12, seed=999)["throughput_per_slot"]
    thr_poison = evaluate_policy(poisoned.global_q, dsa_cfg=cfg, n_episodes=12, seed=999)["throughput_per_slot"]
    thr_defend = evaluate_policy(defended.global_q, dsa_cfg=cfg, n_episodes=12, seed=999)["throughput_per_slot"]

    # Poisoning destroys throughput under plain FedAvg.
    assert thr_poison < 0.2 * thr_clean + 1e-9
    # A robust aggregator (median) recovers most of the throughput.
    assert thr_defend > 0.4 * thr_clean


def test_secure_aggregation_path_runs():
    cfg = DSAConfig(n_channels=4, n_users=3)
    res = federated_dsa_round(
        n_clients=6, n_malicious=0, dsa_cfg=cfg, q_cfg=QLearnConfig(episodes=4),
        method="fedavg", secure_agg=True, seed=2,
    )
    assert res.secure_agg is True
    assert res.global_q.shape == (n_states(cfg.n_channels), n_actions(cfg.n_channels))
    assert np.all(np.isfinite(res.global_q))


# ── end-to-end pipeline ─────────────────────────────────────────────────
def test_pipeline_emits_only_legal_actions_and_chains_evidence():
    cfg = DSAConfig()
    res = federated_dsa_round(n_clients=8, n_malicious=0, dsa_cfg=cfg,
                              q_cfg=QLearnConfig(episodes=8), method="krum", seed=1)
    poison_q = craft_poison_q(res.global_q.shape, target_channel=0, magnitude=50.0)

    tmp = Path(tempfile.mkdtemp()) / "ev.jsonl"
    store = JsonlEvidenceStore(tmp)

    illegal = 0
    for i in range(24):
        obs = i % n_states(cfg.n_channels)
        q = res.global_q if i % 2 == 0 else poison_q
        # Request an over-EIRP power to force the Shield to clamp.
        out = decide_and_record(
            q, obs, method="krum", evidence=store, decision_id=f"dsa-{i}",
            rng_seed=i, requested_tx_power_dBm=42.0,
        )
        a = out.safe_action
        if a.get("emit", True):
            lo = a["frequency_hz"] - a["bandwidth_hz"] / 2
            hi = a["frequency_hz"] + a["bandwidth_hz"] / 2
            eirp = a["tx_power_dBm"] + a["antenna_gain_dBi"]
            if lo < 3.40e9 - 1e-3 or hi > 3.50e9 + 1e-3 or eirp > 33.0 + 1e-6:
                illegal += 1
        # The SafetyCertificate must be embedded in the audit record.
        assert "safety_certificate" in out.record.chosen_action

    # Regardless of poisoning, the Shield kept every emitted action legal.
    assert illegal == 0
    # The hash chain is intact.
    assert store.verify() == -1
    assert len(store) == 24


def test_pipeline_clamps_over_eirp():
    cfg = DSAConfig(n_channels=4, n_users=3)
    res = federated_dsa_round(n_clients=6, n_malicious=0, dsa_cfg=cfg,
                              q_cfg=QLearnConfig(episodes=4), method="median", seed=1)
    out = decide_and_record(res.global_q, 0b1010, method="median",
                            requested_tx_power_dBm=50.0)
    a = out.safe_action
    if a.get("emit", True):
        assert a["tx_power_dBm"] + a["antenna_gain_dBi"] <= 33.0 + 1e-6


# ── realistic poisoning attacks (ALIE / Fang) ───────────────────────────
def test_alie_probit_is_accurate():
    from horizon_ric.spectrum.attacks import _phi_inv

    assert abs(_phi_inv(0.975) - 1.959964) < 1e-4
    assert abs(_phi_inv(0.5)) < 1e-6
    assert abs(_phi_inv(0.84134) - 1.0) < 1e-3


def test_alie_attack_hides_inside_variance_envelope():
    from horizon_ric.federated import robust
    from horizon_ric.spectrum.attacks import alie_attack

    rng = np.random.default_rng(0)
    benign = [rng.normal(0.0, 1.0, size=200) for _ in range(12)]
    honest_mean = np.mean(benign, axis=0)
    mal = alie_attack(benign, 8)  # 40% Byzantine — ALIE bites here
    agg = robust.coordinate_median(benign + mal)
    bias = float(np.linalg.norm(agg - honest_mean))
    # ALIE leaks a NONZERO but bounded bias through the median (it is not a
    # silver-bullet defeat, and it is not zero — the honest middle ground).
    assert bias > 0.5


def test_fang_krum_attack_can_be_selected_by_krum():
    from horizon_ric.federated import robust
    from horizon_ric.spectrum.attacks import fang_attack_krum

    rng = np.random.default_rng(1)
    benign = [rng.normal(0.0, 1.0, size=120) for _ in range(16)]
    n_byz = 4  # 20% — within Krum breakdown (n=20 > 2*4+2=10)
    mal = fang_attack_krum(benign, n_byz)
    res = robust.krum(benign + mal, f=n_byz)
    # The Fang attack is tuned so Krum may select a malicious point. We assert
    # the attack at least biases Krum's pick relative to the honest mean.
    honest_mean = np.mean(benign, axis=0)
    assert float(np.linalg.norm(res.aggregate - honest_mean)) > 1.0


def test_fang_median_attack_drags_median():
    from horizon_ric.federated import robust
    from horizon_ric.spectrum.attacks import fang_attack_median

    rng = np.random.default_rng(2)
    benign = [rng.normal(0.0, 1.0, size=150) for _ in range(14)]
    honest_mean = np.mean(benign, axis=0)
    clean = float(np.linalg.norm(robust.coordinate_median(benign) - honest_mean))
    attacked = float(
        np.linalg.norm(robust.coordinate_median(benign + fang_attack_median(benign, 6)) - honest_mean)
    )
    assert attacked > clean  # the directed-deviation attack moves the median
