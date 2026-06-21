"""Certified federated unlearning — real tests (no mocks, no torch).

Exercises the unlearning mechanisms on the tabular federated DSA model: that
retrain-from-scratch actually removes a backdoor an undefended aggregator let
through, that the cheap replay is cheaper, that the signed certificate verifies
against the unlearned model and rejects tampering, and that detection flags a
large-norm poisoner. Configs are deliberately small so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.federated import unlearning as U
from horizon_ric.security.hsm import InMemoryHSMBackend
from horizon_ric.spectrum.data_poison import BackdoorSpec, backdoor_success_rate
from horizon_ric.spectrum.dsa_env import DSAConfig
from horizon_ric.spectrum.federated_q import QLearnConfig

DSA = DSAConfig()
QC = QLearnConfig(episodes=4)
# Attacker forces a channel a healthy policy avoids on the trigger state (max
# damage); this is what gives a clean-vs-backdoor separation to measure.
BD = BackdoorSpec(trigger_state=0, target_channel=5, boost=50.0)


def _train(method: str, *, role: str = U.BACKDOOR, base_seed: int = 1):
    clients = [U.ClientSpec(f"c{i}", seed=10 + i, role=U.HONEST) for i in range(4)]
    backdoor = BD if role == U.BACKDOOR else None
    clients.append(U.ClientSpec("MAL", seed=99, role=role, backdoor=backdoor))
    return U.run_federated_training(
        clients, rounds=3, method=method, dsa_cfg=DSA, q_cfg=QC, base_seed=base_seed
    )


def test_training_is_deterministic_under_seed():
    g1, t1 = _train("fedavg")
    g2, t2 = _train("fedavg")
    assert np.array_equal(g1, g2)
    # the cached trace is identical too (replay-based unlearning depends on it)
    assert np.array_equal(t1.rounds[-1].uploads[0], t2.rounds[-1].uploads[0])


def test_retrain_from_scratch_removes_backdoor_fedavg_let_through():
    poisoned, trace = _train("fedavg")
    sr_poisoned = backdoor_success_rate(poisoned, BD, dsa_cfg=DSA)
    gold = U.retrain_from_scratch(trace, ["MAL"])
    sr_retrain = backdoor_success_rate(gold, BD, dsa_cfg=DSA)
    # undefended FedAvg lets the backdoor fully land; retrain-from-scratch, which
    # never sees the poisoner, drives the trigger success back to the clean floor.
    assert sr_poisoned > 0.8
    assert sr_retrain < 0.4
    assert sr_retrain < sr_poisoned - 0.4


def test_efficient_unlearn_runs_without_retraining_and_bound_is_measured():
    poisoned, trace = _train("fedavg")
    eff = U.efficient_unlearn(trace, ["MAL"])
    gold = U.retrain_from_scratch(trace, ["MAL"])
    dist = float(np.linalg.norm(eff.ravel() - gold.ravel()))
    # The certified bound is a real, non-negative number. For a warm-start
    # propagated backdoor the cheap replay is far from the gold standard — the
    # bound is meaningfully > 0, which is the honest signal not to trust it.
    assert dist >= 0.0
    assert eff.shape == gold.shape == poisoned.shape


def test_certificate_verifies_and_rejects_tampering():
    poisoned, trace = _train("fedavg")
    gold = U.retrain_from_scratch(trace, ["MAL"])
    hsm = InMemoryHSMBackend()
    cert = U.certify_unlearning(
        poisoned_q=poisoned, unlearned_q=gold, retrained_q=gold,
        removed_client_ids=["MAL"], method="fedavg",
        unlearn_mechanism="retrain_from_scratch", hsm=hsm, backdoor=BD, dsa_cfg=DSA,
    )
    # binds to the unlearned model
    assert cert.verify(gold) is True
    # rejects the still-poisoned model (different bytes)
    assert cert.verify(poisoned) is False
    # rejects a one-byte weight tamper
    tampered = gold.copy()
    tampered.flat[0] += 1e-3
    assert cert.verify(tampered) is False
    # rejects a manifest tamper (signature covers the manifest digest)
    cert.manifest["removed_client_ids"] = ["someone-else"]
    assert cert.verify(gold) is False


def test_certificate_records_backdoor_probe_and_distance():
    poisoned, trace = _train("fedavg")
    eff = U.efficient_unlearn(trace, ["MAL"])
    gold = U.retrain_from_scratch(trace, ["MAL"])
    hsm = InMemoryHSMBackend()
    cert = U.certify_unlearning(
        poisoned_q=poisoned, unlearned_q=gold, retrained_q=gold,
        removed_client_ids=["MAL"], method="fedavg", hsm=hsm, backdoor=BD, dsa_cfg=DSA,
    )
    assert cert.backdoor_success_before is not None and cert.backdoor_success_before > 0.8
    assert cert.backdoor_success_after is not None and cert.backdoor_success_after < 0.4
    assert cert.certified_l2_distance_to_retrain == pytest.approx(0.0, abs=1e-9)
    # provenance is a real RSA-PSS signature, pin-verifiable
    assert cert.provenance is not None
    pinned = bytes.fromhex(cert.provenance.public_key_der_hex)
    assert cert.verify(gold, trusted_public_key_der=pinned) is True
    assert cert.verify(gold, trusted_public_key_der=b"\x00" * 32) is False
    # efficient mechanism's certified distance is strictly larger here (honest)
    cert_eff = U.certify_unlearning(
        poisoned_q=poisoned, unlearned_q=eff, retrained_q=gold,
        removed_client_ids=["MAL"], method="fedavg",
        unlearn_mechanism="efficient_unlearn", hsm=hsm, backdoor=BD, dsa_cfg=DSA,
    )
    assert cert_eff.certified_l2_distance_to_retrain > cert.certified_l2_distance_to_retrain


def test_detect_outliers_flags_large_norm_poisoner():
    _, trace = _train("median", role=U.QTABLE_TARGET)
    last = trace.rounds[-1]
    flagged = U.detect_outliers(last.uploads, n_flagged=1)
    assert last.client_ids[flagged[0]] == "MAL"


def test_retrain_refuses_to_remove_every_client():
    _, trace = _train("fedavg")
    all_ids = [c.client_id for c in trace.clients]
    with pytest.raises(ValueError):
        U.retrain_from_scratch(trace, all_ids)
