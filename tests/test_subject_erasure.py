"""Subject-level certified erasure (GDPR Art. 17) — real tests (no mocks, no torch).

Exercises :mod:`horizon_ric.federated.erasure` on the tabular federated DSA
model: that the transition log is correctly subject-tagged, that erasing a
subject removes exactly that subject's rows and changes the Q-table, that replay
training is deterministic, that FedAvg subject erasure is EXACT (certified L2
distance-to-retrain == 0.0 — the strong, provable Art. 17 claim), and that the
signed certificate verifies against the post-erasure model, rejects the
pre-erasure model and any manifest tamper, and pin-verifies with the embedded
public key. Configs are deliberately small so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.federated import erasure as E
from horizon_ric.security.hsm import InMemoryHSMBackend
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv, n_actions, n_states
from horizon_ric.spectrum.federated_q import QLearnConfig

DSA = DSAConfig()
QC = QLearnConfig(episodes=4)
NS, NA = n_states(DSA.n_channels), n_actions(DSA.n_channels)


def _collect(subjects, *, seed=7):
    env = DSAEnv(cfg=DSA, seed=seed)
    return E.collect_subject_transitions(env, QC, subjects, seed=seed)


def _federation(n_clients=3, *, base_seed=1):
    """Subject-tagged transition log per client (subjects unique per client)."""
    logs = {}
    for ci in range(n_clients):
        cid = f"cell-{ci}"
        subjects = [f"{cid}-sub-{j}" for j in range(3)]
        logs[cid] = _collect(subjects, seed=100 * base_seed + ci + 1)
    return logs


def test_collect_tags_transitions_to_given_subjects():
    subjects = ["alice", "bob", "carol"]
    log = _collect(subjects)
    assert len(log) > 0
    seen = {t.subject_id for t in log}
    # every transition is owned by one of the requested subjects
    assert seen.issubset(set(subjects))
    # round-robin over 4 episodes / 3 subjects touches multiple subjects
    assert len(seen) >= 2
    # transitions are well-formed Transition records
    assert all(isinstance(t, E.Transition) for t in log)
    assert all(0 <= t.state < NS and 0 <= t.action < NA for t in log)


def test_erase_subject_removes_exactly_that_subject_and_changes_q():
    subjects = ["alice", "bob", "carol"]
    log = _collect(subjects)
    target = "alice"
    n_target = sum(1 for t in log if t.subject_id == target)
    assert n_target > 0

    full = E.train_q_from_transitions(log, n_states_=NS, n_actions_=NA, cfg=QC)
    erased = E.erase_subject(log, target, n_states_=NS, n_actions_=NA, cfg=QC)

    # erasing a contributing subject changes the table
    assert not np.array_equal(full, erased)

    # erase_subject is exactly: replay the log with that subject's rows removed
    kept = [t for t in log if t.subject_id != target]
    assert len(kept) == len(log) - n_target
    expected = E.train_q_from_transitions(kept, n_states_=NS, n_actions_=NA, cfg=QC)
    assert np.array_equal(erased, expected)


def test_train_q_from_transitions_is_deterministic():
    log = _collect(["alice", "bob", "carol"])
    q1 = E.train_q_from_transitions(log, n_states_=NS, n_actions_=NA, cfg=QC)
    q2 = E.train_q_from_transitions(log, n_states_=NS, n_actions_=NA, cfg=QC)
    assert np.array_equal(q1, q2)


def test_federated_erase_fedavg_is_exact():
    logs = _federation()
    target = "cell-1"
    subject = sorted({t.subject_id for t in logs[target]})[0]

    hsm = InMemoryHSMBackend()
    global_before, global_after, cert = E.federated_erase_subject(
        client_logs=logs, target_client=target, subject_id=subject,
        dsa_cfg=DSA, q_cfg=QC, method="fedavg", hsm=hsm,
    )
    # the global policy actually changed
    assert not np.array_equal(global_before, global_after)
    # FedAvg (linear) erasure is EXACT — distance to retrain is 0.0
    assert cert.certified_l2_distance_to_retrain == pytest.approx(0.0, abs=1e-9)


def test_certificate_verifies_rejects_and_pins():
    logs = _federation()
    target = "cell-1"
    subject = sorted({t.subject_id for t in logs[target]})[0]

    hsm = InMemoryHSMBackend()
    global_before, global_after, cert = E.federated_erase_subject(
        client_logs=logs, target_client=target, subject_id=subject,
        dsa_cfg=DSA, q_cfg=QC, method="fedavg", hsm=hsm,
    )
    # binds to the post-erasure global
    assert cert.verify(global_after) is True
    # rejects the pre-erasure global (different bytes)
    assert cert.verify(global_before) is False

    # pin-verify against the embedded public key works; a wrong pinned key fails
    assert cert.provenance is not None
    pinned = bytes.fromhex(cert.provenance.public_key_der_hex)
    assert cert.verify(global_after, trusted_public_key_der=pinned) is True
    assert cert.verify(global_after, trusted_public_key_der=b"\x00" * 32) is False

    # a manifest tamper makes verify() False (the signature covers the manifest)
    cert.manifest["subject_id"] = "someone-else"
    assert cert.verify(global_after) is False


def test_manifest_records_erased_transition_count():
    logs = _federation()
    target = "cell-1"
    subject = sorted({t.subject_id for t in logs[target]})[0]
    n_expected = sum(1 for t in logs[target] if t.subject_id == subject)
    assert n_expected > 0

    hsm = InMemoryHSMBackend()
    _, _, cert = E.federated_erase_subject(
        client_logs=logs, target_client=target, subject_id=subject,
        dsa_cfg=DSA, q_cfg=QC, method="fedavg", hsm=hsm,
    )
    assert cert.manifest["n_transitions_erased"] == n_expected
