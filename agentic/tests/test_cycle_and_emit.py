"""The serialisation point, and the emission binding that has to go with it.

The first test in this file is the one that justifies `cycle.py` existing. An
adversarial review reproduced it: with agents submitting separately, each
transaction committed and the protected slice was cut from 50 MHz to 12.5 MHz
anyway — the exact harm the package opens with, straight through the component
built to prevent it. Everything the aggregate layer does was conditional on
batching that nothing caused.
"""

from __future__ import annotations

import pytest
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor
from horizon_agentic.bundle import SafetyTransaction, TransactionRefused
from horizon_agentic.cycle import TransactionCycle
from horizon_agentic.emit import emittable, verify_binding
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)

from horizon_ric.shield import default_terrestrial_shield

ROOT = "operator-root"
CELL = "cell-3450-A"
FLOOR_HZ = 20e6
NOW = 1_700_000_000.0

BASELINE = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}


def policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice"}), 0),
            "energy-agent": AuthorityGrant(
                "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}), 20
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), 15
            ),
        },
    )


def txn(**kw) -> SafetyTransaction:
    return SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
        ),
        policy=policy(),
        aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)],
        clock=lambda: "2026-01-01T00:00:00Z",
        **kw,
    )


def energy(bandwidth_hz=50e6, nonce="e1"):
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": bandwidth_hz},
        mutates=frozenset({"bandwidth_hz"}),
        resource_id=CELL,
        nonce=nonce,
        issued_at=NOW,
    )


def slice_(share=0.25, nonce="s1"):
    return AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": share, "embb": 1.0 - share},
        },
        mutates=frozenset({"prb_allocation"}),
        resource_id=CELL,
        nonce=nonce,
        issued_at=NOW,
    )


def cycle(**kw) -> TransactionCycle:
    return TransactionCycle(
        txn(**kw),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )


# ── the gap this module closes ───────────────────────────────────────────
def test_separate_submissions_both_commit_without_a_cycle() -> None:
    """Documented, not fixed by wishing: this is why `cycle.py` exists.

    Each agent evaluated alone commits, because alone each is impeccable. The
    net effect on the cell is 50 MHz x 0.25 = 12.5 MHz for a slice with a
    20 MHz commitment — the harm the aggregate layer exists to prevent,
    delivered through it.
    """
    alone_energy = txn().evaluate([energy()], context={"baseline_action": dict(BASELINE)})
    alone_slice = txn().evaluate([slice_()], context={"baseline_action": dict(BASELINE)})
    assert alone_energy.committed and alone_slice.committed

    net_bandwidth = alone_energy.members[0].action["bandwidth_hz"]
    net_share = alone_slice.members[0].action["prb_allocation"]["safety_critical"]
    assert net_bandwidth * net_share == pytest.approx(12.5e6)


def test_the_cycle_decides_them_together() -> None:
    """Same two agents, submitted independently, now decided as one epoch."""
    c = cycle()
    assert c.submit(energy()).accepted
    assert c.submit(slice_()).accepted
    result = c.close()
    assert result.committed, result.refusals
    # The conflict is seen, and the culpable request is not applied.
    assert {d.agent_id for d in result.resolution.dropped} == {"energy-agent"}
    emitted = {m.agent_id: m.action for m in result.members}
    assert set(emitted) == {"slice-agent"}
    assert emitted["slice-agent"]["bandwidth_hz"] == 100e6


def test_submission_returns_a_receipt_not_a_decision() -> None:
    """An agent decided on submission would have been decided in isolation."""
    c = cycle()
    receipt = c.submit(energy())
    assert receipt.accepted and receipt.epoch == 0
    assert not hasattr(receipt, "committed")
    assert c.pending == 1


def test_the_baseline_is_read_from_the_operator_not_the_caller() -> None:
    """The anti-smuggling check is only meaningful if the frame is not agent-supplied."""
    seen: list[dict] = []

    def source() -> dict:
        seen.append({})
        return dict(BASELINE)

    c = TransactionCycle(
        txn(), clock=lambda: NOW, baseline_source=source, window_s=0.0
    )
    c.submit(slice_())
    # A caller-supplied baseline must not override the operator's.
    result = c.close(context={"baseline_action": {"bandwidth_hz": 1.0}})
    assert seen, "the baseline source was not consulted"
    assert result.committed, result.refusals
    assert result.certificate.baseline_digest


def test_one_request_per_principal_per_epoch() -> None:
    c = cycle()
    assert c.submit(slice_(nonce="a")).accepted
    second = c.submit(slice_(nonce="b"))
    assert not second.accepted
    assert "already has a request" in second.reason


def test_a_full_epoch_refuses_further_submissions() -> None:
    c = TransactionCycle(
        txn(), clock=lambda: NOW, baseline_source=lambda: dict(BASELINE),
        window_s=0.0, max_members=1,
    )
    assert c.submit(slice_()).accepted
    full = c.submit(energy())
    assert not full.accepted and "full" in full.reason


def test_closing_rotates_the_epoch() -> None:
    c = cycle()
    c.submit(slice_())
    assert c.epoch == 0
    c.close()
    assert c.epoch == 1 and c.pending == 0


def test_tick_does_nothing_while_the_window_is_open() -> None:
    now = [NOW]
    c = TransactionCycle(
        txn(), clock=lambda: now[0], baseline_source=lambda: dict(BASELINE),
        window_s=5.0,
    )
    c.submit(slice_())
    assert c.tick() is None
    now[0] = NOW + 6.0
    assert c.tick() is not None


def test_tick_does_not_certify_empty_epochs() -> None:
    """Otherwise every tick appends a certificate saying nothing happened."""
    now = [NOW]
    c = TransactionCycle(
        txn(), clock=lambda: now[0], baseline_source=lambda: dict(BASELINE),
        window_s=1.0,
    )
    now[0] = NOW + 100.0
    assert c.tick() is None
    assert c.history == ()


def test_decision_is_independent_of_submission_order() -> None:
    """Two agents racing must produce the same epoch decision either way."""
    outcomes = set()
    for order in ([energy(), slice_()], [slice_(), energy()]):
        c = cycle()
        for env in order:
            c.submit(env)
        result = c.close()
        outcomes.add(
            (
                tuple(sorted(m.agent_id for m in result.members)),
                tuple(sorted(d.agent_id for d in result.resolution.dropped)),
            )
        )
    assert len(outcomes) == 1, outcomes


# ── emission binding ─────────────────────────────────────────────────────
def test_a_refused_transaction_yields_nothing_emittable() -> None:
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    result = txn().evaluate(
        [
            AgentActionEnvelope(
                agent_id="slice-agent",
                agent_version="1.0.0",
                target_domain="ran",
                granted_scopes=frozenset({"slice"}),
                delegation_chain=(ROOT, "slice-agent"),
                requested_action={
                    **starved,
                    "prb_allocation": {"safety_critical": 0.25, "embb": 0.75},
                },
                mutates=frozenset({"prb_allocation"}),
                resource_id=CELL,
                nonce="s",
                issued_at=NOW,
            )
        ],
        context={"baseline_action": starved},
    )
    assert not result.committed
    with pytest.raises(TransactionRefused):
        emittable(result)


def test_refused_members_carry_valid_per_action_certificates() -> None:
    """The hazard, stated as a test.

    Every member of a refused bundle has a clean per-action certificate — that
    is the premise of the design, not a defect. A downstream gate checking only
    that certificate would admit the action and be right by its own rules,
    which is why the transaction binding has to exist before any emission path
    does.
    """
    c = cycle()
    c.submit(energy())
    c.submit(slice_())
    result = c.close()
    assert result.committed
    dropped_ids = {d.agent_id for d in result.resolution.dropped}
    assert dropped_ids == {"energy-agent"}
    # The dropped agent's action was individually safe throughout.
    alone = txn().evaluate([energy()], context={"baseline_action": dict(BASELINE)})
    assert alone.committed and alone.members[0].certificate.safe
    # And it is not in what may be emitted.
    assert {e.agent_id for e in emittable(result)} == {"slice-agent"}


def test_the_binding_names_the_transaction_and_verifies() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    c = cycle(signing_key=key)
    c.submit(slice_())
    result = c.close()
    actions = emittable(result)
    assert len(actions) == 1
    binding = actions[0].binding
    assert binding.transaction_id == "epoch-0"
    assert binding.epoch == 0
    assert verify_binding(binding, result.certificate, public_key=key.public_key())


def test_a_binding_against_a_refused_certificate_does_not_verify() -> None:
    """The check that gives the binding its meaning."""
    import dataclasses

    c = cycle()
    c.submit(slice_())
    result = c.close()
    binding = emittable(result)[0].binding
    refused = dataclasses.replace(result.certificate, committed=False)
    assert not verify_binding(binding, refused)


def test_an_altered_certificate_breaks_the_binding() -> None:
    import dataclasses

    c = cycle()
    c.submit(slice_())
    result = c.close()
    binding = emittable(result)[0].binding
    altered = dataclasses.replace(result.certificate, transaction_id="epoch-99")
    assert not verify_binding(binding, altered)


def test_the_certificate_records_what_it_was_decided_against() -> None:
    """Replayability: a record that cannot reconstruct the decision is not evidence."""
    c = cycle()
    c.submit(slice_())
    result = c.close()
    cert = result.certificate
    assert cert.schema_version
    assert cert.baseline_digest and len(cert.baseline_digest) == 64
    assert cert.epoch == 0
    ids = {a["id"] for a in cert.aggregates}
    assert ids == {"absolute_slice_capacity_floor"}
    # Thresholds, not just names: the same invariant with a different floor
    # decides differently.
    assert cert.aggregates[0]["min_capacity_hz"] == FLOOR_HZ
