"""Each trust check must be able to refuse on its own, or it is not a check.

The gate's value is entirely in its ability to say no. A gate that passes
everything is indistinguishable from the unconditional trust it replaced, and
would be worse, because it would look like an answer. So every test here drives
one check to refusal with everything else valid, and the batch-level tests pin
the two properties that are easy to get wrong: refused state must be
unreachable, and a refused batch must not teach the gate anything.
"""

from __future__ import annotations

import pytest
from horizon_agentic.telemetry_trust import (
    FieldBound,
    TelemetryRecord,
    TelemetryTrustError,
    TelemetryTrustGate,
    TrustPolicy,
    canonical_record_bytes,
    sign_record,
)

SECRET_A = b"source-a-secret"
SECRET_B = b"source-b-secret"

BOUNDS = {
    "rsrp_dBm": FieldBound(-140.0, -40.0, "dBm"),
    "prb_utilisation": FieldBound(0.0, 1.0, ""),
    "interference_dBm": FieldBound(-140.0, -30.0, "dBm"),
}


def policy(**overrides) -> TrustPolicy:
    base = dict(
        secrets={"sensor-a": SECRET_A, "sensor-b": SECRET_B},
        max_age_s=5.0,
        bounds=BOUNDS,
        required_fields=frozenset({"rsrp_dBm"}),
        corroborate=frozenset(),
        corroboration_tolerance={},
    )
    base.update(overrides)
    return TrustPolicy(**base)


class FrozenClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def record(source="sensor-a", *, at=1000.0, seq=1, secret=SECRET_A, **fields):
    if not fields:
        fields = {"rsrp_dBm": -90.0}
    return sign_record(
        TelemetryRecord(source_id=source, observed_at=at, sequence=seq, fields=fields),
        secret,
    )


def test_a_well_formed_record_is_admitted() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record()])
    assert verdict.trusted, verdict.refusals
    assert verdict.state["rsrp_dBm"] == -90.0


def test_unknown_source_is_refused() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(source="sensor-rogue", secret=SECRET_A)])
    assert not verdict.trusted
    assert any(c.check_id == "source_authenticated" for c in verdict.refusals)


def test_wrong_key_is_refused() -> None:
    """Authentication must be the tag, not the name.

    A source that merely *claims* a registered id is the whole attack the
    signature exists to stop.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(source="sensor-a", secret=SECRET_B)])
    assert not verdict.trusted
    assert any(c.check_id == "source_authenticated" for c in verdict.refusals)


def test_tampered_field_invalidates_the_tag() -> None:
    signed = record()
    tampered = TelemetryRecord(
        source_id=signed.source_id,
        observed_at=signed.observed_at,
        sequence=signed.sequence,
        fields={"rsrp_dBm": -60.0},
        auth_tag=signed.auth_tag,
    )
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    assert not gate.admit([tampered]).trusted


def test_stale_record_is_refused() -> None:
    gate = TelemetryTrustGate(policy(max_age_s=5.0), clock=FrozenClock(1000.0))
    verdict = gate.admit([record(at=990.0)])
    assert not verdict.trusted
    assert any(c.check_id == "freshness" for c in verdict.refusals)


def test_record_from_the_future_is_refused() -> None:
    """Skew is bounded in both directions.

    A record timestamped ahead of the receiver is as much a signal of a
    misconfigured or hostile source as a stale one, and treating the future as
    automatically fresh is how a replay with a bumped clock gets in.
    """
    gate = TelemetryTrustGate(policy(max_skew_s=1.0), clock=FrozenClock(1000.0))
    verdict = gate.admit([record(at=1030.0)])
    assert not verdict.trusted
    assert any(c.check_id == "freshness" for c in verdict.refusals)


def test_replayed_record_is_refused() -> None:
    clock = FrozenClock()
    gate = TelemetryTrustGate(policy(), clock=clock)
    first = record(seq=1)
    assert gate.admit([first]).trusted
    verdict = gate.admit([first])
    assert not verdict.trusted
    assert any(c.check_id == "replay" for c in verdict.refusals)


def test_stale_sequence_is_refused_even_with_fresh_content() -> None:
    """Sequence, not just byte-identity.

    A source replaying old *content* under a new timestamp produces different
    bytes, so digest matching alone would let it through.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    assert gate.admit([record(seq=5)]).trusted
    verdict = gate.admit([record(seq=3, rsrp_dBm=-88.0)])
    assert not verdict.trusted
    assert any(c.check_id == "replay" for c in verdict.refusals)


def test_a_refused_batch_does_not_advance_the_high_water_mark() -> None:
    """The subtle one.

    If a refused record still raised the sequence high-water mark, an attacker
    could burn a refused record at a high sequence to lock out the genuine
    source — or, worse, a refused replay could raise the mark past the real
    stream. Gate state may only advance on an admitted batch.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    # Refused for range, at a high sequence.
    bad = record(seq=100, rsrp_dBm=999.0)
    assert not gate.admit([bad]).trusted
    # The legitimate stream at a lower sequence must still be admissible.
    assert gate.admit([record(seq=2)]).trusted


def test_out_of_range_field_is_refused() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(rsrp_dBm=+10.0)])
    assert not verdict.trusted
    assert any(c.check_id == "field_range" for c in verdict.refusals)


def test_undeclared_field_is_refused_rather_than_ignored() -> None:
    """An unchecked channel beside the checked ones is the failure mode.

    Silently forwarding a field nobody wrote a bound for would let a source
    introduce arbitrary state into a decision while every declared check
    reported green.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(rsrp_dBm=-90.0, undeclared_thing=1.0)])
    assert not verdict.trusted
    assert any(c.check_id == "field_range" for c in verdict.refusals)


def test_missing_required_field_is_a_refusal_not_a_zero() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(prb_utilisation=0.5)])
    assert not verdict.trusted
    assert any(c.check_id == "required_fields" for c in verdict.refusals)


def test_empty_batch_is_refused() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    assert not gate.admit([]).trusted


def test_critical_field_needs_two_independent_sources() -> None:
    pol = policy(
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 2.0},
        required_fields=frozenset({"interference_dBm"}),
    )
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    single = record(interference_dBm=-95.0)
    verdict = gate.admit([single])
    assert not verdict.trusted
    assert any(c.check_id == "corroboration" for c in verdict.refusals)


def test_corroborating_sources_must_agree_within_tolerance() -> None:
    pol = policy(
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 2.0},
        required_fields=frozenset({"interference_dBm"}),
    )
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    verdict = gate.admit(
        [
            record(source="sensor-a", secret=SECRET_A, interference_dBm=-95.0),
            record(source="sensor-b", secret=SECRET_B, interference_dBm=-70.0),
        ]
    )
    assert not verdict.trusted
    assert any(c.check_id == "corroboration" for c in verdict.refusals)


def test_agreeing_sources_are_admitted() -> None:
    pol = policy(
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 2.0},
        required_fields=frozenset({"interference_dBm"}),
    )
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    verdict = gate.admit(
        [
            record(source="sensor-a", secret=SECRET_A, interference_dBm=-95.0),
            record(source="sensor-b", secret=SECRET_B, interference_dBm=-94.0),
        ]
    )
    assert verdict.trusted, verdict.refusals


def test_corroborated_merge_takes_the_median_not_the_first_source() -> None:
    """Corrected, and worth recording why.

    An earlier version of this module took the reading from the
    lexicographically first source, and this test asserted that was "the only
    rule an attacker cannot influence by changing its measurement". That claim
    was false, and an adversarial review reproduced it: the attacker does not
    change its measurement, it changes its *name*. Registering as
    ``aaa-sensor`` wins every contested field, permanently and for free — a
    registration-time choice, not an ongoing attack.

    The median needs two of three sources to move, so a single hostile source
    is bounded rather than authoritative.
    """
    pol = policy(
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 5.0},
        required_fields=frozenset({"interference_dBm"}),
    )
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    verdict = gate.admit(
        [
            record(source="sensor-b", secret=SECRET_B, interference_dBm=-92.0),
            record(source="sensor-a", secret=SECRET_A, interference_dBm=-95.0),
        ]
    )
    assert verdict.trusted, verdict.refusals
    assert verdict.state["interference_dBm"] == -93.5


def test_a_lexicographically_first_source_no_longer_dictates() -> None:
    """The attack the previous rule permitted, now bounded.

    ``aaa-sensor`` reports a value at the edge of the tolerance window; two
    honest sources agree. Under the old rule the attacker's number was the
    admitted state outright. Under the median it cannot be, and the honest
    reading survives.
    """
    pol = TrustPolicy(
        secrets={"aaa-sensor": b"a", "gnb-1": b"b", "gnb-2": b"c"},
        max_age_s=5.0,
        bounds=BOUNDS,
        required_fields=frozenset({"interference_dBm"}),
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 5.0},
        min_corroborating_sources=3,
    )
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    verdict = gate.admit(
        [
            record(source="aaa-sensor", secret=b"a", interference_dBm=-91.0),
            record(source="gnb-1", secret=b"b", interference_dBm=-95.0),
            record(source="gnb-2", secret=b"c", interference_dBm=-95.5),
        ]
    )
    assert verdict.trusted, verdict.refusals
    assert verdict.state["interference_dBm"] == -95.0
    assert verdict.state["interference_dBm"] != -91.0


def test_admitted_state_is_read_only() -> None:
    """Editing the world model after the gate certified it must not be possible."""
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record()])
    assert verdict.trusted
    with pytest.raises(TypeError):
        verdict.state["rsrp_dBm"] = 999.0  # type: ignore[index]


def test_a_non_ascii_auth_tag_is_refused_not_raised() -> None:
    """`hmac.compare_digest` raises TypeError on non-ASCII str arguments.

    An exception escaping the gate is not a refusal: it propagates out of the
    transaction and defeats the fail-closed discipline the whole module is
    built on — from a party holding no key at all.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    tampered = TelemetryRecord("sensor-a", 1000.0, 1, {"rsrp_dBm": -90.0}, "é" * 64)
    verdict = gate.admit([tampered])
    assert not verdict.trusted
    assert any(c.check_id == "source_authenticated" for c in verdict.refusals)


def test_one_bad_record_does_not_veto_the_whole_batch() -> None:
    """Otherwise anyone who can put a packet on the bus silences the domain.

    Per-record failures belong in the evidence, but the verdict must turn on
    the batch-level checks over the records that authenticated. The earlier
    `all(c.passed for c in checks)` folded in every per-record check, which
    made the `authentic` filter unreachable and handed a denial-of-service to
    an unauthenticated party.
    """
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit(
        [
            record(seq=1),
            record(source="sensor-rogue", secret=SECRET_A, seq=1),
        ]
    )
    assert verdict.trusted, [f"{c.check_id}: {c.detail}" for c in verdict.refusals]
    assert verdict.state["rsrp_dBm"] == -90.0
    assert any(
        c.check_id == "source_authenticated" and not c.passed for c in verdict.checks
    ), "the rejected record must still appear in the evidence"


def test_in_batch_duplicates_are_refused() -> None:
    """Replay state advances after the batch, so two identical records in one
    batch were both admitted."""
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    duplicate = record(seq=7)
    verdict = gate.admit([duplicate, duplicate])
    assert not verdict.trusted
    assert any("within one batch" in c.detail for c in verdict.refusals)


def test_a_huge_sequence_jump_cannot_wedge_a_source() -> None:
    """One record at 2**63 would otherwise silence a source permanently."""
    gate = TelemetryTrustGate(policy(max_sequence_gap=100), clock=FrozenClock())
    assert gate.admit([record(seq=1)]).trusted
    wedge = gate.admit([record(seq=2**63)])
    assert not wedge.trusted
    assert any("jumps" in c.detail for c in wedge.refusals)
    assert gate.admit([record(seq=2)]).trusted, "the genuine source must survive"


def test_a_source_cannot_report_fields_outside_its_remit() -> None:
    """Nothing otherwise binds a source to the fields it is entitled to send."""
    pol = policy(allowed_fields={"sensor-a": frozenset({"rsrp_dBm"})})
    gate = TelemetryTrustGate(pol, clock=FrozenClock())
    verdict = gate.admit([record(rsrp_dBm=-90.0, prb_utilisation=0.5)])
    assert not verdict.trusted
    assert any(c.check_id == "field_authorization" for c in verdict.refusals)


def test_non_finite_values_cannot_be_signed() -> None:
    """`json.dumps` emits bare NaN/Infinity, which is not valid JSON.

    A tag computed over bytes no conforming parser can read is verifiable by
    nobody but us, which defeats the point of authenticating at all.
    """
    with pytest.raises(ValueError):
        canonical_record_bytes(
            TelemetryRecord("sensor-a", 1000.0, 1, {"rsrp_dBm": float("nan")})
        )


def test_int_and_float_timestamps_produce_the_same_bytes() -> None:
    """Same instant, same tag. A JSON round-trip must not invalidate a record."""
    a = TelemetryRecord("sensor-a", 1700000000, 1, {"rsrp_dBm": -90.0})
    b = TelemetryRecord("sensor-a", 1700000000.0, 1, {"rsrp_dBm": -90.0})
    assert canonical_record_bytes(a) == canonical_record_bytes(b)


def test_canonical_bytes_are_domain_separated() -> None:
    """A per-source secret reused for another message with a compatible JSON
    shape would otherwise permit cross-protocol forgery."""
    from horizon_agentic.telemetry_trust import DOMAIN_TAG

    raw = canonical_record_bytes(TelemetryRecord("s", 1.0, 1, {"rsrp_dBm": -90.0}))
    assert raw.startswith(DOMAIN_TAG)


def test_refused_state_cannot_be_read() -> None:
    gate = TelemetryTrustGate(policy(), clock=FrozenClock())
    verdict = gate.admit([record(rsrp_dBm=999.0)])
    assert not verdict.trusted
    with pytest.raises(TelemetryTrustError):
        _ = verdict.state


def test_canonical_bytes_exclude_the_tag() -> None:
    """A signature cannot cover itself, and the bytes must be stable."""
    unsigned = TelemetryRecord("sensor-a", 1000.0, 1, {"rsrp_dBm": -90.0})
    signed = sign_record(unsigned, SECRET_A)
    assert canonical_record_bytes(unsigned) == canonical_record_bytes(signed)
    reordered = TelemetryRecord(
        "sensor-a", 1000.0, 1, {"rsrp_dBm": -90.0, "prb_utilisation": 0.5}
    )
    shuffled = TelemetryRecord(
        "sensor-a", 1000.0, 1, {"prb_utilisation": 0.5, "rsrp_dBm": -90.0}
    )
    assert canonical_record_bytes(reordered) == canonical_record_bytes(shuffled)


def test_policy_rejects_unenforceable_corroboration() -> None:
    """A corroborated field with no tolerance would silently never be checked."""
    with pytest.raises(ValueError):
        TrustPolicy(
            secrets={},
            max_age_s=5.0,
            bounds=BOUNDS,
            corroborate=frozenset({"interference_dBm"}),
            corroboration_tolerance={},
        )


def test_policy_rejects_required_fields_it_cannot_bound() -> None:
    with pytest.raises(ValueError):
        TrustPolicy(
            secrets={},
            max_age_s=5.0,
            bounds={"rsrp_dBm": FieldBound(-140.0, -40.0, "dBm")},
            required_fields=frozenset({"unbounded_field"}),
        )
