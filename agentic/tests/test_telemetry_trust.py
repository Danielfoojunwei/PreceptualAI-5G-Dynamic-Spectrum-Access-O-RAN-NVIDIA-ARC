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


def test_merge_is_not_value_dependent() -> None:
    """A hostile source must not be able to steer the admitted value.

    Merging by mean, newest, max or min would all let one corroborating source
    move the result by choosing what it reports. Selecting by source id is
    boring on purpose: it is the only rule an attacker cannot influence by
    changing its measurement.
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
    assert verdict.state["interference_dBm"] == -95.0  # sensor-a sorts first


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
