"""A trust gate for telemetry, so enforcement stops assuming its inputs are true.

Horizon-RIC's projection argument is about *outputs*: whatever a planner
proposes, only the projected action reaches the radio, so an out-of-licence
emission is unrepresentable. That argument is untouched by a lying sensor — and
that is exactly its limit. Projection guarantees the emitted action satisfies
the invariants *as evaluated against the state it was given*. Feed it false
state and it will compute a different, still-"safe", still-wrong action, and
sign a certificate saying so.

That was tolerable while one planner consumed telemetry the same operator
produced. It stops being tolerable when many agents share one cross-domain data
layer, because a single stale, replayed or mis-attributed stream then steers
several closed loops at once, and every one of them produces a valid signature
over a decision made from fiction.

This module does not prove telemetry is truthful — nothing can, from inside the
receiver. It removes the *unconditional* trust assumption and replaces it with
a stated, checkable one: state is admitted only when it arrives from a known
source, recently, without replay, inside physically declared bounds, and — for
fields nominated as critical — corroborated by a second independent source.
Anything missing is a refusal, not a default.

Six checks, each of which can independently refuse:

``source_authenticated``
    The ``source_id`` is registered and the record's HMAC verifies over its
    canonical bytes. Compared with :func:`hmac.compare_digest`.
``required_fields``
    Every field the policy declares mandatory is present. A missing field is a
    refusal, never a zero.
``freshness``
    Observed within ``max_age_s``, and not implausibly far in the future
    (clock skew is bounded, not assumed away).
``replay``
    Per-source sequence numbers strictly increase, and a canonical digest seen
    before is refused even if its sequence looks fresh.
``field_range``
    Declared fields lie inside their physical bounds. An *undeclared* field is
    also a refusal: silently forwarding state nobody wrote a bound for is how
    an unchecked channel opens.
``corroboration``
    Fields nominated as critical must agree, within tolerance, across at least
    two independently authenticated sources.

The gate is stateful by necessity — replay detection has no meaning without
memory — so one instance belongs to one trust domain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import statistics
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "TelemetryRecord",
    "FieldBound",
    "TrustPolicy",
    "TrustCheck",
    "TrustVerdict",
    "TelemetryTrustGate",
    "canonical_record_bytes",
    "sign_record",
    "TelemetryTrustError",
]


DOMAIN_TAG = b"horizon-telemetry-v1\x00"


class TelemetryTrustError(RuntimeError):
    """Raised when refused state is read as though it had been admitted."""


@dataclass(frozen=True)
class TelemetryRecord:
    """One measurement report from one source.

    ``sequence`` is per-source and must strictly increase; it is what makes
    replay detectable without a synchronised clock. ``auth_tag`` is a hex
    HMAC-SHA256 over :func:`canonical_record_bytes`, which deliberately
    excludes the tag itself.
    """

    source_id: str
    observed_at: float
    sequence: int
    fields: Mapping[str, float]
    auth_tag: str = ""


def canonical_record_bytes(record: TelemetryRecord) -> bytes:
    """Deterministic bytes a signature is computed over.

    Sorted keys and separators without whitespace, matching the discipline in
    ``horizon_ric.shield.signing.canonical_certificate_bytes``: two peers that
    serialise the same record must produce identical bytes, or the tag is
    unverifiable for reasons unrelated to authenticity. ``auth_tag`` is
    excluded — a signature cannot cover itself.
    """
    payload = {
        "source_id": str(record.source_id),
        # Coerced, both of them. `1700000000` and `1700000000.0` are the same
        # instant and produced different bytes, so any JSON round-trip over the
        # wire invalidated an otherwise valid tag.
        "observed_at": float(record.observed_at),
        "sequence": int(record.sequence),
        "fields": {str(k): float(v) for k, v in sorted(record.fields.items())},
    }
    # allow_nan=False: json.dumps emits bare `NaN`/`Infinity`, which is not
    # valid JSON. A tag over bytes no conforming parser can read is not
    # verifiable by anyone but us.
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    # Domain separation: a per-source secret reused for another message with a
    # compatible JSON shape would otherwise permit cross-protocol forgery.
    return DOMAIN_TAG + body


def sign_record(record: TelemetryRecord, secret: bytes) -> TelemetryRecord:
    """Return ``record`` with a valid ``auth_tag`` under ``secret``."""
    tag = hmac.new(secret, canonical_record_bytes(record), hashlib.sha256).hexdigest()
    return TelemetryRecord(
        source_id=record.source_id,
        observed_at=record.observed_at,
        sequence=record.sequence,
        fields=dict(record.fields),
        auth_tag=tag,
    )


@dataclass(frozen=True)
class FieldBound:
    """Physical bounds for one telemetry field, with its unit for the record."""

    lo: float
    hi: float
    unit: str

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclass(frozen=True)
class TrustPolicy:
    """What this trust domain requires before state may influence an action.

    ``secrets`` maps a source id to its shared HMAC key. A source absent here
    is unknown, and unknown is refused — there is no anonymous-but-plausible
    tier.

    ``max_age_s`` and ``max_skew_s`` bound staleness in both directions. A
    record from the future is as suspicious as one from last week; a small
    positive skew is allowed because real clocks disagree.

    ``bounds`` must cover every field a record carries. Fields outside it are
    refused rather than passed through, which is what stops an unchecked
    channel opening beside the checked ones.

    ``corroborate`` names the fields that a single source may not establish
    alone; ``corroboration_tolerance`` is the absolute agreement window, per
    field, between independent sources.
    """

    secrets: Mapping[str, bytes]
    max_age_s: float
    bounds: Mapping[str, FieldBound]
    required_fields: frozenset[str] = frozenset()
    corroborate: frozenset[str] = frozenset()
    corroboration_tolerance: Mapping[str, float] = field(default_factory=dict)
    max_skew_s: float = 1.0
    max_sequence_gap: int = 10_000
    min_corroborating_sources: int = 2
    # Which fields each source may report. A source absent from this mapping
    # may report anything bounded, which is the permissive default; naming a
    # source pins it. Without this, any registered source can inject any
    # bounded field and — if it sorts first — override the honest reading.
    allowed_fields: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(self.corroborate) - set(self.corroboration_tolerance)
        if missing:
            raise ValueError(
                "corroborated fields without a tolerance are unenforceable: "
                f"{sorted(missing)}"
            )
        unbounded = (set(self.required_fields) | set(self.corroborate)) - set(self.bounds)
        if unbounded:
            raise ValueError(
                "policy requires fields it declares no bound for: " f"{sorted(unbounded)}"
            )


@dataclass(frozen=True)
class TrustCheck:
    """One named check and what it decided, kept for the evidence record."""

    check_id: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TrustVerdict:
    """The gate's decision, and the state it admits — if any.

    ``state`` is populated only when ``trusted`` is true. Reading it otherwise
    raises: the whole point is that there is no path to partially-trusted
    state, so a caller cannot proceed on the half of the telemetry that
    happened to pass.
    """

    trusted: bool
    checks: tuple[TrustCheck, ...]
    _state: Mapping[str, float] | None = None

    @property
    def refusals(self) -> tuple[TrustCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def state(self) -> Mapping[str, float]:
        if not self.trusted or self._state is None:
            raise TelemetryTrustError(
                "refused telemetry has no admissible state; refusals: "
                + "; ".join(f"{c.check_id}: {c.detail}" for c in self.refusals)
            )
        return self._state


class TelemetryTrustGate:
    """Stateful admission control for telemetry.

    One instance per trust domain. It remembers the highest sequence and the
    set of canonical digests seen per source, which is what replay detection
    requires; sharing an instance across domains would let one domain's history
    silence another's replay.
    """

    def __init__(
        self,
        policy: TrustPolicy,
        *,
        clock: Callable[[], float],
        initial_high_water: Mapping[str, int] | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock
        # Seeding from a persisted snapshot is what closes the restart window.
        # Without it, a restart resets every high-water mark to None and the
        # first record from each source sets the floor with no lower bound —
        # so everything inside `max_age_s` replays cleanly. Persisting the
        # marks is cheap (one integer per source) and strictly more useful
        # than persisting digests, which the sequence rule already subsumes.
        self._high_water: dict[str, int] = dict(initial_high_water or {})

    def high_water_marks(self) -> Mapping[str, int]:
        """A snapshot to persist, so replay defence survives a restart.

        Callers should write this after an admitted batch and pass it back as
        ``initial_high_water`` on the next start. Doing so is the deployment's
        responsibility: this class is in-memory by design, and pretending to
        own durability it does not have would be worse than saying so.
        """
        return dict(self._high_water)

    # ── per-record checks ────────────────────────────────────────────────
    def _authenticate(self, record: TelemetryRecord) -> TrustCheck:
        secret = self._policy.secrets.get(record.source_id)
        if secret is None:
            return TrustCheck(
                "source_authenticated",
                False,
                f"source {record.source_id!r} is not registered in this trust domain",
            )
        if not record.auth_tag:
            return TrustCheck(
                "source_authenticated",
                False,
                f"source {record.source_id!r} supplied no auth tag",
            )
        expected = hmac.new(
            secret, canonical_record_bytes(record), hashlib.sha256
        ).hexdigest()
        # Compare bytes, not str. `hmac.compare_digest` raises TypeError on
        # str arguments containing non-ASCII, so a tag of "e"*64 with an
        # accent escaped as an exception rather than a refusal — from a party
        # holding no key. An exception out of the gate defeats the whole
        # fail-closed discipline; bytes comparison never raises.
        if not hmac.compare_digest(
            expected.encode("ascii"), record.auth_tag.encode("utf-8", "replace")
        ):
            return TrustCheck(
                "source_authenticated",
                False,
                f"auth tag for source {record.source_id!r} does not verify",
            )
        return TrustCheck(
            "source_authenticated", True, f"source {record.source_id!r} verified"
        )

    def _freshness(self, record: TelemetryRecord) -> TrustCheck:
        now = self._clock()
        age = now - record.observed_at
        if age > self._policy.max_age_s:
            return TrustCheck(
                "freshness",
                False,
                f"record from {record.source_id!r} is {age:.3f}s old, "
                f"limit {self._policy.max_age_s:.3f}s",
            )
        if age < -self._policy.max_skew_s:
            return TrustCheck(
                "freshness",
                False,
                f"record from {record.source_id!r} is {-age:.3f}s in the future, "
                f"skew allowance {self._policy.max_skew_s:.3f}s",
            )
        return TrustCheck("freshness", True, f"age {age:.3f}s within limit")

    def _replay(self, record: TelemetryRecord) -> TrustCheck:
        # No digest set. An earlier version kept every canonical digest per
        # source to catch byte-identical replays, which the strictly-increasing
        # sequence rule already refuses — a replay carries the same sequence.
        # It was measured at ~155 bytes per entry and never freed: a
        # memory-exhaustion path reachable by a *legitimate* high-rate source,
        # for no additional refusal power.
        high = self._high_water.get(record.source_id)
        if high is not None and record.sequence <= high:
            return TrustCheck(
                "replay",
                False,
                f"sequence {record.sequence} from {record.source_id!r} does not "
                f"exceed high-water mark {high}",
            )
        if high is not None and record.sequence - high > self._policy.max_sequence_gap:
            # Without this bound, a single record at 2**63 raises the mark past
            # anything the genuine source will ever emit and silences it
            # permanently. A forward jump is as suspicious as a backward one.
            return TrustCheck(
                "replay",
                False,
                f"sequence {record.sequence} from {record.source_id!r} jumps "
                f"{record.sequence - high} past the high-water mark "
                f"{high}; limit {self._policy.max_sequence_gap}",
            )
        return TrustCheck(
            "replay", True, f"sequence {record.sequence} from {record.source_id!r} is new"
        )

    def _ranges(self, record: TelemetryRecord) -> TrustCheck:
        problems: list[str] = []
        if not record.fields:
            # "Trusted nothing" is indistinguishable from trusted state to a
            # caller that only reads `verdict.trusted`.
            return TrustCheck(
                "field_range", False, f"record from {record.source_id!r} carries no fields"
            )
        for name, value in sorted(record.fields.items()):
            bound = self._policy.bounds.get(name)
            if bound is None:
                problems.append(f"{name}: no declared bound in this policy")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                problems.append(f"{name}: not a number")
                continue
            if numeric != numeric or numeric in (float("inf"), float("-inf")):
                problems.append(f"{name}: non-finite")
                continue
            if not bound.contains(numeric):
                problems.append(
                    f"{name}={numeric:g}{bound.unit} outside "
                    f"[{bound.lo:g}, {bound.hi:g}]{bound.unit}"
                )
        if problems:
            return TrustCheck("field_range", False, "; ".join(problems))
        return TrustCheck(
            "field_range", True, f"{len(record.fields)} field(s) within declared bounds"
        )

    def _fields_permitted(self, record: TelemetryRecord) -> TrustCheck:
        allowed = self._policy.allowed_fields.get(record.source_id)
        if allowed is None:
            return TrustCheck(
                "field_authorization", True, "no per-source field restriction"
            )
        outside = sorted(set(record.fields) - set(allowed))
        if outside:
            return TrustCheck(
                "field_authorization",
                False,
                f"source {record.source_id!r} reported fields outside its remit: "
                f"{outside}",
            )
        return TrustCheck(
            "field_authorization", True, f"{len(record.fields)} field(s) within remit"
        )

    # ── admission ────────────────────────────────────────────────────────
    def admit(self, records: Sequence[TelemetryRecord]) -> TrustVerdict:
        """Admit the state carried by ``records``, or refuse the whole batch.

        Refusal is all-or-nothing on purpose. A caller handed "the fields that
        passed" would build an action from a partial world model and have no
        way to know which part was missing, which is a worse failure than not
        acting.

        Gate state (sequence high-water marks, seen digests) advances only on
        an admitted batch. A refused batch must not teach the gate anything,
        or a refused replay would raise the high-water mark and let the next
        one through.
        """
        checks: list[TrustCheck] = []

        if not records:
            checks.append(
                TrustCheck("required_fields", False, "no telemetry records supplied")
            )
            return TrustVerdict(False, tuple(checks))

        # Batch-level, not per-record. A duplicate is not "one bad packet from
        # a stranger" — which must not veto the batch — but the same source
        # sending the same sequence twice, which means either a replay or a
        # confused sensor. Neither is a state you want to build an action from,
        # so it refuses the batch.
        keys = [(r.source_id, int(r.sequence)) for r in records]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})

        authentic: list[TelemetryRecord] = []
        for record in records:
            per_record = [
                self._authenticate(record),
                self._freshness(record),
                self._replay(record),
                self._ranges(record),
                self._fields_permitted(record),
            ]
            checks.extend(per_record)
            if all(c.passed for c in per_record):
                authentic.append(record)

        merged: dict[str, float] = {}
        contributors: dict[str, list[tuple[str, float]]] = {}
        for record in authentic:
            for name, value in record.fields.items():
                contributors.setdefault(name, []).append((record.source_id, float(value)))

        # Merge rule, corrected. An earlier version took the reading from the
        # lexicographically first source and claimed that was "the only rule an
        # attacker cannot influence by changing its measurement". That was
        # wrong in a way worth recording: the attacker does not change its
        # measurement, it changes its *name*. Registering as `aaa-sensor` wins
        # every contested field, forever, for free.
        #
        # For corroborated fields the median is taken instead: with three
        # sources an attacker must control two to move it, where the previous
        # rule needed only a well-chosen id. For uncorroborated fields the
        # lowest-sorting source still decides, and that is exactly why a field
        # that matters belongs in `corroborate`.
        for name, seen in contributors.items():
            values = [value for _, value in sorted(seen)]
            if name in self._policy.corroborate and len(values) >= 2:
                merged[name] = statistics.median(values)
            else:
                merged[name] = values[0]

        batch_checks = [
            TrustCheck(
                "required_fields",
                not (missing := sorted(set(self._policy.required_fields) - set(merged))),
                f"missing required field(s): {missing}" if missing else "all present",
            ),
            self._corroborate(contributors),
            TrustCheck(
                "batch_duplicates",
                not duplicates,
                f"duplicate (source, sequence) within one batch: {duplicates}"
                if duplicates
                else "no duplicates in batch",
            ),
        ]
        checks.extend(batch_checks)

        # The verdict turns on the BATCH checks, evaluated over the records
        # that authenticated — not on every per-record check. Folding those in
        # made the `authentic` filter unreachable and handed a denial of
        # service to anyone able to put one bad packet on the bus. Per-record
        # failures stay in `checks` as evidence, which is where they belong.
        trusted = bool(authentic) and all(c.passed for c in batch_checks)
        if trusted:
            for record in authentic:
                self._high_water[record.source_id] = max(
                    self._high_water.get(record.source_id, record.sequence),
                    record.sequence,
                )
        # Read-only: a caller mutating admitted state would be editing the
        # world model after the gate certified it.
        return TrustVerdict(
            trusted, tuple(checks), MappingProxyType(dict(merged)) if trusted else None
        )

    def _corroborate(
        self, contributors: Mapping[str, Iterable[tuple[str, float]]]
    ) -> TrustCheck:
        problems: list[str] = []
        for name in sorted(self._policy.corroborate):
            seen = sorted(set(contributors.get(name, ())))
            sources = {source for source, _ in seen}
            need = self._policy.min_corroborating_sources
            if len(sources) < need:
                problems.append(
                    f"{name}: needs {need} independent sources, saw {len(sources)}"
                )
                continue
            values = [value for _, value in seen]
            spread = max(values) - min(values)
            tolerance = self._policy.corroboration_tolerance[name]
            if spread > tolerance:
                problems.append(
                    f"{name}: sources disagree by {spread:g}, tolerance {tolerance:g}"
                )
        if problems:
            return TrustCheck("corroboration", False, "; ".join(problems))
        return TrustCheck(
            "corroboration",
            True,
            f"{len(self._policy.corroborate)} critical field(s) corroborated",
        )
