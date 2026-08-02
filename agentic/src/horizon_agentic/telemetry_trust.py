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
from dataclasses import dataclass, field
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
        "source_id": record.source_id,
        "observed_at": record.observed_at,
        "sequence": record.sequence,
        "fields": {str(k): float(v) for k, v in sorted(record.fields.items())},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


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

    def __init__(self, policy: TrustPolicy, *, clock: Callable[[], float]) -> None:
        self._policy = policy
        self._clock = clock
        self._high_water: dict[str, int] = {}
        self._seen: dict[str, set[str]] = {}

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
        if not hmac.compare_digest(expected, record.auth_tag):
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
        digest = hashlib.sha256(canonical_record_bytes(record)).hexdigest()
        if digest in self._seen.get(record.source_id, set()):
            return TrustCheck(
                "replay",
                False,
                f"record from {record.source_id!r} is a byte-identical replay",
            )
        high = self._high_water.get(record.source_id)
        if high is not None and record.sequence <= high:
            return TrustCheck(
                "replay",
                False,
                f"sequence {record.sequence} from {record.source_id!r} does not "
                f"exceed high-water mark {high}",
            )
        return TrustCheck(
            "replay", True, f"sequence {record.sequence} from {record.source_id!r} is new"
        )

    def _ranges(self, record: TelemetryRecord) -> TrustCheck:
        problems: list[str] = []
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

        authentic: list[TelemetryRecord] = []
        for record in records:
            per_record = [
                self._authenticate(record),
                self._freshness(record),
                self._replay(record),
                self._ranges(record),
            ]
            checks.extend(per_record)
            if all(c.passed for c in per_record):
                authentic.append(record)

        merged: dict[str, float] = {}
        contributors: dict[str, list[tuple[str, float]]] = {}
        for record in authentic:
            for name, value in record.fields.items():
                contributors.setdefault(name, []).append((record.source_id, float(value)))

        # Deterministic merge: for a field seen by several sources, take the
        # reading from the lexicographically first source. Anything
        # value-dependent (mean, median, newest) would let a hostile source
        # move the admitted value by choosing what it reports.
        for name, seen in contributors.items():
            merged[name] = sorted(seen)[0][1]

        missing = sorted(set(self._policy.required_fields) - set(merged))
        checks.append(
            TrustCheck(
                "required_fields",
                not missing,
                f"missing required field(s): {missing}" if missing else "all present",
            )
        )

        checks.append(self._corroborate(contributors))

        trusted = all(c.passed for c in checks)
        if trusted:
            for record in authentic:
                self._high_water[record.source_id] = max(
                    self._high_water.get(record.source_id, record.sequence), record.sequence
                )
                self._seen.setdefault(record.source_id, set()).add(
                    hashlib.sha256(canonical_record_bytes(record)).hexdigest()
                )
        return TrustVerdict(trusted, tuple(checks), merged if trusted else None)

    def _corroborate(
        self, contributors: Mapping[str, Iterable[tuple[str, float]]]
    ) -> TrustCheck:
        problems: list[str] = []
        for name in sorted(self._policy.corroborate):
            seen = sorted(set(contributors.get(name, ())))
            sources = {source for source, _ in seen}
            if len(sources) < 2:
                problems.append(
                    f"{name}: needs 2 independent sources, saw {len(sources)}"
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
