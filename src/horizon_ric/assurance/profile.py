"""Machine-readable *assurance profile* emitted from a live Shield.

**New in WP1.** A :class:`~horizon_ric.shield.shield.Shield` is configured in
code — band edges, EIRP ceiling, PAPR ceiling, neural-RX tolerance, protected
slice floor, PFD mask. Those numbers are the whole regulatory claim, and until
now they existed only as constructor arguments at some call site. An auditor
asking "what was this cell's Shield actually enforcing on 2026-07-30?" had no
artefact to read.

This module answers that by reflecting a *running* Shield instance into a plain
JSON-serialisable dict:

    profile = emit_profile(shield, profile_id="cell-42/band-n78")
    digest  = profile_digest(profile)     # pin it; compare it after a redeploy

The profile is derived from the live object, not hand-maintained, so it cannot
drift from what is being enforced. Its shape is published as
``docs/schemas/invariant-profile-v1.schema.json``.

:func:`profile_digest` hashes the profile through the **same** canonical JSON
form the certificate signer uses
(:func:`horizon_ric.shield.signing.canonical_certificate_bytes`: ``sort_keys``,
tight separators). That is deliberate: a deployment can pin its Shield
configuration by digest in exactly the units its signed certificates are already
denominated in, and a profile digest stamped into
``SafetyCertificate.model_provenance`` means the certificate carries a
cryptographic reference to the envelope that produced it.

Stdlib only, like the rest of the Shield.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from horizon_ric.assurance.planner import (
    OPTIONAL_ACTION_KEYS,
    OUTPUT_SENTINEL_KEYS,
    REQUIRED_ACTION_KEYS,
)
from horizon_ric.shield.invariants import Invariant
from horizon_ric.shield.shield import Shield, ShieldConfig

#: Version of the profile *document shape* (not of any operator's profile).
#: Bump the minor for an additive field, the major for a breaking one, and
#: update ``docs/schemas/invariant-profile-v1.schema.json`` in the same commit —
#: ``tests/test_assurance_schemas.py`` fails otherwise.
PROFILE_SCHEMA_VERSION = "1.0.0"

# Dataclass fields whose value is an injected collaborator, not an
# operator-programmable limit. ``LawfulInterceptInvariant.li`` holds an
# ``LIConstraint`` carrying the operator's LIMF rule catalogue: it is opaque by
# design (the rApp sees warrant *tags*, never warrant content), it is not a
# dataclass, and ``dataclasses.asdict`` would deep-copy it into something
# unserialisable. We never touch the value — we record only that enforcement is
# delegated. That is why this module uses ``fields`` + ``getattr`` rather than
# ``dataclasses.asdict`` on the invariant as a whole.
_DELEGATED_FIELDS: dict[str, dict[str, Any]] = {
    "li": {"delegated_to": "LIConstraint"},
}

_JSON_SCALARS: tuple[type, ...] = (bool, int, float, str)


def _limit_value(value: Any) -> Any:
    """Render one dataclass field value as JSON, or mark it opaque.

    Sequences (``ConstellationLegalityInvariant.allowed_orders`` is a tuple)
    become lists; mappings are walked; anything else is reported as opaque
    rather than being coerced, so a profile can never claim to describe a limit
    it did not actually read.
    """
    if value is None or type(value) in _JSON_SCALARS:
        return value
    if isinstance(value, dict):
        return {str(k): _limit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_limit_value(v) for v in value]
    return {"opaque": type(value).__name__}


def invariant_descriptor(inv: Invariant) -> dict[str, Any]:
    """Describe one invariant: its id, its class, and its programmable limits.

    ``limits`` is every dataclass field except ``id`` (the id is already the
    descriptor's own key, and it is the last field on every concrete invariant
    in :mod:`horizon_ric.shield.invariants`). A field named in
    :data:`_DELEGATED_FIELDS` is reported as delegated without its value ever
    being read.

    A non-dataclass invariant is legal — :class:`Invariant` is a structural
    Protocol, nothing inherits — and yields ``limits: {}`` rather than an error.
    Its id and class name are still recorded, so the chain stays complete.
    """
    limits: dict[str, Any] = {}
    if dataclasses.is_dataclass(inv) and not isinstance(inv, type):
        for field in dataclasses.fields(inv):
            if field.name == "id":
                continue
            if field.name in _DELEGATED_FIELDS:
                limits[field.name] = dict(_DELEGATED_FIELDS[field.name])
                continue
            limits[field.name] = _limit_value(getattr(inv, field.name, None))
    return {
        "id": str(inv.id),
        "class_name": type(inv).__name__,
        "limits": limits,
    }


def _chain(shield: Shield) -> list[Invariant]:
    """The Shield's invariant instances, in enforcement order.

    ``Shield`` exposes only ``invariant_ids`` publicly, so this reaches for the
    private list to reflect the actual limits. If a future ``Shield`` stops
    carrying ``_invariants``, we degrade to id-only descriptors instead of
    failing — an incomplete profile is still auditable, a crashing emitter is
    not.
    """
    raw = getattr(shield, "_invariants", None)
    if isinstance(raw, list):
        return list(raw)
    return []


def _max_passes(shield: Shield) -> int:
    """``ShieldConfig.max_passes`` off the live shield, with the class default."""
    default = ShieldConfig().max_passes
    cfg = getattr(shield, "_cfg", None)
    raw = getattr(cfg, "max_passes", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    return int(default)


def emit_profile(shield: Shield, *, profile_id: str, description: str = "") -> dict[str, Any]:
    """Reflect ``shield`` into a JSON-serialisable assurance profile.

    ``invariant_chain`` is a **list, and its order is semantically significant**.
    :meth:`Shield.dispose` walks the chain in order on every projection pass, so
    an earlier invariant's projection is what a later one sees. That is load
    bearing, not incidental:

    * ``NumericSanityInvariant`` is first in every default chain so the physics
      and regulatory checks downstream only ever see finite values — NaN
      comparisons are false in surprising ways and a negative bandwidth appears
      to fit inside a band.
    * ``LawfulInterceptInvariant`` is placed immediately after it in
      ``default_terrestrial_shield`` / ``default_ntn_shield`` so a
      fail-closed LI refusal short-circuits the pass before any power or
      spectrum rewriting happens.
    * Spectrum/power precede the AI-PHY envelopes, so a fallback to a classical
      block is decided on an already-legal carrier.

    Two profiles with the same invariants in a different order therefore
    describe different enforcement and must produce different
    :func:`profile_digest` values. They do, because JSON arrays are ordered and
    ``sort_keys`` sorts keys, never array elements.
    """
    chain = _chain(shield)
    if chain:
        descriptors = [invariant_descriptor(inv) for inv in chain]
    else:
        # Degraded path: ids only (see ``_chain``). Recorded honestly as such.
        descriptors = [
            {"id": str(inv_id), "class_name": "unknown", "limits": {}}
            for inv_id in shield.invariant_ids
        ]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "description": description,
        "invariant_chain": descriptors,
        "max_passes": _max_passes(shield),
        "action_contract": {
            "required_keys": list(REQUIRED_ACTION_KEYS),
            "optional_keys": list(OPTIONAL_ACTION_KEYS),
            "forbidden_keys": list(OUTPUT_SENTINEL_KEYS),
            "json_primitive_values_only": True,
        },
    }


def canonical_profile_bytes(profile: dict[str, Any]) -> bytes:
    """The profile's canonical JSON — byte-identical rules to the cert signer.

    ``sort_keys=True`` with tight separators, no ``default=`` handler. A profile
    containing a non-JSON-primitive value raises ``TypeError`` here, which is
    the point: an unserialisable profile must not be silently digestible.
    """
    return json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")


def profile_digest(profile: dict[str, Any]) -> str:
    """SHA-256 hex digest of :func:`canonical_profile_bytes`.

    Stable across calls and across processes for an equal profile, so a
    deployment can pin its enforced envelope by digest and detect a silently
    reconfigured Shield after a redeploy.
    """
    return hashlib.sha256(canonical_profile_bytes(profile)).hexdigest()


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "canonical_profile_bytes",
    "emit_profile",
    "invariant_descriptor",
    "profile_digest",
]
