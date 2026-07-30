"""The WP1 planner interface — the seam between an AI planner and the Shield.

**New in WP1.** Before this module there was no planner interface anywhere in
``src/``: no ABC, no Protocol, no injection point.
:class:`horizon_ric.rapp.pipeline.DecisionPipeline` hard-codes its planner as
private methods (``_candidates`` / ``_choose`` / ``_shield_action``) and
constructs its own :class:`~horizon_ric.shield.shield.Shield`. **This module
does not change that.** ``DecisionPipeline`` does *not* yet implement or consume
:class:`Planner`; adopting it is deliberately deferred so WP1 can publish and
freeze the contract before anything is rewired. Treat this file as the *declared*
interface, not as a description of the current pipeline.

What the interface pins
-----------------------
A planner is anything that turns an observation into a *proposed action*::

    action = planner.propose(observation)          # the AI proposes
    disposition = shielded(planner, shield, obs, decision_id="d-1")

The Shield already disposes safely for any input — it never raises for an
unsatisfiable action, it stamps ``emit_blocked`` and returns a certificate with
non-empty ``violated_ids``. So the value this module adds is *not* safety; it is
**failing at the boundary instead of at signing time**:

1. **JSON-primitive values only.**
   :func:`horizon_ric.shield.signing.canonical_certificate_bytes` does
   ``json.dumps(cert.to_dict() minus signature fields, sort_keys=True,
   separators=(",", ":"))`` with **no** ``default=`` handler. The certificate
   embeds the planner's proposed dict verbatim under ``action_proposed``, so a
   ``Decimal``, a ``datetime``, an ``Enum``, a ``set``, ``bytes``, or a
   non-``float``-subclass numpy scalar anywhere in the action raises
   ``TypeError`` *at signing time* — long after the decision was made, on the
   evidence path, where the failure is hardest to attribute. This module rejects
   it at ``propose()`` instead.
2. **Output sentinels are not planner inputs.** ``emit_blocked`` and
   ``shield_fallback_to`` (:data:`~horizon_ric.shield.invariants.FALLBACK_KEY`)
   are written *by* the Shield. A planner that presets either one is forging the
   Shield's own verdict, so the contract forbids them outright.
3. **The de facto required keys become de jure.** The required action keys are
   currently enforced only implicitly, inside
   ``NumericSanityInvariant._problems``. :data:`REQUIRED_ACTION_KEYS` and
   :data:`OPTIONAL_ACTION_KEYS` state them explicitly, and
   :func:`validate_proposed_action` mirrors that invariant's domains so a
   planner author learns the contract from a list of sentences rather than from
   a blocked emit.

The machine-readable form of this contract is
``docs/schemas/proposed-action-v1.schema.json``;
``tests/test_assurance_schemas.py`` pins the two against each other.

Stdlib only — this module is importable on the decision path.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Mapping, Protocol, runtime_checkable

from horizon_ric.shield.certificate import ShieldDisposition
from horizon_ric.shield.invariants import FALLBACK_KEY, Action, Context
from horizon_ric.shield.shield import Shield

#: What a planner is given. A read-only mapping: telemetry, KPMs, channel
#: estimates, whatever the deployment supplies. Deliberately unconstrained —
#: WP1 pins the planner's *output* contract, not its input featurisation.
Observation = Mapping[str, Any]

#: Keys every conformant proposed action must carry.
#:
#: ``frequency_hz`` / ``bandwidth_hz`` / ``tx_power_dBm`` are the numeric domains
#: ``NumericSanityInvariant`` checks unconditionally. ``block`` is required here
#: even though the Shield reads it defensively via ``.get("block", "unknown")``:
#: it is the AI-PHY/RIC block identity recorded on the certificate and the key
#: that dispatches the AI-PHY invariants, so an action without it is
#: unattributable evidence.
REQUIRED_ACTION_KEYS: tuple[str, ...] = (
    "block",
    "frequency_hz",
    "bandwidth_hz",
    "tx_power_dBm",
)

#: Keys a planner may carry. Each is either domain-checked when present by
#: ``NumericSanityInvariant`` or consumed by a downstream invariant
#: (``PfdCeilingInvariant``, ``NeuralRxEnvelopeInvariant``,
#: ``ConstellationLegalityInvariant``, ``ProtectedSliceFloorInvariant``).
#:
#: This tuple is *not* an allow-list: the action shape is open, and a planner may
#: carry deployment-specific keys the Shield ignores. Extra keys must still be
#: JSON-primitive and must not be output sentinels.
OPTIONAL_ACTION_KEYS: tuple[str, ...] = (
    "antenna_gain_dBi",
    "sat_antenna_gain_dBi",
    "predicted_tbler",
    "baseline_tbler",
    "demap_confidence",
    "papr_dB",
    "constellation_order",
    "ntn",
    "slant_range_m",
    "prb_allocation",
)

#: Keys the Shield *writes*. A planner must never set them.
OUTPUT_SENTINEL_KEYS: tuple[str, ...] = ("emit_blocked", FALLBACK_KEY)

# Exact JSON primitive types. Deliberately compared by ``type(...) is`` rather
# than ``isinstance``, because subclasses are exactly the trap: ``numpy.float64``
# subclasses ``float`` and happens to serialise, ``numpy.int64`` does not
# subclass ``int`` and raises, and ``IntEnum`` serialises as its integer and so
# silently loses its identity in the signed evidence. Rejecting the whole family
# at the boundary is cheaper than reasoning about which members survive
# ``json.dumps``.
_JSON_SCALARS: tuple[type, ...] = (bool, int, float, str)


class PlannerContractError(ValueError):
    """A planner emitted an action that violates the WP1 action contract.

    Raised by :func:`shielded` *before* the Shield is consulted, so a
    non-conformant planner fails at the boundary with an attributable message
    rather than at certificate-signing time with a bare ``TypeError``.
    """

    def __init__(self, planner_id: str, problems: list[str]) -> None:
        self.planner_id = planner_id
        self.problems = list(problems)
        detail = "; ".join(self.problems)
        super().__init__(
            f"planner {planner_id!r} proposed a non-conformant action "
            f"({len(self.problems)} problem(s)): {detail}"
        )


@runtime_checkable
class Planner(Protocol):
    """Anything that proposes an action for the Shield to dispose of.

    Structural, like :class:`horizon_ric.shield.invariants.Invariant`: an
    implementation matches the shape, it does not inherit. That keeps a learned
    policy head, a rule-based fallback, a replayed trace, and a test double all
    equally admissible.

    ``planner_id`` is a stable string recorded on the evidence path so a
    disposition can be attributed to the thing that proposed it (mirroring
    ``Invariant.id``).
    """

    planner_id: str

    def propose(self, observation: Observation) -> Action: ...


def planner_contract_problems(candidate: Any) -> list[str]:
    """Report every way ``candidate`` fails to be a :class:`Planner`.

    ``isinstance(x, Planner)`` is not a conformance check and must not be used
    as one. ``runtime_checkable`` verifies attribute *presence* only: a planner
    whose method is ``propose(self)`` returning some private dataclass passes
    ``isinstance`` and then raises ``TypeError`` at the call site, because the
    arity was never compared. That is a worse failure than a clean rejection,
    since it looks like conformance right up to the moment it is used.

    So this checks the signature. It is the executable form of requirement P-1
    in ``docs/conformance/ASSURANCE_PROFILE.md`` and closes that document's
    first known gap; an empty list means conformant.
    """
    problems: list[str] = []

    planner_id = getattr(candidate, "planner_id", None)
    if planner_id is None:
        problems.append("missing the required 'planner_id' attribute")
    elif not isinstance(planner_id, str) or not planner_id:
        problems.append(
            f"planner_id must be a non-empty str, got {planner_id!r}"
        )

    propose = getattr(candidate, "propose", None)
    if propose is None:
        problems.append("missing the required 'propose' method")
        return problems
    if not callable(propose):
        problems.append("'propose' is not callable")
        return problems

    try:
        sig = inspect.signature(propose)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exotic callables
        problems.append(f"cannot introspect 'propose': {type(exc).__name__}")
        return problems

    # One positional parameter, the observation. `propose` is read off the
    # instance, so `self` is already bound and must not be counted.
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    takes_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if not positional and not takes_var_positional:
        problems.append(
            "propose() takes no observation argument; the contract is "
            f"propose(observation) -> Action, got propose{sig}"
        )
    elif len(required) > 1:
        problems.append(
            f"propose() requires {len(required)} positional arguments; the "
            f"contract is propose(observation) -> Action, got propose{sig}"
        )

    return problems


def conforms_to_planner(candidate: Any) -> bool:
    """``True`` when ``candidate`` satisfies the :class:`Planner` contract.

    Prefer :func:`planner_contract_problems` when reporting to a human — it
    says *why*.
    """
    return not planner_contract_problems(candidate)


def _json_problems(value: Any, path: str) -> list[str]:
    """Recursively report every value that ``json.dumps`` could not safely emit.

    Mirrors the *exact* serialisation the certificate signer performs, minus the
    ``sort_keys`` ordering: no ``default=`` hook, so anything outside the JSON
    primitive set is a hard failure.
    """
    if value is None or type(value) in _JSON_SCALARS:
        if type(value) is float and not math.isfinite(value):
            return [
                f"{path}: {value!r} is not representable as strict JSON "
                "(json.dumps emits the non-standard NaN/Infinity literals)"
            ]
        return []
    if isinstance(value, dict):
        problems: list[str] = []
        for key, item in value.items():
            if type(key) is not str:
                problems.append(
                    f"{path}: mapping key {key!r} is {type(key).__name__}, not str — "
                    "canonical JSON requires string keys (sort_keys cannot order "
                    "mixed-type keys)"
                )
                continue
            problems.extend(_json_problems(item, f"{path}.{key}"))
        return problems
    if isinstance(value, (list, tuple)):
        problems = []
        for index, item in enumerate(value):
            problems.extend(_json_problems(item, f"{path}[{index}]"))
        return problems
    return [
        f"{path}: value of type {type(value).__name__} is not JSON-primitive — "
        "horizon_ric.shield.signing.canonical_certificate_bytes calls json.dumps "
        "with no default= handler, so this raises TypeError at signing time"
    ]


def _finite(value: Any) -> float | None:
    """``float(value)`` when it is a finite real number, else ``None``.

    Deliberately identical in behaviour to ``NumericSanityInvariant._number`` so
    the boundary check and the invariant cannot disagree about what "finite"
    means.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _domain_problems(action: Mapping[str, Any]) -> list[str]:
    """Mirror ``NumericSanityInvariant._problems`` on the planner's own output."""
    problems: list[str] = []

    frequency = _finite(action.get("frequency_hz"))
    if frequency is None or frequency <= 0.0:
        problems.append("frequency_hz must be finite and > 0")

    bandwidth = _finite(action.get("bandwidth_hz"))
    if bandwidth is None or bandwidth <= 0.0:
        problems.append("bandwidth_hz must be finite and > 0")

    if _finite(action.get("tx_power_dBm")) is None:
        problems.append("tx_power_dBm must be finite")

    for name in ("antenna_gain_dBi", "sat_antenna_gain_dBi"):
        if name in action and _finite(action[name]) is None:
            problems.append(f"{name} must be finite")

    for name in ("predicted_tbler", "baseline_tbler", "demap_confidence"):
        if name not in action:
            continue
        value = _finite(action[name])
        if value is None or not 0.0 <= value <= 1.0:
            problems.append(f"{name} must be finite and in [0, 1]")

    if "papr_dB" in action:
        papr = _finite(action["papr_dB"])
        if papr is None or papr < 0.0:
            problems.append("papr_dB must be finite and >= 0")

    if "constellation_order" in action:
        order = _finite(action["constellation_order"])
        if order is None or order <= 0.0 or not order.is_integer():
            problems.append("constellation_order must be a positive integer")

    if bool(action.get("ntn", False)):
        slant = _finite(action.get("slant_range_m"))
        if slant is None or slant <= 0.0:
            problems.append(
                "ntn is truthy so slant_range_m must be present, finite and > 0 "
                "(PfdCeilingInvariant otherwise silently assumes a ~550 km LEO "
                "slant range, which is a fabricated regulatory margin)"
            )

    if "prb_allocation" in action:
        alloc = action["prb_allocation"]
        if not isinstance(alloc, dict):
            problems.append("prb_allocation must be an object of slice_id -> fraction")
        else:
            for slice_id, share in alloc.items():
                if type(slice_id) is not str:
                    problems.append("prb_allocation keys must be slice-id strings")
                elif _finite(share) is None:
                    problems.append(f"prb_allocation[{slice_id!r}] must be a finite number")

    return problems


def validate_proposed_action(action: Any) -> list[str]:
    """Every way ``action`` breaches the WP1 planner contract, as sentences.

    An **empty list means conformant**. Problems reported, in order:

    * ``action`` is not a mapping at all;
    * a key in :data:`REQUIRED_ACTION_KEYS` is missing;
    * an output sentinel (:data:`OUTPUT_SENTINEL_KEYS`) is preset — the planner
      is forging the Shield's verdict;
    * a value anywhere in the structure is not JSON-primitive, which would raise
      ``TypeError`` inside ``canonical_certificate_bytes`` once the action is
      embedded in a certificate as ``action_proposed`` (recursive: nested dicts
      and lists are walked);
    * ``ntn`` is truthy without a usable ``slant_range_m``;
    * a conditionally-checked key is out of the domain
      ``NumericSanityInvariant`` enforces.

    This is a *contract* check, not a safety check. It says nothing about
    whether the action is inside the safe set — that is the Shield's job, and
    the Shield is still the only thing that decides. A conformant action can be
    (and often is) wildly unsafe.
    """
    if not isinstance(action, Mapping):
        return [
            f"proposed action must be a mapping, got {type(action).__name__} — "
            "horizon_ric.shield.invariants.Action is dict[str, Any]"
        ]

    problems: list[str] = []

    for key in REQUIRED_ACTION_KEYS:
        if key not in action:
            problems.append(f"missing required key {key!r}")

    for sentinel in OUTPUT_SENTINEL_KEYS:
        if sentinel in action:
            problems.append(
                f"{sentinel!r} is a Shield OUTPUT sentinel and must never be set by a "
                "planner; the Shield writes it when it refuses or falls back"
            )

    problems.extend(_json_problems(dict(action), "action"))
    problems.extend(_domain_problems(action))
    return problems


def shielded(
    planner: Planner,
    shield: Shield,
    observation: Observation,
    *,
    decision_id: str,
    context: Context | None = None,
) -> ShieldDisposition:
    """Adapter: ``planner.propose`` → contract check → ``shield.dispose``.

    The contract check is **not** optional and runs *before* the Shield. A
    non-conformant planner raises :class:`PlannerContractError` here, at the
    boundary, naming the planner and listing every problem — instead of
    producing a certificate that explodes with a bare ``TypeError`` when
    ``horizon_ric.shield.signing`` tries to serialise it.

    Everything after the check is unchanged Shield behaviour: ``dispose`` never
    raises for an unsatisfiable action, it stamps ``emit_blocked`` on the safe
    action and returns a certificate with non-empty ``violated_ids``. So a
    caller of this function still gets a disposition for an unsafe proposal —
    the exception is reserved strictly for a *malformed* one.
    """
    action = planner.propose(observation)
    problems = validate_proposed_action(action)
    if problems:
        planner_id = str(getattr(planner, "planner_id", type(planner).__name__))
        raise PlannerContractError(planner_id, problems)
    return shield.dispose(action, dict(context or {}), decision_id=decision_id)


__all__ = [
    "OPTIONAL_ACTION_KEYS",
    "OUTPUT_SENTINEL_KEYS",
    "REQUIRED_ACTION_KEYS",
    "Observation",
    "Planner",
    "PlannerContractError",
    "conforms_to_planner",
    "planner_contract_problems",
    "shielded",
    "validate_proposed_action",
]
