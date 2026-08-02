#!/usr/bin/env python3
"""Enumerate the gate-bearing constants of Horizon-RIC, from source, with provenance.

A **gate-bearing constant** is a numeric or threshold constant whose value
decides whether a *published gate* passes. Gate G2 asserts that every published
gate reproduces *without retuning any constant*; that assertion is only checkable
if the constants are enumerated and each is tied to the gate it moves. This
module is that enumeration.

Extraction, not transcription
-----------------------------
Every value here is read from the source file's **AST** at the pinned location,
never hand-copied. A transcribed value is correct once and then rots the next
time someone edits the source; an AST-extracted one cannot disagree with the
source because it *is* the source. This mirrors OCUDU's ``extract_catalogue.py``:
``generate_inventory()`` rebuilds the whole record set from source, and
``audit/tests/test_constant_inventory.py`` re-runs it and compares byte-for-byte
against the committed ``inventory.json`` — so a constant that moves without the
inventory being regenerated fails the drift guard rather than sailing through.

Each record carries: the symbol, the file and 1-indexed line, the current value,
its unit, and the published gate(s) whose pass/fail depends on it (named by the
verifier script). ``dependency`` states *how* the gate depends on it — ``sha256``
when the gate pins the file's SHA-256 (G1, G6, G8 style: any edit to the literal
changes the digest and fails the pin), ``behavioural`` when the gate re-runs a
check whose outcome the constant moves, or both.

Run ``python inventory.py --out inventory.json`` to regenerate the committed file.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# audit/constants/inventory.py  ->  repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_JSON = Path(__file__).resolve().parent / "inventory.json"

SCHEMA = "horizon-ric.audit.constant-inventory/1"


# ---------------------------------------------------------------------------
# Specification of every gate-bearing constant (provenance is committed here;
# the *value* and *line* are extracted from source, never written here).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstantSpec:
    """Where a gate-bearing constant lives and which gate(s) depend on it.

    ``kind`` selects the AST/live resolver:

    * ``dataclass_field`` — a class-body attribute default
      (``container`` = class name, ``symbol`` = attribute);
    * ``func_default``    — a function/method parameter default
      (``container`` = ``func`` or ``Class.method``, ``symbol`` = parameter);
    * ``module_const``    — a module-level assignment (``container`` = ``""``).
    """

    symbol: str
    module: str            # dotted import path
    file: str              # repo-relative source file
    kind: str
    container: str
    unit: str
    gates: tuple[str, ...]           # verifier scripts whose gate depends on it
    dependency: tuple[str, ...]      # "sha256" and/or "behavioural"
    note: str


@dataclass(frozen=True)
class ConstantRecord:
    """A resolved inventory row: the spec plus the extracted value and line."""

    symbol: str
    file: str
    line: int
    value: Any
    unit: str
    kind: str
    container: str
    module: str
    dependent_gates: tuple[str, ...]
    dependency: tuple[str, ...]
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "file": self.file,
            "line": self.line,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "container": self.container,
            "module": self.module,
            "dependent_gates": list(self.dependent_gates),
            "dependency": list(self.dependency),
            "note": self.note,
        }


_SHIELD = "src/horizon_ric/shield/shield.py"
_INV = "src/horizon_ric/shield/invariants.py"
_UCB = "src/horizon_ric/planners/ucb_spectrum.py"
_AGG = "agentic/src/horizon_agentic/aggregate.py"
_OCUDU = "ocudu/src/horizon_ocudu/rc_slice_quota.py"

_G1 = "scripts/verify_g1_second_planner.py"
_G6 = "agentic/verify_g6_multi_agent.py"
_G8 = "ocudu/verify_g8_ocudu_conformance.py"
_POISON = "scripts/verify_poisoning_shield.py"

CONSTANT_SPECS: tuple[ConstantSpec, ...] = (
    # ── shield.py ────────────────────────────────────────────────────────
    ConstantSpec(
        symbol="max_passes",
        module="horizon_ric.shield.shield",
        file=_SHIELD,
        kind="dataclass_field",
        container="ShieldConfig",
        unit="projection_passes",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note=(
            "Bound on Shield projection passes before fail-closed. G1 pins "
            "shield.py's SHA-256, so any edit to the literal fails the gate. "
            "Behaviourally the default terrestrial/NTN chains converge in <=1 "
            "pass, so the disposition is insensitive to the magnitude above 1 "
            "(8 is conservative headroom) — the swap harness flips the "
            "disposition only by dropping it to 0. See swap_harness finding."
        ),
    ),
    ConstantSpec(
        symbol="max_eirp_dBm",
        module="horizon_ric.shield.shield",
        file=_SHIELD,
        kind="func_default",
        container="default_terrestrial_shield",
        unit="dBm",
        gates=(_G1, _POISON),
        dependency=("sha256",),
        note=(
            "EIRP ceiling the terrestrial shield builder installs by default; "
            "mirrors MaxEirpInvariant.max_eirp_dBm and is what G1's oracle grades "
            "against (declared_constants.max_eirp_dBm = 33.0)."
        ),
    ),
    # ── invariants.py ────────────────────────────────────────────────────
    ConstantSpec(
        symbol="max_eirp_dBm",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="MaxEirpInvariant",
        unit="dBm",
        gates=(_G1, _POISON),
        dependency=("sha256", "behavioural"),
        note=(
            "Regulatory EIRP ceiling: EIRP_dBm = tx_power + antenna_gain must "
            "not exceed it. G1 pins invariants.py; the check flips at the "
            "boundary."
        ),
    ),
    ConstantSpec(
        symbol="floor",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="ProtectedSliceFloorInvariant",
        unit="fraction",
        gates=(_G1, _POISON),
        dependency=("sha256", "behavioural"),
        note=(
            "PRB fraction reserved for the safety-critical slice that an AI "
            "planner cannot reallocate away. G1 pins invariants.py; the slice "
            "floor check flips at the boundary."
        ),
    ),
    ConstantSpec(
        symbol="max_pfd_dBW_m2_MHz",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="PfdCeilingInvariant",
        unit="dBW/m^2/MHz",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note="ITU-R-style NTN downlink power-flux-density ceiling. G1 pins invariants.py.",
    ),
    ConstantSpec(
        symbol="guard_band_hz",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="SpectralMaskInvariant",
        unit="Hz",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note=(
            "Guard band kept inside each licensed band edge for ACLR. Default 0 "
            "preserves band-edge-flush behaviour; G1 runs it at 2 MHz "
            "(declared_constants.guard_band_hz) and pins invariants.py."
        ),
    ),
    ConstantSpec(
        symbol="tolerance_dB",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="NeuralRxEnvelopeInvariant",
        unit="dB",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note="Neural-RX allowed regression past the classical LMMSE baseline. G1 pins invariants.py.",
    ),
    ConstantSpec(
        symbol="min_confidence",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="NeuralRxEnvelopeInvariant",
        unit="fraction",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note="Minimum demap confidence for a verified neural-RX admit. G1 pins invariants.py.",
    ),
    ConstantSpec(
        symbol="max_papr_dB",
        module="horizon_ric.shield.invariants",
        file=_INV,
        kind="dataclass_field",
        container="ConstellationLegalityInvariant",
        unit="dB",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note="PAPR ceiling on a learned constellation before classical fallback. G1 pins invariants.py.",
    ),
    # ── ucb_spectrum.py (the 9x6 arm grid + PA envelope) ─────────────────
    ConstantSpec(
        symbol="frequency_arms",
        module="horizon_ric.planners.ucb_spectrum",
        file=_UCB,
        kind="func_default",
        container="UcbSpectrumPlanner.__init__",
        unit="count",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note=(
            "Centre-frequency arms in the planner's grid (the 9 of the 9x6 grid). "
            "G1 pins ucb_spectrum.py and its non-vacuity check requires the grid "
            "to still contain illegal arms."
        ),
    ),
    ConstantSpec(
        symbol="power_arms",
        module="horizon_ric.planners.ucb_spectrum",
        file=_UCB,
        kind="func_default",
        container="UcbSpectrumPlanner.__init__",
        unit="count",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note="Transmit-power arms in the planner's grid (the 6 of the 9x6 grid). G1 pins ucb_spectrum.py.",
    ),
    ConstantSpec(
        symbol="pa_ceiling_dBm",
        module="horizon_ric.planners.ucb_spectrum",
        file=_UCB,
        kind="func_default",
        container="UcbSpectrumPlanner.__init__",
        unit="dBm",
        gates=(_G1,),
        dependency=("sha256", "behavioural"),
        note=(
            "Power-amplifier ceiling (HARDWARE, above the 33 dBm regulatory "
            "ceiling on purpose) so the planner can request illegal emissions. "
            "G1 pins ucb_spectrum.py."
        ),
    ),
    ConstantSpec(
        symbol="pa_floor_dBm",
        module="horizon_ric.planners.ucb_spectrum",
        file=_UCB,
        kind="func_default",
        container="UcbSpectrumPlanner.__init__",
        unit="dBm",
        gates=(_G1,),
        dependency=("sha256",),
        note="Power-amplifier floor of the planner's power grid. G1 pins ucb_spectrum.py.",
    ),
    # ── aggregate.py (cross-agent invariants, G6) ────────────────────────
    ConstantSpec(
        symbol="min_capacity_hz",
        module="horizon_agentic.aggregate",
        file=_AGG,
        kind="dataclass_field",
        container="AbsoluteSliceCapacityFloor",
        unit="Hz",
        gates=(_G6,),
        dependency=("sha256", "behavioural"),
        note=(
            "Absolute spectrum (Hz) a protected slice must retain across a "
            "bundle — the cross-agent floor no per-action share can express. "
            "G6 exercises the aggregate chain; the check flips at the boundary."
        ),
    ),
    ConstantSpec(
        symbol="max_total_eirp_dBm",
        module="horizon_agentic.aggregate",
        file=_AGG,
        kind="dataclass_field",
        container="AggregateEirpBudget",
        unit="dBm",
        gates=(_G6,),
        dependency=("sha256", "behavioural"),
        note="Site-level EIRP budget summed (linear) over distinct carriers. G6.",
    ),
    ConstantSpec(
        symbol="guard_hz",
        module="horizon_agentic.aggregate",
        file=_AGG,
        kind="dataclass_field",
        container="SpectralSeparation",
        unit="Hz",
        gates=(_G6,),
        dependency=("sha256", "behavioural"),
        note="Guard separation required between carriers placed by different agents. G6.",
    ),
    ConstantSpec(
        symbol="max_pfd_dBW_m2_MHz",
        module="horizon_agentic.aggregate",
        file=_AGG,
        kind="dataclass_field",
        container="AggregatePfdCeiling",
        unit="dBW/m^2/MHz",
        gates=(_G6,),
        dependency=("sha256", "behavioural"),
        note="Aggregate PFD ceiling at a ground point summed (linear) over beams. G6.",
    ),
    ConstantSpec(
        symbol="tolerance",
        module="horizon_agentic.aggregate",
        file=_AGG,
        kind="dataclass_field",
        container="PrbConservation",
        unit="fraction",
        gates=(_G6,),
        dependency=("sha256", "behavioural"),
        note="Slack on the sum-of-shares <= 1 conservation check across a bundle. G6.",
    ),
    # ── ocudu rc_slice_quota.py (E2SM-RC slice-quota, G8) ────────────────
    ConstantSpec(
        symbol="MIN_PRB_POLICY_RATIO",
        module="horizon_ocudu.rc_slice_quota",
        file=_OCUDU,
        kind="module_const",
        container="",
        unit="ran_parameter_id",
        gates=(_G8,),
        dependency=("sha256", "behavioural"),
        note=(
            "E2SM-RC RAN Parameter ID for Min PRB Policy Ratio, asserted equal to "
            "the id OCUDU declares in the extracted catalogue. G8 fails if it "
            "drifts from OCUDU."
        ),
    ),
    ConstantSpec(
        symbol="MAX_PRB_POLICY_RATIO",
        module="horizon_ocudu.rc_slice_quota",
        file=_OCUDU,
        kind="module_const",
        container="",
        unit="ran_parameter_id",
        gates=(_G8,),
        dependency=("sha256", "behavioural"),
        note="E2SM-RC RAN Parameter ID for Max PRB Policy Ratio, asserted against the catalogue. G8.",
    ),
    ConstantSpec(
        symbol="DEDICATED_PRB_POLICY_RATIO",
        module="horizon_ocudu.rc_slice_quota",
        file=_OCUDU,
        kind="module_const",
        container="",
        unit="ran_parameter_id",
        gates=(_G8,),
        dependency=("sha256", "behavioural"),
        note="E2SM-RC RAN Parameter ID for Dedicated PRB Policy Ratio, asserted against the catalogue. G8.",
    ),
    ConstantSpec(
        symbol="max_ratio",
        module="horizon_ocudu.rc_slice_quota",
        file=_OCUDU,
        kind="dataclass_field",
        container="SliceQuota",
        unit="percent",
        gates=(),
        dependency=(),
        note=(
            "Default PRB max ratio (percent) on a SliceQuota. FINDING: no "
            "published gate is currently sensitive to this default — it is "
            "Horizon-side input validation, not asserted by G8, so it could be "
            "retuned freely. Recorded so the claim is honest, not to imply a gate."
        ),
    ),
)


# ---------------------------------------------------------------------------
# AST extraction (value + line) — the source is the single source of truth.
# ---------------------------------------------------------------------------
class _ModuleIndex:
    """AST index of one source file: module consts, class fields, functions."""

    def __init__(self, tree: ast.Module) -> None:
        self.module_consts: dict[str, tuple[ast.expr, int]] = {}
        self.class_fields: dict[tuple[str, str], tuple[ast.expr, int]] = {}
        self.funcs: dict[str, ast.FunctionDef] = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._record_assign(node, self.module_consts)
            elif isinstance(node, ast.FunctionDef):
                self.funcs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self._index_class(node)

    @staticmethod
    def _targets(node: ast.stmt) -> list[str]:
        if isinstance(node, ast.AnnAssign):
            return [node.target.id] if isinstance(node.target, ast.Name) else []
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        return []

    def _record_assign(
        self, node: ast.stmt, into: dict[str, tuple[ast.expr, int]]
    ) -> None:
        value = getattr(node, "value", None)
        if value is None:
            return
        for name in self._targets(node):
            into[name] = (value, node.lineno)

    def _index_class(self, cls: ast.ClassDef) -> None:
        for item in cls.body:
            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                value = getattr(item, "value", None)
                if value is None:
                    continue
                for name in self._targets(item):
                    self.class_fields[(cls.name, name)] = (value, item.lineno)
            elif isinstance(item, ast.FunctionDef):
                self.funcs[f"{cls.name}.{item.name}"] = item


def _func_default_node(func: ast.FunctionDef, arg_name: str) -> tuple[ast.expr, int]:
    """The default value node (and its line) for ``arg_name`` of ``func``."""
    args = func.args
    # Positional (and positional-or-keyword) args: defaults align to the tail.
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    if defaults:
        first_with_default = len(positional) - len(defaults)
        for offset, arg in enumerate(positional[first_with_default:]):
            if arg.arg == arg_name:
                node = defaults[offset]
                return node, node.lineno
    # Keyword-only args: 1:1 with kw_defaults (which may hold None entries).
    for arg, node in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == arg_name and node is not None:
            return node, node.lineno
    raise KeyError(f"no default for parameter {arg_name!r} in {func.name}")


_INDEX_CACHE: dict[str, _ModuleIndex] = {}


def _index_for(file_rel: str) -> _ModuleIndex:
    if file_rel not in _INDEX_CACHE:
        path = REPO_ROOT / file_rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _INDEX_CACHE[file_rel] = _ModuleIndex(tree)
    return _INDEX_CACHE[file_rel]


def resolve_record(spec: ConstantSpec) -> ConstantRecord:
    """Extract ``spec``'s value and line from source by AST (no import needed)."""
    index = _index_for(spec.file)
    if spec.kind == "module_const":
        node, line = index.module_consts[spec.symbol]
    elif spec.kind == "dataclass_field":
        node, line = index.class_fields[(spec.container, spec.symbol)]
    elif spec.kind == "func_default":
        func = index.funcs[spec.container]
        node, line = _func_default_node(func, spec.symbol)
    else:  # pragma: no cover - guarded by the spec table
        raise ValueError(f"unknown kind {spec.kind!r}")
    value = ast.literal_eval(node)
    return ConstantRecord(
        symbol=spec.symbol,
        file=spec.file,
        line=line,
        value=value,
        unit=spec.unit,
        kind=spec.kind,
        container=spec.container,
        module=spec.module,
        dependent_gates=spec.gates,
        dependency=spec.dependency,
        note=spec.note,
    )


# ---------------------------------------------------------------------------
# Live value (imported from source) — used by the equivalence test.
# ---------------------------------------------------------------------------
def live_value(spec: ConstantSpec) -> Any:
    """The value as the *running* code sees it, imported from ``spec.module``.

    This deliberately takes a different path from :func:`resolve_record` (import
    + reflection vs. AST): the equivalence test asserts the two agree, so a
    committed inventory that matched neither a stale AST nor the live import
    would be caught.
    """
    module = importlib.import_module(spec.module)
    if spec.kind == "module_const":
        return getattr(module, spec.symbol)
    if spec.kind == "dataclass_field":
        return getattr(getattr(module, spec.container), spec.symbol)
    if spec.kind == "func_default":
        obj: Any = module
        for part in spec.container.split("."):
            obj = getattr(obj, part)
        return inspect.signature(obj).parameters[spec.symbol].default
    raise ValueError(f"unknown kind {spec.kind!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Inventory assembly / serialisation.
# ---------------------------------------------------------------------------
def generate_inventory() -> dict[str, Any]:
    """Rebuild the whole inventory from source. The regeneration entry point."""
    records = [resolve_record(spec) for spec in CONSTANT_SPECS]
    records.sort(key=lambda r: (r.file, r.line, r.symbol))
    return {
        "schema": SCHEMA,
        "description": (
            "Gate-bearing constants of Horizon-RIC: numeric/threshold constants "
            "whose value decides whether a published gate passes. Regenerated "
            "from source by audit/constants/inventory.py (AST extraction, never "
            "transcription). 'dependent_gates' names the verifier script(s); "
            "'dependency' is 'sha256' when the gate pins the file's digest and/or "
            "'behavioural' when a gate re-runs a check the constant moves. This is "
            "the enumeration gate G2 ('no constant retuned') is checked against."
        ),
        "generated_by": "audit/constants/inventory.py",
        "constant_count": len(records),
        "constants": [r.to_json() for r in records],
    }


def inventory_json_text() -> str:
    """Canonical serialisation used for both writing and the drift-guard diff."""
    return json.dumps(generate_inventory(), indent=2, sort_keys=True) + "\n"


def load_committed_inventory(path: Path = INVENTORY_JSON) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the regenerated inventory here (default: stdout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed inventory.json differs from a fresh regen",
    )
    args = parser.parse_args()

    text = inventory_json_text()

    if args.check:
        committed = INVENTORY_JSON.read_text(encoding="utf-8") if INVENTORY_JSON.exists() else ""
        if committed != text:
            print("inventory.json is STALE — rerun with --out to regenerate", file=sys.stderr)
            return 1
        print(f"inventory.json is up to date ({len(CONSTANT_SPECS)} constants)", file=sys.stderr)
        return 0

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(CONSTANT_SPECS)} constants)", file=sys.stderr)
    else:
        print(text, end="")

    for spec in CONSTANT_SPECS:
        rec = resolve_record(spec)
        gates = ", ".join(rec.dependent_gates) or "(no gate — see note)"
        print(
            f"  {rec.file}:{rec.line}  {rec.container}.{rec.symbol} = "
            f"{rec.value!r} {rec.unit}  ->  {gates}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
