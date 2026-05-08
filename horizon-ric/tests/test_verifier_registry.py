"""Verifier registry — register / get / chain composition.

Exercises the real built-ins (gso_pfd, itu_spectral_mask, edge_gpu, epfd,
li_jurisdiction) and validates that a custom verifier registered at
runtime is invoked by the constraint layer's ``verifier_chain`` path —
which is the same path the rApp's emit path uses.
"""

from __future__ import annotations

import pytest

from horizon_ric.contracts.constraint_layer import ConstraintViolation
from horizon_ric.policy.constraints import (
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)
from horizon_ric.policy.li_constraint import LIJurisdictionRule
from horizon_ric.policy.verifier_registry import (
    Verifier,
    VerifierRegistry,
    get_verifier,
    list_verifiers,
    register_verifier,
    run_verifier_chain,
)


# ─── Built-ins ───────────────────────────────────────────────────────────


def test_builtin_verifiers_registered():
    """All five named built-ins must be registered."""
    names = list_verifiers()
    for required in (
        "gso_pfd",
        "itu_spectral_mask",
        "edge_gpu",
        "epfd",
        "li_jurisdiction",
    ):
        assert required in names, f"missing built-in verifier {required!r}"


def test_spectral_mask_violation_emitted():
    """Out-of-band frequency must be flagged by the itu_spectral_mask
    verifier and projected onto the band edge."""
    v = get_verifier("itu_spectral_mask")
    action = {"frequency_hz": 50e9}
    ctx = {"permitted_band_hz": (10e9, 12e9)}
    viols = v.check(action, ctx)
    assert len(viols) == 1
    assert viols[0].constraint_id == "itu_spectral_mask"

    feasible, corr = v.project(action, ctx)
    assert feasible["frequency_hz"] == 12e9  # clipped to band high
    assert len(corr) == 1


def test_edge_gpu_violation_routes_to_regional():
    """Over-budget GPU memory must trigger placement to regional/cloud."""
    v = get_verifier("edge_gpu")
    action = {"ai_workload_gpu_gb": 32.0}
    ctx = {}
    viols = v.check(action, ctx)
    assert any(x.constraint_id == "edge_gpu_capacity" for x in viols)

    feasible, corr = v.project(action, ctx)
    assert feasible["ai_workload_gpu_gb"] == 0.0
    assert feasible["workload_placement"] == "regional_edge_or_cloud"
    assert any(c.constraint_id == "edge_gpu_capacity" for c in corr)


def test_li_jurisdiction_strips_protected_ues():
    """A custom-config verifier must drop UEs flagged by the LIMF."""
    rules = [
        LIJurisdictionRule(
            rule_id="warrant_de_001",
            protected_ue_ids={"ue_target_42"},
            allowed_jurisdictions={"DE", "EU"},
        ),
    ]
    v = get_verifier("li_jurisdiction", rules)
    action = {"affected_ue_ids": ["ue_target_42", "ue_normal_7"]}
    feasible, corr = v.project(action, {})
    assert "ue_target_42" not in feasible["affected_ue_ids"]
    assert any(c.constraint_id == "li_protected_ue" for c in corr)


# ─── Custom verifier round-trip ──────────────────────────────────────────


class _CountingVerifier(Verifier):
    """Sentinel verifier — records every invocation. Used to prove that
    the chain *actually* called us (not silently no-op'd)."""

    constraint_id = "_test_counting"
    invocations: list[str] = []

    def __init__(self, config=None):
        self.cfg = config

    def check(self, action, ctx):
        _CountingVerifier.invocations.append("check")
        if action.get("__force_violation__"):
            return [
                ConstraintViolation(
                    constraint_id=self.constraint_id,
                    severity="hard",
                    margin_dB=None,
                    message="forced",
                )
            ]
        return []

    def project(self, action, ctx):
        _CountingVerifier.invocations.append("project")
        out = dict(action)
        if out.pop("__force_violation__", False):
            corr = [
                ConstraintViolation(
                    constraint_id=self.constraint_id,
                    severity="hard",
                    margin_dB=None,
                    message="forced-projected",
                )
            ]
            return out, corr
        return out, []


def test_register_custom_verifier_runs_in_chain():
    """A user verifier registered at runtime must be invoked by the
    PreceptualAIConstraintLayer when listed in its ``verifier_chain``."""
    _CountingVerifier.invocations = []
    register_verifier("__test_counting__", _CountingVerifier)
    assert "__test_counting__" in list_verifiers()

    layer = PreceptualAIConstraintLayer(
        PreceptualAIConstraintConfig(),
        verifier_chain=["__test_counting__"],
    )
    feasible, corr = layer.project({"__force_violation__": True}, {})
    assert "__force_violation__" not in feasible  # popped by projection
    assert any(c.constraint_id == "_test_counting" for c in corr)
    assert "project" in _CountingVerifier.invocations


def test_unknown_verifier_raises_loudly():
    """Asking for an unregistered verifier must raise with a list."""
    with pytest.raises(KeyError) as exc:
        get_verifier("__never_registered__")
    assert "registered:" in str(exc.value)


def test_run_verifier_chain_helper():
    """The composition helper must apply projects in order then check
    residuals."""
    cfg = PreceptualAIConstraintConfig()
    cfg.gso_arcs = [GSOArcEntry("test_gso", longitude_deg=0.0)]

    feasible, corrections, residual = run_verifier_chain(
        ["itu_spectral_mask", "edge_gpu"],
        action={"frequency_hz": 99e9, "ai_workload_gpu_gb": 64.0},
        ctx={"permitted_band_hz": (10e9, 12e9)},
        configs={"itu_spectral_mask": cfg, "edge_gpu": cfg},
    )
    assert feasible["frequency_hz"] == 12e9
    assert feasible["ai_workload_gpu_gb"] == 0.0
    assert len(corrections) >= 1
    # After projection, no residual violations should remain on those ids.
    for r in residual:
        assert r.constraint_id not in {"itu_spectral_mask", "edge_gpu_capacity"}


def test_separate_registry_isolation():
    """Two registries must not share state."""
    r1 = VerifierRegistry()
    r2 = VerifierRegistry()
    r1.register("only_in_r1", _CountingVerifier)
    assert "only_in_r1" in r1.list_verifiers()
    assert "only_in_r1" not in r2.list_verifiers()
