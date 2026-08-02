"""Authority must be enforced, not recorded.

The distinction this file exists to pin: it is easy to build an envelope that
carries agent identity, permissions and a delegation chain, log all of it, and
still admit every action — at which point the envelope is provenance metadata
wearing the costume of access control. Each test below drives a refusal that
would not happen if the field were merely recorded.
"""

from __future__ import annotations

import pytest
from horizon_agentic.envelope import (
    ALWAYS_WRITABLE,
    SCOPE_ACTION_KEYS,
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
    authorize,
)

from horizon_ric.assurance.planner import (
    OPTIONAL_ACTION_KEYS,
    OUTPUT_SENTINEL_KEYS,
    REQUIRED_ACTION_KEYS,
)

ROOT = "operator-root"


def make_policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(ROOT, frozenset({"ran", "core"}),
                                 frozenset(SCOPE_ACTION_KEYS), 0),
            "orchestrator": AuthorityGrant(
                "orchestrator", frozenset({"ran"}),
                frozenset({"spectrum", "slice"}), 10
            ),
            "spectrum-agent": AuthorityGrant(
                "spectrum-agent", frozenset({"ran"}), frozenset({"spectrum"}), 20
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), 15
            ),
            "rogue-widener": AuthorityGrant(
                # Granted more than its delegator holds — the policy is
                # internally inconsistent, and the chain check must catch it.
                "rogue-widener", frozenset({"ran", "core"}),
                frozenset({"spectrum", "slice", "phy"}), 30
            ),
        },
    )


def envelope(**overrides) -> AgentActionEnvelope:
    base = dict(
        agent_id="spectrum-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "orchestrator", "spectrum-agent"),
        requested_action={
            "block": "spectrum",
            "frequency_hz": 3.45e9,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 20.0,
        },
        mutates=frozenset({"bandwidth_hz"}),
    )
    base.update(overrides)
    return AgentActionEnvelope(**base)


BASELINE = {
    "block": "spectrum",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
}


def test_a_well_formed_envelope_is_authorized() -> None:
    verdict = authorize(envelope(), make_policy(), baseline=BASELINE)
    assert verdict.authorized, verdict.problems


def test_no_baseline_refuses_undeclared_state_rather_than_waving_it_through() -> None:
    """The degraded mode must stay sound, not silently permissive.

    Found by adversarial review, reproduced end to end: a slice-scoped agent
    carrying `tx_power_dBm` it did not declare as a mutation was authorized
    when no baseline was supplied, and the Shield then clamped the value to
    something *legal* and emitted it. Clamping restores physics, not authority
    — the agent still moved transmit power with no spectrum scope.
    """
    no_baseline = authorize(envelope(), make_policy())
    assert not no_baseline.authorized
    assert any("no baseline was supplied" in p for p in no_baseline.problems)

    # Declaring everything it carries is sound without a baseline, so the
    # degraded mode is restrictive rather than unusable.
    complete = authorize(
        envelope(mutates=frozenset({"frequency_hz", "bandwidth_hz", "tx_power_dBm"})),
        make_policy(),
    )
    assert complete.authorized, complete.problems


def test_the_operator_root_may_not_present_envelopes() -> None:
    """A one-element chain skips narrowing entirely.

    `(ROOT,)` from `agent_id=ROOT` satisfies every structural check — starts at
    root, ends at the presenting agent, no repeats — and `zip(chain, chain[1:])`
    is empty, so no narrowing is evaluated and the caller inherits every scope
    the root holds. Reproduced by adversarial review.
    """
    verdict = authorize(
        AgentActionEnvelope(
            agent_id=ROOT,
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset(SCOPE_ACTION_KEYS),
            delegation_chain=(ROOT,),
            requested_action=dict(BASELINE),
            mutates=frozenset({"tx_power_dBm"}),
        ),
        make_policy(),
        baseline=BASELINE,
    )
    assert not verdict.authorized
    assert any("may not present envelopes" in p for p in verdict.problems)
    assert any("no delegation step" in p for p in verdict.problems)


def test_ungranted_agent_is_refused() -> None:
    verdict = authorize(envelope(agent_id="nobody"), make_policy())
    assert not verdict.authorized


def test_agent_cannot_mutate_outside_its_scope() -> None:
    """The load-bearing case.

    A spectrum agent changing ``prb_allocation`` produces a perfectly valid
    action — every physics and regulatory invariant would pass it. Only
    authority can object.
    """
    verdict = authorize(
        envelope(
            requested_action={
                "block": "spectrum",
                "frequency_hz": 3.45e9,
                "bandwidth_hz": 20e6,
                "tx_power_dBm": 20.0,
                "prb_allocation": {"safety_critical": 0.05, "embb": 0.95},
            },
            mutates=frozenset({"bandwidth_hz", "prb_allocation"}),
        ),
        make_policy(),
    )
    assert not verdict.authorized
    assert any("prb_allocation" in p for p in verdict.problems)


def test_declaring_state_is_not_mutating_it() -> None:
    """A slice agent must be able to state the bandwidth it allocates within.

    ``REQUIRED_ACTION_KEYS`` forces every action to carry ``frequency_hz``,
    ``bandwidth_hz`` and ``tx_power_dBm``, so an authority model that keyed off
    mere presence would make a legal slice action impossible to express. The
    line is between declaring observed state and requesting a change to it.
    """
    verdict = authorize(
        AgentActionEnvelope(
            agent_id="slice-agent",
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset({"slice"}),
            delegation_chain=(ROOT, "orchestrator", "slice-agent"),
            requested_action={**BASELINE, "block": "slice",
                              "prb_allocation": {"safety_critical": 0.2, "embb": 0.8}},
            mutates=frozenset({"prb_allocation"}),
        ),
        make_policy(),
        baseline={**BASELINE, "block": "slice"},
    )
    assert verdict.authorized, verdict.problems


def test_undeclared_change_to_declared_state_is_smuggling() -> None:
    """The check that makes ``mutates`` mean anything.

    Without a baseline comparison, an agent could change ``bandwidth_hz`` while
    declaring it mutates only ``prb_allocation``, and key-scoping would pass
    it: the mutated set is in scope, and the out-of-scope key is "merely
    declared". This is the exact bypass.
    """
    verdict = authorize(
        AgentActionEnvelope(
            agent_id="slice-agent",
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset({"slice"}),
            delegation_chain=(ROOT, "orchestrator", "slice-agent"),
            requested_action={
                **BASELINE,
                "block": "slice",
                "bandwidth_hz": 20e6,  # <- changed, but not declared
                "prb_allocation": {"safety_critical": 0.2, "embb": 0.8},
            },
            mutates=frozenset({"prb_allocation"}),
        ),
        make_policy(),
        baseline={**BASELINE, "block": "slice"},
    )
    assert not verdict.authorized
    assert any("without declaring them as mutations" in p for p in verdict.problems)


def test_mutation_must_be_carried_by_the_action() -> None:
    verdict = authorize(
        envelope(mutates=frozenset({"bandwidth_hz", "antenna_gain_dBi"})),
        make_policy(),
    )
    assert not verdict.authorized
    assert any("does not carry" in p for p in verdict.problems)


def test_agent_cannot_set_shield_output_sentinels() -> None:
    """An agent that could set ``shield_fallback_to`` could forge a fallback.

    These keys are the Shield's own output. An action arriving with them
    pre-set is either confused or lying, and both are refusals.
    """
    for sentinel in sorted(OUTPUT_SENTINEL_KEYS):
        verdict = authorize(
            envelope(
                requested_action={
                    "block": "spectrum",
                    "frequency_hz": 3.45e9,
                    "bandwidth_hz": 20e6,
                    "tx_power_dBm": 20.0,
                    sentinel: True,
                },
                mutates=frozenset({"bandwidth_hz", sentinel}),
            ),
            make_policy(),
        )
        assert not verdict.authorized, sentinel
        assert any(sentinel in p for p in verdict.problems), sentinel


def test_over_claiming_scopes_is_refused_not_granted() -> None:
    verdict = authorize(
        envelope(granted_scopes=frozenset({"spectrum", "slice"})), make_policy()
    )
    assert not verdict.authorized
    assert any("claims scopes it was not granted" in p for p in verdict.problems)


def test_wrong_domain_is_refused() -> None:
    verdict = authorize(envelope(target_domain="core"), make_policy())
    assert not verdict.authorized
    assert any("domain" in p for p in verdict.problems)


def test_chain_must_start_at_the_operator_root() -> None:
    verdict = authorize(
        envelope(delegation_chain=("orchestrator", "spectrum-agent")), make_policy()
    )
    assert not verdict.authorized
    assert any("operator root" in p for p in verdict.problems)


def test_chain_must_end_at_the_presenting_agent() -> None:
    verdict = authorize(
        envelope(delegation_chain=(ROOT, "orchestrator", "slice-agent")), make_policy()
    )
    assert not verdict.authorized
    assert any("ends at" in p for p in verdict.problems)


def test_delegation_may_not_widen_scopes() -> None:
    """Without this rule the chain is decoration.

    ``orchestrator`` holds {spectrum, slice}. ``rogue-widener`` is granted
    {spectrum, slice, phy}, so the delegation step gains a scope its delegator
    never had. If widening were permitted, any agent could reach any authority
    by naming a suitable delegator.
    """
    policy = make_policy()
    verdict = authorize(
        AgentActionEnvelope(
            agent_id="rogue-widener",
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset({"phy"}),
            delegation_chain=(ROOT, "orchestrator", "rogue-widener"),
            requested_action={
                "block": "phy",
                "frequency_hz": 3.45e9,
                "bandwidth_hz": 20e6,
                "tx_power_dBm": 20.0,
                "papr_dB": 7.0,
            },
            mutates=frozenset({"papr_dB"}),
        ),
        policy,
    )
    assert not verdict.authorized
    assert any("widens scopes" in p for p in verdict.problems)


def test_delegation_may_not_widen_domains() -> None:
    policy = make_policy()
    verdict = authorize(
        AgentActionEnvelope(
            agent_id="rogue-widener",
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset({"spectrum"}),
            delegation_chain=(ROOT, "orchestrator", "rogue-widener"),
            requested_action={
                "block": "spectrum",
                "frequency_hz": 3.45e9,
                "bandwidth_hz": 20e6,
                "tx_power_dBm": 20.0,
            },
        ),
        policy,
    )
    assert not verdict.authorized
    assert any("widens domains" in p for p in verdict.problems)


def test_chain_may_not_loop() -> None:
    verdict = authorize(
        envelope(
            delegation_chain=(ROOT, "orchestrator", ROOT, "orchestrator", "spectrum-agent")
        ),
        make_policy(),
    )
    assert not verdict.authorized
    assert any("repeats a principal" in p for p in verdict.problems)


def test_empty_chain_is_refused() -> None:
    verdict = authorize(envelope(delegation_chain=()), make_policy())
    assert not verdict.authorized


def test_all_problems_are_reported_not_just_the_first() -> None:
    """A refusal that names one violation at a time is an oracle.

    Returning them all denies an attacker the ability to probe the policy one
    field at a time.
    """
    verdict = authorize(
        envelope(
            target_domain="core",
            granted_scopes=frozenset({"spectrum", "slice"}),
            delegation_chain=("orchestrator", "spectrum-agent"),
        ),
        make_policy(),
    )
    assert not verdict.authorized
    assert len(verdict.problems) >= 3


def test_priority_is_read_from_policy_never_from_the_envelope() -> None:
    """An agent that can assert its own precedence has none.

    ``AgentActionEnvelope`` has no priority field at all, which is the
    strongest form of this guarantee: there is nothing for an agent to set.
    """
    assert not hasattr(envelope(), "priority")
    assert make_policy().priority_of("spectrum-agent") == 20


def test_scopes_cover_the_real_key_contract() -> None:
    """Drift guard against ``horizon_ric``.

    If the upstream action contract grows a key, this fails until a scope
    accounts for it. Without this, a new action field would be writable by
    nobody (silently breaking agents) or — worse, if the check were inverted —
    by everybody.
    """
    covered = set(ALWAYS_WRITABLE)
    for keys in SCOPE_ACTION_KEYS.values():
        covered |= set(keys)
    contract = set(REQUIRED_ACTION_KEYS) | set(OPTIONAL_ACTION_KEYS)
    assert contract - covered == set(), (
        "action keys with no scope: " f"{sorted(contract - covered)}"
    )
    assert covered - contract == set(), (
        "scopes grant keys the action contract does not define: "
        f"{sorted(covered - contract)}"
    )


def test_scopes_are_disjoint() -> None:
    """Two scopes granting the same key would make authority ambiguous."""
    seen: dict[str, str] = {}
    for scope, keys in SCOPE_ACTION_KEYS.items():
        for key in keys:
            assert key not in seen, f"{key} granted by both {seen.get(key)} and {scope}"
            seen[key] = scope


def test_policy_rejects_unknown_scopes() -> None:
    with pytest.raises(ValueError):
        AuthorityPolicy(
            root_principal=ROOT,
            grants={
                ROOT: AuthorityGrant(ROOT, frozenset({"ran"}), frozenset({"made-up"}), 0)
            },
        )


def test_policy_rejects_a_root_without_a_grant() -> None:
    with pytest.raises(ValueError):
        AuthorityPolicy(root_principal="ghost", grants={})
