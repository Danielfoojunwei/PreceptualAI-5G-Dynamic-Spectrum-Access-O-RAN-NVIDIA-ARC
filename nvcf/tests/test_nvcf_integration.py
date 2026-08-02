"""Unit coverage for the NVCF binding.

Run with::

    PYTHONPATH=src:agentic/src:ocudu/src:nvcf/src \\
        /home/user/venv/bin/python -m pytest nvcf/tests -q

Gate G10 covers the composed path against a real socket. These cover the
pieces individually, and the edges G10 does not reach.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]

from horizon_agentic.envelope import SCOPE_ACTION_KEYS  # noqa: E402
from horizon_nvcf import protocol  # noqa: E402
from horizon_nvcf.agent import (  # noqa: E402
    NvcfAgentBinding,
    NvcfHostedAgent,
    agent_id_for,
)
from horizon_nvcf.client import (  # noqa: E402
    MAX_INLINE_PAYLOAD_BYTES,
    NvcfClient,
    NvcfClientConfig,
)
from horizon_nvcf.protocol import Disposition, header

FN = "fn-0001"
VER = "ver-0001"


class Recorded:
    def __init__(self, steps: list[tuple[int, dict[str, str], bytes]]) -> None:
        self.steps = list(steps)
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, method: str, url: str, headers: Any, body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url))
        self.last_headers = dict(headers)
        return self.steps.pop(0) if self.steps else (500, {}, b"")


def client(steps: list[tuple[int, dict[str, str], bytes]], **kw: Any) -> NvcfClient:
    cfg = NvcfClientConfig(
        function_id=FN,
        version_id=VER,
        api_key="k",
        poll_interval_s=0.0,
        max_polls=kw.pop("max_polls", 4),
        deadline_s=kw.pop("deadline_s", 5.0),
    )
    return NvcfClient(cfg, Recorded(steps), sleep=lambda _s: None, **kw)


def binding(**kw: Any) -> NvcfAgentBinding:
    base = {
        "function_id": FN,
        "version_id": VER,
        "target_domain": "terrestrial",
        "granted_scopes": frozenset({"spectrum"}),
        "mutates": frozenset({"frequency_hz"}),
        "resource_id": "cell-1",
        "delegation_chain": ("operator-root", agent_id_for(FN, VER)),
    }
    base.update(kw)
    return NvcfAgentBinding(**base)  # type: ignore[arg-type]


def body(action: dict[str, Any], **extra: Any) -> bytes:
    return json.dumps({"action": action, **extra}).encode()


# ── protocol ────────────────────────────────────────────────────────────────


def test_paths_come_from_the_contract_not_from_us():
    c = protocol.contract()
    assert protocol.invoke_path("A", "B") == c["invoke"]["invokeFunction_1"][
        "path"
    ].replace("{functionId}", "A").replace("{versionId}", "B")
    assert protocol.invoke_path("A") == c["invoke"]["invokeFunction"]["path"].replace(
        "{functionId}", "A"
    )


def test_an_undeclared_header_raises_rather_than_being_sent():
    # A typo'd header against a real service is a silent no-op. This is what
    # turns that into a failure at the point of use.
    with pytest.raises(protocol.NvcfProtocolError):
        header("NVCF-MADE-UP")


def test_header_lookup_is_case_insensitive_but_returns_the_canonical_name():
    assert header("nvcf-reqid") == "NVCF-REQID"


def test_a_status_declared_for_invoke_but_not_poll_is_unknown_on_poll():
    # 429 is declared on the invocation endpoints and NOT on the status
    # endpoint. Classifying it as "throttled" on a poll would be inventing a
    # meaning for a response the endpoint was never specified to return.
    assert protocol.classify(429, "invoke") == Disposition.THROTTLED
    assert protocol.classify(429, "poll") == Disposition.UNKNOWN


# ── client ──────────────────────────────────────────────────────────────────


def test_202_then_200_resolves_and_counts_polls():
    rid = header("NVCF-REQID")
    c = client([(202, {rid: "r1"}, b""), (200, {rid: "r1"}, body({"a": 1}))])
    out = c.invoke({"x": 1})
    assert out.ok and out.polls == 1 and out.request_id == "r1"


def test_202_without_a_request_id_is_unknown_not_pending():
    out = client([(202, {}, b"")]).invoke({"x": 1})
    assert out.disposition == Disposition.UNKNOWN
    assert out.payload is None


def test_poll_exhaustion_yields_pending_with_no_payload():
    rid = header("NVCF-REQID")
    out = client([(202, {rid: "r"}, b"")] * 6, max_polls=3).invoke({"x": 1})
    assert out.disposition == Disposition.PENDING
    assert out.payload is None and out.polls == 3


def test_302_is_not_followed():
    out = client([(302, {"Location": "https://evil.example/x"}, b"")]).invoke({"x": 1})
    assert out.disposition == Disposition.REDIRECT
    assert out.payload is None
    assert "will not follow a redirect" in out.detail


def test_oversized_payload_is_refused_not_truncated():
    out = client([]).invoke({"blob": "x" * (MAX_INLINE_PAYLOAD_BYTES + 1)})
    assert out.disposition == Disposition.REFUSED
    assert "Refusing rather than truncating" in out.detail


def test_transport_failure_is_refused_not_raised():
    def boom(*_a: Any, **_kw: Any) -> Any:
        raise OSError("connection reset")

    c = NvcfClient(
        NvcfClientConfig(function_id=FN, version_id=VER), boom, sleep=lambda _s: None
    )
    out = c.invoke({"x": 1})
    assert out.disposition == Disposition.REFUSED and out.payload is None


def test_the_versioned_path_is_used_when_a_version_is_configured():
    c = client([(200, {}, body({"a": 1}))])
    c.invoke({"x": 1})
    assert f"/functions/{FN}/versions/{VER}" in c._transport.calls[0][1]  # type: ignore[attr-defined]


def test_poll_seconds_header_is_sent_under_its_contract_name():
    c = client([(200, {}, body({"a": 1}))])
    c.invoke({"x": 1})
    assert header("NVCF-POLL-SECONDS") in c._transport.last_headers  # type: ignore[attr-defined]


# ── agent binding ───────────────────────────────────────────────────────────


def test_binding_without_a_version_is_rejected():
    with pytest.raises(ValueError, match="version_id is required"):
        binding(version_id="")


def test_binding_cannot_claim_to_mutate_outside_its_scopes():
    with pytest.raises(ValueError, match="mutates"):
        binding(granted_scopes=frozenset({"slice"}), mutates=frozenset({"frequency_hz"}))


def test_agent_refuses_a_client_pointed_at_a_different_version():
    c = NvcfClient(
        NvcfClientConfig(function_id=FN, version_id="other"),
        Recorded([]),
        sleep=lambda _s: None,
    )
    with pytest.raises(ValueError, match="behind the operator's back"):
        NvcfHostedAgent(c, binding())


def test_out_of_scope_keys_are_dropped_rather_than_failing_the_proposal():
    # Dropping is right: written_keys is what authorize() grades, so passing
    # an out-of-scope key would refuse the WHOLE envelope because a model
    # added an annotation.
    env, rej = NvcfHostedAgent(
        client([(200, {}, body({"frequency_hz": 3.5e9, "prb_allocation": 0.5}))]),
        binding(),
    ).propose({}, block="spectrum")
    assert rej is None and env is not None
    assert "prb_allocation" not in env.requested_action
    assert env.requested_action["frequency_hz"] == 3.5e9


def test_out_of_range_values_are_kept_for_the_shield_to_grade():
    # The opposite of the above: an out-of-RANGE value is the model asking
    # for something, and the Shield is what must answer. Dropping it here
    # would hide the request from the thing that exists to judge it.
    env, _ = NvcfHostedAgent(
        client([(200, {}, body({"frequency_hz": 1.0e12}))]), binding()
    ).propose({}, block="spectrum")
    assert env is not None and env.requested_action["frequency_hz"] == 1.0e12


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, None])
def test_unusable_values_refuse_the_proposal(bad: Any):
    env, rej = NvcfHostedAgent(
        client([(200, {}, json.dumps({"action": {"frequency_hz": bad}}).encode())]),
        binding(),
    ).propose({}, block="spectrum")
    assert env is None and rej is not None


def test_a_proposal_of_only_metadata_is_not_an_action():
    env, rej = NvcfHostedAgent(
        client([(200, {}, body({"model": "x", "confidence": 0.9}))]), binding()
    ).propose({}, block="spectrum")
    assert env is None and rej is not None


def test_agent_id_binds_function_and_version():
    assert agent_id_for("f", "v") == "nvcf:f/v"
    with pytest.raises(ValueError):
        agent_id_for("f", "")


def test_every_granted_scope_is_a_real_scope():
    with pytest.raises(ValueError, match="unknown scope"):
        binding(granted_scopes=frozenset({"not-a-scope"}), mutates=frozenset())
    assert set(binding().granted_scopes) <= set(SCOPE_ACTION_KEYS)


# ── contract drift ──────────────────────────────────────────────────────────


def test_committed_contract_matches_a_fresh_extraction():
    rc = subprocess.run(
        [sys.executable, str(REPO / "nvcf" / "extract_contract.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr


def test_ci_installs_every_extra_this_subtree_needs():
    # The ocudu job once passed locally and failed in CI because asn1tools
    # lives in the `oran` extra, not `dev`. G10 imports horizon_ocudu, which
    # needs it. This is the guard that learned from that.
    wf = (REPO / ".github" / "workflows" / "nvcf.yml").read_text(encoding="utf-8")
    assert "'.[dev,oran]'" in wf, "nvcf.yml must install the oran extra"
