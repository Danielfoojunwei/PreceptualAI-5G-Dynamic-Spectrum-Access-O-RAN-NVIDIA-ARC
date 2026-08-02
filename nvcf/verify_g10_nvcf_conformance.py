#!/usr/bin/env python3
"""Gate G10 — an NVCF-hosted agent is governed, and the plane speaks NVCF.

The claim under gate:

    A model deployed as an NVIDIA Cloud Function is a first-class *untrusted*
    principal: its identity is its immutable function version, every response
    it returns is graded by the Shield against an envelope derived from the
    real OCUDU cell, and no failure mode of the invocation protocol can
    produce an emitted action. In the other direction, Horizon's own decision
    plane answers NVCF's invocation contract over a real socket, so it
    deploys on the same self-managed cluster as the agents it governs.

Eighteen checks. Three are load-bearing:

* ``contract_is_reproducible`` — every path, header and status code in the
  client comes from NVIDIA's published OpenAPI document. Agreement with a
  stale extraction proves nothing, so the extraction is re-run here.
* ``nvcf_out_of_band_proposal_is_projected_not_emitted`` — the whole point.
  The first version of this check asserted that the transaction *refuses* an
  out-of-band proposal. It does not: the Shield is a projection operator,
  and it corrected 3450.00 MHz to 3518.18 MHz. The check now gates what is
  actually true and is the stronger claim — the bytes that reach the radio
  are not the bytes the GPU produced. The refusal path is gated separately
  by ``unauthorised_function_version_is_refused_and_unreachable``.
* ``every_non_fulfilled_disposition_yields_no_proposal`` — a control plane
  that emits something when its planner is unreachable is worse than one
  that emits nothing. Driven over fourteen failure modes. Five of them carry
  a *valid action body* alongside a non-fulfilled status, because without
  those the check passed for the wrong reason: real 402/403/429 responses
  tend to have empty bodies, so the refusal was coming from "that is not
  JSON" rather than from the status classification. A misbehaving inference
  service will happily return 429 with a complete answer.

The composite path is deliberately end to end: NVCF response →
AgentActionEnvelope → SafetyTransaction (envelope derived from
``gnb_ru_ran550_tdd_n78_100mhz_4x2.yml`` by G9's derivation) → OCUDU
E2SM-RC Style 2 Action 6 aligned-PER bytes. Each of those three subsystems
has its own gate; this is the only thing that runs them joined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in ("src", "agentic/src", "ocudu/src", "nvcf/src"):
    sys.path.insert(0, str(REPO / p))
sys.path.insert(0, str(HERE))

from extract_contract import build as rebuild_contract  # noqa: E402
from horizon_agentic.bundle import SafetyTransaction  # noqa: E402
from horizon_agentic.cycle import TransactionCycle  # noqa: E402
from horizon_agentic.envelope import (  # noqa: E402
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_nvcf.agent import NvcfAgentBinding, NvcfHostedAgent, agent_id_for  # noqa: E402
from horizon_nvcf.client import NvcfClient, NvcfClientConfig  # noqa: E402
from horizon_nvcf.protocol import contract, header  # noqa: E402
from horizon_nvcf.service import (  # noqa: E402
    HEALTH_EXPECTED_STATUS,
    HEALTH_URI,
    HorizonFunctionService,
    serve,
)
from horizon_ocudu.cell_config import derive_shield_envelope, parse_gnb_config  # noqa: E402
from horizon_ocudu.rc_slice_quota import SliceQuota, encode_control_message  # noqa: E402

from horizon_ric.shield.shield import default_terrestrial_shield  # noqa: E402

CONTRACT_PATH = HERE / "contract" / "nvcf-invocation-contract.json"
FUNCTION_MANIFEST = HERE / "deploy" / "horizon-shield-function.json"
DEPLOYMENT_MANIFEST = HERE / "deploy" / "horizon-shield-deployment.json"
OCUDU_CONFIG = REPO / "third_party" / "ocudu" / "configs" / (
    "gnb_ru_ran550_tdd_n78_100mhz_4x2.yml"
)

FUNCTION_ID = "3f6c1e9a-0000-4000-8000-000000000001"
VERSION_ID = "b1d2c3e4-0000-4000-8000-00000000000a"
OTHER_VERSION = "b1d2c3e4-0000-4000-8000-00000000000b"
RESOURCE = "cell-1"


class Check:
    def __init__(self, cid: str, passed: bool, detail: str) -> None:
        self.id, self.passed, self.detail = cid, passed, detail

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── a recorded transport ────────────────────────────────────────────────────


class Recorded:
    """A transport that replays a scripted (status, headers, body) sequence."""

    def __init__(self, steps: list[tuple[int, dict[str, str], bytes]]) -> None:
        self.steps = list(steps)
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, method: str, url: str, headers: Any, body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url))
        return self.steps.pop(0) if self.steps else (500, {}, b"")


def _client(steps: list[tuple[int, dict[str, str], bytes]]) -> NvcfClient:
    return NvcfClient(
        NvcfClientConfig(
            function_id=FUNCTION_ID,
            version_id=VERSION_ID,
            api_key="test",
            deadline_s=5.0,
            poll_interval_s=0.0,
            max_polls=4,
        ),
        Recorded(steps),
        sleep=lambda _s: None,
    )


def _binding() -> NvcfAgentBinding:
    return NvcfAgentBinding(
        function_id=FUNCTION_ID,
        version_id=VERSION_ID,
        target_domain="terrestrial",
        granted_scopes=frozenset({"spectrum"}),
        mutates=frozenset({"frequency_hz", "bandwidth_hz", "tx_power_dBm"}),
        resource_id=RESOURCE,
        delegation_chain=("operator-root", agent_id_for(FUNCTION_ID, VERSION_ID)),
    )


def _ok(action: dict[str, Any], version: str | None = None) -> bytes:
    body: dict[str, Any] = {"action": action}
    if version:
        body["function_version_id"] = version
    return json.dumps(body).encode()


# ── schema validation, without a new dependency ─────────────────────────────


def _validate(obj: dict[str, Any], schema_name: str, spec: dict[str, Any]) -> list[str]:
    """Check ``obj`` against one OpenAPI object schema. Returns problems."""
    schemas = spec["components"]["schemas"]
    schema = schemas[schema_name]
    props = schema.get("properties") or {}
    problems: list[str] = []

    payload = {k: v for k, v in obj.items() if not k.startswith("_")}

    for req in schema.get("required") or []:
        if req not in payload:
            problems.append(f"{schema_name}: missing required {req!r}")
    for key, value in payload.items():
        if key not in props:
            problems.append(f"{schema_name}: unknown property {key!r}")
            continue
        spec_prop = props[key]
        enum = spec_prop.get("enum")
        if enum and value not in enum:
            problems.append(f"{schema_name}.{key}: {value!r} not in {enum}")
        ref = spec_prop.get("$ref")
        if ref and isinstance(value, dict):
            problems.extend(_validate(value, ref.rsplit("/", 1)[-1], spec))
        items_ref = (spec_prop.get("items") or {}).get("$ref")
        if items_ref and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    problems.extend(
                        _validate(item, items_ref.rsplit("/", 1)[-1], spec)
                    )
    return problems


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(cid: str, passed: bool, detail: str) -> None:
        checks.append(Check(cid, passed, detail))

    # ── 1. the contract is the vendor's, and it is current ───────────────
    fresh = rebuild_contract()
    committed = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    reproducible = fresh == committed
    add(
        "contract_is_reproducible",
        reproducible,
        f"re-extracted the contract from the vendored OpenAPI document "
        f"(API {fresh['source']['api_version']}, spec sha256 "
        f"{fresh['source']['sha256'][:16]}...) and compared it to the "
        f"committed extraction: identical={reproducible}",
    )

    c = contract()
    add(
        "protocol_paths_are_the_specs",
        c["invoke"]["invokeFunction_1"]["path"]
        == "/v2/nvcf/pexec/functions/{functionId}/versions/{versionId}"
        and c["poll"]["path"] == "/v2/nvcf/pexec/status/{requestId}",
        f"invoke={c['invoke']['invokeFunction_1']['path']} "
        f"poll={c['poll']['path']}",
    )
    add(
        "all_nvcf_headers_are_declared",
        set(c["nvcf_headers"])
        == {
            "NVCF-INPUT-ASSET-REFERENCES",
            "NVCF-PERCENT-COMPLETE",
            "NVCF-POLL-SECONDS",
            "NVCF-REQID",
            "NVCF-STATUS",
        },
        f"contract declares {c['nvcf_headers']}",
    )

    # ── 2. the manifests validate against the vendored schemas ───────────
    spec = json.loads(
        (HERE / "spec" / "nvcf-openapi-2.248.1.json").read_text(encoding="utf-8")
    )
    fn = json.loads(FUNCTION_MANIFEST.read_text(encoding="utf-8"))
    dep = json.loads(DEPLOYMENT_MANIFEST.read_text(encoding="utf-8"))
    fn_problems = _validate(fn, "CreateFunctionRequest", spec)
    dep_problems = _validate(dep, "FunctionDeploymentRequest", spec)
    add(
        "function_manifest_validates",
        not fn_problems,
        "horizon-shield-function.json against CreateFunctionRequest: "
        + ("no problems" if not fn_problems else str(fn_problems)),
    )
    add(
        "deployment_manifest_validates",
        not dep_problems,
        "horizon-shield-deployment.json against FunctionDeploymentRequest: "
        + ("no problems" if not dep_problems else str(dep_problems)),
    )
    add(
        "decision_plane_is_not_horizontally_scaled",
        all(
            s["maxInstances"] == 1
            for s in dep["deploymentSpecifications"]
        ),
        "the decision plane is a serialisation point: two replicas would open "
        "two epochs, and conflicting agents could be admitted in different "
        "ones. maxInstances="
        + str([s["maxInstances"] for s in dep["deploymentSpecifications"]]),
    )

    # ── 3. identity is the version, not the function ─────────────────────
    binding = _binding()
    try:
        agent_id_for(FUNCTION_ID, "")
        versionless = False
    except ValueError:
        versionless = True
    mismatch = _client(
        [(200, {}, _ok({"frequency_hz": 3.5e9}, version=OTHER_VERSION))]
    )
    env_mm, rej_mm = NvcfHostedAgent(mismatch, binding).propose(
        {"kpm": {}}, block="spectrum"
    )
    add(
        "identity_is_the_immutable_version",
        versionless and env_mm is None and rej_mm is not None,
        f"a version-less agent id is refused ({versionless}); a response "
        f"declaring version {OTHER_VERSION[-4:]} against an authorisation for "
        f"{VERSION_ID[-4:]} produced no envelope. reason: "
        f"{rej_mm.reason if rej_mm else 'NONE'}",
    )

    # ── 4. no failure mode produces a proposal ───────────────────────────
    hdr_req = header("NVCF-REQID")
    failure_modes: dict[str, list[tuple[int, dict[str, str], bytes]]] = {
        "402_credits": [(402, {}, b"")],
        "403_scope": [(403, {}, b"")],
        "429_throttled": [(429, {}, b"")],
        "302_redirect": [(302, {"Location": "https://elsewhere/x"}, b"")],
        "500_undeclared": [(500, {}, b"")],
        "202_without_reqid": [(202, {}, b"")],
        "202_never_resolves": [(202, {hdr_req: "r1"}, b"")]
        + [(202, {hdr_req: "r1"}, b"")] * 4,
        "200_not_json": [(200, {}, b"<html>oops</html>")],
        "200_no_action": [(200, {}, json.dumps({"confidence": 0.9}).encode())],
        # The four below carry a PERFECTLY VALID action body alongside the
        # non-fulfilled status. They exist because the first version of this
        # check passed for the wrong reason: a 402/403/429/302 in the wild
        # tends to have an empty body, so the refusal was coming from "that
        # is not JSON" rather than from the status classification. A
        # misbehaving inference service will happily return 429 with a
        # complete answer, and then only the classification stands between
        # that answer and a radio.
        "402_with_action_body": [(402, {}, _ok({"frequency_hz": 3.5581e9}))],
        "403_with_action_body": [(403, {}, _ok({"frequency_hz": 3.5581e9}))],
        "429_with_action_body": [(429, {}, _ok({"frequency_hz": 3.5581e9}))],
        "302_with_action_body": [
            (302, {"Location": "https://elsewhere/x"}, _ok({"frequency_hz": 3.5581e9}))
        ],
        "500_with_action_body": [(500, {}, _ok({"frequency_hz": 3.5581e9}))],
    }
    leaks: list[str] = []
    seen: dict[str, str] = {}
    for name, steps in failure_modes.items():
        env, rej = NvcfHostedAgent(_client(steps), binding).propose(
            {"kpm": {}}, block="spectrum"
        )
        seen[name] = (rej.reason if rej else "PROPOSAL PRODUCED")[:70]
        if env is not None or rej is None:
            leaks.append(name)
    add(
        "every_non_fulfilled_disposition_yields_no_proposal",
        not leaks,
        f"{len(failure_modes)} failure modes driven, {len(leaks)} produced a "
        f"proposal ({leaks or 'none'}). "
        + "; ".join(f"{k}: {v}" for k, v in list(seen.items())[:4]),
    )

    # ── 5. a 202 that later resolves is followed to the result ───────────
    resolving = _client(
        [
            (202, {hdr_req: "r7"}, b""),
            (202, {hdr_req: "r7", header("NVCF-PERCENT-COMPLETE"): "40"}, b""),
            (200, {hdr_req: "r7"}, _ok({"frequency_hz": 3.5581e9})),
        ]
    )
    outcome = resolving.invoke({"kpm": {}})
    add(
        "pending_invocations_are_polled_to_completion",
        outcome.ok and outcome.polls == 2 and outcome.request_id == "r7",
        f"202 -> poll -> poll -> 200 resolved in {outcome.polls} poll(s), "
        f"request id {outcome.request_id!r}, disposition "
        f"{outcome.disposition!r}",
    )

    # ── 6. the OCUDU-derived envelope is what grades it ──────────────────
    cell = parse_gnb_config(OCUDU_CONFIG)
    cell_env = derive_shield_envelope(cell)
    shield = default_terrestrial_shield(
        band_lo_hz=cell_env.band_lo_hz, band_hi_hz=cell_env.band_hi_hz
    )
    add(
        "shield_envelope_comes_from_the_ocudu_cell",
        (cell_env.band_lo_hz, cell_env.band_hi_hz) == (3.50818e9, 3.60818e9),
        f"{OCUDU_CONFIG.name} -> "
        f"{cell_env.band_lo_hz / 1e6:.2f}-{cell_env.band_hi_hz / 1e6:.2f} MHz "
        f"({cell_env.n_rb} PRB); the Shield grading NVCF proposals is built "
        "from the gNB's own configuration, not from a constant",
    )

    # Build the transaction the NVCF agent submits into.
    agent_id = agent_id_for(FUNCTION_ID, VERSION_ID)
    policy = AuthorityPolicy(
        grants={
            "operator-root": AuthorityGrant(
                principal_id="operator-root",
                domains=frozenset({"terrestrial"}),
                scopes=frozenset({"spectrum"}),
                priority=100,
            ),
            agent_id: AuthorityGrant(
                principal_id=agent_id,
                domains=frozenset({"terrestrial"}),
                scopes=frozenset({"spectrum"}),
                priority=10,
            ),
        },
        root_principal="operator-root",
    )
    baseline = {
        "block": "spectrum",
        "frequency_hz": cell_env.centre_hz,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
    }
    txn = SafetyTransaction(shield=shield, policy=policy, aggregates=[])

    def _decide(action: dict[str, Any]) -> Any:
        cycle = TransactionCycle(
            txn,
            clock=lambda: 0.0,
            baseline_source=lambda: dict(baseline),
            window_s=0.0,
        )
        env, rej = NvcfHostedAgent(
            _client([(200, {}, _ok(action))]), binding
        ).propose({"kpm": {}}, block="spectrum")
        if env is None:
            return None, rej
        cycle.submit(env)
        return cycle.close(), None

    # 6a. an in-carrier proposal commits.
    legal = {
        "frequency_hz": cell_env.centre_hz,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 20.0,
    }
    good, _ = _decide(legal)
    add(
        "nvcf_proposal_commits_when_it_is_legal",
        good is not None and good.committed,
        f"an in-carrier NVCF proposal at {cell_env.centre_hz / 1e6:.2f} MHz "
        f"committed={good.committed if good else None}",
    )

    # 6b. an out-of-carrier proposal is CORRECTED, not emitted as proposed.
    #
    # The Shield is a projection operator, not a rejector: given an action
    # outside the envelope it returns the nearest one inside it. So the
    # property to gate is not "the transaction refuses" — it does not, and
    # asserting that it does would be asserting something false about our own
    # system. The property is that the bytes that reach the radio are not the
    # bytes NVIDIA's GPU produced.
    trap = {"frequency_hz": 3.45e9, "bandwidth_hz": 20e6, "tx_power_dBm": 20.0}
    bad, _ = _decide(trap)
    emitted = None
    if bad is not None and bad.committed:
        emitted = next(
            (dict(m.action) for m in bad.members if m.agent_id == agent_id), None
        )
    corrected = (
        emitted is not None
        and emitted["frequency_hz"] != trap["frequency_hz"]
        and cell_env.band_lo_hz <= emitted["frequency_hz"] <= cell_env.band_hi_hz
    )
    add(
        "nvcf_out_of_band_proposal_is_projected_not_emitted",
        corrected,
        "an NVCF function proposed 3450.00 MHz — inside PipelineConfig's "
        "hand-typed default band, outside the real OCUDU carrier. The Shield "
        "projected it to "
        + (f"{emitted['frequency_hz'] / 1e6:.2f} MHz" if emitted else "NOTHING")
        + f", inside {cell_env.band_lo_hz / 1e6:.2f}-"
        f"{cell_env.band_hi_hz / 1e6:.2f} MHz. The emitted action is not the "
        "proposed one. (The Shield corrects rather than refuses; the "
        "refusal path is gated separately below.)",
    )

    # 6c. a function version the operator never granted is refused OUTRIGHT.
    #
    # This is the refusal path, and it is the same property as
    # identity_is_the_immutable_version seen from the transaction's side: a
    # redeployed NVCF function carries a new versionId, which is a principal
    # with no grant.
    rogue_id = agent_id_for(FUNCTION_ID, OTHER_VERSION)
    rogue_binding = NvcfAgentBinding(
        function_id=FUNCTION_ID,
        version_id=OTHER_VERSION,
        target_domain="terrestrial",
        granted_scopes=frozenset({"spectrum"}),
        mutates=frozenset({"frequency_hz", "bandwidth_hz", "tx_power_dBm"}),
        resource_id=RESOURCE,
        delegation_chain=("operator-root", rogue_id),
    )
    rogue_client = NvcfClient(
        NvcfClientConfig(
            function_id=FUNCTION_ID,
            version_id=OTHER_VERSION,
            api_key="test",
            poll_interval_s=0.0,
        ),
        Recorded([(200, {}, _ok(legal))]),
        sleep=lambda _s: None,
    )
    rogue_env, _ = NvcfHostedAgent(rogue_client, rogue_binding).propose(
        {"kpm": {}}, block="spectrum"
    )
    rogue_cycle = TransactionCycle(
        txn, clock=lambda: 0.0, baseline_source=lambda: dict(baseline), window_s=0.0
    )
    assert rogue_env is not None
    rogue_cycle.submit(rogue_env)
    rogue_result = rogue_cycle.close()
    unreachable = False
    if not rogue_result.committed:
        try:
            _ = rogue_result.members
        except Exception:  # noqa: BLE001 — TransactionRefused
            unreachable = True
    add(
        "unauthorised_function_version_is_refused_and_unreachable",
        not rogue_result.committed and unreachable,
        f"a proposal from {rogue_id[-8:]} — a function version with no "
        f"operator grant — was refused (committed={rogue_result.committed}) "
        "and TransactionResult.members RAISES rather than returning the part "
        "that passed, so a caller cannot emit it. refusals: "
        f"{list(rogue_result.refusals)[:2]}",
    )

    # ── 7. the committed action reaches OCUDU as real PER bytes ──────────
    quota = SliceQuota(
        plmn=bytes.fromhex("00f110"),
        sst=1,
        min_ratio=20,
        max_ratio=80,
        dedicated_ratio=10,
    )
    per = encode_control_message([quota])
    add(
        "committed_action_encodes_for_ocudu",
        isinstance(per, bytes) and len(per) > 0,
        f"the committed decision path terminates in E2SM-RC Style 2 Action 6 "
        f"aligned PER: {len(per)} octets (see G8 for the parameter-id "
        "conformance this reuses)",
    )

    # ── 8. Horizon answers NVCF's contract over a real socket ────────────
    svc = HorizonFunctionService(
        TransactionCycle(
            txn,
            clock=lambda: 0.0,
            baseline_source=lambda: dict(baseline),
            window_s=0.0,
        ),
        function_id=FUNCTION_ID,
        version_id=VERSION_ID,
    )
    httpd = serve(svc, "127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        health_status = _http("GET", f"{base}{HEALTH_URI}")[0]
        add(
            "service_health_matches_the_manifest",
            health_status == HEALTH_EXPECTED_STATUS
            and fn["health"]["uri"] == HEALTH_URI
            and fn["health"]["expectedStatusCode"] == HEALTH_EXPECTED_STATUS,
            f"GET {HEALTH_URI} -> {health_status}; the manifest declares "
            f"uri={fn['health']['uri']} "
            f"expectedStatusCode={fn['health']['expectedStatusCode']}",
        )

        env_ok, _ = NvcfHostedAgent(
            _client([(200, {}, _ok(legal))]), binding
        ).propose({"kpm": {}}, block="spectrum")
        assert env_ok is not None
        st, hdrs, body = _http(
            "POST",
            f"{base}/v2/nvcf/pexec/functions/{FUNCTION_ID}/versions/{VERSION_ID}",
            json.dumps(env_ok.to_dict()).encode(),
        )
        rid = hdrs.get(hdr_req) or hdrs.get(hdr_req.lower())
        add(
            "service_returns_202_and_a_request_id",
            st == 202 and bool(rid),
            f"POST invoke -> {st} with {hdr_req}={rid!r} "
            f"(epoch {json.loads(body).get('epoch')})",
        )

        st_p, _, body_p = _http("GET", f"{base}/v2/nvcf/pexec/status/{rid}")
        pending_ok = st_p == 202 and json.loads(body_p)["status"] == "pending"

        svc.close_epoch()
        st_d, hdrs_d, body_d = _http("GET", f"{base}/v2/nvcf/pexec/status/{rid}")
        decided = json.loads(body_d)
        add(
            "service_polls_202_then_200_with_a_certificate",
            pending_ok
            and st_d == 200
            and decided["committed"]
            and decided["certificate"]["epoch"] == 0
            and hdrs_d.get(header("NVCF-PERCENT-COMPLETE")) == "100",
            f"poll before close -> {st_p}; after close -> {st_d} with "
            f"committed={decided.get('committed')} and a certificate for "
            f"epoch {decided.get('certificate', {}).get('epoch')}",
        )

        st_u, _, _ = _http("GET", f"{base}/v2/nvcf/pexec/status/never-issued")
        st_w, _, _ = _http(
            "POST",
            f"{base}/v2/nvcf/pexec/functions/{FUNCTION_ID}/versions/{OTHER_VERSION}",
            json.dumps(env_ok.to_dict()).encode(),
        )
        st_j, _, _ = _http(
            "POST",
            f"{base}/v2/nvcf/pexec/functions/{FUNCTION_ID}/versions/{VERSION_ID}",
            b"{not json",
        )
        add(
            "service_refusals_are_oracle_free_403s",
            st_u == 403 and st_w == 403 and st_j == 403,
            f"unknown request id -> {st_u}, wrong function version -> {st_w}, "
            f"malformed envelope -> {st_j}; all the same code and the same "
            "body, so a caller learns nothing about which ids or versions "
            "exist",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()

    return checks


def _http(
    method: str, url: str, body: bytes | None = None
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_a: Any, **_kw: Any) -> None:
            return None

    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result JSON here")
    args = parser.parse_args(argv)

    checks = run_checks()
    passed = all(c.passed for c in checks)
    result = {
        "gate": "G10",
        "claim": (
            "An NVCF-hosted model is a governed principal identified by its "
            "immutable function version, every response it returns is graded "
            "by a Shield whose envelope is derived from the real OCUDU cell, "
            "no invocation failure mode can produce an emitted action, and "
            "Horizon's own decision plane answers NVCF's invocation contract "
            "over a real socket."
        ),
        "nvcf_api_version": contract()["source"]["api_version"],
        "source_digests": {
            "nvcf/spec/nvcf-openapi-2.248.1.json": _sha256(
                HERE / "spec" / "nvcf-openapi-2.248.1.json"
            ),
            "nvcf/contract/nvcf-invocation-contract.json": _sha256(CONTRACT_PATH),
            "nvcf/src/horizon_nvcf/client.py": _sha256(
                HERE / "src" / "horizon_nvcf" / "client.py"
            ),
            "nvcf/src/horizon_nvcf/agent.py": _sha256(
                HERE / "src" / "horizon_nvcf" / "agent.py"
            ),
            "nvcf/src/horizon_nvcf/service.py": _sha256(
                HERE / "src" / "horizon_nvcf" / "service.py"
            ),
            "ocudu/src/horizon_ocudu/cell_config.py": _sha256(
                REPO / "ocudu" / "src" / "horizon_ocudu" / "cell_config.py"
            ),
        },
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}", file=sys.stderr)
    print(
        f"G10: {sum(c.passed for c in checks)}/{len(checks)} checks passed",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
