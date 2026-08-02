#!/usr/bin/env python3
"""Drive the A1 ``assurance`` envelope across a real O-RAN A1 socket.

Starts the **vendored, unmodified** O-RAN-SC A1 interface simulator
(``third_party/sim-a1-interface``, ``near-rt-ric-simulator/src/OSC_2.1.0``) as a
subprocess on a free loopback port, then exercises the certificate-on-the-wire
path against it with Horizon's ``osc_a1`` dialect:

    1. register the four Horizon policy types, carrying the schema-1.1.0
       ``create_schema`` that declares the ``assurance`` envelope;
    2. read the registered type back off the simulator and confirm the
       simulator is holding the extended schema, not ours;
    3. generate a real Ed25519 signing key;
    4. run a real action through a real ``default_terrestrial_shield`` and sign
       the resulting SafetyCertificate;
    5. PUT a policy whose body carries the assurance envelope;
    6. GET the policy back and compare the envelope, canonical byte for
       canonical byte, with what was sent;
    7. verify the returned signature against the returned digest with the
       public key;
    8. GET the A1AP §6.5 policy status and record ``enforceStatus``;
    9. NEGATIVE case — under ``HORIZON_A1_REQUIRE_CERT=1`` a Shield-blocked
       decision must be refused before any HTTP happens, and nothing must
       appear on the simulator;
   10. CONTROL case — register a second policy type carrying the *pre-1.1.0*
       schema and PUT the same envelope-bearing body at it. The simulator
       validates every PUT against the registered ``create_schema``
       (``a1_mediator_controller.a1_controller_create_or_replace_policy_instance``),
       so this must be a 400. Without it, "the schema accepted the envelope"
       would prove nothing about whether anything was validating at all.

The simulator is torn down in a ``finally`` block. Its git commit is recorded
from ``git -C third_party/sim-a1-interface rev-parse HEAD`` so a verifier can
confirm the run used the real vendored software and not a stand-in.

There is no Docker daemon in this environment, so the simulator is started
natively rather than from its published image. It is byte-identical source at a
pinned commit; ``scripts/verify_a1_assurance_wire.py`` re-checks that commit.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature

from horizon_ric.assurance.profile import emit_profile, profile_digest
from horizon_ric.rapp.a1_adapter import (
    DEFAULT_POLICY_TYPES,
    A1Adapter,
    A1AdapterConfig,
    assurance_envelope,
)
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.shield.signing import (
    canonical_certificate_bytes,
    generate_signing_key,
    key_fingerprint,
    signed_certificate,
    verify_certificate,
)

# Relative to the repository root.
SIM_SUBMODULE = "third_party/sim-a1-interface"
SIM_INTERFACE = "OSC_2.1.0"

# Deterministic identifiers so a committed proof diffs cleanly run to run.
# Everything genuinely per-run (the Ed25519 key, ``issued_at``, and therefore
# the certificate digest and signature) is recorded but never compared.
DECISION_ID = "a1-assurance-wire-proof-0001"
POLICY_ID = "horizon-a1-assurance-proof"
BLOCKED_POLICY_ID = "horizon-a1-assurance-proof-blocked"
CONTROL_POLICY_TYPE_ID = 29001
CONTROL_POLICY_ID = "horizon-a1-assurance-proof-control"

PROOF_POLICY_TYPE = "horizon.qos.priority"

# The physics the planner is licensed to command (mirrors
# ``DecisionPipeline._shield_action``).
BAND_LO_HZ = 3.40e9
BAND_HI_HZ = 3.50e9
MAX_EIRP_DBM = 33.0


def _canonical(obj: Any) -> bytes:
    """The canonical JSON form used for every byte-identity claim here."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_proxy_for_loopback() -> None:
    """Keep loopback traffic off any ambient HTTP proxy.

    httpx honours ``HTTP_PROXY``/``NO_PROXY``; a proxied 127.0.0.1 would fail
    in a way that looks like the simulator never came up.
    """
    for name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(name, "")
        hosts = [h for h in current.split(",") if h.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in hosts:
                hosts.append(host)
        os.environ[name] = ",".join(hosts)


def _git_commit(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# the real simulator, started natively
# ---------------------------------------------------------------------------
# main.py hardcodes port 2222: its `isinstance(sys.argv[1], int)` guard is
# always False because argv entries are strings, so the documented port
# argument is dead. We import the module and run its connexion app ourselves on
# the port we chose. No simulator source is modified.
_LAUNCHER = (
    "import sys; import main; "
    "main.app.run(port=int(sys.argv[1]), host='127.0.0.1')"
)


class Simulator:
    """The vendored OSC A1 simulator as a child process."""

    def __init__(self, repo_root: Path, port: int, log_path: Path) -> None:
        sim_root = repo_root / SIM_SUBMODULE / "near-rt-ric-simulator"
        self.repo_root = repo_root
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.sim_root = sim_root
        self.cwd = sim_root / "src" / SIM_INTERFACE
        self.api_path = sim_root / "api" / SIM_INTERFACE
        self.common_path = sim_root / "src" / "common"
        self.log_path = log_path
        self.argv = [sys.executable, "-c", _LAUNCHER, str(port)]
        self.proc: subprocess.Popen[bytes] | None = None
        self.readiness_wait_s = 0.0
        self.healthcheck_status: int | None = None

    def start(self, timeout_s: float = 60.0) -> None:
        for required in (self.cwd / "main.py", self.api_path / "openapi.yaml"):
            if not required.exists():
                raise FileNotFoundError(
                    f"vendored simulator is not checked out: {required} is missing. "
                    f"Run `git submodule update --init {SIM_SUBMODULE}`."
                )
        env = dict(os.environ)
        # Exactly what near-rt-ric-simulator/src/start.sh exports.
        env["APIPATH"] = str(self.api_path)
        env["PYTHONPATH"] = str(self.common_path)
        env["PYTHONUNBUFFERED"] = "1"
        self._log = self.log_path.open("wb")
        self.proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            self.argv,
            cwd=str(self.cwd),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"simulator exited early with code {self.proc.returncode}; "
                    f"log:\n{self.tail_log()}"
                )
            try:
                resp = httpx.get(
                    f"{self.base_url}/a1-p/healthcheck", timeout=2.0, trust_env=False
                )
            except httpx.HTTPError:
                time.sleep(0.25)
                continue
            if resp.status_code == 200:
                self.readiness_wait_s = round(time.monotonic() - t0, 3)
                self.healthcheck_status = resp.status_code
                return
            time.sleep(0.25)
        raise TimeoutError(
            f"simulator did not become healthy on {self.base_url} within "
            f"{timeout_s}s; log:\n{self.tail_log()}"
        )

    def tail_log(self, lines: int = 20) -> str:
        with contextlib.suppress(OSError):
            return "\n".join(
                self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -lines:
                ]
            )
        return ""

    def stop(self) -> int | None:
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover — dev server is docile
                self.proc.kill()
                self.proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self._log.close()
        return self.proc.returncode

    def describe(self) -> dict[str, Any]:
        return {
            "name": "O-RAN-SC sim-a1-interface near-rt-ric-simulator",
            "kind": "real_vendored_submodule",
            "fallback_used": False,
            "fallback_reason": None,
            "upstream": "https://gerrit.o-ran-sc.org/r/sim/a1-interface",
            "interface": SIM_INTERFACE,
            "submodule_path": SIM_SUBMODULE,
            "commit": _git_commit(self.repo_root / SIM_SUBMODULE),
            "commit_command": f"git -C {SIM_SUBMODULE} rev-parse HEAD",
            "source_modified": False,
            "launch": {
                "argv": list(self.argv),
                "cwd": str(self.cwd.relative_to(self.repo_root)),
                "env": {
                    "APIPATH": str(self.api_path.relative_to(self.repo_root)),
                    "PYTHONPATH": str(self.common_path.relative_to(self.repo_root)),
                },
                "note": (
                    "started natively (no Docker daemon in this environment); "
                    "src/start.sh exports the same APIPATH/PYTHONPATH and runs "
                    "the same main.py"
                ),
            },
            "port": self.port,
            "base_url": self.base_url,
            "healthcheck_status": self.healthcheck_status,
            "readiness_wait_s": self.readiness_wait_s,
        }


# ---------------------------------------------------------------------------
# the proof
# ---------------------------------------------------------------------------
def _shield():
    return default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )


def _action(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    action = {
        "block": "policy_emit",
        "policy_type": PROOF_POLICY_TYPE,
        "policy_payload": payload,
        "frequency_hz": (BAND_LO_HZ + BAND_HI_HZ) / 2.0,
        "bandwidth_hz": min(20e6, BAND_HI_HZ - BAND_LO_HZ),
        "tx_power_dBm": min(24.0, MAX_EIRP_DBM - 9.0),
        "antenna_gain_dBi": 5.0,
    }
    action.update(overrides)
    return action


def _base_payload() -> dict[str, Any]:
    return {
        "scope": {"slice_id": "slice-a1-assurance-proof"},
        "qos_objectives": {"priority": 5},
        "rapp_metadata": {"decision_id": DECISION_ID},
    }


async def _run(sim: Simulator, key_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    shield = _shield()
    digest_of_profile = profile_digest(
        emit_profile(
            shield,
            profile_id="a1-assurance-wire-proof/terrestrial",
            description=(
                f"band {BAND_LO_HZ:.0f}-{BAND_HI_HZ:.0f} Hz, "
                f"max EIRP {MAX_EIRP_DBM} dBm"
            ),
        )
    )

    adapter = A1Adapter(
        A1AdapterConfig(
            near_rt_ric_base_url=sim.base_url, dialect="osc_a1", timeout_seconds=10.0
        )
    )
    raw = httpx.AsyncClient(base_url=sim.base_url, timeout=10.0, trust_env=False)
    try:
        # 1. Register the policy types, carrying the schema-1.1.0 create schemas.
        accepted = await adapter.register_policy_types()
        expected_ids = [spec["policy_type_id"] for spec in DEFAULT_POLICY_TYPES.values()]
        type_id = int(DEFAULT_POLICY_TYPES[PROOF_POLICY_TYPE]["policy_type_id"])

        # 2. Read the type back off the simulator: the schema under test is now
        #    the simulator's, not ours.
        readback = await raw.get(f"/a1-p/policytypes/{type_id}")
        stored = readback.json() if readback.status_code == 200 else {}
        stored_schema = stored.get("create_schema", {})
        stored_assurance = stored_schema.get("properties", {}).get("assurance", {})
        result["policy_types"] = {
            "registered_ids": accepted,
            "expected_ids": expected_ids,
            "registration_complete": accepted == expected_ids,
            "schema_version_declared": DEFAULT_POLICY_TYPES[PROOF_POLICY_TYPE]["schema_v"],
            "readback": {
                "policy_type_id": type_id,
                "http_status": readback.status_code,
                "declares_assurance": bool(stored_assurance),
                "assurance_properties": sorted(stored_assurance.get("properties", {})),
                "assurance_required": sorted(stored_assurance.get("required", [])),
                "assurance_additional_properties": stored_assurance.get(
                    "additionalProperties"
                ),
                "assurance_in_top_level_required": "assurance"
                in stored_schema.get("required", []),
                "create_schema_sha256": _sha256(_canonical(stored_schema)),
            },
        }

        # 3. A real Ed25519 key, generated for this run.
        signing_key = generate_signing_key(key_dir / "a1-assurance-proof.pem")
        public_key = signing_key.public_key()
        result["signing"] = {
            "algorithm": "Ed25519",
            "key_source": "horizon_ric.shield.signing.generate_signing_key",
            "signing_key_fingerprint": key_fingerprint(public_key),
        }

        # 4. A real action through a real Shield, then signed.
        payload = _base_payload()
        disposition = shield.dispose(_action(payload), {}, decision_id=DECISION_ID)
        certificate = signed_certificate(disposition.certificate, signing_key)
        canonical = canonical_certificate_bytes(certificate)
        result["certificate"] = {
            "decision_id": certificate.decision_id,
            "loop_tier": certificate.loop_tier,
            "invariant_ids": [c.invariant_id for c in certificate.invariants],
            "safe": certificate.safe,
            "emit_blocked": certificate.emit_blocked,
            "projected": certificate.projected,
            "violated_ids": list(certificate.violated_ids),
            "min_margin_dB": certificate.min_margin_dB,
            "canonical_bytes_len": len(canonical),
            "canonical_bytes_sha256": _sha256(canonical),
            "signature": certificate.signature,
            "signing_key_fingerprint": certificate.signing_key_fingerprint,
            "self_verifies_locally": verify_certificate(certificate, public_key),
            "keys_on_the_wire": 8,
            "keys_in_the_certificate": len(certificate.to_dict()),
        }

        # 5. PUT the policy with the assurance envelope on it.
        envelope = assurance_envelope(certificate, profile_digest=digest_of_profile)
        body = {**payload, "assurance": envelope}
        emitted_id, put_status = await adapter.emit_policy(
            PROOF_POLICY_TYPE,
            body,
            policy_id=POLICY_ID,
            safety_certificate=certificate,
        )

        # 6. GET it back off the simulator and compare canonical bytes.
        fetched = await raw.get(f"/a1-p/policytypes/{type_id}/policies/{emitted_id}")
        returned_body = fetched.json() if fetched.status_code == 200 else {}
        returned_envelope = returned_body.get("assurance", {})
        sent_sha = _sha256(_canonical(envelope))
        returned_sha = _sha256(_canonical(returned_envelope))

        # 7. Verify the RETURNED signature over the object the RETURNED digest
        #    names. The signature is Ed25519 over the certificate's canonical
        #    bytes; the wire carries their SHA-256. So: confirm the bytes we
        #    hold hash to the digest that came back, then verify the returned
        #    signature over exactly those bytes.
        wire_digest = returned_envelope.get("certificate_digest")
        wire_signature = returned_envelope.get("signature")
        digest_matches = wire_digest == _sha256(canonical)
        ed25519_verified = False
        verify_error: str | None = None
        if isinstance(wire_signature, str):
            try:
                public_key.verify(bytes.fromhex(wire_signature), canonical)
                ed25519_verified = True
            except (InvalidSignature, ValueError) as exc:
                verify_error = type(exc).__name__
        signature_verified = bool(digest_matches and ed25519_verified)

        # 8. A1AP §6.5 policy status.
        status_resp = await raw.get(
            f"/a1-p/policytypes/{type_id}/policies/{emitted_id}/status"
        )
        status_body = status_resp.json() if status_resp.status_code == 200 else {}

        listed = await adapter.list_policies(PROOF_POLICY_TYPE)

        result["positive_case"] = {
            "policy_type": PROOF_POLICY_TYPE,
            "policy_type_id": type_id,
            "policy_id": emitted_id,
            "put_status": put_status,
            "sent_body": body,
            "sent_envelope": envelope,
            "sent_envelope_canonical_sha256": sent_sha,
            "get_status": fetched.status_code,
            "returned_body": returned_body,
            "returned_envelope": returned_envelope,
            "returned_envelope_canonical_sha256": returned_sha,
            "envelope_round_trip_byte_identical": sent_sha == returned_sha,
            "envelope_round_trip_equal": envelope == returned_envelope,
            "body_round_trip_byte_identical": _sha256(_canonical(body)) == _sha256(
                _canonical(returned_body)
            ),
            "signature_verified": signature_verified,
            "verification": {
                "digest_on_wire": wire_digest,
                "recomputed_canonical_sha256": _sha256(canonical),
                "digest_matches_canonical_bytes": digest_matches,
                "ed25519_verify_over_canonical_bytes": ed25519_verified,
                "ed25519_verify_error": verify_error,
                "public_key_fingerprint": key_fingerprint(public_key),
            },
            "status_get_status": status_resp.status_code,
            "enforce_status": status_body.get("enforceStatus"),
            "enforce_reason": status_body.get("enforceReason"),
            "policies_listed_after_put": listed,
        }

        # 9. NEGATIVE case: a Shield-blocked decision under
        #    HORIZON_A1_REQUIRE_CERT=1 must never reach the wire.
        blocked_disposition = shield.dispose(
            _action(payload, bandwidth_hz=-1.0), {}, decision_id=f"{DECISION_ID}-blocked"
        )
        blocked_cert = signed_certificate(blocked_disposition.certificate, signing_key)
        blocked_envelope = assurance_envelope(
            blocked_cert, profile_digest=digest_of_profile
        )
        refused = False
        refusal_message: str | None = None
        exception_type: str | None = None
        previous = os.environ.get("HORIZON_A1_REQUIRE_CERT")
        os.environ["HORIZON_A1_REQUIRE_CERT"] = "1"
        try:
            await adapter.emit_policy(
                PROOF_POLICY_TYPE,
                {**payload, "assurance": blocked_envelope},
                policy_id=BLOCKED_POLICY_ID,
                safety_certificate=blocked_cert,
            )
        except ValueError as exc:
            refused = True
            exception_type = type(exc).__name__
            refusal_message = str(exc)
        finally:
            if previous is None:
                os.environ.pop("HORIZON_A1_REQUIRE_CERT", None)
            else:
                os.environ["HORIZON_A1_REQUIRE_CERT"] = previous

        after = await raw.get(
            f"/a1-p/policytypes/{type_id}/policies/{BLOCKED_POLICY_ID}"
        )
        listed_after = await adapter.list_policies(PROOF_POLICY_TYPE)
        result["negative_case"] = {
            "description": (
                "Shield-blocked decision (negative bandwidth is unfixable by "
                "projection) offered to emit_policy with "
                "HORIZON_A1_REQUIRE_CERT=1"
            ),
            "env": {"HORIZON_A1_REQUIRE_CERT": "1"},
            "certificate": {
                "safe": blocked_cert.safe,
                "emit_blocked": blocked_cert.emit_blocked,
                "violated_ids": list(blocked_cert.violated_ids),
            },
            "envelope_that_would_have_shipped": blocked_envelope,
            "refused": refused,
            "exception_type": exception_type,
            "refusal_message": refusal_message,
            "policy_get_status_after": after.status_code,
            "blocked_policy_absent_from_simulator": BLOCKED_POLICY_ID not in listed_after,
            "policies_listed_after": listed_after,
        }

        # 10. CONTROL case: the same body against the pre-1.1.0 schema.
        legacy_schema = json.loads(json.dumps(stored_schema))
        legacy_schema.get("properties", {}).pop("assurance", None)
        control_register = await raw.put(
            f"/a1-p/policytypes/{CONTROL_POLICY_TYPE_ID}",
            json={
                "policy_type_id": CONTROL_POLICY_TYPE_ID,
                "name": "Control: Horizon QoS priority, schema 1.0.0",
                "description": (
                    "Identical to 20001 with the assurance envelope removed from "
                    "create_schema — the pre-1.1.0 shape"
                ),
                "schema_version": "1.0.0",
                "create_schema": legacy_schema,
            },
        )
        control_put = await raw.put(
            f"/a1-p/policytypes/{CONTROL_POLICY_TYPE_ID}"
            f"/policies/{CONTROL_POLICY_ID}",
            json=body,
        )
        control_bare = await raw.put(
            f"/a1-p/policytypes/{CONTROL_POLICY_TYPE_ID}"
            f"/policies/{CONTROL_POLICY_ID}-bare",
            json=payload,
        )
        result["schema_control_case"] = {
            "description": (
                "The simulator validates every policy PUT against the registered "
                "create_schema. The same envelope-bearing body must be rejected "
                "by a policy type whose schema does not declare `assurance`, and "
                "the same body without the envelope must be accepted — otherwise "
                "the positive case proves only that nothing was validating."
            ),
            "policy_type_id": CONTROL_POLICY_TYPE_ID,
            "assurance_declared": "assurance"
            in legacy_schema.get("properties", {}),
            "register_status": control_register.status_code,
            "envelope_bearing_put_status": control_put.status_code,
            "envelope_rejected_by_simulator": control_put.status_code == 400,
            "envelope_free_put_status": control_bare.status_code,
            "envelope_free_accepted": 200 <= control_bare.status_code < 300,
        }

        # Cleanup: roll the proof policy back off the simulator.
        result["positive_case"]["delete_status"] = await adapter.rollback_policy(
            PROOF_POLICY_TYPE, emitted_id
        )
        result["positive_case"]["policies_listed_after_delete"] = (
            await adapter.list_policies(PROOF_POLICY_TYPE)
        )
    finally:
        await raw.aclose()
        await adapter.close()
    return result


def _runtime_block(started_at: float) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "httpx": httpx.__version__,
        "docker_available": False,
        "wall_clock_s": round(time.monotonic() - started_at, 3),
    }


def _judge(result: dict[str, Any]) -> tuple[str, list[str]]:
    """Self-report PASS/FAIL. The gate re-derives this independently."""
    problems: list[str] = []
    types = result.get("policy_types", {})
    readback = types.get("readback", {})
    positive = result.get("positive_case", {})
    negative = result.get("negative_case", {})
    control = result.get("schema_control_case", {})

    if not types.get("registration_complete"):
        problems.append("policy type registration incomplete")
    if not readback.get("declares_assurance"):
        problems.append("simulator is not holding a schema that declares `assurance`")
    if readback.get("assurance_in_top_level_required"):
        problems.append("`assurance` is in the top-level required list")
    if not 200 <= int(positive.get("put_status") or 0) < 300:
        problems.append(f"policy PUT was not 2xx: {positive.get('put_status')}")
    if not positive.get("envelope_round_trip_byte_identical"):
        problems.append("assurance envelope did not survive the round trip")
    if not positive.get("signature_verified"):
        problems.append("signature over the returned digest did not verify")
    if not positive.get("enforce_status"):
        problems.append("no enforceStatus recorded")
    if not negative.get("refused"):
        problems.append("blocked decision was NOT refused")
    if not negative.get("blocked_policy_absent_from_simulator"):
        problems.append("blocked policy reached the simulator")
    if not control.get("envelope_rejected_by_simulator"):
        problems.append("pre-1.1.0 schema did not reject the envelope")
    if not control.get("envelope_free_accepted"):
        problems.append("pre-1.1.0 schema rejected even the envelope-free body")
    return ("PASS" if not problems else "FAIL"), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (holds third_party/sim-a1-interface)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the proof JSON "
        "(default deploy/xapp-e2e/results/a1-assurance-wire-proof.json)",
    )
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_path = args.out or (
        repo_root / "deploy" / "xapp-e2e" / "results" / "a1-assurance-wire-proof.json"
    )

    _no_proxy_for_loopback()
    started_at = time.monotonic()
    port = args.port or _free_port()

    with tempfile.TemporaryDirectory(prefix="a1-assurance-proof-") as tmp:
        tmp_path = Path(tmp)
        sim = Simulator(repo_root, port, tmp_path / "simulator.log")
        proof: dict[str, Any] = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "system_under_test": (
                "Horizon-RIC A1 `assurance` envelope over the osc_a1 dialect "
                "(src/horizon_ric/rapp/a1_adapter.py)"
            ),
        }
        try:
            sim.start()
            proof["simulator"] = sim.describe()
            proof.update(asyncio.run(_run(sim, tmp_path)))
        finally:
            exit_code = sim.stop()
            proof.setdefault("simulator", {})
            proof["simulator"]["exit_code"] = exit_code
            proof["simulator"]["log_tail"] = sim.tail_log(8).splitlines()

    proof["runtime"] = _runtime_block(started_at)
    verdict, problems = _judge(proof)
    proof["result"] = verdict
    proof["self_reported_problems"] = problems
    proof["scope"] = (
        "Proves the SafetyCertificate reference crosses a real O-RAN A1 socket "
        "(official o-ran-sc/sim-a1-interface, OSC_2.1.0) inside Horizon's own "
        "registered policy types, survives round-trip unchanged, and verifies "
        "by Ed25519 signature. Does NOT prove that any near-RT RIC or xApp "
        "consumes or acts on the envelope: no O-RAN specification requires it "
        "and no vendor platform has onboarded it. Not a radio, not operator "
        "traffic, not a conformance certification."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = proof.get("positive_case", {})
    print(
        f"a1-assurance-wire-proof: {verdict} — real "
        f"{proof.get('simulator', {}).get('interface')} simulator @ "
        f"{str(proof.get('simulator', {}).get('commit'))[:12]} on port "
        f"{proof.get('simulator', {}).get('port')}; PUT "
        f"{positive.get('put_status')}, GET {positive.get('get_status')}, "
        f"envelope round-trip "
        f"{positive.get('envelope_round_trip_byte_identical')}, signature "
        f"{positive.get('signature_verified')}, enforceStatus "
        f"{positive.get('enforce_status')} -> {out_path}"
    )
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
