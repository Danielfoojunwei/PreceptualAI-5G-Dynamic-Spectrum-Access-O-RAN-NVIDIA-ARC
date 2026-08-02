#!/usr/bin/env python3
"""Verify a fresh A1 assurance-envelope run against the committed real result.

This gates the one claim the assurance envelope makes: that a Shield
SafetyCertificate now crosses a **real** O-RAN A1 socket as a verifiable
reference, survives the round trip unchanged, and verifies by signature. Before
this, the certificate reached nobody southbound — it was enforced by an
env-gated local refusal, a pre-emit guard chain, and a three-field subset
written into the evidence side-channel. None of that is visible to a near-RT
RIC.

A committed JSON asserting "it works" would be worth nothing here, because every
interesting way for this to be false leaves the file looking healthy:

* the run could have talked to a hand-rolled stub instead of the vendored
  o-ran-sc simulator, so the recorded simulator commit is compared against
  ``git -C third_party/sim-a1-interface rev-parse HEAD`` in the tree being
  verified, not merely against the committed value;
* the round-trip and signature results could be booleans someone set, so the
  digests are **recomputed** here from the recorded envelopes rather than read
  out of the flags;
* the run could have PUT nothing at all and reported success vacuously, so at
  least one policy must have been accepted with a 2xx and read back with the
  envelope on it;
* the schema could have been permissive rather than extended, so the control
  case must show the pre-1.1.0 schema (no ``assurance`` declaration) rejecting
  the identical body with a 400 while accepting it without the envelope.

Unlike the benchmark verifiers, almost nothing here is a float. The comparable
quantities are HTTP status codes, key sets, and hex digests — all exact, so
there is no tolerance band. The per-run quantities (the Ed25519 key, the
certificate's ``issued_at`` and therefore its digest and signature, the port)
are deliberately NOT compared between committed and fresh; they are checked for
internal consistency within the fresh run instead.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SIM_SUBMODULE = "third_party/sim-a1-interface"

# Fields that must reproduce exactly run to run. All are status codes, ids,
# booleans or declared version strings — nothing host- or key-dependent.
STABLE_FIELDS: tuple[tuple[str, ...], ...] = (
    ("simulator", "commit"),
    ("simulator", "interface"),
    ("simulator", "kind"),
    ("simulator", "submodule_path"),
    ("policy_types", "registered_ids"),
    ("policy_types", "expected_ids"),
    ("policy_types", "schema_version_declared"),
    ("policy_types", "readback", "http_status"),
    ("policy_types", "readback", "assurance_properties"),
    ("policy_types", "readback", "assurance_required"),
    ("policy_types", "readback", "create_schema_sha256"),
    ("positive_case", "policy_type"),
    ("positive_case", "policy_type_id"),
    ("positive_case", "policy_id"),
    ("positive_case", "put_status"),
    ("positive_case", "get_status"),
    ("positive_case", "status_get_status"),
    ("positive_case", "delete_status"),
    ("positive_case", "enforce_status"),
    ("negative_case", "exception_type"),
    ("negative_case", "policy_get_status_after"),
    ("schema_control_case", "register_status"),
    ("schema_control_case", "envelope_bearing_put_status"),
    ("schema_control_case", "envelope_free_put_status"),
)

# The envelope's declared shape. A receiver that cannot rely on these keys
# cannot verify anything, so a silent rename is a breaking change.
REQUIRED_ENVELOPE_KEYS = frozenset(
    {"certificate_digest", "safe", "projected", "violated_ids"}
)
VERIFIABLE_ENVELOPE_KEYS = frozenset({"signature", "signing_key_fingerprint"})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(blob: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = blob
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _canonical_sha256(obj: Any) -> str:
    """The same canonical JSON the proof runner and the cert signer use."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _live_simulator_commit(repo_root: Path) -> str | None:
    """``git -C <repo-root>/third_party/sim-a1-interface rev-parse HEAD``.

    None when the submodule is absent or is not a git checkout — which is
    itself a verification failure, not something to paper over.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root / SIM_SUBMODULE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def verify(
    committed: dict[str, Any],
    fresh: dict[str, Any],
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []

    simulator = fresh.get("simulator") or {}
    positive = fresh.get("positive_case") or {}
    negative = fresh.get("negative_case") or {}
    control = fresh.get("schema_control_case") or {}
    readback = (fresh.get("policy_types") or {}).get("readback") or {}
    if not positive:
        errors.append("fresh result has no 'positive_case' block")
        return errors

    # 1. The simulator was the REAL vendored one. The recorded commit is
    #    compared against the working tree's actual submodule HEAD, so a run
    #    against a hand-rolled stub (or against our own deploy/osc_emulator)
    #    cannot pass by simply naming a plausible commit in the JSON.
    live_commit = _live_simulator_commit(repo_root) if repo_root else None
    if repo_root is None:
        errors.append("no --repo-root given, so the simulator commit cannot be checked")
    elif live_commit is None:
        errors.append(
            f"could not read {SIM_SUBMODULE} HEAD under {repo_root} -- the vendored "
            "simulator is not checked out there, so the recorded commit is unverifiable"
        )
    elif simulator.get("commit") != live_commit:
        errors.append(
            f"simulator.commit: recorded {simulator.get('commit')!r} != working-tree "
            f"{SIM_SUBMODULE} HEAD {live_commit!r}"
        )
    if simulator.get("kind") != "real_vendored_submodule":
        errors.append(
            f"simulator.kind is {simulator.get('kind')!r}, not 'real_vendored_submodule' "
            "-- the run did not exercise the official O-RAN-SC simulator"
        )
    if simulator.get("fallback_used") is not False:
        errors.append(
            "simulator.fallback_used is not False: "
            f"{simulator.get('fallback_used')!r} ({simulator.get('fallback_reason')!r})"
        )
    if simulator.get("source_modified") is not False:
        errors.append("simulator.source_modified is not False -- vendored source patched")
    if simulator.get("healthcheck_status") != 200:
        errors.append(
            f"simulator healthcheck was {simulator.get('healthcheck_status')!r}, not 200"
        )

    # 2. NON-VACUITY. A run that PUT nothing, or whose PUT was rejected, or
    #    whose read-back body carried no envelope, proves nothing whatever it
    #    reports. Check the traffic actually happened before checking its
    #    properties.
    put_status = positive.get("put_status")
    if not (isinstance(put_status, int) and 200 <= put_status < 300):
        errors.append(
            f"no policy was accepted: positive_case.put_status is {put_status!r}, not 2xx"
        )
    if positive.get("get_status") != 200:
        errors.append(
            f"the policy was not read back: get_status is {positive.get('get_status')!r}"
        )
    returned = positive.get("returned_envelope")
    sent = positive.get("sent_envelope")
    if not isinstance(returned, dict) or not returned:
        errors.append("returned_envelope is empty -- nothing crossed the wire")
    if not isinstance(sent, dict) or not sent:
        errors.append("sent_envelope is empty -- no envelope was offered to the wire")
    listed = positive.get("policies_listed_after_put") or []
    if positive.get("policy_id") not in listed:
        errors.append(
            f"policy {positive.get('policy_id')!r} absent from the simulator's own "
            f"policy list after the PUT: {listed!r}"
        )
    if not isinstance(returned, dict) or not REQUIRED_ENVELOPE_KEYS <= set(returned):
        errors.append(
            "returned envelope is missing required keys: "
            f"{sorted(REQUIRED_ENVELOPE_KEYS - set(returned or {}))}"
        )
    if not isinstance(returned, dict) or not VERIFIABLE_ENVELOPE_KEYS <= set(returned):
        errors.append(
            "returned envelope carries no signature/fingerprint, so nothing on the "
            "wire is verifiable"
        )

    # 3. The round trip was byte-identical -- recomputed here from the recorded
    #    envelopes, not trusted from the recorded flag. Both the flag and the
    #    two canonical digests have to agree with what the payloads actually
    #    hash to, so perturbing any one of them fails.
    if isinstance(sent, dict) and isinstance(returned, dict):
        sent_sha = _canonical_sha256(sent)
        returned_sha = _canonical_sha256(returned)
        if sent_sha != returned_sha:
            errors.append(
                f"assurance envelope changed on the wire: sent canonical "
                f"sha256 {sent_sha} != returned {returned_sha}"
            )
        if positive.get("sent_envelope_canonical_sha256") != sent_sha:
            errors.append(
                "sent_envelope_canonical_sha256 does not match the recorded "
                f"sent_envelope: {positive.get('sent_envelope_canonical_sha256')!r} "
                f"!= {sent_sha}"
            )
        if positive.get("returned_envelope_canonical_sha256") != returned_sha:
            errors.append(
                "returned_envelope_canonical_sha256 does not match the recorded "
                f"returned_envelope: "
                f"{positive.get('returned_envelope_canonical_sha256')!r} != {returned_sha}"
            )
    if positive.get("envelope_round_trip_byte_identical") is not True:
        errors.append(
            "envelope_round_trip_byte_identical is "
            f"{positive.get('envelope_round_trip_byte_identical')!r}, not True"
        )
    if positive.get("body_round_trip_byte_identical") is not True:
        errors.append(
            "body_round_trip_byte_identical is "
            f"{positive.get('body_round_trip_byte_identical')!r}, not True"
        )

    # 4. The signature over the returned reference verified. Asserted on the
    #    flag AND on both of its components, because a run that recorded
    #    signature_verified without a digest match, or without an Ed25519
    #    verify, has verified nothing.
    verification = positive.get("verification") or {}
    if positive.get("signature_verified") is not True:
        errors.append(
            f"signature_verified is {positive.get('signature_verified')!r}, not True"
        )
    if verification.get("digest_matches_canonical_bytes") is not True:
        errors.append(
            "verification.digest_matches_canonical_bytes is "
            f"{verification.get('digest_matches_canonical_bytes')!r}, not True"
        )
    if verification.get("ed25519_verify_over_canonical_bytes") is not True:
        errors.append(
            "verification.ed25519_verify_over_canonical_bytes is "
            f"{verification.get('ed25519_verify_over_canonical_bytes')!r}, not True"
        )
    if verification.get("ed25519_verify_error") is not None:
        errors.append(
            f"Ed25519 verification error recorded: "
            f"{verification.get('ed25519_verify_error')!r}"
        )

    # 5. The digest on the wire IS the digest of the bytes that were signed.
    #    This is the whole join between the A1 body and the evidence chain: one
    #    hex string, in three places, all of which must agree.
    certificate = fresh.get("certificate") or {}
    canonical_sha = certificate.get("canonical_bytes_sha256")
    wire_digest = verification.get("digest_on_wire")
    for label, value in (
        ("verification.digest_on_wire", wire_digest),
        ("verification.recomputed_canonical_sha256",
         verification.get("recomputed_canonical_sha256")),
        ("returned_envelope.certificate_digest",
         (returned or {}).get("certificate_digest") if isinstance(returned, dict) else None),
    ):
        if value != canonical_sha:
            errors.append(
                f"{label} is {value!r} but the certificate's canonical-bytes "
                f"sha256 is {canonical_sha!r}"
            )
    if not (isinstance(canonical_sha, str) and len(canonical_sha) == 64):
        errors.append(f"certificate.canonical_bytes_sha256 is not a sha256: {canonical_sha!r}")
    if certificate.get("self_verifies_locally") is not True:
        errors.append(
            "the signed certificate does not verify against its own public key "
            f"({certificate.get('self_verifies_locally')!r})"
        )
    # The wire deliberately carries a reference, not a copy of the audit object.
    if certificate.get("keys_on_the_wire") == certificate.get("keys_in_the_certificate"):
        errors.append(
            "the whole certificate is on the wire "
            f"({certificate.get('keys_in_the_certificate')} keys) -- the envelope is "
            "supposed to be a digest-plus-signature reference"
        )

    # 6. The negative case was actually REFUSED, and refused before the wire.
    #    A blocked decision that merely failed to be enforced would not be a
    #    fail-closed guarantee.
    if negative.get("refused") is not True:
        errors.append(
            f"negative case was not refused: refused={negative.get('refused')!r}"
        )
    if negative.get("exception_type") != "ValueError":
        errors.append(
            f"negative case raised {negative.get('exception_type')!r}, not ValueError"
        )
    message = str(negative.get("refusal_message") or "")
    if "HORIZON_A1_REQUIRE_CERT" not in message:
        errors.append(f"refusal message does not name the gate: {message!r}")
    blocked_cert = negative.get("certificate") or {}
    if blocked_cert.get("emit_blocked") is not True or blocked_cert.get("safe") is not False:
        errors.append(
            "the negative case's certificate is not a blocked one "
            f"(safe={blocked_cert.get('safe')!r}, "
            f"emit_blocked={blocked_cert.get('emit_blocked')!r}) -- the refusal proves "
            "nothing about blocked decisions"
        )
    if not blocked_cert.get("violated_ids"):
        errors.append("the blocked certificate reports no violated invariant ids")
    if negative.get("policy_get_status_after") != 404:
        errors.append(
            "the refused policy is not absent from the simulator: GET returned "
            f"{negative.get('policy_get_status_after')!r}, expected 404"
        )
    if negative.get("blocked_policy_absent_from_simulator") is not True:
        errors.append("the refused policy appears in the simulator's policy list")

    # 7. The extension is load bearing. The simulator validates every PUT
    #    against the registered create_schema, so the pre-1.1.0 schema must
    #    reject the identical body (400) and accept it once the envelope is
    #    removed. Without both halves, the positive case is consistent with
    #    nothing validating at all.
    if readback.get("declares_assurance") is not True:
        errors.append("the schema the simulator stored does not declare `assurance`")
    if readback.get("assurance_additional_properties") is not False:
        errors.append(
            "the stored `assurance` schema is not additionalProperties:false "
            f"({readback.get('assurance_additional_properties')!r})"
        )
    if readback.get("assurance_in_top_level_required") is not False:
        errors.append("`assurance` is in the top-level required list -- not optional")
    if sorted(readback.get("assurance_required") or []) != sorted(REQUIRED_ENVELOPE_KEYS):
        errors.append(
            f"stored `assurance` required list is {readback.get('assurance_required')!r}, "
            f"expected {sorted(REQUIRED_ENVELOPE_KEYS)}"
        )
    if control.get("assurance_declared") is not False:
        errors.append("the control policy type still declares `assurance` -- not a control")
    if control.get("envelope_rejected_by_simulator") is not True:
        errors.append(
            "the pre-1.1.0 schema did NOT reject the envelope-bearing body "
            f"(status {control.get('envelope_bearing_put_status')!r}) -- so the "
            "simulator was not validating and the positive case proves nothing"
        )
    if control.get("envelope_free_accepted") is not True:
        errors.append(
            "the control policy type rejected even the envelope-free body "
            f"(status {control.get('envelope_free_put_status')!r}) -- the control is "
            "broken, so its 400 is not attributable to the envelope"
        )

    # 8. The stable surface reproduces exactly against the committed run.
    #    Per-run quantities (key, digest, signature, port, timings) are
    #    excluded by construction -- see the module docstring.
    for path in STABLE_FIELDS:
        want, got = _get(committed, path), _get(fresh, path)
        if want != got:
            errors.append(f"{'.'.join(path)}: committed {want!r} != fresh {got!r}")
    committed_env = _get(committed, ("positive_case", "returned_envelope")) or {}
    if isinstance(returned, dict) and set(committed_env) != set(returned):
        errors.append(
            f"envelope key set changed: committed {sorted(committed_env)} != "
            f"fresh {sorted(returned)}"
        )

    # 9. The run's own verdict, asserted rather than inferred.
    if fresh.get("result") != "PASS":
        errors.append(
            f"fresh run did not self-report PASS: {fresh.get('result')!r} "
            f"{fresh.get('self_reported_problems')!r}"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--committed",
        type=Path,
        default=Path("deploy/xapp-e2e/results/a1-assurance-wire-proof.json"),
    )
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=f"tree holding {SIM_SUBMODULE}, whose HEAD must match the recorded commit",
    )
    args = ap.parse_args()

    errors = verify(_load(args.committed), _load(args.fresh), args.repo_root.resolve())
    if errors:
        print("a1-assurance-wire verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    fresh = _load(args.fresh)
    simulator = fresh["simulator"]
    positive = fresh["positive_case"]
    certificate = fresh["certificate"]
    print(
        "a1-assurance-wire verification PASSED: certificate digest "
        f"{certificate['canonical_bytes_sha256'][:12]} crossed the real "
        f"{simulator['interface']} A1 surface "
        f"({simulator['submodule_path']} @ {simulator['commit'][:12]}, port "
        f"{simulator['port']}) as policy {positive['policy_id']} "
        f"type {positive['policy_type_id']} -- PUT {positive['put_status']}, "
        f"GET {positive['get_status']}, envelope canonical sha256 "
        f"{positive['returned_envelope_canonical_sha256'][:12]} unchanged, "
        f"Ed25519 signature verified against key "
        f"{certificate['signing_key_fingerprint'][:12]}, enforceStatus "
        f"{positive['enforce_status']}; the pre-1.1.0 schema rejected the same body "
        f"with {fresh['schema_control_case']['envelope_bearing_put_status']} and the "
        "blocked decision was refused before the wire "
        f"({fresh['negative_case']['exception_type']}, "
        f"{certificate['keys_on_the_wire']} of "
        f"{certificate['keys_in_the_certificate']} certificate keys on the wire)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
