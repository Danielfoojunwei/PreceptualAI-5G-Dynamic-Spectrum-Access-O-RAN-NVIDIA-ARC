# A1 assurance-envelope proof — the SafetyCertificate on a real A1 socket

_Recorded 2026-07-30. Everything below was produced by real processes on real
sockets: the official O-RAN-SC A1 interface simulator running as a subprocess
from the pinned submodule in [`../../third_party/sim-a1-interface`](../../third_party/sim-a1-interface),
a real Ed25519 key, a real `default_terrestrial_shield` disposition, and real
HTTP over loopback. No `httpx.MockTransport`, no in-repo emulator. The raw
artifact is [`results/a1-assurance-wire-proof.json`](results/a1-assurance-wire-proof.json);
the runner is [`a1_assurance_proof.py`](a1_assurance_proof.py) and the CI gate is
[`../../scripts/verify_a1_assurance_wire.py`](../../scripts/verify_a1_assurance_wire.py)._

## What changed

Until this commit the Shield's `SafetyCertificate` never crossed the A1 wire.
Three mechanisms existed, all of them local to the rApp:

| Mechanism | Where |
| --- | --- |
| Env-gated local refusal, default OFF (`HORIZON_A1_REQUIRE_CERT`) | [`../../src/horizon_ric/rapp/a1_adapter.py`](../../src/horizon_ric/rapp/a1_adapter.py) `emit_policy` |
| Pre-emit guard chain (`missing_safety_certificate`, `shield_blocked`) | [`../../src/horizon_ric/policy/emit_guards.py`](../../src/horizon_ric/policy/emit_guards.py) `run_guard_chain` |
| Three-field certificate subset in the DecisionRecord side-channel | [`../../src/horizon_ric/rapp/pipeline.py`](../../src/horizon_ric/rapp/pipeline.py) step 4 |

A near-RT RIC received a `decision_id` and nothing else. It could not tell a
shielded policy from an unshielded one; a verifier had to join the `decision_id`
to the evidence chain out of band.

Every registered A1 policy type was JSON Schema Draft-07 with
`"additionalProperties": false` at every level, and the only cross-cutting
envelope was `rapp_metadata` with properties exactly
`{decision_id, rapp_version, model_versions}`. There was no slot for a
certificate, a signature, or a violated-invariant list.

There is now. `A1Adapter._policy_create_schema` declares a second optional
envelope, `assurance`, in all four policy types (20001–20004), and
`DecisionPipeline` populates it on every emit.

## The wire shape

JSON over HTTP. No ASN.1 anywhere — A1AP is REST/JSON per
O-RAN.WG2.A1-GAP-v05.00 §5, and this extension adds no new encoding. A complete
policy body as read back off the simulator:

```json
{
  "scope": {"slice_id": "slice-a1-assurance-proof"},
  "qos_objectives": {"priority": 5},
  "rapp_metadata": {"decision_id": "a1-assurance-wire-proof-0001"},
  "assurance": {
    "certificate_digest": "80dc99256ba4b2a3fbda2230ecabec6b096a52cbd271ae904eb765562242b582",
    "signature": "8bbb6e4ba7241484416591d6cac1e2e268ef4739393d5ab4841bb016530e6c20bb2b3a46c742f361f17f82089bd746bfb6d9633dd4e498ec750ef5c8b0fd5602",
    "signing_key_fingerprint": "a8179c037bfa4c2a0560ddd349bc300509c497e77d0781e83add38f91dd9d73f",
    "safe": true,
    "projected": false,
    "violated_ids": [],
    "min_margin_dB": 4.0,
    "profile_digest": "acf0118f4eef41cd1a363770cf411b764eb2a9038b30e9c276a1c12c77e1375b"
  }
}
```

Eight keys, four of them required once the envelope is present
(`certificate_digest`, `safe`, `projected`, `violated_ids`).

| Field | Meaning |
| --- | --- |
| `certificate_digest` | SHA-256 of `canonical_certificate_bytes(cert)` — the *exact* bytes the Ed25519 signature covers. The builder imports that function from [`../../src/horizon_ric/shield/signing.py`](../../src/horizon_ric/shield/signing.py) rather than re-deriving the canonical form, so the digest cannot drift from what was signed. |
| `signature` | Hex Ed25519 signature over those same bytes. Omitted, not nulled, when signing is not configured. |
| `signing_key_fingerprint` | SHA-256 of the raw 32-byte public key. Names *which* key a receiver needs. |
| `safe` | The Shield's verdict: the final action satisfies every hard invariant. |
| `projected` | True when the action had to be rewritten to reach the safe set. |
| `violated_ids` | Invariant ids still violated. Non-empty means the decision was refused, readable without fetching anything. |
| `min_margin_dB` | Tightest dB margin across invariants — worst-case headroom. `null` when no invariant reports a dB quantity. |
| `profile_digest` | Digest of the assurance profile reflected off the live Shield ([`../../src/horizon_ric/assurance/profile.py`](../../src/horizon_ric/assurance/profile.py)), so a receiver can tell *which* invariant set, with which band edges and EIRP ceiling, graded the action — and detect a silently reconfigured Shield after a redeploy. |

The certificate itself serialises to 19 keys, including two variable-length
arrays and two embedded action dicts. That is an audit object, and A1 policy
create is a latency-sensitive control interface, so the wire carries 8 keys, not
19. The full record stays in the evidence store and is joined by
`rapp_metadata.decision_id`; `assurance.certificate_digest` is the cryptographic
link between the two. This is the digest form proposed as requirements A-1, A-2
and A-3 in [`../../docs/conformance/ASSURANCE_PROFILE.md`](../../docs/conformance/ASSURANCE_PROFILE.md)
§6.2, promoted from illustration to a registered field.

`schema_v` moved `1.0.0` → `1.1.0` on all four types. That is A-4 of the same
section and it is not bookkeeping: because every schema is
`additionalProperties: false`, a receiver still holding the 1.0.0 schema
**rejects** a body carrying the envelope. Minor rather than major, because the
envelope is optional and a 1.0.0 body remains a valid 1.1.0 body.

## Backward compatibility

`assurance_envelope(None)` returns `{}`, and the pipeline merges nothing when
the envelope is empty:

```python
envelope = assurance_envelope(certificate, profile_digest=self._profile_digest)
wire_payload = {**payload, "assurance": envelope} if envelope else payload
```

An uncertified caller therefore puts the same bytes on the wire as before. That
is asserted on bytes, not on key sets, in
[`../../tests/test_a1_assurance_envelope.py`](../../tests/test_a1_assurance_envelope.py)
(`test_absent_certificate_leaves_the_policy_body_byte_identical`).

The merge builds a **copy**. `Shield.dispose` shallow-copies the action, so
`certificate.action_proposed["policy_payload"]` is the same dict object as the
payload; mutating it in place would retroactively change the bytes the signature
was computed over and the certificate would stop verifying against itself. For
the same reason `record.chosen_action["policy_payload"]` keeps pointing at the
pre-envelope payload — it must keep matching what the Shield actually graded.
The DecisionRecord's three-field certificate subset is unchanged: the
side-channel and the wire are different consumers.

## The live run

| Item | Value |
| --- | --- |
| Simulator | `o-ran-sc/sim-a1-interface`, `near-rt-ric-simulator/src/OSC_2.1.0` |
| Commit | `be2943f57211f62095dc5434099e331df237aafb` (`git -C third_party/sim-a1-interface rev-parse HEAD`) |
| Upstream | `https://gerrit.o-ran-sc.org/r/sim/a1-interface` |
| Source modifications | none |
| Transport | plain HTTP on a free loopback port (`http://127.0.0.1:<port>`) |
| Dialect | `osc_a1` — `PUT /a1-p/policytypes/{id}`, `PUT /a1-p/policytypes/{id}/policies/{pid}`, `GET .../status` |
| Interpreter | the repo venv, Python 3.11.15 |

There is no Docker daemon in this environment, so the simulator was started
natively instead of from its published image. This is the same source and the
same entry point: `near-rt-ric-simulator/src/start.sh` exports `APIPATH` and
`PYTHONPATH` and runs `src/OSC_2.1.0/main.py`, and the runner exports the same
two variables and runs the same module. One deviation, disclosed: `main.py`
hardcodes port 2222 — its `isinstance(sys.argv[1], int)` guard is always false
because argv entries are strings, so the documented port argument is dead code.
The runner imports the module and runs its connexion app on the chosen port. No
simulator source is patched.

### Sequence and results

| Step | Result |
| --- | --- |
| `GET /a1-p/healthcheck` | 200 |
| `register_policy_types()` | `[20001, 20002, 20003, 20004]`, schema 1.1.0 |
| `GET /a1-p/policytypes/20001` read-back | 200; the schema **the simulator is holding** declares `assurance` with `additionalProperties: false`, required `[certificate_digest, projected, safe, violated_ids]`, and `assurance` absent from the top-level `required` |
| `generate_signing_key` | real Ed25519, fingerprint `a8179c03…` |
| `shield.dispose(...)` | `safe=true`, `emit_blocked=false`, `projected=false`, 5 invariants, `min_margin_dB=4.0` |
| `PUT /a1-p/policytypes/20001/policies/horizon-a1-assurance-proof` | **202** |
| `GET` the policy back | 200; envelope canonical SHA-256 `1f534041e7fb…` — **identical** to what was sent |
| Signature verification | `true` — the returned `certificate_digest` equals SHA-256 of the certificate's canonical bytes, and the returned `signature` verifies as Ed25519 over exactly those bytes |
| `GET .../status` (A1AP §6.5) | 200, `enforceStatus: "NOT_ENFORCED"`, `enforceReason: "OTHER_REASON"` |
| `DELETE` the policy | 202, policy list empty afterwards |

`NOT_ENFORCED` is the honest answer and is not a failure: the simulator sets
`EnforceStatus("NOT_ENFORCED", "OTHER_REASON")` on every create
(`controllers/a1_mediator_controller.py`,
`a1_controller_create_or_replace_policy_instance`) because no xApp sits behind
it. Horizon reports what the RIC actually said.

"Byte-identical" is a precise claim: the envelope as sent and the envelope as
read back hash to the same SHA-256 under
`json.dumps(..., sort_keys=True, separators=(",", ":"))`. Both raw objects and
both digests are in the artifact, and the gate recomputes the digests from the
recorded objects rather than trusting the recorded booleans.

### Negative case — a blocked decision does not reach the wire

A negative bandwidth is unfixable by projection, so the Shield fails closed:
`safe=false`, `emit_blocked=true`, `violated_ids=["numeric_domain_sanity"]`.
Offered to `emit_policy` under `HORIZON_A1_REQUIRE_CERT=1`, it was refused
before any HTTP:

```
ValueError: HORIZON_A1_REQUIRE_CERT is set and the safety certificate for the
horizon.qos.priority emit is not clean (safe=False, emit_blocked=True) — refusing
```

`GET /a1-p/policytypes/20001/policies/horizon-a1-assurance-proof-blocked` → 404,
and the id is absent from the simulator's own policy list. The envelope that
*would* have shipped is recorded in the artifact, showing `safe: false` with a
non-empty `violated_ids` — so even in a deployment that emits refusals, a
receiver reading only the wire can tell a refusal from an endorsement.

### Control case — the schema extension is load bearing

The simulator validates every policy PUT against the registered `create_schema`
with real `jsonschema` (`a1_mediator_controller.py`,
`a1_controller_create_or_replace_policy_instance` → `validate(instance=data,
schema=policy_types[policy_type_id]['create_schema'])`). So the positive 202 is
only meaningful if something was actually validating. The runner registers a
second policy type, 29001, carrying the identical schema with the `assurance`
declaration removed — the pre-1.1.0 shape — and PUTs the same body at it:

| PUT against policy type 29001 (schema without `assurance`) | Status |
| --- | --- |
| the envelope-bearing body | **400** |
| the same body with the envelope removed | **202** |

The 400 is attributable to the envelope and to nothing else. This is also the
concrete demonstration of why A-4's version bump is required rather than
optional.

## Reproducing

```bash
python deploy/xapp-e2e/a1_assurance_proof.py --out /tmp/a1_assurance_fresh.json
python scripts/verify_a1_assurance_wire.py --fresh /tmp/a1_assurance_fresh.json
```

The gate compares the fresh run against
[`results/a1-assurance-wire-proof.json`](results/a1-assurance-wire-proof.json)
and fails on any of: an envelope that changed on the wire, a signature that did
not verify, a `certificate_digest` that is not the SHA-256 of the recorded
canonical bytes, a simulator commit that does not match
`git -C third_party/sim-a1-interface rev-parse HEAD` in the tree being verified,
a negative case that was not refused, a control case that did not return 400, or
a run in which no policy was accepted with a 2xx at all. Per-run quantities —
the Ed25519 key, the certificate's `issued_at` and therefore its digest and
signature, the port, timings — are deliberately not compared between runs; they
are checked for internal consistency within the fresh run instead.

`tests/test_a1_assurance_envelope.py` holds the same contract in the fast test
pack, with a hand-written Draft-07 subset validator (the repo has no
`jsonschema` dependency and this change does not add one).

## Honest scope — what this does and does not prove

**It proves:**

- the certificate reference crosses a real O-RAN A1 surface — the official
  O-RAN-SC A1 interface simulator at a pinned commit, over a real socket, in
  Horizon's own registered policy types;
- the receiving implementation *validates* the body against the registered
  `create_schema` and accepts it only because the schema declares the envelope
  (400 when it does not);
- the envelope survives the round trip unchanged, canonical byte for canonical
  byte;
- the reference is cryptographically verifiable by the receiver: the digest on
  the wire is the SHA-256 of the exact bytes the Ed25519 signature covers, and
  that signature verifies against the named public key;
- a Shield-blocked decision is refused before the wire under
  `HORIZON_A1_REQUIRE_CERT=1`, and a refusal that *is* emitted is legible as a
  refusal.

**It does not prove:**

- that any vendor's near-RT RIC or any xApp **consumes or acts on** the
  assurance envelope. Nothing in the O-RAN specifications requires them to. No
  vendor platform has onboarded this. The envelope is a **proposed extension**
  carried inside Horizon's own policy types (20001–20004), not a standardised
  A1 field, and `docs/conformance/ASSURANCE_PROFILE.md` §6.2 still lists A-1…A-5
  as proposals;
- that the simulator does anything with the envelope beyond storing, validating
  and returning it. It is a simulator: it has no policy handlers, which is why
  `enforceStatus` is `NOT_ENFORCED`;
- that the four other dialects (`legacy`, `osc`, `eiap`, `mantaray`) carry the
  envelope end to end against **real vendor targets**. Only `osc_a1` was
  exercised here on a real socket. The transport claim itself is no longer left
  as reasoning, though: `tests/test_a1_assurance_all_dialects.py` pins, offline
  over `httpx.MockTransport`, that all five dialects deliver the envelope to the
  transport boundary byte for byte and that the digest arriving still covers the
  signed bytes — through each of the three distinct body shapes (`legacy` and
  `osc_a1` PUT a bare payload, `osc` nests it under `policy_data`, `eiap` and
  `mantaray` under `policyData`). It also fails if a sixth dialect is added to
  the adapter without being covered. What remains unproven is the *vendor
  platform* on the far end, not our side of the wire;
- anything about a radio, operator traffic, RAN load, an E2 node, a conformance
  certification, or carrier-scale behaviour;
- that a receiver can *revoke* or *reject* on a failed reference. A-5 of the
  profile proposes reporting such a refusal over the A1AP policy-status surface;
  nothing implements it, and the simulator has no mechanism for it.

Related: [`../XAPP_E2E_PROOF.md`](../XAPP_E2E_PROOF.md) (full pipeline against
the real A1 mediator and `hw-python` xApp),
[`README.md`](README.md) (the xApp harness),
[`../OSC_NONRTRIC_PROOF.md`](../OSC_NONRTRIC_PROOF.md) (NONRTRIC Policy
Management Service).
