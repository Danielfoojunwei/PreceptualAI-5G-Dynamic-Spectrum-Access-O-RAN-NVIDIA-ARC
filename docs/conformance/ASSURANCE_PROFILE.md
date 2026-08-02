# Horizon-RIC Assurance Conformance Profile

**Version:** 0.1.1
**Date:** 30 July 2026
**Changes in 0.1.1:** §7.2 rewritten. Version 0.1.0 stated that no E2SM-RC
existed in this repository; that stopped being true a few commits later, when
the control payload encoder landed. The E2 table row and the non-goal in §2 are
corrected with it. Nothing else changed.
**Status:** **DRAFT.** This document is **candidate input to a standardisation
discussion, not an adopted standard.** No standards body has reviewed it, no
vendor has implemented it, and every requirement in it is open to being wrong.
It is published at this maturity because a profile that is argued about is worth
more than one that is finished in private.
**Work package:** WP1 deliverable D (see `docs/proposal/` for the programme
context).
**Companion:** [`CONFORMANCE.md`](CONFORMANCE.md) records what the repository
implements against published O-RAN and 3GPP specifications. This document is the
opposite direction: what a *profile* would have to say for the assurance layer in
this repository to be checkable by someone who did not write it.

Status legend (same glyphs as [`CONFORMANCE.md`](CONFORMANCE.md)):

| Glyph | Meaning |
| ----- | ------- |
| ✅ | Implemented and continuously tested in CI |
| ⚠️ | Partially implemented — gaps tracked in `docs/EVALUATION_CRITERIA.md` |
| ❌ | Not applicable — out of scope for this component |
| 🟡 | Planned — design exists, code not yet landed |

Requirement keywords **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT** and
**MAY** are used in the RFC 2119 sense. Requirements are numbered by area: `P-n`
planner, `I-n` invariant, `C-n` certificate, `A-n` A1 binding, `E-n` E2 binding.
Each requirement carries a status glyph for *this repository*, which is the only
implementation that exists.

---

## 1. What this profile is for

The profile pins one loop and nothing else:

    action      = planner.propose(observation)     # the AI proposes
    disposition = shield.dispose(action, context)  # the invariant chain disposes
    certificate = disposition.certificate          # the disposition is evidence

Three artefacts make that loop checkable by a third party:

1. an **action contract** — what a planner is allowed to emit, so that a planner
   written by someone else can be dropped into someone else's Shield;
2. an **invariant contract** — what a constraint has to look like and how a chain
   of them composes, so that a regulator-supplied invariant can be added to a
   vendor's chain;
3. a **certificate schema** with a canonical signing form, so the disposition is
   verifiable offline by a party who trusts neither the planner nor the operator.

The profile exists because the interesting failure mode in AI-RAN is not a model
that performs badly. It is a model that performs well and is trusted for its own
safety claim. Every requirement below is chosen so that the safety verdict is
produced by something the model does not control.

### 1.1 What this profile deliberately does not do

* **It does not define a learning algorithm, a reward, a feature set, or a model
  architecture.** The planner's input featurisation is explicitly unconstrained.
* **It does not define an O-RAN message.** It defines a certificate. Section 6
  states plainly that the certificate does not currently cross the A1 wire and
  that the field which would carry it does not exist.
* **It does not certify that a decision is safe.** It certifies that a decision
  was checked against a *declared* invariant set and records the result. An
  invariant nobody wrote is not enforced by anything here, and the certificate
  makes no claim about it.
* **It does not define near-real-time enforcement.** E2SM-RC control payloads
  can now be *constructed* from a disposition, and a refused action cannot be
  constructed at all (§7.2), but nothing here delivers one. Binding a
  certificate to a delivered E2 control action remains WP4 work.
* **It does not replace the O-RAN WG11 security specification, the WG4
  conformance test specification, or 3GPP TS 28.105 AI/ML management.** It sits
  under them. Where they say what to protect, this says what a disposition record
  has to contain.
* **It does not claim adoption.** Zero vendor platforms have onboarded (section
  9, gap 5).

---

## 2. The planner contract

Reference implementation:
[`src/horizon_ric/assurance/planner.py`](../../src/horizon_ric/assurance/planner.py).
Machine-readable form:
[`docs/schemas/proposed-action-v1.schema.json`](../schemas/proposed-action-v1.schema.json).

The planner interface is **new in WP1**. Before it there was no planner interface
anywhere in `src/`: no ABC, no Protocol, no injection point. It is also **not yet
used by the production pipeline** — `DecisionPipeline` hard-codes its planner as
private methods (`_risk` at `src/horizon_ric/rapp/pipeline.py:216`, `_candidates`
at `:226`, `_choose` at `:273`) and constructs its own Shield at `:197`. The
contract is published and frozen first; rewiring the pipeline onto it is
deliberately deferred. Nothing below should be read as a description of what the
running rApp does today.

| Id | Requirement | Status |
| --- | --- | --- |
| P-1 | A planner **MUST** expose a stable `planner_id: str` and a `propose(observation) -> Action` method. Conformance is structural: an implementation matches the shape, it does not inherit (`assurance/planner.py:135`). | ✅ |
| P-2 | A proposed action **MUST** carry `block`, `frequency_hz`, `bandwidth_hz`, `tx_power_dBm` (`assurance/planner.py:76`), in the numeric domains `NumericSanityInvariant` enforces (`src/horizon_ric/shield/invariants.py:73`). | ✅ |
| P-3 | Every value anywhere in the action **MUST** be JSON-primitive. See 2.1. | ✅ |
| P-4 | A planner **MUST NOT** set `emit_blocked` or `shield_fallback_to` (`assurance/planner.py:105`, `shield/invariants.py:36`). See 2.2. | ✅ |
| P-5 | A planner's self-reported quality fields **MUST NOT** be the basis of a satisfaction verdict. They **MAY** be recorded for audit. See 2.3. | ✅ |
| P-6 | A planner **MAY** carry deployment-specific keys the chain ignores. Extra keys are still bound by P-3 and P-4. | ✅ |
| P-7 | A contract violation **MUST** be raised at the planner boundary, before the Shield is consulted, naming the planner (`PlannerContractError`, `assurance/planner.py:117`, raised from `shielded(...)` at `:311`). | ✅ |
| P-8 | `propose` **SHOULD** be deterministic given the same observation and seed, so a disposition can be replayed. Not enforced anywhere. | ⚠️ |

### 2.1 The JSON-primitive rule (P-3), and why it is normative

This looks like a style rule and is not. The certificate embeds the planner's
proposed dict **verbatim** under `action_proposed`
(`src/horizon_ric/shield/certificate.py:110`), and the canonical signing form is
`json.dumps(...)` **with no `default=` handler**
(`src/horizon_ric/shield/signing.py:52`). So a `Decimal`, a `datetime`, an
`Enum`, a `set`, `bytes`, or a `numpy.int64` anywhere in the action raises
`TypeError` at *signing* time: after the decision was taken, on the evidence
path, where the failure is hardest to attribute and where refusing is no longer
free.

Normatively, therefore:

* Values **MUST** be `None`, `bool`, `int`, `float`, `str`, a list of such, or a
  mapping with `str` keys of such. Mapping keys **MUST** be `str`, because
  `sort_keys=True` cannot order mixed-type keys.
* `float` values **MUST** be finite. `json.dumps` emits `NaN` and `Infinity`,
  which are not JSON and which a non-Python verifier will reject.
* Subclasses of the primitive types (`numpy.float64`, `IntEnum`) **SHOULD** be
  converted at the boundary rather than relied on. `numpy.float64` happens to
  serialise; `numpy.int64` does not subclass `int` and raises; an `IntEnum`
  serialises as its integer and silently loses its identity inside signed
  evidence. Rejecting the whole family is cheaper than reasoning about which
  members survive.

### 2.2 Output sentinels (P-4)

`emit_blocked` and `shield_fallback_to` are written *by* the Shield when it
refuses or falls back. A planner that presets either is forging the Shield's own
verdict, so the contract forbids them outright rather than overwriting them.

### 2.3 Self-attestation is not assurance (P-5)

`NeuralRxEnvelopeInvariant` is the worked example
(`shield/invariants.py:472`). It previously graded a neural receiver on its own
`predicted_tbler` and `demap_confidence`, which is exactly the field a poisoned
model controls: a backdoored receiver reports a good error rate and walks
through. It now grades against an independently measured block error rate
supplied in `context["measured_tbler"]` (in production, CRC and HARQ telemetry
from the MAC, which the receiver cannot forge), and treats an absent measurement
as *unverified* and therefore a fallback (`shield/invariants.py:519`). The
self-report is still recorded. It is not trusted.

---

## 3. The invariant contract

Reference implementation: [`src/horizon_ric/shield/invariants.py`](../../src/horizon_ric/shield/invariants.py).
Composition: [`src/horizon_ric/shield/shield.py`](../../src/horizon_ric/shield/shield.py).
Machine-readable declaration of a deployment's chain:
[`docs/schemas/invariant-profile-v1.schema.json`](../schemas/invariant-profile-v1.schema.json),
emitted from a live Shield by
[`src/horizon_ric/assurance/profile.py`](../../src/horizon_ric/assurance/profile.py)
(`emit_profile`, `profile_digest`), also new in WP1.

| Id | Requirement | Status |
| --- | --- | --- |
| I-1 | An invariant **MUST** expose `id: str`, `evaluate(action, context) -> InvariantCheck`, and `project(action, context) -> (action, [ConstraintViolation])` (`shield/invariants.py:39`). | ✅ |
| I-2 | `evaluate` **MUST NOT** mutate the action or the context, and **MUST** report `satisfied` for every input it is given. | ✅ |
| I-3 | `project` **MUST** return a **new** dict and **MUST NOT** mutate the action in place. See 3.1. | ✅ |
| I-4 | When the safe set is unreachable, `project` **MUST** leave the invariant unsatisfied (or set `emit_blocked`) and **MUST NOT** return an action that merely looks compliant. See 3.2. | ✅ |
| I-5 | Chain order is semantically significant and **MUST** be declared as part of a conformance claim. See 3.3. | ⚠️ |
| I-6 | `project` **MUST** be a no-op on an action that already satisfies the invariant. See 3.4. | ✅ |
| I-7 | An invariant **MAY** raise on malformed input, but **MUST NOT** return `satisfied=True` for input it could not evaluate. See 3.5. | ✅ |
| I-8 | A margin **MUST** be labelled with the unit it is in, and an invariant whose natural unit is not dB **MUST NOT** label its margin `"dB"`. See 3.6. | ✅ |
| I-9 | The invariant set, and the numeric bounds each invariant was configured with, **MUST** be emittable in machine-readable form from the running chain, and the emitted document **MUST** be digestible to a stable value. See 3.8. | ✅ |
| I-10 | A disposition **SHOULD** carry the digest of the profile that produced it. Nothing stamps it automatically. See section 9, gap 8. | ⚠️ |

### 3.1 Projection must not mutate in place (I-3)

`Shield.dispose` takes `proposed = dict(action)` once, up front
(`shield/shield.py:138`). That is a **shallow** copy. It then threads the working
action through every invariant's `project` in sequence without copying between
them (`shield/shield.py:150`). An invariant that mutates a nested object in place
therefore corrupts `action_proposed` as well as the working action, and the
certificate becomes a false record of what the planner asked for. Every invariant
in `shield/invariants.py` returns `dict(action)`; the requirement makes that
obligation explicit rather than incidental.

### 3.2 The unreachable-constraint rule (I-4)

This is the requirement most likely to be got wrong by an implementer trying to
be helpful. When a constraint cannot be satisfied, the tempting move is to return
the closest thing to compliance and let the caller sort it out. That produces an
action which passes a shallow inspection and violates the constraint.

The reference behaviour is `ProtectedSliceFloorInvariant.project`
(`shield/invariants.py:345`). If a protected slice's PRB floor cannot be reached
even by zeroing every other slice, it gives the protected slice everything
available, emits a `ConstraintViolation` whose message says `UNREACHABLE`, and
**leaves the invariant unsatisfied** so that `is_feasible()` stays `False` and the
pre-emit guard chain refuses. Failing closed through the refusal path that already
exists beats inventing a second one that callers do not expect.

### 3.3 Ordering (I-5)

`Shield` composes an *ordered* list (`shield/shield.py:50`) and the order carries
meaning:

1. numeric-domain sanity first, because NaN comparisons are false in surprising
   ways and a negative bandwidth can appear to fit inside a band
   (`shield/invariants.py:53`);
2. lawful intercept second, fail-closed, so a refusal cannot be projected away;
3. spectrum and power next, because they define the legal envelope;
4. AI-PHY envelopes last, because their projection sets a *fallback block* and a
   fallback must not re-open a spectrum violation.

This is enforced by construction in `default_terrestrial_shield`
(`shield/shield.py:237`) and `default_ntn_shield` (`:279`) and by nothing else.
A chain assembled by hand can be in any order, and the certificate records the
order it ran in but not whether that order was intended. `emit_profile` makes the
order an inspectable, digestible artefact (3.8) so a *reordered* chain is
detectable after the fact, but nothing declares the intended order up front.
Hence ⚠️.

### 3.4 Idempotent projection (I-6)

Each pass of the projection loop calls `project` on **every** invariant in the
chain, not only on the violated ones (`shield/shield.py:150`). A projection that
perturbs an already-satisfied action therefore fights the other invariants and
burns the pass budget (`ShieldConfig.max_passes = 8`, `shield/shield.py:39`).

### 3.5 Fail closed on malformed input (I-7)

Raising is acceptable. `Shield.check` catches `TypeError`, `ValueError`,
`OverflowError` and `ZeroDivisionError` and records the invariant as
**unsatisfied** with `unit="error"` (`shield/shield.py:80`); the projection loop
does the same and additionally stamps `emit_blocked` (`shield/shield.py:154`).
Untrusted action payloads must fail closed, never crash the control loop and
never skip the remaining evidence construction.

### 3.6 Margins and units (I-8)

`SafetyCertificate.min_margin_dB` is derived by taking the minimum `margin` over
exactly those checks whose `unit == "dB"` (`shield/certificate.py:98`). An
invariant that reports Hz, a fraction, or a boolean under the label `"dB"`
silently poisons the aggregate headroom figure that a reviewer reads first.

### 3.7 The declared invariant set

Eight invariants ship today. The chain membership column is the honest one.

| Invariant id | Class | Terrestrial default chain | NTN default chain | Exported from `shield/__init__.py` |
| --- | --- | :---: | :---: | :---: |
| `numeric_domain_sanity` | `NumericSanityInvariant` | ✅ | ✅ | ✅ |
| `lawful_intercept` | `LawfulInterceptInvariant` | ✅ when an `LIConstraint` is passed | ✅ when an `LIConstraint` is passed | ✅ |
| `spectral_mask_ts38104` | `SpectralMaskInvariant` | ✅ | ✅ | ✅ |
| `max_eirp` | `MaxEirpInvariant` | ✅ | ✅ | ✅ |
| `neural_rx_envelope` | `NeuralRxEnvelopeInvariant` | ✅ | ✅ | ✅ |
| `constellation_legality` | `ConstellationLegalityInvariant` | ✅ | ❌ not in this chain | ✅ |
| `pfd_ceiling_ntn` | `PfdCeilingInvariant` | ❌ not in this chain | ✅ (`shield/shield.py:286`) | ✅ |
| `protected_slice_floor` | `ProtectedSliceFloorInvariant` | ❌ | ❌ | ❌ |

`protected_slice_floor` is in **neither** default chain, is **not** exported from
`shield/__init__.py` or from `shield/invariants.py`'s `__all__`, and its only
consumer in the entire tree is `tests/test_protected_slice_floor.py`. It has
never run in a pipeline. It is the one invariant in the set that bounds a
*service* commitment rather than a regulatory limit, which makes its absence from
the wired chains the most consequential omission in the layer.

The lawful-intercept row is conditional by design: both default chains include it
only when an `LIConstraint` is supplied, and omitting it is documented as lab use
only (`shield/shield.py:223`).

### 3.8 Emitting a deployment's profile (I-9, I-10)

A Shield's regulatory claim lives entirely in its constructor arguments: band
edges, EIRP ceiling, PAPR ceiling, neural-RX tolerance, PFD mask, protected slice
floor. Until WP1 those numbers existed only at a call site, so an auditor asking
what a given cell's Shield was enforcing on a given day had no artefact to read.

`emit_profile(shield, profile_id=...)`
(`src/horizon_ric/assurance/profile.py:139`) reflects a **live** Shield instance
into a JSON-serialisable document carrying `schema_version`, `profile_id`,
`invariant_chain` (ordered, one descriptor with its limits per invariant),
`max_passes`, and the `action_contract` key tuples from section 2. It is derived
from the live object rather than hand-maintained, so it cannot drift from what is
being enforced. `profile_digest` hashes it through the **same** canonical JSON
rules as the certificate signer (4.1), so a deployment can pin its enforced
envelope in the units its signed certificates are already denominated in, and
detect a silently reconfigured Shield after a redeploy. Two chains holding the
same invariants in a different order produce different digests, because JSON
arrays are ordered and `sort_keys` never reorders array elements.

One field is deliberately not rendered: `LawfulInterceptInvariant.li` holds the
operator's LIMF rule catalogue and is recorded as delegated rather than expanded
(`assurance/profile.py:61`). Warrant content is not audit material for an rApp.

What is missing is the link in the other direction. No pipeline calls
`emit_profile`, and no certificate carries a profile digest unless a caller stamps
it into `model_provenance` by hand. That is requirement I-10 and gap 8.

---

## 4. The safety certificate

Reference implementation:
[`src/horizon_ric/shield/certificate.py`](../../src/horizon_ric/shield/certificate.py).
Signing: [`src/horizon_ric/shield/signing.py`](../../src/horizon_ric/shield/signing.py).
Machine-readable form:
[`docs/schemas/safety-certificate-v1.schema.json`](../schemas/safety-certificate-v1.schema.json).

`SafetyCertificate` is a **stdlib frozen dataclass, not a pydantic model**
(`shield/certificate.py:73`), as are `InvariantCheck` (`:49`) and
`ConstraintViolation` (`:26`). That is deliberate: the module is imported on the
real-time path and carries no third-party dependency. It also means there is no
runtime validation of field vocabularies, which section 4.3 records.

It has **18 fields** plus **one derived property**, `min_margin_dB`, so
`to_dict()` emits **19 keys** (`shield/certificate.py:104`).

| Key | Type | Meaning |
| --- | --- | --- |
| `decision_id` | string | Joins the certificate to the DecisionRecord and to `rapp_metadata.decision_id` on the A1 body |
| `issued_at` | string | ISO-8601 UTC |
| `loop_tier` | string | `"rt"` / `"near_rt"` / `"non_rt"`, conventional only (4.3) |
| `block` | string | The AI-PHY or RIC block that produced the proposal |
| `action_proposed` | object | The planner's dict, verbatim |
| `action_safe` | object | What the chain will let through |
| `invariants` | array | One `InvariantCheck` per invariant, **in chain order** |
| `corrections` | array | One `ConstraintViolation` per correction applied |
| `violated_ids` | array of string | Residual violations after the final evaluation |
| `projected` | bool | The action was corrected at least once |
| `fallback_used` | bool | A certified classical fallback was substituted |
| `fallback_to` | string or null | Which fallback |
| `safe` | bool | Final action satisfies every hard invariant |
| `emit_blocked` | bool | The chain could not make the action safe |
| `min_margin_dB` | number or null | **Derived**: min `margin` over checks with `unit == "dB"` |
| `rng_seed` | integer or null | Replay seed |
| `model_provenance` | object or null | Model identity and versions |
| `signature` | string or null | Ed25519 signature, hex |
| `signing_key_fingerprint` | string or null | SHA-256 of the raw public key |

`InvariantCheck` is `{invariant_id, satisfied, margin, unit, detail}`;
`ConstraintViolation` is `{constraint_id, severity, margin_dB, message}`.

| Id | Requirement | Status |
| --- | --- | --- |
| C-1 | A certificate **MUST** carry all 19 keys, with these names and JSON types, including explicit `null` for absent optional values. An implementation in another language **MUST NOT** drop keys it has no value for. | ✅ |
| C-2 | The canonical signing form **MUST** be exactly as specified in 4.1. | ✅ |
| C-3 | `min_margin_dB` **MUST** be derived from `invariants`, never transported as an independent claim. A verifier **SHOULD** recompute it and treat a mismatch as tampering. | ⚠️ recomputation is not implemented |
| C-4 | Verification **MUST** be over the canonical bytes, and a missing, malformed, or invalid signature **MUST** verify as false rather than raise (`shield/signing.py:129`). | ✅ |
| C-5 | Both `action_proposed` and `action_safe` **MUST** be present. The delta between them is the audit object; a certificate carrying only the safe action cannot be reviewed. | ✅ |
| C-6 | `invariants` **MUST** be in chain-execution order, so the order is recoverable from the certificate alone. | ✅ |
| C-7 | A certificate **SHOULD** record the `context` the chain evaluated against. It does not. See section 9, gap 7. | ❌ not implemented |
| C-8 | A certificate **SHOULD** carry the `profile_digest` (3.8) of the invariant chain that produced it, under `model_provenance`, so the enforced bounds are recoverable from the certificate alone. A caller can stamp it; nothing does automatically. | ⚠️ |

### 4.1 Canonical signing form (C-2)

```
payload = cert.to_dict()               # 19 keys
payload.pop("signature")               # a signature cannot cover itself
payload.pop("signing_key_fingerprint")
bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
```

* Algorithm: **Ed25519** over those bytes.
* `signature`: the raw 64-byte signature, **lower-case hex**.
* `signing_key_fingerprint`: **SHA-256 hex of the raw 32-byte public key**
  (`serialization.Encoding.Raw` / `PublicFormat.Raw`, `shield/signing.py:103`).
  It is not a SHA-256 of a PEM, a DER, or an SPKI wrapper, and implementations
  that fingerprint an encoding wrapper will not interoperate.
* The two signature fields are removed rather than nulled, so signing and
  verification agree regardless of whether the certificate has already been
  stamped (`shield/signing.py:42`).
* Signing keys **MUST NOT** be generated implicitly at runtime. A configured but
  unreadable key path is a hard startup error (`shield/signing.py:78`,
  `rapp/pipeline.py:207`).

The consequence of `sort_keys=True` with no `default=` handler is requirement P-3.
The consequence of embedding `action_proposed` verbatim is that the planner's
output shape is part of the signed evidence format, which is why a profile has to
constrain it at all.

### 4.2 What signing does and does not prove

A valid signature proves that the holder of one private key emitted this exact
certificate. It proves nothing about whether the invariant chain was the right
one, whether the bounds were the licensed bounds, or whether the measurement in
`context` was honest. Sections 3.7, 3.8 and 9 are where those questions live.

### 4.3 Conventional vocabularies that the code does not enforce (C-3 note)

* `ConstraintViolation.severity` is annotated `"hard" | "soft"` in a comment
  (`shield/certificate.py:36`) and is a bare `str` at runtime. Every violation
  constructed in `shield/invariants.py` passes `"hard"`; nothing rejects any other
  string. It *is* enforced one layer downstream, where the evidence record types
  it as `Literal["hard", "soft"]` (`src/horizon_ric/evidence/schema.py:97`) and
  the spectrum pipeline coerces anything else to `"hard"`
  (`src/horizon_ric/spectrum/pipeline.py:174`). A conformant certificate producer
  **MUST** emit one of the two strings; do not rely on the certificate layer to
  catch it.
* `loop_tier` has a declared tuple, `LOOP_TIERS = ("rt", "near_rt", "non_rt")`
  (`shield/certificate.py:23`), which is exported and **never read**.
  `Shield.dispose` defaults it to `"non_rt"` and accepts any string
  (`shield/shield.py:133`).

---

## 5. Certificate lifetime and the evidence chain

| Path | What is persisted | Where |
| --- | --- | --- |
| rApp decision loop | A **three-field subset** of the certificate plus signature fields when signed | `src/horizon_ric/rapp/pipeline.py:422` |
| Spectrum loop | The **full** `cert.to_dict()` | `src/horizon_ric/spectrum/pipeline.py:187` |
| A1 wire | Nothing. See section 6 | — |

The rApp side-channel carries `safe`, `projected`, `violated_ids`, and adds
`signature` and `signing_key_fingerprint` only when a signing key is configured
(`rapp/pipeline.py:426`). This is enough to answer "was this decision shielded and
did anything remain violated" and not enough to answer "what were the margins".
The full certificate is persisted only by `src/horizon_ric/spectrum/pipeline.py`,
which never touches A1.

---

## 6. A1 binding

This section is split because the honest answer is split. **The certificate does
not cross the A1 wire.** What exists is a local gate, a guard chain, and a
side-channel.

### 6.1 Implemented today

| Mechanism | What it does | Where | Status |
| --- | --- | --- | --- |
| Env-gated local refusal | When `HORIZON_A1_REQUIRE_CERT` is truthy, `emit_policy` raises `ValueError` if no certificate accompanies the emit, or if `not cert.safe or cert.emit_blocked`. **Default OFF** | `src/horizon_ric/rapp/a1_adapter.py:336` | ⚠️ off by default |
| Pre-emit guard chain | `run_guard_chain` runs before the A1 call; guard ids `missing_safety_certificate` and `shield_blocked` refuse an uncertified or unsafe decision | `src/horizon_ric/policy/emit_guards.py:46` and `:138`, called from `src/horizon_ric/rapp/pipeline.py:376` | ✅ |
| DecisionRecord side-channel | Three certificate fields (`safe`, `projected`, `violated_ids`) plus signature and fingerprint when signed, persisted into the evidence chain, joined to the A1 policy by `decision_id` | `src/horizon_ric/rapp/pipeline.py:422` | ✅ |
| Certificate on the A1 policy body | Does not exist | — | ❌ |

Why the last row is ❌ and not 🟡: it is not a missing implementation of an agreed
field, it is a missing field. `A1Adapter._policy_create_schema`
(`src/horizon_ric/rapp/a1_adapter.py:618`) emits JSON Schema **Draft-07** with
`"additionalProperties": false` **at every level**, for all four registered policy
types. The only cross-cutting envelope is `rapp_metadata`, whose properties are
exactly `{decision_id, rapp_version, model_versions}`, whose `required` is
`["decision_id"]`, and which is itself `additionalProperties: false`
(`a1_adapter.py:625`). There is no slot for a certificate, an invariant list, or a
margin in **any** registered A1 policy type. A near-RT RIC validating the emitted
body against the registered schema would reject a certificate-bearing policy.

So the guarantee that reaches the near-RT RIC today is negative and local: an
unshielded or unsafe decision is *not emitted*. The near-RT RIC cannot tell the
difference between a policy that was shielded and one that was not. It receives a
`decision_id`, and a verifier joins that to the certificate out of band, from the
evidence chain.

### 6.2 Proposed for standardisation

Requirements in this subsection describe a **future** A1 profile version. None of
it is implemented; the field does not exist.

| Id | Proposed requirement | Status |
| --- | --- | --- |
| A-1 | An A1 policy body **should** be able to carry, alongside the policy data, a verifiable reference to the certificate for the decision that produced it. | 🟡 |
| A-2 | The mandatory minimum **should** be a digest form: `decision_id`, `signature`, `signing_key_fingerprint`. It is small, fixed-size, and sufficient for a near-RT RIC to log and for a third party to later demand the matching certificate. | 🟡 |
| A-3 | The full certificate body **should** be optional. A near-RT RIC does not need per-invariant margins to enforce a policy, and mandating a variable-size audit object on a latency-sensitive interface is the wrong trade. | 🟡 |
| A-4 | The change **must** be versioned. Adding a key to `rapp_metadata` is a breaking change for any consumer validating with `additionalProperties: false`, so it requires a new policy type id or a `schema_version` bump. Horizon's own types carry `schema_v: "1.0.0"` (`a1_adapter.py:56`). | 🟡 |
| A-5 | A near-RT RIC that receives a certificate reference **should** be able to refuse a policy whose reference does not verify, and that refusal **should** be reportable over the existing A1AP policy-status surface rather than through a new mechanism. | 🟡 |

An illustrative shape, for discussion only. This is **not** a registered schema
and no code emits it:

```json
"rapp_metadata": {
  "decision_id": "8c2f…",
  "rapp_version": "0.2.0",
  "model_versions": {"policy": "risk_band_rules_v1"},
  "safety_certificate_ref": {
    "decision_id": "8c2f…",
    "signature": "…hex…",
    "signing_key_fingerprint": "…sha256 hex…"
  }
}
```

### 6.3 Dialects

The adapter enforces **five** A1 dialects: `legacy`, `osc`, `osc_a1`, `eiap`,
`mantaray` (`src/horizon_ric/rapp/a1_adapter.py:160`). See
[`docs/SMO_INTEGRATION.md`](../SMO_INTEGRATION.md) for the URL surfaces and body
shapes. The dialect choice is orthogonal to this profile: the certificate is
absent from all five, and adding it would be a per-dialect body change on top of
the schema change in 6.2.

---

## 7. E2 binding

**Receive-only.** `src/horizon_ric/e2/__init__.py:5` states it: the package
"deliberately contains NO E2AP/SCTP transport of its own: the near-RT RIC owns the
E2 association; Horizon consumes the E2SM payloads it surfaces." The package
exports exactly three names: `KpmBridgeError`, `KpmMeasurementBridge`,
`e2_to_pipeline`.

| E2 element | Status | Detail |
| --- | --- | --- |
| E2SM-KPM v03.00 `E2SM-KPM-IndicationMessage` decode | ✅ | Format 1 and 2 yield one event; Format 3 yields one event per `ueMeasReportList` entry (`src/horizon_ric/e2/kpm_bridge.py:401`) |
| E2SM-KPM v03.00 `E2SM-KPM-IndicationHeader` decode | ⚠️ Format 1 only | Reads `colletStartTime` (the spec's own typo, reproduced faithfully) and `senderName` (`kpm_bridge.py:394`) |
| E2AP / SCTP transport | ❌ | The near-RT RIC owns the E2 association, by design |
| RIC Subscription, `ActionDefinition`, `EventTriggerDefinition` | ❌ | Not decoded, not encoded, not vendored |
| E2SM-RC v1.03 `ControlHeader` / `ControlMessage` **encode** | ✅ | `controlHeader-Format1` and `controlMessage-Format1` constructed in aligned PER and decoded back through the same spec (`src/horizon_ric/e2/rc_control.py`, `tests/test_e2_rc_control.py`) |
| E2SM-RC **fail-closed** on a refused disposition | ✅ | `control_from_disposition` raises rather than encoding when the certificate is `emit_blocked`, not `safe`, or carries `violated_ids`; it encodes `safe_action`, never the proposal |
| E2SM-RC control **delivery** | ❌ | No `RICcontrolRequest`, no E2AP, no transport. See `deploy/e2-companion/E2_RC_PROOF.md` for the three independent reasons |
| E2SM-RC RAN Parameter ID assignment | ⚠️ | The encoder's default IDs are local placeholders. The spec assigns them per node via `RANFunctionDefinition-Control-Action-Item`; a real deployment must read them from the E2 node and pass `parameter_ids=` |

The only vendored ASN.1 is
[`src/horizon_ric/e2/asn1/e2sm_kpm_v03.00_standard.asn1`](../../src/horizon_ric/e2/asn1/e2sm_kpm_v03.00_standard.asn1),
a byte-for-byte copy from the FlexRIC `dev` tree at commit
`ef6d722f22191eea74089966983da1f5ec1fedd4`, sha256
`ab473a9cc60bc3e3b1e32d032ce3b7158175c210282a006d662b1c48bc8a573a`, containing
the modules `E2SM-KPM-IEs` and `E2SM-COMMON-IEs`. Provenance and the license note
are in
[`src/horizon_ric/e2/asn1/PROVENANCE.md`](../../src/horizon_ric/e2/asn1/PROVENANCE.md).
It is compiled at runtime by `asn1tools` with `codec="per"`, which is aligned PER,
the transfer syntax E2SM-KPM mandates (`kpm_bridge.py:71`).

| Id | Requirement | Status |
| --- | --- | --- |
| E-1 | Measurement decode **MUST** be name-agnostic: keyed by `measName` when present, falling back to `measID.<n>`, so an unrecognised metric is carried rather than dropped (`kpm_bridge.py:200`). | ✅ |
| E-2 | A derived risk or utility signal **MUST** declare a versioned mapping id on every event it produces. Horizon emits `kpm_risk_v1` under `payload["risk"]["mapping"]` (`kpm_bridge.py:95`). | ✅ |
| E-3 | The mapping **MUST** state which metrics it interprets and what happens when none is present. See 7.1. | ✅ |
| E-4 | An implementation claiming this profile's E2 binding **MUST NOT** describe a KPM-only ingest path as closed-loop control. | ✅ |
| E-5 | The ASN.1 module used for decode **MUST** be committed with provenance and a hash, and the hash **SHOULD** be checked in CI (`tests/test_e2_kpm_bridge.py`). | ✅ |

### 7.1 The `kpm_risk_v1` mapping (E-3)

Decode is name-agnostic, but only **five** TS 28.552 metrics are *interpreted*,
by `_risk_contributions` (`kpm_bridge.py:111`):

| Metric | Contribution |
| --- | --- |
| `RRU.PrbTotDl` | `prb / 100` (percent utilisation) |
| `DRB.RlcSduDelayDl` | `delay / 50.0` (50 ms budget) |
| `DRB.UEThpDl` | `1 - thp / 10000.0` (10 Mbps SLA floor, kbps) |
| `DRB.PdcpSduVolumeDL` | `vol / 100000.0` (100 Mbit reference load, kbit) |
| `DRB.RlcSduTransmittedVolumeDL` | `vol / 100000.0` |

Every contribution is clamped into `[0, 1]`, and the final risk is the **maximum**
contribution: the worst signal wins, which is conservative by design. When none of
the five is present the floor `RISK_FLOOR = 0.05` is used and the dominant metric
is reported as `null` (`kpm_bridge.py:104`, `:130`). Any other metric present in
the indication is carried into `payload["measurements"]` and ignored by the risk
mapping.

### 7.2 E2SM-RC: payloads are constructed and gated; delivery is not

Version 0.1.0 of this document said there was no E2SM-RC anywhere in this
repository. That was true when it was written and false a few commits later, so
it is corrected here rather than left to be discovered.

What exists now. The O-RAN E2SM-RC v1.03 standard ASN.1 is vendored at
[`src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn`](../../src/horizon_ric/e2/asn1/e2sm_rc_v1_03_standard.asn),
byte-for-byte from the same pinned FlexRIC commit as the KPM spec, and
[`src/horizon_ric/e2/rc_control.py`](../../src/horizon_ric/e2/rc_control.py)
compiles it under aligned PER to construct `controlHeader-Format1` and
`controlMessage-Format1` payloads, decoding each one back through the same spec
to the same fields. The provenance argument is stronger than a from-scratch
implementation would give: FlexRIC's own asn1c wire codec for RC was generated
from this same file, so the bytes an E2 node parses and the bytes constructed
here derive from one source text.

**The requirement this adds to the profile.** A conformant implementation must
make the E2 control path subject to the same enforcement as the southbound policy
path. Concretely: the only function that turns a decision into E2SM-RC bytes must
refuse when the certificate is `emit_blocked`, when it is not `safe`, or when
`violated_ids` is non-empty; and when it succeeds it must encode the *projected*
action, never the proposal. Otherwise adding a control path creates a second,
unguarded route to the radio beside the one the Shield governs — which would be a
net loss in assurance despite looking like a gain in capability.
`control_from_disposition` implements this and `tests/test_e2_rc_control.py`
pins both directions.

**What is still out of scope.** Delivery. There is no `RICcontrolRequest`, no
E2AP, and no transport in this package — the near-RT RIC owns the E2 association
by design. [`E2_RC_PROOF.md`](../../deploy/e2-companion/E2_RC_PROOF.md) records
the three independent reasons a delivered control action cannot be demonstrated
here and does not claim one. Binding a certificate to an *accepted and acted upon*
E2 control action remains WP4 work.

**One honest limitation a reader must not miss.** The encoder's default RAN
Parameter IDs are local placeholders, not specification constants. E2SM-RC
assigns them per E2 node through
`RANFunctionDefinition-Control-Action-Item.ran-ControlActionParameters-List`,
which pairs each `ranParameter-ID` with its `ranParameter-name`. A real
deployment must read that list off the node it is controlling and pass
`parameter_ids=`; the defaults exist so the encode path is testable, not because
they mean anything on a live E2 interface.

---

## 8. Conformance criteria a third party could check

Each row is something an outside party can run or inspect without asking us
anything. This is the part of the profile that would have to survive contact with
a real conformance programme.

| Criterion | How to check | Status |
| --- | --- | --- |
| The action contract matches its schema | [`docs/schemas/proposed-action-v1.schema.json`](../schemas/proposed-action-v1.schema.json) pinned against `assurance/planner.py` by `tests/test_assurance_schemas.py` | ✅ |
| The certificate matches its schema | [`docs/schemas/safety-certificate-v1.schema.json`](../schemas/safety-certificate-v1.schema.json) pinned against `shield/certificate.py` by `tests/test_assurance_schemas.py`; the 19 keys are asserted, not described | ✅ |
| A deployment's invariant chain and its numeric bounds are readable and pinnable | `emit_profile` / `profile_digest` in [`src/horizon_ric/assurance/profile.py`](../../src/horizon_ric/assurance/profile.py), validated against [`docs/schemas/invariant-profile-v1.schema.json`](../schemas/invariant-profile-v1.schema.json) by `tests/test_assurance_schemas.py` (`test_emit_profile_validates_against_the_published_profile_schema`, `test_profile_digest_is_stable_across_calls`) | ⚠️ emittable and digestible; no deployment pins a digest and no certificate references one |
| **Gate G1**: a second, independently written planner is shielded with neither side modified | [`src/horizon_ric/planners/ucb_spectrum.py`](../../src/horizon_ric/planners/ucb_spectrum.py), re-executed and checked by [`scripts/verify_g1_second_planner.py`](../../scripts/verify_g1_second_planner.py) in `.github/workflows/test.yml:28`. The verifier pins the SHA-256 of `shield.py`, `invariants.py` and the planner, so editing either side to make the numbers agree fails the gate instead of passing it | ✅ |
| A signed certificate round-trips, and a tampered one does not verify | `tests/test_shield_signing.py` (`test_generate_sign_verify_roundtrip`, `test_tampered_certificate_fails_verification`, `test_wrong_key_fails_verification`) | ✅ |
| Every action the Shield emits satisfies every programmed invariant | `tests/test_shield_properties.py::test_any_emitted_random_action_satisfies_every_programmed_invariant` (randomised) | ✅ |
| Malformed RF values fail closed rather than crash | `tests/test_shield_properties.py::test_nonfinite_or_malformed_rf_values_fail_closed` | ✅ |
| An uncertified or unsafe decision does not reach A1 | `tests/test_pipeline_defects.py` (the `HORIZON_A1_REQUIRE_CERT` gate) and the guard-chain refusal path in `tests/test_pipeline_wiring.py` | ✅ |
| KPM decode is against the committed spec, and the spec matches its upstream hash | `tests/test_e2_kpm_bridge.py::test_committed_asn1_spec_matches_flexric_provenance` plus decode tests over captured FlexRIC PDUs in `tests/fixtures/e2sm_kpm/` | ✅ |
| The A1 body shape per dialect | `tests/test_a1_osc_dialect.py`, `tests/test_a1_osc_a1_dialect.py`, `tests/test_a1_eiap_dialect.py`, `tests/test_a1_mantaray_dialect.py`, `tests/test_multi_vendor_integration.py` | ✅ |
| Every claim in this document resolves to a file | `scripts/check_doc_links.py`, `tests/test_docs_links.py` | ✅ |
| An external party reproduces the headline enforcement result from the published archive without contacting us | Gate G4, WP4 | 🟡 |

What a third party **cannot** check today: that the certificate reached the near-RT
RIC (it does not, section 6), that a vendor platform accepts this profile (none
has, section 9), and that the invariant bounds recorded *in a certificate* are the
licensed bounds (inside a certificate they appear only in free-text `detail`;
requirement C-8).

---

## 9. Known gaps

Listed because a profile whose gaps are unlisted is marketing. Each gap is a
candidate work item for profile version 0.2.0.

1. **The `Invariant` conformance gate is weak.** `Invariant` is a
   `runtime_checkable` structural `Protocol`, not a base class
   (`shield/invariants.py:39`). `isinstance(x, Invariant)` therefore checks only
   that the attribute names exist. It does not check signatures, return types, or
   any of the semantics in section 3: an object with an `id` and two callables that
   return nonsense passes. Worse, no code in the tree performs that `isinstance`
   check at all, so invariant conformance currently rests entirely on the test
   suite and on review. A future version should pin the shape with an explicit
   conformance test-kit that a third-party invariant must pass.

2. **`max_passes` exhaustion is indistinguishable from convergence with a residual
   violation.** The projection loop runs at most `ShieldConfig.max_passes = 8`
   passes (`shield/shield.py:39`, `:144`). Both "the loop ran out of budget" and
   "the loop converged but a hard violation remains" surface identically in the
   certificate, as a non-empty `violated_ids` with `emit_blocked=True`
   (`shield/shield.py:169`). The certificate records neither the pass count nor
   the reason the loop stopped. Operationally both refuse the emit, so nothing
   unsafe ships; diagnostically they are different failures and the audit record
   cannot tell them apart. A future version should add a termination reason and a
   pass count to the certificate.

3. **One invariant has never run, and one runs only in the NTN chain.**
   `pfd_ceiling_ntn` is in the NTN default chain (`shield/shield.py:286`) and not in
   the terrestrial one, which is defensible; `protected_slice_floor` is in
   **neither**, is not exported from
   `shield/__init__.py`, and its only consumer in the whole tree is
   `tests/test_protected_slice_floor.py`. It has never run in a pipeline. The one
   invariant that bounds a service commitment rather than a regulatory limit is
   also the one that is unwired.

4. **The certificate is absent from the A1 wire.** Section 6. The registered A1
   schemas cannot carry it, the field to carry it does not exist, and the
   guarantee that reaches the near-RT RIC is therefore local and negative: unsafe
   decisions are not emitted, but a policy that arrives carries no evidence that
   it was shielded.

5. **No vendor platform has onboarded this.** Zero. The dialect implementations are
   validated against publicly documented surfaces with mock transports, and the
   partner-portal portions of the Ericsson EIAP and Nokia MantaRay contracts are
   not accessible to us (see [`docs/SMO_INTEGRATION.md`](../SMO_INTEGRATION.md)
   §6). Until a vendor implements section 4 and rejects something, this profile is
   a proposal with one implementation.

6. **`DecisionPipeline` does not use the planner interface.** The interface in
   `assurance/planner.py` is published but unwired; the production pipeline still
   hard-codes its planner as private methods and builds its own Shield
   (`rapp/pipeline.py:197`, `:226`, `:273`). So the contract in section 2 is
   enforced today only for callers that go through `shielded(...)`, which means
   the G1 verifier and the tests, not the rApp.

7. **The certificate does not record the context it was evaluated against.**
   `Shield.dispose` takes a `context` and stores none of it
   (`shield/shield.py:126`). `NeuralRxEnvelopeInvariant`'s entire verdict depends
   on `context["measured_tbler"]` (section 2.3), so a verifier holding a valid
   signed certificate still cannot reproduce that verdict. This is the most
   serious of the auditability gaps.

8. **Nothing links a certificate to the profile that produced it.** WP1 closed half
   of this: `emit_profile` and `profile_digest` (`assurance/profile.py`, section
   3.8) make a live Shield's bounds machine-readable and pinnable. The other half
   is open. No pipeline emits a profile, no deployment pins a digest, and the
   certificate carries no profile reference unless a caller stamps one into
   `model_provenance` by hand. Inside a certificate, the bounds a chain enforced
   still appear only in the free-text `detail` string of each `InvariantCheck`, so
   a verifier holding only a certificate cannot check the bounds against a licence
   without parsing prose. Requirements I-10 and C-8.

9. **Field vocabularies are unenforced at the certificate layer.** `severity` and
   `loop_tier` are conventional strings (section 4.3). `LOOP_TIERS` is declared and
   never read.

10. **No conformance test-kit is published.** Section 8 lists checks that live in
    this repository's own test suite. A profile that expected third-party
    implementations would ship an executable kit that does not depend on this
    repository. That is not written.

---

## 10. Versioning of this profile

* This is version **0.1.0**. It will change incompatibly.
* Requirement ids (`P-n`, `I-n`, `C-n`, `A-n`, `E-n`) are stable once published: a
  requirement that is withdrawn is marked withdrawn rather than reused.
* The certificate schema is versioned separately, in
  [`docs/schemas/safety-certificate-v1.schema.json`](../schemas/safety-certificate-v1.schema.json).
  A change to the 19 keys is a new schema version and a new profile version.
* Where this document and the code disagree, **the code is right and this document
  is a bug.** Every claim above cites a file and a line so that the disagreement
  is findable.
