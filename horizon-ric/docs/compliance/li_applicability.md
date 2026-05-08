# Lawful Intercept (LI) Applicability Memo — PreceptualAI rApp

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** Operator legal counsel, LI architect, sign-off authority
**Component under review:** PreceptualAI rApp (`src/horizon_ric/`)
**Status:** Pre-pilot — produced for Tier-1 operator sign-off

---

## 1. Purpose and scope

This memorandum is read by the operator's Lawful Intercept architect, the operator's legal counsel, and (by reference) by any national LI sign-off authority. It establishes whether PreceptualAI requires LI accreditation, what surface it exposes to the operator's Lawful Intercept Mediation Function (LIMF), and what the operator must configure on its side before authorising the rApp to emit A1 policies in production.

The memo is scoped to PreceptualAI v0.2.0 as deployed at the Non-RT RIC (rApp tier). It does NOT cover the operator's underlying Near-RT RIC, gNB, UPF, or LIMF — those remain the operator's accredited LI components.

## 2. Statutory framework

The framework below is cited because each of these instruments imposes distinct obligations on the operator that an rApp can disturb if it acts without constraint:

- **3GPP TS 33.126 (LI Requirements), Release 18** — §4 establishes the principle that any management or optimisation function affecting bearer routing must not be permitted to subvert active warrants.
- **3GPP TS 33.127 (LI Architecture and Functions), Release 18** — §5.4 names the LIMF as the authoritative orchestrator of intercept; §6.2 requires that the Administration Function (ADMF) holds the catalogue of protected targets and that no third-party function (an rApp is third-party from the LIMF's perspective) may relocate a protected target across jurisdictional anchors. §7.1 sets the non-interference principle for management plane functions.
- **3GPP TS 33.128 (LI Stage 3), Release 18** — §5 / §6 define the X1, X2, X3 handover interfaces. PreceptualAI consumes none of these interfaces and emits onto none of them.
- **ETSI TS 103 221-1 / -2** — internal LI handover; same non-interference principle.
- **USA**: CALEA (47 U.S.C. §1001 et seq.) — operator-side obligation; an rApp that disrupts intercept capability creates operator liability under §103.
- **EU**: Directive 2002/58/EC (e-Privacy) Art. 15(1) and Directive (EU) 2016/680 (Law Enforcement Data Directive) — define the lawful basis under which interception data is processed; rApp must not relocate a target out of the warrant's competent jurisdiction.
- **UK**: Investigatory Powers Act 2016, Part 9 (Telecommunications Operator obligations) — Technical Capability Notices may name jurisdictional anchoring requirements.
- **Singapore**: Telecommunications Act 1999 (Cap. 323) and the Telecommunications (Class Licences) Notifications — Singapore IMDA imposes operator-side LI obligations on Class Licence holders.

## 3. PreceptualAI's LI architecture position

PreceptualAI is an **A1 policy emitter at the Non-RT RIC (rApp) tier**. Per O-RAN.WG2.NON-RT-RIC-ARCH and O-RAN.WG2.A1AP-v05.00 §6.3, the rApp issues policy intents (PUT /policies); the actual user-plane traffic is handled by the Near-RT RIC, the gNB-CU/DU, and the UPF.

Concretely:

- PreceptualAI **never sees subscriber traffic, IMSI/SUPI, or content of communication (CC) or intercept-related information (IRI).** The rApp consumes 3GPP TS 28.552 KPIs and TS 28.541 NRM via the O1 adapter (`src/horizon_ric/rapp/o1_adapter.py`); these are management-plane counters, not subscriber data.
- PreceptualAI emits A1 policy intents to the Near-RT RIC. These intents express constraints over slices, gateway placements, and target jurisdictions — never per-IMSI mappings.
- PreceptualAI is therefore **not an LI function** under TS 33.127 §5.4. It is an "external optimisation function" that, *if unconstrained*, could nevertheless disturb the LIMF by emitting a policy that re-anchors a slice carrying a protected UE.

This memo concerns that residual disturbance surface.

## 4. The five protections PreceptualAI implements

### 4.1 LI jurisdiction constraint — `src/horizon_ric/policy/li_constraint.py:78-207`

Class `LIConstraint` (line 78) refuses any action that:

- intersects a protected UE-ID set (`check_feasibility`, line 113-127), or
- would anchor a protected slice in a jurisdiction not on the operator's `allowed_jurisdictions` list (line 129-146).

The constraint is registered as a **hard** constraint (line 100, `hard_constraint_ids` returns `["li_protected_ue", "li_jurisdiction"]`). Per the design comment at lines 30-34, the constraint is hard-only by construction: `lagrangian_violation` (line 194-207) returns a zero tensor unconditionally so that the policy network cannot trade off LI compliance against any reward. **Soft Lagrangian penalties for LI are explicitly forbidden** — a regulator-mandated intercept is never a tunable trade-off.

The rule catalogue (`LIJurisdictionRule`, line 56-75) holds only opaque tags (`rule_id`), UE-ID sets, slice-ID sets, and ISO-3166 alpha-2 jurisdiction codes. **The rApp never sees warrant content** (named in the docstring at line 18-20).

### 4.2 Per-tenant evidence chain — `src/horizon_ric/evidence/store.py:82-142`

The abstract `EvidenceStore` (line 82) stamps every `DecisionRecord` with a `tenant_id` (resolved from the active `TenantScope`) and computes the SHA-256 hash chain **per tenant** (line 113-121, `verify`; line 123-142, `verify_tenant`). When an auditor for tenant A queries the store within a `TenantScope("A")`, only tenant-A records are yielded (`__iter__` line 212-227). Tampering with tenant B's chain therefore does NOT break tenant A's chain — chains are independent (line 128-130).

This protects against a regulator-side concern: an auditor working on a warrant belonging to tenant A must not be able to read tenant B's audit log.

### 4.3 Hash-chain tamper detection — `src/horizon_ric/evidence/store.py:106-142`

`EvidenceStore.verify()` (line 106) walks the chain from index 0 and returns the index of the first record whose stored hash does not match `sha256(prev_hash || canonical_json(record))`. The chain formula is at line 75-79 (`_chain`) using SHA-256. The first record uses the all-zero hash as the prior (line 48). The `_canonical_json` serialiser (line 68-72) sorts keys and uses minimal separators so re-serialisation is deterministic.

This means any post-hoc edit of any historical decision record breaks every subsequent chain hash — providing tamper-evidence to the standard required by NIST SP 800-92 §4.2 and 3GPP TS 33.117 §4.2.5.2.

### 4.4 Operator override flag — `src/horizon_ric/evidence/schema.py:128-129`

`DecisionRecord.operator_override` (boolean, line 128) and `operator_override_reason` (line 129) record when a SOC operator manually overrode the rApp's decision. This is the human-in-the-loop record required by:

- 3GPP TS 33.127 §7.1 (non-interference principle requires that LI architects can pre-empt automated decisions);
- EU AI Act Art. 14 (human oversight, see `docs/compliance/eu_ai_act.md` §4.6);
- ETSI TS 103 221-1 (the operator's LI architect retains override authority).

The lifecycle owner (`src/horizon_ric/rapp/lifecycle.py:50`, class `HorizonRAppLifecycle`) maintains the boot/run/shutdown loop and accepts override events into the audit chain via the rApp dashboard API (`src/horizon_ric/rapp/api_v1.py:349`).

### 4.5 mTLS + JWT auth — `src/horizon_ric/rapp/auth.py` and `src/horizon_ric/security/jwt.py`

All R1, A1, and O1 channels run under O-RAN.WG11 §6 mTLS (`src/horizon_ric/rapp/auth.py:92-152`, `_build_ssl_context`). `production_mode=True` with `verify_tls=False` is a hard-banned combination (lines 112-117) — the constructor raises `ValueError`. The WG11 §6 cipher allow-list is at lines 34-38; it is applied at lines 155-200 with explicit honest-failure logging when OpenSSL rejects TLS-1.3 names (lines 178-200).

JWT issuance is RS256-only (`src/horizon_ric/security/jwt.py:36-37`); HS256 is not accepted. Required claims (`sub`, `tenant`, `roles`, `iss`, `aud`, `iat`, `exp`, `jti`) are enforced both at mint (lines 122-131) and at verify (lines 168-172). Zero-downtime key rotation is at lines 180-198. The middleware (`src/horizon_ric/security/middleware.py:64-162`) enforces tenant-claim equality on every request (line 117-127).

This means an LI auditor presenting a warrant-scoped JWT can only read the records in their authorised tenant.

## 5. Seven LI-affecting actions PreceptualAI CANNOT take

Per 3GPP TS 33.127 §6.2 and §7.1, the following classes of action would, if taken, disturb the LIMF. Each is blocked by a named constraint:

| # | LI-affecting action | Constraint that blocks it | Code reference |
|---|---|---|---|
| 1 | Re-route a UE flagged as a protected target | `LIConstraint._CID_UE` (hard rejection of any action with `affected_ue_ids ∩ protected_ue_ids`) | `policy/li_constraint.py:113-127` |
| 2 | Re-anchor a protected slice into a jurisdiction not on the allowed list | `LIConstraint._CID_JX` (hard rejection unless `target_jurisdiction ∈ allowed_jurisdictions`) | `policy/li_constraint.py:129-146` |
| 3 | Silently rewrite a target jurisdiction | `LIConstraint.project()` deliberately does NOT rewrite `target_jurisdiction` — it surfaces the violation loudly | `policy/li_constraint.py:150-192` (especially design comment 153-161) |
| 4 | Trade off LI compliance against throughput in the learned policy | `lagrangian_violation` returns zero unconditionally — LI is not in the reward signal | `policy/li_constraint.py:194-207` |
| 5 | Persist a decision record outside the operator's tenant chain | `EvidenceStore.append` stamps tenant from the active `TenantScope`; missing scope falls into the `_unscoped_` chain rather than mixing tenants | `evidence/store.py:185-210` and `:266-283` |
| 6 | Cross-tenant read of the audit chain by an auditor | `__iter__` filters by active `TenantScope`; the SQLite backend filters at the `WHERE` clause | `evidence/store.py:212-227`, `:285-298` |
| 7 | Skip mTLS / cert verification in production | `AuthConfig.production_mode=True` + `verify_tls=False` raises `ValueError` at context build time | `rapp/auth.py:111-117` |

All seven are exercised by the test suite (see `tests/test_li_constraint.py`, `tests/test_evidence_store.py`, `tests/test_auth_security.py`).

## 6. Two LI-affecting actions PreceptualAI COULD take if misconfigured

The constraint surface is only as good as the configuration the operator supplies. The following are honest residual risks; each requires operator-side mitigation.

### 6.1 Empty `LIJurisdictionRule` catalogue — fail-closed default (closed in v0.2.1)

**Status:** **CLOSED.** The constructor `LIConstraint(rules=[])` now defaults to `fail_closed=True` (`src/horizon_ric/policy/li_constraint.py:108`). With an empty rule catalogue under the safe default, **every action is rejected** with the synthetic constraint id `li_fail_closed` (severity `hard`). The Devil-A Finding #1 zero-config bypass is no longer reachable.

The rejection is emitted both from `check_feasibility()` (`li_constraint.py:135-152`) and from `project()` (`li_constraint.py:185-200`), so the emit-guard chain refuses to emit an action whether the planner queries feasibility directly or routes through projection.

**No-LI lab/testbed escape hatch.** A legitimate no-LI environment (a dev-bench, a CI fixture, an air-gapped maritime simulator) MUST construct the constraint as:

```python
LIConstraint(
    rules=[],
    fail_closed=False,
    deployment_audit_record="<evidence-chain id of the operator's no-LI election>",
)
```

The non-empty `deployment_audit_record` is enforced at construction time (`li_constraint.py:117-126`); a `ValueError` is raised if it is omitted. Downstream the rApp lifecycle layer persists that audit-record id alongside the deployment manifest, so an external auditor can later answer the question "on what date and on whose authority did this tenant elect to run without LI?"

**Operator deployment check (still recommended):** the operator's production pre-flight should additionally assert `len(li_constraint.rules) >= 1` and that `li_constraint.fail_closed is True`, even though those conditions are now also enforced at the code layer.

### 6.2 Stale rule catalogue

The `LIJurisdictionRule` set is held in memory at `LIConstraint` construction time (`policy/li_constraint.py:96`). If the LIMF adds a new protected UE after rApp boot, the rApp will not pick it up until the catalogue is refreshed.

**Operator mitigation:** the operator must drive a refresh path from the LIMF/ADMF into the rApp. **NOT YET — planned in Phase 2**: an O1 NETCONF subscription path that pushes `LIJurisdictionRule` updates from the operator's ADMF into a hot-swappable in-memory rule store. Today the operator has to restart the rApp (graceful — see lifecycle restart at `src/horizon_ric/rapp/lifecycle.py:164-174`) to pick up rule changes.

In the interim, the operator is responsible for ensuring rApp restart happens within the SLA defined by their LI architect (typically 1-15 minutes for warrant-list propagation).

## 7. Conclusion — deployment checklist for the operator's LI architect

Before authorising PreceptualAI to emit A1 policies in a production environment carrying intercepted bearers, the operator's LI architect MUST verify each of the following.

- [ ] PreceptualAI is registered as an external optimisation function, not as an LI function (TS 33.127 §5.4 categorisation).
- [ ] A non-empty `LIConstraint` rule catalogue is populated from the operator's LIMF/ADMF before the rApp transitions out of `RAppState.REGISTERING` (`src/horizon_ric/rapp/lifecycle.py:135`).
- [ ] The rule catalogue refresh SLA from LIMF→rApp meets the operator's LI architect's requirement (today: rApp restart; planned: live O1 push, Phase 2).
- [ ] mTLS is enabled with `production_mode=True` and `verify_tls=True` on every R1, A1, O1 channel (`src/horizon_ric/rapp/auth.py:80, 112-117`).
- [ ] The evidence store is configured per tenant, with an auditor-scoped read role gated by Casbin (`src/horizon_ric/security/rbac_policy.csv`).
- [ ] The operator's SOC has a documented override procedure that exercises `DecisionRecord.operator_override = True` and writes a non-empty `operator_override_reason` (`src/horizon_ric/evidence/schema.py:128-129`).
- [ ] A periodic chain-verification job runs `EvidenceStore.verify()` (`src/horizon_ric/evidence/store.py:106`); failures generate an Alertmanager P1 incident (`deploy/prometheus/sla_rules.yml`).

Sign-off by the operator's LI architect on the above seven items closes PreceptualAI's residual LI applicability for that operator's deployment.

---

**End of memo.** Questions to: PreceptualAI compliance lead. Spec citations are to the latest published versions of TS 33.126, 33.127, 33.128 and ETSI TS 103 221 as of 2026-05-06.
