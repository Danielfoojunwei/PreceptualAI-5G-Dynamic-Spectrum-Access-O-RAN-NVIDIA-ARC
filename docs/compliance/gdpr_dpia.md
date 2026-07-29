# GDPR Article 35 Data Protection Impact Assessment — Horizon-RIC

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** Operator Data Protection Officer (DPO), supervisory authority on request
**Component:** Horizon-RIC rApp (`src/horizon_ric/`)
**Regulation:** Regulation (EU) 2016/679 (GDPR), in particular Articles 5, 6, 25, 30, 32, and 35.

---

This template DPIA is provided so that an operator deploying Horizon-RIC can complete its own Article 35 obligation. The vendor (Horizon-RIC) is not the controller — the operator is — but the technical-side analysis below is reproducible by the vendor and is therefore documented here once. Section 5 is intentionally left for operator-side completion.

## 1. Description of processing (Art. 35(7)(a))

### 1.1 Nature of processing

Horizon-RIC ingests **management-plane telemetry** from the operator's RAN: per-cell KPIs (3GPP TS 28.552 §6, e.g. `RRU.PrbUsedDl`, `DRB.PdcpSduDelayDl`), per-slice load counters, gateway load gauges, and physical-layer propagation observables (RSRP/RSRQ rolled up per cell). It does NOT ingest IMSI, SUPI, MSISDN, IMEI, location traces, content of communication, or signalling.

Inputs flow through:

- O1 NETCONF subscriptions: `src/horizon_ric/rapp/o1_adapter.py`
- R1 service exposure: `src/horizon_ric/rapp/r1_adapter.py`

These inputs are validated against the strict schemas in `src/horizon_ric/io/schemas.py` (`TelemetryEvent`, `FeatureFrame`) and reduced to a latent `z_resource` representation. The decision pipeline (`src/horizon_ric/policy/`) emits A1 policy intents to the Near-RT RIC (`src/horizon_ric/rapp/a1_adapter.py`).

### 1.2 Identifiers handled

| Identifier | Type | Horizon-RIC handling |
|---|---|---|
| Cell-ID (NCGI / ECGI) | Quasi-identifier | Always handled. Per-cell KPIs are inherent to RAN scheduling. |
| Slice-ID (S-NSSAI) | Tenant identifier | Always handled. |
| Gateway-ID (LADN-DNN, UPF-ID) | Network identifier | Handled when operator-configured. |
| UE-ID | Subscriber pseudonymous ID | **Only** when the operator's LIMF configures the LI constraint (`src/horizon_ric/policy/li_constraint.py:71-73`, `protected_ue_ids`). This is opaque-tag handling — the rApp never receives warrant content, never handles IMSI directly. |
| IMSI / SUPI / MSISDN / IMEI / IMEISV | PII | **NEVER** processed by Horizon-RIC. There is no code path that ingests these identifiers. |

### 1.3 Purposes (Art. 5(1)(b) purpose limitation)

- Optimisation of RAN slice and gateway placement to minimise SLA breach probability
- Predictive risk reporting to the operator's SOC (the rApp dashboard at `src/horizon_ric/rapp/dashboard_api.py`)
- Auditable record of every automated policy decision (the evidence chain at `src/horizon_ric/evidence/store.py`)

No secondary or unrelated purpose. The audit chain is used only for operator-side audit, regulator-on-warrant access, and post-incident replay.

### 1.4 Categories of data subjects

- **Subscribers** — only as cell-density and slice-load aggregates. No subscriber is identifiable from a `z_resource` vector or from the per-cell KPIs.
- **Operator personnel** — names, roles, and JWT subjects of SOC staff who interact with the rApp. Held in the operator's IdP, not in Horizon-RIC; Horizon-RIC stores only the `sub` claim from JWTs in audit logs (`src/horizon_ric/security/middleware.py:110, 153-161`).

### 1.5 Recipients (Art. 30(1)(d))

- The operator's Near-RT RIC (A1 PUT)
- The operator's SOC dashboard (R1 readers)
- The operator's Prometheus / Alertmanager (see `deploy/prometheus/`)
- On warrant, a regulator with credentials in the operator's IdP can read tenant-scoped audit records via `GET /api/v1/audit/verify` (`src/horizon_ric/rapp/api_v1.py:392-398`).

No data is shared with the vendor (Horizon-RIC) in production. Telemetry remains within the operator's network boundary.

### 1.6 Retention (Art. 5(1)(e))

- DecisionRecord audit chain: retention is operator-configured. Recommended floor: the longer of (a) the operator's regulatory log retention (typically 6-24 months) and (b) the SLA dispute window (typically 90 days). See risk #2 in §3.
- Model weights and their signed provenance records (`src/horizon_ric/provenance/signing.py`): retained for the lifetime of the deployment plus the longest in-flight A1 policy lifetime (so an audit can re-instantiate the model that produced any historical decision).
- Telemetry caches (Redis / in-memory): minutes to hours, no persistence by default.

## 2. Necessity and proportionality (Art. 35(7)(b))

### 2.1 Necessity

The processing is necessary because:

- The operator's regulatory obligation under TS 28.554 §6 (KPI reporting) requires the operator to retain RAN management-plane KPIs.
- The operator's contractual SLA obligation requires post-hoc breach evidence — satisfied by the audit chain.
- The operator's TS 33.117 §4.2.5.2 obligation (security log integrity) is satisfied by the SHA-256 chain.

Horizon-RIC's processing is therefore additive to obligations the operator already has.

### 2.2 Proportionality

- Horizon-RIC does NOT process PII; the highest-sensitivity identifier it can be configured to see is an opaque per-UE tag from the operator's LIMF (`policy/li_constraint.py:71-73`).
- The state representation uses dimensionality reduction before any decision; raw KPIs are not stored alongside the decision record (only the `state_hash` and an optional `state_blob_uri` are persisted, see `src/horizon_ric/evidence/schema.py`, `DecisionRecord`). The blob URI points to the operator's object store, not to the vendor.
- The audit chain is per-tenant: an auditor for tenant A cannot read tenant B (`evidence/store.py:212-227`). This is data-minimisation by tenant boundary.

## 3. Identified risks (Art. 35(7)(c))

### Risk 1 — Cell-ID quasi-identifier exposure

**Risk:** Per-cell KPIs are inherent to RAN scheduling but a cell-ID can, in principle, be combined with other data sources (a published cell-tower map plus a coarse user location) to reidentify subscribers in a sparse cell.

**Likelihood:** Low to moderate, depending on cell density.
**Severity:** Moderate — does not on its own identify a subscriber, but reduces the anonymity set.

### Risk 2 — Audit-log over-retention

**Risk:** The DecisionRecord chain accumulates indefinitely if the operator does not configure a retention policy. Holding records past the legitimate purpose violates Art. 5(1)(e) (storage limitation).

**Likelihood:** High in operator deployments without explicit retention configuration.
**Severity:** Moderate — supervisory-authority finding risk.

### Risk 3 — Evidence-store unauthorised access

**Risk:** A misconfigured RBAC policy or a leaked JWT could let a tenant-A user read tenant-B's audit chain.

**Likelihood:** Low (multiple defences in depth).
**Severity:** High when it occurs (cross-tenant disclosure).

### 3.5 Recital 26 quasi-identifier enumeration

GDPR Recital 26 requires that "to determine whether a natural person is identifiable, account should be taken of all the means reasonably likely to be used … either by the controller or by another person to identify the natural person directly or indirectly." This sub-section enumerates the combinations of fields Horizon-RIC handles, classifies the identifiability risk of each, and names the codebase mitigation in force. This addresses Devils-Advocate finding #22.

The combinations are listed from least to most identifiable. The naming convention is `(field_a + field_b + ... )`.

#### 3.5.1 `(cell_id)` alone

**Identifiability:** Not identifying. A cell-ID names a base-station coverage area, not a person. A cell-ID at any urban density covers ≥10⁴ subscribers; even in the sparsest rural cell it covers tens to hundreds.

**Mitigation in our codebase:**
- Cell-IDs are inherent to RAN scheduling and are present in every input KPI under TS 28.552 §6 — there is no version of the rApp's job that can be done without per-cell granularity.
- We do not enrich cell-IDs with cell location at decision time. The cell-ID is a 28-bit / 36-bit identifier in our pipeline, not a (lat, lon) pair. Cell-tower location databases exist publicly, but the operator's controller decision (not ours) governs whether they are joined to the audit chain at retrieval time.

#### 3.5.2 `(cell_id + slice_id)`

**Identifiability:** Possibly identifying for small slices. A slice (S-NSSAI) is operator-defined and may, in principle, contain very few subscribers — e.g., a private MVNO slice for a single enterprise customer. In a small private slice, `(cell, slice)` narrows the anonymity set materially.

**Mitigation in our codebase:**
- The operator-side recommendation (DPIA §4.1, "configure the rApp to drop cells with fewer than k subscribers from the policy decision input set") applies here: a `(cell, slice)` combination with fewer than k subscribers should not enter policy decision input.
- The cell-allowlist filter is exposed via O1 NETCONF (`src/horizon_ric/rapp/o1_adapter.py`); operators with private/enterprise slices are advised to configure the filter at slice-grain.
- Per-slice tenant scoping is enforced at retrieval (`src/horizon_ric/evidence/store.py:212-227, 285-298`): a tenant cannot read another tenant's slice-level audit records.

#### 3.5.3 `(cell_id + slice_id + time_window)`

**Identifiability:** Increasingly identifying as the time window narrows. A 5-minute window of `(cell_id, slice_id, time)` narrows the anonymity set further — for a sparse slice in a sparse cell during a quiet hour, fewer than 10 subscribers may have been served.

**Mitigation in our codebase:**
- Time windows in our audit chain are decision-grain (~1s decision cadence). We do not retain per-second per-(cell, slice) subscriber counts in the audit chain — we retain per-decision aggregates.
- The retention policy (DPIA §1.6, recommended ≥6 months under operator control) applies; old narrow-window aggregates are archived/deleted at the retention horizon under the operator's WORM policy.

#### 3.5.4 `(cell_id + slice_id + RNTI)` — UNIQUE per-second per UE → IDENTIFIABLE

**Identifiability:** **IDENTIFIABLE.** An RNTI (Radio Network Temporary Identifier) is unique to a single UE within a cell at any moment in time. The combination `(cell_id, slice_id, RNTI)` is therefore unique to a single UE, and over even a short observation window can be linked to a subscriber identity by a party with adjacent data (e.g., the operator's HSS or MME).

**Mitigation in our codebase:**
- **Horizon-RIC does NOT ingest, store, or process RNTI.** The input schemas (`src/horizon_ric/io/schemas.py`) consume per-cell aggregates from TS 28.552 KPIs, not per-UE per-RNTI records. There is no field in `src/horizon_ric/io/schemas.py` that holds RNTI.
- The only per-UE identifier the rApp can be configured to see is an opaque per-UE tag from the operator's LIMF (`src/horizon_ric/policy/li_constraint.py:71-73, protected_ue_ids`). This tag is a hashed/scoped identifier the operator generates; it is not an RNTI and is not joinable by the rApp to any subscriber identity. The operator's LI process governs the RNTI-to-tag mapping outside our boundary.
- `__iter__` returning DecisionRecords scopes by tenant (`evidence/store.py:212-227`); even were RNTI somehow persisted upstream, cross-tenant disclosure is fail-closed.

#### 3.5.5 `(cell_id + RNTI history over 5 min)` — IDENTIFIABLE

**Identifiability:** **IDENTIFIABLE.** A 5-minute trajectory of `(cell_id, RNTI)` pairs reconstructs a subscriber's mobility path across cells; combined with a public cell-tower map, this identifies a person's movement. Even without RNTI, a 5-minute trajectory of `(cell_id, opaque-UE-tag)` is identifying.

**Mitigation in our codebase:**
- We don't store RNTI at all (see §3.5.4). We don't store mobility trajectories. Only the `state_hash` (a fixed-size sha256) is persisted in the audit chain (`src/horizon_ric/evidence/schema.py`, `DecisionRecord.state_hash`). The optional `state_blob_uri` points to the operator's object store, not to the vendor's; the operator chooses whether and for how long the latent blobs are retained.
- The latent representation is dimensionality-reduced and is intentionally not invertible to per-UE trajectories; it is derived from cell-aggregate inputs, not per-UE inputs.
- LI-tagged UE handling is per the warrant lifecycle in `docs/compliance/li_applicability.md` §4.1 — the warrant defines retention; rApp-side retention is bounded by the warrant's term.

#### 3.5.6 Summary table

| Combination | Identifiability | Codebase mitigation |
|---|---|---|
| `(cell_id)` | Not identifying | KPIs at cell-grain are necessary; no enrichment with cell location at decision time. |
| `(cell_id + slice_id)` | Possibly, if slice is small | Operator opts into k-anonymity allowlist via O1; per-tenant scoping at evidence read. |
| `(cell_id + slice_id + time_window)` | Increasingly | Decision-grain audit only; operator-controlled retention. |
| `(cell_id + slice_id + RNTI)` | **IDENTIFIABLE** | **RNTI never ingested**; only opaque hashed LI-UE tags via warrant. |
| `(cell_id + RNTI-history over 5 min)` | **IDENTIFIABLE** | **No mobility trajectories stored**; only latent + state_hash in chain. |

The operator's controller-side responsibility under Recital 26 remains to assess "the means reasonably likely to be used … by another person to identify the natural person." Horizon-RIC's contribution is to make the rApp-side surface as small as possible: per-cell aggregates, no RNTI, no mobility, hashed UE tags only when warrant-driven, per-tenant scoping at every read.

### Risk 4 — Model card data lineage

**Risk:** The emitted model cards (`src/horizon_ric/observability/model_card.py`) declare training datasets but, if the operator brings their own training corpus and the vendor's corpus claim is stale, the lineage record may not match reality.

**Likelihood:** Low — provenance signing (`src/horizon_ric/provenance/signing.py`) binds weights to a training manifest, but does not guarantee the human-readable dataset description is accurate.
**Severity:** Moderate — undermines an auditor's ability to verify training data fairness / bias.

## 4. Mitigations (Art. 35(7)(d))

### Mitigation for Risk 1 — Cell-ID quasi-identifier

- Horizon-RIC never persists per-subscriber records; only per-cell aggregates appear in the evidence chain (`src/horizon_ric/evidence/schema.py`).
- The `z_resource` state is hashed before persistence (`evidence/schema.py:107`, `state_hash`); the raw vector is optional and operator-controlled (`state_blob_uri`).
- Operator-side recommendation: configure the rApp to drop cells with fewer than k subscribers from the policy decision input set (k-anonymity at the cell level). Horizon-RIC accepts a cell-allowlist filter via O1 (`src/horizon_ric/rapp/o1_adapter.py`); this is NOT a default and the operator must opt in.

### Mitigation for Risk 2 — Audit-log retention

- Audit log uses a hash-chain (`evidence/store.py:75-79, 106-142`) so deletion of historical records is detectable: an auditor running `verify()` will see the chain break at the deletion point. This means **structured retention** is required: deletions must happen at a chain checkpoint and must be logged.
- Operator-side recommendation: deploy a retention job that, at the operator's chosen retention horizon, **archives** the chain (snapshots + signs the head hash to the operator's WORM store) rather than deleting in place. **NOT YET — planned in Phase 3**: `deploy/runbooks/audit_retention.md` with a sample archive job. Today the operator must implement this themselves.

### Mitigation for Risk 3 — Evidence-store unauthorised access

- Per-tenant chains: `evidence/store.py:106-142` (chains computed independently per tenant).
- TenantScope enforcement on `__iter__`: `evidence/store.py:212-227` (Jsonl) and `evidence/store.py:285-298` (SQLite).
- JWT `tenant` claim required and enforced equal to `X-Tenant` header at the middleware: `src/horizon_ric/security/middleware.py:117-127`.
- Casbin RBAC: `src/horizon_ric/security/rbac.py`, with policy at `src/horizon_ric/security/rbac_policy.csv`.
- mTLS on every R1/A1/O1 channel: `src/horizon_ric/rapp/auth.py`.
- Audit-grade structured event names emitted on every auth grant/deny: `src/horizon_ric/security/middleware.py:38, 90-161`.

### Mitigation for Risk 4 — Model card data lineage

- Each emitted model card (`src/horizon_ric/observability/model_card.py`) records `sha256`, training corpus, and training metrics; model lineage is signed via `src/horizon_ric/provenance/signing.py` and surfaced through `src/horizon_ric/evidence/ai_phy_lineage.py`.
- Provenance at decision time: `ModelVersions` (`evidence/schema.py`) is persisted on every DecisionRecord, so an auditor can reproduce which model produced any decision.
- Operator-side recommendation: when the operator brings their own training corpus, they must re-issue the model card with corrected lineage. There is no automatic propagation today. **NOT YET — planned in Phase 2**: a CI hook that fails the build if a model card's training-corpus block is empty or the `dataset_sha256` is missing.

## 5. DPO consultation record (Art. 35(2)) — operator-side template

```
Operator: ____________________________________________
DPO name: ____________________________________________
Date of DPO consultation: ___________________________
DPO advice (free text):
________________________________________________________________
________________________________________________________________
________________________________________________________________
DPO concurrence with this DPIA's risk classification (Y/N): ___
Operator residual-risk acceptance signature: ________________
```

If the operator's DPO disagrees with any risk classification in §3, the operator should record their alternative classification here and adjust §4 mitigations accordingly. The supervisory authority of the operator's lead Member State must be consulted under Art. 36 if a high residual risk persists.

## 6. Article 30 records of processing — operator-side template

| Art. 30(1) field | Value |
|---|---|
| (a) Name and contact details of the controller | _operator-fills_ |
| (a) Joint controller, if any | (none — vendor is not a controller) |
| (a) Representative of the controller | _operator-fills_ |
| (a) Data Protection Officer | _operator-fills_ |
| (b) Purposes of the processing | RAN policy optimisation; SLA breach prediction; auditable record of automated decisions (see this DPIA §1.3) |
| (c) Categories of data subjects and personal data | Subscribers (cell/slice aggregates only); operator personnel (JWT `sub` claim) — see §1.4 |
| (d) Recipients to whom personal data have been or will be disclosed | Operator's Near-RT RIC, SOC dashboard, Prometheus/Alertmanager, regulator-on-warrant — see §1.5 |
| (e) Transfers to third countries | None by default (vendor receives no operator data in production) |
| (f) Time limits for erasure | Operator-configured; recommended floor in §1.6 |
| (g) General description of the technical and organisational security measures referred to in Art. 32(1) | mTLS (auth.py), JWT RS256 with rotation (jwt.py), Casbin RBAC, per-tenant evidence chain with SHA-256 tamper detection (evidence/store.py), atomic state checkpointer (runtime/state_recovery.py), structured audit event names (security/middleware.py) |

---

**End of DPIA.** This document is signed and dated by the operator's controller and DPO once §5 and §6 are completed. Spec citations are to Regulation (EU) 2016/679 of 27 April 2016 (consolidated text).
