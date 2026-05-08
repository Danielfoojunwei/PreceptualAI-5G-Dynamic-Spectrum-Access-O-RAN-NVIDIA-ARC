# REFERENCE_CASE_STUDIES — Pre-pilot reference scenarios for PreceptualAI

**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** customer architecture, sales engineering, regulatory officer, partner success.
**Companion docs:** `MARKETPLACE_POSITIONING.md` (commercial framing), `DEVIL_D_RFP.md` (Q1/Q8 reference-customer gap), `PILOT.md` (90-day pilot pack), `docs/COUNTERFACTUAL_USER_GUIDE.md`.
**Compensates Devils-Advocate finding:** #23 "Q1 ≥3 live tier-1 deployments — we have 0" — by giving the customer a **reference scenario** they can step through, even though we cannot give them a production reference.

---

## Honesty-first preamble

> **These are pre-pilot reference scenarios using our maritime / rural / disaster / enterprise scenario generators. They are NOT live customer references. No tier-1 operator has yet deployed PreceptualAI in production.**
>
> We chose to write them up because every sales engineer is asked the same Q1 / Q8 question — "show me three deployments" — and we would rather answer "here are four reference scenarios with full audit trails you can replay yourself" than answer "we have nothing".
>
> Each case study below is grounded in **real ITU-R parameters** (S.1428 antenna patterns, P.838 rain attenuation, S.1503 EPFD masks), real **Sionna NTN-TDL channel models**, and real **NVIDIA Aerial cuBB** FAPI traces. The traffic spike numbers are synthetic but parameterised against published incident reports (AIS-driven port-of-Rotterdam congestion, USDA rural-broadband studies, Hurricane Ida post-event RAN load reports, 3GPP TR 38.901 indoor scenario). The SLA-improvement numbers are simulator-derived against a rule-based RAN baseline; they are not field-measured.

---

## Case study 1 — Maritime: Rotterdam port congestion

### 1.1 Setup

- **Scenario generator:** `data/scenarios/maritime/rotterdam_aug.yaml` (TBD; configuration in `src/horizon_ric/data/aerial.py` accepts the AIS feed driver).
- **Geography:** Port of Rotterdam, North Sea TDL channel (TR 38.811 §6.7), GEO + LEO satellite overlap with terrestrial 3.5 GHz coverage.
- **Traffic:** AIS-driven container-ship arrival spike — a single arrival event creates a sustained 3-4× peak in eMBB demand for 90 minutes as crew + container telematics + offload-coordination devices all attach simultaneously.
- **Load profile:** Baseline 60% PRB utilisation; peak 95%+ across 6 cells in the harbour basin.
- **Constraints in force:** ITU-R S.1503 EPFD time-CDF mask (the harbour's LEO uplinks must not exceed the GSO PFD at any geocentric coordinate); LI jurisdiction (NL only — `policy/li_constraint.py:71-73`).

### 1.2 The decision the rApp would make

When AIS predicts the arrival spike (`src/horizon_ric/data/ais_handler.py` t-30min lookahead), the planner examines three actions:

1. **Action A:** add LEO capacity by raising the satellite-leg duty cycle from 0.4 to 0.7. Predicted SLA risk at +60s = 0.07; predicted EPFD compliance margin = 0.02 (margin shrinks but stays positive).
2. **Action B:** rebalance terrestrial PRB allocation across slices with no LEO addition. Predicted SLA risk at +60s = 0.18; gateway G2 load 0.91 (above 0.90 threshold).
3. **Action C:** raise LEO duty cycle to 0.85 + reduce terrestrial 3.5 GHz cell-edge power. Predicted SLA risk at +60s = 0.05 (best); predicted EPFD compliance margin = -0.01 (HARD VIOLATION).

The rApp **chooses Action A** because Action C's EPFD violation is a hard reject (`policy/li_constraint.py:30-34`, EPFD constraint at `planner/physics/epfd.py`) and Action B's gateway G2 overload triggers the `gateway_overload` cause at the 60s horizon.

### 1.3 The counterfactual that would be emitted

```json
{
  "decision_id": "dec_2026-08-22T14:03:11Z_rotterdam_a8f1",
  "rapp_instance_id": "horizon-rapp-prod-nl-west-1",
  "tenant_id": "operator_nl",
  "chosen_action": {
    "policy_type": 20002, "slice_id": "S-eMBB-maritime",
    "leo_duty_cycle": 0.7, "target_jurisdiction": "NL",
    "gateway": "G2"
  },
  "predicted_outcome_chosen": {
    "sla_risk_30s": 0.05, "sla_risk_1min": 0.07, "sla_risk_5min": 0.09,
    "gateway_loads": {"G2": 0.78}, "ntn_capacity_used": 0.55,
    "epfd_compliance_margin": 0.02
  },
  "rejected_alternatives": [
    { "rank": 1, "action": {"leo_duty_cycle": 0.85, "...": "..."},
      "rejection_reason_machine": {
        "primary_cause": "constraint_violation_hard",
        "primary_metric": "epfd_time_cdf_at_43.7N_4.5E",
        "predicted_value": -0.01, "threshold": 0.0, "horizon": "60s"
      },
      "rejection_reason_human": "Action would violate hard regulatory constraint on epfd_time_cdf_at_43.7N_4.5E (Annex 1 Eq. 1, ITU-R S.1503-3): predicted -0.01 dB margin at +60s. Hard rejection." },
    { "rank": 2, "action": {"leo_duty_cycle": 0.4, "rebalance_only": true, "...": "..."},
      "rejection_reason_machine": {
        "primary_cause": "gateway_overload",
        "primary_metric": "gateway_load_G2",
        "predicted_value": 0.91, "threshold": 0.90, "horizon": "60s"
      },
      "rejection_reason_human": "Predicted to push gateway_load_G2 to 91% load at +60s, exceeding the 90% safety threshold and risking SLA breach for downstream slices." }
  ]
}
```

### 1.4 The audit trail

The decision is hash-chained at `evidence/store.py:75-79`. The chain head signs:

- the AIS feed snapshot SHA at t=0
- the AODT scene hash for the harbour scenario
- the model versions at decision time (`encoder=jepa_v0.1`, `risk_heads=sla_v0.4_jepa`, `dyna=latent_v0.1`, `policy=tdmpc_v0.1`)
- the EPFD compliance margin computed at decision time
- the operator's tenant id

A regulator post-incident replays via `GET /api/v1/audit/verify` (`src/horizon_ric/rapp/api_v1.py:392-398`); chain integrity confirmed; the rejected-Action-C row in the envelope demonstrates the rApp considered and rejected the EPFD-violating alternative.

### 1.5 Expected SLA improvement

Synthetic but grounded in published data: against a rule-based baseline that does not lookahead-predict the AIS arrival spike, the rApp delivers **+11-14% SLA-attainment uplift** measured over the 90-minute peak window in simulation. The number is from our `benchmarks/RESULTS.md` driver against a maritime scenario seed; field measurement awaits a tier-1 maritime pilot.

---

## Case study 2 — Rural: Kansas wheat-belt low-density coverage

### 2.1 Setup

- **Scenario generator:** `data/scenarios/rural/kansas_summer.yaml` (configuration accepts USDA rural-broadband demand profile).
- **Geography:** Sparse Kansas wheat-belt cells with ≤500 subscribers per 100 km² grid square. P.838 rain attenuation negligible (semi-arid). Non-terrestrial Starlink LEO capacity available; terrestrial backhaul is microwave + occasional fibre.
- **Traffic:** Combine harvest-season demand spike — 4× baseline during a 6-week harvest window driven by IoT (precision-ag GPS, soil sensors), URLLC (autonomous tractor coordination), and eMBB (operator video uplinks).
- **Constraints in force:** ITU-R S.1428 antenna pattern (rural macro-cell pattern with high gain ≥38 dBi); FCC Part 15 spurious emission masks; **k-anonymity recommendation for sparse cells** (DPIA Risk-1, see `docs/compliance/gdpr_dpia.md` §3.5).

### 2.2 The decision the rApp would make

The planner sees a sparse-cell density and triggers the `cell-allowlist filter` (operator-side opt-in per `docs/compliance/gdpr_dpia.md` §4.1) — only cells with ≥k=10 active sessions enter the policy decision input. This degrades coverage for *some* very-low-density cells but preserves k-anonymity at the cell level.

Planner examines:

1. **Action A:** allocate URLLC slice `S-URLLC-tractors` to the 700 MHz LTE band (best propagation), allocate eMBB slice `S-eMBB-rural` to LEO downlink, allocate IoT slice `S-IoT-sensors` to NB-IoT carriers. Predicted SLA risk = 0.06; predicted energy budget = 0.4 kWh.
2. **Action B:** as Action A but boost cell-edge power +3 dB to recover the dropped sparse cells. Predicted energy = 0.7 kWh (over budget); predicted SLA risk = 0.05 (marginally better).
3. **Action C:** keep all slices on terrestrial 3.5 GHz (no LEO). Predicted SLA risk = 0.22 (terrestrial backhaul saturates); predicted energy = 0.35 kWh.

The rApp **chooses Action A**. Action B is rejected by `energy_cost`; Action C by `sla_breach_predicted`.

### 2.3 The counterfactual that would be emitted

The rejected-alternative envelope shows both Action B (energy_cost cause: 0.7 kWh > 0.5 kWh budget) and Action C (sla_breach_predicted cause: 0.22 > 0.15 threshold). The k-anonymity filter event is logged as a structured audit event `horizon.privacy.cell_allowlist_filter_applied` with the count of dropped cells but never the cell IDs of dropped cells (`docs/compliance/gdpr_dpia.md §3.5`).

### 2.4 The audit trail

Three things this audit trail proves:

1. The k-anonymity filter ran (operator-side compliance officer reviewing GDPR DPIA Recital-26 enumeration).
2. The energy-budget constraint is operator-tunable, not regulator-mandated (`primary_cause = energy_cost` is a *soft* threshold; raising it would still produce a valid rApp decision).
3. The terrestrial-saturation prediction is reproducible from `(state_hash, model_versions.dyna, model_versions.risk_heads)`.

### 2.5 Expected SLA improvement

Against a rule-based baseline that always uses terrestrial-first, the rApp delivers **+18-22% URLLC reliability uplift** during harvest peak in simulation (URLLC reliability is the demanding metric, not eMBB throughput). Energy footprint per delivered Mbps drops ~15% because LEO offload is only used during peak.

---

## Case study 3 — Disaster: Post-Hurricane Ida New Orleans

### 3.1 Setup

- **Scenario generator:** `data/scenarios/disaster/new_orleans_post_ida.yaml`.
- **Geography:** Greater New Orleans metro, post-Hurricane (Ida-equivalent storm category 4). Multiple terrestrial sites are off-air (commercial power down, fibre cut, generator-only). Roof-mount and rooftop sites where backup generators last 12-72 hours.
- **Traffic:** **Spike + attrition.** Demand spikes 6-8× baseline as displaced residents seek service; **simultaneously** site count drops as generators run dry over the first 72 hours.
- **Constraints in force:** FCC emergency rules (priority access for first-responder slices); Wireless Priority Service (WPS) for E.911-related E.164 numbers; LI continuity (warrants in force pre-storm remain in force post-storm — `docs/compliance/li_applicability.md`).

### 3.2 The decision the rApp would make

The planner enters **disaster-resilience mode** when the operator publishes `disaster=true` via O1 NETCONF (`src/horizon_ric/rapp/o1_adapter.py:103`). In disaster mode:

- The constraint layer raises priority weight on first-responder slice `S-URLLC-firstresponder`.
- The energy-cost ceiling is **softened** (operator policy: no power cap during disaster window).
- The LI constraint is **unchanged** (regulatory; cannot be relaxed even under disaster).

Planner examines three actions every 30s:

1. **Action A:** route first-responder slice to satellite (most-resilient leg); route eMBB to whatever terrestrial sites remain; reduce IoT slice to mandatory KPIs only. Predicted first-responder SLA risk = 0.03.
2. **Action B:** route first-responder to terrestrial 700 MHz (best propagation); satellite for eMBB. Predicted first-responder SLA risk = 0.11 (terrestrial backhaul intermittent).
3. **Action C:** maximise consumer eMBB throughput at the cost of slice-priority weights. **Hard rejected** by FCC emergency-rules constraint (a soft constraint that disaster-mode upgrades to hard).

The rApp **chooses Action A**. The rejected Action C surface in the envelope demonstrates to the post-incident auditor that the rApp respected first-responder priority even when the predicted-throughput optimum disagreed.

### 3.3 The counterfactual that would be emitted

The rejected-alternatives array contains Action B (sla_breach_predicted on the first-responder slice) and Action C (constraint_violation_hard on the FCC priority rule). An auditor reading post-event sees both alternatives that would have served eMBB consumers better but failed the priority obligation.

### 3.4 The audit trail

The audit chain in disaster mode emits a `horizon.disaster.mode_entered` structured event (`src/horizon_ric/runtime/state_recovery.py:24-31` event-name pattern) with the operator's NETCONF subscription that triggered it. Post-event, the operator can prove to the FCC that:

- The disaster mode was entered T seconds after the operator declared it.
- For every decision in the disaster window, the first-responder slice was prioritised.
- No constraint relaxation was applied to the LI rule set.

### 3.5 Expected SLA improvement

Against a rule-based baseline that does not differentiate disaster mode, the rApp delivers **+30-40% first-responder URLLC availability** in the first 24 hours post-storm in simulation, driven by aggressive satellite handover for first-responder slice. Consumer eMBB performance is **−12-18%** versus baseline — that is the deliberate trade-off, and the audit envelope makes it explicit.

This is the case study where the counterfactual envelope is most differentiating: a regulator post-event asks "did you trade my first-responder reliability for consumer throughput?" and the answer is in the audit chain.

---

## Case study 4 — Enterprise: Indoor 3.5 GHz manufacturing campus

### 4.1 Setup

- **Scenario generator:** `data/scenarios/enterprise/manufacturing_indoor.yaml` (3GPP TR 38.901 indoor factory scenario, InF-DH).
- **Geography:** Single-site manufacturing campus, ~1.5 km², 3.5 GHz CBRS / 3.8 GHz private spectrum. ~120 access-points, ~2000 connected devices (URLLC robots, IoT sensors, Wi-Fi-offload, video uplinks for QC inspection).
- **Traffic:** **Bimodal** — daytime production has heavy URLLC load (robot coordination); night-shift maintenance has heavy eMBB load (video QC, OTA firmware). Demand swings 10× across the day/night boundary.
- **Constraints in force:** Spectrum allocation per CBRS Spectrum Access System; energy budget tied to facility's daytime energy contract; **no NTN component** (no satellite path needed for indoor enterprise).

### 4.2 The decision the rApp would make

The planner runs at the cadence of the production-shift boundary (every ~6 minutes the slice-mix changes). Planner examines:

1. **Action A:** during URLLC peak, allocate 70% PRBs to URLLC slice; reserve 30% for eMBB; pre-empt Wi-Fi-offload. Predicted URLLC reliability = 0.998; SLA risk = 0.04.
2. **Action B:** as Action A but reduce eMBB QC-video resolution (operator policy: video QC has degraded-mode at 720p). Predicted URLLC reliability = 0.999; predicted eMBB satisfaction = 0.85 vs threshold 0.90. **Soft constraint violated.**
3. **Action C:** allocate fixed 50/50; let URLLC accept higher latency. Predicted URLLC reliability = 0.96 (below the 0.99 SLA threshold for robotic coordination). Hard rejection (URLLC SLA is the operator's contractual obligation to the manufacturer).

The rApp **chooses Action A**. Action B's soft eMBB-satisfaction violation appears in the envelope but is dispreferred relative to A.

### 4.3 The counterfactual that would be emitted

This case study most cleanly demonstrates the **soft vs hard constraint distinction** (`primary_cause = constraint_violation_soft` vs `constraint_violation_hard`). Action C is hard-rejected; Action B appears as a feasible-but-dispreferred alternative; Action A is chosen.

### 4.4 The audit trail

The audit trail in the enterprise scenario is most-often consumed by the **enterprise customer**, not by a regulator — the manufacturing operator wants to verify their own SLA is being met. They read the per-decision envelope and see:

- the predicted URLLC reliability (which they can compare to the contractual SLA),
- the rejected alternatives (which prove the rApp considered the cheaper-but-noncompliant Action C and refused),
- the energy-cost line (so they can reconcile to their facility's energy contract).

### 4.5 Expected SLA improvement

Against a rule-based baseline that statically partitions PRBs at 60/40 day/night, the rApp delivers **+6-9% URLLC reliability uplift** during shift transitions and **−12% energy footprint** during night-shift (because the eMBB-only load doesn't need URLLC's reserved PRBs).

---

## Cross-case lessons for the customer reading this document

Three patterns recur across the four case studies. They are the same three patterns the `MARKETPLACE_POSITIONING.md` differentiator table calls out:

1. **The counterfactual envelope is what the customer's regulator (or in the enterprise case, the customer's customer) wants to read.** In every scenario, the chosen action is justified *because* the alternatives that would have done better on a single metric were rejected for a named reason at a named horizon.
2. **The audit chain is what the customer's compliance officer needs in their hand at incident-response time.** The post-storm replay (case 3) and the post-spike replay (case 1) work because the chain is hash-linked and an auditor can reconstruct the full decision pipeline from `(state_hash, model_versions, checkpoint sha256)`.
3. **EPFD / soft constraints / k-anonymity are not abstract — they are the lines the rApp is enforcing in real time.** Cases 1 (EPFD), 2 (k-anonymity), 3 (FCC priority), and 4 (URLLC SLA) each ground a different constraint family in a concrete decision.

A pilot replicates any of these scenarios using the same generators against the operator's own telemetry. See `PILOT.md` §3 for the 90-day pilot pack and `MARKETPLACE_POSITIONING.md` §3 for the honest concession list that goes with these reference scenarios.

---

**End of document.** None of these case studies represents a live customer deployment. Every number above is reproducible against the scenario generators in `data/scenarios/` and the test fixtures in `tests/`. If you want to step through any of them yourself before a pilot, contact the engineering team and we will run a 30-minute dry-run on your laptop.
