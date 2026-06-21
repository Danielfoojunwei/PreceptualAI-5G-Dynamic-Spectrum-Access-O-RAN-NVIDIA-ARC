# Horizon-RIC — a security & trust rApp for AI-RAN

**Horizon-RIC makes AI-RAN decisions safe and auditable *by construction*.** It
is an O-RAN Non-RT-RIC **rApp** that wraps the decisions a neural-PHY block or
RIC policy head produces — and guarantees that a *poisoned, drifted, or
adversarial* model still cannot emit an unsafe or illegal radio policy.

It does **not** train or run the neural model. It sits beside the SMO / Non-RT
RIC and acts as a security envelope around the AI's *output*. That is the
AI-RAN-native security idea: bind the AI decision to the RAN's own physical and
regulatory invariants, at the RAN's own timescales.

> Scope note: this repository is deliberately narrow — **security in AI-RAN**.
> It is torch-free and installs on a stock Python 3.10+ with no accelerator. The
> earlier broad "UHCI / NTN orchestration" codebase has been removed; the
> security core is the product.

---

## The four mechanisms

| # | Mechanism | What it defends | Where |
|---|---|---|---|
| 1 | **Decision Safety Shield** | A poisoned/adversarial model cannot emit an illegal policy. The AI proposes; a verified envelope of RAN-physics + spectrum + AI-PHY + lawful-intercept invariants disposes, projecting to a safe action or a **certified classical fallback** and emitting a `SafetyCertificate`. | `src/horizon_ric/shield/` |
| 2 | **At-decision-time evidence** | Tamper-evidence + replayability. SHA-256 hash-chained, RFC-3161-anchored `DecisionRecord`s with a counterfactual (rejected alternatives + reason + pinned RNG seed). | `src/horizon_ric/evidence/`, `policy/counterfactual.py` |
| 3 | **Model provenance** | The "byte-consistent but untrusted/backdoored model" gap. A signature over `(weights ‖ training-manifest)` (RSA-PSS, HSM-held key), verified on promotion. | `src/horizon_ric/provenance/` |
| 4 | **Robust federated aggregation** | Model-poisoning (Byzantine) clients. Krum / coordinate-median / trimmed-mean cap a malicious client's influence; Shamir secure aggregation hides individual updates. | `src/horizon_ric/federated/` |

Supporting (Zero-Trust platform table-stakes, per NIST SP 800-207): mTLS, JWT,
Casbin RBAC with tenant domains, HSM key custody, NIS2 24h incident reporter
(`src/horizon_ric/security/`, `rapp/`).

The full mapping of O-RAN WG11 / OWASP-ML / 3GPP TR 33.898 threats to these
controls — with an honest GAP register — is in
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

---

## The Shield in one snippet

```python
from horizon_ric.shield import default_terrestrial_shield
from horizon_ric.policy.li_constraint import LIConstraint

shield = default_terrestrial_shield(
    band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0,
    li_constraint=LIConstraint(rules=[], fail_closed=False,
                               deployment_audit_record="lab-001"),
)

# A *poisoned* neural-RX decision: out-of-band, over-power, illegal 1024-QAM,
# blown PAPR, and a TBLER regression versus the classical baseline.
poisoned = {
    "block": "neural_rx", "frequency_hz": 3.55e9, "bandwidth_hz": 20e6,
    "tx_power_dBm": 40.0, "antenna_gain_dBi": 6.0,
    "constellation_order": 1024, "papr_dB": 12.0,
    "predicted_tbler": 0.4, "baseline_tbler": 0.05, "demap_confidence": 0.1,
}

disp = shield.dispose(poisoned, decision_id="d1", rng_seed=7)
assert disp.certificate.safe and not disp.certificate.emit_blocked
# → safe action: in-band 3.49 GHz, EIRP 33 dBm, 256-QAM, PAPR 8.5 dB,
#   fell back to a certified classical receiver — and a SafetyCertificate
#   recording every correction for the audit chain.
```

---

## The benchmark (attack → defense)

```bash
python benchmarks/poisoning_shield_benchmark.py --decisions 10000 --poison-rate 0.30
```

Representative result:

```
SHIELD:    ~1500 illegal emits without the Shield → 0 with it   [PASS]
FEDERATED: FedAvg dist 202.2  vs  Krum 12.1 / median 7.8 / trimmed 8.9
```

The Shield reduces illegal air-interface emits from a 30%-poisoned decision
stream to **zero**; robust aggregation cuts a Byzantine federation's pull on the
global model by ~20×.

---

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # torch-free; installs in seconds
pytest tests/ -m "not integration and not slow" -q
python benchmarks/poisoning_shield_benchmark.py --decisions 2000
```

Optional extras: `.[oran]` (NETCONF/YANG + ASN.1 for the O1/A1 wire),
`.[otel]` (OpenTelemetry export), `.[persistence]` (Postgres evidence store).

Run the rApp daemon (registers over R1, emits A1 policies, serves
`/healthz` `/readyz` `/metrics`):

```bash
horizon-rapp --once          # boot + readiness check + shutdown (smoke)
```

---

## Architecture

```
 telemetry  ┌──────────────────────────────────────────────────────────────┐
 (Aerial    │                        Horizon-RIC rApp                        │
  cuBB /    │                                                                │
  E2 KPM)   │   neural-PHY / RIC head ──▶ Decision Safety Shield ──▶ A1 emit │
     │      │      (external; the AI         (shield/)              (rapp/)   │
     └──────┼──────▶ proposes)                   │                     │      │
            │                                    ▼                     ▼      │
            │   provenance/  ◀── verify    evidence chain        Zero-Trust   │
            │   (signed weights)           (SHA-256 + RFC-3161)   (security/) │
            │   federated/ (robust agg)    + counterfactual                   │
            └──────────────────────────────────────────────────────────────┘
```

- `shield/` — invariants (`SpectralMaskInvariant` TS 38.104, `MaxEirpInvariant`,
  `NeuralRxEnvelopeInvariant`, `ConstellationLegalityInvariant`,
  `LawfulInterceptInvariant` TS 33.127), the `Shield` orchestrator, and the
  `SafetyCertificate`.
- `evidence/` — hash-chained store, RFC-3161 anchor, model card, AI-PHY lineage.
- `provenance/` — sign / verify model artefacts.
- `federated/` — `robust.py` (Krum/median/trimmed) + `secure.py` (Shamir).
- `rapp/` — R1 / A1 / O1 adapters, auth, lifecycle, health.
- `security/` — JWT, Casbin RBAC, HSM, tenant isolation, NIS2 reporter.

---

## Honest status

- **What is real and tested:** the Shield, provenance, robust/secure
  aggregation, evidence chain, RBAC/JWT/tenant, R1/A1/O1 adapters — all covered
  by the fast test pack and the benchmark above.
- **Invariant calibration:** the spectral-mask / EIRP limits must be pinned to
  each deployment's licensed band; the defaults are illustrative.
- **Known gaps (see `docs/THREAT_MODEL.md`):** no DP accountant for
  membership-inference/inversion yet; no inference-API extraction rate-limiting
  yet; telemetry source authentication depends on the vendor (Aerial / E2); HSM
  production custody (CloudHSM / Luna) is documented, not bundled.

## AI-RAN Alliance positioning

This targets the **AI-for-RAN** track as a *security & trust* contribution: a
demonstrable, benchmarked control set for the O-RAN WG11 / OWASP-ML AI/ML threat
surface, framed as an open innovation + benchmark (the normative security spec
work belongs to O-RAN WG11 / 3GPP, not to this project).

License: Apache-2.0.
