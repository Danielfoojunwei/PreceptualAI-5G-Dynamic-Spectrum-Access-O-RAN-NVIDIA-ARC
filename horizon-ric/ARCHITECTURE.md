# PreceptualAI Architecture

*Date: 2026-05-08 (canonical to v3 trust-layer wave). Full per-section deep dive in [`README.md`](README.md) Part II.*

This document is the **engineer's map** of PreceptualAI. It tells you where
things live, what depends on what, and exactly where to plug in a new
data source / modality / sink / model.

**v3 added modules.** Trust layer + AI-RAN integration shipped these new modules — read [`README.md`](README.md) §9 for the full ASCII topology including:
- `policy/{neural_rx_decision, dpod_activation, learned_constellation_decision}.py` (M1, AI-PHY decisions with counterfactual envelope)
- `planner/physics/{otfs, fdss, grant_free_sic}.py` (M4, 6G modulation primitives)
- `evidence/ai_phy_lineage.py` (M5, TS 28.105 §7.4 with AI-PHY block lineage)
- `runtime/{atomic_promotion, shadow_executor, artefact_vault, loop_state}.py` (M7+M8+M9, LCM atomics)
- `data/{aerial.py::DLDBLiveConsumer, lineage.py}` (M2 + M9, NVIDIA Aerial Data Lake live consumer + GDPR data manifest)
- `integrations/{nvidia_arc_ota, viavi_d4ai, viavi_digital_twin}.py` (M2 + M3, vendor sandbox bindings)

## Topology

```
                         ┌──────────────────┐
                         │      Operator     │
                         │  (UI, evidence,   │
                         │   A1 policies,    │
                         │   audits, alerts) │
                         └────────┬──────────┘
                                  │ HTTP /healthz /readyz /metrics
                                  │ /api  (FastAPI)
                                  ▼
   ┌────────────────────────  rApp ────────────────────────────────┐
   │                                                              │
   │   ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
   │   │  Sources    │───►│  Encoder     │───►│  World Model   │  │
   │   │  (io/       │    │  (encoder/)  │    │  (core/)       │  │
   │   │   connectors)│   │              │    │  Liquid CfC +  │  │
   │   └─────────────┘    │  JEPA + HGT  │    │  physics       │  │
   │          ▲           │  + Perceiver │    │  residual      │  │
   │          │           └──────┬───────┘    └────────┬───────┘  │
   │          │                  │                     │          │
   │   ┌──────┴──────┐    ┌──────▼──────┐    ┌────────▼───────┐   │
   │   │  Plugins    │    │ Constraint  │    │  Planner       │   │
   │   │  (entry-pts)│    │ Layer       │◄───┤ TD-MPC2        │   │
   │   └─────────────┘    │ (policy/    │    │ (policy/       │   │
   │                      │  EPFD,      │    │  td_mpc.py)    │   │
   │                      │  ITU mask,  │    └────────┬───────┘   │
   │                      │  edge GPU)  │             │           │
   │                      └─────────────┘             │           │
   │                              ▲                   ▼           │
   │                              │         ┌──────────────────┐  │
   │                              │         │  Action Sink     │  │
   │                              │         │  A1Adapter ───►  │  │
   │                              │         │  Near-RT RIC     │  │
   │                              │         └──────────────────┘  │
   │                              │                   │           │
   │                       ┌──────┴────────┐    ┌─────▼───────┐   │
   │                       │ Evidence Store│◄───┤ DecisionRec │   │
   │                       │ (JSONL/SQLite,│    │ persistence │   │
   │                       │  hash chain)  │    └─────────────┘   │
   │                       └───────────────┘                      │
   │                                                              │
   │   Cross-cutting:                                             │
   │     federated/ (weights-only FedAvg via flwr)                │
   │     trading/   (sealed-bid auction, Paillier HE)             │
   │     planner/physics/ (ITU-R, 3GPP TR 38.811, SGP4, geodesy)  │
   │     contracts/ (DomainAdapter, ConstraintLayer, MetricSuite) │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
```

## Package map

| Package | Purpose | Stable contract |
|---|---|---|
| `horizon_ric.io` | Schemas + Connector ABCs + registry | `TelemetryEvent`, `FeatureFrame`, `PolicyAction`, `AuditRecord`, `Source`, `Sink` |
| `horizon_ric.io.connectors` | Reference connectors (file, http, kafka, grpc) | Auto-registered on import |
| `horizon_ric.contracts` | Domain-adapter / constraint-layer / metric ABCs | `DomainAdapter`, `ConstraintLayer`, `MetricSuite` |
| `horizon_ric.planner.physics` | Pure-function physics (ITU-R, 3GPP, SGP4, WGS-84) | Stateless functions; all numbers cite a published spec section |
| `horizon_ric.core` | Liquid CfC cell, physics-residual, timing budgets | `CfCCell`, `PhysicsResidualHead`, `l1_admit/l2_/a1_admit` |
| `horizon_ric.heads` | Risk heads (SLA shipped; beam/gateway/compute Phase 2) | `SLARiskHead` |
| `horizon_ric.policy` | Constraint layer + projection | `PreceptualAIConstraintLayer` |
| `horizon_ric.evidence` | Tamper-evident audit log | `JsonlEvidenceStore`, `SqliteEvidenceStore` |
| `horizon_ric.rapp` | R1, A1 adapters, lifecycle, health endpoints | `HorizonRAppLifecycle`, `R1Adapter`, `A1Adapter` |
| `horizon_ric.scenarios` | Synthetic generators for offline training | `MaritimeSyntheticGenerator` |
| `horizon_ric.agent` | Edge inference loop | `EdgeAgentConfig`, `main()` |
| `horizon_ric.encoder` | Resource-State JEPA encoder (Phase 1 — design specced) | TBD |
| `horizon_ric.federated` | Weights-only FL (Phase 2 — design specced) | TBD |
| `horizon_ric.trading` | Cross-operator HE auction (Phase 3 — design specced) | TBD |

## Stable I/O contract

Every cross-module data flow uses one of four Pydantic models:

| Schema | Producer | Consumer | Flow direction |
|---|---|---|---|
| `TelemetryEvent` | Sources, sensors | Encoder, audit | Inbound |
| `FeatureFrame` | Encoder | World model, planner | Internal |
| `PolicyAction` | Planner | A1 adapter, audit | Outbound |
| `AuditRecord` | Decision-record builder | Evidence store, regulators | Outbound |

All four carry `schema_version`. Major bumps are coordinated; new fields
are additive (`extra="allow"` on TelemetryEvent + FeatureFrame +
AuditRecord; strict on PolicyAction since it's regulatory-bound).

## How to plug in something new

### 1. New data source (Kafka topic, gRPC stream, REST endpoint, file)

The smallest possible change: pick the right reference connector, drop a
config dict in `deploy/config/<env>.yaml`, point the rApp at it. No code.

```yaml
sources:
  - kind: kafka
    config:
      name: starlink_kpm_stream
      bootstrap_servers: kafka.prod:9092
      topic: telemetry.starlink.kpm
      group_id: horizon-ric-prod
  - kind: http
    config:
      name: openweather_grib_poll
      url: https://api.openweathermap.org/data/3.0/onecall
      bearer_token: ${OPENWEATHER_KEY}
      poll_interval_seconds: 60
```

### 2. New modality (e.g. ISL link telemetry, RIS reflectarray state)

1. Add the modality identifier to `horizon_ric.io.schemas.Modality`
   (additive — never remove).
2. (Optional but recommended) write a payload submodel in
   `horizon_ric.io.payloads` so consumers can validate strictly.
3. The encoder dispatches on `event.modality`; add a tokeniser branch
   under `horizon_ric.encoder.tokenizers/<modality>.py`.

### 3. New connector (custom transport)

Inherit from `Source`, `Sink`, or `BiConnector`; declare a `Config`
inner class subclassing `ConnectorConfig`; register in either of two
ways:

```python
# In-process
from horizon_ric.io.registry import register_source
register_source("my_transport", MyTransportSource)
```

```toml
# External package — no code change to PreceptualAI
[project.entry-points."horizon_ric.connectors"]
my_transport = "my_pkg.transports:MyTransportSource"
```

Both routes converge at `registry.get_source("my_transport", config_dict)`.

### 4. New physics module (e.g. ITU-R P.840 cloud attenuation)

1. Add a pure function under `horizon_ric.planner.physics.<spec>.py`
   following the same docstring convention (cite spec section + paragraph).
2. Vendor any tabulated data into `_itu_tables.py`.
3. Export from `planner/physics/__init__.py`.
4. If the function feeds the constraint layer, add a hook to
   `policy.constraints.PreceptualAIConstraintConfig`.

### 5. New ML head (beam_risk, gateway_risk, compute_risk)

Inherit from `nn.Module`; reuse `core._two_hot.two_hot_loss` for
multi-OOM-scale targets; expose `predict()` + `loss()` + `calibration_summary()`
to match the SLA head contract; export from `heads/__init__.py`.

### 6. New A1 policy type

1. Add the type to `DEFAULT_POLICY_TYPES` in `rapp.a1_adapter`.
2. Write the type's JSON schema in `_policy_create_schema(policy_type)`
   — no permissive `additionalProperties: True`.
3. The A1 adapter's PolicyStatus polling (Phase 1.5) auto-picks it up.

## Layered timing contract (engineering rule)

Every component must satisfy ONE of three latency contracts:

| Layer | Budget | Contract gate | What can run here |
|---|---|---|---|
| L1 (PHY) | 50 µs | `core.timing_budgets.l1_admit()` | distilled CfC heads only; no full models |
| L2 (MAC) | 200 µs | `l2_inference_budget_us()` | small MLP (≤ 25 k params) |
| L3 / A1 | 90 ms | `a1_admit()` | full world model + planner + EPFD aggregation |

A model that fails its layer's admit gate must either be distilled, run
in a slower layer, or be precomputed into a `policy_lut` cache.

## Security posture

* All R1 / A1 traffic over `https://`. mTLS + OAuth2 bearer enforced via
  `httpx.AsyncClient` headers (Phase 1.5 — see `rapp.auth`).
* Federated learning is **weights-only** — no raw KPMs leave the device
  (project memory: `project_uhci_constraints.md`).
* Audit log is hash-chain tamper-evident (`evidence.store`).
* Connector configs supporting credentials accept `${ENV_VAR}`
  expansion; never log decoded values.

## Versioning

* Schemas: SemVer in `schema_version` (additive minor; breaking major).
* Connectors: independent per package — declare a `__version__` and
  surface it in connector metadata.
* The rApp itself: `horizon_ric.__version__` is the single source of
  truth, baked into A1 emissions and audit records as `model_versions.rapp`.

## Diagnostics

| Question | Where to look |
|---|---|
| What sources are configured? | `registry.list_connectors()` + lifecycle log |
| Is the rApp ready? | `GET /readyz` (200 if RUNNING) |
| What does the encoder think the channel state is? | `tr38811.channel_state(env, freq, elev)` |
| Why was an action rejected? | `evidence` store: rejection reason machine + human |
| Did audit log change? | `JsonlEvidenceStore.verify()` returns -1 if intact |
| What's the EPFD? | `epfd_down(...)` returns aggregate + per-emitter |
| Why is inference slow? | `core.timing_budgets.inference_budget_ms(thermal_state, ...)` |
