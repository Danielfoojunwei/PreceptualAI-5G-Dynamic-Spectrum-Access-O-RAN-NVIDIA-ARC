# UHCI Architecture

This document explains the repository as a **UHCI-first system**. In this codebase, **Universal Heterogeneous Connectivity Intelligence (UHCI)** is the primary architectural concept: a connectivity-intelligence stack that spans **provider modeling, telecom-physics-aware propagation, heterogeneous environment construction, structural and temporal learning, policy inference, low-latency serving, O-RAN integration, and lifecycle governance**.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

The most important architectural point is simple. The repository should be read as a **heterogeneous connectivity intelligence system** whose center of gravity is provider-aware reasoning across terrestrial and non-terrestrial classes, physics regimes, graph-structured state, continuous-time decision dynamics, and deployment-facing telecom control loops. A line-by-line audit shows that these layers are not peripheral add-ons; they are the core of the implemented architecture.[1] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

| Architectural layer | Primary role in UHCI | Principal repository anchors |
|---|---|---|
| World and provider layer | Defines what kinds of connectivity assets exist and what priors they carry | `provider_registry.py`, `itu_propagation.py`, `fr3_propagation.py` |
| Environment and data layer | Builds the controllable world, state, actions, rewards, and empirical ingestion path | `unified_connectivity_env.py`, `data_pipeline.py`, `oran.py` |
| Representation layer | Encodes heterogeneous system state structurally and temporally | `hetero_gnn_encoder.py`, `ltc_cell.py`, `ltc_cell_cfc.py` |
| Decision layer | Produces actions over the richer UHCI state and provider space | `universal_spectrum_agent.py` |
| Runtime and serving layer | Turns trained intelligence into a low-latency callable service | `inference_engine.py`, `dapp_engine.py`, `server.py`, `aerial_adapter.py` |
| Control-plane and governance layer | Connects the intelligence to O-RAN loops and model-management flows | `e2_adapter.py`, `oran.py`, `rapp_trainer.py` |
| Evidence layer | Defines what has been benchmarked and how claims should be bounded | `benchmarks/benchmark.py`, benchmark artifacts, tests |

## Architectural Thesis

UHCI treats wireless decision-making as a **heterogeneous control problem** rather than a narrow channel-selection problem. The core architectural bet is that future connectivity intelligence must reason across **different provider classes, different physical regimes, different temporal horizons, and different operational insertion points**. That is why the repository contains a provider registry, multiple propagation modules, a unified environment, graph-oriented encoding, continuous-time temporal modules, real-time inference code, and Non-RT lifecycle management in the same codebase.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

> The architectural shift is from **single-environment spectrum selection** to **multi-layer heterogeneous connectivity intelligence**.

This shift matters because the intelligence problem is larger than immediate spectral occupancy. The system may need to compare or coordinate connectivity options whose behavior differs in **latency, Doppler exposure, handover interval, bandwidth, coverage, propagation impairment, observability, and control-loop timing**. A repository that models only one of those axes is not solving the full problem UHCI is aimed at.[1] [2] [3] [4]

## Layer 1: Provider Taxonomy and World Modeling

UHCI begins by defining the decision world properly. `provider_registry.py` is one of the most important files in the repository because it formalizes the fact that the system is reasoning over **provider classes**, not just over anonymous channels. The implementation organizes heterogeneous connectivity options such as **LEO, MEO, GEO, HAPS, FR1, FR3, ISAC, and WiFi 7**, together with structured priors such as altitude, coverage, bandwidth, latency range, Doppler characteristics, handover period, and coexistence assumptions.[1]

This provider taxonomy is not cosmetic. It determines how the rest of the system should interpret state, construct actions, and reason about performance tradeoffs. In architectural terms, the provider registry gives UHCI its **ontology**: it specifies what kinds of things the controller can reason about and how those things differ before learning begins.[1]

| Provider-modeling concern | Why it exists in UHCI | Principal anchor |
|---|---|---|
| Provider identity | Prevents all connectivity options from collapsing into one flat action pool | `provider_registry.py` |
| Latency and bandwidth priors | Encodes service-quality constraints at the provider level | `provider_registry.py` |
| Doppler and handover characteristics | Matters for NTN and moving-platform reasoning | `provider_registry.py` |
| Coverage and coexistence assumptions | Shapes feasible and desirable control decisions | `provider_registry.py` |

A key implication follows from this design. UHCI is not simply a smarter spectrum picker; it is a controller built to reason over **heterogeneous connectivity assets whose operating envelopes differ by construction**.[1]

## Layer 2: Telecom Physics and Propagation Realism

The provider ontology is paired with an explicit physics layer. `itu_propagation.py` and `fr3_propagation.py` show that the repository does not treat the environment as a purely abstract simulator. Instead, it encodes propagation behavior that is relevant to real telecom conditions, especially for **non-terrestrial networking, atmospheric attenuation, and FR3-specific path-loss behavior**.[2] [3]

This physics layer matters because learned control without propagation realism can produce models that look strong under simplified assumptions but break down when path characteristics change materially across provider classes and bands. By making propagation modules first-class components, the repository pushes UHCI closer to **telecom-shaped intelligence** rather than generic sequence modeling with wireless-themed variable names.[2] [3]

| Physics-facing component | What it contributes to UHCI |
|---|---|
| `itu_propagation.py` | Standards-aware attenuation and NTN-relevant propagation logic |
| `fr3_propagation.py` | FR3-specific path-loss behavior and coexistence-relevant modeling |
| `provider_registry.py` | Supplies the provider priors that tell the physics layer what kinds of links exist |
| `oran.py` | Connects a physics-aware control viewpoint to a live O-RAN-facing surface |

Taken together, these files mean that UHCI is grounded in the idea that **provider choice is inseparable from propagation conditions**.[1] [2] [3]

## Layer 3: Unified Environment and Real-Data Pathways

The next architectural layer is the environment system, and `unified_connectivity_env.py` is the central file. This module is where the provider ontology and propagation assumptions become a **controllable world** with observations, actions, rewards, provider interactions, and step dynamics.[4] The environment should therefore be read as the **operational substrate of UHCI** rather than as a narrow simulation convenience.

The design is important because it turns heterogeneous connectivity into an explicit decision surface. Rather than assuming one uniform wireless setting, the environment composes information about providers, performance state, dynamics, and rewards into a state-action formulation aligned with UHCI’s broader ambition.[4]

`data_pipeline.py` extends this by handling the path from external data sources into the modeled environment.[5] That is significant because it shows the repository is not limited to purely synthetic workflows. The architecture includes an explicit place where richer empirical telecom data can be ingested, normalized, and made useful to the control layer.[5]

`oran.py` extends the environment logic toward a live O-RAN-style control framing, showing how observations and actions can be situated inside an operational loop rather than being trapped in offline experimentation only.[6]

| Environment concern | Where it is implemented |
|---|---|
| Unified heterogeneous-provider simulation | `unified_connectivity_env.py` |
| Real-data ingestion and normalization | `data_pipeline.py` |
| O-RAN-facing control and observation flow | `oran.py` |
| Additional environment modules | Supporting environment implementations under `src/preceptualai/env/` |

Architecturally, this means the environment layer is no longer merely a training convenience. It is the **world-construction layer** for UHCI.[4] [5] [6]

## Layer 4: Structural Representation Learning

Heterogeneous connectivity requires a representation that can preserve relations among system entities rather than flattening them prematurely. That is the role of `hetero_gnn_encoder.py`.[7] This file is one of the clearest pieces of evidence that UHCI is meant to operate on **structured heterogeneous state** rather than on a single undifferentiated feature vector.

A graph-oriented encoder is a natural fit for this setting because provider classes, links, users, and system constraints are relational. The encoder can preserve distinctions among entity types and their interactions in a way that a purely flat state representation often cannot.[7]

| Representation concern | Why it matters in UHCI | Principal anchor |
|---|---|---|
| Heterogeneous entity handling | Different provider and network elements should not be merged blindly | `hetero_gnn_encoder.py` |
| Relational inductive bias | Connectivity decisions depend on relationships, not only local scalars | `hetero_gnn_encoder.py` |
| Structural compression for control | Produces a learned latent state suitable for downstream policy modules | `hetero_gnn_encoder.py` |

The structural layer shows that the repository is aimed at **relational intelligence over connectivity systems**, not merely recurrent processing of a flat state tensor.[7]

## Layer 5: Temporal Intelligence with CfC and Continuous-Time Backends

UHCI also treats time as a first-class modeling problem. The repository contains both `ltc_cell.py` and `ltc_cell_cfc.py`, which is important because it shows the temporal layer supports more than one continuous-time reasoning backend.[8] [9]

The presence of `ltc_cell_cfc.py` shows that the repository supports **CfC-capable continuous-time temporal reasoning**, which better matches the idea that wireless and connectivity control unfold over irregular and mixed timescales. At the same time, `ltc_cell.py` remains part of the temporal toolkit, giving the stack multiple ways to represent memory and adaptation over time.[8] [9]

This is one of the most important conceptual points in the repository. UHCI is not just state modeling plus a policy head; it is a system whose intelligence depends on handling **how conditions evolve over time**, including persistence, adaptation, and mixed control horizons.[8] [9]

| Temporal component | Role in the architecture |
|---|---|
| `ltc_cell.py` | Continuous-time temporal backend available in the core stack |
| `ltc_cell_cfc.py` | Continuous-time backend that makes CfC-style reasoning part of UHCI |
| Temporal integration inside the core stack | Lets the repository evaluate richer time-aware control strategies under one larger system frame |

The right way to read this layer is therefore not “the project uses one old recurrent cell,” but rather “the project treats **adaptive temporal reasoning** as a core requirement of heterogeneous connectivity intelligence.”[8] [9]

## Layer 6: Universal Agent and Decision Logic

At the decision layer, `universal_spectrum_agent.py` is the clearest sign that UHCI has become the repository’s primary system identity.[10] The file expresses a broader agentic abstraction that sits above the heterogeneous environment, structured representation modules, and temporal backends.

This matters because the agent layer is where the repository stops being a set of telecom-informed feature processors and becomes a **controller**. The universal agent translates the richer UHCI latent state into actions over connectivity options and system decisions.[10]

Empirical training and evaluation lanes still matter because they provide measured evidence for parts of the system. Under a correct UHCI-first reading, those lanes should be understood as **validation surfaces** inside the broader architecture rather than as the definition of the entire project.[10] [13]

| Decision component | Architectural interpretation |
|---|---|
| `universal_spectrum_agent.py` | UHCI-native control agent over heterogeneous state and richer action semantics |
| Benchmark-oriented policy lane | One measured learning path inside the broader system |
| Supporting actor/critic and core modules | Submodules and alternative building blocks within the larger research stack |

That distinction is essential to the documentation strategy. A repository can contain measured evaluation lanes and a much larger platform architecture at the same time. Here, **UHCI is the platform architecture**.[10] [13]

## Layer 7: Runtime Inference and Serving

A major reason the repository is more than an academic prototype is its runtime layer. `inference_engine.py` provides a runtime abstraction for loading and executing models in deployed settings.[11] `server.py` exposes this through a gRPC service with prediction, health, and metrics surfaces, turning the model into a live callable service rather than a static training artifact.[12]

This deployment-aware structure matters because telecom intelligence is only useful if it can be executed under service constraints. The runtime layer answers the question of **how the learned control logic is actually served, monitored, and queried**.[11] [12]

| Runtime surface | Role in deployed UHCI |
|---|---|
| `inference_engine.py` | General runtime abstraction for model loading and inference |
| `server.py` | gRPC prediction service, health checks, metrics, and observability |
| Supporting runtime modules | Tie deployment concerns back to model packaging and state handling |

This layer turns UHCI from a research formulation into something closer to a **telecom software component**.[11] [12]

## Layer 8: Low-Latency RT-RIC and AI-RAN Deployment Path

`dapp_engine.py` is one of the strongest files in the repository for demonstrating operational seriousness. It implements a low-latency path designed for RT-RIC-style deployment and latency-sensitive execution, including persistent hidden state and runtime optimizations oriented toward fast inference.[13] The file includes explicit latency-oriented machinery rather than assuming that a training-time model can be deployed unchanged.

`aerial_adapter.py` extends this runtime story toward NVIDIA Aerial and ARC-oriented deployment profiles, connecting the architecture to AI-RAN-style execution assumptions where AI and RAN workloads share accelerated infrastructure.[14]

| Deployment surface | What it contributes |
|---|---|
| `dapp_engine.py` | Near-real-time low-latency inference with persistent temporal state |
| `aerial_adapter.py` | Accelerator-native packaging path for AI-RAN-style deployment |
| `server.py` | Service boundary for integrating the runtime into surrounding infrastructure |

This layer is strategically important because it shows that UHCI is not only about learning a policy; it is also about **executing that policy where telecom control software can actually use it**.[12] [13] [14]

## Layer 9: O-RAN Control-Plane Integration

The repository includes explicit O-RAN-facing logic, especially through `oran.py` and `e2_adapter.py`.[6] [15] These files show how UHCI can connect to control loops that consume telemetry and produce policy-aligned actions in a way that reflects O-RAN concepts and boundaries.

This part of the architecture is important because it answers a practical adoption question. A controller can be technically interesting and still be operationally irrelevant if it has no integration surface. The presence of environment-side O-RAN abstractions and lower-level E2-facing logic means the repository is trying to define **where UHCI fits into live telecom control workflows**.[6] [15]

| O-RAN-facing component | Architectural meaning |
|---|---|
| `oran.py` | Frames live observation and control-loop context for UHCI |
| `e2_adapter.py` | Encodes the E2-facing measurement and control boundary |
| `server.py` | Exposes a service surface that can sit alongside control-plane infrastructure |

This is the point where the repository becomes a **closed-loop control architecture**, not just a training stack.[6] [12] [15]

## Layer 10: Non-RT Lifecycle Governance and Policy Distribution

`rapp_trainer.py` extends the system from training and inference into **lifecycle governance**.[16] This is a particularly important file because it encodes how models are cataloged, approved, deployed, monitored for degradation, and refreshed over time. It also creates policy-generation and distribution surfaces consistent with longer-timescale telecom management workflows.[16]

This means UHCI operates across multiple timescales. Some logic is near-real-time and inference-oriented. Some is service-oriented. Some belongs to governance and model-management loops. That multi-timescale structure is one of the clearest signs that the codebase is architecturally broader than any single benchmark workflow.[13] [16]

| Lifecycle component | Role in the architecture |
|---|---|
| `rapp_trainer.py` | Model cataloging, approval, monitoring, deployment, and retraining |
| A1 and policy-generation logic | Bridges learned models to longer-timescale control artifacts |
| Degradation detection and retraining pathways | Makes model maintenance part of the system, not an external afterthought |

## End-to-End Architectural Flow

The UHCI system can be read end to end as a pipeline from world definition to control execution.

```text
Provider taxonomy and priors
        +
Telecom propagation modeling
        ↓
Unified heterogeneous environment
        +
Real-data ingestion pathways
        ↓
Graph-structured state encoding
        +
Continuous-time temporal reasoning
        ↓
Universal decision agent
        ↓
Low-latency runtime and gRPC serving
        ↓
O-RAN integration, E2 interaction, and lifecycle governance
```

| End-to-end stage | Main files | Output of the stage |
|---|---|---|
| World definition | `provider_registry.py`, `itu_propagation.py`, `fr3_propagation.py` | A physics-aware heterogeneous connectivity universe |
| Decision environment | `unified_connectivity_env.py`, `data_pipeline.py`, `oran.py` | A controllable state-action world with empirical pathways |
| Representation | `hetero_gnn_encoder.py`, `ltc_cell.py`, `ltc_cell_cfc.py` | A structural-temporal latent state |
| Decision | `universal_spectrum_agent.py` | A learned control action |
| Runtime | `inference_engine.py`, `dapp_engine.py`, `server.py` | A callable low-latency service |
| Deployment and governance | `aerial_adapter.py`, `e2_adapter.py`, `rapp_trainer.py` | Operational integration and lifecycle management |

## Evidence Boundaries and Benchmark Interpretation

A good architecture document must also state what is and is not fully benchmarked. The repository does include empirical benchmark evidence through `benchmarks/benchmark.py` and the associated benchmark artifacts.[17] Those files support careful claims about measured performance in the included benchmark setting.

However, the correct architectural interpretation is not that every UHCI subsystem has already been validated by one equally mature end-to-end benchmark suite. The proper reading is more precise.

> **The repository implements a broad UHCI architecture, while the clearest directly aggregated benchmark evidence remains concentrated in the included benchmark suite.**

That distinction does not weaken the architectural claim. It strengthens the documentation by separating **implementation-backed scope** from **benchmark-backed scope**.[16] [17]

| Surface | Evidence posture |
|---|---|
| Included benchmark suite | Directly benchmarked with result artifacts |
| Provider, physics, and environment stack | Strongly implementation-backed and code-audited |
| Graph and temporal UHCI stack | Strongly implementation-backed; benchmark claims should be specific and bounded |
| Runtime and serving path | Code-backed, latency-conscious, and deployment-oriented |
| O-RAN and lifecycle layers | Architecturally explicit and implementation-backed, dependent on deployment context |

## How to Read the Repository Now

A new reader should understand the repository in the following order. First, UHCI defines the system-level ambition: a universal intelligence layer for heterogeneous connectivity. Second, the provider, physics, and environment stack define the world that intelligence operates in. Third, the structural and temporal modules define how the system reasons. Fourth, the runtime and O-RAN modules define how that intelligence is operationalized. Fifth, the included benchmark suite provides the clearest current empirical anchor for what has already been measured end to end.

| If you want to understand... | Read next |
|---|---|
| UHCI at a strategic and plain-language level | [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) and [`UHCI_RESEARCH_GUIDE.md`](UHCI_RESEARCH_GUIDE.md) |
| Training and empirical evidence boundaries | [`TRAINING_AND_EVALUATION.md`](TRAINING_AND_EVALUATION.md) and [`DATA_AND_REPRODUCIBILITY.md`](DATA_AND_REPRODUCIBILITY.md) |
| Runtime, xApp, dApp, and deployment surfaces | [`DEPLOYMENT_AND_XAPP_GUIDE.md`](DEPLOYMENT_AND_XAPP_GUIDE.md) |
| Configuration, commands, and interfaces | [`API_AND_INTERFACES.md`](API_AND_INTERFACES.md) |
| Directory-level repository structure | [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) |

## References

[1]: [Provider taxonomy in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[2]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[3]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[4]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[5]: [Unified data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[6]: [O-RAN environment module in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[7]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[8]: [Core LTC cell in `src/preceptualai/core/ltc_cell.py`](../src/preceptualai/core/ltc_cell.py)
[9]: [CfC-capable temporal backend in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[10]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[11]: [Inference engine in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[12]: [gRPC serving module in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[13]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[14]: [NVIDIA Aerial adapter in `src/preceptualai/xapp/aerial_adapter.py`](../src/preceptualai/xapp/aerial_adapter.py)
[15]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[16]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[17]: [Benchmark harness in `../benchmarks/benchmark.py`](../benchmarks/benchmark.py)
