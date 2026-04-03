# PreceptualAI

**PreceptualAI** is best understood as a repository for **Universal Heterogeneous Connectivity Intelligence (UHCI)**: a software-defined intelligence layer for making connectivity and spectrum decisions across **terrestrial, non-terrestrial, and hybrid wireless environments**. The codebase is not merely a narrow reinforcement-learning benchmark. A full line-by-line audit of the repository shows a much larger system that combines **provider-aware world modeling, telecom-physics-aware propagation, graph-structured state encoding, adaptive temporal reasoning, learned decision policies, real-data pathways, low-latency runtime execution, O-RAN-facing control integration, and lifecycle governance** into one architecture.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

This README is intentionally the most comprehensive document in the repository. A lay reader should be able to understand **what problem the repository addresses, why the problem matters now, what the solution actually is, why the approach is defensible, what evidence is already measured, and how the codebase is organized into a reconstructable system**. A researcher or engineer should be able to use this document as the top-level map for rebuilding the full UHCI stack from the rest of the Markdown documentation and the implementation itself.[13] [14] [15]

| Reader | Best starting path |
|---|---|
| Lay reader, operator, partner, or investor | Read this file, then [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) and [`docs/UHCI_RESEARCH_GUIDE.md`](docs/UHCI_RESEARCH_GUIDE.md). |
| Researcher or ML engineer | Read this file, then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md), and [`docs/DATA_AND_REPRODUCIBILITY.md`](docs/DATA_AND_REPRODUCIBILITY.md). |
| Deployment engineer or telecom integrator | Read this file, then [`docs/DEPLOYMENT_AND_XAPP_GUIDE.md`](docs/DEPLOYMENT_AND_XAPP_GUIDE.md) and [`docs/API_AND_INTERFACES.md`](docs/API_AND_INTERFACES.md). |
| Contributor or maintainer | Read [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md), [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md), and [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md). |

## The Problem We Solve

Modern wireless systems are no longer well described as one base station choosing one channel inside one fixed propagation regime. Real-world connectivity increasingly spans **multiple provider classes, multiple spectrum regimes, multiple control timescales, and multiple deployment surfaces**. A decision system may need to reason not only about interference and occupancy, but also about whether the right next action lies in **FR1, FR3, WiFi 7, HAPS, or a non-terrestrial option such as LEO, MEO, or GEO**, each with different latency, coverage, Doppler, handover, and coexistence behavior.[1] [6] [7] [16] [17]

The challenge is therefore broader than classical dynamic spectrum access. The real problem is that **connectivity intelligence must now operate across a heterogeneous wireless world while remaining fast enough for control loops, grounded enough in telecom physics, and structured enough to fit programmable deployment surfaces such as O-RAN and AI-RAN**.[18] [19] [20] [21]

> The central problem is no longer only “which channel should I pick next?” It is **how to build one decision layer that can reason across heterogeneous connectivity options, adapt over time, and operate inside real telecom software control loops**.

| Structural constraint | Why it matters operationally | Why narrow systems break down |
|---|---|---|
| Connectivity is heterogeneous | Different provider classes expose different constraints and opportunities | Flat single-environment formulations hide crucial distinctions |
| Propagation regimes differ | FR3, NTN, and atmospheric effects shape real performance envelopes | Simplified channel assumptions fail to capture deployment reality |
| Control timescales are mixed | Fast inference, persistent state, and slower governance loops all matter | One fixed-clock controller is an architectural mismatch |
| Telecom stacks are becoming programmable | O-RAN creates live software insertion points for control intelligence | Offline-only research code is insufficient |
| AI and RAN are converging operationally | AI-RAN creates infrastructure pathways for learned control | Legacy packaging does not fit accelerator-native operations |
| Claims must be auditable | Operators and researchers need reproducible evidence | Poorly documented systems are hard to trust or adopt |

## Our Solution: UHCI

**UHCI** is the repository’s answer to this problem. In code, UHCI is a layered architecture that combines **provider taxonomy, physics-aware environment modeling, heterogeneous state encoding, adaptive temporal reasoning, policy learning, runtime serving, and telecom lifecycle management** into one coherent system.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

The architecture starts by defining the world properly. `provider_registry.py` formalizes provider classes such as **LEO, MEO, GEO, HAPS, FR1, FR3, ISAC, and WiFi 7**, together with structured priors such as latency ranges, bandwidth, Doppler, handover periods, coverage, and coexistence assumptions.[1] That provider layer is then connected to explicit propagation modules such as `itu_propagation.py` and `fr3_propagation.py`, making telecom physics part of the decision problem rather than a post-hoc narrative.[6] [7]

On top of that world model, `unified_connectivity_env.py` constructs a heterogeneous control environment. The representation stack includes graph-oriented encoding in `hetero_gnn_encoder.py` and continuous-time temporal backends in `ltc_cell_cfc.py`, including **CfC** and related adaptive temporal pathways.[2] [4] [5] The policy layer is broadened through `universal_spectrum_agent.py`, which makes the repository’s center of gravity a **general connectivity-intelligence agent** rather than a narrow benchmark workflow.[3]

| UHCI layer | What it contributes | Principal code anchors |
|---|---|---|
| Provider ontology | Defines which connectivity classes exist and how they differ | `src/preceptualai/env/provider_registry.py` |
| Propagation realism | Encodes standards- and band-aware wireless behavior | `src/preceptualai/env/itu_propagation.py`, `src/preceptualai/env/fr3_propagation.py` |
| Unified environment | Builds observations, actions, rewards, and provider interactions | `src/preceptualai/env/unified_connectivity_env.py` |
| Real-data pipeline | Connects the environment to richer telecom datasets and ingestion paths | `src/preceptualai/env/data_pipeline.py` |
| Structural representation | Encodes heterogeneous system state relationally | `src/preceptualai/core/hetero_gnn_encoder.py` |
| Temporal intelligence | Provides adaptive continuous-time reasoning, including CfC-capable pathways | `src/preceptualai/core/ltc_cell_cfc.py` |
| Universal decision layer | Produces actions over the richer UHCI state space | `src/preceptualai/core/universal_spectrum_agent.py` |
| Runtime and deployment | Serves, executes, and governs the learned intelligence | `src/preceptualai/xapp/` |

## What UHCI Encompasses in This Repository

A line-by-line repository audit shows that UHCI is not just an environment rename and not just a model swap. It spans multiple technical layers that, in many projects, would be split across separate research and production repositories.

```text
Provider taxonomy + telecom physics
                ↓
Unified heterogeneous environment + data ingestion
                ↓
Graph-structured encoding + continuous-time temporal reasoning
                ↓
Universal agent / learned control policy
                ↓
Low-latency inference + serving + AI-RAN packaging
                ↓
E2 / O-RAN integration + Non-RT lifecycle governance
```

| UHCI subsystem | Repository evidence | Architectural meaning |
|---|---|---|
| Provider taxonomy | `provider_registry.py` | The system reasons across provider classes, not anonymous channels only |
| Physics layer | `itu_propagation.py`, `fr3_propagation.py` | Decisions are constrained by propagation-aware modeling |
| Unified environment | `unified_connectivity_env.py` | The world state is heterogeneous and provider-aware |
| Data ingestion | `data_pipeline.py` | The stack is designed to connect to richer empirical telecom inputs |
| Universal agent | `universal_spectrum_agent.py` | Decision logic is broader than one narrow benchmark actor |
| Graph encoder | `hetero_gnn_encoder.py` | Connectivity state is represented structurally, not just as flat vectors |
| Continuous-time backend | `ltc_cell_cfc.py` | Time dynamics are treated as a first-class modeling problem |
| RT inference path | `dapp_engine.py` | The system includes a near-real-time execution lane |
| Serving surface | `server.py`, `inference_engine.py` | The model is exposed as a callable live service |
| Control-plane integration | `oran.py`, `e2_adapter.py` | The architecture is designed with O-RAN loop boundaries in mind |
| Lifecycle governance | `rapp_trainer.py` | Model approval, monitoring, degradation checks, and retraining are explicit |
| AI-RAN packaging | `aerial_adapter.py` | The stack contemplates accelerator-native telecom deployment |

## Why This Architecture Is Defensible

The repository’s defensibility comes from **combination, structure, and execution path**, not from a single isolated modeling trick.

First, the code combines **heterogeneous-provider reasoning** with **telecom-physics-aware modeling**. This matters because a connectivity controller that ignores provider differences, latency envelopes, handover dynamics, and propagation constraints is not addressing the full real-world problem.[1] [2] [6] [7]

Second, the architecture combines **structural state encoding** with **continuous-time temporal reasoning**. That is important because heterogeneous connectivity is relational and mixed-timescale by nature. A graph-oriented encoder and adaptive temporal backend are more aligned with that reality than a generic fixed-timescale flat controller.[3] [4] [5]

Third, the repository includes a genuine **deployment story**. The presence of runtime abstractions, gRPC serving, RT-oriented inference, E2-facing integration, and Non-RT lifecycle governance means the codebase is trying to answer not only whether a model can be trained, but also **how it would be executed, monitored, and updated inside a telecom software environment**.[8] [9] [10] [11] [12]

| Defensibility vector | Why it matters | Repository evidence |
|---|---|---|
| **Heterogeneous-provider worldview** | Expands value from narrow spectrum selection to broader connectivity intelligence | `provider_registry.py`, `unified_connectivity_env.py` |
| **Physics-aware modeling** | Grounds decision logic in telecom reality | `itu_propagation.py`, `fr3_propagation.py` |
| **Graph-structured state** | Preserves provider identity and system relationships | `hetero_gnn_encoder.py` |
| **Adaptive temporal reasoning** | Aligns controller memory with mixed wireless timescales | `ltc_cell_cfc.py` |
| **Universal control abstraction** | Makes the system extensible beyond one benchmark workflow | `universal_spectrum_agent.py` |
| **Deployment realism** | Bridges research code to callable service interfaces and real-time paths | `server.py`, `dapp_engine.py`, `aerial_adapter.py` |
| **Lifecycle governance** | Treats retraining, approval, rollout, and degradation as part of the system | `rapp_trainer.py` |
| **Auditable evidence posture** | Separates benchmark-backed claims from broader architecture claims | Benchmark artifacts and docs system |

## Why Now

The timing argument for UHCI is strong because **infrastructure, standards, and policy are converging**.

NVIDIA’s AI-RAN framing highlights a future in which AI and radio workloads coexist on accelerated infrastructure, creating a concrete execution substrate for learned telecom control.[18] O-RAN has made software-driven control loops more operationally real through the **Near-RT RIC**, **xApp**, and related open interfaces, which means the industry now has insertion points for programmable network intelligence.[19] At the same time, public-sector spectrum strategy is moving toward more adaptive, coexistence-aware, and innovation-oriented spectrum use, as reflected in the U.S. National Spectrum R&D Plan and NTIA’s AI-RAN-oriented program direction.[20] [21]

> The opportunity exists now because the industry finally has both **a place to run connectivity intelligence** and **a reason to demand more adaptive cross-provider decision systems**.

| Timing driver | What changed | Why UHCI fits now |
|---|---|---|
| **AI-RAN infrastructure** | AI and RAN workloads increasingly share accelerated platforms | UHCI already includes serving, RT inference, and AI-RAN packaging surfaces |
| **O-RAN control surfaces** | RIC-oriented software loops make deployment practical | UHCI includes xApp, dApp, E2, and lifecycle-oriented modules |
| **Dynamic spectrum pressure** | Policy momentum favors more adaptive coexistence-aware operation | Provider- and propagation-aware control becomes more valuable |
| **Need for cross-layer reasoning** | Future networks span terrestrial and non-terrestrial assets | UHCI explicitly models heterogeneous providers and propagation regimes |
| **Demand for auditability** | Operators and researchers increasingly demand code-backed claims | The repository now includes a documentation system designed for reconstruction and diligence |

## Technical Architecture, End to End

UHCI can be read as a full control stack. The environment layer defines what the system can observe and act on. The intelligence layer transforms that state into a structured latent representation and chooses an action. The runtime layer turns that decision into a callable service. The governance layer manages how models are approved, monitored, and refreshed over time.[1] [2] [3] [4] [5] [8] [9] [10] [11] [12]

| Stage | Primary files | What happens |
|---|---|---|
| World definition | `provider_registry.py`, `itu_propagation.py`, `fr3_propagation.py` | The connectivity universe and its physical constraints are defined |
| Decision environment | `unified_connectivity_env.py`, `data_pipeline.py` | Observations, actions, rewards, and empirical data pathways are assembled |
| Representation learning | `hetero_gnn_encoder.py`, `ltc_cell_cfc.py` | Heterogeneous state is encoded structurally and temporally |
| Decision policy | `universal_spectrum_agent.py` | The agent chooses a control action over the richer state space |
| Runtime execution | `inference_engine.py`, `dapp_engine.py`, `server.py` | The trained model is executed and exposed as a service |
| Telecom integration | `oran.py`, `e2_adapter.py`, `aerial_adapter.py` | The model is connected to O-RAN and accelerator-oriented deployment paths |
| Lifecycle governance | `rapp_trainer.py` | Model cataloging, approval, monitoring, policy generation, and retraining are handled |

## Measured Performance and Safe Benchmark Boundaries

The repository includes empirical benchmark evidence, but that evidence should be described precisely. The most mature directly aggregated benchmark surface currently present in the repository is the control benchmark implemented in `benchmarks/benchmark.py` and summarized in `benchmark_summary.json`.[13] [14] That benchmark provides measured results under a specific included evaluation setting.

The safe interpretation is therefore straightforward: **the repository contains measured improvement on important canonical control metrics within the included benchmark suite**, while the broader UHCI architecture is primarily supported by code-level implementation evidence and subsystem-level design, not by a single equally mature end-to-end aggregated benchmark covering every deployment and provider scenario.[13] [14]

| Metric in the included benchmark suite | Reported result | Safe interpretation |
|---|---:|---|
| **Success rate** | **0.6337 ± 0.0044** | Best reported result in the included benchmark suite on successful transmissions.[14] |
| **Collision rate** | **0.3663 ± 0.0044** | Best reported result in the included benchmark suite on lowest collision rate.[14] |
| **Spectral efficiency** | **0.6337 ± 0.0044** | Best reported result in the included benchmark suite on spectral efficiency.[14] |
| **Mean reward** | Best reported value is higher for another included model | The repository should not claim universal reward leadership from this artifact alone.[14] |
| **Fairness** | Near-leading but not first in the included evaluation | The repository should describe fairness carefully and precisely.[14] |
| **Mean inference latency** | Real-time-capable in the included benchmark framing | The repository should not claim fastest latency across all included models from this artifact alone.[14] |

This distinction matters. It lets the documentation make a strong, honest statement: **the codebase already shows a broad UHCI system architecture, and it already contains measured leadership on key benchmark metrics inside the included evaluation lane, but the architecture is broader than the current single aggregated benchmark story**.

## How UHCI Reaches Deployment

One of the strongest aspects of the codebase is that it goes beyond training. `dapp_engine.py` implements a low-latency inference path with persistent temporal state and deployment-oriented optimizations. `server.py` exposes gRPC prediction, health, and metrics interfaces, making the intelligence layer inspectable as a live service. `inference_engine.py` provides the main runtime abstraction for loading and executing models.[8] [9] [10]

At a slower lifecycle timescale, `rapp_trainer.py` provides model cataloging, approval, deployment logic, degradation monitoring, and policy-generation functionality aligned with a Non-RT RIC or SMO-style setting.[12] `e2_adapter.py` and `oran.py` express the lower-level control and measurement boundaries that connect the decision system to O-RAN-style loops.[11] `aerial_adapter.py` extends the story toward NVIDIA Aerial and ARC-oriented deployment profiles, which is why the repository fits the broader AI-RAN timing argument.[11] [18]

| Deployment surface | What it does |
|---|---|
| `src/preceptualai/xapp/dapp_engine.py` | Executes low-latency inference for RT-oriented deployment |
| `src/preceptualai/xapp/server.py` | Serves the model over gRPC with health and metrics endpoints |
| `src/preceptualai/xapp/inference_engine.py` | Provides the main loaded-model runtime abstraction |
| `src/preceptualai/xapp/e2_adapter.py` | Encodes the E2-facing measurement and control boundary |
| `src/preceptualai/env/oran.py` | Frames the live O-RAN control-loop environment |
| `src/preceptualai/xapp/rapp_trainer.py` | Manages cataloging, approval, monitoring, and policy artifacts |
| `src/preceptualai/xapp/aerial_adapter.py` | Packages the runtime for NVIDIA Aerial and ARC-oriented deployment |

## Our Moats

The repository’s moats are best understood as **compound system moats** rather than as one secret algorithm.

The first moat is **system breadth with internal coherence**. The codebase does not stop at one policy network. It ties together provider ontology, propagation realism, structured representation learning, continuous-time temporal reasoning, runtime execution, and telecom lifecycle logic in one inspectable system.[1] [2] [3] [4] [5] [8] [9] [10] [11] [12]

The second moat is **domain-shaped architecture**. Provider priors, ITU-aligned propagation logic, FR3-specific behavior, and O-RAN-facing deployment modules create a code structure that is much harder to reproduce than a generic RL agent trained in a toy simulator.[1] [6] [7] [11] [12]

The third moat is **evidence discipline**. The repository now separates benchmark-backed claims from architectural claims and makes the rebuild path explicit through a documentation system intended for lay readers, engineers, and researchers alike.[13] [14] [15]

| Moat | Why it is hard to replicate quickly |
|---|---|
| Integrated heterogeneous-provider stack | Requires more than just a model; it requires a coherent world model and control formulation |
| Physics-aware connectivity modeling | Requires telecom-domain priors and propagation-aware environment design |
| Temporal and structural intelligence combined | Requires both graph reasoning and adaptive time modeling, not only flat sequence learning |
| Telecom deployment path | Requires runtime, observability, O-RAN integration, and lifecycle management |
| Documentation and evidence clarity | Reduces ambiguity for adopters while increasing trust and reconstructability |

## What the Repository Is and Is Not

This repository **is** a comprehensive, code-backed architecture for UHCI, including heterogeneous-provider modeling, propagation-aware environments, structural and temporal learning modules, runtime serving, and telecom lifecycle integration.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

It **is not** yet best described as a single uniform benchmarked product where every subsystem has exactly the same empirical maturity. The strongest aggregated quantitative evidence remains concentrated in the included benchmark suite, while the broader UHCI architecture is presently best understood as implementation-backed, technically extensive, and deployment-oriented.[13] [14]

That distinction is important because it allows the repository to be read honestly and seriously. A careful reader should conclude that **the architectural scope is already broad, the benchmark evidence is real but bounded, and the overall repository is larger than any one benchmarked model path**.

## How to Recreate the System from Documentation

A careful reader can rebuild the repository in stages. The recommended path is to understand the system architecture first, then install and run the environment and training surfaces, then validate the benchmark evidence, and finally move into deployment-facing integration.

| Build objective | Recommended path |
|---|---|
| Understand the problem, solution, moats, and timing | This README and [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) |
| Understand the full technical decomposition | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Understand the complete UHCI research stack | [`docs/UHCI_RESEARCH_GUIDE.md`](docs/UHCI_RESEARCH_GUIDE.md) |
| Install and run the repository locally | [`docs/SETUP_AND_QUICKSTART.md`](docs/SETUP_AND_QUICKSTART.md) |
| Reproduce benchmark evidence responsibly | [`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md) and [`docs/DATA_AND_REPRODUCIBILITY.md`](docs/DATA_AND_REPRODUCIBILITY.md) |
| Integrate or deploy the runtime | [`docs/DEPLOYMENT_AND_XAPP_GUIDE.md`](docs/DEPLOYMENT_AND_XAPP_GUIDE.md) and [`docs/API_AND_INTERFACES.md`](docs/API_AND_INTERFACES.md) |
| Navigate the repository structurally | [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) |

## Documentation System

The repository now uses a layered documentation system specifically to prevent the project from being misread as a narrow one-model codebase. The documentation is organized so that a layman can understand the business and technical narrative, while a researcher or engineer can reconstruct the implementation and its empirical boundaries from Markdown alone.

| Documentation file | Purpose |
|---|---|
| [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md) | Master map of the documentation system |
| [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) | Plain-language explanation of UHCI and why it matters |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Detailed decomposition of the full technical system |
| [`docs/UHCI_RESEARCH_GUIDE.md`](docs/UHCI_RESEARCH_GUIDE.md) | Deepest explanation of the heterogeneous-connectivity research stack |
| [`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md) | Benchmark and empirical-evidence guide |
| [`docs/DATA_AND_REPRODUCIBILITY.md`](docs/DATA_AND_REPRODUCIBILITY.md) | Boundaries of what is directly reproducible |
| [`docs/DEPLOYMENT_AND_XAPP_GUIDE.md`](docs/DEPLOYMENT_AND_XAPP_GUIDE.md) | Serving, xApp, dApp, and deployment path |
| [`docs/API_AND_INTERFACES.md`](docs/API_AND_INTERFACES.md) | Scripts, configuration surfaces, and protocol contracts |
| [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) | Directory-by-directory map of the repository |

## References

[1]: [Provider taxonomy in `src/preceptualai/env/provider_registry.py`](src/preceptualai/env/provider_registry.py)
[2]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](src/preceptualai/env/unified_connectivity_env.py)
[3]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](src/preceptualai/core/universal_spectrum_agent.py)
[4]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](src/preceptualai/core/hetero_gnn_encoder.py)
[5]: [CfC temporal backend in `src/preceptualai/core/ltc_cell_cfc.py`](src/preceptualai/core/ltc_cell_cfc.py)
[6]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](src/preceptualai/env/itu_propagation.py)
[7]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](src/preceptualai/env/fr3_propagation.py)
[8]: [Unified real-data pipeline in `src/preceptualai/env/data_pipeline.py`](src/preceptualai/env/data_pipeline.py)
[9]: [RT-oriented dApp engine in `src/preceptualai/xapp/dapp_engine.py`](src/preceptualai/xapp/dapp_engine.py)
[10]: [gRPC serving module in `src/preceptualai/xapp/server.py`](src/preceptualai/xapp/server.py)
[11]: [O-RAN environment surface in `src/preceptualai/env/oran.py`](src/preceptualai/env/oran.py) and [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](src/preceptualai/xapp/e2_adapter.py)
[12]: [Non-RT RIC lifecycle service in `src/preceptualai/xapp/rapp_trainer.py`](src/preceptualai/xapp/rapp_trainer.py)
[13]: [Benchmark harness in `benchmarks/benchmark.py`](benchmarks/benchmark.py)
[14]: [Aggregated benchmark summary in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](benchmarks/results/benchmark_results_full/benchmark_summary.json)
[15]: [Documentation index in `docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md)
[16]: [UHCI training entry point in `scripts/train_uhci.py`](scripts/train_uhci.py)
[17]: [Project manifest in `pyproject.toml`](pyproject.toml)
[18]: [NVIDIA, "AI-RAN Solutions for 5G & 6G Cellular Networks"](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)
[19]: [O-RAN Software Community documentation](https://docs.o-ran-sc.org/en/k-release/)
[20]: [NITRD, "National Spectrum Research and Development Plan 2024"](https://www.nitrd.gov/pubs/National-Spectrum-RD-Plan-2024.pdf)
[21]: [NTIA, "NTIA Seeks Feedback on New Direction for Innovation Fund That Focuses on AI-RAN"](https://www.ntia.gov/blog/2026/ntia-seeks-feedback-new-direction-innovation-fund-focuses-ai-ran)
