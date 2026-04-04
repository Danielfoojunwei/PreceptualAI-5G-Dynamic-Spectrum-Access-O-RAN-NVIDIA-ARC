# PreceptualAI UHCI: Universal Heterogeneous Connectivity Intelligence

**PreceptualAI** is best understood as a repository for **Universal Heterogeneous Connectivity Intelligence (UHCI)**: a software-defined intelligence layer for making connectivity, spectrum, and control decisions across **terrestrial, non-terrestrial, and hybrid wireless systems**. The codebase is broader than a narrow reinforcement-learning benchmark. A repository-wide audit shows a layered architecture that combines **provider-aware world modeling, telecom-physics-aware propagation, heterogeneous graph encoding, adaptive continuous-time temporal reasoning, learned decision policies, low-latency serving, O-RAN-facing control integration, and lifecycle governance** into one system.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16]

This README has been redesigned so that a technical reader, operator, researcher, or collaborator can understand **what problem UHCI addresses, why the timing matters, what each subsystem does, how the technologies interact, what empirical evidence is already present in the repository, and which standards and research references anchor the design**.[17] [18] [19] [20] [21] [22] [23] [24] [25] [26] [27] [28] [29] [30] [31] [32] [33] [34]

| Reader | Best starting path |
|---|---|
| Executive, partner, or first-time reader | Read this README, then `docs/UHCI_ARCHITECTURE_REFERENCE_EN.md`. |
| Researcher or ML engineer | Read this README, then `docs/ARCHITECTURE.md`, `docs/TRAINING_AND_EVALUATION.md`, and `docs/UHCI_ARCHITECTURE_REFERENCE_EN.md`. |
| Telecom integrator or deployment engineer | Read this README, then `docs/API_AND_INTERFACES.md`, `docs/DEPLOYMENT_AND_XAPP_GUIDE.md`, and `docs/UHCI_ARCHITECTURE_REFERENCE_EN.md`. |
| Chinese-speaking reader | Read `docs/UHCI_ARCHITECTURE_REFERENCE_ZH.md` after this README. |

## Executive Summary

Modern wireless control is no longer a single-base-station, single-band optimization problem. Real deployments increasingly span **FR1, FR3, Wi-Fi 7, HAPS, ISAC, and non-terrestrial options such as LEO, MEO, and GEO**, each with different latency envelopes, propagation behavior, mobility dynamics, and operational constraints.[1] [7] [8] [22] [23] [24] The core architectural question is therefore larger than channel selection alone. The question is how to build **one intelligence layer** that can reason across heterogeneous connectivity options, preserve structural relationships, adapt over irregular timescales, and still fit programmable telecom execution surfaces such as **O-RAN** and **AI-RAN**.[12] [13] [14] [25] [26] [27] [31] [32]

UHCI answers that problem with a layered system. At the bottom, a provider registry and propagation modules define the physical and operational world. In the middle, a unified environment and data pipeline transform that world into a structured decision process. On top of that, heterogeneous graph encoding and continuous-time temporal models compress the state into actionable representations for a universal decision policy. Around the policy, the repository provides serving interfaces, near-real-time execution paths, E2-oriented integration surfaces, lifecycle governance, and an optional federated-learning extension.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [18] [19] [20]

> “Native Artificial Intelligence (AI) is the enabler technology for 6G. The RAN Intelligence Controller (RIC) of O-RAN is the potential approach for native AI.” — *O-RAN next Generation Research Group, RR-2023-02* [25]

> “A unified data ingestion model is emerging as a key requirement.” — *O-RAN next Generation Research Group, RR-2023-03* [26]

These statements align closely with the repository design: the codebase includes a unified data pipeline, O-RAN-facing surfaces, model lifecycle control, and runtime execution paths intended for programmable network intelligence rather than offline experimentation only.[9] [10] [11] [12] [13] [14]

## The Problem UHCI Solves

Wireless systems are becoming **heterogeneous by construction**. Networks increasingly blend terrestrial radio, non-terrestrial assets, edge compute, and software-controlled orchestration. The result is that a controller must reason not only about spectrum occupancy and interference, but also about provider identity, coverage geometry, propagation regime, mobility, latency, coexistence, and deployment surface.[1] [2] [7] [8] [22] [23] [24]

Classical formulations become brittle in this setting because they flatten the world too early. If all links are treated as interchangeable channels, the system loses the distinctions that determine whether a decision is actually deployable. UHCI instead treats heterogeneity as a first-class modeling assumption. It preserves multiple provider types, explicit propagation modules, structural relationships among entities, and mixed-timescale temporal behavior before producing actions.[1] [2] [4] [5] [6] [7] [8]

| Structural challenge | Why it matters operationally | How UHCI addresses it |
|---|---|---|
| Connectivity classes differ | LEO, GEO, HAPS, FR1, FR3, and Wi-Fi expose different coverage, latency, and interference patterns | A provider ontology formalizes the heterogeneous world before learning begins.[1] |
| Propagation is regime-dependent | NTN and terrestrial links obey different physical assumptions | Dedicated propagation modules keep telecom physics inside the control loop.[7] [8] [23] [24] |
| State is relational | Network entities, links, and contexts interact through typed relationships | A heterogeneous graph encoder preserves node and relation structure.[4] [30] |
| Time is irregular | Wireless dynamics evolve across fast and slow timescales | LTC and CfC-capable temporal modules support adaptive continuous-time memory.[5] [6] [28] [29] |
| Deployment is programmable | O-RAN and AI-RAN create live insertion points for intelligence | Runtime, gRPC, E2, and lifecycle modules bridge training to operations.[10] [11] [12] [13] [14] [25] [26] [27] [31] [32] |
| Continuous improvement matters | Models must be monitored, retrained, and redeployed safely | Non-RT lifecycle services and federated interfaces are explicit in the repository.[14] [18] [19] [20] |

## Overall System Architecture

The architecture below summarizes how the repository organizes the full UHCI stack from operating context to runtime control and learning lifecycle.

![UHCI Overall Architecture](docs/assets/uhci_overall_architecture.png)

UHCI is organized as a **stack of cooperating layers** rather than a single monolithic model. The lower layers define what the world is, the middle layers define how the world is encoded and predicted, and the upper layers define how intelligence is served, integrated, and maintained. This is important because telecom intelligence fails in practice when any of these layers is absent. A policy without a realistic world model is not trustworthy, and a model without a deployment surface is not operationally useful.[1] [2] [3] [4] [5] [7] [8] [10] [11] [12] [14] [25] [26] [27]

| Architecture layer | Primary responsibility | Principal repository anchors |
|---|---|---|
| World model and provider ontology | Define provider classes, priors, and heterogeneous connectivity categories | `src/preceptualai/env/provider_registry.py` [1] |
| Telecom physics | Model NTN and terrestrial propagation behavior | `src/preceptualai/env/itu_propagation.py`, `src/preceptualai/env/fr3_propagation.py` [7] [8] |
| Unified environment | Transform the wireless world into observations, actions, rewards, and transitions | `src/preceptualai/env/unified_connectivity_env.py` [2] |
| Data ingestion and normalization | Feed empirical telecom signals and measurements into the environment | `src/preceptualai/env/data_pipeline.py` [9] |
| Structural representation | Encode typed entities and relations without flattening heterogeneity away | `src/preceptualai/core/hetero_gnn_encoder.py` [4] |
| Temporal intelligence | Model irregular and continuous-time dynamics | `src/preceptualai/core/ltc_cell.py`, `src/preceptualai/core/ltc_cell_cfc.py` [5] [6] |
| Universal decision layer | Learn and execute connectivity or spectrum actions | `src/preceptualai/core/universal_spectrum_agent.py` [3] |
| Serving and real-time runtime | Expose the learned intelligence as a callable system | `src/preceptualai/xapp/inference_engine.py`, `src/preceptualai/xapp/dapp_engine.py`, `src/preceptualai/xapp/server.py` [10] [11] [12] |
| O-RAN integration | Connect control logic to O-RAN semantics and E2 surfaces | `src/preceptualai/env/oran.py`, `src/preceptualai/xapp/e2_adapter.py` [13] [14] |
| Lifecycle governance | Monitor, approve, retrain, and redeploy models | `src/preceptualai/xapp/rapp_trainer.py` [15] |
| Federated extension | Support distributed training and model exchange across sites | `src/preceptualai/federated/aggregator.py`, `proto/preceptualai_fl.proto` [19] [20] |

## How the Technologies Work Together

The key to understanding UHCI is to follow the **end-to-end transformation of information**. The system does not begin with a neural network. It begins with a formal description of which kinds of connectivity resources exist and what their engineering priors look like. The provider registry expresses that ontology and assigns structured characteristics to provider categories such as LEO, MEO, GEO, HAPS, FR1, FR3, ISAC, and Wi-Fi 7.[1] That registry is then paired with propagation logic so that any downstream control logic remains tethered to link realism rather than only to abstract reward shaping.[7] [8] [23] [24]

Once the world has been defined, the unified environment turns it into a decision process with explicit state, action, reward, and transition semantics. The data pipeline expands the same environment into a more empirical operating mode by supporting richer telemetry and dataset ingestion.[2] [9] This design is important because O-RAN-native and AI-native architectures increasingly depend on large, multi-layer data flows, and O-RAN’s own research reports describe unified data ingestion and distributed intelligence as central architectural requirements.[25] [26]

The representation stage then preserves structure instead of discarding it. The heterogeneous graph encoder is well aligned with this need because heterogeneous graph research explicitly addresses settings in which entities and links have different types and semantic roles.[4] [30] For wireless intelligence, that means providers, links, nodes, and contexts can be encoded as a relational system instead of being collapsed into a flat vector too early.

Temporal modeling sits alongside structural modeling because wireless control is not only relational; it is also dynamic across mixed timescales. Liquid Time-constant Networks and Closed-form Continuous-time Neural Models provide the research basis for the repository’s LTC and CfC-capable temporal backends. These model classes are designed for time-continuous sequence processing, stable bounded dynamics, and efficient continuous-time reasoning, which makes them relevant when network state evolves irregularly rather than at one fixed step size.[5] [6] [28] [29]

Finally, the universal agent consumes the encoded state and produces actions over the broader heterogeneous connectivity space. Around the agent, the repository provides an inference engine, a low-latency dApp engine, a gRPC serving surface, O-RAN-facing adapters, lifecycle services, and federated-learning contracts. In other words, the model is not treated as the whole system. It is treated as one component inside an operational intelligence architecture.[3] [10] [11] [12] [13] [14] [15] [18] [19] [20]

| Technology | What it does in UHCI | Why it is technically appropriate |
|---|---|---|
| Provider registry | Encodes heterogeneous resource classes and priors | Prevents the controller from assuming all links are equivalent.[1] |
| ITU and FR3 propagation modules | Constrain behavior using telecom propagation logic | Keeps decisions physically grounded.[7] [8] [23] [24] |
| Unified environment | Converts the wireless system into a learnable control process | Provides the operational abstraction on which training and inference depend.[2] |
| Data pipeline | Normalizes and feeds richer telecom data sources | Aligns with O-RAN’s emphasis on unified data ingestion.[9] [26] |
| Heterogeneous GNN | Encodes typed entities and relations | Matches the relational structure of heterogeneous wireless systems.[4] [30] |
| LTC and CfC | Model irregular continuous-time dynamics | Matches multi-timescale network behavior and offers efficient sequence modeling.[5] [6] [28] [29] |
| Universal policy agent | Produces decisions over the full heterogeneous space | Generalizes beyond one narrow benchmark policy.[3] |
| dApp and inference runtime | Enable near-real-time execution | Makes the controller operationally usable.[10] [11] |
| gRPC and E2 surfaces | Expose the model to external systems and control planes | Support deployable software integration.[12] [14] |
| rApp lifecycle services | Govern training, approval, deployment, and degradation response | Treat model management as part of the architecture, not an afterthought.[15] |
| Federated contracts and aggregator | Extend learning across distributed sites | Support multi-site model improvement under heterogeneous deployment conditions.[19] [20] |

## Runtime, Control, and Lifecycle View

The second architecture view focuses on how UHCI behaves across online control, O-RAN interaction, offline learning, and optional federated updates.

![UHCI Control and Lifecycle](docs/assets/uhci_control_lifecycle.png)

This control-lifecycle view shows that UHCI contains **two tightly linked loops**. The first is an online loop in which telemetry becomes state, state becomes encoded representation, and the policy produces actions that affect the network. The second is a slower learning and governance loop in which observations are accumulated, models are trained and benchmarked, deployment decisions are made, and updated models are pushed back into runtime. This separation matches how O-RAN research describes near-real-time and non-real-time functions, and it helps explain why the repository contains both execution-focused modules and lifecycle-focused modules.[10] [11] [12] [13] [14] [15] [25] [26] [27]

| Loop | Main purpose | Main repository anchors |
|---|---|---|
| Online control loop | Observe, encode, infer, and act on current wireless conditions | `oran.py`, `hetero_gnn_encoder.py`, `ltc_cell_cfc.py`, `universal_spectrum_agent.py`, `inference_engine.py`, `dapp_engine.py`, `e2_adapter.py` [4] [6] [10] [11] [13] [14] |
| Service interface loop | Expose inference and metrics to external systems | `server.py`, `proto/preceptualai.proto` [12] [18] |
| Offline learning loop | Train, evaluate, and select model variants | `scripts/train_uhci.py`, `benchmarks/benchmark.py`, benchmark artifacts [16] [17] |
| Governance loop | Approve, monitor, and redeploy models through non-RT workflows | `rapp_trainer.py` [15] |
| Federated loop | Collect distributed updates and aggregate models across sites | `aggregator.py`, `proto/preceptualai_fl.proto` [19] [20] |

## Subsystems and Their Roles

UHCI is easier to understand when each subsystem is separated by responsibility instead of by directory name alone. The table below summarizes the major subsystems, what they own, and why they are necessary to the whole architecture.

| Subsystem | What it owns | Why it matters to the full system |
|---|---|---|
| Provider and ontology subsystem | Provider categories, priors, heterogeneity assumptions | Defines the design space of connectivity choices.[1] |
| Propagation subsystem | NTN and terrestrial radio behavior | Prevents learning from diverging from telecom reality.[7] [8] |
| Environment subsystem | State, action, reward, transition logic | Converts physics and provider logic into a controllable problem.[2] |
| Data subsystem | Ingestion, normalization, and empirical pathways | Connects the intelligence layer to operational measurements.[9] |
| Graph representation subsystem | Typed node and edge encoding | Captures multi-entity relations in a structured form.[4] [30] |
| Temporal subsystem | LTC and CfC-capable recurrent logic | Preserves evolving context over irregular timescales.[5] [6] [28] [29] |
| Decision subsystem | Universal policy and critic logic | Produces actions over the heterogeneous action space.[3] |
| Runtime subsystem | Inference engine, dApp, serving | Makes the learned controller callable and low-latency.[10] [11] [12] |
| O-RAN interface subsystem | O-RAN environment surface and E2 adapter | Bridges the model to programmable RAN control semantics.[13] [14] [25] [26] [27] |
| Lifecycle subsystem | rApp-style monitoring, approval, retraining | Makes model governance explicit.[15] |
| Federated subsystem | Aggregation and distributed contracts | Extends the architecture across multiple sites and clients.[19] [20] |

## Benchmarks and Measured Evidence

The benchmark suite included in the repository already provides useful evidence, but the evidence is **multi-dimensional**, not one-dimensional. Different models lead on different criteria. That distinction matters for honest documentation.[16] [17]

In the included `benchmark_summary.json`, **`sac_ltc`** is the strongest model on **success rate**, **collision rate** when lower is better, and **spectral efficiency**. **`sac_lstm`** is strongest on **mean reward** and **inference latency**. **`sac_lfm`** is strongest on **Jain fairness**. These results suggest that the repository already supports a meaningful trade-space across operational performance, fairness, and runtime cost rather than a single universal winner.[17]

| Model | Mean reward | Success rate | Collision rate | Spectral efficiency | Jain fairness | Mean inference (ms) | P99 inference (ms) |
|---|---|---|---|---|---|---|---|
| `sac_lfm` | 35.8460 ± 2.3739 | 0.6273 ± 0.0056 | 0.3727 ± 0.0056 | 0.6273 ± 0.0056 | **0.99636 ± 0.00008** | 1.2802 ± 0.0539 | 6.8623 ± 0.0781 |
| `sac_lstm` | **50.0267 ± 1.8831** | 0.6251 ± 0.0047 | 0.3749 ± 0.0047 | 0.6251 ± 0.0047 | 0.99495 ± 0.00076 | **0.8276 ± 0.0200** | **2.2159 ± 0.2607** |
| `ppo_lstm` | 49.5080 ± 1.3341 | 0.6262 ± 0.0055 | 0.3738 ± 0.0055 | 0.6262 ± 0.0055 | 0.99518 ± 0.00044 | 4.1140 ± 0.0565 | 6.1734 ± 0.0942 |
| `sac_ltc` | 38.3313 ± 2.1778 | **0.6337 ± 0.0044** | **0.3663 ± 0.0044** | **0.6337 ± 0.0044** | 0.99618 ± 0.00105 | 1.5829 ± 0.0261 | 2.5141 ± 0.1348 |

A careful reading of these numbers supports three claims. First, the repository already contains **low-latency viable controllers**, because all listed mean inference times are in the millisecond range and the strongest latency result is below one millisecond.[17] Second, the temporal architecture family is meaningful, because LSTM-, LTC-, and related variants expose different operating points rather than collapsing into equivalent performance.[5] [6] [17] Third, benchmark interpretation must remain honest: the repository supports a strong architecture story, but different deployment goals may favor different model families.[17]

## Why UHCI Is Timely

The timing for UHCI is unusually strong because infrastructure, standards, and policy are converging. The joint 3GPP and O-RAN perspective on AI adoption argues that standardization is essential for industry alignment in 5G-Advanced and 6G, while O-RAN research reports frame native AI, distributed intelligence, unified data ingestion, and RIC-centered execution as core architectural themes.[25] [26] [27] NVIDIA’s AI-RAN framing reinforces the infrastructure side by highlighting a world in which AI and RAN workloads coexist on accelerated platforms.[31] Public-spectrum policy documents and NTIA’s AI-RAN-focused direction reinforce the operational pressure for more adaptive, programmable, and auditable wireless intelligence.[33] [34]

| Timing driver | External signal | Why it matters for this repository |
|---|---|---|
| AI-native RAN standardization | Joint 3GPP and O-RAN perspective on AI adoption [27] | UHCI already combines learning with programmable RAN control surfaces |
| Native and cross-domain AI in O-RAN | O-RAN nGRG reports [25] [26] | The repository includes data ingestion, lifecycle control, and distributed intelligence elements |
| AI-RAN infrastructure readiness | NVIDIA AI-RAN materials [31] | The repository includes serving, runtime, and accelerator-oriented adapters |
| NTN and heterogeneous connectivity pressure | 3GPP NTN and channel modeling anchors [22] [23] | UHCI models terrestrial and non-terrestrial providers together |
| Earth-space propagation realism | ITU-R propagation guidance [24] | The propagation subsystem is explicitly part of the architecture |
| Public-sector push for adaptive spectrum systems | National Spectrum R&D Plan and NTIA direction [33] [34] | Auditability and dynamic control become more strategically important |

## Repository Map

The repository is already organized like a multi-layer system. The redesigned documentation makes that structure easier to read at a glance.

| Path | Purpose |
|---|---|
| `src/preceptualai/core/` | Decision models, structural encoders, and temporal backends |
| `src/preceptualai/env/` | Provider ontology, propagation, environments, and data ingestion |
| `src/preceptualai/xapp/` | Inference runtime, serving, O-RAN adapters, and lifecycle services |
| `src/preceptualai/federated/` | Federated aggregation logic |
| `proto/` | Formal inference and federated-learning contracts |
| `scripts/` | Training entry points and orchestration scripts |
| `benchmarks/` | Evaluation harnesses and result artifacts |
| `docs/` | System-level documentation, deployment guides, and architecture references |
| `docs/assets/` | Architecture visuals and diagrams embedded in documentation |

## Recommended Reading Order

This README is now the top-level entry point, but the repository also contains deeper references for architecture, training, deployment, and API-level integration.

| If you want to understand... | Read next |
|---|---|
| The full architecture explanation in English | `docs/UHCI_ARCHITECTURE_REFERENCE_EN.md` |
| The full Chinese translation | `docs/UHCI_ARCHITECTURE_REFERENCE_ZH.md` |
| Repository internals directory by directory | `docs/REPOSITORY_MAP.md` |
| Training and evaluation details | `docs/TRAINING_AND_EVALUATION.md` |
| Deployment and xApp integration | `docs/DEPLOYMENT_AND_XAPP_GUIDE.md` |
| APIs and service contracts | `docs/API_AND_INTERFACES.md` |

## References

[1]: [Provider taxonomy in `src/preceptualai/env/provider_registry.py`](src/preceptualai/env/provider_registry.py)
[2]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](src/preceptualai/env/unified_connectivity_env.py)
[3]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](src/preceptualai/core/universal_spectrum_agent.py)
[4]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](src/preceptualai/core/hetero_gnn_encoder.py)
[5]: [LTC module in `src/preceptualai/core/ltc_cell.py`](src/preceptualai/core/ltc_cell.py)
[6]: [CfC temporal backend in `src/preceptualai/core/ltc_cell_cfc.py`](src/preceptualai/core/ltc_cell_cfc.py)
[7]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](src/preceptualai/env/itu_propagation.py)
[8]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](src/preceptualai/env/fr3_propagation.py)
[9]: [Unified real-data pipeline in `src/preceptualai/env/data_pipeline.py`](src/preceptualai/env/data_pipeline.py)
[10]: [Inference engine in `src/preceptualai/xapp/inference_engine.py`](src/preceptualai/xapp/inference_engine.py)
[11]: [RT-oriented dApp engine in `src/preceptualai/xapp/dapp_engine.py`](src/preceptualai/xapp/dapp_engine.py)
[12]: [gRPC serving module in `src/preceptualai/xapp/server.py`](src/preceptualai/xapp/server.py)
[13]: [O-RAN environment surface in `src/preceptualai/env/oran.py`](src/preceptualai/env/oran.py)
[14]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](src/preceptualai/xapp/e2_adapter.py)
[15]: [Non-RT RIC lifecycle service in `src/preceptualai/xapp/rapp_trainer.py`](src/preceptualai/xapp/rapp_trainer.py)
[16]: [Benchmark harness in `benchmarks/benchmark.py`](benchmarks/benchmark.py)
[17]: [Aggregated benchmark summary in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](benchmarks/results/benchmark_results_full/benchmark_summary.json)
[18]: [Training entry point in `scripts/train_uhci.py`](scripts/train_uhci.py)
[19]: [Federated aggregator in `src/preceptualai/federated/aggregator.py`](src/preceptualai/federated/aggregator.py)
[20]: [Federated-learning service contract in `proto/preceptualai_fl.proto`](proto/preceptualai_fl.proto)
[21]: [Inference service contract in `proto/preceptualai.proto`](proto/preceptualai.proto)
[22]: [3GPP TR 38.821, "Solutions for NR to support Non-Terrestrial Networks (NTN)"](https://www.3gpp.org/dynareport/38821.htm)
[23]: [3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to 100 GHz"](https://www.3gpp.org/dynareport/38901.htm)
[24]: [ITU-R P.618, "Propagation data and prediction methods required for the design of Earth-space telecommunication systems"](https://www.itu.int/rec/R-REC-P.618)
[25]: [O-RAN next Generation Research Group, "O-RAN Native AI Architecture Description," RR-2023-02](https://mediastorage.o-ran.org/ngrg-rr/nGRG-RR-2023-02-Native%20AI%20Architecture%20Description-v1.2.pdf)
[26]: [O-RAN next Generation Research Group, "Research Report on Native and Cross-domain AI: State of the art and future outlook," RR-2023-03](https://mediastorage.o-ran.org/ngrg-rr/nGRG-RR-2023-03-Research-Report-on-Native-and-Cross-domain-AI-v1_1.pdf)
[27]: [X. Lin, L. Kundu, C. Dick, and S. Velayutham, "Embracing AI in 5G-Advanced Towards 6G: A Joint 3GPP and O-RAN Perspective," arXiv:2209.04987](https://arxiv.org/abs/2209.04987)
[28]: [R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks," arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
[29]: [R. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray, M. Tschaikowski, G. Teschl, and D. Rus, "Closed-form Continuous-time Neural Models," Nature Machine Intelligence 4, 992--1003 (2022)](https://arxiv.org/abs/2106.13898)
[30]: [X. Wang, H. Ji, C. Shi, B. Wang, P. Cui, P. S. Yu, and Y. Ye, "Heterogeneous Graph Attention Network," arXiv:1903.07293](https://arxiv.org/abs/1903.07293)
[31]: [NVIDIA, "AI-RAN Solutions for 5G and 6G Cellular Networks"](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)
[32]: [O-RAN Software Community documentation](https://docs.o-ran-sc.org/en/latest/)
[33]: [NITRD, "National Spectrum Research and Development Plan 2024"](https://www.nitrd.gov/pubs/National-Spectrum-RD-Plan-2024.pdf)
[34]: [NTIA, "NTIA Seeks Feedback on New Direction for Innovation Fund That Focuses on AI-RAN"](https://www.ntia.gov/blog/2026/ntia-seeks-feedback-new-direction-innovation-fund-focuses-ai-ran)
