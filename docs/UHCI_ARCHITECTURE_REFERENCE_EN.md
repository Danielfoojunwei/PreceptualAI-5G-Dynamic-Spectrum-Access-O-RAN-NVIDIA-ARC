# PreceptualAI UHCI Architecture Reference

**Author:** Manus AI  
**Repository:** `Danielfoojunwei/PreceptualAI-Universal-Heterogeneous-Connectivity-Intelligence-UHCI-`

## Introduction

This document provides a full architectural interpretation of **PreceptualAI Universal Heterogeneous Connectivity Intelligence (UHCI)**. It is written as a system-level reference for readers who need to understand not only what the repository contains, but also **how the major subsystems fit together, why the chosen technologies are technically appropriate, which standards and research references support the design, and how a complete end-to-end UHCI stack can be reconstructed from the codebase**.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16]

The core conclusion of this architectural review is that the repository is best understood not as a narrow reinforcement-learning project, but as a **multi-layer connectivity-intelligence platform**. It combines a provider ontology, telecom-physics-aware propagation, a unified control environment, a heterogeneous graph representation layer, continuous-time temporal intelligence, a universal decision policy, low-latency runtime interfaces, O-RAN integration surfaces, non-real-time lifecycle governance, and optional federated learning into a coherent system architecture.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17] [18] [19] [20]

| Architectural question | Short answer |
|---|---|
| What problem does UHCI solve? | It creates a single intelligence layer for heterogeneous terrestrial, non-terrestrial, and hybrid connectivity decisions. |
| Why is this different from a narrow RL benchmark? | The repository includes world modeling, propagation, structural representation, serving, telecom control-plane integration, governance, and distributed-learning interfaces. |
| Why does the architecture matter now? | AI-native RAN, O-RAN programmability, AI-RAN infrastructure, and heterogeneous connectivity pressure are converging in 5G-Advanced and 6G workflows.[21] [22] [23] [24] [25] [26] [27] |
| Why is the stack defensible? | It aligns modeling choices with the structure of the wireless problem instead of forcing everything into a flat environment or offline training loop. |

## System Goal

The architectural goal of UHCI is to enable **decision-making across multiple connectivity domains that differ in physics, provider identity, and control semantics**, while still exposing an execution model that can fit near-real-time inference, non-real-time lifecycle management, and distributed or federated deployment settings.[1] [2] [3] [10] [11] [12] [14] [15] [19] [20]

This matters because modern wireless deployments no longer operate within one homogeneous resource pool. Real systems increasingly combine terrestrial radio, non-terrestrial platforms, edge resources, and software-defined control surfaces. As a result, a controller must reason over a broader state space that includes latency regimes, propagation conditions, coverage geometry, coexistence behavior, and orchestration constraints rather than simple instantaneous channel occupancy only.[1] [7] [8] [21] [22] [23] [24]

> “Native Artificial Intelligence (AI) is the enabler technology for 6G. The RAN Intelligence Controller (RIC) of O-RAN is the potential approach for native AI.” — *O-RAN next Generation Research Group, RR-2023-02* [25]

> “A unified data ingestion model is emerging as a key requirement.” — *O-RAN next Generation Research Group, RR-2023-03* [26]

These statements are particularly relevant because the repository contains both a **unified data pathway** and a **RIC-oriented execution and lifecycle story**, which means the implementation direction is aligned with the broader shift toward AI-native wireless control.[9] [13] [14] [15] [25] [26] [27]

## Overall Architecture

The repository organizes UHCI as a layered system. The first diagram below shows the complete architecture from inputs and domain priors to runtime and lifecycle control.

![UHCI Overall Architecture](assets/uhci_overall_architecture.png)

The architecture is not a monolithic model. It is a **pipeline of cooperating layers** in which each layer serves a distinct systems purpose. The provider and physics layers define what the world looks like. The environment and data layers define how the world is transformed into a control problem. The graph and temporal layers define how complex state is represented. The policy layer defines how actions are chosen. The runtime and governance layers define how those actions are served, integrated, monitored, and improved over time.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [18] [19] [20]

| Layer | Core function | Main code anchors | Architectural meaning |
|---|---|---|---|
| Provider ontology | Define heterogeneous provider classes and priors | `provider_registry.py` [1] | Establishes the design space of connectivity choices |
| Telecom physics | Model terrestrial and NTN propagation | `itu_propagation.py`, `fr3_propagation.py` [7] [8] | Grounds the control problem in realistic wireless behavior |
| Unified environment | Build observations, actions, rewards, and transitions | `unified_connectivity_env.py` [2] | Converts the world into a learnable decision process |
| Data ingestion | Normalize and route richer operational inputs | `data_pipeline.py` [9] | Connects the architecture to empirical telecom data |
| Structural representation | Encode typed nodes, links, and relations | `hetero_gnn_encoder.py` [4] | Preserves system structure rather than flattening it away |
| Temporal intelligence | Model irregular, mixed-timescale state evolution | `ltc_cell.py`, `ltc_cell_cfc.py` [5] [6] | Gives the policy adaptive memory over time |
| Decision policy | Produce learned control actions | `universal_spectrum_agent.py` [3] | Implements the central UHCI controller |
| Runtime and serving | Execute and expose low-latency inference | `inference_engine.py`, `dapp_engine.py`, `server.py` [10] [11] [12] | Makes the intelligence callable in operational settings |
| O-RAN control interface | Bridge model outputs to RAN-facing semantics | `oran.py`, `e2_adapter.py` [13] [14] | Connects learning to programmable control loops |
| Lifecycle governance | Manage approval, monitoring, retraining, and rollout | `rapp_trainer.py` [15] | Treats model management as part of the system |
| Federated extension | Aggregate multi-site learning updates | `aggregator.py`, `preceptualai_fl.proto` [19] [20] | Extends UHCI across distributed deployments |

## Subsystem-by-Subsystem Explanation

### Provider Ontology and Heterogeneous World Modeling

The first architectural subsystem is the **provider ontology**, centered on `provider_registry.py`. This module is important because it determines **what the controller is allowed to reason about**. Rather than modeling the world as one abstract pool of interchangeable channels, the repository formalizes categories such as **LEO, MEO, GEO, HAPS, FR1, FR3, ISAC, and Wi-Fi 7**, together with attributes such as latency tendencies, coverage, handover assumptions, bandwidth, and deployment priors.[1]

This design choice changes the problem formulation significantly. If the world is heterogeneous, then the controller must preserve the identity and semantics of different connectivity options. The provider layer therefore acts as the architectural boundary between raw wireless diversity and the rest of the learning stack.

| Provider subsystem contribution | Why it matters |
|---|---|
| Encodes provider categories explicitly | Prevents all links from being treated as equivalent |
| Associates structured priors with each provider type | Gives the rest of the stack access to semantically meaningful constraints |
| Supports terrestrial and non-terrestrial reasoning in one abstraction | Enables a broader connectivity intelligence story than classical DSA alone |

### Telecom Physics and Propagation Modeling

The second subsystem is the **propagation layer**, which includes `itu_propagation.py` and `fr3_propagation.py`.[7] [8] These modules matter because real network decisions depend on physical behavior, not on reward functions alone. The repository’s propagation layer ties the control problem to telecom modeling assumptions related to terrestrial and Earth-space communication.

This architecture is consistent with the standards context implied by the repository. The non-terrestrial side maps naturally to **3GPP TR 38.821** on solutions for NR support of non-terrestrial networks and **ITU-R P.618** on propagation data and prediction methods for Earth-space telecommunication systems.[21] [23] The terrestrial side maps naturally to **3GPP TR 38.901**, which studies channel models from 0.5 to 100 GHz.[22]

| Propagation reference | Architectural relevance |
|---|---|
| 3GPP TR 38.821 [21] | Provides a standards anchor for non-terrestrial network assumptions and heterogeneous NTN provider framing |
| 3GPP TR 38.901 [22] | Provides a standards anchor for terrestrial and broad-band channel-model reasoning |
| ITU-R P.618 [23] | Provides a standards anchor for Earth-space propagation modeling |

### Unified Environment and Data Pipeline

The third subsystem is the **unified environment layer**, centered on `unified_connectivity_env.py` and extended by `data_pipeline.py` and `oran.py`.[2] [9] [13] Here, the repository converts a heterogeneous wireless world into a formal control process. Observations, action spaces, rewards, and transitions are assembled from the provider and propagation logic so that the intelligence layer operates on a coherent decision surface.

The data pipeline is especially important because O-RAN’s cross-domain AI report emphasizes that next-generation wireless systems must deal with **large amounts of disparate data across multiple layers** and that a **unified data ingestion model** is becoming a key architectural requirement.[26] The presence of a dedicated `data_pipeline.py` strongly supports the interpretation that the repository is designed for richer real-data workflows rather than synthetic simulation only.[9] [26]

| Environment-layer file | Main architectural role |
|---|---|
| `unified_connectivity_env.py` | Defines the core learning environment and operational control semantics |
| `data_pipeline.py` | Ingests and normalizes data for more realistic operating scenarios |
| `oran.py` | Extends the environment into O-RAN-facing control semantics |

### Heterogeneous Graph Representation

The fourth subsystem is the **structural representation layer**, centered on `hetero_gnn_encoder.py`.[4] This subsystem is essential because UHCI’s target problem is relational. Providers, links, contexts, and constraints are not isolated scalar values. They exist in a typed system of entities and interactions.

The architectural choice is well supported by research on **Heterogeneous Graph Attention Networks (HAN)**, which explicitly addresses graphs containing multiple node and link types and introduces hierarchical attention across both neighbors and meta-path semantics.[30] That research is directly relevant to UHCI because the repository’s problem domain naturally includes typed providers, typed relationships, and multiple modes of interaction.

> HAN introduces a “novel heterogeneous graph neural network based on hierarchical attention, including node-level and semantic-level attentions.” — *Wang et al.* [30]

This is exactly the kind of inductive bias that makes sense when the problem is not homogeneous channel selection, but heterogeneous connectivity reasoning.

### Continuous-Time Temporal Intelligence

The fifth subsystem is the **temporal intelligence layer**, implemented through `ltc_cell.py` and `ltc_cell_cfc.py`.[5] [6] This subsystem is necessary because wireless control unfolds over mixed and irregular timescales. Some conditions shift rapidly, while others persist over longer intervals. A system that assumes one fixed timescale risks losing important temporal structure.

The design is strongly supported by two research threads. **Liquid Time-constant Networks (LTCs)** introduce time-continuous recurrent models with liquid time constants and stable bounded dynamics.[28] **Closed-form Continuous-time Neural Models** show how liquid dynamics can be approximated in closed form, reducing reliance on numerical differential-equation solvers and enabling efficient continuous-time sequence modeling.[29]

| Temporal research anchor | Relevance to UHCI |
|---|---|
| Liquid Time-constant Networks [28] | Supports adaptive continuous-time memory with bounded dynamics |
| Closed-form Continuous-time Neural Models [29] | Supports efficient CfC-style sequence modeling for irregular temporal data |

Because the repository includes a CfC-capable temporal backend, the architecture can be interpreted as explicitly trying to balance temporal expressiveness with runtime practicality.[6] [29]

### Universal Decision Layer

The sixth subsystem is the **decision layer**, centered on `universal_spectrum_agent.py`.[3] This file signals that the repository’s main controller is not limited to a single benchmark formulation. The naming and surrounding architecture imply a **universal agent** for a broader heterogeneous connectivity problem.

The decision layer is therefore best interpreted as the point where structural and temporal intelligence are fused into actions over the heterogeneous resource space. It stands on top of the environment, graph encoder, and temporal backends rather than replacing them.

### Runtime, Serving, and Low-Latency Execution

The seventh subsystem is the **runtime and serving layer**, centered on `inference_engine.py`, `dapp_engine.py`, and `server.py`.[10] [11] [12] This layer is crucial because it converts a trained policy into an operational service. Without this layer, the repository would still be valuable for research, but not for telecom deployment.

The `dapp_engine.py` module is especially important because it provides a low-latency execution surface. This aligns with the near-real-time side of programmable telecom systems, in which decisions must be served quickly enough to matter operationally.[11] [25] [26] Meanwhile, the gRPC server and protocol definitions show that the repository is not limited to internal Python calls; it defines a formal service boundary for external consumption.[12] [18]

### O-RAN Integration and Lifecycle Governance

The eighth subsystem is the **O-RAN and lifecycle layer**, centered on `oran.py`, `e2_adapter.py`, and `rapp_trainer.py`.[13] [14] [15] This layer is what connects the repository to the broader conversation about programmable, AI-native radio access networks.

The O-RAN research reports are especially relevant here. The O-RAN native AI architecture document identifies the **RIC as a potential approach for native AI**, while the cross-domain AI report emphasizes the importance of distributed intelligence, unified data ingestion, AI lifecycle management, and collaboration across disaggregated RAN and between RAN and core domains.[25] [26] The joint 3GPP and O-RAN perspective similarly frames standardization and AI-enabled traffic steering as central to the 5G-Advanced to 6G path.[27]

Taken together, those references make the repository’s O-RAN-facing surfaces more meaningful. They are not cosmetic integrations. They are part of a plausible architecture for placing learned connectivity intelligence into programmable RAN control workflows.

### Federated Learning and Distributed Updates

The ninth subsystem is the **federated extension**, centered on `src/preceptualai/federated/aggregator.py` and `proto/preceptualai_fl.proto`.[19] [20] This subsystem indicates that UHCI is not limited to centralized training only. It can also support a distributed model lifecycle in which local clients or sites contribute updates to a broader global model.

This is architecturally significant because distributed wireless systems often exhibit site heterogeneity, privacy constraints, and localized observations. The federated subsystem therefore extends the lifecycle story from centralized governance to multi-site learning.

## How the Subsystems Work Together

The second diagram below focuses on the interaction between online control, O-RAN interfaces, offline learning, and optional federated updates.

![UHCI Control and Lifecycle](assets/uhci_control_lifecycle.png)

The end-to-end interaction can be described as a sequence of transformations.

First, provider and propagation modules define the physical and operational structure of the world. Second, the unified environment and data pipeline turn that world into a stream of observations and normalized inputs. Third, the graph and temporal modules encode that state while preserving structure and temporal context. Fourth, the universal decision layer produces actions. Fifth, runtime and serving modules expose those actions to low-latency systems and external interfaces. Sixth, O-RAN-facing components and lifecycle services govern where and how those actions are deployed and updated. Seventh, optional federated workflows allow additional training signals to arrive from distributed clients.[1] [2] [3] [4] [5] [6] [9] [10] [11] [12] [13] [14] [15] [19] [20]

| Step | Input | Transformation | Output |
|---|---|---|---|
| 1 | Provider classes and standards priors | Ontology and propagation modeling | Structured wireless world |
| 2 | Wireless world plus measurements | Environment construction and data normalization | Decision-ready observations |
| 3 | Structured observations | Graph encoding and temporal memory | Latent system representation |
| 4 | Latent representation | Universal policy inference | Connectivity or spectrum actions |
| 5 | Actions | Runtime serving and integration | Callable low-latency control outputs |
| 6 | Runtime outputs and monitoring | Lifecycle governance and O-RAN coupling | Approved, monitored, and updateable deployments |
| 7 | Multi-site model feedback | Federated aggregation | Improved global or personalized models |

## Why the Technology Choices Are Appropriate

The repository’s main strength is not that it uses fashionable components, but that **its components match the problem structure**.

The provider registry is appropriate because the wireless world is heterogeneous. The propagation layer is appropriate because connectivity choices are shaped by physics. The graph encoder is appropriate because the world contains typed entities and relationships. Continuous-time temporal models are appropriate because network conditions evolve over irregular timescales. Runtime and O-RAN layers are appropriate because useful telecom intelligence must eventually exist inside programmable operational loops rather than only inside offline training notebooks.[1] [2] [4] [5] [6] [7] [8] [10] [11] [12] [13] [14] [15] [25] [26] [27]

| Design choice | Problem property it matches |
|---|---|
| Provider ontology | Heterogeneous resource classes |
| Propagation modules | Physics-constrained wireless behavior |
| Heterogeneous GNN | Typed graph-structured state |
| LTC and CfC temporal backends | Irregular, mixed-timescale dynamics |
| Universal agent | Broad, multi-domain action space |
| Low-latency serving | Near-real-time operational requirements |
| O-RAN-facing modules | Programmable RAN control loops |
| Federated extension | Distributed, site-specific learning settings |

## Benchmarks and Evidence

The repository contains benchmark outputs that are useful for grounding claims in measured evidence rather than architecture aspiration alone.[16] [17] The benchmark artifact `benchmark_summary.json` shows that different temporal and policy variants occupy different positions in the trade-space across reward, success rate, fairness, and latency.[17]

In particular, the strongest performance on **success rate**, **collision rate**, and **spectral efficiency** comes from `sac_ltc`, which reports **0.6337 ± 0.0044** success rate, **0.3663 ± 0.0044** collision rate, and **0.6337 ± 0.0044** spectral efficiency.[17] The strongest performance on **mean reward** and **inference latency** comes from `sac_lstm`, which reports **50.0267 ± 1.8831** mean reward and **0.8276 ± 0.0200 ms** mean inference latency.[17] The strongest **Jain fairness** score comes from `sac_lfm`, which reports **0.99636 ± 0.00008**.[17]

| Model | Key strength | Important interpretation |
|---|---|---|
| `sac_ltc` | Best operational success, lowest collision, strongest spectral efficiency | Supports the value of continuous-time temporal intelligence in the benchmark suite |
| `sac_lstm` | Best reward and best latency | Shows that some deployment settings may prioritize faster recurrent execution |
| `sac_lfm` | Best fairness | Suggests fairness objectives may favor a different temporal or memory profile |
| `ppo_lstm` | Strong competitive baseline | Helps establish that the repository’s evaluation is multi-model rather than single-model |

These results support a nuanced reading of the repository. The architecture already has measured evidence of low-latency feasibility and differentiated model behavior, but it should not be documented as though one single model dominates every metric. The right conclusion is that **UHCI provides a strong platform for exploring the trade-space between operational performance, fairness, and runtime efficiency**.[17]

## Why the Architecture Matters Now

The timing for UHCI is strong because multiple external trends are converging. The joint 3GPP and O-RAN perspective on AI adoption makes clear that standardization is central to industry alignment for AI in 5G-Advanced and 6G.[27] O-RAN research reports frame the RIC, unified data ingestion, distributed intelligence, lifecycle management, and collaboration across domains as central future architecture themes.[25] [26] NVIDIA’s AI-RAN framing shows that accelerated infrastructure is becoming a realistic substrate for co-located AI and RAN workloads.[31] Meanwhile, public-sector research and policy documents show continued pressure toward more adaptive and innovation-friendly spectrum systems.[33] [34]

| Trend | Why it strengthens the UHCI story |
|---|---|
| AI-native RAN and 6G research | UHCI already adopts an architecture that mixes learning, lifecycle, and telecom control surfaces |
| O-RAN programmability | UHCI includes near-real-time and non-real-time integration points |
| AI-RAN infrastructure | UHCI includes runtime and accelerator-oriented deployment pathways |
| NTN growth | UHCI explicitly includes non-terrestrial provider classes and propagation logic |
| Auditability requirements | UHCI includes benchmarks, protocols, and explicit documentation surfaces |

## Reconstructing the Repository as a Complete System

A reader who wants to rebuild the full system should think of the repository in stages. The first stage is to reconstruct the **provider, propagation, and environment worldview**. The second stage is to reconstruct the **graph and temporal representation stack**. The third stage is to reconstruct the **decision and training pipeline**. The fourth stage is to reconstruct the **runtime, serving, and O-RAN integration surfaces**. The fifth stage is to reconstruct the **lifecycle and federated-learning extensions**.[1] [2] [3] [4] [5] [6] [9] [10] [11] [12] [13] [14] [15] [18] [19] [20]

| Rebuild stage | Files to inspect first |
|---|---|
| Wireless world definition | `provider_registry.py`, `itu_propagation.py`, `fr3_propagation.py` |
| Control environment | `unified_connectivity_env.py`, `data_pipeline.py`, `oran.py` |
| Representation learning | `hetero_gnn_encoder.py`, `ltc_cell.py`, `ltc_cell_cfc.py` |
| Policy learning | `universal_spectrum_agent.py`, `scripts/train_uhci.py` |
| Runtime and interfaces | `inference_engine.py`, `dapp_engine.py`, `server.py`, `proto/preceptualai.proto` |
| O-RAN and lifecycle | `e2_adapter.py`, `rapp_trainer.py` |
| Distributed learning | `aggregator.py`, `proto/preceptualai_fl.proto` |
| Evaluation | `benchmarks/benchmark.py`, benchmark results |

## Conclusion

The most important architectural conclusion is that **UHCI is a systems architecture, not just a model**. Its main value lies in the way it combines heterogeneous provider reasoning, propagation realism, structured representation learning, continuous-time temporal intelligence, deployable runtime interfaces, O-RAN integration, and lifecycle governance into one connected design.[1] [2] [3] [4] [5] [6] [7] [8] [10] [11] [12] [13] [14] [15]

This is precisely why the repository deserves documentation that emphasizes **architecture, subsystem relationships, technology fit, evidence, and standards grounding** rather than only training commands. When interpreted in that way, the repository becomes much easier to understand, evaluate, and extend.

## References

[1]: [Provider taxonomy in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[2]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[3]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[4]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[5]: [LTC module in `src/preceptualai/core/ltc_cell.py`](../src/preceptualai/core/ltc_cell.py)
[6]: [CfC temporal backend in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[7]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[8]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[9]: [Unified real-data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[10]: [Inference engine in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[11]: [RT-oriented dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[12]: [gRPC serving module in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[13]: [O-RAN environment surface in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[14]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[15]: [Non-RT RIC lifecycle service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[16]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[17]: [Aggregated benchmark summary in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
[18]: [Inference service contract in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[19]: [Federated aggregator in `src/preceptualai/federated/aggregator.py`](../src/preceptualai/federated/aggregator.py)
[20]: [Federated-learning service contract in `proto/preceptualai_fl.proto`](../proto/preceptualai_fl.proto)
[21]: [3GPP TR 38.821, "Solutions for NR to support Non-Terrestrial Networks (NTN)"](https://www.3gpp.org/dynareport/38821.htm)
[22]: [3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to 100 GHz"](https://www.3gpp.org/dynareport/38901.htm)
[23]: [ITU-R P.618, "Propagation data and prediction methods required for the design of Earth-space telecommunication systems"](https://www.itu.int/rec/R-REC-P.618)
[24]: [O-RAN Software Community documentation](https://docs.o-ran-sc.org/en/latest/)
[25]: [O-RAN next Generation Research Group, "O-RAN Native AI Architecture Description," RR-2023-02](https://mediastorage.o-ran.org/ngrg-rr/nGRG-RR-2023-02-Native%20AI%20Architecture%20Description-v1.2.pdf)
[26]: [O-RAN next Generation Research Group, "Research Report on Native and Cross-domain AI: State of the art and future outlook," RR-2023-03](https://mediastorage.o-ran.org/ngrg-rr/nGRG-RR-2023-03-Research-Report-on-Native-and-Cross-domain-AI-v1_1.pdf)
[27]: [X. Lin, L. Kundu, C. Dick, and S. Velayutham, "Embracing AI in 5G-Advanced Towards 6G: A Joint 3GPP and O-RAN Perspective," arXiv:2209.04987](https://arxiv.org/abs/2209.04987)
[28]: [R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks," arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
[29]: [R. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray, M. Tschaikowski, G. Teschl, and D. Rus, "Closed-form Continuous-time Neural Models," Nature Machine Intelligence 4, 992--1003 (2022)](https://arxiv.org/abs/2106.13898)
[30]: [X. Wang, H. Ji, C. Shi, B. Wang, P. Cui, P. S. Yu, and Y. Ye, "Heterogeneous Graph Attention Network," arXiv:1903.07293](https://arxiv.org/abs/1903.07293)
[31]: [NVIDIA, "AI-RAN Solutions for 5G and 6G Cellular Networks"](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)
[32]: [NITRD, "National Spectrum Research and Development Plan 2024"](https://www.nitrd.gov/pubs/National-Spectrum-RD-Plan-2024.pdf)
[33]: [NTIA, "NTIA Seeks Feedback on New Direction for Innovation Fund That Focuses on AI-RAN"](https://www.ntia.gov/blog/2026/ntia-seeks-feedback-new-direction-innovation-fund-focuses-ai-ran)
