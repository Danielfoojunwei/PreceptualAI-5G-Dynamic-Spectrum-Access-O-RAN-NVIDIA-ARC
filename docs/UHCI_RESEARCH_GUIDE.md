# UHCI Research Guide

This guide explains the repository’s **Universal Heterogeneous Connectivity Intelligence (UHCI)** research surface as a complete technical architecture rather than as an optional appendix to a narrower benchmark workflow. A line-by-line reading of the code shows that UHCI is the repository’s broader system of record for reasoning across **heterogeneous providers, multiple propagation regimes, structured graph state, temporal dynamics, real-data pathways, and deployment-aware control loops**.[1] [2] [3] [4] [5] [6] [7] [8] [9]

The most important mindset shift is that UHCI is not just “more environments.” It is a different formulation of the problem itself. Instead of asking which channel to use inside one narrow simulation, UHCI asks a larger question: **how should an intelligent controller choose among heterogeneous connectivity options under changing physics, timing, provider constraints, and deployment conditions?**[1] [2] [3] [4]

| UHCI dimension | What it expands beyond | Why it matters |
|---|---|---|
| Provider scope | Single-stack or single-band reasoning | Future networks mix terrestrial and non-terrestrial systems |
| Physics realism | Simplified channel assumptions | Real decisions are constrained by propagation and attenuation |
| State structure | Flat observations | Connectivity decisions are relational and graph-structured |
| Temporal modeling | One fixed recurrent design | Mixed-timescale dynamics need adaptive sequence reasoning |
| Runtime surface | Offline research-only training | Telecom intelligence must also be deployable and governable |

## What UHCI Means in This Repository

In this repository, UHCI is the umbrella architecture that unifies the following layers. First, the code defines a **provider ontology** that captures distinct connectivity classes and their operating characteristics. Second, it defines **propagation modules** that encode telecom physics. Third, it defines a **unified environment** that turns these ingredients into a control problem. Fourth, it defines a **universal agent and representation stack** that can process heterogeneous state. Finally, it defines **runtime, serving, and governance surfaces** that make the resulting intelligence operationally relevant.[1] [2] [3] [4] [5] [6] [7] [8]

> UHCI is best understood as a **world model plus decision model plus runtime model** for heterogeneous connectivity intelligence.

| UHCI layer | Principal repository anchors |
|---|---|
| Provider ontology | `provider_registry.py` |
| Propagation and channel realism | `itu_propagation.py`, `fr3_propagation.py` |
| Unified control environment | `unified_connectivity_env.py` |
| Data ingestion and empirical feeds | `data_pipeline.py` |
| Universal decision stack | `universal_spectrum_agent.py`, `hetero_gnn_encoder.py`, `ltc_cell_cfc.py` |
| Training entry point | `train_uhci.py` |
| Deployment surfaces | `inference_engine.py`, `dapp_engine.py`, `server.py`, `oran.py`, `e2_adapter.py`, `rapp_trainer.py` |

## Provider Taxonomy and Heterogeneous Connectivity Scope

The clearest starting point for understanding UHCI is `src/preceptualai/env/provider_registry.py`.[1] This file defines the system’s provider taxonomy and encodes physical and operational parameters that distinguish one class of connectivity from another. The result is a structured provider universe rather than a generic action list.

The code indicates that UHCI is meant to reason across a mix of terrestrial and non-terrestrial options, including classes such as **LEO, MEO, GEO, HAPS, FR1, FR3, ISAC, and WiFi 7**.[1] That matters because each of these classes has different latency, coverage, handover, Doppler, and bandwidth behavior. A controller that ignores those differences is not really solving the heterogeneous-connectivity problem.

| Provider class | Broad interpretation inside UHCI |
|---|---|
| LEO / MEO / GEO | Satellite connectivity classes with different orbital and latency properties |
| HAPS | High-altitude platform systems bridging terrestrial and NTN characteristics |
| FR1 / FR3 | Terrestrial cellular regimes with different propagation and coexistence behavior |
| ISAC | Integrated sensing and communication surface relevant to advanced 6G workflows |
| WiFi 7 | High-performance local-area access option inside the unified control space |

## Provider Physics as Research Prior

UHCI does not just label providers categorically. The registry also encodes a **physics prior** through parameters such as altitude, coverage radius, supported bands, bandwidth, EIRP, path loss, SNR assumptions, noise figure, latency range, Doppler, handover period, and channel decorrelation timescale.[1]

This is a major design choice. It means the research stack is not asking the model to infer every property from scratch. Instead, it gives the environment enough structured prior information to make the control problem physically meaningful.

| Encoded provider attribute | Why it matters scientifically |
|---|---|
| Altitude and coverage | Determines geometry, reach, and likely path-loss behavior |
| Latency range | Shapes suitability for delay-sensitive traffic |
| Doppler and decorrelation time | Directly affects temporal stability and memory requirements |
| Band support and bandwidth | Constrains capacity and coexistence possibilities |
| Handover period | Changes the effective control horizon |

## Physics Layer: ITU and FR3 Propagation

Two important modules show that UHCI is grounded in wireless physics rather than only abstract reward shaping. `src/preceptualai/env/itu_propagation.py` implements ITU-aligned propagation logic, while `src/preceptualai/env/fr3_propagation.py` handles FR3-specific path-loss and coexistence considerations.[2] [3]

These files matter for both lay and expert readers. For a lay reader, they mean the system does not pretend the radio channel is magic. For a researcher, they mean the environment embeds domain priors that can make learned behavior more credible than purely toy abstractions.

| Propagation module | Research contribution |
|---|---|
| `itu_propagation.py` | Introduces standards-aligned attenuation and NTN-relevant propagation realism |
| `fr3_propagation.py` | Captures FR3-specific channel behavior and coexistence constraints |

A useful way to think about this layer is that it defines the **physical grammar** of the UHCI world. The agent can only be intelligent if the environment reflects the channel constraints that real systems face.[2] [3]

## Unified Connectivity Environment

`src/preceptualai/env/unified_connectivity_env.py` is the file that turns provider modeling and physics into a real decision environment.[4] It constructs the observation space, state transitions, provider interactions, step logic, and reward composition for the larger heterogeneous-connectivity problem.

The practical significance of this file is that it moves the repository from a narrow dynamic-spectrum benchmark toward a more general **connectivity-control simulator and experimentation surface**. This environment is where the repository most clearly declares that the unit of reasoning is no longer only a channel, but a structured multi-provider connectivity context.[4]

| Environment function | Why it matters |
|---|---|
| Observation construction | Makes heterogeneous provider state visible to the model |
| Action semantics | Defines how connectivity choices are expressed |
| Reward composition | Encodes trade-offs among success, quality, and system objectives |
| Provider interaction logic | Lets the controller reason over a structured connectivity ecosystem |

## Real-Data and Empirical Data Pathways

The UHCI stack is not limited to synthetic simulation. `src/preceptualai/env/data_pipeline.py` implements a broader data-ingestion surface and references richer telecom data flows, including UCC MISL and Colosseum-style sources.[5] This means the architecture is designed to move beyond toy environments and engage with more realistic empirical inputs.

The correct documentation posture is careful. The code strongly supports the claim that **the repository implements real-data pathways**. It does not automatically mean that every user can reproduce every data-backed workflow from a clean machine without obtaining external datasets and preparing the right environment.[5]

| Data-pathway statement | Safe interpretation |
|---|---|
| The repository supports richer telecom data ingestion | Yes |
| UHCI is designed to go beyond pure simulation | Yes |
| All real-data experiments are turnkey for every user | No |

## Universal Agent and Structured Intelligence

`src/preceptualai/core/universal_spectrum_agent.py` is the clearest sign that the repository’s modeling center of gravity has shifted toward UHCI.[6] The file defines a broader decision-making abstraction than a narrow benchmarked agent path. It is designed to sit on top of heterogeneous representations, temporal modules, and optional supporting components, making it a more suitable conceptual center for the repository’s current architecture.

This matters because it tells the reader how to interpret the rest of the code. The repository is not just a single policy implementation with extra utilities. It is moving toward a **universal control agent** for connectivity intelligence.[6]

## Graph and Relational Representation

UHCI requires more than a flat feature vector because the environment contains multiple entities and relations. `src/preceptualai/core/hetero_gnn_encoder.py` addresses this by encoding **heterogeneous graph-structured state**.[7] This is a meaningful design choice because provider-aware reasoning is naturally relational. Different nodes, links, provider classes, and contextual features interact; they are not merely independent scalar inputs.

| Why graph structure matters | Consequence for UHCI |
|---|---|
| Providers differ systematically | The model should preserve entity identity |
| Connectivity options interact | The encoder should reason about relations, not just isolated values |
| System state is structured | A graph encoder is often a more natural representation than a flat MLP |

## Temporal Backends: CfC, LTC, and Beyond

Temporal reasoning is another place where UHCI broadens the repository’s identity. `src/preceptualai/core/ltc_cell_cfc.py` exposes the temporal backend surface and includes **CfC**, **LTC**, and related sequence-modeling pathways.[8] This is architecturally important because it shows the repository is not committed to one recurrent block as the only possible temporal mechanism.

The right research interpretation is that UHCI requires **pluggable temporal intelligence**. Some conditions may be better handled by continuous-time memory, some by closed-form continuous-time updates, and some by richer sequence-modeling backends. The code reflects that flexibility.[7] [8]

| Temporal backend idea | Research meaning |
|---|---|
| LTC | Adaptive continuous-time memory used in the included benchmark suite and still useful in UHCI |
| CfC | Alternative continuous-time backend that broadens the temporal surface |
| Other temporal pathways in the encoder stack | Support experimentation with broader sequence reasoning under one architecture |

## Training Surface and Experimental Control

`scripts/train_uhci.py` is the main executable entry point for UHCI experiments.[9] It exposes flags and options that make clear the repository is supporting more than a toy benchmark. The training path includes support for richer experimental settings, real-data-related flags, TelecomTS-oriented workflows, and smoothing or domain-specific controls that do not belong to a narrow benchmark workflow.[9]

| UHCI experimental artifact | Role |
|---|---|
| `train_uhci.py` | Main CLI surface for UHCI research runs |
| `unified_connectivity_env.py` | Supplies the broader decision world |
| `provider_registry.py` | Supplies provider ontology and physics priors |
| `data_pipeline.py` | Supplies external-data and richer empirical pathways |

## Runtime, O-RAN, and Lifecycle Surfaces

One of the most important findings from the code audit is that UHCI extends into runtime and deployment surfaces. `src/preceptualai/xapp/inference_engine.py`, `src/preceptualai/xapp/dapp_engine.py`, and `src/preceptualai/xapp/server.py` provide runtime, low-latency, and serving layers for the model.[10] [11] [12] `src/preceptualai/env/oran.py` and `src/preceptualai/xapp/e2_adapter.py` link the intelligence stack to O-RAN-facing control logic.[13] [14] `src/preceptualai/xapp/rapp_trainer.py` adds longer-timescale governance, model approval, degradation monitoring, and policy generation.[15]

This means UHCI is not just a research environment. It is a research environment that is already being designed with deployment, observability, and lifecycle concerns in mind.

| Runtime layer | What it contributes to UHCI |
|---|---|
| `inference_engine.py` | Common runtime execution abstraction |
| `dapp_engine.py` | Low-latency inference path aligned with RT-RIC-style execution |
| `server.py` | gRPC prediction, health, and metrics service surface |
| `oran.py` | O-RAN-oriented environment and loop semantics |
| `e2_adapter.py` | E2-facing measurement and control integration |
| `rapp_trainer.py` | Non-RT RIC governance, approval, and lifecycle management |

## What UHCI Is Not

UHCI should not be described carelessly. The repository does **not** prove that every UHCI subsystem has already been benchmarked end to end with the same maturity as the currently included benchmark suite. The code strongly supports the claim that the repository contains a serious heterogeneous-connectivity research and deployment scaffold. It does **not** by itself justify saying that the entire UHCI stack has already achieved literature-wide state of the art across every benchmark and deployment setting.[4] [5] [9] [16] [17]

| Claim type | Safe? |
|---|---|
| The repository implements a serious UHCI architecture | Yes |
| UHCI includes provider, physics, graph, temporal, runtime, and governance layers | Yes |
| The full UHCI stack is uniformly benchmarked end to end like the current benchmark suite | No |
| The repository proves literature-wide SOTA for every UHCI surface | No |

## How to Approach UHCI Responsibly

A new reader should approach UHCI as the repository’s main architectural direction, but should do so methodically. Start by understanding the provider registry and unified environment, because those files define the world that the rest of the system reasons over. Then inspect the universal agent and graph or temporal modules. After that, move into the training entry point and finally into the runtime and O-RAN-facing surfaces.[1] [4] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15]

| Step | Recommended reading and action |
|---|---|
| 1 | Read `provider_registry.py` to understand the connectivity ontology |
| 2 | Read `itu_propagation.py` and `fr3_propagation.py` to understand the physics layer |
| 3 | Read `unified_connectivity_env.py` to understand state, action, and reward design |
| 4 | Read `universal_spectrum_agent.py`, `hetero_gnn_encoder.py`, and `ltc_cell_cfc.py` |
| 5 | Inspect `train_uhci.py --help` and design a controlled experiment |
| 6 | Explore `data_pipeline.py` if you need richer empirical input |
| 7 | Move into runtime and O-RAN-facing modules if deployment is relevant |

## Recommended Research Questions

The repository naturally supports several high-value research questions. These include how graph structure improves provider-aware decision quality, how continuous-time temporal backends compare under heterogeneous dynamics, how physics-aware priors affect learning stability, and how deployment constraints shape the choice of architecture.

| Research question | Why the repository is suitable |
|---|---|
| Does provider-aware graph structure improve control quality? | The environment and encoder explicitly support relational state |
| How do CfC and LTC compare under mixed-timescale dynamics? | The temporal backend surface is already implemented |
| How much does explicit propagation realism change policy behavior? | ITU and FR3 propagation modules are present |
| Can heterogeneous-control models be moved toward O-RAN deployment? | Runtime, E2, and lifecycle modules are already included |

## References

[1]: [Provider taxonomy in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[2]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[3]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[4]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[5]: [Unified data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[6]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[7]: [Heterogeneous graph encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[8]: [CfC and temporal backend implementation in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[9]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[10]: [Runtime inference abstraction in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[11]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[12]: [gRPC serving module in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[13]: [O-RAN environment module in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[14]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[15]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[16]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[17]: [Aggregated benchmark artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
