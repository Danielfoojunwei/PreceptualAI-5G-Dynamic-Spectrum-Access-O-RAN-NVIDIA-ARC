# Data and Reproducibility

This document explains **what can be reproduced directly from the repository, what depends on optional data or infrastructure, and how performance claims should be bounded** when the project is read correctly as a **UHCI-first system**. The central point is that the repository implements a broad **Universal Heterogeneous Connectivity Intelligence (UHCI)** architecture spanning provider modeling, propagation-aware simulation, heterogeneous state construction, structural and temporal learning, deployment-facing inference, and telecom control-loop integration.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

A careful reader should distinguish among three kinds of evidence. The first is **implementation evidence**, which comes from source code, protocols, configuration files, and deployment modules. The second is **benchmark evidence**, which comes from the included benchmark harness and saved summary artifacts. The third is **contextual evidence**, which explains why the architecture matters strategically but does not by itself prove measured repository performance.[1] [2] [13] [14]

| Evidence type | What it proves well | What it does not prove by itself |
|---|---|---|
| Source code | What UHCI implements architecturally | That every subsystem has already been benchmarked equally |
| Configuration files | What workflows, assumptions, and runtime surfaces the repository expects | That all external environments are available to every reader |
| Benchmark artifacts | What the included empirical suite actually measured | Literature-wide state of the art |
| Runtime and deployment modules | That serving, dApp, xApp, O-RAN, and governance pathways exist in code | That production deployment is turnkey in every environment |
| External ecosystem references | Why the timing and architecture matter | That the repository has measured superiority in those external settings |

## Reproducibility Philosophy

A repository is reproducible when a new reader can answer four questions clearly. **What is the system? Which parts are directly runnable? Which claims are numerically validated? Which parts require extra assumptions?** The documentation for this repository is organized around exactly those questions.

The first answer is that the system is **UHCI**, not merely a narrow single-model benchmark. The second answer is that the repository contains both a directly reproducible benchmark path and a much broader heterogeneous-connectivity architecture. The third answer is that the strongest shared numeric validation remains concentrated in the included benchmark suite. The fourth answer is that several advanced surfaces, including real-data ingestion, telecom-control integration, and richer deployment contexts, depend on optional datasets, infrastructure, or environment-specific setup.[1] [2] [3] [4] [5] [6] [10] [11] [12] [13]

> The most honest reading of the repository is that **its architectural scope is broader than its fully aggregated benchmark scope**.

## What Is Directly Reproducible

Several important parts of the repository are directly reproducible from a clean checkout. A reader can inspect the benchmark methodology, configuration surfaces, training entry points, exported protocol definitions, and saved benchmark summary without requiring hidden notebooks or private services.[1] [13] [14] [15]

The broader UHCI implementation is also directly inspectable as code. A reader can audit the provider taxonomy, unified environment, ITU and FR3 propagation logic, graph-structured encoder, CfC-capable temporal backend, universal agent abstraction, low-latency runtime, gRPC service, E2-facing adapter, and Non-RT lifecycle logic as implementation evidence even when a single end-to-end public benchmark artifact is not yet the main proof surface for every module.[2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

| Surface | Directly reproducible from repository alone? | Why |
|---|---|---|
| Benchmark methodology | Yes | Harness, config, and saved summary are present |
| Benchmark summary | Yes | JSON artifact is included |
| Core UHCI architecture | Yes, as implementation evidence | Source code for provider, environment, model, runtime, and governance layers is present |
| Real-data pathway in full | Partially | The code path is present, but datasets and preparation steps may be external |
| O-RAN or operator deployment in full | Partially | Integration code exists, but external infrastructure is context-dependent |
| AI-RAN-oriented packaging path | Partially | Packaging and adapter code exist, but accelerator environment is external |

## The Strongest Quantitative Evidence

The benchmark harness in `benchmarks/benchmark.py`, its configuration file, and the saved summary artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json` remain the strongest direct empirical anchors in the repository.[13] [14] [15] These artifacts support specific claims and should be treated as the primary source for quantitative language anywhere in the documentation.

The safest quantitative conclusion is precise rather than promotional: **within the included benchmark suite, the repository reports leading results on key canonical control metrics such as success rate, lowest collision rate, and spectral efficiency**. At the same time, the artifact does **not** show universal leadership on every metric, and it does **not** establish literature-wide state of the art across all external methods, datasets, or telecom environments.[14] [15]

| Benchmark claim boundary | Supported directly? | Source of proof |
|---|---|---|
| Best reported result in the included suite on success rate | Yes | `benchmark_summary.json` |
| Best reported result in the included suite on lowest collision rate | Yes | `benchmark_summary.json` |
| Best reported result in the included suite on spectral efficiency | Yes | `benchmark_summary.json` |
| Best on every metric in the included suite | No | Reward, fairness, and latency show mixed leadership |
| Literature-wide state of the art across all external baselines | No | External methods are not comprehensively benchmarked here |
| Full end-to-end UHCI benchmark maturity equal to the included benchmark suite | No | The architecture is broader than the aggregated benchmark set |

This distinction is central to the documentation strategy. The repository can responsibly claim **measured improvement on important telecom-control metrics within the included benchmark suite**, but it should not collapse that into a blanket claim that every UHCI subsystem has already been benchmarked exhaustively in the same way.[13] [14] [15]

## UHCI as Implementation-Backed Evidence

The repository’s broader UHCI system is strongly supported by implementation evidence. `provider_registry.py` encodes heterogeneous provider classes and capability assumptions. `itu_propagation.py` and `fr3_propagation.py` encode propagation realism and regime-specific channel behavior. `unified_connectivity_env.py` constructs a provider-aware and heterogeneous decision surface. `universal_spectrum_agent.py`, `hetero_gnn_encoder.py`, and `ltc_cell_cfc.py` expose the core reasoning stack that combines structural encoding with temporal control. The runtime stack then extends that architecture into serving, dApp execution, O-RAN integration, and lifecycle governance.[2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

That evidence is strong because it proves that the architecture exists in code, not only in marketing language or conceptual diagrams. However, implementation-backed evidence should still be documented differently from benchmark-backed evidence. It supports statements such as **“the repository implements a heterogeneous-provider control architecture”** or **“the repository includes deployment-facing O-RAN and governance surfaces.”** It does **not** automatically support claims such as **“the full UHCI stack is benchmarked end to end as field-leading across all settings.”**[2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

| UHCI surface | Strongest evidence type | Safe documentation posture |
|---|---|---|
| Provider taxonomy and heterogeneous capability modeling | Implementation evidence | State that the repository encodes these structures in code |
| ITU and FR3 propagation modeling | Implementation evidence | State that the repository includes telecom-oriented physics layers |
| Unified heterogeneous environment | Implementation evidence | State that the repository supports this experimental surface |
| Universal agent, graph encoder, and CfC-capable temporal reasoning | Implementation evidence | State that the repository implements these learning modules |
| Low-latency runtime, serving, and governance | Implementation plus runtime utilities | State that the repository includes operational execution paths |
| Full architecture-wide superiority claims | Not directly established | Avoid unless supported by a specific benchmark or deployment study |

## Real-Data Reproducibility Boundaries

The repository contains a meaningful real-data pathway through `src/preceptualai/env/data_pipeline.py` and related UHCI surfaces.[6] That is important evidence that the codebase is designed to operate beyond purely synthetic simulation. The code is structured to ingest richer telecom data sources into the broader connectivity-intelligence stack rather than treating data realism as an afterthought.[1] [6]

At the same time, the documentation should draw a clear line between **a coded pathway for real-data use** and **a fully self-contained public data package that every reader can run immediately**. The repository directly proves the former. It does not automatically guarantee the latter unless the user also has access to the relevant datasets, schemas, formats, and environment assumptions.[6]

| Real-data statement | Safe? | Why |
|---|---|---|
| The repository contains real-data ingestion pathways | Yes | The data pipeline is implemented in code |
| The repository is designed to operate beyond toy simulation | Yes | UHCI environment and data-ingestion modules make that explicit |
| Every reader can reproduce all real-data experiments immediately from a clean machine | No | External datasets and preprocessing setup may be required |

## Deployment Reproducibility Boundaries

The runtime and deployment stack is also partly reproducible and partly context-dependent. A reader can inspect the service interface, runtime abstraction, low-latency dApp path, O-RAN integration surfaces, lifecycle-governance logic, and AI-RAN packaging modules directly from the repository.[8] [9] [10] [11] [12] [16] [17] Those are strong implementation signals.

However, a complete operator-grade deployment depends on external infrastructure such as RIC platforms, policy routing, cluster configuration, accelerator availability, networking, and operational controls that cannot be inferred solely from the repository.[8] [9] [10] [11] [12] [16] [17]

> The code supports a strong claim that the repository is **deployment-oriented**. It does not support the careless claim that deployment is **infrastructure-free**.

| Deployment statement | Safe? | Why |
|---|---|---|
| The repository exposes serving, health, and metrics surfaces | Yes | Server and proto contracts are explicit |
| The repository includes low-latency dApp execution logic | Yes | `dapp_engine.py` implements and benchmarks this path |
| The repository includes O-RAN-facing and lifecycle-governance code | Yes | `oran.py`, `e2_adapter.py`, and `rapp_trainer.py` are explicit |
| The repository includes AI-RAN-oriented packaging support | Yes | `aerial_adapter.py` and container assets are present |
| The repository alone is a complete production deployment package for every operator | No | External infrastructure remains necessary |

## Recommended Reproduction Sequence

A careful reader who wants the highest-confidence reproduction path should move from the most mature evidence to the more context-dependent surfaces. This sequence minimizes confusion because it lets the reader validate the strongest numerical evidence first and then expand into broader UHCI workflows and deployment contexts.[13] [14] [15]

| Stage | Recommended action | Why this order matters |
|---|---|---|
| 1 | Install the development environment and verify basic repository setup | Confirms that the local environment is stable |
| 2 | Run a short training and evaluation loop | Establishes the simplest working learning path |
| 3 | Run the benchmark harness | Reproduces the strongest aggregated quantitative evidence |
| 4 | Inspect the benchmark summary and configuration | Grounds performance language in saved artifacts |
| 5 | Read and run UHCI-specific entry points | Expands from benchmarked control into heterogeneous connectivity intelligence |
| 6 | Explore real-data pathways | Adds external-data complexity only after the core path is understood |
| 7 | Explore serving, dApp, xApp, and lifecycle modules | Approaches deployment only after the model and evidence story are clear |

## Source-of-Truth Hierarchy

When narrative documents and executable artifacts diverge, the repository should be read according to a clear source-of-truth hierarchy. This is especially important because the codebase spans benchmark claims, runtime behavior, strategic narrative, and broader architectural ambition.

| If the question is about... | Prefer these sources first |
|---|---|
| What UHCI includes architecturally | `src/preceptualai/env/`, `src/preceptualai/core/`, `src/preceptualai/xapp/`, and `scripts/train_uhci.py` |
| What was benchmarked quantitatively | `benchmarks/benchmark.py`, `benchmark_config.yaml`, and `benchmark_summary.json` |
| What can be served or deployed | `proto/`, `server.py`, `dapp_engine.py`, `e2_adapter.py`, `rapp_trainer.py`, and deployment assets |
| What dependencies and tooling are expected | `pyproject.toml` and runtime config files |
| How to phrase claims safely | Benchmark artifacts first, then implementation evidence, then contextual references |

This hierarchy is what allows the documentation to remain both ambitious and honest. It acknowledges the full UHCI architecture without pretending that every implemented surface has identical empirical maturity.[1] [2] [3] [4] [5] [6] [13] [14] [15]

## What a Responsible Reader Should Conclude

A lay reader should conclude that the repository contains **real measured improvements on important canonical control metrics** and a much larger **UHCI architecture** designed for heterogeneous connectivity intelligence, telecom realism, and deployment-oriented workflows.[2] [3] [4] [5] [8] [9] [10] [11] [12] [14]

A technical reader should conclude something more specific: the repository offers **high-confidence reproducibility for the included benchmark suite**, **strong implementation-level evidence for the broader UHCI architecture**, and **partial but meaningful reproducibility for real-data and operator-facing deployment surfaces subject to optional datasets and infrastructure**.[1] [2] [3] [4] [5] [6] [8] [9] [10] [11] [12] [13] [14] [15]

## References

[1]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[2]: [Provider registry in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[3]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[4]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[5]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[6]: [Unified data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[7]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[8]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[9]: [CfC and related temporal module in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[10]: [Runtime inference abstraction in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[11]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[12]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[13]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[14]: [Aggregated benchmark artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
[15]: [Benchmark configuration in `benchmarks/benchmark_config.yaml`](../benchmarks/benchmark_config.yaml)
[16]: [O-RAN environment module in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[17]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
