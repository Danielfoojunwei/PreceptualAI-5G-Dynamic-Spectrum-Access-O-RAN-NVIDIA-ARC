# API and Interfaces

This document explains the repository’s most important **user-facing, integration-facing, and system-facing interfaces** when the codebase is read correctly as a **UHCI-first architecture**. In this repository, interfaces are not limited to a single inference endpoint. They include **training entry points, environment and provider-model interfaces, propagation-aware configuration surfaces, runtime inference abstractions, protobuf contracts, O-RAN-facing control boundaries, and lifecycle-governance pathways**.[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]

The key idea is that **Universal Heterogeneous Connectivity Intelligence (UHCI)** is exposed through several layers of control. A researcher interacts through scripts, configuration, and experiment modules. A developer interacts through package-level abstractions in `env`, `core`, `xapp`, and `federated`. An operator or integrator interacts through runtime configuration, service contracts, telemetry-facing boundaries, and lifecycle-management surfaces. A telecom systems builder may additionally interact through E2, A1, xApp, dApp, and rApp-oriented interfaces represented in the deployment modules.[2] [3] [4] [5] [6] [7] [8] [9]

| Interface layer | What it exposes | Why it matters |
|---|---|---|
| Command-line layer | Training, evaluation, export, and benchmark workflows | Fastest path to reproduction and experimentation |
| Configuration layer | Experiment, runtime, benchmark, and dependency profiles | Makes assumptions explicit and editable |
| Environment layer | Providers, propagation models, state construction, and data ingestion | Defines the world UHCI reasons over |
| Model layer | Universal agent, graph encoders, and temporal backends | Defines the learnable decision machinery |
| Protocol layer | Formal service and federated contracts | Defines integration boundaries for external systems |
| Runtime layer | Inference and serving APIs | Turns trained policies into callable services |
| Control-plane layer | O-RAN, E2, and lifecycle hooks | Connects intelligence to telecom operation |
| Governance layer | Approval, rollout, degradation, and retraining logic | Makes deployment manageable over time |

## Command-Line Interfaces

The repository exposes most practical workflows through scripts. These are the primary interfaces for researchers, developers, and operators who want to run the system without coupling directly to internal Python modules.[1]

A UHCI-first reading changes the meaning of these scripts. `train.py` and `evaluate.py` remain useful as stable entry points into the included benchmark workflow, but `train_uhci.py` is the script that most clearly exposes the repository’s broader identity as a **Universal Heterogeneous Connectivity Intelligence** system. It is the clearest top-level interface into heterogeneous-provider training, expanded environment semantics, and broader connectivity-intelligence workflows.[1] [2]

| Script | Primary role | UHCI-first interpretation |
|---|---|---|
| `scripts/train.py` | Stable training entry point | Fastest on-ramp into the repository’s benchmarked learning surface |
| `scripts/evaluate.py` | Checkpoint evaluation | Validates trained policy artifacts against expected metrics |
| `scripts/export_model.py` | Model export | Bridges trained artifacts into deployment-ready formats |
| `scripts/train_uhci.py` | Advanced heterogeneous-connectivity training | Main CLI surface for UHCI experimentation and system-level learning workflows |
| `benchmarks/run_full_benchmark.py` | Multi-agent benchmark runner | Reproduces the repository’s clearest aggregated quantitative evidence surface |

These scripts should be treated as the first interface layer because they encode the intended user workflows more explicitly than any individual internal module.[1] [2]

## Configuration Interfaces

The configuration files are part of the API because they define the parameters that external users are expected to manipulate in order to run experiments, serve models, or validate benchmarks. In other words, configuration is not just implementation detail here; it is a **contract surface**.[3] [4] [13]

| Configuration file | What it controls | Interface significance |
|---|---|---|
| `xapp_config.yaml` | Serving and runtime behavior | Main deployment-time control surface |
| `benchmarks/benchmark_config.yaml` | Benchmark schedule, seeds, environment settings, and agent hyperparameters | Reproducibility contract for the included benchmark suite |
| `pyproject.toml` | Dependency groups, tooling, validation boundaries, and install profiles | Installation and developer-surface contract |

A careful reader should note that configuration boundaries also define what is optional in the repository. Some UHCI pathways require broader dependency support and richer runtime assumptions than the narrower benchmark suite workflow. That distinction is visible in both the dependency profiles and the runtime-oriented configuration surfaces.[3] [4] [13]

## Environment and World-Model Interfaces

One of the most important things a narrow single-model reading misses is that UHCI is not only a policy model. It is also an **interface to a structured heterogeneous world**. The files in `src/preceptualai/env/` define how the system represents providers, propagation, external data, O-RAN state, and heterogeneous action semantics.[5] [6] [7] [8] [9]

`provider_registry.py` is especially important because it defines the provider taxonomy and capability surface that the broader system reasons over. `unified_connectivity_env.py` provides the main heterogeneous environment interface, while `itu_propagation.py` and `fr3_propagation.py` expose physics-layer assumptions used to shape network realism and propagation behavior.[5] [6] [7] [8]

| Environment interface | What it exposes conceptually |
|---|---|
| `provider_registry.py` | Provider taxonomy, capabilities, and heterogeneity assumptions |
| `unified_connectivity_env.py` | Main UHCI environment with heterogeneous state, action, and reward structure |
| `itu_propagation.py` | ITU-oriented propagation and attenuation modeling |
| `fr3_propagation.py` | FR3-specific propagation behavior and coexistence assumptions |
| `data_pipeline.py` | External-data ingestion and preprocessing pathways |
| `oran.py` | O-RAN-oriented operational context and live control-loop environment surfaces |

This environment layer matters because it defines **what UHCI can know and act upon**. Without these interfaces, the system would collapse into a generic policy learner. With them, it becomes a structured connectivity-intelligence stack grounded in providers, physics, and telecom-specific operating context.[5] [6] [7] [8] [9]

## Model and Decision Interfaces

Below the environment layer, the repository exposes model-facing interfaces that define how information is encoded, remembered, and converted into actions. These are not merely internal implementation details; they are architectural extension points for anyone modifying UHCI’s decision core.[10] [11] [12]

`universal_spectrum_agent.py` is the clearest high-level modeling interface because it binds together encoding, temporal reasoning, and policy execution inside a broader agent abstraction. `hetero_gnn_encoder.py` exposes the structural encoding path for heterogeneous entities and relations, while `ltc_cell_cfc.py` exposes temporal backends associated with continuous-time and CfC-style sequential reasoning.[10] [11] [12]

| Model interface | Role in the repository |
|---|---|
| `universal_spectrum_agent.py` | Main UHCI agent abstraction combining perception, temporal reasoning, and action generation |
| `hetero_gnn_encoder.py` | Graph-structured encoding interface for heterogeneous connectivity state |
| `ltc_cell_cfc.py` | Temporal reasoning interface for CfC and related recurrent control dynamics |

A UHCI-first documentation posture matters here because the repository is not just a single-agent model plus wrappers. It is a broader system in which **world modeling, structural encoding, temporal memory, and action generation** are exposed through separable interfaces that advanced users can inspect or extend.[10] [11] [12]

## Inference Service Interface

The repository’s main serving contract is defined in `proto/preceptualai.proto`, while the service implementation lives in `src/preceptualai/xapp/server.py`.[14] [15] This is the clearest formal API surface for the deployed intelligence layer.

The service includes prediction, health, metrics, and streaming-oriented pathways. In practical terms, this means the model is exposed not merely as a Python object, but as a networked service with explicit operational semantics.[14] [15]

| Proto element | Integration meaning |
|---|---|
| `PreceptualAI` service | Main serving boundary for runtime prediction |
| `PredictRequest` | Request payload for inference |
| `PredictResponse` | Response payload carrying model output |
| Health-related messages | Liveness and readiness surface |
| Metrics-related messages | Observability and runtime-statistics surface |

This service boundary matters because it provides a concrete way to integrate UHCI-aligned intelligence into larger systems without depending on unstable internal Python contracts.[14] [15]

## Runtime Inference Interfaces

Below the public service surface, the repository contains explicit runtime abstractions that shape how inference is executed. `src/preceptualai/xapp/inference_engine.py` provides a generalized model-execution interface, while `src/preceptualai/xapp/dapp_engine.py` provides a more specialized low-latency path oriented toward RT-style execution.[16] [17]

This distinction is important. The inference engine is the general execution layer, whereas the dApp engine is the optimized interface for near-real-time operation with persistent temporal state, optional quantization, and accelerator-aware execution paths.[16] [17]

| Runtime interface | Role |
|---|---|
| `inference_engine.py` | Common runtime abstraction for loading and executing trained models |
| `dapp_engine.py` | Low-latency inference path for RT-oriented or near-real-time deployment |
| `server.py` | gRPC wrapper around the runtime layer |

In a UHCI-first system, runtime interfaces are not incidental. They are part of the architecture that allows learned connectivity intelligence to operate under realistic timing and systems constraints.[15] [16] [17]

## O-RAN and Control-Plane Interfaces

The repository also includes interfaces that sit much closer to telecom control-plane semantics than a generic machine-learning API would. `src/preceptualai/env/oran.py` provides an O-RAN-facing operational surface, while `src/preceptualai/xapp/e2_adapter.py` handles E2-related measurement and control interaction patterns.[9] [18]

These files should be understood as **system interfaces**, even if they are not simple REST or gRPC endpoints. They encode how the intelligence stack expects to receive telemetry, structure state, and emit control-like actions in an O-RAN-aligned context.[9] [18]

| Control-plane interface | What it exposes conceptually |
|---|---|
| `oran.py` | O-RAN-oriented environment and loop semantics |
| `e2_adapter.py` | E2-facing measurement subscription and control-action boundary |
| `rapp_trainer.py` | Governance and policy-generation surfaces for longer-timescale lifecycle control |

This makes the repository’s interface story richer than a standard ML service. UHCI is exposed not only through inference APIs, but also through **telecom control-loop and governance boundaries**.[9] [18] [19]

## Federated-Learning Interface

The federated-learning interface is defined separately in `proto/preceptualai_fl.proto`.[20] This is an important architectural choice because it keeps distributed-learning coordination distinct from live inference serving. The federated proto models concepts such as registration, round progression, tensor transport, and upload or download of model weights.[20]

| Federated proto concept | Meaning |
|---|---|
| Registration | Device or client onboarding into a federated workflow |
| Weight upload or download | Synchronization of model parameters across rounds |
| Round status and metrics | Visibility into distributed-training progress |
| Tensor transport | Concrete data path for model-state exchange |

This separation indicates that the repository treats federated coordination as a first-class integration domain rather than as an implementation detail hidden inside the main serving API.[20]

## Lifecycle and Governance Interfaces

One of the most overlooked interface layers in the repository is the lifecycle-governance surface exposed by `src/preceptualai/xapp/rapp_trainer.py`.[19] This module handles model registration, approval, deployment tracking, degradation detection, retraining triggers, and policy generation. That means it functions as a management interface for the system’s longer-timescale operational behavior.[19]

| Governance interface | Function |
|---|---|
| Model catalog operations | Tracks available trained models and versions |
| Approval and deployment state | Separates trained artifacts from deployable artifacts |
| Degradation monitoring | Detects when retraining or rollback may be needed |
| Policy-generation surface | Produces deployment-consumable control artifacts |

For telecom deployments, this layer is often as important as the prediction API itself, because safe systems need to govern when and how models are activated, replaced, or rolled back.[19]

## Package-Level Interfaces

In addition to explicit scripts and protocols, the package structure itself exposes conceptual interfaces. Different directories represent different extension points in the broader UHCI system. A contributor who wants to modify provider logic should not need to rewrite the serving layer, and a deployment engineer who wants to change runtime behavior should not need to redesign the graph encoder.

| Package | Interface role in the repository |
|---|---|
| `core` | Modeling primitives, temporal modules, encoders, and universal-agent building blocks |
| `agent` | Earlier policy-training orchestration surfaces |
| `env` | Environment, provider taxonomy, propagation, and data-ingestion interfaces |
| `export` | Artifact conversion and runtime packaging support |
| `xapp` | Runtime serving, dApp execution, O-RAN integration, and lifecycle surfaces |
| `federated` | Distributed-learning coordination logic |
| `proto` | Formal contract definitions for integration |

A UHCI-first reading matters here because `env`, `core`, and `xapp` together define the main system surface. The repository is not just “model code plus a server.” It is **world modeling plus intelligence plus runtime plus governance**.

## Interface Guidance

When building on top of the repository, prefer the most explicit interface available. Use scripts for reproducible workflows, configuration files for parameter control, protobuf definitions for integration understanding, and runtime modules only when you deliberately need lower-level control.[1] [3] [14] [15] [16]

> The safest integration strategy is to depend on **declared interfaces first** and on deep internal implementation details only when you are intentionally extending the core architecture.

That advice is especially important here because the codebase contains both a directly benchmarked evaluation surface and a broader UHCI architecture. Integrators should avoid assuming that every experimental abstraction is as stable as the top-level scripts and protocol surfaces.[1] [2] [14]

## Recommended Reading

A reader who wants to understand the repository’s interfaces in context should continue with the following documents.

| If you want to understand... | Read next |
|---|---|
| The system architecture behind these interfaces | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| How deployment uses the runtime and control interfaces | [`DEPLOYMENT_AND_XAPP_GUIDE.md`](DEPLOYMENT_AND_XAPP_GUIDE.md) |
| How to run the repository locally | [`SETUP_AND_QUICKSTART.md`](SETUP_AND_QUICKSTART.md) |
| How training and benchmarking relate to the interfaces | [`TRAINING_AND_EVALUATION.md`](TRAINING_AND_EVALUATION.md) |

## References

[1]: [Project scripts directory](../scripts)
[2]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[3]: [Runtime configuration in `xapp_config.yaml`](../xapp_config.yaml)
[4]: [Project manifest in `pyproject.toml`](../pyproject.toml)
[5]: [Provider registry in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[6]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[7]: [ITU propagation module in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[8]: [FR3 propagation module in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[9]: [O-RAN environment module in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[10]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[11]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[12]: [CfC and related temporal module in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[13]: [Benchmark configuration in `benchmarks/benchmark_config.yaml`](../benchmarks/benchmark_config.yaml)
[14]: [Inference protobuf in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[15]: [Serving runtime in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[16]: [Runtime inference abstraction in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[17]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[18]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[19]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[20]: [Federated protobuf in `proto/preceptualai_fl.proto`](../proto/preceptualai_fl.proto)
