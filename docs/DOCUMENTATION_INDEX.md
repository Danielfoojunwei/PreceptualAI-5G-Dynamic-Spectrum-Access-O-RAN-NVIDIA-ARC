# Documentation Index

This repository now uses a **UHCI-first documentation system**. The documentation treats **Universal Heterogeneous Connectivity Intelligence (UHCI)** as the primary repository system and organizes every major guide around the full heterogeneous-connectivity architecture: provider modeling, propagation physics, graph and temporal reasoning, runtime services, O-RAN control loops, lifecycle governance, and reproducible evaluation surfaces.

The result is a documentation set that serves two audiences at once. A lay reader can understand **what UHCI is, why heterogeneous connectivity now matters, what technical problem the repository solves, and why the system is strategically relevant to O-RAN and AI-RAN**. A researcher, engineer, or operator can understand **how UHCI is implemented, how its subsystems fit together, where the empirical evidence is strong, which parts are benchmarked versus exploratory, and how to recreate the stack from code and documentation alone**.

| Document | Primary audience | UHCI-first role in the documentation system |
|---|---|---|
| [`README.md`](../README.md) | Everyone | The main narrative entry point. It explains the problem, the UHCI solution, technical moats, why now, verified empirical claims, architectural layers, and how the rest of the documentation set should be read. |
| [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) | Lay readers, operators, first-time technical readers | A plain-language explanation of UHCI as a connectivity-intelligence system spanning terrestrial, non-terrestrial, and O-RAN control surfaces. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Engineers and researchers | The technical decomposition of UHCI into environment, agent, temporal modeling, provider graph, physics layer, runtime, deployment, and lifecycle-governance subsystems. |
| [`UHCI_RESEARCH_GUIDE.md`](UHCI_RESEARCH_GUIDE.md) | Researchers and advanced builders | The deepest conceptual and implementation guide to UHCI, including provider taxonomy, real-data pathways, graph reasoning, temporal backends, and experimental surfaces. |
| [`TRAINING_AND_EVALUATION.md`](TRAINING_AND_EVALUATION.md) | ML engineers and researchers | Documents the repository’s empirical evaluation surfaces, including the included benchmark suite, UHCI training pathways, and the boundaries between measured results and broader implementation scope. |
| [`DATA_AND_REPRODUCIBILITY.md`](DATA_AND_REPRODUCIBILITY.md) | Reviewers, researchers, due-diligence readers | Separates what is directly benchmark-validated, what is implementation-supported, and what requires optional data, tooling, or deployment infrastructure. |
| [`DEPLOYMENT_AND_XAPP_GUIDE.md`](DEPLOYMENT_AND_XAPP_GUIDE.md) | Deployment engineers and operators | Explains how UHCI maps into serving, xApp, dApp, rApp, gRPC, A1, E2, and AI-RAN deployment surfaces. |
| [`API_AND_INTERFACES.md`](API_AND_INTERFACES.md) | Integration engineers and developers | Documents the executable interfaces, scripts, configuration surfaces, runtime contracts, and protocol boundaries that expose UHCI as a working system. |
| [`SETUP_AND_QUICKSTART.md`](SETUP_AND_QUICKSTART.md) | New users and developers | Explains how to install and run the fastest credible UHCI-first pathways for local experimentation, benchmarking, and architecture exploration. |
| [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) | Contributors, maintainers, reviewers | Gives a directory-by-directory mental model of the codebase, explicitly showing which directories implement UHCI core logic, support infrastructure, benchmarks, and deployment surfaces. |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Contributors | Explains how to extend the UHCI stack without blurring the distinction between benchmarked, deployable, and exploratory components. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Everyone | Provides debugging paths for installation, optional dependencies, real-data pipelines, serving, and O-RAN or AI-RAN-facing surfaces. |

## Recommended Reading Paths

The documentation is now organized around **reader intent** and **depth of technical need**, with UHCI treated as the system of record across research, engineering, and deployment pathways.

| Reader goal | Recommended path |
|---|---|
| Understand what UHCI is and why it matters | `README.md` → `SYSTEM_OVERVIEW.md` → `UHCI_RESEARCH_GUIDE.md` |
| Understand the full technical architecture | `README.md` → `ARCHITECTURE.md` → `API_AND_INTERFACES.md` |
| Understand the benchmark and evidence story | `README.md` → `TRAINING_AND_EVALUATION.md` → `DATA_AND_REPRODUCIBILITY.md` |
| Recreate the repository locally | `README.md` → `SETUP_AND_QUICKSTART.md` → `ARCHITECTURE.md` → `TRAINING_AND_EVALUATION.md` |
| Understand O-RAN, xApp, dApp, and rApp deployment surfaces | `README.md` → `DEPLOYMENT_AND_XAPP_GUIDE.md` → `API_AND_INTERFACES.md` |
| Extend UHCI as a research platform | `README.md` → `UHCI_RESEARCH_GUIDE.md` → `REPOSITORY_MAP.md` → `DEVELOPER_GUIDE.md` |

## UHCI-First Documentation Principles

The revised documentation architecture follows a few explicit rules so the repository remains understandable even though it spans multiple technical layers.

| Principle | How it is applied |
|---|---|
| **UHCI is the system of record** | Documents describe the repository first as a universal heterogeneous-connectivity intelligence stack, not as a narrow single-model benchmark artifact. |
| **Benchmarks are separated from architecture** | Measured benchmark results are documented clearly, but they are not allowed to stand in for claims about the whole UHCI stack. |
| **Deployment is part of the architecture** | rApp, dApp, xApp, serving, E2, A1, and AI-RAN pathways are treated as first-class technical surfaces rather than afterthoughts. |
| **Physics and provider modeling are explicit** | Provider taxonomy, propagation modeling, NTN surfaces, and coexistence logic are documented as core components of UHCI. |
| **Code-backed claims take priority** | When narrative language and implementation differ, the implementation and benchmark artifacts are treated as the stronger source of truth. |
| **Evaluation artifacts remain subordinate to system architecture** | Benchmark-specific agents remain documented only as evidence surfaces inside the larger UHCI system rather than as the repository’s identity. |

## Source-of-Truth Guidance

The repository includes narrative docs, benchmark artifacts, runtime code, and protocol definitions. For technical due diligence, readers should treat executable artifacts as the strongest source of truth.

| If the question is about... | Strongest source |
|---|---|
| What UHCI encompasses architecturally | `src/preceptualai/env/`, `src/preceptualai/core/`, `src/preceptualai/xapp/`, and `scripts/train_uhci.py` |
| Provider taxonomy and heterogeneous physics | `provider_registry.py`, `itu_propagation.py`, `fr3_propagation.py`, and `unified_connectivity_env.py` |
| Runtime serving and control-loop deployment | `src/preceptualai/xapp/server.py`, `dapp_engine.py`, `rapp_trainer.py`, `e2_adapter.py`, and `oran.py` |
| Benchmark metrics and performance claims | `benchmarks/benchmark.py`, benchmark configuration, and benchmark result artifacts |
| Installation and dependency boundaries | `pyproject.toml` and `SETUP_AND_QUICKSTART.md` |
| Protocol and service interfaces | `proto/`, `API_AND_INTERFACES.md`, and deployment documentation |

This index is intentionally explicit. Its purpose is to prevent the repository from being misunderstood as a narrow single-model project when the implementation is clearly a broader **UHCI architecture** spanning environment modeling, propagation physics, learning, serving, control integration, and governance.
