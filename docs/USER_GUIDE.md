# User Guide

This guide explains how to install, run, evaluate, and operate **PreceptualAI** as a **Universal Heterogeneous Connectivity Intelligence (UHCI)** system. The repository should be understood as a layered telecom intelligence stack that combines **heterogeneous provider modeling**, **physics-aware propagation**, **graph and temporal decision models**, **runtime serving**, **O-RAN-facing integration**, and **deployment-oriented control surfaces** rather than as a single narrow benchmark script.[1] [2] [3] [4] [5] [6] [7] [8] [9]

A practical reading of the repository begins with one central question: **how can an intelligent controller choose among multiple connectivity options under changing wireless conditions, deployment constraints, and provider-specific trade-offs?** The codebase answers that question through UHCI. Some parts of the repository provide directly reproducible benchmark evidence, while other parts establish architecture, runtime, and integration scope through implementation. This guide keeps those two forms of evidence separate so users can understand what is already measured versus what is already built.[1] [2] [3] [4] [5] [6]

| Reader goal | Best starting path | Why this path is recommended |
|---|---|---|
| I want the repository running quickly | Install the package in editable mode and verify imports | Confirms that the local environment is valid before deeper experiments |
| I want the strongest reproducible quantitative evidence | Run the benchmark harness and inspect saved artifacts | Grounds performance interpretation in the clearest repository-backed metrics |
| I want to understand the full UHCI system | Run `train_uhci.py`, then read the architecture and research guides | Connects execution with the codebase’s broader technical model |
| I want to deploy or integrate with telecom control loops | Study the xApp, server, E2, and governance modules after the base install works | Runtime and integration paths are richer, but also more environment-dependent |

## 1. What This Repository Is

PreceptualAI is not just an optimizer for a single radio-control toy problem. In the current repository, it is a **UHCI platform** that models a world of heterogeneous connectivity providers, embeds wireless propagation priors, constructs a unified control environment, trains a universal decision agent, and exposes runtime surfaces for low-latency inference, observability, and telecom-oriented integration.[1] [2] [3] [4] [5] [6] [7] [8] [9]

The practical meaning of UHCI is that the system is designed to reason across **provider classes**, **frequency regimes**, **terrestrial and non-terrestrial settings**, **real-data pathways**, and **deployment constraints** rather than only across a flat list of interchangeable channels. The repository therefore contains both research modules and operational modules, and users should expect the most advanced deployment paths to require more infrastructure than the simplest local benchmark route.[1] [2] [3] [4] [7] [8] [9]

| UHCI layer | Primary repository anchors | User-facing meaning |
|---|---|---|
| Provider taxonomy | `src/preceptualai/env/provider_registry.py` | Defines what kinds of connectivity the system can reason about |
| Physics layer | `src/preceptualai/env/itu_propagation.py`, `src/preceptualai/env/fr3_propagation.py` | Adds realistic propagation constraints |
| Unified environment | `src/preceptualai/env/unified_connectivity_env.py` | Turns heterogeneous connectivity into a control problem |
| Data pathways | `src/preceptualai/env/data_pipeline.py` | Supports richer empirical workflows beyond pure toy simulation |
| Universal agent | `src/preceptualai/core/universal_spectrum_agent.py` | Main decision-making abstraction for UHCI |
| Representation and time | `src/preceptualai/core/hetero_gnn_encoder.py`, `src/preceptualai/core/ltc_cell_cfc.py` | Encodes structured state and sequence dynamics |
| Training entry point | `scripts/train_uhci.py` | Main research CLI for broader UHCI experimentation |
| Runtime and deployment | `src/preceptualai/xapp/*.py`, `proto/preceptualai.proto` | Makes the system callable, observable, and integrable |

## 2. Requirements

The project manifest declares Python `>=3.10` and organizes optional dependency groups for development, benchmarks, O-RAN-oriented integration, GPU runtime, and UHCI research extensions.[10] Users should think of these extras as **capability layers** rather than as one monolithic install requirement. A clean local environment is strongly recommended because the repository spans training, serving, protocol generation, deployment tooling, and optional telecom integrations.[10]

| Requirement | Recommended version | Notes |
|---|---|---|
| Python | 3.10 or newer | 3.11 is a practical choice for local development |
| pip | Current release | Upgrade before installation |
| Virtual environment | Any standard tool | Strongly recommended to isolate optional dependencies |
| Docker | Optional | Useful for deployment and container-oriented workflows |
| NVIDIA GPU | Optional | Relevant for accelerated inference and selected deployment paths |

## 3. Clone and Install

Start by cloning the repository and creating an isolated environment. Editable installation is the best default because it keeps scripts, source code, and documentation aligned while you explore the system.

```bash
git clone https://github.com/Danielfoojunwei/PreceptualAI-5G-Dynamic-Spectrum-Access-O-RAN-NVIDIA-ARC.git
cd PreceptualAI-5G-Dynamic-Spectrum-Access-O-RAN-NVIDIA-ARC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This base path gives you the package itself plus the development workflow without forcing every optional subsystem into the first environment. That is important because some advanced surfaces, especially graph-heavy UHCI research and O-RAN-facing runtime work, rely on extra packages that not every user needs on day one.[10]

| Install profile | Command | Use it when |
|---|---|---|
| Base development | `pip install -e ".[dev]"` | You want the repository running locally and want testing, linting, and scripts |
| Benchmark path | `pip install -e ".[dev,benchmarks]"` | You want to reproduce the included benchmark results |
| UHCI research path | `pip install -e ".[uhci]"` | You want to explore graph-heavy or broader heterogeneous-connectivity workflows |
| Telecom integration path | `pip install -e ".[oran]"` | You want xApp- and O-RAN-oriented integration modules |
| Accelerated local path | `pip install -e ".[dev,benchmarks,gpu]"` | You want accelerated runtime or export experiments |

## 4. Verify the Installation

Before attempting training or deployment, verify that the package imports successfully and that the main script surfaces are visible. This step removes ambiguity early and is the fastest way to confirm that the environment is valid.

```bash
python -c "import preceptualai; print('preceptualai import OK')"
python scripts/train.py --help
python scripts/train_uhci.py --help
```

If these commands succeed, you have confirmed that the base package and the two main research entry points are available. At that stage, you can choose whether to proceed first into the **benchmark suite workflow** or directly into the broader **UHCI workflow**.[10] [11] [12]

## 5. Quick Start Paths

The repository supports more than one credible first execution path. The right path depends on whether your priority is **quantitative evidence**, **system understanding**, or **runtime integration**.

### 5.1 Fastest Evidence Path

If your primary goal is to reproduce the strongest shared quantitative evidence in the repository, start with the benchmark harness.

```bash
pip install -e ".[dev,benchmarks]"
python benchmarks/run_full_benchmark.py --config benchmarks/benchmark_config.yaml
```

This route matters because the saved benchmark artifacts provide the clearest repository-grounded source for benchmark-backed claims about control metrics such as success rate, collision rate, and spectral efficiency inside the included benchmark suite.[13] [14] [15]

### 5.2 Fastest UHCI Path

If your primary goal is to explore the repository’s broader system identity, move into the UHCI training entry point.

```bash
pip install -e ".[uhci]"
python scripts/train_uhci.py --help
python scripts/train_uhci.py --num-steps 1000
```

This command path is the most direct way to experience the repository as a **provider-aware, heterogeneous-connectivity system** rather than as only a narrow benchmark environment.[11] [1] [4] [5] [6]

| Quick start path | Strongest outcome | Main limitation |
|---|---|---|
| Benchmark harness | Best directly reproducible quantitative evidence | Covers the included benchmark suite, not every deployed subsystem |
| UHCI training entry point | Best view of the broader architecture | May require optional research extras and deeper environment assumptions |
| Runtime server | Best view of deployment surfaces | Assumes you already have a valid model artifact and config |

## 6. Training and Evaluation

The repository contains two conceptually different empirical surfaces. One is the **included benchmark suite**, which is the clearest source of shared numerical evidence. The other is the broader **UHCI experimentation lane**, which exposes richer modeling and deployment ideas through code and CLI surfaces.[11] [13] [14] [15]

### 6.1 Included Benchmark Suite

Use the benchmark harness when you want the cleanest reproduction path for the repository’s saved performance artifacts.

```bash
python benchmarks/run_full_benchmark.py --config benchmarks/benchmark_config.yaml
```

The output artifacts in `benchmarks/results/benchmark_results_full/` are the safest place to ground performance claims because they summarize the results of the included benchmark evaluation setup.[13] [14] [15]

### 6.2 UHCI Research Lane

Use `scripts/train_uhci.py` when you want to explore the full system’s heterogeneous-control formulation.

```bash
python scripts/train_uhci.py \
    --num-steps 5000 \
    --providers LEO MEO GEO HAPS FR1 FR3 ISAC WIFI7 \
    --temporal-backend cfc
```

The training entry point also exposes richer switches for real-data-related workflows, alternative temporal backends, and optional modeling components. Users should interpret this as evidence that the repository supports a broader research program, not as proof that every possible switch combination has already been benchmarked equally in the public artifact set.[11]

| Training surface | Primary command | What it demonstrates |
|---|---|---|
| Included benchmark suite | `python benchmarks/run_full_benchmark.py --config ...` | Measured performance in the included benchmark suite |
| UHCI experimentation | `python scripts/train_uhci.py ...` | Broader heterogeneous-connectivity architecture and experimentation scope |
| Core training lane | `python scripts/train.py ...` | A narrower training surface that remains present in the repository |

### 6.3 Safe Interpretation of Results

A critical documentation rule for this repository is that **implementation scope** and **benchmark scope** are not identical. The codebase demonstrates that UHCI includes provider modeling, propagation, graph encoding, temporal reasoning, runtime serving, O-RAN-facing control, and governance layers. The benchmark artifacts demonstrate measured quantitative performance for the included evaluation lane. These two facts are both valuable, but they should not be conflated.[1] [2] [3] [4] [5] [6] [13] [14] [15]

| Evidence type | Strongest proof source | Safe claim boundary |
|---|---|---|
| Architecture breadth | Source code across `env/`, `core/`, and `xapp/` | What the repository implements |
| Reproducible empirical metrics | Benchmark artifacts in `benchmarks/results/...` | What the repository has quantitatively demonstrated in the included suite |
| Deployment intent | Serving, E2, O-RAN, and governance modules | That the system is designed for operational integration |

## 7. UHCI Concepts That Matter Most

The fastest way to understand the repository technically is to focus on four linked ideas: **provider heterogeneity**, **physics-aware simulation**, **structured representation**, and **deployment-aware intelligence**.

First, the provider registry defines the classes of connectivity the system can reason about. Second, the propagation modules constrain those choices with realistic channel behavior. Third, the environment constructs a state and reward surface over those ingredients. Fourth, the universal agent and encoder stack convert that structured world into actions that can later be served or integrated into telecom control loops.[1] [2] [3] [4] [5] [6]

| UHCI concept | Why it matters to a user |
|---|---|
| Provider taxonomy | Explains why the system is not just choosing among interchangeable channels |
| ITU and FR3 propagation | Explains why decisions are tied to realistic wireless conditions |
| Heterogeneous GNN encoding | Explains how structured multi-entity state is represented |
| CfC / LTC temporal backends | Explains how mixed-timescale dynamics are modeled |
| Runtime serving and O-RAN modules | Explains how learned intelligence becomes operational software |

## 8. Runtime Serving

The runtime stack shows that PreceptualAI is meant to be more than an offline training project. The repository exposes protocol definitions, an inference engine abstraction, a low-latency runtime path, and a server surface for prediction, health checks, and metrics.[16] [17] [18] [19]

A practical first runtime step is to inspect `xapp_config.yaml`, prepare a compatible model artifact, and launch the server.

```bash
python -m preceptualai.xapp.server --config xapp_config.yaml
```

This path should be treated as a **later-stage operational track** rather than as the first thing every new user must run. It becomes most meaningful after you already understand the model and its training path.[19] [20]

| Runtime component | File | Purpose |
|---|---|---|
| Protocol definition | `proto/preceptualai.proto` | Defines the gRPC service contract |
| Inference abstraction | `src/preceptualai/xapp/inference_engine.py` | Standardizes model execution backends |
| Low-latency path | `src/preceptualai/xapp/dapp_engine.py` | Supports near-real-time execution flow |
| Service surface | `src/preceptualai/xapp/server.py` | Exposes prediction, health, and metrics interfaces |

## 9. gRPC and Interface Surface

The protocol file indicates that the repository formalizes prediction and observability through a gRPC interface.[16] This matters because it turns the trained system into a service rather than leaving it as a research checkpoint only. The exact request and response semantics should always be confirmed against `proto/preceptualai.proto`, but the broad service pattern includes prediction, health, and metrics-oriented calls.[16] [19]

Users integrating with the runtime should treat the protocol contract and the server implementation as the authoritative interface pair. The documentation can describe the pattern, but the `.proto` file and serving module remain the canonical source of truth when there is any ambiguity.[16] [19]

## 10. O-RAN, E2, and Telecom Integration

A major differentiator of the repository is that the intelligence stack does not stop at model inference. The codebase includes O-RAN- and RIC-oriented surfaces that move the system toward real telecom control-loop integration. `oran.py` captures live O-RAN environment concepts, `e2_adapter.py` supports E2-facing interaction, and `rapp_trainer.py` introduces longer-timescale lifecycle and governance behavior consistent with non-real-time management flows.[20] [21] [22]

For users coming from traditional ML projects, this means the repository should be read partly as a **telecom systems project** rather than only as a model-training project. Some modules exist to support orchestration, lifecycle approval, health monitoring, and operator-facing control semantics, not only raw prediction quality.[20] [21] [22]

| Telecom-facing layer | File | Operational meaning |
|---|---|---|
| O-RAN environment semantics | `src/preceptualai/env/oran.py` | Connects intelligence to telecom control context |
| E2 integration | `src/preceptualai/xapp/e2_adapter.py` | Bridges runtime decisions to measurement/control pathways |
| rApp lifecycle governance | `src/preceptualai/xapp/rapp_trainer.py` | Adds model approval, monitoring, and policy lifecycle logic |

## 11. Federated and Distributed Learning

The repository also includes federated-learning surfaces. These should be interpreted as evidence that the system is designed to support distributed training and coordination patterns, particularly where raw edge data should not be centralized. Users should treat these modules as part of the repository’s broader operational ambition, while still validating the exact workflow against the current code and any deployment-specific prerequisites in their environment.[23]

A prudent approach is to read the implementation before promising a specific federated topology in production. The repository clearly contains this capability surface, but environment-specific deployment details may vary depending on infrastructure, dataset availability, and orchestration assumptions.[23]

## 12. Docker and Deployment Paths

The repository includes container-oriented deployment assets, including telecom- and accelerator-oriented packaging. This is especially relevant for users exploring AI-RAN-style deployment pathways or accelerated serving surfaces.[24] [25]

A Docker workflow is useful when you want to control runtime dependencies more tightly than in a local editable install. However, container execution should still be treated as a **deployment track**, not as the only legitimate way to use the repository. For most readers, understanding the system and reproducing a benchmark locally is the more appropriate first step.[24] [25]

| Deployment asset | Purpose |
|---|---|
| `docker-compose` assets | Multi-service local orchestration |
| `Dockerfile` variants | Standardized build environments |
| `Dockerfile.aerial` | AI-RAN / NVIDIA Aerial-oriented packaging path |

## 13. Monitoring and Observability

The serving stack includes health and metrics surfaces, which means the system is designed to be observable in operation. That is a significant property for telecom-facing AI, because usefulness in production depends not only on prediction quality but also on whether the system can be monitored, debugged, and governed over time.[19] [22]

Users deploying the server should inspect the configuration file and runtime modules to understand metrics exposure, logging expectations, and health behavior. The exact metric names and export patterns are implementation details that should be validated directly against the running service and the current source code.[19] [20]

## 14. Configuration

`xapp_config.yaml` is the primary practical starting point for runtime configuration. It expresses environment assumptions, model parameters, inference configuration, transport settings, and monitoring behavior.[26] A user should read this file together with the serving and inference modules because configuration fields make the most sense when interpreted against the runtime code that consumes them.[17] [19] [26]

| Config category | Why it matters |
|---|---|
| Environment | Determines what kind of world or input assumptions the runtime expects |
| Model | Aligns runtime with the trained model artifact |
| Inference | Selects backend, precision, and device assumptions |
| gRPC | Controls service exposure |
| Monitoring | Controls health, logging, and metrics behavior |

## 15. Troubleshooting

Most user-facing issues fall into a small number of categories. The most common are missing optional dependencies, mismatched execution expectations, and deployment tracks being attempted before the simpler research path is stable.

If import verification fails, rebuild the environment and reinstall from the manifest. If benchmark scripts fail, ensure that the benchmark extras are installed. If UHCI exploration fails, check that the UHCI-specific extras and any graph-related packages are present. If runtime serving fails, verify the config, the model artifact, and the backend assumptions before assuming the model itself is at fault.[10] [11] [13] [19] [26]

| Problem pattern | Likely cause | Recommended fix |
|---|---|---|
| Import errors | Missing extras or stale virtual environment | Recreate the environment and reinstall from `pyproject.toml` |
| Benchmark failure | Missing benchmark dependencies | Install `.[dev,benchmarks]` and rerun |
| UHCI script failure | Missing research extras or graph packages | Install `.[uhci]` and verify optional dependencies |
| Server startup failure | Bad config path, missing model, or backend mismatch | Validate `xapp_config.yaml` and runtime artifacts |
| Telecom integration failure | Missing external infrastructure or environment assumptions | Treat O-RAN and E2 paths as later-stage integrations |

## 16. Frequently Asked Questions

### What is the simplest correct mental model for this repository?

The best concise answer is that PreceptualAI is a **UHCI system**: a layered intelligence stack for choosing well across changing connectivity options in modern programmable wireless environments.[1] [4] [5] [6]

### Is the repository only about one benchmark?

No. The repository contains a benchmark suite, but the codebase is broader than that. It includes provider modeling, propagation, graph and temporal learning, runtime serving, O-RAN-facing integration, and governance modules.[1] [2] [3] [4] [5] [6] [17] [18] [19] [20] [21] [22]

### Can I claim that the full UHCI stack is already benchmarked end to end?

Not safely. The repository’s benchmark artifacts support measured claims for the included evaluation lane. The broader codebase proves substantial implementation scope. Those two facts should be presented together, but not collapsed into a stronger claim than the evidence supports.[13] [14] [15]

### Do I need a GPU?

No. A GPU can be useful for accelerated inference, export, or larger experiments, but it is not a universal prerequisite for understanding the repository or running the simplest local paths.[10] [24] [25]

### Do I need telecom infrastructure to learn the repository?

No. Most readers should begin with local installation, documentation, and the benchmark or UHCI training path. Telecom integration is a later-stage operational path.[11] [13] [19] [20] [21] [22]

### Where should I go after this guide?

If you want the big picture, read [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md). If you want the technical structure, read [`ARCHITECTURE.md`](ARCHITECTURE.md). If you want the research framing, read [`UHCI_RESEARCH_GUIDE.md`](UHCI_RESEARCH_GUIDE.md). If you want empirical boundaries, read [`TRAINING_AND_EVALUATION.md`](TRAINING_AND_EVALUATION.md) and [`DATA_AND_REPRODUCIBILITY.md`](DATA_AND_REPRODUCIBILITY.md).

## 17. Recommended Reading Order

The best reading order depends on whether you are a lay reader, a researcher, or an operator. In all cases, the aim is to move from high-level orientation toward the exact subsystem you care about without losing the UHCI-first picture.

| Reader type | Read in this order |
|---|---|
| Lay reader | `README.md` → `SYSTEM_OVERVIEW.md` → `SETUP_AND_QUICKSTART.md` |
| Researcher | `README.md` → `UHCI_RESEARCH_GUIDE.md` → `ARCHITECTURE.md` → `TRAINING_AND_EVALUATION.md` |
| Deployment engineer | `README.md` → `ARCHITECTURE.md` → `DEPLOYMENT_AND_XAPP_GUIDE.md` → `API_AND_INTERFACES.md` |
| Contributor | `README.md` → `REPOSITORY_MAP.md` → `DEVELOPER_GUIDE.md` |

## References

[1]: [Provider registry in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[2]: [ITU propagation in `src/preceptualai/env/itu_propagation.py`](../src/preceptualai/env/itu_propagation.py)
[3]: [FR3 propagation in `src/preceptualai/env/fr3_propagation.py`](../src/preceptualai/env/fr3_propagation.py)
[4]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[5]: [Data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[6]: [Universal spectrum agent in `src/preceptualai/core/universal_spectrum_agent.py`](../src/preceptualai/core/universal_spectrum_agent.py)
[7]: [Heterogeneous GNN encoder in `src/preceptualai/core/hetero_gnn_encoder.py`](../src/preceptualai/core/hetero_gnn_encoder.py)
[8]: [Temporal backends in `src/preceptualai/core/ltc_cell_cfc.py`](../src/preceptualai/core/ltc_cell_cfc.py)
[9]: [System overview](SYSTEM_OVERVIEW.md)
[10]: [Project manifest in `pyproject.toml`](../pyproject.toml)
[11]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[12]: [Legacy training entry point in `scripts/train.py`](../scripts/train.py)
[13]: [Benchmark guide in `benchmarks/README.md`](../benchmarks/README.md)
[14]: [Benchmark configuration in `benchmarks/benchmark_config.yaml`](../benchmarks/benchmark_config.yaml)
[15]: [Benchmark summary artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
[16]: [gRPC protocol in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[17]: [Inference engine in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[18]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[19]: [Serving runtime in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[20]: [O-RAN environment surface in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[21]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[22]: [rApp trainer in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[23]: [Federated module directory](../src/preceptualai/federated)
[24]: [Docker assets](../docker)
[25]: [AI-RAN Dockerfile](../docker/Dockerfile.aerial)
[26]: [Runtime configuration in `xapp_config.yaml`](../xapp_config.yaml)
