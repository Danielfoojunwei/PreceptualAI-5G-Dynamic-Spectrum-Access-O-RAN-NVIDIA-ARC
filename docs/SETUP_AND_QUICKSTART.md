# Setup and Quickstart

This guide explains how to get the repository running locally in the fastest credible way while keeping the repository’s **UHCI-first identity** intact. The goal is not merely to make one narrow training lane runnable. The goal is to help a new reader understand, install, and exercise the repository as a **Universal Heterogeneous Connectivity Intelligence (UHCI)** system with clear entry points for experimentation, benchmarking, and deployment-oriented exploration.[1] [2] [3] [4] [5] [6] [7] [8]

A practical setup strategy should move from the simplest validated path to the richer system surfaces. That means first verifying the local environment, then exercising the strongest directly reproducible benchmark path, then moving into the broader UHCI training entry point, and finally exploring runtime, O-RAN, and deployment modules.[1] [2] [6] [7] [8]

| Reader goal | Best starting path |
|---|---|
| I want the repository running quickly | Install the package and verify imports and scripts |
| I want the strongest directly reproducible evidence | Run the included benchmark suite and inspect saved artifacts |
| I want to understand the full system | Run the UHCI training entry point and read the architecture docs |
| I want to explore deployment | Inspect the runtime, server, dApp, xApp, and telecom-facing modules |

## Requirements

The project manifest declares Python `>=3.10` and exposes both base dependencies and optional extras for development, benchmarks, O-RAN-related integration, GPU runtime, and UHCI-oriented research extensions.[1] A new user should treat those extras as **progressive capability layers**, not as requirements that must all be installed on day one.

| Requirement category | Notes |
|---|---|
| Python | Version `3.10` or newer |
| Core compute | PyTorch-based model training and inference |
| Packaging | Editable installs are recommended for development |
| Benchmark extras | Add when reproducing the included benchmark results |
| O-RAN extras | Add when exploring telecom-control integration |
| UHCI extras | Add when exploring graph-heavy or advanced heterogeneous-connectivity workflows |

## Clone the Repository

Start by cloning the repository and entering the project directory.

```bash
git clone https://github.com/Danielfoojunwei/PreceptualAI-5G-Dynamic-Spectrum-Access-O-RAN-NVIDIA-ARC.git
cd PreceptualAI-5G-Dynamic-Spectrum-Access-O-RAN-NVIDIA-ARC
```

If you are working from a feature branch or pull request, check out that branch before installing dependencies.

## Create a Virtual Environment

Create and activate an isolated Python environment so the repository’s dependencies do not interfere with system packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

This is especially important because the repository spans research code, runtime services, benchmark tooling, and optional telecom or deployment integrations.[1]

## Install the Project

For most readers, the correct first install is an editable development install.

```bash
pip install -e ".[dev]"
```

That gives you the package itself, developer tooling, and the basic local workflow without forcing every optional subsystem into the initial environment.[1]

## Optional Install Profiles

The manifest is intentionally explicit about extras because the repository spans a wide range of use cases. You should install only the profiles that match the surface you intend to use.[1]

| Extra | Adds |
|---|---|
| `dev` | `pytest`, coverage, Ruff, mypy, pre-commit, typing helpers |
| `benchmarks` | Stable-Baselines3, plotting libraries, benchmark tooling |
| `oran` | `ricxappframe`, `mdclogpy` for O-RAN-oriented runtime work |
| `gpu` | `onnxruntime-gpu` |
| `uhci` | `torch-geometric`, `torch-scatter`, `torch-sparse` |

Example installs:

```bash
pip install -e ".[dev,benchmarks]"
pip install -e ".[dev,benchmarks,gpu]"
pip install -e ".[oran]"
pip install -e ".[uhci]"
```

## Verify the Installation

Before running training or serving workflows, verify that the package imports cleanly and that the main repository surfaces are discoverable.

```bash
python -c "import preceptualai; print('preceptualai import OK')"
python scripts/train.py --help
python scripts/train_uhci.py --help
```

A successful import and visible CLI help output tell you that the local environment is ready for the first credible execution path.[2] [3]

## Fastest High-Confidence Reproduction Path

If your goal is to validate the strongest directly reproducible evidence in the repository, start with the included benchmark suite rather than assuming that the full deployment stack is the first execution path. Install the benchmark extras and run the benchmark harness.[4] [5]

```bash
pip install -e ".[dev,benchmarks]"
python benchmarks/run_full_benchmark.py --config benchmarks/benchmark_config.yaml
```

This route matters because it reproduces the clearest aggregated metrics in the repository and gives you the safest basis for interpreting public performance claims.[4] [5] [6]

| What this step gives you | Why it matters |
|---|---|
| Aggregated benchmark metrics | Establishes the strongest directly reproducible quantitative evidence |
| Canonical control metrics | Grounds claims about success rate, collision rate, and spectral efficiency |
| Evidence boundaries | Helps separate benchmark-backed claims from architecture-backed claims |

## Move to the UHCI Training Entry Point

Once the environment is stable and you have reproduced the strongest shared evidence path, the next step is to exercise the repository as a **UHCI system**. The clearest executable entry point for that is `scripts/train_uhci.py`, which exposes a broader training surface for heterogeneous connectivity experiments.[2]

```bash
pip install -e ".[uhci]"
python scripts/train_uhci.py --help
```

The exact flags and runtime assumptions depend on the experiment you want to run, but this script is the right place to begin if your goal is to understand the repository beyond the current benchmark workflow. It connects more directly to provider-aware environments, richer state construction, and broader heterogeneous-connectivity workflows.[2] [7] [8]

## Inspect the Core UHCI Architecture

With installation verified and at least one runnable path completed, the next useful step is to inspect the major implementation surfaces that define UHCI as a whole system. These are the files that explain what the repository really encompasses beyond a single training loop.[7] [8] [9] [10] [11]

| Subsystem | Primary files to inspect |
|---|---|
| Provider and environment modeling | `src/preceptualai/env/provider_registry.py`, `src/preceptualai/env/unified_connectivity_env.py` |
| Propagation realism | `src/preceptualai/env/itu_propagation.py`, `src/preceptualai/env/fr3_propagation.py` |
| Agent and learning core | `src/preceptualai/core/universal_spectrum_agent.py`, `src/preceptualai/core/hetero_gnn_encoder.py`, `src/preceptualai/core/ltc_cell_cfc.py` |
| Runtime and serving | `src/preceptualai/xapp/inference_engine.py`, `src/preceptualai/xapp/dapp_engine.py`, `src/preceptualai/xapp/server.py` |
| Telecom integration and governance | `src/preceptualai/env/oran.py`, `src/preceptualai/xapp/e2_adapter.py`, `src/preceptualai/xapp/rapp_trainer.py` |

## Serving and Telecom Runtime Setup

If your goal is runtime serving or telecom-facing integration, you will likely need the base install plus O-RAN extras and runtime configuration. The repository’s serving surface is centered on the gRPC server and protocol definitions in `proto/`, while the operational stack extends into dApp execution, E2-facing integration, and lifecycle governance.[10] [11] [12] [13]

```bash
pip install -e ".[oran]"
python -m preceptualai.xapp.server --config xapp_config.yaml
```

This path should be treated as a later-stage setup track, not the first requirement for understanding the repository.

## Optional Real-Data and Deployment Paths

After the core setup is working, you can move into more advanced surfaces such as the real-data pipeline, low-latency runtime execution, O-RAN-facing control loops, and AI-RAN-oriented deployment pathways.[7] [10] [11] [12] [13]

These paths are meaningful, but they are also more environment-dependent than the basic local install. Some require additional datasets, integration targets, containers, or accelerators. That should be understood as a normal property of a deployment-oriented telecom intelligence stack rather than as a flaw in the repository.

## Common Setup Strategy

The best operational strategy is incremental. Do **not** try to activate every optional surface on day one. Install the base development environment, verify it, reproduce the benchmark suite, then add UHCI, serving, or telecom extras one layer at a time. This reduces ambiguity when something fails.

| If you are... | Start with |
|---|---|
| A new engineer | `.[dev]` |
| A benchmark-focused researcher | `.[dev,benchmarks]` |
| A telecom deployment engineer | Base install plus `.[oran]` |
| A UHCI researcher | Benchmark or base install first, then `.[uhci]` |

## Common Setup Problems

Most installation issues fall into a small number of categories: optional dependency gaps, Python version mismatch, missing runtime libraries, or external-infrastructure assumptions being treated as local prerequisites.

| Problem | Likely cause | Practical fix |
|---|---|---|
| Import errors during installation | Missing optional packages or stale virtual environment | Recreate the environment and reinstall from `pyproject.toml` |
| Benchmark scripts fail early | Dependency mismatch or missing extras | Reinstall with `.[dev,benchmarks]` |
| UHCI-specific exploration fails | Graph or research extras are missing | Add `.[uhci]` and review subsystem-specific assumptions |
| Deployment modules fail locally | Missing infrastructure such as containers, services, or RIC components | Treat deployment as a later-stage path, not the first setup step |

## Recommended Reading Order After Setup

Once your local environment works, follow the documents below in order. This sequence will help you move from “the code runs” to “I understand the full UHCI system.”

| Goal | Read next |
|---|---|
| Understand the big picture quickly | [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) |
| Understand the technical structure | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Understand interfaces and integration points | [`API_AND_INTERFACES.md`](API_AND_INTERFACES.md) |
| Understand empirical evidence and claim boundaries | [`TRAINING_AND_EVALUATION.md`](TRAINING_AND_EVALUATION.md) and [`DATA_AND_REPRODUCIBILITY.md`](DATA_AND_REPRODUCIBILITY.md) |
| Understand deployment and telecom runtime surfaces | [`DEPLOYMENT_AND_XAPP_GUIDE.md`](DEPLOYMENT_AND_XAPP_GUIDE.md) |

## References

[1]: [Project manifest in `pyproject.toml`](../pyproject.toml)
[2]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[3]: [Primary scripts directory](../scripts)
[4]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[5]: [Benchmark configuration in `benchmarks/benchmark_config.yaml`](../benchmarks/benchmark_config.yaml)
[6]: [Aggregated benchmark artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
[7]: [Provider registry in `src/preceptualai/env/provider_registry.py`](../src/preceptualai/env/provider_registry.py)
[8]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[9]: [System overview](SYSTEM_OVERVIEW.md)
[10]: [Inference protocol in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[11]: [Serving runtime in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[12]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[13]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
