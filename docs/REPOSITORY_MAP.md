# Repository Map

This document provides a directory-by-directory mental model of the repository. It is written for readers who want to move through the codebase without guessing what each folder is for. The goal is not only orientation. It is to let a new contributor or reviewer infer **which files define the primary UHCI system, which files define interfaces, and which files represent benchmark, deployment, or specialized research surfaces**.[1]

| Path | What you should think when you see it |
|---|---|
| `src/preceptualai/` | The Python package that implements the system |
| `scripts/` | Human-facing entry points for common workflows |
| `benchmarks/` | Controlled empirical comparison harnesses and outputs |
| `proto/` | Explicit service contracts |
| `docker/` | Packaging and runtime deployment artifacts |
| `tests/` | Validation surfaces for stable and advanced components |
| `docs/` | Repository-wide explanatory documentation |

## Top Level

The top level of the repository contains the packaging manifest, runtime configuration, demonstration entry points, and the documentation front door. `pyproject.toml` is especially important because it encodes the dependency groups and validation boundaries that determine how the code should be approached operationally.[2]

| File | Why it matters |
|---|---|
| `README.md` | Main narrative and entry point for the repository |
| `pyproject.toml` | Dependencies, optional extras, lint/type-check scope |
| `xapp_config.yaml` | Runtime configuration for serving workflows |
| `demo.py` | High-level demonstration pathway |

## `src/preceptualai/`

The source package is the heart of the codebase. It contains the primary UHCI stack together with benchmark, deployment, and specialized research modules. A useful reading order is `env` → `core` → `xapp` → `agent` / `export` / `federated`.

| Subpackage | What it contains |
|---|---|
| `core/` | Temporal building blocks, actor/critic logic, replay infrastructure, advanced encoders |
| `agent/` | Trainable agent implementations, benchmarked policies, and supporting learning variants |
| `env/` | Simulation, real-data, O-RAN, AODT, NTN, UHCI, and provider abstractions |
| `export/` | Model-export helpers for serving and deployment |
| `federated/` | Distributed-learning components |
| `xapp/` | Inference-serving and runtime logic |

## `scripts/`

The `scripts/` directory is where a human operator usually starts interacting with the repository. These scripts encapsulate the workflows that would otherwise require importing modules manually. Conceptually, the directory contains UHCI training, evaluation, export, benchmarking support, and deployment-adjacent entry points.[1]

| Script category | Typical purpose |
|---|---|
| Training scripts | Fit UHCI-related models and benchmark workflows |
| Evaluation scripts | Score saved checkpoints or run deterministic assessment |
| Export scripts | Convert trained artifacts into serving-friendly formats |
| Client/runtime scripts | Exercise inference or distributed-learning endpoints |

## `benchmarks/`

The benchmark directory holds the empirical evaluation machinery. It includes the general harness, configuration files, standalone runners, benchmarked model implementations, visualization helpers, and saved output artifacts. This is where you go when you want to understand what the repository means by “improvement” and how those numbers are computed.[3]

| Benchmark file | Role |
|---|---|
| `benchmark.py` | General benchmark harness and metric definitions |
| `benchmark_config.yaml` | Matched experiment configuration |
| `run_full_benchmark.py` | Convenience runner for a fixed experimental sweep |
| `visualize.py` | Figure and report generation |
| `results/` | Saved numerical outputs and figures |

## `proto/`

The `proto/` directory formalizes the system’s external interfaces. These are not implementation details. They are the contract layer for prediction and federated-learning flows. A reviewer who wants to understand integration boundaries should inspect this directory early.[4]

| File | Interface meaning |
|---|---|
| `preceptualai.proto` | Inference requests, responses, health, and metrics |
| `preceptualai_fl.proto` | Registration, model transfer, round status, and training telemetry |

## `docker/`

The repository’s container assets are where the system moves closer to deployment reality. The main Dockerfile packages the serving runtime, while additional files support development, federated-learning stacks, or Aerial-oriented environments.[5]

| Asset | Deployment role |
|---|---|
| `Dockerfile` | Main runtime container |
| `Dockerfile.aerial` | NVIDIA Aerial-oriented build path |
| `docker-compose.fl.yaml` | Multi-client federated deployment example |

## `tests/`

The tests directory contains both default-installation and advanced test surfaces. Not every file is equally suitable as a first validation target. The repository manifest makes that clear by constraining the default validation scopes.[2] A sensible reader should therefore treat the tests as a layered set rather than a flat list.

| Test posture | Meaning |
|---|---|
| Default-installation tests | Best place to validate a local installation |
| Advanced UHCI or GPU-heavy tests | Useful for deeper development, but may require extra dependencies |
| Integration-style tests | Closer to system workflows than to pure unit isolation |

## `docs/`

The documentation directory is intended to make the repository independently understandable. It is where lay explanation, architecture description, reproducibility guidance, interface references, and troubleshooting have been separated into reader-friendly layers instead of one monolithic file.

## Suggested Exploration Order

A new technical reader should begin with the README, then inspect the manifest, then move through the core architecture before looking at interfaces or deployment surfaces. This order prevents confusion caused by the repository’s breadth.

| Goal | Good path |
|---|---|
| Understand the primary UHCI stack | `README.md` → `src/preceptualai/env/provider_registry.py` → `src/preceptualai/env/unified_connectivity_env.py` → `src/preceptualai/core/universal_spectrum_agent.py` → `src/preceptualai/xapp/` |
| Understand measurement and claims | `benchmarks/benchmark.py` → benchmark artifacts → `docs/DATA_AND_REPRODUCIBILITY.md` |
| Understand service boundaries | `proto/` → `src/preceptualai/xapp/` → `docker/` |
| Understand the broader experimental surface | `scripts/train_uhci.py` → `src/preceptualai/core/hetero_gnn_encoder.py` → `src/preceptualai/core/ltc_cell_cfc.py` → benchmark and runtime docs |

## References

[1]: [Documentation index in `docs/DOCUMENTATION_INDEX.md`](./DOCUMENTATION_INDEX.md)
[2]: [Project manifest in `pyproject.toml`](../pyproject.toml)
[3]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[4]: [Inference protobuf in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[5]: [Main Docker runtime in `docker/Dockerfile`](../docker/Dockerfile)
