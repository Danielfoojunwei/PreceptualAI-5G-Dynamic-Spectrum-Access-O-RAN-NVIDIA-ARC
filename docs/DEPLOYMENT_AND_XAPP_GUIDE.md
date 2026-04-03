# Deployment and xApp Guide

This guide explains how the repository moves from trained UHCI artifacts to a **deployable connectivity-intelligence runtime**. The key point is that deployment in this codebase is not treated as a thin wrapper around one research model. The implementation includes a layered operational path spanning **model export, runtime abstraction, low-latency execution, serving interfaces, O-RAN-facing control integration, Non-RT lifecycle governance, and AI-RAN-oriented packaging**.[1] [2] [3] [4] [5] [6] [7] [8] [9]

A correct UHCI-first reading changes the interpretation of the deployment layer. The repository is not merely exporting a narrow single-agent controller behind a generic API. It is implementing a broader **Universal Heterogeneous Connectivity Intelligence deployment stack** in which trained policies can be served, accelerated, monitored, integrated into RIC-style software loops, governed over time, and adapted to telecom-oriented accelerated infrastructure.[2] [3] [4] [6] [7] [8] [9]

| Deployment layer | Role inside UHCI | Principal repository anchors |
|---|---|---|
| Model export | Converts trained artifacts into portable runtime assets | `src/preceptualai/export/`, `scripts/export_model.py` |
| Runtime abstraction | Loads and runs trained policies consistently across execution contexts | `src/preceptualai/xapp/inference_engine.py` |
| Low-latency execution | Provides RT-oriented inference behavior with persistent state | `src/preceptualai/xapp/dapp_engine.py` |
| Service serving | Exposes prediction, streaming, health, and metrics surfaces | `src/preceptualai/xapp/server.py`, `proto/preceptualai.proto` |
| O-RAN loop integration | Connects runtime decisions to telecom control-loop semantics | `src/preceptualai/env/oran.py`, `src/preceptualai/xapp/e2_adapter.py` |
| Lifecycle governance | Handles versioning, approval, retraining, degradation, and policy output | `src/preceptualai/xapp/rapp_trainer.py` |
| AI-RAN packaging | Bridges execution to NVIDIA Aerial and ARC-oriented deployment paths | `src/preceptualai/xapp/aerial_adapter.py`, `docker/Dockerfile.aerial` |

## Deployment Philosophy

The repository’s deployment philosophy is that an intelligence system is only useful if it can cross the boundary from training code to operational software. That is why the codebase includes protocol definitions, runtime configuration, gRPC serving, low-latency inference paths, O-RAN-facing adapters, container assets, and governance surfaces rather than stopping at checkpoint generation.[1] [2] [3] [4] [5] [6] [7] [8]

> The deployment stack treats UHCI as a **living control service**, not just as a trained model file.

This is particularly important in telecom settings, where a controller must fit into software-defined control loops, satisfy latency expectations, expose observable service health, and support policy and lifecycle management rather than one-off offline inference.[3] [4] [6] [7] [8]

## Export Flow

The export path is the first bridge from training to runtime. The repository includes an export package under `src/preceptualai/export/` and a user-facing script in `scripts/export_model.py`.[1] The documented path centers on ONNX export, which is a sensible deployment boundary because it turns a learned policy into a more portable runtime artifact suitable for serving and accelerator-aware execution.[1]

```bash
python scripts/export_model.py \
  --checkpoint results/quickstart/checkpoint_final.pt \
  --output models/preceptualai_actor.onnx
```

In a UHCI-first framing, export is not just about freezing one trained model. It is the mechanism that makes the broader connectivity-intelligence stack runnable inside serving, dApp, xApp, and AI-RAN-oriented execution surfaces.[1] [2] [3] [4]

| Export concern | Why it matters |
|---|---|
| Portable model format | Reduces coupling between the training stack and runtime host |
| Reproducible deployment artifact | Makes serving and performance testing more consistent |
| Runtime compatibility | Enables use in xApp, dApp, and accelerator-facing pathways |

## Runtime Abstraction

`src/preceptualai/xapp/inference_engine.py` is the repository’s main runtime abstraction for loading models and performing inference consistently across deployment contexts.[2] This file matters because it separates core model execution from higher-level serving, control-plane, and orchestration concerns. In architectural terms, it acts as the **runtime kernel** of the deployment stack.

A useful mental model is that the inference engine provides the shared execution layer on top of which several deployment surfaces are built. That design is consistent with the rest of the repository, which supports both general service-style inference and tighter low-latency execution paths.[2] [3] [4]

## Low-Latency dApp Path

The most deployment-specific runtime module is `src/preceptualai/xapp/dapp_engine.py`. This file implements a low-latency inference path designed for **RT-oriented execution**, including persistent temporal state and accelerator-aware options such as **INT8 quantization** and **CUDA graph capture**.[3]

The module also includes benchmarking utilities that report **mean**, **median**, **p99`, and minimum latency statistics. That is architecturally important because it shows that runtime feasibility is treated as a measurable property inside the code rather than as a vague aspiration.[3]

| dApp capability | Operational significance |
|---|---|
| Persistent hidden state | Supports sequential and continuous-time control across repeated inference steps |
| INT8 quantization option | Reduces inference cost in constrained deployment settings |
| CUDA graph option | Minimizes per-inference overhead on supported accelerators |
| Latency benchmarking | Provides measurable evidence for runtime suitability |

In the UHCI architecture, this module is one of the clearest signals that the repository is trying to close the gap between research models and near-real-time control execution.[3]

## gRPC Serving Surface

`src/preceptualai/xapp/server.py` exposes the intelligence layer as a gRPC service, while `proto/preceptualai.proto` defines the corresponding protocol contract.[4] [5] Together, these files formalize the repository’s service boundary and make integration assumptions explicit.

The service supports prediction, streaming prediction, health reporting, and metrics exposure. That matters because it means the deployment layer is not just a local script interface. It has a real software boundary that external services or orchestration layers can call in a structured way.[4] [5]

| Service surface | Purpose |
|---|---|
| `Predict` | Unary inference request-response path |
| Streaming prediction | Sequential inference surface for repeated state updates |
| Health endpoint | Lets operators and integrators monitor readiness |
| Metrics endpoint | Supports observability and runtime inspection |

This is one of the reasons the repository should not be interpreted as notebook-scale experimentation. The explicit service contract means the deployment layer is built as a callable control component rather than as a one-off local model wrapper.[4] [5]

## xApp, E2, and O-RAN Integration

The deployment story extends beyond generic serving because the repository also contains explicit **O-RAN-facing integration logic**. `src/preceptualai/env/oran.py` models a live O-RAN control context, while `src/preceptualai/xapp/e2_adapter.py` handles lower-level E2-style subscription and control interactions.[6] [7]

This changes the meaning of the xApp framing in the repository. The code is not using the term as marketing language. It is modeling a software component that consumes measurements, maintains state, and emits decisions in a way that is conceptually aligned with a **Near-RT RIC control application**.[6] [7] [10]

> In this repository, xApp framing means the intelligence is designed to live inside a **telecom control loop**, not merely behind a generic inference API.

| O-RAN-facing component | What it contributes |
|---|---|
| `oran.py` | Provides an environment and control-loop abstraction tied to O-RAN-style operation |
| `e2_adapter.py` | Encodes measurement subscription and control-message behavior near the E2 boundary |
| `server.py` | Exposes callable runtime surfaces for integration into larger systems |

A careful reader should still note the deployment boundary. The repository contains the intelligence-side logic and integration scaffolding, but a full production operator deployment would still require external RIC infrastructure, policy distribution, networking, security, and operations systems that are not bundled here.[4] [6] [7] [10]

## Non-RT RIC and Lifecycle Governance

A particularly important part of the deployment architecture is `src/preceptualai/xapp/rapp_trainer.py`, which models the **Non-RT RIC or SMO lifecycle layer**.[8] This file goes beyond serving and inference by tracking model versions, approval status, deployment state, online degradation, retraining triggers, and policy generation for downstream deployment.[8]

The module also produces policy artifacts aligned with telecom control-governance thinking, which shows that the repository explicitly considers how a learned controller is governed over time rather than only how it is initially served.[8]

| Lifecycle function | Why it matters |
|---|---|
| Model cataloging | Makes versioned deployment explicit |
| Approval flow | Separates trained models from deployable models |
| Online degradation monitoring | Supports safe retraining and rollback logic |
| Policy generation | Bridges model state into control-plane-consumable artifacts |

This governance layer is one of the strongest reasons to describe the repository as a **system architecture** rather than as a narrow machine-learning experiment.[8]

## AI-RAN and NVIDIA Aerial Packaging

The repository also includes a deployment path aimed at **AI-RAN-style accelerated infrastructure**. `src/preceptualai/xapp/aerial_adapter.py` packages models for NVIDIA ARC profiles and Aerial-oriented assumptions, while `docker/Dockerfile.aerial` provides a containerization surface aligned with that deployment path.[9] [11]

This matters because AI-RAN changes the question from “can the model run?” to “can the model run where telecom intelligence and radio workloads are increasingly converging?” NVIDIA’s public AI-RAN framing emphasizes shared accelerated infrastructure for AI and radio functions, which makes this packaging surface strategically relevant.[12]

| AI-RAN deployment asset | Role |
|---|---|
| `aerial_adapter.py` | Maps trained execution into NVIDIA Aerial and ARC-oriented deployment profiles |
| `docker/Dockerfile.aerial` | Container base for Aerial-oriented or accelerator-specific deployment |
| Main runtime container | Generic serving path for non-Aerial deployment experiments |

A careful documentation posture should distinguish between **repository support for AI-RAN-oriented packaging** and a claim of complete field deployment readiness. The repository clearly implements the former. The latter depends on external infrastructure and operator context beyond what is contained in the repository.[9] [11] [12]

## Configuration and Containerization

The runtime configuration file `xapp_config.yaml` captures key deployment parameters for the serving layer, while the Docker assets package the runtime stack into reproducible containers.[4] [11] The repository also includes a federated compose example in `docker/docker-compose.fl.yaml`, which is useful for understanding how multiple clients and an aggregation process can be run together in a distributed learning workflow.[11] [13]

| Asset | Best use |
|---|---|
| `xapp_config.yaml` | Runtime configuration for serving and xApp-style deployment |
| `docker/Dockerfile` | General inference-service container |
| `docker/Dockerfile.aerial` | NVIDIA Aerial-oriented container path |
| `docker/docker-compose.fl.yaml` | Multi-client federated experimentation |

## Practical Deployment Sequence

A sensible deployment sequence starts with model export and service validation before moving to low-latency or O-RAN-facing integration. This reduces risk because it separates artifact correctness, service behavior, latency behavior, and control-plane integration into manageable stages.

| Stage | Recommended focus |
|---|---|
| 1 | Export a trained model and verify the artifact loads correctly |
| 2 | Start the gRPC serving surface and validate prediction, health, and metrics flows |
| 3 | Benchmark the low-latency dApp path if near-real-time execution matters |
| 4 | Integrate with O-RAN-facing telemetry and E2 control surfaces |
| 5 | Add lifecycle governance, policy generation, and retraining logic |
| 6 | Package for AI-RAN or accelerator-specific deployment if needed |

## Deployment Boundaries and Safe Claims

The repository’s deployment stack is substantial, but the documentation should still be explicit about what the code does and does not prove on its own.

| Claim | Safe? | Why |
|---|---|---|
| The repository contains export and serving pathways | Yes | Export helpers, service contracts, and runtime code are explicit |
| The repository exposes a gRPC inference contract | Yes | The protobuf and server implementation define this clearly |
| The repository supports xApp and dApp-oriented deployment thinking | Yes | The code models both general serving and lower-latency execution |
| The repository includes O-RAN-facing control-loop integration scaffolding | Yes | `oran.py` and `e2_adapter.py` make that visible |
| The repository includes lifecycle governance for deployed models | Yes | `rapp_trainer.py` explicitly implements this surface |
| The repository supports AI-RAN-oriented packaging | Yes | `aerial_adapter.py` and Aerial container assets are included |
| The repository alone constitutes a full production operator deployment | No | External infrastructure, networking, policy routing, security, and operations remain outside the repository |

## Recommended Reading

A reader who wants to understand deployment in the context of the full UHCI stack should continue in the following order.

| If you want to understand... | Read next |
|---|---|
| The full technical system behind deployment | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| The interfaces exposed to integrators | [`API_AND_INTERFACES.md`](API_AND_INTERFACES.md) |
| How the system is configured and run locally | [`SETUP_AND_QUICKSTART.md`](SETUP_AND_QUICKSTART.md) |
| How lifecycle and empirical boundaries work | [`DATA_AND_REPRODUCIBILITY.md`](DATA_AND_REPRODUCIBILITY.md) |

## References

[1]: [Export package in `src/preceptualai/export/`](../src/preceptualai/export)
[2]: [Runtime inference abstraction in `src/preceptualai/xapp/inference_engine.py`](../src/preceptualai/xapp/inference_engine.py)
[3]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[4]: [Serving runtime in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[5]: [Inference protobuf in `proto/preceptualai.proto`](../proto/preceptualai.proto)
[6]: [O-RAN environment module in `src/preceptualai/env/oran.py`](../src/preceptualai/env/oran.py)
[7]: [E2 adapter in `src/preceptualai/xapp/e2_adapter.py`](../src/preceptualai/xapp/e2_adapter.py)
[8]: [Non-RT RIC rApp training service in `src/preceptualai/xapp/rapp_trainer.py`](../src/preceptualai/xapp/rapp_trainer.py)
[9]: [NVIDIA Aerial adapter in `src/preceptualai/xapp/aerial_adapter.py`](../src/preceptualai/xapp/aerial_adapter.py)
[10]: [O-RAN Software Community documentation](https://docs.o-ran-sc.org/en/k-release/)
[11]: [Aerial-oriented container in `docker/Dockerfile.aerial`](../docker/Dockerfile.aerial)
[12]: [NVIDIA, "AI-RAN Solutions for 5G & 6G Cellular Networks"](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)
[13]: [Federated compose stack in `docker/docker-compose.fl.yaml`](../docker/docker-compose.fl.yaml)
