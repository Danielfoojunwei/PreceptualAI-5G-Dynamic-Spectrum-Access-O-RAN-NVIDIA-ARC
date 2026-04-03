# Training and Evaluation

This document explains how **training, benchmarking, and runtime validation** work when the repository is read correctly as a **UHCI-first system**. In this codebase, **Universal Heterogeneous Connectivity Intelligence (UHCI)** is the main architectural object: a stack that combines provider-aware world modeling, propagation-aware simulation, heterogeneous state construction, structural and temporal learning, learned decision policies, and deployment-facing runtime surfaces.[1] [2] [3] [4] [5] [6] [7]

The empirical story should therefore be read in layers. The repository contains one directly aggregated benchmark suite with saved summary artifacts, a broader UHCI training lane built around heterogeneous-provider reasoning and richer data pathways, and a deployment-oriented evaluation lane focused on latency and operational execution behavior.[1] [2] [3] [4] [5] [6] [7] The correct interpretation is not that only the benchmark workflow matters. The correct interpretation is that **the broader UHCI system is primary, and the included benchmark suite is the clearest currently summarized quantitative evidence surface inside that larger architecture**.

| Empirical surface | What it evaluates | Evidence maturity in the repository |
|---|---|---|
| Included benchmark suite | Controlled agent-to-agent performance under the included benchmark harness | Highest directly aggregated maturity |
| UHCI training lane | Heterogeneous-provider learning, real-data-aware pathways, and broader connectivity intelligence workflows | Strong implementation support with lighter aggregated reporting |
| Runtime and deployment lane | Inference latency, serving behavior, and operational execution characteristics | Code-backed and deployment-relevant |

## How to Read the Evidence Story

A serious reader should separate **architectural scope** from **benchmark scope**. The architecture now includes provider taxonomies, ITU and FR3 propagation logic, unified heterogeneous environments, graph-structured encoders, CfC-capable temporal backends, universal-agent abstractions, RT-oriented inference, O-RAN surfaces, and lifecycle governance.[3] [4] [5] [6] [7] However, the strongest repository-wide numeric comparison that is already summarized into a common artifact remains the included benchmark suite implemented in `benchmarks/benchmark.py` and reported in `benchmark_summary.json`.[1] [2]

> The safest empirical statement is that **UHCI is the repository’s primary system architecture, while the included benchmark suite is the strongest current quantitative anchor for measured performance claims**.

| Question | Strongest source of truth |
|---|---|
| Which results are directly summarized across matched training and evaluation runs? | `benchmarks/benchmark.py` and `benchmark_summary.json` |
| Where does the broader UHCI training path begin? | `scripts/train_uhci.py`, `unified_connectivity_env.py`, and `data_pipeline.py` |
| Where are deployment-latency behaviors evaluated? | `dapp_engine.py` and runtime-serving modules |
| What claims are safe to make publicly? | Benchmark artifacts, configuration files, and reproducibility documentation |

## UHCI Training Stack

The repository’s primary training story now lives in the **UHCI stack**. `scripts/train_uhci.py` is the clearest training entry point for this broader architecture. It exposes a training workflow oriented toward **heterogeneous connectivity intelligence**, including richer configuration surfaces, real-data-aware options, and domain-specific controls that go beyond a narrowly simplified control formulation.[3]

This UHCI training lane is supported by `unified_connectivity_env.py`, which constructs a provider-aware decision environment rather than a single undifferentiated control surface, and by `data_pipeline.py`, which introduces external-data ingestion pathways for more realistic workflows.[4] [5] When read together with the provider registry and core agent modules, these files show that the repository is architected to train over a **broader decision problem involving provider identity, propagation regime, and heterogeneous connectivity state**, not only over a simplified channel-choice benchmark.[3] [4] [5]

| UHCI training component | Role in the empirical workflow |
|---|---|
| `scripts/train_uhci.py` | Main entry point for advanced UHCI experiments |
| `src/preceptualai/env/unified_connectivity_env.py` | Builds heterogeneous-provider state, action, and reward structure |
| `src/preceptualai/env/data_pipeline.py` | Connects synthetic and external-data workflows into the environment |
| `src/preceptualai/env/provider_registry.py` | Supplies provider taxonomy and system priors used in UHCI workflows |
| `src/preceptualai/core/universal_spectrum_agent.py` | Provides a broader agent abstraction aligned with the UHCI stack |
| `src/preceptualai/core/hetero_gnn_encoder.py` | Supplies structural state encoding for heterogeneous entities |
| `src/preceptualai/core/ltc_cell_cfc.py` | Supplies CfC-capable temporal reasoning inside the broader control stack |

The right empirical claim boundary here is careful but strong. The code directly supports the claim that the repository contains a serious **UHCI experimentation and training framework**. It does **not** by itself prove that every UHCI subsystem already has the same aggregated benchmark maturity as the included benchmark suite.[1] [2] [3] [4] [5]

## Included Benchmark Suite Inside UHCI

The repository also contains a directly summarized benchmark suite implemented in `benchmarks/benchmark.py`.[1] This suite remains important because it provides the clearest shared numeric evidence surface in the repository. It standardizes environment construction, agent instantiation, training loops, deterministic evaluation, and multi-seed aggregation into a single result summary artifact.[1] [2]

Within a UHCI-first reading, this benchmark suite should be understood as **one measured subsystem inside the broader architecture**. It is not the whole identity of the repository, but it is still the strongest current source for bounded quantitative claims about canonical control metrics such as success, collisions, and spectral efficiency.[1] [2]

| Benchmark responsibility | Repository anchor | Why it matters |
|---|---|---|
| Agent creation | `make_agent()` in `benchmarks/benchmark.py` | Keeps the included agent comparisons structured and matched |
| Off-policy training | `train_off_policy()` | Covers SAC-family agents in a common evaluation loop |
| PPO training | `train_ppo()` | Preserves an on-policy comparison path |
| Deterministic evaluation | `evaluate_agent()` | Produces consistent held-out metrics |
| Aggregated reporting | `benchmark_summary.json` | Defines the strongest public claim boundary |

## Canonical Metrics and What They Mean

One of the strengths of the repository is that the benchmark harness makes its metric semantics explicit. That matters because telecom-control systems are multi-objective systems. A model can look strong on reward while being weaker on operationally important metrics such as collisions or successful transmissions. The documentation therefore treats the benchmark metrics as describing **different dimensions of control quality**, not as redundant versions of the same outcome.[1]

| Metric | Meaning in the repository | Why it matters |
|---|---|---|
| `mean_reward` | Average episode reward under the defined objective | Useful for optimization tracking, but not sufficient by itself |
| `success_rate` | Successful transmissions divided by total steps | Strong operational indicator of connectivity quality |
| `collision_rate` | Collisions divided by total steps | Safety and coexistence indicator; lower is better |
| `spectral_efficiency` | Mean throughput-oriented utilization signal | Measures how effectively radio opportunities are used |
| `jain_fairness` | Jain fairness index over throughput allocation | Captures balance and resource-allocation equity |
| `mean_inference_ms` | Mean per-step inference time | Practical deployment-latency indicator |
| `p99_inference_ms` | Tail latency under evaluation | Important for near-real-time reliability |

The most defensible interpretation is that **success rate, collision rate, and spectral efficiency** deserve special emphasis for telecom-control credibility because they map more directly to transmission quality and coexistence performance than reward alone.[1] [2]

## Verified Measured Results in the Included Benchmark Suite

The saved benchmark summary supports several strong, but carefully bounded, claims. Within the included benchmark suite, the reported leading results on **success rate**, **lowest collision rate**, and **spectral efficiency** come from the strongest temporal-control path in that evaluation. At the same time, the suite does **not** show universal leadership on every metric: another included model leads on mean reward, and another model slightly edges the fairness metric. The documentation should therefore speak precisely rather than overclaiming.[2]

| Metric in the included benchmark suite | Reported result | Safe interpretation |
|---|---:|---|
| **Success rate** | **0.6337 ± 0.0044** | Best reported result in the included benchmark suite on successful transmissions.[2] |
| **Collision rate** | **0.3663 ± 0.0044** | Best reported result in the included benchmark suite on lowest collision rate.[2] |
| **Spectral efficiency** | **0.6337 ± 0.0044** | Best reported result in the included benchmark suite on spectral efficiency.[2] |
| **Mean reward** | Higher value reported for another included model | The repository should not claim universal reward leadership from this artifact alone.[2] |
| **Jain fairness** | Near-leading but not first in the included evaluation | Fairness claims should be carefully bounded and precise.[2] |
| **Mean inference latency** | Real-time-capable in the included benchmark framing | The artifact supports low-latency viability, but not universal fastest-model claims.[2] |

This is the right way to communicate performance claims in a serious UHCI documentation set. The repository can credibly state that it contains **measured leadership on important canonical control metrics within the included benchmark suite**, while also stating that the broader UHCI architecture is larger than the current single aggregated benchmark story.[1] [2]

## Why Operational Leadership and Reward Leadership Can Diverge

The included results make an important methodological point. In connectivity control, the model with the highest optimization reward is not automatically the model with the best operational behavior. A controller can obtain a strong reward score while still causing more collisions or producing a weaker success rate than another model. In telecom contexts, **fewer collisions and more reliable successful transmissions** often matter more than isolated reward leadership.[1] [2]

> In a wireless-control system, the most important model is often the one that best balances **success, efficiency, coexistence safety, and latency**, not merely the one that maximizes one reward number.

This is why the repository’s documentation emphasizes **canonical control metrics** rather than collapsing the empirical story into a single headline reward figure.[1] [2]

## Runtime and Deployment Evaluation

Training quality is not the only evaluation surface in this repository. The deployment path introduces another form of evidence: **latency and operational execution quality**. `src/preceptualai/xapp/dapp_engine.py` implements a low-latency inference engine designed for RT-oriented execution with persistent temporal state and accelerator-aware optimizations such as quantization and graph-capture-oriented acceleration paths.[6]

That module includes runtime benchmarking utilities that measure latency statistics such as **mean**, **median**, **p99**, and **minimum** inference time. This is architecturally significant because it shows that the repository treats runtime feasibility as something measurable rather than something assumed.[6]

`src/preceptualai/xapp/server.py` complements this by exposing the model through a serving interface with prediction, streaming prediction, health, and metrics endpoints, which broadens evaluation from training curves to live-service behavior.[7]

| Runtime evaluation surface | What is measured or exposed |
|---|---|
| `src/preceptualai/xapp/dapp_engine.py` | Low-latency inference behavior and latency-distribution statistics |
| `src/preceptualai/xapp/server.py` | Prediction serving, streaming, health checks, and metrics surfaces |
| Comparative benchmark summary | Per-step inference latency for the included evaluated models |

Taken together, these files show that UHCI evaluation spans both **learning performance** and **deployment readiness signals**.[2] [6] [7]

## Reproducible Workflows

For a reader who wants to recreate the empirical story responsibly, the safest sequence is to begin with the included benchmark suite, then move to the broader UHCI training surface, and only after that inspect runtime or deployment-oriented execution paths. This sequencing is useful because it lets the user first validate the repository’s most mature shared numeric evidence before adding optional dependencies, richer data pathways, and lower-latency deployment complexity.[1] [2] [3] [5]

| Goal | Recommended command path |
|---|---|
| Quick benchmark-suite validation | `python scripts/train.py --num_steps 5000 --output_dir results/quickstart` then `python scripts/evaluate.py --checkpoint results/quickstart/checkpoint_final.pt` |
| Full benchmark-suite reproduction | `pip install -e ".[dev,benchmarks]"` then `python benchmarks/run_full_benchmark.py --config benchmarks/benchmark_config.yaml` |
| UHCI exploration | Inspect `python scripts/train_uhci.py --help`, then run a controlled UHCI experiment after benchmark-suite validation |
| Runtime-latency inspection | Review or instrument the dApp inference path and serving stack |

The benchmark configuration file remains the main reproducibility anchor for the included benchmark suite because it defines seeds, training schedules, evaluation episodes, environment shape, and per-agent hyperparameters.[8]

## What the Current Evidence Does Not Prove

The documentation should also be explicit about what the repository’s current evidence does **not** prove. The included benchmark suite does not establish literature-wide state of the art across all external methods, all datasets, or all deployment settings. It also does not prove that every UHCI subsystem has already been benchmarked end to end at the same maturity level as the directly summarized benchmark suite.[2]

| Claim type | Supported directly by current repository evidence? |
|---|---|
| Best reported result in the included benchmark on success, collisions, and spectral efficiency | Yes |
| Universally best-performing system across all metrics and all settings | No |
| Literature-wide state of the art across all external methods | No |
| Existence of a substantial UHCI experimentation stack | Yes |
| Existence of deployment-oriented latency evaluation surfaces | Yes |
| Full end-to-end UHCI benchmark maturity equal to the included benchmark suite | No |

## Recommended Interpretation for Readers

A lay reader should come away with a simple and accurate conclusion: the repository contains **real measured improvements on important canonical control metrics in its included benchmark suite**, and it also contains a much broader **UHCI architecture** designed for heterogeneous connectivity intelligence.[1] [2] [3] [4] [5]

A technical reader should come away with a more precise conclusion: the repository’s **empirical maturity is layered**. The strongest aggregated benchmark evidence lives in the included benchmark suite, while the broader UHCI environment, provider, agent, runtime, and deployment surfaces provide the code-backed architecture for a next-generation connectivity-intelligence system whose full end-to-end empirical story is broader than the current one-summary benchmark artifact.[1] [2] [3] [4] [5] [6] [7]

## References

[1]: [Benchmark harness in `benchmarks/benchmark.py`](../benchmarks/benchmark.py)
[2]: [Aggregated benchmark artifact in `benchmarks/results/benchmark_results_full/benchmark_summary.json`](../benchmarks/results/benchmark_results_full/benchmark_summary.json)
[3]: [UHCI training entry point in `scripts/train_uhci.py`](../scripts/train_uhci.py)
[4]: [Unified connectivity environment in `src/preceptualai/env/unified_connectivity_env.py`](../src/preceptualai/env/unified_connectivity_env.py)
[5]: [Unified data pipeline in `src/preceptualai/env/data_pipeline.py`](../src/preceptualai/env/data_pipeline.py)
[6]: [RT-RIC dApp engine in `src/preceptualai/xapp/dapp_engine.py`](../src/preceptualai/xapp/dapp_engine.py)
[7]: [gRPC serving module in `src/preceptualai/xapp/server.py`](../src/preceptualai/xapp/server.py)
[8]: [Benchmark configuration in `benchmarks/benchmark_config.yaml`](../benchmarks/benchmark_config.yaml)
