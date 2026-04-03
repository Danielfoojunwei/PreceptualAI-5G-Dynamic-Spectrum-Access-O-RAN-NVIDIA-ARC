# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-03

### Added — UHCI (Universal Heterogeneous Connectivity Intelligence)

- Universal Heterogeneous Connectivity Intelligence (UHCI) paradigm — unified AI for all connectivity providers
- ProviderRegistry with 8 provider types: LEO, MEO, GEO, HAPS (NTN) + FR1, FR3, ISAC, WiFi7 (terrestrial)
- 3GPP-compliant physics parameters for each provider (altitude, freq bands, path loss, Doppler, RTT, etc.)
- HeteroGNNEncoder with type-specific node projections and edge-conditioned multi-head attention
- HeteroGNNTemporalEncoder combining spatial GNN with temporal CfC/Mamba/LTC backends
- UniversalSpectrumAgent with KAN actor for interpretable, regulatory-compliant policies
- UnifiedConnectivityEnv — vectorized Gymnasium environment for all 8 providers simultaneously
- KAN (Kolmogorov-Arnold Network) actor with B-spline basis functions and spline audit for regulatory compliance
- Closed-form Continuous-time (CfC) cell — analytical solution, no ODE stepping needed
- Mamba encoder — selective state space model with linear O(L) temporal complexity
- LTC-GPU encoder — fused CUDA kernel for GPU-resident ODE integration
- Multi-scale LTC cell for heterogeneous channel decorrelation timescales
- DiffusionAugmenter (DDPM) for synthetic spectrum data generation
- FNO (Fourier Neural Operator) channel surrogate with physics-constrained loss
- SmODE power control neuron — Lipschitz-constrained for 3GPP-compliant power ramps
- LTC World Model + Dyna trainer for model-based imagined rollouts
- GPU-resident replay buffer (zero CPU-GPU copies during training)
- Prioritized Experience Replay with sum-tree
- Curriculum learning scheduler for progressive environment difficulty
- Real data pipeline supporting UCC MISL, Colosseum, and TelecomTS (HuggingFace)
- Satellite environment with Keplerian orbital dynamics and Doppler computation
- ITU-R P.618-13 Earth-space propagation model
- 3GPP TR 38.901 FR3 (7-24 GHz) channel model
- 6G ISAC dual-function (sensing + communication) environment
- NVIDIA Aerial adapter for GPU-accelerated L1/L2 integration
- dApp engine for direct L1 PHY access on NVIDIA ARC
- rApp trainer for Non-RT RIC model lifecycle management
- Hybrid discrete (channel selection) + continuous (power control) actor
- GPU-vectorized simulation environment (sim_vectorized.py)
- UHCI training script (scripts/train_uhci.py) with all feature flags
- UHCI test suite (4 test modules, 2172 lines)

### Changed

- Package renamed from `spectrai` to `preceptualai`
- Version bumped to 0.2.0
- All imports updated to preceptualai namespace
- Protobuf files renamed to preceptualai_pb2.py / preceptualai_pb2_grpc.py
- README updated for Preceptual.ai branding

## [0.1.0] - 2026-03-15

### Added — SAC-LTC DSA Engine

- Liquid Time-Constant (LTC) neural ODE cell with input-dependent time constants
- Multi-layer LTC encoder producing fixed-dimensional latent vectors
- Soft Actor-Critic (discrete) with automatic entropy tuning
- Twin Q-networks with Polyak target averaging
- 1M-step replay buffer
- Simulated DSA environment (10 channels, Markov PU dynamics P_on=0.3, P_off=0.5)
- Real 5G environment (UCC MISL dataset, 188K measurements, 83 Irish operator traces)
- O-RAN environment with E2 interface integration
- NVIDIA AODT environment
- ONNX export pipeline (opset 17)
- TensorRT compilation (fp32/fp16/int8)
- Async gRPC inference server (port 50051)
- Protobuf service definitions (inference + federated learning)
- Hybrid Federated Learning with structural/tau weight separation
- FL aggregator with FedAvg + personalized tau_mix_ratio blending
- FL client for edge device training
- Cold-start elimination via global model registration
- Prometheus metrics collection (counters, histograms, gauges)
- Structured JSON logging with structlog
- Pydantic-validated configuration schema
- Docker images (production, development, NVIDIA Aerial)
- Docker Compose for single-server and FL stack deployment
- GitHub Actions CI (ruff lint, mypy type-check, pytest with coverage)
- 74 pytest tests across 11 modules
- Reproducible benchmark suite: SAC-LTC vs SAC-LSTM vs SAC-LFM vs PPO-LSTM
- Benchmark results: SAC-LTC achieves 63.37% +/- 0.44% success rate (best), 1.58ms mean inference
- Publication-quality benchmark visualization (learning curves, performance bars, ablation, LaTeX table)
- Training, evaluation, and model export scripts
- xApp gRPC test client
- Complete user guide, business docs, competitive analysis, pricing model

### Empirical Results (v0.1)

- SAC-LTC: 63.37% success, 0.634 spectral efficiency, 1.58ms inference, 2.51ms P99
- SAC-LSTM: 62.51% success, 0.625 spectral efficiency, 0.83ms inference
- SAC-LFM: 62.73% success, 0.627 spectral efficiency, 1.28ms inference
- PPO-LSTM: 62.62% success, 0.626 spectral efficiency, 4.11ms inference
- All within 10ms O-RAN Near-RT RIC budget
