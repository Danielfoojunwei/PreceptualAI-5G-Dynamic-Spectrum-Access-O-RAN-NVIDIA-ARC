# SpectrAI User Guide

Complete documentation for installing, training, deploying, and operating SpectrAI.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Training Guide](#training-guide)
5. [Model Export](#model-export)
6. [xApp Deployment](#xapp-deployment)
7. [gRPC API Reference](#grpc-api-reference)
8. [Federated Learning](#federated-learning)
9. [Docker Deployment](#docker-deployment)
10. [Monitoring](#monitoring)
11. [Configuration Reference](#configuration-reference)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| pip | 22+ | |
| Docker | 24+ | Optional, for containerized deployment |
| NVIDIA GPU | Compute 7.0+ | Optional, for TensorRT acceleration |
| CUDA | 12.0+ | Required only with GPU |

---

## Installation

### Option 1: From Source (Recommended for Development)

```bash
git clone https://github.com/Danielfoojunwei/SAC-LTC.git
cd SAC-LTC
pip install -e ".[dev]"
```

### Option 2: pip

```bash
pip install spectrai
```

### Option 3: Docker

```bash
docker pull ghcr.io/spectrai/spectrai:latest
docker run --gpus all -p 50051:50051 -p 9090:9090 ghcr.io/spectrai/spectrai:latest
```

### Verify Installation

```bash
python -c "import spectrai; print(spectrai.__version__)"
pytest tests/ -v  # 76 tests should pass
```

---

## Quick Start

Get from zero to first prediction in 5 commands:

```bash
# 1. Install
pip install -e .

# 2. Train (simulated environment)
python scripts/train.py --num_steps 5000

# 3. Export to ONNX
python scripts/export_model.py --checkpoint results/checkpoint.pt

# 4. Start the gRPC inference server
python -m spectrai.xapp.server --config xapp_config.yaml

# 5. Run inference
python scripts/xapp_client.py --num_requests 10
```

---

## Training Guide

### Simulated Environment

Train a SAC-LTC agent against the built-in spectrum simulator:

```bash
python scripts/train.py \
    --num_steps 100000 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --device cuda
```

Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `--num_steps` | 100000 | Total training steps |
| `--lr` | 3e-4 | Learning rate for all optimizers |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Polyak averaging coefficient |
| `--batch_size` | 256 | Replay buffer sample size |
| `--buffer_size` | 1000000 | Replay buffer capacity |
| `--learning_starts` | 1000 | Steps before first update |
| `--hidden_dim` | 128 | LTC encoder hidden dimension |
| `--latent_dim` | 128 | Latent representation dimension |
| `--num_layers` | 2 | Number of LTC encoder layers |
| `--dt` | 1.0 | ODE integration time step |
| `--device` | cuda | Training device (cuda/cpu) |

### Real 5G Data (UCC MISL Dataset)

```bash
python demo.py --data_dir /path/to/5g_traces
```

The real 5G environment expects CSVs from the UCC MISL dataset (83 traces, 188K measurements). Each CSV should contain columns for RSRP, RSRQ, SINR, and timestamp.

### Custom Data

Create a `Real5GEnv` with your own CSV files:

```python
from spectrai.env.real5g import Real5GEnvironment

env = Real5GEnvironment(
    data_dir="/path/to/your/csvs",
    num_channels=10,
    sequence_length=16,
)
```

---

## Model Export

### ONNX Export

```bash
python scripts/export_model.py --checkpoint results/checkpoint.pt
```

Output: `models/actor.onnx`

### TensorRT Compilation

```bash
python scripts/export_model.py \
    --checkpoint results/checkpoint.pt \
    --tensorrt \
    --precision fp16
```

Supported precisions: `fp32`, `fp16`, `int8`

Output: `models/actor.trt`

---

## xApp Deployment

### Configuration

Create or edit `xapp_config.yaml`:

```yaml
environment:
  type: simulated
  num_channels: 10
  num_features: 3
  sequence_length: 16

model:
  hidden_dim: 128
  latent_dim: 128
  num_layers: 2
  dt: 1.0

inference:
  model_path: models/actor.onnx
  backend: auto          # auto | tensorrt | onnx | pytorch
  precision: fp16
  max_batch_size: 1
  device: cuda

grpc:
  host: 0.0.0.0
  port: 50051
  max_workers: 4
  reflection: true

monitoring:
  prometheus_port: 9090
  log_level: INFO
  log_format: json
```

### Starting the Server

```bash
python -m spectrai.xapp.server --config xapp_config.yaml
```

The server starts with:
- gRPC on port 50051
- Prometheus metrics on port 9090
- Automatic backend selection: TensorRT → ONNX Runtime → PyTorch

### Health Check

```bash
grpcurl -plaintext localhost:50051 spectrai.InferenceService/GetHealth
```

---

## gRPC API Reference

| Endpoint | Type | Description |
|---|---|---|
| `Predict` | Unary | Single spectrum allocation prediction |
| `PredictStream` | Server streaming | Continuous prediction stream |
| `GetHealth` | Unary | Server health and backend status |
| `GetMetrics` | Unary | Inference latency and throughput metrics |

### Predict

**Request:**
```protobuf
message PredictRequest {
    repeated float observation = 1;  // Flattened [sequence_length × input_dim]
    int32 batch_size = 2;            // Optional, default 1
}
```

**Response:**
```protobuf
message PredictResponse {
    repeated float action_probs = 1;  // Channel selection probabilities
    int32 selected_action = 2;        // Argmax channel
    float latency_ms = 3;             // Inference time
}
```

### Client Example

```bash
python scripts/xapp_client.py --num_requests 500 --host localhost --port 50051
```

---

## Federated Learning

### Concept

SpectrAI's federated learning uses a hybrid aggregation strategy:
- **Structural weights** (encoder, critics): globally averaged across all sites
- **Time-constant (tau) weights**: partially personalized per site via `tau_mix_ratio`
- **Privacy**: raw spectrum data never leaves the edge device

### Running the FL Demo

```bash
python -m spectrai.federated.demo \
    --data_dir /path/to/5g_traces \
    --num_rounds 10 \
    --num_clients 5 \
    --local_steps 200
```

### FL Configuration

| Parameter | Default | Description |
|---|---|---|
| `tau_mix_ratio` | 0.5 | Blend ratio for tau weights (0 = fully local, 1 = fully global) |
| `num_rounds` | 10 | Number of federated rounds |
| `local_steps_per_round` | 200 | Local training steps per round |
| `aggregation_method` | `hybrid` | Aggregation: `fedavg`, `fedprox`, `hybrid` |
| `fedprox_mu` | 0.01 | FedProx proximal penalty coefficient |

### Cold-Start Elimination

New cell sites immediately receive the global model, providing reasonable performance from the first inference without requiring local training data.

### Docker FL Stack

```bash
docker compose -f docker/docker-compose.fl.yaml up
```

This starts:
- 1 FL aggregator server
- 3 simulated edge clients
- Prometheus + Grafana monitoring

---

## Docker Deployment

### Single Server

```bash
docker compose up
```

Services:
- `spectrai-server`: gRPC inference on :50051
- `prometheus`: Metrics on :9090

### Federated Learning Stack

```bash
docker compose -f docker/docker-compose.fl.yaml up
```

### NVIDIA ARC Deployment

For NVIDIA ARC-Compact (L4) or ARC-Pro (Blackwell RTX PRO):

```bash
docker build -f docker/Dockerfile.aerial -t spectrai-aerial .
docker run --gpus all --runtime=nvidia \
    -p 50051:50051 -p 9090:9090 \
    spectrai-aerial
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SPECTRAI_CONFIG` | `xapp_config.yaml` | Config file path |
| `SPECTRAI_LOG_LEVEL` | `INFO` | Logging level |
| `SPECTRAI_DEVICE` | `cuda` | Compute device |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

---

## Monitoring

### Prometheus Metrics

SpectrAI exposes the following metrics on the Prometheus port (default 9090):

**Counters:**
| Metric | Description |
|---|---|
| `spectrai_predictions_total` | Total predictions served |
| `spectrai_errors_total` | Total inference errors |

**Histograms:**
| Metric | Description |
|---|---|
| `spectrai_inference_latency_seconds` | Per-inference latency |
| `spectrai_grpc_request_duration_seconds` | gRPC request duration |

**Gauges:**
| Metric | Description |
|---|---|
| `spectrai_model_loaded` | 1 if model is loaded, 0 otherwise |
| `spectrai_active_backend` | Current backend (trt=3, onnx=2, pytorch=1) |
| `spectrai_collision_rate` | Current spectrum collision rate |
| `spectrai_success_rate` | Current channel access success rate |

### Grafana Setup

1. Add Prometheus data source: `http://prometheus:9090`
2. Import the dashboard from `docker/grafana/dashboards/spectrai.json`

---

## Configuration Reference

Full YAML configuration with all fields, types, and defaults:

### environment

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `simulated` / `oran` / `aodt` | `simulated` | Environment backend |
| `num_channels` | int | 10 | Number of spectrum channels |
| `num_features` | int | 3 | Features per channel |
| `sequence_length` | int | 16 | Observation history length |
| `max_steps` | int | 200 | Max steps per episode |
| `reward_success` | float | 1.0 | Reward for successful access |
| `reward_collision` | float | -1.0 | Penalty for collision |
| `reward_switch` | float | -0.1 | Penalty for channel switching |

### model

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | int | 128 | LTC hidden dimension |
| `latent_dim` | int | 128 | Latent representation size |
| `num_layers` | int | 2 | LTC encoder layers |
| `dt` | float | 1.0 | ODE integration step size |

### inference

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | str | `models/actor.onnx` | Path to model file |
| `backend` | `auto`/`tensorrt`/`onnx`/`pytorch` | `auto` | Inference backend |
| `precision` | `fp32`/`fp16`/`int8` | `fp16` | Compute precision |
| `max_batch_size` | int | 1 | Maximum batch size |
| `device` | str | `cuda` | Compute device |

### grpc

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | str | `0.0.0.0` | Bind address |
| `port` | int | 50051 | gRPC port |
| `max_workers` | int | 4 | Thread pool size |
| `reflection` | bool | true | Enable gRPC reflection |

### monitoring

| Field | Type | Default | Description |
|---|---|---|---|
| `prometheus_port` | int | 9090 | Metrics port |
| `log_level` | str | `INFO` | Log level |
| `log_format` | `json`/`console` | `json` | Log output format |
| `log_file` | str | null | Optional log file path |

---

## Troubleshooting

### Model fails to load

```
InferenceEngineError: Failed to load TensorRT engine from models/actor.trt
```

**Cause:** TensorRT engine was built with a different CUDA/TRT version.
**Fix:** Re-export with your current TRT version:
```bash
python scripts/export_model.py --checkpoint results/checkpoint.pt --tensorrt
```

### gRPC connection refused

**Cause:** Server not running or wrong port.
**Fix:** Check server logs and ensure port 50051 is not in use:
```bash
ss -tlnp | grep 50051
```

### CUDA out of memory

**Fix:** Reduce batch size or use CPU:
```yaml
inference:
  max_batch_size: 1
  device: cpu
```

### Federated learning clients disconnect

**Cause:** Network timeout or aggregator overloaded.
**Fix:** Increase `control_request_timeout_s` in xApp config and check network connectivity between edge nodes and aggregator.

### Tests fail with import errors

**Fix:** Ensure dev dependencies are installed:
```bash
pip install -e ".[dev]"
```

---

## FAQ

**Q: What spectrum bands does SpectrAI support?**
A: SpectrAI is band-agnostic. It works with any spectrum data that can be represented as channel occupancy features. It has been validated on sub-6 GHz 5G data from the UCC MISL dataset.

**Q: Can I use SpectrAI without an NVIDIA GPU?**
A: Yes. The PyTorch backend runs on CPU. GPU acceleration (via TensorRT or ONNX Runtime) is recommended for production latency targets.

**Q: How much training data do I need?**
A: The simulated environment requires no external data. For real 5G deployment, we recommend at least 10,000 measurements per cell site. With federated learning, new sites benefit immediately from the global model.

**Q: Does SpectrAI comply with O-RAN specifications?**
A: Yes. SpectrAI implements E2SM-KPM for metrics collection and E2SM-RC for control actions, compatible with the O-RAN Near-RT RIC architecture.

**Q: What is the inference latency?**
A: Mean 3.14ms, P99 3.96ms on NVIDIA L4 GPU — well within the O-RAN Near-RT RIC 10ms budget.

**Q: Can I run multiple xApp instances?**
A: Yes. Each instance is stateless after model loading. Use a load balancer for horizontal scaling.

**Q: Is my spectrum data sent to the cloud?**
A: No. With federated learning, only model weight updates leave the edge. Raw spectrum data stays on-site.

**Q: What's the difference between `fedavg`, `fedprox`, and `hybrid` aggregation?**
A: `fedavg` averages all weights equally. `fedprox` adds a proximal penalty to prevent drift. `hybrid` (recommended) averages structural weights globally while keeping time-constant (tau) weights partially personalized per site.
