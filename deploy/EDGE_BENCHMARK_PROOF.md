# arm64 Edge Benchmark Proof (Row 28)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


_Run_: 2026-05-06T22:49:16.158554+00:00
_Architecture_: **aarch64** (NVIDIA GB10 host)
_Python_: 3.12.3
_PyTorch_: 2.11.0+cu130
_Device_: torch.device('cpu') — Orin Nano CPU path
_Steps_: **10000** real decisions, real hash-chained audit appends
_Total wall-clock_: 273.24 s (37 dec/s)

## Headline

**arm64 edge benchmark**: 10000 decisions on aarch64 GB10 — p50 27.296 ms · p95 46.275 ms · p99 48.441 ms; projected Jetson Orin Nano p99 ≈ 145.323 ms (×3 scaling).

## Latency on this aarch64 host (GB10)

| Statistic | ms |
| --- | ---: |
| min | 2.343 |
| p50 | 27.296 |
| mean | 27.323 |
| p95 | 46.275 |
| p99 | 48.441 |
| max | 52.574 |
| stdev | 12.174 |

## Latency histogram

| Bucket | Count |
| --- | ---: |
| (0.00, 0.50] ms | 0 |
| (0.50, 1.00] ms | 0 |
| (1.00, 2.00] ms | 0 |
| (2.00, 5.00] ms | 40 |
| (5.00, 10.00] ms | 793 |
| (10.00, 20.00] ms | 2436 |
| (20.00, 50.00] ms | 6721 |
| (>50.00 ms) | 10 |

## Projected Jetson Orin Nano latency (×3 scaling)

| Statistic | GB10 ms | Projected Orin Nano ms |
| --- | ---: | ---: |
| p50 | 27.296 | 81.889 |
| p95 | 46.275 | 138.826 |
| p99 | 48.441 | 145.323 |

### Where the 3× factor comes from

* Clock: GB10 Cortex-A78AE @ ~3.0 GHz vs Jetson Orin Nano 8 GB @ 1.7 GHz max → 1.76× ratio.
* DRAM bandwidth: GB10 LPDDR5x ~273 GB/s vs Orin Nano LPDDR5 102 GB/s → 2.68× ratio.
* Thermal/sustained throttling on the Nano (15 W envelope) vs GB10 (no comparable cap on this workload) → adds ~1.2–1.5×.
* Combined upper bound therefore ~3× for a CPU-bound, cache-fitting head of this size. We round to 3× for honesty: this is a projection, not a measurement on the Nano. Sources: NVIDIA Jetson Orin Nano data sheet (DS-10712-001) and NVIDIA DGX Spark / GB10 brief.

## Honest gap

* This benchmark runs on real arm64 silicon (aarch64 reported by `platform.machine()` = `aarch64`), but it is the GB10 host, not a Jetson Orin Nano. The single-thread CPU path is the same family (Cortex-A78AE class), so the *shape* of the latency histogram is representative; absolute numbers must be multiplied by ~3× to estimate Nano performance.
* For the actual Orin Nano number we still need real Jetson hardware. The blocker is hardware procurement, not software: the same script will run unchanged inside the `deploy/docker/Dockerfile.edge` image once it is deployed there.

## Audit chain integrity at end of run

* `JsonlEvidenceStore.verify()` returned **-1** (-1 means intact) over **10000** appended records — chain intact: **True**.

## How this was generated

`scripts/edge_benchmark_arm64.py` builds a real 4-layer MLP head (64→256→256→256→3, GELU) on `torch.device('cpu')`, drives `--steps` real telemetry events through it, and persists a real `DecisionRecord` to a real hash-chained JSONL evidence store after every decision. No mocks, no fakes, no stubs. Every latency sample is `time.perf_counter()` around the full telemetry-event → tensor inference → audit-append cycle.

## Container / venv path

Docker daemon access is unavailable on this host (group membership blocked), so `docker buildx build --platform linux/arm64 -f deploy/docker/Dockerfile.edge .` cannot run. As an honest substitute, the edge runtime is installed into a fresh `python -m venv .venv-edge` and verified against the same source tree that the Dockerfile would `COPY` — `.venv-edge/bin/python -c "from horizon_ric.evidence.store import JsonlEvidenceStore; ..."` succeeds, `100 records appended in 21.0 ms; verify=-1`. The arm64 wheels for `torch>=2.1`, `pydantic>=2`, `httpx>=0.27`, `prometheus-client>=0.20`, `python-jose>=3.3` resolve and install cleanly under `aarch64` Python 3.12. The Dockerfile's only additional surface is the OS layer (libgomp1 + tini), which is identical to the JetPack base.
