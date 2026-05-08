# NVIDIA_INTEGRATION.md — PreceptualAI's NVIDIA-first stack

> PreceptualAI is built as a first-class consumer of the NVIDIA RAN stack: **Aerial (cuBB SDK)**, **ARC**, **AODT** and **Sionna**. This document is the per-component map: what we integrate, what file in our codebase consumes it, what is tested vs stubbed, and the 3-month roadmap to deepen each integration.

---

## 1. The four NVIDIA components

| Component | What it is | Upstream repo / portal |
|---|---|---|
| **Aerial (cuBB SDK)** | NVIDIA's GPU-accelerated 5G NR L1 (cuPHY) + L2 (cuMAC) software stack with a FAPI L2/L1 split. Ships as a CUDA SDK with FAPI traces, FH parquet captures and a cuBB H5 test-vector corpus. | https://developer.nvidia.com/aerial — NGC: `nvidia/aerial-cuda-accelerated-ran` |
| **ARC** | Aerial RAN Computer reference platform — the deployment target for cuBB on Grace Hopper / GB200 nodes. Supplies the runtime envelope PreceptualAI's rApp talks to. | https://developer.nvidia.com/aerial-ran-computer-1 |
| **AODT** | Aerial Omniverse Digital Twin — RAN-scale, USD-based digital twin of a deployment, with scene geometry, ephemeris and ray-tracing hooks. | https://developer.nvidia.com/aerial-omniverse-digital-twin |
| **Sionna** | Open-source GPU-accelerated link-level simulator: 3GPP TDL profiles, ray-traced channels, OFDM PHY simulation. | https://github.com/NVlabs/sionna |

---

## 2. Integration map (per-component)

### 2.1 Aerial / cuBB

- **What we consume**: FAPI event traces (DL_TTI.request, UL_DCI, RX_DATA.indication, etc.), FH parquet captures (cuPHY ↔ cuMAC), the **114 cuBB H5 test vectors**, cuMAC vectors, the **`cubb_24_3`** release artefacts, and the ASIM / MLSIM / TRTengine reference vectors.
- **On disk**: `data/aerial/` — **1.9 GB** today, expandable from NGC.
- **Codebase consumer**: `src/horizon_ric/data/aerial.py` — parses FAPI events into the existing `TelemetryEvent` schema (`io/schemas.py`); registers as a connector via `io/registry.py`.
- **Tested**: parquet replay round-trip on a subset of FAPI message types; cuBB H5 metadata walk.
- **Stubbed**: full FAPI message-type coverage (we cover the core L2-scheduling subset today); live FAPI-over-socket consumer.

### 2.2 ARC (Aerial RAN Computer)

- **What we consume**: ARC is the deployment target, not a data source. PreceptualAI's rApp lifecycle (`rapp/lifecycle.py`) targets ARC as a runtime profile (CUDA + DOCA + FAPI runtime). Helm chart variants under `deploy/` will carry ARC node-selectors.
- **Tested**: nothing on real ARC hardware yet — we run on commodity x86 + Jetson Orin Nano dev kits.
- **Stubbed**: ARC-specific NUMA / DOCA tuning; cuPHY-collocated low-latency rApp variant.

### 2.3 AODT (Aerial Omniverse Digital Twin)

- **What we consume**: AODT scenario bundles — UE / gNB ephemeris, scene geometry, antenna placements.
- **Codebase consumer**: `src/horizon_ric/data/aodt.py` — walks the bundle and emits ephemeris events as `TelemetryEvent`s; passes the spatial prior into `planner/physics/spatial_prior.py`.
- **Tested**: bundle walker emits ephemeris; deterministic round-trip on a fixture bundle.
- **Stubbed**: **full USD scene parsing is Phase 1.5**; **real-time AODT WebSocket is Phase 2**. Today the adapter consumes a static export, not the live twin.

### 2.4 Sionna

- **What we consume**: Sionna's 3GPP TDL channel profiles and (later) ray-traced channels.
- **Codebase consumer**: `src/horizon_ric/data/sionna_channel.py` — wraps Sionna TDL profile generation and surfaces channel-state tensors (K-factor, σ_SF, τ_RMS) to cross-check `planner/physics/tr38811.py`.
- **Tested**: TDL profile cross-check on (env, freq, elev) tuples — used in H4 measurement criterion #1.
- **Stubbed**: full Sionna ray-tracing pipeline driven from AODT geometry; OFDM end-to-end PHY-in-the-loop simulation. Both are Phase 2.

---

## 3. Tested vs stubbed — at a glance

| Adapter | File | Tested today | Stubbed / Phase 1.5 | Phase 2 |
|---|---|---|---|---|
| Aerial cuBB FAPI replay | `data/aerial.py` | parquet round-trip (subset) | full FAPI message-type coverage | live FAPI socket |
| Aerial cuBB H5 corpus | `data/aerial.py` | metadata walk | per-vector regression suite | NGC pull automation |
| ARC deployment | `deploy/`, `rapp/lifecycle.py` | x86 / Orin Nano | ARC-node Helm profile | cuPHY-collocated rApp |
| AODT scenario bundle | `data/aodt.py` | ephemeris-event walk | full USD scene parse | real-time WebSocket |
| Sionna link-level | `data/sionna_channel.py` | TDL profile cross-check | RT integration via AODT | full PHY-in-the-loop |
| DeepMIMO (NVIDIA-adjacent) | `data/deepmimo.py` | scenario `.mat` load | full ASU suite ingest | runtime channel swap |

If a row says "stubbed", a paying customer should not assume that surface area is production-grade. We do not paper over this.

---

## 4. The 3-month roadmap to deepen the integration

### Month 1 — Phase 1.5 close-out
- Land **full FAPI message-type coverage** in `data/aerial.py`; regress against the 114 cuBB H5 vectors.
- Land **full USD scene parsing** in `data/aodt.py`; replace the ephemeris-only walker.
- Land **DeepMIMO full-suite ingest** with sharded manifest.
- Add an **NVIDIA-integration CI job** that runs the cuBB H5 vectors through the encoder on every PR.

### Month 2 — Live signals
- **Live FAPI socket consumer** in `data/aerial.py`: subscribe to a running cuBB instance over the FAPI L2/L1 socket; fall back to parquet replay on disconnect.
- **AODT WebSocket** consumer: subscribe to a running AODT instance, emit `TelemetryEvent`s on scene changes, drive `planner/physics/spatial_prior.py` from the live twin.
- **Sionna RT pipeline** driven from the AODT geometry — close the loop where the same scene that drives the spatial prior also drives the channel sim.

### Month 3 — ARC + reference contribution
- **ARC-node Helm profile** in `deploy/`; tested on an ARC reference node (target: NVIDIA partner lab).
- **cuPHY-collocated rApp** variant: a near-RT decision path that runs in the cuBB process address space for sub-ms inference (separate from the Non-RT rApp).
- **Reference contribution**: publish the FAPI replay → world-model adapter as a reference rApp under OSC NONRTRIC, citing the NVIDIA stack as the canonical data source.

---

## 5. Honest disclosure

- We integrate **with** the NVIDIA stack; we are not part of NVIDIA. No endorsement is implied.
- Trademarks: Aerial, ARC, AODT, NVIDIA, Omniverse, Sionna belong to NVIDIA Corporation.
- The DeepMIMO dataset is from KAUST / authors' own publication; it is NVIDIA-adjacent (heavily used by Aerial users) but not an NVIDIA product.
- Every released checkpoint's model card lists which NVIDIA datasets it consumed by SHA-256 hash. If a hash is missing, the checkpoint did not see that data.
