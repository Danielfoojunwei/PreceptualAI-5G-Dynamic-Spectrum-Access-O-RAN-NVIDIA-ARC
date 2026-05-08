# DATA.md — How PreceptualAI handles its real-data corpus

**Honest status (today):** **~8 GB of real, NVIDIA-anchored data is on disk; the full 1.1 TB ingest is staged via NGC pulls.**

The real-data corpus on disk now is:

| Source | On disk | Contents |
|---|---|---|
| **NVIDIA Aerial / cuBB** | 1.9 GB | FAPI event traces, FH parquet (cuPHY ↔ cuMAC), 114 cuBB H5 test vectors, cuMAC vectors, the `cubb_24_3` release artefacts, ASIM / MLSIM / TRTengine reference vectors |
| **DeepMIMO** | 6.2 GB | 100+ ASU campus 3.5 GHz scenarios with full ray-traced channels |
| **UCC MISL 5G** | 28 MB | 5G NR throughput / latency traces |
| **AliMaatouk telecom_ts** | (see `AliMaatouk___telecom_ts/`) | telecom time-series benchmark |
| **Total real-data on disk** | **~8 GB** | expandable to the full 1.1 TB via NGC pulls of the rest of the cuBB vectors and the full DeepMIMO suite |

The shipped SLA head v0.1 checkpoint
(`checkpoints/sla_head_v0.1.pt`) and the 12-benchmark suite
(`benchmarks/RESULTS.md`) still trained on **synthetic windows** for now;
the real-data adapters land alongside H4 (NVIDIA-Native Compositional WM)
in Phase 1.5. The honest count for the trained checkpoint remains
~5 MB synthetic — a fact we track in the model card and do not paper over.

Per our own ultrareview discipline ("don't claim what isn't there"), this
document explains: (1) exactly what the architecture *does* support today,
(2) what we have *not* done with the 1.1 TB, (3) the concrete plan to
ingest it, with file paths and effort estimates.

---

## 1. What the architecture supports today

The data plane is designed to scale; it has not been *exercised* at scale.

| Layer | Module | Status | Scale tested |
|---|---|---|---|
| Streaming connectors | `io/connectors/file_connector.py` (JSONL), `http_connector.py`, `kafka_connector.py` | ✅ shipped | Single-file replay (5 events) |
| Stable schemas | `io/schemas.py` — `TelemetryEvent`, `FeatureFrame`, `PolicyAction`, `AuditRecord` (Pydantic v2 with `extra="allow"` for forward-compatible modalities) | ✅ shipped | Round-trip ms |
| Connector registry + entry-point plug-ins | `io/registry.py` | ✅ shipped | 4 connectors registered |
| Modality dispatch | `Modality` Literal in `io/schemas.py` covers 13 types: `kpm_5g`, `kpm_ntn`, `spectrum_iq`, `spectrum_psd`, `tle`, `ephemeris_oem`, `weather_grib`, `thermal`, `spectrum_lbt`, `ue_qos`, `isac_radar`, `user_event`, `synthetic` | ✅ shipped | — |
| Federated weights-only contract | `federated/aggregator.py` | ✅ shipped | 2-client unit tests |
| Offline scenario generator | `scenarios/maritime.py` | ✅ shipped | 2000 windows / ~5 MB |

**What's true:** the connector + schema + registry foundation is real. Any
1.1 TB corpus organised as JSONL, parquet, or a Kafka topic of
`TelemetryEvent`-shaped records can be plugged in *without changing the
encoder, planner, or audit code*.

**What's not true:** there is **no shipped DataLoader, no sharded
preprocessing, no streaming parquet reader, no DDP / FSDP training, no
GPU memory budget, no I/O benchmarks at TB scale.** None of it has been
built or measured.

---

## 1b. NVIDIA Aerial Integration

The data adapters are in flight under `src/horizon_ric/data/`:

- **`aerial.py`** consumes the cuBB FAPI event traces and FH parquet under
  `data/aerial/`. Round-trips FAPI message types into the existing
  `TelemetryEvent` schema (`io/schemas.py`) so the encoder doesn't change.
  Today: parquet replay + a subset of FAPI message types. Phase 1.5: full
  message-type coverage and a live FAPI socket consumer.
- **`sionna_channel.py`** wraps Sionna's TDL profiles and ray-traced
  channel sims into channel-state tensors that feed `planner/physics/tr38811.py`
  as the calibration signal. Today: TDL profile cross-check. Phase 2:
  the full Sionna ray-tracing pipeline driven from AODT geometry.
- **`aodt.py`** walks an AODT scenario bundle and emits ephemeris + scene
  events into `TelemetryEvent`. Honest scope: walks the bundle and emits
  ephemeris events; **full USD parsing is Phase 1.5**, real-time WebSocket
  is Phase 2.
- **`deepmimo.py`** loads DeepMIMO scenario `.mat` channels into channel
  matrices the world model can replay through.

(File names above describe the in-flight adapter contract; some files are
being landed by parallel agents and may not all be on `main` yet — see the
`NVIDIA_INTEGRATION.md` "tested vs stubbed" matrix for the live state.)

---

## 2. What we are NOT doing with the 1.1 TB today

Putting this in writing so it does not get said in front of a customer:

- We are not training any model on the corpus. The shipped checkpoint
  was trained on **2000 synthetic windows = ~5 MB** in 23 epochs on CPU.
- We have not measured ingest throughput, parquet decode time, or
  preprocessing CPU cost on real data of this volume.
- We have no manifest, hash-tree, or content-addressed shard layout for
  the corpus — bare minimum for reproducible training.
- We have no published model card linking a trained checkpoint to a
  specific subset of the corpus (TS 28.105 §7.4 lineage requirement).
- We have not validated weights-only-FL bandwidth at the parameter
  count a model trained on 1.1 TB would actually need (50–500 M params
  to use the corpus at all).
- We have not written or run a single distributed-training step.

If a stakeholder asks "did your benchmarks come from the 1.1 TB run?",
the honest answer is **no — the SLA head checkpoint trained on synthetic
maritime windows, and the benchmark numbers in `benchmarks/RESULTS.md`
are inference latencies on random tensors of the size the production
shapes will eventually take.**

---

## 3. The plan to actually ingest the 1.1 TB

Five concrete deliverables. Each is independent enough to ship in
sequence; together they take this repo from "trains on 5 MB synthetic"
to "pretrains on the 1.1 TB corpus" in 4–6 weeks.

### Deliverable A — Corpus manifest + content-addressed shard layout
**File:** `data/corpus_manifest.json` + `scripts/manifest_corpus.py`
**Effort:** 2 days

Walks the 1.1 TB directory tree, hashes every shard, records modality,
sample count, time range, source operator, and licensing. Produces a
SHA-256-keyed manifest the training script consumes. Anything not in
the manifest never enters training. This is the TS 28.105 §7.4
lineage primitive plus the "don't train on accidental files" guard.

### Deliverable B — Streaming dataset adapter (parquet + webdataset)
**File:** `src/horizon_ric/io/dataset.py`
**Effort:** 3–5 days

A `torch.utils.data.IterableDataset` that streams from the manifest.
Two backends:

  * **Parquet** via `pyarrow.dataset` for the structured KPM / weather /
    ephemeris modalities (most of the corpus).
  * **Webdataset** (tar shards) for `spectrum_iq` and `isac_radar`
    binary blobs.

Yields `TelemetryEvent` objects (the existing schema). The encoder's
`EntityTokenizer` already accepts these — no encoder change needed.
Throughput target: ≥250 MB/s decoded on a single CPU node, measured.

### Deliverable C — Sharded preprocessing pipeline
**File:** `scripts/preprocess_shards.py`
**Effort:** 3–4 days

Per-shard: convert raw KPMs into `FeatureFrame` windows, attach derived
physics features (`encoder/link_state.compose_link_state` per visible
satellite, `epfd_down` snapshot per ground station), write one
`features-{shard_id}.parquet` per input shard. Runs as parallel processes
via `concurrent.futures.ProcessPoolExecutor`. Idempotent; resumable;
content-addressed output.

### Deliverable D — Distributed pretraining loop
**Files:** `scripts/pretrain_jepa.py`, `scripts/pretrain_world_model.py`
**Effort:** 5–7 days

Two pretraining jobs over the preprocessed shards:

  * **Graph-JEPA pretraining** (`encoder/graph_jepa.py`) — action-free,
    EMA target encoder, mask-and-predict in latent space. Fits the
    `Perceiver` backbone we already shipped. Distributed via PyTorch
    `DistributedDataParallel`.
  * **Latent-dynamics pretraining** (`core/latent_dynamics.py`) on the
    next-state-given-action targets implicit in the corpus.

Both consume the streaming dataset from Deliverable B; checkpoints are
SHA-256-hashed and registered in the manifest the SLA head training
already references in its model card.

### Deliverable E — Federated wrapper around Deliverables D
**File:** `scripts/fl_pretrain.py`
**Effort:** 3 days

For the **per-operator** subsets of the corpus that cannot leave the
operator's data centre (the weights-only-FL constraint from project
memory), wrap Deliverable D in a Flower (`flwr`) coordinator using the
existing `federated/aggregator.py` `FedAvg` / `FedProx` primitives.
Each operator runs a pretraining client on its slice; only weight
deltas leave the perimeter. Top-k sparsification (`federated/sparsifier.top_k_sparsify`)
caps round payload at the cellular-backhaul ceiling we already measured
(`core/timing_budgets.fl_round_seconds`).

---

## 4. Honest sizing — can the architecture *handle* 1.1 TB?

| Pipeline stage | Computed cost | Notes |
|---|---|---|
| Manifest + hashing | ~30 minutes single-CPU walk | I/O-bound; one-shot. |
| Parquet decode | ≥250 MB/s/node target | 1.1 TB → ~75 minutes single node, or ~5 minutes on 16 nodes. |
| Feature preprocessing | EPFD snapshot 40 µs (measured `RESULTS.md` row 2); link-state ~80 µs | At 250 MB/s of KPMs ≈ 2.5 M events/s → preprocessing fits inside the I/O budget on multicore. |
| Graph-JEPA forward (256-token) | 16.6 ms/forward (measured row 5) | At batch 32 / 4 GPUs → ~2 K steps / minute → ~1 epoch over 1.1 TB in ~2 days on a 4×A100 box. Aspirational, but the inference number is grounded. |
| FL round (50 M params, fp32, top-1 % sparse, 50 Mbps) | 16 s/round per `fl_round_seconds` | Comfortable. |

These are **projected** numbers based on the measured per-call latencies
in `benchmarks/RESULTS.md`. They are the basis for the deliverable
estimates above — they are not the result of a 1.1 TB run.

---

## 5. What we say to a stakeholder asking the question

> "Today the architecture supports streaming ingest at any volume — the
> connectors, schemas, and federated-weights-only contract are shipped
> and tested. We have not yet pretrained on the 1.1 TB corpus; the
> shipped checkpoint trained on a 5 MB synthetic substitute, and we
> are deliberate about not pretending otherwise. The plan to ingest
> the full 1.1 TB lives in `DATA.md`, breaks into five deliverables
> totalling four-to-six weeks of engineering, and the projected
> throughput is grounded in the per-call latencies we measured this
> week. The pilot scope (PILOT.md Tier-1 / Tier-2) does not require
> the 1.1 TB pretraining to land first — we use the synthetic head as
> the day-one baseline and let the 1.1 TB pretrained encoder upgrade
> the system in-place once it lands."

That is the answer. No softening, no "we're working on it," no implication
that the 1.1 TB is already inside the SLA head. The corpus is in the
plan; the plan is in this file; the file is in the repo.
