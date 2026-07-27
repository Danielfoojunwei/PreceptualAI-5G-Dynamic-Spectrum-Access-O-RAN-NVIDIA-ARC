# Datasets index

Every dataset this repository uses, with provenance and integrity hashes.
Policy: **small, redistributable data is committed directly** (with a tracked
`SHA256SUMS`); **large or non-redistributable data is fetched on demand** by a
build script that pins the source and hard-fails on a SHA-256 mismatch. The
DeepMIMO raw scenario is the latter — it is a 100+ MB external ray-tracing
archive whose download carried no separate dataset license, so committing it
would both bloat the repository and redistribute data we have no license to
redistribute.

| Name | Purpose | Size | Vendoring | Source | SHA-256 / manifest | License / usage |
| --- | --- | --- | --- | --- | --- | --- |
| `datasets/spectrum_dsa` (`traces.jsonl`, 1,620 rows) | Federated DSA decision traces through the real Shield + hash-chained evidence store, honest vs poisoned; evaluation input for `benchmarks/secure_dsa_benchmark.py` | 930 KB | **Committed directly** (fully synthetic, deterministic — `generate.py` reproduces it) | Generated in-repo by `datasets/spectrum_dsa/generate.py` (fixed seeds) | [`datasets/spectrum_dsa/SHA256SUMS`](spectrum_dsa/SHA256SUMS): `traces.jsonl` = `313627efc8186f9758f6608c2fe9d1077409dbdbb96dffa971dc4f47f8a70245` | Apache-2.0 (this repo); see [`DATASHEET.md`](spectrum_dsa/DATASHEET.md) |
| `datasets/deepmimo_asu_3p5` (raw scenario + generated features) | External-data evaluation of DSA channel selection and Shield legality on site-specific Wireless InSite ray tracing (`benchmarks/deepmimo_dsa_benchmark.py`) | Raw scenario 100+ MB (external); derived features 4,096-row JSONL (regenerated, gitignored) | **Fetched on demand with pinned checksum** — `build.py` downloads via the pinned `deepmimo==4.0.0` package and hard-fails unless the archive matches `SOURCE_ARCHIVE_SHA256`; raw data and row-level features are deliberately never committed | DeepMIMO v4.0.0 (release commit `cbe3a3ea`), scenario `asu_campus_3p5`, via `dm.download()` | [`datasets/deepmimo_asu_3p5/manifest.json`](deepmimo_asu_3p5/manifest.json): archive `80e4a4983847023158ed61a5a489025d72f4edcdac89edf4ee1323699e1b7da3`, source tree `42f9c6eb…`, features `ab4414a1…` (host-exact; CI compares floats at 1e-4 dB) | DeepMIMO tool Apache-2.0; the scenario archive shipped **no separate dataset license**, so it is not redistributed here — see [`DATASHEET.md`](deepmimo_asu_3p5/DATASHEET.md) |
| `examples/replay_telemetry.jsonl` (30 events) | Deterministic `kpm_5g` telemetry replay input for the xApp E2E pipeline (`deploy/xapp-e2e/source-replay.yaml`) | 2.5 KB | **Committed directly** | Authored in-repo | `85f5c95ae61d5a6341a72594c98d9a1a5fc3ea0cb48842a2e394d6aea8017e21` | Apache-2.0 (this repo) |
| `benchmarks/results/*.json` (15 files) | Recorded benchmark/evidence outputs (DSA poisoning, DP privacy, jamming, DeepMIMO DSA, secure aggregation, …) referenced by the docs and proofs | 158 KB total | **Committed directly** | Produced by the corresponding `benchmarks/*.py` suites in the locked `deps/locks/horizon.txt` environment | [`benchmarks/results/SHA256SUMS`](../benchmarks/results/SHA256SUMS) (one line per file) | Apache-2.0 (this repo) |

Verify committed data at any time:

```bash
(cd datasets/spectrum_dsa && sha256sum -c SHA256SUMS)
(cd benchmarks/results && sha256sum -c SHA256SUMS)
sha256sum examples/replay_telemetry.jsonl   # expect 85f5c95a…
```

Not datasets (deliberately untracked): `data/e2e/` holds per-run runtime state
(`audit.jsonl`, `state.json`) and is gitignored; `datasets/deepmimo_asu_3p5/generated/`
holds the regenerable DeepMIMO features and is gitignored per the datasheet's
no-redistribution rationale.
