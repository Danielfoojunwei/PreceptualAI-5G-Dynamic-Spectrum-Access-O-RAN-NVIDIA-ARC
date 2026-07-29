#!/usr/bin/env python3
"""Build a deterministic DSA feature set from DeepMIMO's ASU 3.5 GHz scenario.

The raw scenario and per-receiver features are deliberately not committed.
This script downloads the published ray-tracing archive, pins it by SHA-256,
uses DeepMIMO itself to generate SISO OFDM channels, and writes a compact local
JSONL feature set plus a tracked provenance manifest.

Run with Python 3.11+ after ``pip install '.[realdata]'``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.data.lineage import compute_dataset_sha256

# Attempts for the upstream scenario download. The archive is CDN-hosted and
# returns transient 5xx; backoff is 2s, 4s, 8s between attempts.
_DOWNLOAD_ATTEMPTS = 4

SCENARIO = "asu_campus_3p5"
DEEPMIMO_VERSION = "4.0.0"
DEEPMIMO_RELEASE_COMMIT = "cbe3a3eae426d0d0a2bd1fdf95ce011c536a3dd9"
# Pinned SHA-256 of the downloaded scenario archive (asu_campus_3p5
# *downloaded.zip). The build hard-fails on any mismatch so a changed or
# corrupted upstream artifact can never silently enter the pipeline. Matches
# the tracked manifest.json "source_archive_sha256".
SOURCE_ARCHIVE_SHA256 = (
    "80e4a4983847023158ed61a5a489025d72f4edcdac89edf4ee1323699e1b7da3"
)
SCENARIO_DOCS_URL = "https://deepmimo.net/docs/tutorials/1_getting_started.html"
DEEPMIMO_RELEASE_URL = "https://github.com/DeepMIMO/DeepMIMO/releases/tag/v4.0.0"
MATRICES = [
    "rx_pos",
    "tx_pos",
    "power",
    "phase",
    "delay",
    "aoa_az",
    "aoa_el",
    "aod_az",
    "aod_el",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_or_resolve(
    *,
    scenario_dir: Path | None,
    cache_dir: Path,
    source_archive: Path | None,
) -> tuple[Path, Path]:
    if scenario_dir is not None:
        resolved = scenario_dir.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"scenario directory not found: {resolved}")
        if source_archive is None:
            raise ValueError("--source-archive is required with --scenario-dir")
        archive = source_archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"source archive not found: {archive}")
        return resolved, archive

    import deepmimo as dm

    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = cache_dir / "deepmimo_scenarios" / SCENARIO
    old_cwd = Path.cwd()
    try:
        # DeepMIMO extracts under ./deepmimo_scenarios. Pin the working
        # directory so no untracked 100+ MB tree appears in the repository.
        os.chdir(cache_dir)
        # dm.download() does NOT raise on an HTTP error: it prints
        # "Download failed: <status>" and returns normally, so the only
        # reliable success signal is the scenario directory appearing. The
        # upstream archive is served from a CDN that does return transient 5xx
        # (observed: 503 from f005.backblazeb2.com), and without a retry a
        # single blip fails the whole realdata workflow — and with it every
        # real-data reproduction claim at once. Retry with backoff.
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            if resolved.is_dir():
                break
            if attempt:
                time.sleep(2**attempt)
                print(
                    f"retrying DeepMIMO download for {SCENARIO} "
                    f"(attempt {attempt + 1}/{_DOWNLOAD_ATTEMPTS})"
                )
            try:
                dm.download(SCENARIO, output_dir=str(cache_dir))
            except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
                print(f"DeepMIMO download attempt {attempt + 1} raised: {exc!r}")
    finally:
        os.chdir(old_cwd)

    archives = sorted(cache_dir.rglob(f"{SCENARIO}*downloaded.zip"))
    if not resolved.is_dir():
        # Do not blame a missing directory: after N attempts this is an upstream
        # availability problem, and saying so is the difference between a
        # 30-second re-run and a hunt for a nonexistent code regression.
        raise FileNotFoundError(
            f"DeepMIMO scenario {SCENARIO!r} was not downloaded after "
            f"{_DOWNLOAD_ATTEMPTS} attempts (expected {resolved}). The upstream "
            "scenario host failed to serve the archive — check the "
            "'Download failed:' lines above for the HTTP status. This is an "
            "upstream availability failure, not a reproduction regression; "
            "re-run the job once the host recovers."
        )
    if not archives:
        raise FileNotFoundError(f"DeepMIMO source archive not found below {cache_dir}")
    return resolved, archives[0]


def _round_floats(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def _write_features(
    *,
    scenario_dir: Path,
    features_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    import deepmimo as dm

    package_version = importlib.metadata.version("deepmimo")
    if package_version != DEEPMIMO_VERSION:
        raise RuntimeError(
            f"expected deepmimo=={DEEPMIMO_VERSION}, found {package_version}"
        )

    # DeepMIMO 4.0 accepts an absolute scenario path for its data matrices but
    # still resolves params.json through its configured scenarios directory.
    # Point both lookups at the same downloaded tree, then restore global state.
    previous_scenarios_folder = dm.config.get("scenarios_folder")
    dm.config.set("scenarios_folder", str(scenario_dir.parent))
    try:
        # First load only enough data to choose receivers that have at least one
        # ray-traced propagation path.
        census = dm.load(
            str(scenario_dir),
            matrices=["rx_pos", "power"],
            max_paths=10,
        )
        candidate_count = int(census.power.shape[0])
        valid_indices = np.flatnonzero(np.any(np.isfinite(census.power), axis=1))
        if sample_count > len(valid_indices):
            raise ValueError(
                f"requested {sample_count} rows but only "
                f"{len(valid_indices)} receivers have paths"
            )

        sample_offsets = np.linspace(
            0,
            len(valid_indices) - 1,
            sample_count,
            dtype=np.int64,
        )
        receiver_indices = valid_indices[sample_offsets]
        dataset = dm.load(
            str(scenario_dir),
            rx_sets={0: receiver_indices},
            matrices=MATRICES,
            max_paths=10,
        )
    finally:
        dm.config.set("scenarios_folder", previous_scenarios_folder)

    params = dm.ChannelParameters()
    params.bs_antenna.shape = [1, 1]
    params.ue_antenna.shape = [1, 1]
    params.num_paths = 10
    params.ofdm.subcarriers = 1024
    params.ofdm.selected_subcarriers = np.linspace(0, 1023, 60, dtype=np.int64)
    params.ofdm.bandwidth = 100e6
    params.doppler = False
    params.freq_domain = True

    channels = dataset.compute_channels(params)
    power_w = np.abs(channels[:, 0, 0, :]) ** 2
    subband_power = np.stack(
        [chunk.mean(axis=1) for chunk in np.split(power_w, 6, axis=1)],
        axis=1,
    )
    subband_gain_dbw = 10.0 * np.log10(np.maximum(subband_power, 1e-30))
    if not np.all(np.isfinite(subband_gain_dbw)):
        raise RuntimeError("DeepMIMO transform produced a non-finite feature")

    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, receiver_index in enumerate(receiver_indices):
            gains = subband_gain_dbw[row]
            record = {
                "receiver_index": int(receiver_index),
                "position_m": _round_floats(dataset.rx_pos[row]),
                "subband_gain_dbw": _round_floats(gains),
                "best_subband": int(np.argmax(gains)),
                "worst_subband": int(np.argmin(gains)),
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return {
        "candidate_receivers": candidate_count,
        "receivers_with_paths": int(len(valid_indices)),
        "sampled_receivers": int(sample_count),
        "subband_gain_dbw_min": round(float(np.min(subband_gain_dbw)), 6),
        "subband_gain_dbw_max": round(float(np.max(subband_gain_dbw)), 6),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    scenario_dir, source_archive = _download_or_resolve(
        scenario_dir=args.scenario_dir,
        cache_dir=args.cache_dir,
        source_archive=args.source_archive,
    )
    source_archive_sha256 = _sha256_file(source_archive)
    if source_archive_sha256 != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            "DeepMIMO scenario archive failed its pinned checksum: expected "
            f"{SOURCE_ARCHIVE_SHA256}, got {source_archive_sha256} for "
            f"{source_archive}"
        )
    stats = _write_features(
        scenario_dir=scenario_dir,
        features_path=args.features,
        sample_count=args.samples,
    )
    manifest = {
        "dataset": "DeepMIMO ASU Campus 3.5 GHz",
        "scenario": SCENARIO,
        "data_kind": "site-specific Wireless InSite ray tracing; not over-the-air capture",
        "deepmimo_version": DEEPMIMO_VERSION,
        "deepmimo_release_commit": DEEPMIMO_RELEASE_COMMIT,
        "deepmimo_release_url": DEEPMIMO_RELEASE_URL,
        "scenario_docs_url": SCENARIO_DOCS_URL,
        "source_archive_sha256": source_archive_sha256,
        "source_tree_sha256": compute_dataset_sha256(scenario_dir),
        "features_sha256": compute_dataset_sha256(args.features),
        "features_sha256_scope": (
            "Exact JSONL bytes for this build. Cross-host CI requires the same "
            "source hashes, transform, structure and safety outcomes, and compares "
            "floating outputs with 0.0001 dB absolute tolerance."
        ),
        "features_committed": False,
        "licensing": {
            "deepmimo_tool": "Apache-2.0",
            "scenario_archive": (
                "No separate license file was present in the downloaded scenario archive; "
                "raw and row-level derived data are therefore not redistributed here."
            ),
        },
        "transform": {
            "antenna": "SISO isotropic",
            "carrier_frequency_hz": 3_500_000_000,
            "bandwidth_hz": 100_000_000,
            "ofdm_total_subcarriers": 1024,
            "ofdm_selected_subcarriers": 60,
            "subbands": 6,
            "paths_per_receiver": 10,
            "receiver_sampling": "evenly spaced over receivers with >=1 finite path",
        },
        **stats,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/horizon-deepmimo"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
