#!/usr/bin/env python3
"""Extract real per-path angular geometry from DeepMIMO's ASU 3.5 GHz scenario.

The coverage build (``build.py``) keeps only the wideband received gain. The
angular capabilities — beam management, beam-level mobility/handover, and
anti-jam beam nulling — need the *ray geometry*: per-path power, phase, angle of
departure (AoD az/el) and arrival (AoA az/el), and delay. This script extracts
those, for the same deterministic, evenly-spaced receivers ``build.py`` samples,
and writes a compact JSONL plus a checksum-pinned provenance manifest. The raw
scenario and the row-level features are not committed (see the manifest
licensing note); the manifest binds the extraction by SHA-256.

Deterministic and checksum-gated exactly like ``build.py``: it hard-fails if the
downloaded scenario archive does not match its pinned SHA-256, and it samples
receivers by ``np.linspace`` (no RNG), so a given scenario yields identical
bytes. Run with Python 3.11+ after ``pip install '.[realdata]'``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.data.lineage import compute_dataset_sha256

SCENARIO = "asu_campus_3p5"
DEEPMIMO_VERSION = "4.0.0"
DEEPMIMO_RELEASE_COMMIT = "cbe3a3eae426d0d0a2bd1fdf95ce011c536a3dd9"
SOURCE_ARCHIVE_SHA256 = (
    "80e4a4983847023158ed61a5a489025d72f4edcdac89edf4ee1323699e1b7da3"
)
SCENARIO_DOCS_URL = "https://deepmimo.net/docs/tutorials/1_getting_started.html"
DEEPMIMO_RELEASE_URL = "https://github.com/DeepMIMO/DeepMIMO/releases/tag/v4.0.0"
MAX_PATHS = 10
# The scenario's base station is an 8-element uniform linear array (the DeepMIMO
# default bs_antenna shape for this scenario); the angular capabilities beamform
# over it. Recorded here so every downstream loop shares one array definition.
BS_ARRAY_ELEMENTS = 8
BS_ARRAY_SPACING_WAVELENGTHS = 0.5
CARRIER_HZ = 3.5e9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(scenario_dir: Path | None, cache_dir: Path, source_archive: Path | None) -> tuple[Path, Path]:
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
        os.chdir(cache_dir)
        if not resolved.is_dir():
            dm.download(SCENARIO, output_dir=str(cache_dir))
    finally:
        os.chdir(old_cwd)
    archives = sorted(cache_dir.rglob(f"{SCENARIO}*downloaded.zip"))
    if not resolved.is_dir() or not archives:
        raise FileNotFoundError(f"DeepMIMO did not produce the expected scenario under {cache_dir}")
    return resolved, archives[0]


def _extract(scenario_dir: Path, features_path: Path, sample_count: int) -> dict[str, Any]:
    import deepmimo as dm

    package_version = importlib.metadata.version("deepmimo")
    if package_version != DEEPMIMO_VERSION:
        raise RuntimeError(f"expected deepmimo=={DEEPMIMO_VERSION}, found {package_version}")

    previous = dm.config.get("scenarios_folder")
    dm.config.set("scenarios_folder", str(scenario_dir.parent))
    try:
        dataset = dm.load(
            str(scenario_dir),
            matrices=[
                "rx_pos", "tx_pos", "power", "phase", "delay",
                "aoa_az", "aoa_el", "aod_az", "aod_el",
            ],
            max_paths=MAX_PATHS,
        )
    finally:
        dm.config.set("scenarios_folder", previous)

    power = np.asarray(dataset.power)  # [n_rx, MAX_PATHS], dBW per path (NaN = no path)
    candidate_count = int(power.shape[0])
    valid = np.flatnonzero(np.any(np.isfinite(power), axis=1))
    if sample_count > len(valid):
        raise ValueError(f"requested {sample_count} rows but only {len(valid)} receivers have paths")
    offsets = np.linspace(0, len(valid) - 1, sample_count, dtype=np.int64)
    receiver_indices = valid[offsets]

    rx_pos = np.asarray(dataset.rx_pos)
    tx_pos = np.asarray(dataset.tx_pos)[0]
    phase = np.asarray(dataset.phase)
    delay = np.asarray(dataset.delay)
    aod_az, aod_el = np.asarray(dataset.aod_az), np.asarray(dataset.aod_el)
    aoa_az, aoa_el = np.asarray(dataset.aoa_az), np.asarray(dataset.aoa_el)

    def _round(values: np.ndarray, ndigits: int) -> list[float]:
        return [round(float(v), ndigits) for v in values]

    all_aod_az: list[float] = []
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("w", encoding="utf-8", newline="\n") as stream:
        for rx in receiver_indices:
            finite = np.flatnonzero(np.isfinite(power[rx]))
            # Strongest-first path ordering (deterministic).
            order = finite[np.argsort(power[rx][finite])[::-1]]
            paths = []
            for p in order:
                paths.append(
                    {
                        "power_dbw": round(float(power[rx][p]), 4),
                        "phase_deg": round(float(phase[rx][p]), 4),
                        "delay_ns": round(float(delay[rx][p]) * 1e9, 4),
                        "aod_az": round(float(aod_az[rx][p]), 4),
                        "aod_el": round(float(aod_el[rx][p]), 4),
                        "aoa_az": round(float(aoa_az[rx][p]), 4),
                        "aoa_el": round(float(aoa_el[rx][p]), 4),
                    }
                )
                all_aod_az.append(float(aod_az[rx][p]))
            record = {
                "receiver_index": int(rx),
                "position_m": _round(rx_pos[rx], 4),
                "n_paths": int(len(paths)),
                "paths": paths,
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return {
        "candidate_receivers": candidate_count,
        "receivers_with_paths": int(len(valid)),
        "sampled_receivers": int(sample_count),
        "tx_position_m": _round(tx_pos, 4),
        "aod_az_deg_min": round(float(np.min(all_aod_az)), 4),
        "aod_az_deg_max": round(float(np.max(all_aod_az)), 4),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    scenario_dir, source_archive = _resolve(args.scenario_dir, args.cache_dir, args.source_archive)
    archive_sha = _sha256_file(source_archive)
    if archive_sha != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"DeepMIMO scenario archive failed its pinned checksum: expected "
            f"{SOURCE_ARCHIVE_SHA256}, got {archive_sha} for {source_archive}"
        )
    stats = _extract(scenario_dir, args.features, args.samples)
    manifest = {
        "dataset": "DeepMIMO ASU Campus 3.5 GHz (angular ray geometry)",
        "scenario": SCENARIO,
        "data_kind": "site-specific Wireless InSite ray tracing; not over-the-air capture",
        "deepmimo_version": DEEPMIMO_VERSION,
        "deepmimo_release_commit": DEEPMIMO_RELEASE_COMMIT,
        "deepmimo_release_url": DEEPMIMO_RELEASE_URL,
        "scenario_docs_url": SCENARIO_DOCS_URL,
        "source_archive_sha256": archive_sha,
        "source_tree_sha256": compute_dataset_sha256(scenario_dir),
        "features_sha256": compute_dataset_sha256(args.features),
        "features_sha256_scope": (
            "Exact JSONL bytes for this build. features_sha256 is host-dependent "
            "(gains/angles rounded to 4 decimals); cross-host reproduction is "
            "verified on source_tree_sha256 plus tolerance-bounded outputs."
        ),
        "features_committed": False,
        "bs_array": {
            "elements": BS_ARRAY_ELEMENTS,
            "spacing_wavelengths": BS_ARRAY_SPACING_WAVELENGTHS,
            "carrier_hz": CARRIER_HZ,
            "geometry": "uniform linear array (ULA)",
        },
        "licensing": {
            "deepmimo_tool": "Apache-2.0",
            "scenario_archive": (
                "No separate license file was present in the downloaded scenario archive; "
                "raw and row-level derived data are therefore not redistributed here."
            ),
        },
        "paths_per_receiver": MAX_PATHS,
        "receiver_sampling": "evenly spaced over receivers with >=1 finite path",
        **stats,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/horizon-deepmimo"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/angular_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/angular_manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
