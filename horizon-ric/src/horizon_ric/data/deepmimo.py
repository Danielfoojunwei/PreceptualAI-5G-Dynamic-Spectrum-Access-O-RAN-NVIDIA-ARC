"""DeepMIMO scenario reader.

DeepMIMO drops are arranged as one directory per ray-tracing run, e.g.
``data/deepmimo/asu_campus_3p5_dyn/insite_3.5ghz_5r_1d_0s/``. Each
directory carries a `params.json` (ray-tracer config), an
`objects.json` (scene geometry), and a fan of `.npz` files holding the
ray-traced channel components keyed by `<field>_t<TXi>_tx<TXj>_r<RXk>`.

In some drops the channels are stored as `.mat` files instead of
`.npz`; both are supported. Each scenario emits one
`TelemetryEvent(modality="kpm_ntn")` carrying the params, the
objects-file path, the list of UE positions, and a manifest of
available channel files.

Multi-scenario suite
====================
The reader knows about the following standard DeepMIMO scenarios; each
maps to one top-level directory under ``data/deepmimo/<scenario_id>/``.
Sub-directories with ``params.json`` are treated as scene drops; if the
scenario directory itself carries ``params.json`` it counts as a single
scene.

  asu_campus_3p5    – ASU campus, sub-6 GHz (3.5 GHz)
  boston5g_28       – Boston, mmWave 28 GHz
  boston5g_3p5      – Boston, sub-6 GHz 3.5 GHz
  city_0_newyork_28 – Manhattan, mmWave 28 GHz
  city_01_villas_florida_1gp – Florida villa scenario
  asu_campus_3p5_dyn – ASU campus dynamic (multi-time-step)

`DeepMIMOScenarioReader.list_known_scenarios()` returns the list, and
:func:`iterate_all_scenarios` iterates every scene from every locally
present scenario.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorIOError,
    ConnectorState,
    Source,
)
from horizon_ric.io.schemas import Modality, TelemetryEvent

# ─── multi-scenario registry ────────────────────────────────────────────


@dataclass(frozen=True)
class ScenarioSpec:
    """One known DeepMIMO scenario and where to find its public mirror."""

    scenario_id: str           # local directory name under data/deepmimo/
    description: str
    hf_repo: str               # HuggingFace dataset repo, e.g. "DeepMIMO/asu_campus_3p5"
    hf_revision: str = "main"


_KNOWN_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        scenario_id="asu_campus_3p5",
        description="ASU campus, sub-6 GHz (3.5 GHz) static ray-tracing",
        hf_repo="DeepMIMO/asu_campus_3p5",
    ),
    ScenarioSpec(
        scenario_id="boston5g_28",
        description="Boston, mmWave 28 GHz",
        hf_repo="DeepMIMO/Boston5G_28",
    ),
    ScenarioSpec(
        scenario_id="boston5g_3p5",
        description="Boston, sub-6 GHz 3.5 GHz",
        hf_repo="DeepMIMO/Boston5G_3p5",
    ),
    ScenarioSpec(
        scenario_id="city_0_newyork_28",
        description="Manhattan, mmWave 28 GHz",
        hf_repo="DeepMIMO/city_0_newyork_28",
    ),
    ScenarioSpec(
        scenario_id="city_01_villas_florida_1gp",
        description="Florida villas, sub-6 GHz",
        hf_repo="DeepMIMO/city_01_villas_florida_1gp",
    ),
    ScenarioSpec(
        scenario_id="asu_campus_3p5_dyn",
        description="ASU campus dynamic (multi-time-step)",
        hf_repo="DeepMIMO/asu_campus_3p5_dyn",
    ),
)


def _default_root() -> Path:
    """Top-level directory where every scenario lives."""
    return Path("/home/danielfoojunwei/Preceptualv1/data/deepmimo")


def _load_npz_first(path: Path) -> tuple[str, np.ndarray] | None:
    try:
        with np.load(path) as d:
            keys = list(d.keys())
            if not keys:
                return None
            return keys[0], np.asarray(d[keys[0]])
    except Exception:  # pragma: no cover - tolerate corrupted files
        return None


class DeepMIMOScenarioReader:
    """Iterate scenarios under a DeepMIMO directory.

    Parameters
    ----------
    scenarios_dir
        Top-level directory containing one sub-directory per scene-step
        (e.g. ``asu_campus_3p5_dyn``). Each sub-dir must have a
        `params.json`.
    """

    modality: Modality = "kpm_ntn"

    def __init__(
        self,
        scenarios_dir: str | Path,
        source_id: str = "deepmimo",
        max_scenes: int | None = None,
    ):
        self.dir = Path(scenarios_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"DeepMIMO directory not found: {self.dir}")
        self.source_id = source_id
        # Discover scene directories: subfolders that contain `params.json`.
        scenes = [
            p for p in sorted(self.dir.iterdir())
            if p.is_dir() and (p / "params.json").exists()
        ]
        # If the scenario directory itself carries `params.json`, treat it as
        # a single (flat) scene drop. This is how the static asu_campus_3p5,
        # boston5g_28, etc. drops are organised.
        if not scenes and (self.dir / "params.json").exists():
            scenes = [self.dir]
        if not scenes:
            raise ConnectorIOError(
                f"no DeepMIMO scenes found in {self.dir} "
                "(expected sub-directories with params.json, or a flat "
                "scenario directory containing params.json directly)"
            )
        if max_scenes is not None:
            scenes = scenes[:max_scenes]
        self._scenes: list[Path] = scenes

    @staticmethod
    def list_known_scenarios() -> list[ScenarioSpec]:
        """Return the canonical 6 DeepMIMO scenarios this codebase knows."""
        return list(_KNOWN_SCENARIOS)

    @staticmethod
    def iterate_all_scenarios(
        root: str | Path | None = None,
        max_scenes_per_dataset: int | None = None,
    ) -> Iterator[TelemetryEvent]:
        """Iterate every locally-present known scenario; emit one
        `TelemetryEvent` per scene under each scenario's sub-tree.

        Scenarios that are not on disk are silently skipped; callers can use
        :py:meth:`list_known_scenarios` to discover what is missing and use
        ``scripts/download_deepmimo.py`` to fetch.
        """
        base = Path(root) if root is not None else _default_root()
        for spec in _KNOWN_SCENARIOS:
            scenario_dir = base / spec.scenario_id
            if not scenario_dir.exists():
                continue
            try:
                reader = DeepMIMOScenarioReader(
                    scenario_dir,
                    source_id=f"deepmimo:{spec.scenario_id}",
                    max_scenes=max_scenes_per_dataset,
                )
            except ConnectorIOError:
                continue
            yield from reader

    @property
    def scenes(self) -> list[Path]:
        return list(self._scenes)

    def _scan_scene(self, scene: Path) -> dict:
        params = json.loads((scene / "params.json").read_text())
        objects_path = scene / "objects.json"
        # Channel manifests: enumerate matrix-bearing files.
        channel_files: list[str] = []
        for ext in ("*.npz", "*.mat"):
            channel_files.extend(sorted(p.name for p in scene.glob(ext)))
        # UE positions: prefer the merged rx-positions array if present.
        positions: list[list[float]] = []
        rx_pos_files = sorted(scene.glob("rx_pos_*.npz"))
        if rx_pos_files:
            loaded = _load_npz_first(rx_pos_files[0])
            if loaded is not None:
                _, arr = loaded
                # Take a small bounded sample to keep payload manageable.
                head = arr[: min(len(arr), 16)]
                positions = head.tolist() if hasattr(head, "tolist") else list(head)
        return {
            "scene_dir": str(scene),
            "params": params,
            "objects_path": str(objects_path) if objects_path.exists() else None,
            "channel_files": channel_files,
            "n_channel_files": len(channel_files),
            "positions": positions,
        }

    def __iter__(self) -> Iterator[TelemetryEvent]:
        ts0 = datetime.now(tz=timezone.utc)
        for idx, scene in enumerate(self._scenes):
            payload = self._scan_scene(scene)
            yield TelemetryEvent(
                event_id=f"deepmimo-{scene.name}-{idx}",
                modality=self.modality,
                source_id=self.source_id,
                ts_utc=ts0,
                sequence=idx,
                payload=payload,
                tags={"scene": scene.name, "format": "deepmimo"},
            )


# ─── registry adapter ────────────────────────────────────────────────────


class _DeepMIMOConfig(ConnectorConfig):
    scenarios_dir: str
    max_scenes: int | None = None


class DeepMIMOSource(Source):
    """Async `Source` wrapper around `DeepMIMOScenarioReader`."""

    Config = _DeepMIMOConfig
    cfg: _DeepMIMOConfig

    async def connect(self) -> None:
        self._reader = DeepMIMOScenarioReader(
            self.cfg.scenarios_dir, max_scenes=self.cfg.max_scenes
        )
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def stream(self):
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("DeepMIMOSource not connected")
        for ev in self._reader:
            yield ev


def summarize_scenario(scenario_dir: str | Path) -> dict:
    """Inspect a single DeepMIMO scenario directory and return a JSON-able
    summary suitable for ``data/deepmimo/manifest.json``.

    The returned dict has the keys:

    * ``path``           — absolute scenario path
    * ``scene_name``     — scenario directory name
    * ``n_scenes``       — number of scene drops (1 for flat scenarios)
    * ``n_ues``          — number of receivers (UEs) detected
    * ``n_bs``           — number of distinct transmitter (BS) indices
    * ``n_antennas``     — number of antennas / paths per UE in the
      first power tensor (DeepMIMO stores per-path complex gains;
      ``n_paths`` doubles as the per-UE channel-vector length)
    * ``channel_tensor_shape`` — shape of the first power tensor as a
      tuple of ints, e.g. ``(131931, 10)`` for asu_campus_3p5
    * ``rss_min`` / ``rss_max`` / ``rss_mean`` — RSS distribution over
      the first power tensor (linear scale)
    * ``channel_pickle_sha256`` — sha256 of the first power file
      (the closest thing DeepMIMO has to a "channel pickle"); ``None``
      if no power file is present.
    """
    import hashlib
    import re

    path = Path(scenario_dir)
    if not path.exists():
        raise FileNotFoundError(path)

    # Discover scene drops.
    scenes: list[Path]
    if (path / "params.json").exists():
        scenes = [path]
    else:
        scenes = sorted(
            d for d in path.iterdir() if d.is_dir() and (d / "params.json").exists()
        )
    if not scenes:
        return {
            "path": str(path),
            "scene_name": path.name,
            "n_scenes": 0,
            "n_ues": 0,
            "n_bs": 0,
            "n_antennas": 0,
            "channel_tensor_shape": None,
            "rss_min": None,
            "rss_max": None,
            "rss_mean": None,
            "channel_pickle_sha256": None,
        }

    head = scenes[0]
    files = list(head.iterdir())

    # Distinct tx (BS) and rx (UE-set) indices from filename pattern,
    # which is `<field>_t<TT>_tx<XX>_r<RR>.{mat,npz}`. We anchor the
    # regex on the underscore separators to avoid the "tx" inside the
    # "r" capture.
    tx_idx: set[int] = set()
    rx_idx: set[int] = set()
    t_idx: set[int] = set()
    for f in files:
        m_tx = re.search(r"_tx(\d+)_", f.name)
        m_rx = re.search(r"_r(\d+)\.", f.name)
        m_t = re.search(r"_t(\d+)_", f.name)
        if m_tx:
            tx_idx.add(int(m_tx.group(1)))
        if m_rx:
            rx_idx.add(int(m_rx.group(1)))
        if m_t:
            t_idx.add(int(m_t.group(1)))

    # First power tensor is the canonical "channel" surrogate.
    power_files = sorted(
        [f for f in files if f.name.startswith("power_") and f.suffix in (".mat", ".npz")]
    )
    rx_pos_files = sorted(
        [f for f in files if f.name.startswith("rx_pos_") and f.suffix in (".mat", ".npz")]
    )

    channel_shape: tuple[int, ...] | None = None
    rss_stats: dict[str, float] = {}
    channel_sha: str | None = None
    n_ues = 0
    n_antennas = 0

    if power_files:
        pf = power_files[0]
        if pf.suffix == ".mat":
            from scipy.io import loadmat

            arr = loadmat(str(pf))["power"]
        else:
            with np.load(pf) as d:
                k = list(d.keys())[0]
                arr = np.asarray(d[k])
        channel_shape = tuple(int(x) for x in arr.shape)
        if arr.ndim == 2:
            n_ues, n_antennas = channel_shape  # type: ignore[misc]
        else:
            n_ues = int(arr.shape[0])
            n_antennas = int(np.prod(arr.shape[1:]))
        # RSS distribution: linear-power, then convert min/mean/max to
        # dB-equivalent only if user wants — keep linear here for
        # numerical reproducibility.
        finite = arr[np.isfinite(arr)]
        if finite.size:
            rss_stats = {
                "rss_min": float(np.min(finite)),
                "rss_max": float(np.max(finite)),
                "rss_mean": float(np.mean(finite)),
            }
        h = hashlib.sha256()
        h.update(pf.read_bytes())
        channel_sha = h.hexdigest()
    elif rx_pos_files:
        # Fall back to receiver-position file shape.
        pf = rx_pos_files[0]
        if pf.suffix == ".mat":
            from scipy.io import loadmat

            arr = loadmat(str(pf))["rx_pos"]
        else:
            with np.load(pf) as d:
                k = list(d.keys())[0]
                arr = np.asarray(d[k])
        n_ues = int(arr.shape[0])
        channel_shape = tuple(int(x) for x in arr.shape)

    return {
        "path": str(path),
        "scene_name": path.name,
        "n_scenes": len(scenes),
        "n_ues": int(n_ues),
        "n_bs": len(tx_idx) if tx_idx else 1,
        "n_ue_groups": len(rx_idx),
        "n_time_snapshots": len(t_idx),
        "n_antennas": int(n_antennas),
        "channel_tensor_shape": list(channel_shape) if channel_shape is not None else None,
        "rss_min": rss_stats.get("rss_min"),
        "rss_max": rss_stats.get("rss_max"),
        "rss_mean": rss_stats.get("rss_mean"),
        "channel_pickle_sha256": channel_sha,
    }


__all__ = [
    "DeepMIMOScenarioReader",
    "DeepMIMOSource",
    "ScenarioSpec",
    "summarize_scenario",
]
