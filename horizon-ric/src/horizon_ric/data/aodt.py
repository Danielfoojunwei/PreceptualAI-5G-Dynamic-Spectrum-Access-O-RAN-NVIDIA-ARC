"""Adapter for NVIDIA Aerial Omniverse Digital Twin (AODT) scenario bundles.

AODT bundles ship as Pixar-format USD files (`.usd` / `.usda` / `.usdc`)
describing the deployed network topology — UE clusters, gNB/cell sites,
satellite ephemeris, beam footprints. This loader uses **real OpenUSD**
(`pxr.Usd`, `pxr.UsdGeom`) to walk the prim hierarchy and extract real
ECEF positions from `xformOp:translate`.

Per-prim type comes from one of:

  * `customData["aodt_type"]` on the prim (preferred — matches the AODT
    schema convention),
  * the prim's path / name (e.g. `/World/UE/UE_0` → "ue",
    `/World/Sat/Sat_LEO_0` → "satellite"),
  * the default kind "other" if neither yields a match.

A complementary JSON loader is retained for convenience (some AODT
exports include sidecar `*.json` topology files alongside the `.usd`),
but the canonical extraction path is real USD parsing.

There is no stand-in. If `pxr.Usd` cannot be imported, opening a USD
file raises ImportError with install instructions.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from horizon_ric.io.schemas import TelemetryEvent

logger = logging.getLogger(__name__)


_KNOWN_KINDS = {"ue", "cell", "gnb", "satellite", "beam", "gateway"}


_USD_INSTALL_HINT = (
    "OpenUSD Python bindings (pxr.Usd) are required to parse AODT .usd / "
    ".usda scene files. Install with `pip install usd-core` on "
    "x86_64 Linux / macOS / Windows. On aarch64 hosts (e.g. Jetson, "
    "GB10) usd-core has no PyPI wheel; build from source via "
    "`python OpenUSD/build_scripts/build_usd.py --no-imaging --no-tools "
    "<install_dir>` and add `<install_dir>/lib/python` to PYTHONPATH."
)


def _import_pxr() -> tuple[Any, Any]:
    """Return (Usd, UsdGeom) modules or raise ImportError with install hint."""
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(_USD_INSTALL_HINT) from exc
    return Usd, UsdGeom


class AODTScenario:
    """Walk an AODT bundle and produce entity / event views.

    Use:
        scen = AODTScenario("/path/to/aodt_bundle")
        if scen:
            sp_input = scen.to_spatial_prior_input()
            for ev in scen.to_telemetry_events():
                bus.publish(ev)
        else:
            log.info("No AODT scenario entities found at %s", scen.path)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entities: dict[str, list[dict[str, Any]]] = {
            "ues": [],
            "cells": [],
            "satellites": [],
            "beams": [],
            "other": [],
        }
        self._files_found: dict[str, list[str]] = {
            "json": [],
            "usd": [],
            "parquet": [],
            "binary": [],
        }
        self._load()

    def __bool__(self) -> bool:
        return any(len(v) > 0 for v in self.entities.values())

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.entities.values())
        return (
            f"AODTScenario(path={self.path!s}, entities={n}, "
            f"json={len(self._files_found['json'])}, "
            f"usd={len(self._files_found['usd'])})"
        )

    # ── Loaders ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning("AODT path does not exist: %s", self.path)
            return

        # Allow path to be a single .usd / .usda file, not just a directory.
        if self.path.is_file():
            self._load_one(self.path)
            return

        if not self.path.is_dir():
            logger.warning("AODT path is neither file nor directory: %s", self.path)
            return

        for p in sorted(self.path.rglob("*")):
            if p.is_dir():
                continue
            self._load_one(p)

        if not self:
            logger.info(
                "AODTScenario at %s parsed %d JSON / %d USD / %d parquet "
                "/ %d opaque files but found no scenario entities. "
                "Check that the bundle contains a .usd/.usda scene "
                "describing UE / cell / satellite Xforms.",
                self.path,
                len(self._files_found["json"]),
                len(self._files_found["usd"]),
                len(self._files_found["parquet"]),
                len(self._files_found["binary"]),
            )

    def _load_one(self, p: Path) -> None:
        suffix = p.suffix.lower()
        if suffix == ".json":
            self._files_found["json"].append(str(p))
            self._load_json(p)
        elif suffix in (".usd", ".usda", ".usdc"):
            self._files_found["usd"].append(str(p))
            self._load_usd(p)
        elif suffix == ".parquet":
            self._files_found["parquet"].append(str(p))
            self._load_parquet(p)
        else:
            self._files_found["binary"].append(str(p))
            logger.debug(
                "AODT: opaque file %s (%s) — no scenario entities",
                p.name,
                suffix or "no-ext",
            )

    def _load_json(self, p: Path) -> None:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("AODT: %s not parseable as JSON (%s)", p, exc)
            return

        if isinstance(data, dict):
            for key, target in (
                ("ues", "ues"),
                ("UEs", "ues"),
                ("cells", "cells"),
                ("gNBs", "cells"),
                ("satellites", "satellites"),
                ("beams", "beams"),
            ):
                if key in data and isinstance(data[key], list):
                    for raw in data[key]:
                        ent = self._normalise_entity(raw, default_kind=target[:-1])
                        if ent is not None:
                            self.entities[target].append(ent)
            return

        if isinstance(data, list):
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                ent = self._normalise_entity(raw, default_kind="other")
                if ent is None:
                    continue
                bucket = self._bucket_for_kind(ent["kind"])
                self.entities[bucket].append(ent)

    def _load_usd(self, p: Path) -> None:
        Usd, UsdGeom = _import_pxr()
        stage = Usd.Stage.Open(str(p))
        if stage is None:
            logger.warning("AODT: USD stage failed to open: %s", p)
            return

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Xform):
                continue
            translate = self._read_translate(prim, UsdGeom)
            if translate is None:
                continue
            kind = self._kind_for_prim(prim)
            ent: dict[str, Any] = {
                "id": prim.GetPath().pathString,
                "kind": kind,
                "position_xyz_m": [
                    float(translate[0]),
                    float(translate[1]),
                    float(translate[2]),
                ],
                "placement": "ecef",
            }
            self.entities[self._bucket_for_kind(kind)].append(ent)

    @staticmethod
    def _read_translate(prim: Any, UsdGeom: Any) -> tuple[float, float, float] | None:
        """Read xformOp:translate from a UsdGeom.Xform prim, if present."""
        # Direct attribute lookup is the most robust across USD versions.
        attr = prim.GetAttribute("xformOp:translate")
        if attr and attr.HasAuthoredValue():
            tx = attr.Get()
            if tx is not None and len(tx) == 3:
                return float(tx[0]), float(tx[1]), float(tx[2])

        # Fall back to UsdGeom.Xformable to handle composed xform ops.
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                tx = op.Get()
                if tx is not None and len(tx) == 3:
                    return float(tx[0]), float(tx[1]), float(tx[2])
        return None

    @classmethod
    def _kind_for_prim(cls, prim: Any) -> str:
        # 1) customData["aodt_type"] — the canonical AODT schema field.
        try:
            cd = prim.GetCustomData()
            if cd and "aodt_type" in cd:
                k = str(cd["aodt_type"]).lower()
                if k in _KNOWN_KINDS:
                    return k
        except Exception:
            pass
        # 2) Path-based inference: /World/UE/... → "ue", etc.
        path = prim.GetPath().pathString.lower()
        for k in ("satellite", "sat", "ue", "cell", "gnb", "beam", "gateway"):
            if f"/{k}" in path or path.endswith(f"/{k}"):
                return "satellite" if k == "sat" else k
        # 3) Fallback to name heuristic.
        name = prim.GetName().lower()
        for k in ("satellite", "sat", "ue", "cell", "gnb", "beam", "gateway"):
            if k in name:
                return "satellite" if k == "sat" else k
        return "other"

    def _load_parquet(self, p: Path) -> None:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            logger.debug(
                "AODT: parquet %s found but pyarrow not installed; skipping",
                p,
            )
            return
        try:
            table = pq.read_table(p)  # pragma: no cover
        except Exception as exc:  # pragma: no cover
            logger.warning("AODT: parquet %s unreadable: %s", p, exc)
            return
        cols = set(table.column_names)  # pragma: no cover
        if {"id", "x", "y", "z"}.issubset(cols):  # pragma: no cover
            df = table.to_pandas()
            for _, row in df.iterrows():
                ent = {
                    "id": str(row["id"]),
                    "kind": str(row.get("kind", "other")),
                    "position_xyz_m": [
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    ],
                    "placement": "ecef",
                }
                self.entities[self._bucket_for_kind(ent["kind"])].append(ent)

    # ── Normalisation helpers ────────────────────────────────────────

    @staticmethod
    def _bucket_for_kind(kind: str) -> str:
        kind = kind.lower()
        if kind in ("ue", "user_equipment"):
            return "ues"
        if kind in ("cell", "gnb", "gnodeb"):
            return "cells"
        if kind in ("satellite", "sat"):
            return "satellites"
        if kind == "beam":
            return "beams"
        return "other"

    @staticmethod
    def _guess_kind_from_name(name: str) -> str:
        n = name.lower()
        for k in ("satellite", "sat", "ue", "cell", "gnb", "beam", "gateway"):
            if k in n:
                return k
        return "other"

    def _normalise_entity(
        self, raw: dict[str, Any], default_kind: str = "other"
    ) -> dict[str, Any] | None:
        ent_id = (
            raw.get("id")
            or raw.get("name")
            or raw.get("identifier")
        )
        kind_raw = str(raw.get("kind") or raw.get("type") or default_kind).lower()
        if kind_raw not in _KNOWN_KINDS:
            kind_raw = self._guess_kind_from_name(kind_raw) or default_kind

        pos: list[float] | None = None
        if "position_xyz_m" in raw and len(raw["position_xyz_m"]) == 3:
            pos = [float(x) for x in raw["position_xyz_m"]]
        elif {"x", "y", "z"}.issubset(raw):
            pos = [float(raw["x"]), float(raw["y"]), float(raw["z"])]
        elif "ecef" in raw and isinstance(raw["ecef"], (list, tuple)):
            pos = [float(x) for x in raw["ecef"]]

        if ent_id is None and pos is None:
            return None

        ent: dict[str, Any] = {
            "id": str(ent_id) if ent_id is not None else f"aodt-{uuid.uuid4().hex[:8]}",
            "kind": kind_raw,
            "position_xyz_m": pos if pos is not None else [0.0, 0.0, 0.0],
        }
        if "lat_deg" in raw and "lon_deg" in raw:
            ent["lat_deg"] = float(raw["lat_deg"])
            ent["lon_deg"] = float(raw["lon_deg"])
            ent["height_m"] = float(raw.get("height_m", 0.0))
            ent["placement"] = "ground"
        else:
            ent["placement"] = "ecef"
        return ent

    # ── Output adapters ──────────────────────────────────────────────

    def all_entities(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bucket in self.entities.values():
            out.extend(bucket)
        return out

    def to_spatial_prior_input(self) -> list[dict[str, Any]]:
        from horizon_ric.planner.physics.geodesy import ECEF  # local import

        out: list[dict[str, Any]] = []
        for ent in self.all_entities():
            if ent.get("placement") == "ground" and {
                "lat_deg",
                "lon_deg",
            }.issubset(ent):
                out.append(
                    {
                        "placement": "ground",
                        "lat_deg": ent["lat_deg"],
                        "lon_deg": ent["lon_deg"],
                        "height_m": ent.get("height_m", 0.0),
                        "id": ent["id"],
                        "kind": ent["kind"],
                    }
                )
            else:
                x, y, z = ent.get("position_xyz_m", [0.0, 0.0, 0.0])
                out.append(
                    {
                        "placement": "ecef",
                        "ecef": ECEF(float(x), float(y), float(z)),
                        "id": ent["id"],
                        "kind": ent["kind"],
                    }
                )
        return out

    def to_telemetry_events(
        self,
        source_id: str = "aodt-scenario",
        t0_utc: datetime | None = None,
    ) -> Iterator[TelemetryEvent]:
        if t0_utc is None:
            t0_utc = datetime.now(timezone.utc)
        if t0_utc.tzinfo is None:
            raise ValueError("t0_utc must be timezone-aware UTC")
        for sat in self.entities["satellites"]:
            yield TelemetryEvent(
                event_id=str(uuid.uuid4()),
                modality="ephemeris_oem",
                source_id=source_id,
                ts_utc=t0_utc,
                payload={
                    "satellite_id": sat["id"],
                    "position_ecef_m": sat.get(
                        "position_xyz_m", [0.0, 0.0, 0.0]
                    ),
                    "kind": "satellite",
                },
                tags={"origin": "aodt"},
            )


__all__ = ["AODTScenario"]
