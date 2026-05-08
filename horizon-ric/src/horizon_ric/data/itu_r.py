"""ITU-R reference propagation map readers.

ITU publishes gridded reference data alongside Recommendations P.837 (rain
rate), P.836 (water vapor), P.839 (rain height), P.840 (cloud water),
P.453 (wet refractivity / N_wet), and P.1510 (mean surface temperature).
We ship a download script (`scripts/download_itu_data.py`) that drops the
extracted ZIPs under `data/itu_r/<rec_id>/`.

This module exposes those maps as `ITUMapReader` instances and a thin
per-rec subclass for the most common queries (`P837Map`, `P839Map`,
etc.). The readers do bilinear interpolation over the published lat/lon
grid and gracefully fall back to documented constants if the data
files aren't present yet — that way PreceptualAI keeps working before the
download has run, and tests can still execute without network access.

Coordinate conventions:

    * Input is always (lat_deg ∈ [-90, 90], lon_deg ∈ [-180, 180]).
    * The reader normalises lon to whatever range the underlying grid
      uses (P.839 stores 0..360, P.1510 stores -180..180, P.453 stores
      -180..180). Wrap-around at the dateline is handled by extending the
      grid by one column.

All readers are cheap to construct and cache the underlying numpy
arrays, so repeated `lookup()` calls are O(log n) per axis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_LOG = logging.getLogger(__name__)

# Default fallbacks per ITU-R common-engineering values. These match the
# defaults already used in `planner.physics.propagation` so that disabling
# the maps is a no-op for legacy callers.
_FALLBACK_RAIN_HEIGHT_KM = 3.0          # P.839-4 mid-latitude default
_FALLBACK_WATER_VAPOR_G_M3 = 7.5        # P.836 mid-latitude default
_FALLBACK_RAIN_RATE_R001_MM_HR = 30.0   # P.837 typical mid-latitude
_FALLBACK_TEMPERATURE_K = 288.15        # P.1510 ~15°C
_FALLBACK_NWET = 40.0                   # P.453 mid-latitude
_FALLBACK_CLOUD_WATER_KG_M2 = 0.5       # P.840 typical


def _default_data_dir() -> Path:
    """Resolve the on-disk root for downloaded ITU data.

    The download script lives in horizon-ric/, but the data lives under
    Preceptualv1/data/itu_r/ (gitignored). We resolve relative to this
    file so callers don't have to set cwd.
    """
    # src/horizon_ric/data/itu_r.py -> Preceptualv1/data/itu_r
    here = Path(__file__).resolve()
    # Walk up until we find a `data/itu_r` sibling, falling back to
    # Preceptualv1/data/itu_r.
    candidate = here.parents[3] / "data" / "itu_r"
    if candidate.exists():
        return candidate
    candidate = here.parents[4] / "data" / "itu_r"
    return candidate


@dataclass
class _Grid:
    """A 2-D grid sampled on (lat, lon)."""

    lats: np.ndarray   # shape (n_lat,) — monotonic
    lons: np.ndarray   # shape (n_lon,) — monotonic, in either [-180,180] or [0,360]
    values: np.ndarray  # shape (n_lat, n_lon)

    def __post_init__(self) -> None:
        if self.values.shape != (self.lats.size, self.lons.size):
            raise ValueError(
                f"grid shape mismatch: values={self.values.shape}, "
                f"expected=({self.lats.size}, {self.lons.size})"
            )


def _read_matrix(path: Path) -> np.ndarray:
    """Whitespace- or tab-delimited float matrix. Tolerant of mixed seps."""
    return np.loadtxt(path)


def _axis_from_2d(matrix: np.ndarray, axis: str) -> np.ndarray:
    """ITU files store lat/lon as full 2-D meshes; collapse to 1-D."""
    if axis == "lat":
        # Lat usually constant along columns, varies along rows.
        return matrix[:, 0].astype(float)
    elif axis == "lon":
        return matrix[0, :].astype(float)
    else:
        raise ValueError(f"axis must be lat or lon, got {axis}")


def _normalise_grid(grid: _Grid) -> _Grid:
    """Make lat ascending and ensure lon is sorted; values reordered to match."""
    lats = grid.lats
    lons = grid.lons
    values = grid.values

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]
    # If lon is descending, flip too.
    if lons.size > 1 and lons[0] > lons[-1]:
        lons = lons[::-1]
        values = values[:, ::-1]
    return _Grid(lats=lats, lons=lons, values=values)


def _to_grid_lon(lon_deg: float, grid_lons: np.ndarray) -> float:
    """Translate input lon ∈ [-180,180] to whatever convention the grid uses."""
    lo, hi = float(grid_lons[0]), float(grid_lons[-1])
    # Tolerance for grid that wraps to 360 vs 359.x
    if hi > 180.5:
        # Grid stored on [0, 360]
        return lon_deg % 360.0
    # Grid stored on [-180, 180]; just wrap any out-of-range input.
    x = ((lon_deg + 180.0) % 360.0) - 180.0
    return x


def _bilinear(grid: _Grid, lat_deg: float, lon_deg: float) -> float:
    """Bilinear interpolation with longitude wrap-around at the dateline."""
    if not (-90.0 <= lat_deg <= 90.0):
        raise ValueError(f"lat_deg must be in [-90, 90], got {lat_deg}")

    g_lon = _to_grid_lon(lon_deg, grid.lons)

    lats = grid.lats
    lons = grid.lons
    n_lat = lats.size
    n_lon = lons.size

    # Lat index: clamp into [0, n_lat-2].
    i = int(np.searchsorted(lats, lat_deg, side="right") - 1)
    i = max(0, min(i, n_lat - 2))
    lat0, lat1 = float(lats[i]), float(lats[i + 1])
    t_lat = 0.0 if lat1 == lat0 else (lat_deg - lat0) / (lat1 - lat0)
    t_lat = float(np.clip(t_lat, 0.0, 1.0))

    # Lon index with wrap-around.
    j = int(np.searchsorted(lons, g_lon, side="right") - 1)
    if j < 0:
        # Below first lon — wrap to last/first pair.
        j0 = n_lon - 1
        j1 = 0
        # Distance from lons[j0] (upper end) to g_lon, treating wrap.
        # Compute fractional position as if lons extended.
        dlon = (lons[1] - lons[0]) if n_lon > 1 else 1.0
        # Use lons[j0]-360 as the "left" sample if grid is 0..360.
        if lons[-1] > 180.5:
            left = float(lons[j0]) - 360.0
        else:
            left = float(lons[j0]) - 360.0
        right = float(lons[j1])
        t_lon = (g_lon - left) / (right - left) if right != left else 0.0
    elif j >= n_lon - 1:
        # At or past last lon — wrap forward.
        j0 = n_lon - 1
        j1 = 0
        left = float(lons[j0])
        if lons[-1] > 180.5:
            right = float(lons[j1]) + 360.0
        else:
            right = float(lons[j1]) + 360.0
        t_lon = (g_lon - left) / (right - left) if right != left else 0.0
    else:
        j0, j1 = j, j + 1
        left, right = float(lons[j0]), float(lons[j1])
        t_lon = (g_lon - left) / (right - left) if right != left else 0.0

    t_lon = float(np.clip(t_lon, 0.0, 1.0))

    v00 = float(grid.values[i, j0])
    v01 = float(grid.values[i, j1])
    v10 = float(grid.values[i + 1, j0])
    v11 = float(grid.values[i + 1, j1])

    v0 = v00 * (1.0 - t_lon) + v01 * t_lon
    v1 = v10 * (1.0 - t_lon) + v11 * t_lon
    return float(v0 * (1.0 - t_lat) + v1 * t_lat)


# Per-rec data-discovery rules. Each entry tells the loader where to find
# the lat, lon, and value matrices for a given (rec_id, percentage_time).
# `value_chooser` is a callable receiving `(percentage_time, files_list)`
# and returning the file to use, or None if none match.
def _list_text_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".txt", ".csv"}])


def _find_lat_lon(files: list[Path]) -> tuple[Optional[Path], Optional[Path]]:
    lat = lon = None
    for p in files:
        name = p.name.lower()
        if name.startswith("lat"):
            lat = lat or p
        elif name.startswith("lon"):
            lon = lon or p
        elif name.startswith("lat_t"):
            lat = lat or p
        elif name.startswith("lon_t"):
            lon = lon or p
    return lat, lon


class ITUMapReader:
    """Reader for an ITU-R published gridded data file.

    The reader looks under `<data_dir>/<rec_id>/` for the lat/lon mesh and
    a value file matching the requested `percentage_time` (or annual map).
    If the data isn't present, `lookup()` returns the documented fallback
    constant for the rec_id and emits a single warning per process.

    Args:
        rec_id: ITU recommendation ID, e.g. "P.839-4".
        data_dir: root directory containing the extracted ZIPs. Defaults
            to `Preceptualv1/data/itu_r`.
    """

    _warned_missing: set[str] = set()

    def __init__(self, rec_id: str, data_dir: Optional[Path] = None) -> None:
        self.rec_id = rec_id
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        self.rec_dir = self.data_dir / rec_id
        self._grid_cache: dict[str, _Grid] = {}
        self._available = self.rec_dir.exists() and any(
            p for p in self.rec_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".csv"}
        )
        if not self._available and not self.rec_dir.exists():
            # Sometimes the dir exists but with TXT files in subdirs only —
            # rglob already covers that, so this is just a directory check.
            pass

    # ---- file selection -------------------------------------------------

    # Subclasses override this with a substring (lower-case) that the
    # value-file name must contain. None means "no constraint".
    _value_name_prefer: Optional[str] = None

    def _value_path(self, percentage_time: Optional[float]) -> Optional[Path]:
        """Pick the value file matching `percentage_time` (or annual)."""
        if not self._available:
            return None
        files = _list_text_files(self.rec_dir)
        # Exclude lat/lon meshes, land-sea masks, readmes.
        candidates = [p for p in files if not p.name.lower().startswith(("lat", "lon"))
                      and "landsea" not in p.name.lower()
                      and "readme" not in p.name.lower()]
        if self._value_name_prefer is not None:
            preferred = [p for p in candidates
                         if self._value_name_prefer in p.name.lower()
                         or self._value_name_prefer in str(p).lower()]
            if preferred:
                candidates = preferred
        if not candidates:
            return None

        if percentage_time is None:
            # Prefer the "annual" / "T_Annual" / "h0" type files.
            for kw in ("annual", "h0", "_annual", "mean"):
                for p in candidates:
                    if kw in p.name.lower():
                        return p
            return candidates[0]

        # Match files like RR_001_v3.txt, DN_00d10_v1.txt — encoded as ddd
        # or NNdNN with a percentage. Accept percentage as float.
        target = percentage_time
        # Encode "0.01" as "00d01", "0.1" as "00d10", "1.0" as "01d00", etc.
        def _encode(p: float) -> str:
            integer = int(p)
            frac = int(round((p - integer) * 100))
            return f"{integer:02d}d{frac:02d}"

        encoded = _encode(target)
        for p in candidates:
            if encoded in p.name.lower().replace(".", ""):
                return p
        # Fall back to numerical match: look at trailing digits in stem.
        best = None
        best_diff = float("inf")
        rx = re.compile(r"(\d+(?:[._]\d+)?)")
        for p in candidates:
            stem = p.stem.replace("d", ".")
            for m in rx.findall(stem):
                try:
                    val = float(m.replace("_", "."))
                except ValueError:
                    continue
                d = abs(val - target)
                if d < best_diff:
                    best_diff = d
                    best = p
        return best

    # ---- grid loading ---------------------------------------------------

    def _load_grid(self, value_path: Path) -> _Grid:
        cache_key = str(value_path)
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        # Lat/Lon files live next to the value file, or one level up.
        search_dirs = [value_path.parent, self.rec_dir]
        lat_path = lon_path = None
        for d in search_dirs:
            files = _list_text_files(d)
            la, lo = _find_lat_lon(files)
            if la is not None and lat_path is None:
                lat_path = la
            if lo is not None and lon_path is None:
                lon_path = lo
            if lat_path and lon_path:
                break

        if lat_path is None or lon_path is None:
            raise FileNotFoundError(
                f"No lat/lon mesh under {self.rec_dir}; "
                f"expected files starting with 'Lat' and 'Lon'."
            )

        lat_mat = _read_matrix(lat_path)
        lon_mat = _read_matrix(lon_path)
        values = _read_matrix(value_path)

        # ITU lat/lon files are typically 2-D meshes matching the value
        # matrix shape. Collapse to 1-D using the appropriate axis.
        if lat_mat.ndim == 1:
            lats = lat_mat
        else:
            lats = _axis_from_2d(lat_mat, "lat")
        if lon_mat.ndim == 1:
            lons = lon_mat
        else:
            lons = _axis_from_2d(lon_mat, "lon")

        grid = _Grid(lats=lats, lons=lons, values=values)
        grid = _normalise_grid(grid)
        self._grid_cache[cache_key] = grid
        return grid

    # ---- public API -----------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the data ZIP has been downloaded and extracted."""
        return self._available

    def fallback_value(self) -> float:
        """Documented engineering default when the data isn't on disk."""
        rid = self.rec_id.split("-")[0]
        return {
            "P.837": _FALLBACK_RAIN_RATE_R001_MM_HR,
            "P.836": _FALLBACK_WATER_VAPOR_G_M3,
            "P.839": _FALLBACK_RAIN_HEIGHT_KM,
            "P.840": _FALLBACK_CLOUD_WATER_KG_M2,
            "P.453": _FALLBACK_NWET,
            "P.1510": _FALLBACK_TEMPERATURE_K,
        }.get(rid, float("nan"))

    def _warn_missing_once(self) -> None:
        if self.rec_id in ITUMapReader._warned_missing:
            return
        ITUMapReader._warned_missing.add(self.rec_id)
        _LOG.warning(
            "ITU-R %s data not found under %s — using fallback value %s. "
            "Run scripts/download_itu_data.py for high-fidelity lookups.",
            self.rec_id, self.rec_dir, self.fallback_value(),
        )

    def lookup(
        self,
        lat_deg: float,
        lon_deg: float,
        percentage_time: Optional[float] = None,
    ) -> float:
        """Bilinear lookup at (lat, lon).

        Args:
            lat_deg: latitude in degrees, [-90, 90].
            lon_deg: longitude in degrees, accepts [-180, 180] or [0, 360].
            percentage_time: only used for P.837 (R0.01 vs R0.001) and
                similar percentage-indexed maps; ignored otherwise.

        Returns:
            Interpolated value, or the documented fallback constant if the
            data files aren't on disk.
        """
        if not self._available:
            self._warn_missing_once()
            return self.fallback_value()

        try:
            value_path = self._value_path(percentage_time)
            if value_path is None:
                self._warn_missing_once()
                return self.fallback_value()
            grid = self._load_grid(value_path)
            return _bilinear(grid, lat_deg, lon_deg)
        except (FileNotFoundError, ValueError, OSError) as exc:
            _LOG.warning(
                "ITU-R %s lookup failed (%s); falling back to %s.",
                self.rec_id, exc, self.fallback_value(),
            )
            return self.fallback_value()


# ---------------- Convenience subclasses ---------------------------------


class P837Map(ITUMapReader):
    """Annual rainfall rate exceeded P% of the time (mm/hr). Defaults to R0.01."""

    _value_name_prefer = "rr"  # P.837 annual files are RR_PPP.TXT

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.837-7", data_dir=data_dir)

    def lookup(self, lat_deg: float, lon_deg: float,
               percentage_time: Optional[float] = 0.01) -> float:
        return super().lookup(lat_deg, lon_deg, percentage_time)


class P836Map(ITUMapReader):
    """Surface water-vapor density (g/m³)."""

    _value_name_prefer = "rho"  # P.836 ships RHO_*.TXT (water-vapor density)

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.836-6", data_dir=data_dir)


class P839Map(ITUMapReader):
    """Rain height (km above mean sea level), 0°C isotherm.

    Per P.839-4, rain height = 0°C-isotherm height + 0.36 km.
    """

    _value_name_prefer = "h0"  # P.839 ships h0.txt (0°C isotherm) + Lat/Lon

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.839-4", data_dir=data_dir)

    def lookup(
        self,
        lat_deg: float,
        lon_deg: float,
        percentage_time: Optional[float] = None,
    ) -> float:
        h0 = super().lookup(lat_deg, lon_deg, percentage_time)
        # If the data file was missing we returned the fallback (already
        # rain height in km), so don't add the offset.
        if not self.is_available:
            return h0
        return h0 + 0.36


class P840Map(ITUMapReader):
    """Total columnar liquid water (kg/m²)."""

    _value_name_prefer = "lwc"  # P.840 ships LWC*.TXT

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.840-9", data_dir=data_dir)


class P453Map(ITUMapReader):
    """Wet-term refractivity N_wet (dimensionless / N-units).

    P.453-14 ships annual NWET maps under
    `P.453_NWET_Maps/P.453_NWET_Maps_Annual/` as `NWET_Annual_*.TXT`.
    """

    _value_name_prefer = "nwet"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.453-14", data_dir=data_dir)


class P1510Map(ITUMapReader):
    """Mean annual surface temperature (K). The annual map ships in K."""

    _value_name_prefer = "t_annual"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("P.1510-1", data_dir=data_dir)


__all__ = [
    "ITUMapReader",
    "P837Map",
    "P836Map",
    "P839Map",
    "P840Map",
    "P453Map",
    "P1510Map",
]
