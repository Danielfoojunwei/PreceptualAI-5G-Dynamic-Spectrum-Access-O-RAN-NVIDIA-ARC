"""Real-data LEO TLE → NGSO emitter pipeline.

This module connects the public CelesTrak TLE catalog to PreceptualAI's
EPFD aggregation pipeline. Every other piece of `planner/physics/`
already accepts ECEF positions — what was missing was a production
adapter that propagates a real catalog of two-line element sets and
yields `NGSOEmitter` values at any UTC instant.

Typical use::

    prop = TLEConstellationPropagator("data/orbital/celestrak/starlink.tle")
    emitters = prop.visible_from(es_lat=51.92, es_lon=4.48, t_utc=now,
                                 min_elevation_deg=10.0)
    epfd = epfd_down(es_lat_deg=51.92, ..., ngso_satellites=emitters)

The class is constellation-agnostic (Starlink, OneWeb, Iridium-NEXT,
Globalstar, etc.) — pass any 3-line CelesTrak TLE file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from sgp4.api import WGS84, Satrec

from horizon_ric.planner.physics.epfd import (
    EPFDTimeCDF,
    NGSOEmitter,
    epfd_time_cdf,
)
from horizon_ric.planner.physics.geodesy import look_angles_to_target
from horizon_ric.planner.physics.orbital import sgp4_state, state_ecef_m


def _parse_tle_file(text: str) -> list[tuple[str, str, str]]:
    """Parse a 3-line-group TLE file into (name, line1, line2) triples.

    CelesTrak's standard format is:

        NAME OF SAT
        1 NNNNNU ...
        2 NNNNN ...

    Blank lines are skipped. Records that don't have line1/line2 starting
    with ``"1 "`` and ``"2 "`` are skipped (with no error) — CelesTrak
    occasionally emits comment lines that we want to tolerate.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(lines) + 1 and i + 1 < len(lines):
        if i + 2 >= len(lines):
            break
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        # Defensive: skip if the next two don't look like TLE numeric lines.
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name.strip(), l1, l2))
            i += 3
        else:
            # Mis-aligned record — advance one line and try again.
            i += 1
    return out


class TLEConstellationPropagator:
    """Propagate a TLE catalog to NGSOEmitter snapshots.

    Caches one ``Satrec`` per satellite at construction time so per-call
    cost is just the SGP4 propagation (~30 µs each). Re-parsing of the
    TLE file is deliberately *not* deferred — we want a clear failure
    mode at construction if the file is malformed.

    Parameters
    ----------
    tle_file_path : Path | str
        Path to a 3-line-group TLE file (CelesTrak / space-track format).
    default_eirp_dBW : float
        EIRP assigned to every emitter. -15 dBW/MHz is a common Ku-band
        per-beam value used in ITU EPFD studies; tune per scenario.
    name_prefix : str
        Fallback prefix when a TLE record's name field is empty.

    Raises
    ------
    FileNotFoundError
        If ``tle_file_path`` doesn't exist.
    ValueError
        If the file contains no valid 3-line TLE groups.
    """

    def __init__(
        self,
        tle_file_path: Path | str,
        default_eirp_dBW: float = -15.0,
        name_prefix: str = "ngso",
    ) -> None:
        self._path = Path(tle_file_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"TLE file not found: {self._path} — "
                f"run scripts/download_tles.py first"
            )
        self._default_eirp_dBW = float(default_eirp_dBW)
        self._name_prefix = name_prefix

        text = self._path.read_text(encoding="utf-8", errors="replace")
        records = _parse_tle_file(text)
        if not records:
            raise ValueError(
                f"No valid 3-line TLE groups found in {self._path}"
            )

        self._records: list[tuple[str, Satrec]] = []
        for idx, (name, l1, l2) in enumerate(records):
            try:
                sat = Satrec.twoline2rv(l1, l2, WGS84)
            except Exception:  # noqa: BLE001 — skip malformed lines
                continue
            display_name = name if name else f"{name_prefix}-{idx}"
            self._records.append((display_name, sat))

        if not self._records:
            raise ValueError(
                f"All TLE records in {self._path} failed to parse via sgp4"
            )

        # Keep the raw lines around so users can re-instantiate / inspect.
        self._raw_records = records

    # ─── introspection ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    @property
    def path(self) -> Path:
        return self._path

    def names(self) -> list[str]:
        return [name for name, _ in self._records]

    def slice(self, n: int) -> "TLEConstellationPropagator":
        """Return a new propagator restricted to the first ``n`` sats.

        Useful for "politeness gates" in demos — Starlink has ~10 K sats
        and propagating all of them per frame swamps any other work.
        """
        if n <= 0:
            raise ValueError("slice n must be positive")
        sliced = TLEConstellationPropagator.__new__(TLEConstellationPropagator)
        sliced._path = self._path
        sliced._default_eirp_dBW = self._default_eirp_dBW
        sliced._name_prefix = self._name_prefix
        sliced._records = self._records[:n]
        sliced._raw_records = self._raw_records[:n]
        return sliced

    # ─── propagation ──────────────────────────────────────────────────

    def at(self, t_utc: datetime) -> list[NGSOEmitter]:
        """Propagate every cached satellite to ``t_utc`` and return emitters.

        Each satellite is converted via SGP4 (TEME/ECI) → ECEF using the
        IAU 1982 GMST already implemented in ``orbital.eci_to_ecef_km``.
        Satellites whose SGP4 propagation errors out (decayed, bad TLE)
        are silently dropped — we'd rather miss one sat than abort the
        whole snapshot.
        """
        emitters: list[NGSOEmitter] = []
        for name, sat in self._records:
            # We re-build the line pair via Satrec → reuse sgp4_state for
            # consistency. Cheaper path: call sat.sgp4 directly here.
            try:
                # sgp4_state takes the original lines; instead we invoke
                # sat.sgp4 ourselves to skip re-parsing.
                state = _sgp4_state_from_satrec(sat, t_utc)
            except Exception:  # noqa: BLE001
                continue
            ecef = state_ecef_m(state)
            emitters.append(
                NGSOEmitter(
                    position_ecef=ecef,
                    eirp_dBW=self._default_eirp_dBW,
                    name=name or f"{self._name_prefix}-?",
                )
            )
        return emitters

    def visible_from(
        self,
        es_lat_deg: float,
        es_lon_deg: float,
        t_utc: datetime,
        min_elevation_deg: float = 10.0,
        es_height_m: float = 0.0,
    ) -> list[NGSOEmitter]:
        """Return only those emitters above the local horizon at the ES.

        ``min_elevation_deg`` is the typical >5°-or-10° mask that real
        receivers use; below that the link budget is dominated by
        atmospheric attenuation and the satellite isn't actually usable.
        """
        all_emitters = self.at(t_utc)
        out: list[NGSOEmitter] = []
        for em in all_emitters:
            look = look_angles_to_target(
                es_lat_deg, es_lon_deg, em.position_ecef, es_height_m
            )
            if look.elevation_deg >= min_elevation_deg:
                out.append(em)
        return out

    def epfd_time_cdf_real(
        self,
        es_lat_deg: float,
        es_lon_deg: float,
        wanted_gso_lon_deg: float,
        es_diameter_m: float,
        es_frequency_hz: float,
        t_start_utc: datetime,
        duration_s: float = 3_600.0,
        step_s: float = 60.0,
        es_height_m: float = 0.0,
        aperture_efficiency: float = 0.65,
        reference_bandwidth_hz: float = 1e6,
        transmit_bandwidth_hz: float = 1e6,
    ) -> EPFDTimeCDF:
        """Convenience wrapper — runs ITU-R S.1503 time-CDF using ``self.at``.

        This is the production entry point: it threads the cached SGP4
        ensemble into ``epfd_time_cdf`` so callers don't need to know
        how the constellation is sourced. Drop-in compatible with any
        post-processing that expects an ``EPFDTimeCDF`` (e.g. percentile
        masks from ITU-R Article 22).
        """
        return epfd_time_cdf(
            es_lat_deg=es_lat_deg,
            es_lon_deg=es_lon_deg,
            wanted_gso_lon_deg=wanted_gso_lon_deg,
            es_diameter_m=es_diameter_m,
            es_frequency_hz=es_frequency_hz,
            constellation_at=self.at,
            t_start_utc=t_start_utc,
            duration_s=duration_s,
            step_s=step_s,
            es_height_m=es_height_m,
            aperture_efficiency=aperture_efficiency,
            reference_bandwidth_hz=reference_bandwidth_hz,
            transmit_bandwidth_hz=transmit_bandwidth_hz,
        )


# ─── internal: avoid Satrec re-parse per call ───────────────────────────


def _sgp4_state_from_satrec(sat: Satrec, t_utc: datetime):
    """SGP4-propagate an already-parsed Satrec to UTC instant.

    Mirrors ``orbital.sgp4_state`` semantics but skips the
    ``Satrec.twoline2rv`` call (already done at constructor time).
    """
    from datetime import timezone

    from sgp4.api import jday

    from horizon_ric.planner.physics.orbital import OrbitalState

    if t_utc.tzinfo is None:
        raise ValueError("t_utc must be timezone-aware")
    t_utc = t_utc.astimezone(timezone.utc)
    jd, fr = jday(
        t_utc.year, t_utc.month, t_utc.day,
        t_utc.hour, t_utc.minute,
        t_utc.second + t_utc.microsecond / 1e6,
    )
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed (code {err})")
    return OrbitalState(
        epoch_utc=t_utc, r_eci_km=tuple(r), v_eci_km_s=tuple(v)
    )


__all__ = ["TLEConstellationPropagator"]


# Suppress "unused" warnings for re-exports we may want later.
_ = (sgp4_state, Iterable)
