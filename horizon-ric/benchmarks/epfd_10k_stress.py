"""EPFD 10k random LEO scenario stress test (Row 12).

Produces a real EPFD aggregate distribution over 10,000 distinct
(ES_lat, ES_lon, wanted_GSO_lon, time_offset, freq_band) tuples drawn
with a fixed RNG seed for reproducibility.

For each scenario:
    1. Randomly choose 200 Starlink satellites from the live CelesTrak TLE
       catalogue and propagate them via real SGP4 (vendored python-sgp4)
       to the scenario time.
    2. Place a 1.2 m Ku-band ES (12 GHz downlink) at the scenario coords,
       pointing at a wanted GSO at the scenario's GSO_lon.
    3. Run `epfd_down(...)` and record EPFD, n_visible, and exceedance
       of the Article 22 representative -160 dBW/m² floor.

Result is written to `benchmarks/epfd_10k.json` with per-scenario rows
plus aggregate stats. Headline number printed to stdout.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.planner.physics import NGSOEmitter, epfd_down  # noqa: E402
from horizon_ric.planner.physics.orbital import (  # noqa: E402
    sgp4_state,
    state_ecef_m,
)


# ── configuration ───────────────────────────────────────────────────────────
TLE_PATH = Path("/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak/starlink.tle")
N_SCENARIOS = 10_000
N_SATS_PER_SCENARIO = 200
SEED = 20260506
ARTICLE22_FLOOR_DBW_PER_M2 = -160.0  # representative Ku-band single-entry mask
OUT_PATH = ROOT / "benchmarks" / "epfd_10k.json"

# Frequency bands sampled per scenario (Ku-band downlink set).
FREQ_BANDS_HZ: list[tuple[str, float]] = [
    ("ku_10p7ghz", 10.70e9),
    ("ku_11p2ghz", 11.20e9),
    ("ku_11p7ghz", 11.70e9),
    ("ku_12p0ghz", 12.00e9),
    ("ku_12p2ghz", 12.20e9),
    ("ku_12p5ghz", 12.50e9),
]

# Window of permissible scenario times: 24 h around 2026-05-06 12:00 UTC.
SIM_BASE = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
TIME_WINDOW_S = 24 * 3600


def _read_tles(path: Path) -> list[tuple[str, str, str]]:
    """Parse a 3-line TLE file (NAME / line1 / line2)."""
    if not path.exists():
        raise FileNotFoundError(f"TLE catalogue not found: {path}")
    raw = [ln for ln in path.read_text().splitlines() if ln.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(raw) + 1:
        if i + 2 >= len(raw):
            break
        name, l1, l2 = raw[i].strip(), raw[i + 1], raw[i + 2]
        # python-sgp4 wants exactly 69-char lines starting with "1 " / "2 "
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name, l1, l2))
            i += 3
        else:
            # Skip stray non-tle line.
            i += 1
    return out


def _percentile(sorted_vals: list[float], p_pct: float) -> float:
    """Lower-tail percentile from an ascending list (p_pct=99.999 etc)."""
    if not sorted_vals:
        return float("-inf")
    idx = int(round((p_pct / 100.0) * (len(sorted_vals) - 1)))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def main() -> int:
    rng = random.Random(SEED)
    tles = _read_tles(TLE_PATH)
    if len(tles) < N_SATS_PER_SCENARIO:
        raise RuntimeError(
            f"Need at least {N_SATS_PER_SCENARIO} TLEs; found {len(tles)}"
        )
    print(
        f"[epfd-10k] loaded {len(tles)} Starlink TLEs from {TLE_PATH.name}",
        flush=True,
    )

    rows: list[dict] = []
    epfd_values: list[float] = []
    n_violations = 0
    n_no_visible = 0

    t_start_wall = time.time()

    for sid in range(N_SCENARIOS):
        # Scenario parameters.
        es_lat = rng.uniform(-65.0, 65.0)            # avoid poles where Starlink coverage thins
        es_lon = rng.uniform(-180.0, 180.0)
        # Wanted GSO within ±60° of ES longitude so it is visibly above horizon.
        gso_offset = rng.uniform(-50.0, 50.0)
        wanted_gso_lon = ((es_lon + gso_offset) + 180.0) % 360.0 - 180.0
        time_offset_s = rng.uniform(0.0, TIME_WINDOW_S)
        scenario_t = SIM_BASE + timedelta(seconds=time_offset_s)
        band_name, freq_hz = rng.choice(FREQ_BANDS_HZ)

        # Sample 200 sats without replacement.
        idxs = rng.sample(range(len(tles)), N_SATS_PER_SCENARIO)
        emitters: list[NGSOEmitter] = []
        for k in idxs:
            name, l1, l2 = tles[k]
            try:
                state = sgp4_state(l1, l2, scenario_t)
            except RuntimeError:
                # SGP4 deep-space convergence failure: skip this sat.
                continue
            emitters.append(
                NGSOEmitter(
                    position_ecef=state_ecef_m(state),
                    eirp_dBW=-15.0,    # representative Starlink user-link EIRP density (dBW/MHz)
                    name=name.strip() or f"sat-{k}",
                )
            )

        result = epfd_down(
            es_lat_deg=es_lat,
            es_lon_deg=es_lon,
            wanted_gso_lon_deg=wanted_gso_lon,
            es_diameter_m=1.2,
            es_frequency_hz=freq_hz,
            ngso_satellites=emitters,
            reference_bandwidth_hz=1e6,
            transmit_bandwidth_hz=1e6,
        )
        epfd = result.epfd_dBW_per_m2
        n_vis = result.n_visible
        violated = epfd > ARTICLE22_FLOOR_DBW_PER_M2
        if violated:
            n_violations += 1
        if n_vis == 0:
            n_no_visible += 1
        epfd_values.append(epfd)

        if (sid + 1) % 1000 == 0:
            elapsed = time.time() - t_start_wall
            print(
                f"[epfd-10k] {sid + 1:>5}/{N_SCENARIOS} "
                f"elapsed={elapsed:6.1f}s  rate={(sid + 1) / elapsed:6.1f} sc/s  "
                f"viol_so_far={n_violations}",
                flush=True,
            )

        rows.append({
            "id": sid,
            "es_lat": es_lat,
            "es_lon": es_lon,
            "wanted_gso_lon": wanted_gso_lon,
            "time_offset_s": time_offset_s,
            "freq_band": band_name,
            "freq_hz": freq_hz,
            "epfd_dBW_per_m2": epfd,
            "n_visible": n_vis,
            "violation": bool(violated),
        })

    wall_clock_s = time.time() - t_start_wall

    # Aggregate stats. Sort once, ascending.
    sorted_e = sorted(epfd_values)
    n = len(sorted_e)
    mean_e = sum(epfd_values) / n
    median_e = sorted_e[n // 2]
    p99_999 = _percentile(sorted_e, 99.999)
    p99_99 = _percentile(sorted_e, 99.99)
    p99_9 = _percentile(sorted_e, 99.9)
    p50 = _percentile(sorted_e, 50.0)
    # ITU Article 22 expresses limits as "EPFD must not be exceeded for more
    # than 0.001 % of the time" → that is the (100 − 0.001) percentile.
    p99_999_top = _percentile(sorted_e, 99.999)

    payload = {
        "config": {
            "n_scenarios": N_SCENARIOS,
            "n_sats_per_scenario": N_SATS_PER_SCENARIO,
            "seed": SEED,
            "tle_source": str(TLE_PATH),
            "n_tles_available": len(tles),
            "article22_floor_dBW_per_m2": ARTICLE22_FLOOR_DBW_PER_M2,
            "wall_clock_s": wall_clock_s,
            "sim_base_utc": SIM_BASE.isoformat(),
        },
        "aggregate": {
            "mean_dBW_per_m2": mean_e,
            "median_dBW_per_m2": median_e,
            "p50_dBW_per_m2": p50,
            "p99_9_dBW_per_m2": p99_9,
            "p99_99_dBW_per_m2": p99_99,
            "p99_999_dBW_per_m2": p99_999,
            "n_violations": n_violations,
            "n_no_visible": n_no_visible,
            "violation_rate": n_violations / n,
        },
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    # Headline.
    print()
    print(
        f"{N_SCENARIOS:,} scenarios run, {n_violations} violations, "
        f"P_0.001%={p99_999_top:.2f} dBW/m^2 "
        f"(wall_clock={wall_clock_s:.1f} s)"
    )
    print(f"[epfd-10k] aggregate written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
