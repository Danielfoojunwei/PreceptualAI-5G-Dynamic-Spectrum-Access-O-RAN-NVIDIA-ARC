"""Real-data showcase: CelesTrak Starlink TLEs → EPFD aggregate at Rotterdam.

This is the FIRST end-to-end demo against real public data — every other
demo in this repo runs against a synthetic Walker constellation. It:

    1. Loads ``data/orbital/celestrak/starlink.tle`` from CelesTrak.
    2. Restricts to the first 50 sats (politeness gate; full Starlink is
       ~10 K and would dominate a 5-second demo).
    3. Propagates them to "now" via SGP4 + ECI→ECEF.
    4. Filters to those visible from Rotterdam (51.9244 °N, 4.4777 °E)
       above a 10° elevation mask.
    5. Computes an ITU-R S.1503-3 EPFD-down snapshot against a wanted
       GSO at 10 °E.
    6. Prints aggregate EPFD + the top-3 contributing emitters.
    7. Runs ``epfd_time_cdf_real`` over 1 hour at 60 s cadence.
    8. Prints the 0.001 %, 0.01 % and 50 % percentiles + visibility stats.

If the TLE file isn't present (e.g. the parallel download hasn't
finished), the script prints an explicit "no TLEs downloaded yet" line
and exits non-zero — never crashes with a stack trace.

Usage
-----

    .venv/bin/python scripts/demo_real_constellation.py            # 50 sats
    .venv/bin/python scripts/demo_real_constellation.py --all-sats # full cat
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from horizon_ric.data.tle_pipeline import TLEConstellationPropagator  # noqa: E402
from horizon_ric.planner.physics.epfd import epfd_down  # noqa: E402


# CelesTrak download script writes here (see scripts/download_tles.py).
DEFAULT_TLE_PATH = (
    Path("/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak/starlink.tle")
)

# Rotterdam port — same coordinate set used by scripts/e2e_simulation.py.
ES_LAT, ES_LON = 51.9244, 4.4777
ES_HEIGHT_M = 0.0
WANTED_GSO_LON_DEG = 10.0
ES_DIAMETER_M = 1.2
ES_FREQUENCY_HZ = 12e9   # Ku-band downlink

DEFAULT_N_SATS = 50
DEFAULT_DURATION_S = 3_600.0
DEFAULT_STEP_S = 60.0


def _format_dB(x: float) -> str:
    return f"{x:>8.2f} dBW/m^2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tle-path", type=Path, default=DEFAULT_TLE_PATH,
        help="Path to a 3-line CelesTrak TLE file (default: starlink).",
    )
    parser.add_argument(
        "--all-sats", action="store_true",
        help="Disable the 50-satellite politeness gate (slow!).",
    )
    parser.add_argument(
        "--n-sats", type=int, default=DEFAULT_N_SATS,
        help="How many sats to keep when --all-sats is not set.",
    )
    parser.add_argument(
        "--duration-s", type=float, default=DEFAULT_DURATION_S,
        help="Time-CDF window length in seconds.",
    )
    parser.add_argument(
        "--step-s", type=float, default=DEFAULT_STEP_S,
        help="Time-CDF sampling cadence in seconds.",
    )
    args = parser.parse_args()

    # ── 1. Load the TLE file (gracefully) ─────────────────────────────
    if not args.tle_path.exists():
        print(
            f"no TLEs downloaded yet — run scripts/download_tles.py first\n"
            f"  expected: {args.tle_path}"
        )
        return 2

    print(f"[load] reading TLEs from {args.tle_path}")
    full = TLEConstellationPropagator(args.tle_path)
    print(f"[load] parsed {len(full)} satellites")

    if args.all_sats:
        prop = full
        print(f"[load] using ALL {len(prop)} sats (--all-sats)")
    else:
        prop = full.slice(args.n_sats)
        print(f"[load] using first {len(prop)} sats (politeness gate)")

    # ── 2-3. Propagate to NOW ─────────────────────────────────────────
    now = datetime.now(timezone.utc)
    print(f"[now]  t_utc = {now.isoformat()}")

    t0 = time.monotonic()
    all_emitters = prop.at(now)
    dt_propagate_ms = (time.monotonic() - t0) * 1e3
    print(
        f"[prop] {len(all_emitters)} sats propagated in "
        f"{dt_propagate_ms:.1f} ms"
    )

    # ── 4. Visibility from Rotterdam @ ≥10° elev ──────────────────────
    visible = prop.visible_from(
        ES_LAT, ES_LON, now, min_elevation_deg=10.0,
        es_height_m=ES_HEIGHT_M,
    )
    print(
        f"[vis]  visible from Rotterdam (>10 deg elev): "
        f"{len(visible)} / {len(all_emitters)}"
    )

    # ── 5-6. EPFD snapshot ────────────────────────────────────────────
    snap = epfd_down(
        es_lat_deg=ES_LAT,
        es_lon_deg=ES_LON,
        wanted_gso_lon_deg=WANTED_GSO_LON_DEG,
        es_diameter_m=ES_DIAMETER_M,
        es_frequency_hz=ES_FREQUENCY_HZ,
        ngso_satellites=visible,
        es_height_m=ES_HEIGHT_M,
    )
    print()
    print("=" * 60)
    print(f"EPFD-down snapshot @ Rotterdam vs GSO @ {WANTED_GSO_LON_DEG} deg E")
    print("=" * 60)
    print(f"  aggregate              {_format_dB(snap.epfd_dBW_per_m2)}")
    print(f"  visible NGSOs (in agg) {snap.n_visible}")
    print()
    if snap.per_emitter_dBW_per_m2:
        # Top-3 by linear power (i.e. highest dB).
        top3 = sorted(
            snap.per_emitter_dBW_per_m2, key=lambda kv: kv[1], reverse=True,
        )[:3]
        print("  top-3 per-emitter contributors:")
        for name, db in top3:
            print(f"    {name:<32} {_format_dB(db)}")
    else:
        print("  (no NGSO above horizon — nothing to aggregate)")

    # ── 7-8. Time-CDF ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(
        f"EPFD time-CDF over {args.duration_s/60:.0f} min @ "
        f"{args.step_s:.0f} s cadence"
    )
    print("=" * 60)
    t1 = time.monotonic()
    cdf = prop.epfd_time_cdf_real(
        es_lat_deg=ES_LAT,
        es_lon_deg=ES_LON,
        wanted_gso_lon_deg=WANTED_GSO_LON_DEG,
        es_diameter_m=ES_DIAMETER_M,
        es_frequency_hz=ES_FREQUENCY_HZ,
        t_start_utc=now,
        duration_s=args.duration_s,
        step_s=args.step_s,
        es_height_m=ES_HEIGHT_M,
    )
    dt_cdf_s = time.monotonic() - t1
    n_samples = len(cdf.samples_dB)
    print(f"  samples              {n_samples}")
    print(f"  duration             {cdf.duration_s:.0f} s")
    print(f"  step                 {cdf.step_s:.0f} s")
    print(f"  n_visible_max        {cdf.n_visible_max}")
    print(f"  n_visible_mean       {cdf.n_visible_mean:.2f}")
    print(f"  P_0.001%             {_format_dB(cdf.percentile(0.001))}")
    print(f"  P_0.01%              {_format_dB(cdf.percentile(0.01))}")
    print(f"  P_50%                {_format_dB(cdf.percentile(50.0))}")
    print(f"  mean (linear→dB)     {_format_dB(cdf.mean_dB())}")
    print(f"  max                  {_format_dB(cdf.max_dB())}")
    print(f"  wall_clock           {dt_cdf_s:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
