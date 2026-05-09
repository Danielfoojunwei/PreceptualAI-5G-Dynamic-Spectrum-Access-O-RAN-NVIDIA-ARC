"""Download public LEO/MEO/GEO TLE catalogs from CelesTrak.

CelesTrak mirrors the public element sets from space-track.org. We fetch
the active TLE for each constellation group, store one file per group,
and write a manifest with sha256 + downloaded_at_utc + sat counts.

Be polite: 1 s sleep between groups, descriptive User-Agent. If a single
group fails (404, transient network), we log it and continue — partial
catalogs are still useful for PreceptualAI scenarios.

Usage:
    python scripts/download_tles.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# (group_name_slug, celestrak_GROUP_param)
GROUPS: list[tuple[str, str]] = [
    ("starlink", "starlink"),
    ("oneweb", "oneweb"),
    ("iridium-next", "iridium-NEXT"),
    ("globalstar", "globalstar"),
    ("planet", "planet"),
    ("spire", "spire"),
    ("swarm", "swarm"),
    ("orbcomm", "orbcomm"),
    ("amateur", "amateur"),
    ("weather", "weather"),
    ("noaa", "noaa"),
    ("goes", "goes"),
    ("science", "science"),
    ("stations", "stations"),
    ("geo", "geo"),
    ("intelsat", "intelsat"),
    ("ses", "ses"),
    ("galileo", "galileo"),
    ("gps-ops", "gps-ops"),
    ("beidou", "beidou"),
    ("glo-ops", "glo-ops"),
]

URL_TEMPLATE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
# CelesTrak hosts a supplemental stream (operator-published) for some
# constellations. Used as a fallback when the main GP endpoint is rate-limited
# (CelesTrak returns "GP data has not updated since your last successful
# download…" for ~2 hours per IP+group).
SUPPLEMENTAL_URL_TEMPLATE = (
    "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?"
    "FILE={group}&FORMAT=tle"
)
SUPPLEMENTAL_FALLBACK = {"starlink", "oneweb"}
USER_AGENT = (
    "horizon-ric/0.1 (+https://github.com/horizon-ric; "
    "research data fetcher for LEO/GEO orbital catalogs)"
)
OUT_DIR = Path("/home/danielfoojunwei/Preceptualv1/data/orbital/celestrak")
SLEEP_S = 1.0
TIMEOUT_S = 60.0


class _RateLimited(Exception):
    """CelesTrak says the GP data hasn't been updated since our last fetch."""


def _count_sats(text: str) -> int:
    """Count 3-line TLE records (NAME / line1 / line2)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) // 3


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(
                url, resp.status, f"non-200: {resp.status}", resp.headers, None
            )
        return resp.read()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: list[tuple[str, str]] = []
    total_sats = 0
    total_bytes = 0

    for slug, group_param in GROUPS:
        url = URL_TEMPLATE.format(group=group_param)
        out_path = OUT_DIR / f"{slug}.tle"
        try:
            print(f"[fetch] {slug:<14}  {url}", flush=True)
            try:
                data = _fetch(url)
                text = data.decode("utf-8", errors="replace")
                # CelesTrak rate-limits per IP+group with this body:
                if (
                    "GP data has not updated" in text
                    and slug in SUPPLEMENTAL_FALLBACK
                ):
                    raise _RateLimited(text[:120])
                if "No GP data found" in text or _count_sats(text) == 0:
                    raise RuntimeError(
                        f"empty / no-data response: {text[:120]!r}"
                    )
            except (urllib.error.HTTPError, _RateLimited) as primary_err:
                # Fall back to the supplemental (operator-published) stream
                # for constellations that publish there.
                if slug not in SUPPLEMENTAL_FALLBACK:
                    raise
                sup_url = SUPPLEMENTAL_URL_TEMPLATE.format(group=group_param)
                print(
                    f"        primary failed ({primary_err}); trying "
                    f"supplemental: {sup_url}",
                    flush=True,
                )
                data = _fetch(sup_url)
                text = data.decode("utf-8", errors="replace")
                if "No GP data found" in text or _count_sats(text) == 0:
                    raise RuntimeError(
                        f"supplemental empty: {text[:120]!r}"
                    )
                url = sup_url  # record actual URL used
            out_path.write_bytes(data)
            n_sats = _count_sats(text)
            total_sats += n_sats
            total_bytes += len(data)
            manifest.append({
                "group": slug,
                "celestrak_group": group_param,
                "url": url,
                "n_satellites": n_sats,
                "sha256": _sha256(data),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "file_bytes": len(data),
                "file_path": str(out_path),
            })
            print(f"        ok  {n_sats:>5} sats  {len(data):>8} bytes", flush=True)
        except Exception as exc:  # noqa: BLE001 — must not abort whole batch
            print(f"        FAIL {type(exc).__name__}: {exc}", flush=True)
            failures.append((slug, f"{type(exc).__name__}: {exc}"))

        time.sleep(SLEEP_S)  # be polite even after failure

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "celestrak.org",
                "groups": manifest,
                "failures": [{"group": g, "error": e} for g, e in failures],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mb = total_bytes / (1024 * 1024)
    n_groups_ok = len(manifest)
    print(
        f"\nSUMMARY: downloaded {n_groups_ok} groups, {total_sats} total satellites, "
        f"{mb:.2f} MB"
    )
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for slug, err in failures:
            print(f"  - {slug}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
