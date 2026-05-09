"""Download ITU-R reference propagation data files for PreceptualAI.

ITU publishes reference data files (gridded maps for rainfall, water vapor,
rain height, refractivity, temperature, etc.) alongside its Recommendations.
We fetch each ZIP, extract the contents, record SHA-256 + size + ITU rec ID
into a manifest, and continue past 404s.

Usage:
    .venv/bin/python scripts/download_itu_data.py

Outputs land under `data/itu_r/<rec_id>/` (the data dir is gitignored).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# (rec_id, primary URL, [optional fallback URLs])
RECOMMENDATIONS: list[tuple[str, str, list[str]]] = [
    (
        "P.837-7",
        # ITU only ships the rain-rate ZIP under the -S (superseded) suffix
        # even though P.837-7 is currently in force; the -I URL 404s.
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.837-7-201706-S!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.837-7-201706-I!!ZIP-E.zip",
            # P.837-8 (09/2025) supersedes -7; use it as a last-ditch fallback.
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.837-8-202509-I!!ZIP-E.zip",
        ],
    ),
    (
        "P.836-6",
        # P.836-6 was re-posted with a 201712 stamp (not 201709).
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.836-6-201712-I!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.836-6-201709-I!!ZIP-E.zip",
            # Previous in-force revision (P.836-5) ships the same maps.
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.836-5-201309-S!!ZIP-E.zip",
        ],
    ),
    (
        "P.839-4",
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-S!!ZIP-E.zip",
        ],
    ),
    (
        "P.840-9",
        # P.840-9 (08/2023) is currently in force but ITU has not posted a
        # consolidated ZIP for it; the maps live with the -8 (08/2019)
        # revision which is the most recent rev that ships data files.
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-8-201908-S!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-9-202308-I!!ZIP-E.zip",
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-9-202308-S!!ZIP-E.zip",
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-7-201712-S!!ZIP-E.zip",
        ],
    ),
    (
        "P.453-14",
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-I!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-S!!ZIP-E.zip",
        ],
    ),
    (
        "P.1510-1",
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1510-1-201706-I!!ZIP-E.zip",
        [
            "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1510-1-201706-S!!ZIP-E.zip",
        ],
    ),
]

USER_AGENT = (
    "horizon-ric/0.1 (+https://github.com/horizon-ric; "
    "research data fetcher for ITU-R propagation maps)"
)
TIMEOUT_S = 30.0
OUT_DIR = Path("/home/danielfoojunwei/Preceptualv1/data/itu_r")


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


def _extracted_size(directory: Path) -> int:
    total = 0
    for p in directory.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    failures: list[dict] = []
    total_extracted_bytes = 0
    n_ok = 0

    # Carry forward any prior-run manifest so we don't re-download recs that
    # already landed on disk (some ITU ZIPs are >200 MB).
    prior_manifest_path = OUT_DIR / "manifest.json"
    prior_entries: dict[str, dict] = {}
    if prior_manifest_path.exists():
        try:
            prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            for entry in prior.get("recommendations", []):
                prior_entries[entry["rec_id"]] = entry
        except (OSError, json.JSONDecodeError):
            prior_entries = {}

    for rec_id, primary_url, fallbacks in RECOMMENDATIONS:
        # Skip if this rec already extracted on disk and matches prior manifest.
        prior = prior_entries.get(rec_id)
        if prior is not None:
            extracted_dir = Path(prior.get("extracted_dir", ""))
            if extracted_dir.is_dir() and any(extracted_dir.iterdir()):
                ext_bytes = _extracted_size(extracted_dir)
                if ext_bytes > 0:
                    manifest_entries.append(prior)
                    total_extracted_bytes += ext_bytes
                    n_ok += 1
                    print(
                        f"[skip ] {rec_id:<10}  already on disk "
                        f"({ext_bytes} B)",
                        flush=True,
                    )
                    continue

        urls_to_try = [primary_url, *fallbacks]
        attempted: list[dict] = []
        last_err: str | None = None
        success = False

        for url in urls_to_try:
            try:
                print(f"[fetch] {rec_id:<10}  {url}", flush=True)
                data = _fetch(url)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                attempted.append({"url": url, "error": last_err})
                print(f"        FAIL {last_err}", flush=True)
                continue

            zip_path = OUT_DIR / f"{rec_id}.zip"
            zip_path.write_bytes(data)
            extract_dir = OUT_DIR / rec_id
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                # ITU sometimes nests inner ZIPs (e.g. P.453 ships
                # "P.453_NWET_Maps.zip" containing two more zips). Recurse
                # one level deep so all data files land on disk.
                for inner in list(extract_dir.rglob("*.zip")):
                    inner_dir = inner.parent / inner.stem
                    inner_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        with zipfile.ZipFile(inner, "r") as zf:
                            zf.extractall(inner_dir)
                    except zipfile.BadZipFile:
                        continue
                    # And one more level — some inner zips themselves nest.
                    for inner2 in list(inner_dir.rglob("*.zip")):
                        inner2_dir = inner2.parent / inner2.stem
                        inner2_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            with zipfile.ZipFile(inner2, "r") as zf:
                                zf.extractall(inner2_dir)
                        except zipfile.BadZipFile:
                            continue
            except zipfile.BadZipFile as exc:
                last_err = f"BadZipFile: {exc}"
                attempted.append({"url": url, "error": last_err})
                print(f"        FAIL extract: {last_err}", flush=True)
                continue

            ext_bytes = _extracted_size(extract_dir)
            total_extracted_bytes += ext_bytes
            n_ok += 1
            success = True
            manifest_entries.append({
                "rec_id": rec_id,
                "url": url,
                "zip_sha256": _sha256(data),
                "zip_bytes": len(data),
                "extracted_bytes": ext_bytes,
                "extracted_dir": str(extract_dir),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"        ok  zip={len(data):>9} B  extracted={ext_bytes:>9} B",
                flush=True,
            )
            break

        if not success:
            msg = (
                f"MISSING: {rec_id} — try manual download from itu.int "
                f"(last error: {last_err})"
            )
            print(msg, flush=True)
            failures.append({
                "rec_id": rec_id,
                "status": "permanently_failed",
                "error": last_err or "unknown",
                "attempted_urls": attempted,
            })

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "itu.int",
        "recommendations": manifest_entries,
        "failures": failures,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mb = total_extracted_bytes / (1024 * 1024)
    print(
        f"\nDownloaded {n_ok}/{len(RECOMMENDATIONS)} ITU-R recommendations, "
        f"total {mb:.2f} MB extracted"
    )
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f['rec_id']}: {f['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
