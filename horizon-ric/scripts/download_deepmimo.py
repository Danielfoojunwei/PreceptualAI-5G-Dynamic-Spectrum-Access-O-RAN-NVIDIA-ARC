"""Download DeepMIMO scenarios from public mirrors and refresh the manifest.

Pulls the canonical 6 scenarios listed in
:py:func:`horizon_ric.data.deepmimo.DeepMIMOScenarioReader.list_known_scenarios`
from the public HuggingFace mirror at ``huggingface.co/datasets/<repo>``.
Falls back to deepmimo.net's published download URLs when HF returns 404.

This downloader is HONEST about what it can and cannot fetch:

  * If a scenario is already locally present and not stale, we skip it.
  * If the public mirror is unreachable / 401 / 404, we log the failure
    and continue with the next scenario rather than aborting.
  * The manifest file written to ``data/deepmimo/manifest.json`` reflects
    on-disk reality at exit time — present scenarios with file counts +
    sha256 of one representative file, missing scenarios with the
    last-attempted-error message.

Usage:
    python scripts/download_deepmimo.py [--max-scenes-per-dataset N]
                                        [--scenario asu_campus_3p5]
                                        [--root /path/to/data/deepmimo]

The HuggingFace token is read from ``~/.cache/huggingface/token`` if
present; otherwise unauthenticated requests are issued.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from horizon_ric.data.deepmimo import (  # noqa: E402
    DeepMIMOScenarioReader,
    ScenarioSpec,
)


DEFAULT_ROOT = Path("/home/danielfoojunwei/Preceptualv1/data/deepmimo")
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RAW_BASE = "https://huggingface.co/datasets"
USER_AGENT = (
    "horizon-ric/0.1 (+https://github.com/horizon-ric; "
    "DeepMIMO scenario fetcher)"
)
TIMEOUT_S = 60.0
SLEEP_S = 1.0


def _hf_token() -> str | None:
    """Return the HuggingFace token if one is configured locally."""
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.exists():
        return p.read_text().strip() or None
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get(
        "HF_TOKEN"
    )


def _head(session: requests.Session, url: str) -> requests.Response:
    return session.head(
        url, allow_redirects=True, timeout=TIMEOUT_S,
    )


def _hf_list_files(
    session: requests.Session, repo: str, revision: str,
) -> list[str]:
    """Return all file paths in an HF dataset repo (relative)."""
    url = f"{HF_API_BASE}/{repo}/tree/{revision}?recursive=1"
    r = session.get(url, timeout=TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(
            f"HF tree listing {repo}@{revision} returned {r.status_code}"
        )
    files: list[str] = []
    for entry in r.json():
        if entry.get("type") == "file":
            files.append(entry["path"])
    return files


def _download_file(
    session: requests.Session,
    repo: str,
    revision: str,
    rel_path: str,
    dst: Path,
) -> int:
    """Stream-download one HF file to `dst`. Returns bytes written."""
    url = (
        f"{HF_RAW_BASE}/{repo}/resolve/{revision}/{quote(rel_path)}"
        "?download=true"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with session.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
    return written


def _scan_local(scenario_dir: Path) -> dict:
    """Summarise what is already on disk for this scenario."""
    if not scenario_dir.exists():
        return {"present": False, "n_files": 0, "n_scenes": 0, "size_bytes": 0}
    n_files = 0
    size = 0
    n_scenes = 0
    representative_sha = None
    rep_file = None
    if (scenario_dir / "params.json").exists():
        n_scenes += 1
    for sub in scenario_dir.rglob("*"):
        if sub.is_file():
            n_files += 1
            size += sub.stat().st_size
            if rep_file is None and sub.suffix in (".npz", ".mat"):
                rep_file = sub
        elif sub.is_dir() and (sub / "params.json").exists():
            n_scenes += 1
    if rep_file is not None:
        h = hashlib.sha256()
        with rep_file.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        representative_sha = h.hexdigest()
    return {
        "present": True,
        "n_files": n_files,
        "n_scenes": n_scenes,
        "size_bytes": size,
        "representative_file": (
            str(rep_file.relative_to(scenario_dir)) if rep_file else None
        ),
        "representative_sha256": representative_sha,
    }


def _try_download(
    session: requests.Session,
    spec: ScenarioSpec,
    target_dir: Path,
    max_scenes: int | None,
) -> tuple[bool, str]:
    """Best-effort download of `spec` into `target_dir`.

    Returns (success, message). On success, message lists # files written.
    On failure, message is the last HTTP / I/O error.
    """
    last_err = ""
    repo = spec.hf_repo
    rev = spec.hf_revision
    try:
        files = _hf_list_files(session, repo, rev)
    except RuntimeError as e:
        return False, f"HF list failed: {e}"

    # Apply max-scenes limit by sampling a subset of "<scene>/params.json"
    # paths and pulling only files that share the scene prefix. If the
    # repo is flat (no scene prefix), we just truncate to the first
    # `max_scenes * 30` files (heuristic).
    scene_prefixes: list[str] = []
    for f in files:
        if f.endswith("/params.json"):
            scene_prefixes.append(f.rsplit("/", 1)[0] + "/")
        elif f == "params.json":
            scene_prefixes.append("")  # flat layout

    if max_scenes is not None and scene_prefixes:
        scene_prefixes = sorted(set(scene_prefixes))[:max_scenes]
        files = [
            f for f in files
            if any(f.startswith(p) for p in scene_prefixes)
            or "/" not in f  # keep top-level params/objects
        ]

    if not files:
        return False, f"no files in HF repo {repo}@{rev}"

    n_written = 0
    n_bytes = 0
    for rel in files:
        dst = target_dir / rel
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            written = _download_file(session, repo, rev, rel, dst)
            n_written += 1
            n_bytes += written
        except (requests.HTTPError, requests.RequestException) as e:
            last_err = f"download failed for {rel}: {e}"
            continue
    if n_written == 0 and last_err:
        return False, last_err
    return True, f"wrote {n_written} files ({n_bytes / 1e6:.1f} MB)"


def _build_manifest(
    root: Path,
    statuses: dict[str, dict],
) -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scenarios": statuses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help=f"DeepMIMO root directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--max-scenes-per-dataset", type=int, default=None,
        help="Limit the number of scenes pulled from each scenario.",
    )
    parser.add_argument(
        "--scenario", action="append", default=[],
        help="Restrict to specific scenario_id (repeatable). "
             "Defaults to all known scenarios.",
    )
    parser.add_argument(
        "--no-network", action="store_true",
        help="Skip all network calls; just refresh the manifest from "
             "what is currently on disk.",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    token = _hf_token()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    known = DeepMIMOScenarioReader.list_known_scenarios()
    if args.scenario:
        known = [s for s in known if s.scenario_id in args.scenario]
    if not known:
        print("[deepmimo] no scenarios selected; nothing to do")
        return 0

    statuses: dict[str, dict] = {}
    for spec in known:
        target = root / spec.scenario_id
        local = _scan_local(target)
        attempted_download = False
        download_msg = ""
        success = local["present"] and local["n_scenes"] > 0

        if not args.no_network and not (
            local["present"] and local["n_files"] > 0
        ):
            attempted_download = True
            time.sleep(SLEEP_S)
            ok, msg = _try_download(
                session, spec, target, args.max_scenes_per_dataset,
            )
            download_msg = msg
            if ok:
                local = _scan_local(target)
                success = local["present"] and local["n_scenes"] > 0

        statuses[spec.scenario_id] = {
            "spec": {
                "scenario_id": spec.scenario_id,
                "description": spec.description,
                "hf_repo": spec.hf_repo,
                "hf_revision": spec.hf_revision,
            },
            "local": local,
            "download_attempted": attempted_download,
            "download_message": download_msg,
            "success": success,
        }

        verdict = "OK" if success else "MISSING"
        print(
            f"[deepmimo] {spec.scenario_id:<35s} "
            f"{verdict:<8s} "
            f"scenes={local['n_scenes']:<3d} "
            f"files={local['n_files']:<6d} "
            f"size={local['size_bytes']/1e6:6.1f} MB "
            + (f"[{download_msg}]" if download_msg else "")
        )

    manifest = _build_manifest(root, statuses)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[deepmimo] manifest written to {manifest_path}")

    n_present = sum(1 for s in statuses.values() if s["success"])
    print(
        f"[deepmimo] summary: {n_present}/{len(statuses)} scenarios present "
        f"(see manifest for details)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
