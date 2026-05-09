"""Download the full space-track.org GP catalog.

Reads `SPACE_TRACK_IDENTITY` and `SPACE_TRACK_PASSWORD` from the
environment. The catalog is written as TLE text to the path provided
on argv (default: `data/space_track/full_catalog.tle`).

Run::

    SPACE_TRACK_IDENTITY=user@example.com SPACE_TRACK_PASSWORD='...' \
        python scripts/download_space_track_catalog.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from horizon_ric.data.space_track import SpaceTrackClient


async def _main(out_path: Path) -> int:
    async with SpaceTrackClient() as st:
        return await st.download_full_catalog(out_path)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/space_track/full_catalog.tle")
    n = asyncio.run(_main(out))
    print(f"wrote {n} satellites to {out}")
