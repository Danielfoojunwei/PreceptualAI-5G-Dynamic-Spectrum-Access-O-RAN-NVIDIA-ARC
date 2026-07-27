"""Thin re-export shim for the Horizon-RIC production daemon.

The real implementation lives in :mod:`horizon_ric.rapp.daemon` so the
installed wheel ships it (`horizon-rapp --source-config ...` works from a
plain ``pip install``). This shim keeps the historical
``scripts/run_horizon_rapp.py`` invocation path and the
``from scripts.run_horizon_rapp import run_daemon`` import used by docs,
CI and tests working unchanged.

Usage::

    .venv/bin/python scripts/run_horizon_rapp.py --source-config local-dev.yaml
    .venv/bin/python scripts/run_horizon_rapp.py --once --report-json report.json

See ``horizon_ric.rapp.daemon`` for the full source-config schema and the
per-event pipeline contract.
"""

from __future__ import annotations

import sys

from horizon_ric.rapp.daemon import *  # noqa: F401,F403 — re-export shim
from horizon_ric.rapp.daemon import main

if __name__ == "__main__":
    sys.exit(main())
