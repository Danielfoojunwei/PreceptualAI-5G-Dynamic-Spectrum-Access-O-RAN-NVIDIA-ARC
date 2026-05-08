"""Thin CLI dispatcher for `python -m horizon_ric.cli`.

This module provides a stable, narrow entrypoint used by the production
systemd unit (`deploy/systemd/horizon-ric-orin.service`):

    /usr/bin/python3 -m horizon_ric.cli serve --bind 127.0.0.1:8083

Subcommands:
  serve   — boot the rApp lifecycle with health/api on the given bind address.

The actual lifecycle logic lives in `horizon_ric.rapp.lifecycle.main`; this
shim only translates the `--bind HOST:PORT` form into the env vars that
lifecycle._config_from_env() consumes (HORIZON_HEALTH_HOST / HORIZON_HEALTH_PORT).
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_bind(bind: str) -> tuple[str, int]:
    if ":" not in bind:
        raise SystemExit(f"--bind must be HOST:PORT, got {bind!r}")
    host, _, port_s = bind.rpartition(":")
    try:
        port = int(port_s)
    except ValueError as exc:
        raise SystemExit(f"--bind port must be integer, got {port_s!r}") from exc
    return host or "127.0.0.1", port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horizon_ric.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run the rApp lifecycle daemon")
    p_serve.add_argument("--bind", default="127.0.0.1:8083",
                         help="HOST:PORT for the operator API (default 127.0.0.1:8083)")
    p_serve.add_argument("--once", action="store_true",
                         help="boot, verify readiness, exit (smoke test)")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        host, port = _parse_bind(args.bind)
        os.environ["HORIZON_HEALTH_HOST"] = host
        os.environ["HORIZON_HEALTH_PORT"] = str(port)
        from horizon_ric.rapp.lifecycle import main as lifecycle_main
        forwarded: list[str] = []
        if args.once:
            forwarded.append("--once")
        return lifecycle_main(forwarded)

    return 2


if __name__ == "__main__":
    sys.exit(main())
