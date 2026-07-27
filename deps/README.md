# deps/ — locked Python environments

Horizon-RIC's **authoritative** Python dependency declaration lives in
[`pyproject.toml`](../pyproject.toml). The lockfiles in `deps/locks/` are
*not* an alternative declaration — they are verbatim `pip freeze` snapshots
of the three virtualenvs in which this repository's E2E / benchmark evidence
was actually produced, kept so that the exact tested environments can be
reproduced bit-for-bit.

| Lockfile | Snapshot of | Interpreter | Purpose |
| --- | --- | --- | --- |
| `locks/horizon.txt` | Horizon-RIC's own venv (`pip install -e .` of this repo) | Python 3.11 | The environment the rApp, test suite, and benchmark suites ran in. |
| `locks/xapp.txt` | The O-RAN-SC `hw-python` xApp venv built by `deploy/xapp-e2e/run_stack.sh` | Python 3.11 (3.12+ breaks: `hiredis 2.0.0`'s `setup.py` imports the removed stdlib `imp` module) | The `ricxappframe` 2.2.0 stack the real xApp ran on, including the required `protobuf==3.20.3` pin (`ricxappframe` ships `_pb2.py` stubs the protobuf 4.x/5.x runtime refuses to load). `hw-python` itself appears as an editable install pinned to the exact upstream commit. |
| `locks/a1sim.txt` | The O-RAN-SC A1 simulator (`sim/a1-interface` near-rt-ric-simulator) venv | Python 3.11 | Flask 2.2.5 / connexion 2.14.2 stack used when the rApp was exercised against the OSC A1 simulator (upstream's Dockerfile pins the same two top-level packages). |

Reproduce an environment with, e.g.:

```bash
python3.11 -m venv venv-xapp
venv-xapp/bin/pip install -r deps/locks/xapp.txt
```

Notes:

- These are exact-version snapshots taken on Linux/x86-64 with Python
  3.11.15; on another platform pip may need to build sdists or select
  different wheels.
- For day-to-day development of Horizon-RIC itself, use
  `pip install -e '.[dev]'` per the top-level README; use `horizon.txt`
  only when you need to replay the exact tested environment.
- The O-RAN-SC source components these venvs drive are vendored as pinned
  git submodules under [`third_party/`](../third_party/MANIFEST.md), and
  `scripts/fetch_dependencies.sh` reproduces the same checkouts without
  submodules.
