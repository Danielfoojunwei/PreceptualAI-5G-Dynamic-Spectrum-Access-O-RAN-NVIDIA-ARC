# third_party/ — vendored O-RAN dependency manifest

Every external source component the Horizon-RIC E2E stack was built from in
the recorded evidence runs, pinned to the **exact commit** that was used.
Machine-readable copy: [`manifest.json`](manifest.json).

## Vendoring policy — read this first

- **Vendored as pinned git submodule** means the component lives under
  `third_party/<name>` as a git submodule whose gitlink records the exact
  upstream commit. The superproject stores ~200 bytes per submodule (URL +
  commit), *not* the source tree, so the repository stays small while the
  checkout is exactly reproducible (`git submodule update --init`).
- Committing these trees (or the multi-GB datasets/build outputs they
  produce) directly as blobs would bloat the repository by hundreds of MB
  and break normal clone/review workflows, which is why nothing in this
  directory is stored as in-repo source.
- `scripts/fetch_dependencies.sh` reproduces the same pinned checkouts
  *without* submodules (plain clones + `checkout --detach <pin>`) and
  applies the one local patch, for consumers who prefer not to use
  submodules.
- Submodules pin **clean upstream commits**. Local modifications are never
  hidden inside a submodule: the single functional patch is a tracked file
  (see `local_patch` below) applied at build time, and scripted config
  mutations are performed by `deploy/xapp-e2e/run_stack.sh` itself.
- Build artifacts (`.build/`, `build/`, venvs) are never part of a
  submodule: a submodule records a commit, so local build output in a
  checkout does not enter the superproject history.

## Source dependencies

All six are vendored as pinned git submodules (`submodule: true`); none are
manifest-only. Pins verified against the working checkouts that produced
the evidence in `deploy/xapp-e2e/results/` and `benchmarks/results/`.

| Name | Role in the stack | Origin URL | Pinned commit | Version | License | Build command | Submodule | Local patch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ric-plt-lib-rmr` | RMR message router (C library) — the RIC message bus linking the A1 mediator and the xApp | https://gerrit.o-ran-sc.org/r/ric-plt/lib/rmr | `8b9a214906a40b338def981d5c16b4ea247176fb` | 4.9.4 | Apache-2.0 | `mkdir .build && cd .build && cmake .. -DDEV_PKG=1 -DCMAKE_INSTALL_PREFIX=/usr/local && make install` then same without `-DDEV_PKG=1` for the runtime lib (see `deploy/xapp-e2e/run_stack.sh`) | true | none |
| `ric-plt-a1` | Near-RT RIC A1 mediator (Go) — serves A1-P `/A1-P/v2` northbound, forwards `A1_POLICY_REQ` over RMR, records `enforceStatus` in SDL | https://gerrit.o-ran-sc.org/r/ric-plt/a1 | `09a757b4fd63198d8690d50b52bfd04552d47f1f` | 3.2.3 | Apache-2.0 | `go build -o a1mediator ./cmd/a1.go` (Go >= 1.22 per `go.mod`; go1.24.7 used here; needs librmr installed) | true | `deploy/xapp-e2e/patches/a1mediator-policy-resp-type.patch` — upstream `Consume()` asserts `policy_type_id` is a JSON number but its own `A1_POLICY_REQ` serialises it as a string, so the stock binary panics on the echoed `A1_POLICY_RESP`; the patch accepts both encodings. Applied at build time by `run_stack.sh` / `fetch_dependencies.sh`; the submodule pins the clean upstream commit. |
| `ric-app-hw-python` | Official O-RAN-SC reference xApp — consumes A1 policies over RMR (`A1_POLICY_REQ`/`A1_POLICY_RESP`), the real policy consumer in the xApp E2E proof | https://gerrit.o-ran-sc.org/r/ric-app/hw-python | `a6d00525aa2f62f2457d84731da2b7d89e0b1013` | 1.1.1 (pkg 0.0.1) | Apache-2.0 | `python3.11 -m venv venv-xapp && venv-xapp/bin/pip install -e . && venv-xapp/bin/pip install 'protobuf==3.20.3'` | true | none (not a patch: `run_stack.sh` adds `"xapp_name": "hw-python"` to `init/config-file.json` at run time — the key upstream expects the RIC helm configmap to inject — and rewrites the JSON formatting in doing so) |
| `sim-a1-interface` | O-RAN-SC A1 simulator (`near-rt-ric-simulator`, `STD_2.0.0` flavour) — used for the OSC A1 conformance smoke (`scripts/osc_a1_live_smoke.py`) | https://gerrit.o-ran-sc.org/r/sim/a1-interface | `be2943f57211f62095dc5434099e331df237aafb` | 2.8.1-13-gbe2943f (master tip at pin time) | Apache-2.0 | `docker build near-rt-ric-simulator/` or run `near-rt-ric-simulator/src` under the `deps/locks/a1sim.txt` venv | true | none committed. The local build inserted this sandbox's HTTPS-proxy CA into the Dockerfile and relaxed the `python3=3.10.15-r0` apk pin (that exact alpine package revision is no longer served); both are environment-specific build fixes, not functional changes, and are deliberately not vendored. |
| `ocudu` | OCUDU O-RAN CU/DU stack (C++17, srsRAN lineage) — RAN-side component for E2 integration; its build was still in progress when this manifest was written, so no OCUDU-derived evidence is recorded yet | https://gitlab.com/ocudu/ocudu.git | `f46f5804e53fede1e5c3353420ce1bcd3f4e57c6` | `dev` branch tip, 2026-07-24 (also `origin/HEAD` at pin time) | BSD-3-Clause-Open-MPI | `cmake -B build && cmake --build build` (out-of-tree `build/` is local output only — the submodule records a commit, never build artifacts) | true | none |
| `flexric` | FlexRIC (Mosaic5G/OAI) near-RT RIC + E2 agent — cloned and pinned for the E2 leg of the stack; **not yet built or exercised** in this session, pinned now so the eventual E2 results are reproducible | https://gitlab.eurecom.fr/mosaic5g/flexric.git | `ef6d722f22191eea74089966983da1f5ec1fedd4` | `v2.0.0-224-gef6d722f` (dev tip at pin time) | OAI Collaborative Standards Software License v1.0 (CSSL); `LICENSES/` carries per-file SPDX licenses | Standard upstream flow (`mkdir build && cd build && cmake .. && make`); not yet validated here | true | none |

## Python package stacks (PyPI)

Exact `pip freeze` snapshots live in [`deps/locks/`](../deps/README.md):

| Environment | Lockfile | Key pins | Purpose |
| --- | --- | --- | --- |
| Horizon-RIC rApp venv | `deps/locks/horizon.txt` (75 entries) | authoritative spec in `pyproject.toml` | Environment the rApp, tests and benchmarks ran in (Python 3.11.15) |
| xApp venv (`venv-xapp`) | `deps/locks/xapp.txt` (19 entries) | `ricxappframe==2.2.0`, `protobuf==3.20.3`, `hw-python` editable @ `a6d0052` | The real xApp runtime; protobuf pinned because `ricxappframe` ships pre-3.19 `_pb2.py` stubs; needs Python <= 3.11 (`hiredis 2.0.0` setup.py imports removed stdlib `imp`) |
| A1-sim venv (`venv-a1sim`) | `deps/locks/a1sim.txt` (23 entries) | `Flask==2.2.5`, `connexion==2.14.2` (same top-level pins as upstream Dockerfile) | Runs the A1 simulator outside docker |

## System dependencies (exact apt packages)

| Component | Toolchain / packages |
| --- | --- |
| RMR (C) | `build-essential` (gcc, make), `cmake` |
| A1 mediator (Go) | Go >= 1.22 (`go.mod`; built here with go1.24.7 from upstream tarball — Ubuntu's `golang-go` works if >= 1.22), plus installed librmr |
| SDL backend | `redis-server` (the O-RAN-SC dbaas image is plain redis) |
| hw-python xApp | `python3.11`, `python3.11-venv`, `python3.11-dev`; runtime librmr; PyPI pin `protobuf==3.20.3` |
| A1 simulator | `docker.io` (image build) — or `python3.11` + `deps/locks/a1sim.txt` |
| OCUDU | `cmake`, `build-essential`, `libfftw3-dev`, `libmbedtls-dev`, `libsctp-dev`, `libyaml-cpp-dev`, `libgtest-dev`, `libzmq3-dev` |
| FlexRIC | `cmake`, `build-essential` (see upstream README for optional SWIG bindings; not yet built here) |

## Reproducing the stack

```bash
git submodule update --init                      # materialise all six pins
scripts/fetch_dependencies.sh /tmp/oran-deps     # …or plain clones + patch, no submodules
deploy/xapp-e2e/run_stack.sh                     # build + run RMR, patched mediator, xApp
```
