# Reproducing Horizon-RIC's enforcement result — the front door

This directory is the **single entry point** an external party uses to reproduce
Horizon-RIC's verification gates without contacting us. You do not need to read
the CI workflows or hunt through `scripts/` to find which of ~20 gate scripts to
run, in what order, against which committed result. Run one command.

```bash
python audit/reproduce.py
```

That runs every **offline** gate — the ones that reproduce on a clean checkout
with nothing but the Python standard library, the `cryptography` package, and
the data already committed to this repository. No Horizon install, no Docker, no
git submodules, no network. It prints a `PASS` / `SKIP` / `FAIL` table and exits
non-zero if any gate that actually ran failed.

Everything the runner knows lives in [`MANIFEST.json`](MANIFEST.json): the
machine-readable catalogue of every gate, its exact command, the committed
evidence it checks, and the class of requirement you must satisfy to run it.

## The four commands

| Command | What it does |
| --- | --- |
| `python audit/reproduce.py` | Run every offline gate (`requirement: none`). Exit non-zero on any real failure. |
| `python audit/reproduce.py --list` | Print the full catalogue as a table (every gate, its requirement, its CI workflow, its command). Runs nothing. |
| `python audit/reproduce.py --all` | Attempt **every** gate, discovering at runtime whether each requirement is met on this host and **skipping** — loudly, with a reason — the ones that are not. |
| `python audit/reproduce.py --only <id> ...` | Run only the named gate id(s). |

A skip is always loud and always carries a reason. The runner never turns an
unmet requirement into a silent pass, and never reports success on a red gate.

## What "offline reproduction" means, honestly

The headline enforcement numbers (for example: on 8000 decisions drawn from
measured DeepMIMO campus geometry, 4658 requested actions are illegal and **0**
survive the Shield) live in committed result JSON under `benchmarks/results/`.
The CI job that produces those numbers rebuilds them from the raw DeepMIMO
ray-tracing dataset, which is a multi-gigabyte download and is **not** committed.

So the offline gates here do not re-derive the numbers from raw rays — that is
the `network`-class path below. Instead, each offline gate re-runs the **same
verifier CI runs**, pointed at the committed evidence, in a committed-vs-committed
self-check. That self-check is not a tautology: the verifiers assert the
substantive invariants directly, so they fail on bad committed data. For example
the poisoning→Shield verifier re-asserts:

* zero illegal emissions survive the Shield, and zero in-spec-but-harmful ones;
* a **non-zero** count of illegal requests on the unguarded path (a vacuous run
  proves nothing);
* the poisoned-planner vs real-geometry split of those violations;
* that the adjudicating oracle is still declared independent of the Shield.

If the committed evidence were tampered to hide a surviving illegal emission, or
its oracle were made to grade its own work, the offline gate goes red. That is
what makes running it evidence rather than ceremony.

## Requirement classes

`reproduce.py` discovers requirement satisfaction **at runtime** — it does not
assume. Each gate declares one of:

| Requirement | Met when… | Discovered by |
| --- | --- | --- |
| `none` | always | — |
| `horizon` | the Horizon-RIC package imports | `import horizon_ric` in a child process |
| `docker` | a Docker daemon is reachable | `docker` on `PATH` **and** `docker info` succeeds |
| `submodule:<name>` | `third_party/<name>` is a non-empty checkout | directory exists and is non-empty |
| `network` | never attempted automatically | see below |

`network`-class gates need a dataset (re)download before they can run at all, so
`reproduce.py` **never** attempts them — even under `--all` they skip with the
manual command. To run them yourself, follow the DeepMIMO build documented in
`datasets/deepmimo_asu_3p5/` (`build.py` / `build_angular.py`), then invoke the
command shown by `--list` for `deepmimo_reproduction` / `data_dependence`.

A gate may additionally declare a runtime import probe (for example
`a1_assurance_proof` needs the vendored simulator's `requests`/`connexion`
runtime, `pip install -r deps/locks/a1sim.txt`). When such a module is missing
the gate **skips** with the missing module named, rather than failing.

## Reproducing the non-offline gates

```bash
# Everything the host can support, skipping the rest with reasons:
python audit/reproduce.py --all

# Horizon-class gates: install the package first, then --all picks them up.
pip install -e .

# submodule-class gates: check the submodule out first.
git submodule update --init third_party/ocudu third_party/sim-a1-interface

# docker-class gate: start a Docker daemon, then --all picks it up.
```

## The committed evidence fixture

The offline evidence auditor `audit/verify_evidence.py` runs against
[`fixtures/evidence-sample.jsonl`](fixtures/evidence-sample.jsonl): a small,
committed, self-verifying per-tenant hash chain. Its hashes are computed by the
**same** `horizon_audit.chain.chain_hash` the verifier recomputes, so the
cryptographic linkage is genuine — the auditor independently re-derives every
hash. Rebuild it deterministically with:

```bash
python audit/fixtures/build_evidence_sample.py
```

## The gate catalogue at a glance

Run `python audit/reproduce.py --list` for the authoritative, always-current
table. As committed, the manifest catalogues 20 gates: 12 run offline
(`requirement: none`), the rest need Horizon (`g1_second_planner`,
`g6_multi_agent`), a submodule (`g8_ocudu_conformance`, `a1_assurance_wire`,
`a1_assurance_proof`), Docker (`xapp_e2e_proof`), or the DeepMIMO dataset over
the network (`deepmimo_reproduction`, `data_dependence`).
