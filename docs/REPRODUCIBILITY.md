# Reproducibility of the closed-loop benchmarks

Horizon's four closed-loop benchmarks and the DSA aggregate benchmark are
*reproduction targets*: the `realdata` workflow rebuilds the licence-gated
DeepMIMO features from a checksum-pinned archive, re-runs each loop, and a
verifier asserts the committed numbers still hold. This note records the one
dependency that can break that contract silently, and what we do about it.

## The failure mode: numpy is effectively a seed

Every closed-loop benchmark draws randomness from `numpy.random.default_rng`.
numpy's own `Generator` documentation is explicit:

> **No Compatibility Guarantee.** `Generator` does not provide a version
> compatibility guarantee. In particular, as better algorithms evolve the bit
> stream may change.

So a numpy upgrade is indistinguishable, from outside, from a reseed. The loops
still run and the physics is unchanged, but every stochastic field can move —
and it would surface as a red reproduction check, which reads like an
unreproducible-science alarm rather than the dependency bump it actually is.

That matters here because a 12-seed sweep showed the reproduction gates have
very little headroom:

| gate | margin | seed-induced spread | outcome |
| --- | --- | --- | --- |
| `krum_rmse < baseline` (`verify_federated_coverage.py`) | 0.6501 dB | sd **1.8687 dB** | fails at 1 of 12 seeds |
| `robust_target_receiver == 9` (exact compare) | — | 4 distinct values | reproduces at **8 of 12** seeds |

Those gates are unchanged for now (see `deploy/shield-learning/ERRATA.md`, P5),
so the pin below is what keeps them from flaking.

## What we do

1. **`realdata.yml` pins numpy explicitly** — `pip install 'numpy==2.2.6'`
   before the extras. This is not a new constraint, it freezes what the job
   already resolved to.
2. **`pyproject.toml` bounds numpy at `<3`** so a major release cannot silently
   reseed any consumer of the library.
3. **Every result JSON carries a `runtime` block** (`horizon_ric.runtime_env`)
   recording the numpy version, interpreter version and implementation. If a
   reproduction ever fails, `git diff` on the result JSON says immediately
   whether the environment moved.

## Why CI's numpy differs from the committed results — deliberately

The committed result JSONs were produced under **numpy 2.4.6**; the `realdata`
job runs under **numpy 2.2.6**. This is not drift, and it is not fixable by
pinning harder:

- The `realdata` extra installs `deepmimo==4.0.0`, whose own metadata requires
  `numpy<2.3,>=1.19.5`. 2.2.6 is the newest numpy satisfying it.
- `deps/locks/horizon.txt` pins `numpy==2.4.6` for the **runtime** environment.
  That pin is **unsatisfiable alongside the realdata extra**, which is why the
  `realdata` workflow deliberately does *not* install from the lockfile.

The results reproduce exactly across those two versions today — that is an
empirical observation, not a guarantee numpy offers. The `runtime` block is what
makes it checkable rather than assumed.

**Consequently no verifier asserts the `runtime` block.** It is diagnostic
only; a gate on it would fail on every CI run by construction.
`tests/test_runtime_env.py` enforces that no `verify_*.py` script references it.

## If a reproduction check goes red

1. `git diff benchmarks/results/<name>.json` — if the `runtime` block moved, the
   environment changed and you are looking at a reseed, not a physics
   regression. Check the numpy version first.
2. If the environment is identical, it is a genuine regression; bisect the loop
   or substrate change.
3. If it is a Krum-RMSE or `robust_target_receiver` assertion specifically, note
   the margins in the table above before concluding anything — those two are
   known to be seed-fragile and are scheduled to be re-gated on separation
   rather than on a 0.35σ margin.
