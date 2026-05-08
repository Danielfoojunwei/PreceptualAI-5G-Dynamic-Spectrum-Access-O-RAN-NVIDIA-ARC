# Code Quality Polish Report

Pass executed under the "no-behaviour-change, surgical fixes only" rules.
Final smoke test: **67 / 67 passed** (test_health_endpoints, test_federated,
test_li_constraint, test_dns_cache, test_secure_aggregation,
test_lora_adapter, test_drift_detector, test_private_auction).

---

## 1. TODO / FIXME / XXX / HACK / STUB sweep

| Phase | Total occurrences in `src/horizon_ric` |
|-------|----------------------------------------|
| Before | 2 |
| After  | 2 |
| Closed inline | 0 |
| Re-categorised as Phase-2/3 | 0 (already explicit) |

Both surviving markers live in
`src/horizon_ric/trading/private_auction.py`:

* line 357: `STUB` marker on `PrivateSecondPriceAuction._mpc_blind_rank`
  — already documented as a Phase-3 deliverable in the docstring.
* line 379: `TODO(phase-3)` referencing the DFKNT-2006 secure-comparison
  primitive that replaces the stub.

Both are explicit deferrals tied to the GAPS_TO_PILOT roadmap; leaving
untouched per the "already explicitly deferred → leave alone"
instruction.

---

## 2. Pyflakes warnings

| Phase | Lines emitted |
|-------|---------------|
| Before | 55 |
| After  | 18 |

The 18 remaining are all intentional:

* 7 are forward-reference type hints already annotated `# noqa: F821`
  (string-quoted refs in `rapp/health.py`, `rapp/a1_adapter.py`,
  `rapp/r1_adapter.py`, `contracts/domain_adapter.py`). Pyflakes does
  not honour `noqa`; ruff does.
* 2 are gated probe-imports in `rapp/__init__.py` (re-exported behind
  the ncclient try/except).
* 1 is `data/sionna_channel.py` `import sionna` — already `noqa: F401`
  (import-availability probe).
* 6 are dead-store `F841` unused locals (e.g. `lo` in `data/itu_r.py`,
  `n_polar` in `planner/physics/geodesy.py`). These read as
  documentation aids in numerical code; removing them is a behavioural
  edit (deletes a load), so left for a focused numerical-correctness
  pass.
* 2 are `attrs`/`empty` in `rapp/o1_adapter.py` and
  `encoder/spatial_prior.py` — same category.

37 unused imports were closed. Specifically:

* Removed 2 truly unused names from `evidence/rfc3161.py`
  (`VerifierBuilder`, `_RFCVerificationError`).
* Added `# noqa: F401  (re-exported)` on the gated O1 import in
  `rapp/__init__.py` so ruff stops flagging the deliberate re-export.
* Ruff `--fix` removed 33 stale single-name imports across `rapp/`,
  `policy/`, `scenarios/`, `trading/`, `sla/`, `continual/`,
  `federated/`, `integrations/`, `runtime/`, `planner/physics/`,
  `security/`, `data/`, `observability/`, `core/`, `io/connectors/`,
  `contracts/`.

---

## 3. Ruff (E,F,W,I)

| Phase | Total | F401 | I001 | F841 | E501 | E741 |
|-------|-------|------|------|------|------|------|
| Before | 67 | 39 | 20 | 6 | 1 | 1 |
| After  | 8 | 0 | 0 | 6 | 1 | 1 |

Auto-fix invoked on `--select=I` (import sort) and `--select=F401`.
Six remaining F841 unused-variable warnings are the same numerical
dead-stores discussed above; not auto-fixed because removal counts as
a behavioural change. The single `E501` and `E741` are isolated style
nits in `data/itu_r.py` (line length on a comment, single-letter
ambiguous name `l` in a Lagrange interpolation snippet).

---

## 4. Mypy lite (`runtime` + `evidence`)

Total: **6 errors**, top categories:

| Count | Error class |
|-------|-------------|
| 3 | `[arg-type]` — `dict | Awaitable` mismatch in `state_recovery.save_state`; `tuple[str|int, ...]` vs declared `tuple[str, ...]` in `dns_cache._Entry` |
| 2 | `[no-any-return]` — `chaos_test.py:62` returns Any from int-typed fn; `evidence/rfc3161.py:129` returns Any from bytes-typed fn |
| 1 | `[return-value]` — companion to the dns_cache arg-type |

Not addressed; volume below the 3-line "trivial" bar and would touch
behavioural surfaces (await semantics in `save_state`,
runtime narrowing in `dns_cache`).

---

## 5. Docstring coverage — Wave-5 modules

All five Wave-5 modules already had module docstrings ≥ 3 lines and
`__all__` blocks. Coverage table for *public* defs:

| Module | Module doc | Public classes (with doc) | Public top-level fns (with doc) |
|--------|------------|---------------------------|---------------------------------|
| `federated/secure_aggregation.py` | yes | 2/2 | n/a (only methods) |
| `continual/lora_adapter.py`       | yes | 1/1 | 3/3 |
| `continual/drift_detector.py`     | yes | 4/4 | n/a |
| `trading/private_auction.py`      | yes | **7/7** (was 4/7) | n/a |
| `runtime/dns_cache.py`            | yes | 2/2 | 1/1 |

Three public dataclasses in `private_auction.py` (`Commitment`,
`Reveal`, `PaillierCiphertext`, `PrivateAuctionResult`) had no
docstring; added minimal Args-style summaries describing the role of
each field.

Public-method docstring coverage is partial across all five modules —
short trivial methods (`__init__`, `to_bytes`, `to_b64`, `update`,
`reset`, `close`, `submit_*`) remain undocumented. Adding boilerplate
docstrings to every two-line wrapper is not a quality improvement, so
left as-is per the "don't gold-plate" rule.

---

## 6. `__all__` exports

All five Wave-5 modules already export an explicit `__all__` listing
their public names. Verified each is non-empty and contains the
intended public surface; no edits required.

---

## 7. Empty `pass` bodies

26 `pass`-statements found across `runtime/`, `rapp/`, `evidence/`,
`security/`, `data/`. **All** are legitimate exception-swallows inside
`except` blocks (chaos targets, cleanup paths, optional-import gates,
metric-emission failures), not unimplemented method stubs. None
warranted `raise NotImplementedError` or removal.

---

## Files touched

1. `src/horizon_ric/evidence/rfc3161.py` — removed 2 unused imports.
2. `src/horizon_ric/rapp/__init__.py` — added `noqa: F401` on the
   gated O1 re-export.
3. `src/horizon_ric/trading/private_auction.py` — added 4 dataclass
   docstrings.
4. Ruff `--fix --select=I,F401` touched the following files for
   import sorting / unused-import removal:
   * `src/horizon_ric/rapp/api.py`
   * `src/horizon_ric/rapp/lifecycle.py`
   * `src/horizon_ric/policy/td_mpc_planner.py`
   * `src/horizon_ric/policy/constraints.py`
   * `src/horizon_ric/scenarios/maritime.py`
   * `src/horizon_ric/sla/escalation.py`
   * `src/horizon_ric/sla/policy.py`
   * `src/horizon_ric/continual/drift_detector.py`
   * `src/horizon_ric/federated/aggregator.py`
   * `src/horizon_ric/federated/secure_aggregation.py`
   * `src/horizon_ric/integrations/raas.py`
   * `src/horizon_ric/runtime/leader_election.py`
   * `src/horizon_ric/runtime/graceful_degradation.py`
   * `src/horizon_ric/runtime/_chaos_target.py`
   * `src/horizon_ric/runtime/liveness.py`
   * `src/horizon_ric/planner/physics/doppler.py`
   * `src/horizon_ric/planner/physics/ntn_timing.py`
   * `src/horizon_ric/security/nis2_reporter.py`
   * `src/horizon_ric/data/aodt.py`
   * `src/horizon_ric/data/sionna_channel.py`
   * `src/horizon_ric/data/aerial.py`
   * `src/horizon_ric/observability/model_card.py`
   * `src/horizon_ric/core/encoder_registry.py`
   * `src/horizon_ric/io/connectors/kafka_connector.py`
   * `src/horizon_ric/contracts/metric_suite.py`
   * `src/horizon_ric/trading/private_auction.py` (also imports)

(Some files were touched by both manual edits and ruff auto-fix.)

---

## Honesty check

* No TODOs were "fixed inline" — all surviving markers are legitimate
  Phase-3 deferrals.
* The dead-store unused locals (`F841`) and the `E741`/`E501` nits are
  **not** fixed; they're listed here to make the residual surface
  visible.
* Mypy errors are **not** fixed; only counted and categorised.
* No tests altered; no behavioural code changed.
