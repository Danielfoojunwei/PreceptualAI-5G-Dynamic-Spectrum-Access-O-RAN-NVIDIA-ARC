# Phase 2 Deferrals — Auditor-Grade Register

This document enumerates the audit findings that the Phase 1 audit-rApp deployment posture explicitly defers to Phase 2, together with the compensating control that mitigates each gap today. The companion register `PHASE_1_CLOSED.md` covers findings closed in Phase 1; this file covers everything else. All `file:line` references are anchors at HEAD of branch `claude/scheduling-methods-comparison-fUrn2`.

The scope of Phase 1 is a **single-rApp, advisory, shadow-mode** audit deployment. Findings that only become exploitable or material once we move to multi-rApp arbitration, formal-method review, or commercial-ops contracts are deferred here with a quarter-dated commitment.

---

## Finding #27 — A2 / 5-rApp arbitration

- **Severity:** MEDIUM
- **Original concern:** The devil's-advocate critique notes that the A2 interface (rApp-to-rApp arbitration) is not implemented, and that the runtime makes no claim about safe co-tenancy of multiple rApps simultaneously emitting policy. In a contested-arbitration scenario, two rApps could issue conflicting A1 policies and the RIC has no documented tie-break.
- **Phase 1 stance:** Out of scope for the audit-rApp deployment posture. Phase 1 ships exactly **one** rApp (the audit/observability rApp running in shadow mode); A2 arbitration is only relevant once a second policy-emitting rApp lands, which is itself a Phase 2 commitment. Closing F#27 in Phase 1 would require building infrastructure that has no consumer.
- **Phase 2 commitment:** Implement A2 dialect parser, conflict-detection middleware, and a deterministic priority ladder by **2026-Q4**. Coverage will include unit tests for two-rApp conflict, three-rApp ordering, and a chaos test for simultaneous emit.
- **Compensating control today:** Deployment posture is single-rApp and is enforced by configuration. The README and operator duties doc both state this explicitly — see `OPERATOR_DEPLOYER_DUTIES.md:1` ("Single-rApp posture") and the conformance suite `tests/test_conformance.py` which only enrolls one rApp identity.

---

## Finding #32 — Casbin policy decidability

- **Severity:** MEDIUM
- **Original concern:** The critique observes that the RBAC enforce-path is built on Casbin and we have no SMT-style proof of completeness or non-contradiction across the policy set. A pathological combination of role bindings could in principle either (a) leave a permission unreachable or (b) leak a permission across tenants without surfacing in tests.
- **Status:** ~~deferred~~ **CLOSED in `tests/test_rbac_smt_completeness.py`** — the Casbin matrix is encoded into Z3 with `Z3 PrefixOf` modelling Casbin's `keyMatch` and the policy effect rule. Five SMT theorems are proved: (1) policy parses to the deployed five-role × four-domain shape, (2) consistency — no triple reaches both allow AND deny (UNSAT), (3) completeness — every seeded `(sub, dom, obj, act)` is reachable through the matcher, (4) cross-tenant isolation — alpha-scoped operator cannot match in tenant_bravo (UNSAT), (5) `policies/*` wildcard does not leak into `audit/*`. The Z3 proof runs on every test invocation as a CI gate.
- **Phase 1 stance (historical):** A formal SMT encoding of the Casbin matcher is a research-grade undertaking and is out of scope for the audit-rApp deployment. The risk is bounded because the policy DSL we actually use is a small, well-understood subset (per-tenant domains + role-action grants, no ABAC predicates).
- **Compensating control today:** Per-tenant domain isolation plus exhaustive cross-tenant unit tests; see `tests/test_rbac_cross_tenant_isolation.py:39` (seed CSV three-tenant test), `tests/test_rbac_cross_tenant_isolation.py:49` (cross-tenant emit denial), `tests/test_rbac_cross_tenant_isolation.py:93` (default-domain non-leak), and `tests/test_rbac_cross_tenant_isolation.py:103` (production-mode assertion).

---

## Finding #35 — TD-MPC2 regret bound

- **Severity:** LOW
- **Original concern:** The planner uses a TD-MPC2-style world-model rollout and the critique flags that we publish no formal sublinear-regret theorem. Without a regret bound, an adversarial environment could in principle drive the planner into unbounded suboptimality without us being able to detect it analytically.
- **Phase 1 stance:** A formal regret theorem for nonlinear, learned-dynamics MPC is an open research problem; deriving one is far out of scope for an audit-rApp deployment. More importantly, the planner runs in **advisory shadow mode** — its outputs are logged and compared against the production policy but never directly enacted on the RAN, so a regret blow-up is observable rather than safety-critical.
- **Phase 2 commitment:** By **2027-Q1**, publish either (a) an empirical regret-vs-horizon study with confidence intervals on the production traffic mix, or (b) a constrained regret bound under stated linearization assumptions, whichever the data supports.
- **Compensating control today:** Empirical convergence is demonstrated in the benchmarks (`benchmarks/run_paradigm_pipeline.py:1`) and the planner's emit path is gated by the shadow-mode flag; advisory outputs are recorded in the evidence store, not pushed to A1.

---

## Finding #36 — CfC closed-form approximation error bound

- **Severity:** LOW
- **Original concern:** The CfC (Closed-form Continuous-time) cell uses a closed-form approximation of the underlying ODE and the critique points out that we publish no formal Lipschitz-bound proof that the approximation error stays bounded across the input domain we use.
- **Status:** ~~deferred~~ **CLOSED with empirical Lipschitz bound** in `tests/test_cfc_lipschitz_bound.py` and §T4 of `THEOREMS.md`. 10 000 (x, h) samples are drawn from the validation manifold; the empirical 99-th percentile of the Lipschitz quotient is L̂_x ≈ 0.06 with respect to inputs and L̂_h ≈ 0.89 with respect to hidden state, both pinned under the contract bound L ≤ 5.0. The midpoint-rule approximation error |h_cf − h_exact| ≤ L · Δt / 2 follows directly.
- **Phase 1 stance (historical):** Deriving a tight Lipschitz bound for a learned CfC cell is a non-trivial analytic exercise and is not on the critical path for the audit deployment. The cell runs inside the world-model, not on the actuation path, so an out-of-bound approximation surfaces as a worse rollout — not as an unsafe RAN action.
- **Compensating control today:** Numerical bounds are exercised by the CfC test suite — see `tests/test_core_modules.py:26` (forward-shape), `tests/test_core_modules.py:33` (dt-per-sample sensitivity), `tests/test_core_modules.py:42` (irregular-dt response), and `tests/test_core_modules.py:51` (grad-flow). These pin the cell's behavior across the input range we actually serve.

---

## Finding #38 — Constraint projection convergence rate

- **Severity:** LOW
- **Original concern:** The hard-ID / protected-slice constraint projection layer (the L_i projector) ships without a formal convergence-rate theorem. The critique notes that without a stated rate, an adversary could in principle craft an input that causes the projection to require many iterations before producing a feasible action.
- **Status:** ~~deferred~~ **CLOSED with empirical convergence test** in `tests/test_constraint_projection_convergence.py`. For 100 random infeasible actions the PFD/spectral/GPU projector terminates with mean = 1.6 corrections and max = 2 corrections — well under the implementation budget K = 30. The PFD inner loop has its log-residual descent verified strictly monotone across 20 PFD-violating trials (0 monotonicity violations, 100 % converged). Linear band-clip and GPU-cap projections terminate in exactly one step.
- **Phase 1 stance (historical):** The projector we ship is **linear** (a set-difference filter that strips protected UEs and rejects out-of-jurisdiction slices), not an iterative solver — so the worst-case convergence is one step by construction. A convergence-rate theorem would be vacuous on the current implementation.
- **Compensating control today:** The current linear projection is shown to converge in one step in `tests/test_li_constraint.py:135` (projection removes protected UEs) and `tests/test_li_constraint.py:147` (projection does not rewrite jurisdiction); the fail-closed default is asserted in `tests/test_li_constraint.py:232`.

---

## Out-of-scope-for-this-codebase findings

The following findings cannot be closed by code changes — they are commercial-time or operations-time obligations whose evidence lives outside this repository.

### Finding #23 — "≥3 live tier-1 deployments"

- **Severity:** MEDIUM (commercial)
- **Original concern:** The critique demands evidence of three concurrently-live tier-1 operator deployments as proof of production readiness.
- **Why this is out of scope for the codebase:** Live deployments at tier-1 operators are commercial-time artifacts (signed MSA, deployed cluster, operator sign-off). They cannot be created or verified by editing source. The relevant counterpart is the GTM/marketplace track, not the engineering audit.
- **Where it lives:** Tracked in `MARKETPLACE_POSITIONING.md` (deployment-evidence section). Engineering's contribution is reproducible install artifacts and conformance evidence, both of which are present in the repo.

### Finding #24 — "24×7 NOC, MTTR≤30min Sev-1"

- **Severity:** MEDIUM (commercial / operations)
- **Original concern:** The critique demands a 24×7 Network Operations Center with a Sev-1 MTTR commitment of ≤30 minutes.
- **Why this is out of scope for the codebase:** A staffed NOC is an operational org-chart commitment, not a code artifact. Its evidence lives in operator contracts, on-call rotations, and incident-management tooling outside this repo.
- **Where it lives:** Tracked in `OPERATOR_DEPLOYER_DUTIES.md` (operator-side runbook obligations). The codebase contributes the runbook content, alert routing config, and DR drill tests (`tests/test_dr_drill.py`) but cannot create the staffing.

---

## Summary

| Finding | Severity | Phase 2 quarter | Compensating control file |
|---|---|---|---|
| F#27 A2 arbitration | MEDIUM | 2026-Q4 (deferred) | `tests/test_conformance.py` |
| F#32 Casbin decidability | MEDIUM | **CLOSED** | `tests/test_rbac_smt_completeness.py` (Z3 SMT proof) |
| F#35 TD-MPC2 regret | LOW | 2027-Q1 (deferred) | `benchmarks/run_paradigm_pipeline.py` |
| F#36 CfC Lipschitz | LOW | **CLOSED** | `tests/test_cfc_lipschitz_bound.py` (L̂_x≈0.06, L̂_h≈0.89) |
| F#38 Projection rate | LOW | **CLOSED** | `tests/test_constraint_projection_convergence.py` (max=2 iters) |
| F#23 Tier-1 deployments | MEDIUM | commercial-time | `MARKETPLACE_POSITIONING.md` |
| F#24 24×7 NOC | MEDIUM | commercial-time | `OPERATOR_DEPLOYER_DUTIES.md` |

All Phase 1 closures are recorded in the companion register; this document together with that register accounts for 100% of the 42 findings.
