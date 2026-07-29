# Errata — shield-learning and federated-coverage results

A five-agent adversarial post-mortem re-ran every headline in
`deploy/shield-learning/*.md`, `deploy/federated-coverage/FEDERATED_COVERAGE_PROOF.md`
and the corresponding result JSONs against the same real DeepMIMO data. Most of
what we published about *shielded learning* turns out to be a property of our own
harness rather than of the Shield. This file corrects the record. It is written
before the fixes land so the claims and the corrections are auditable together.

**The one claim that survived everything:** no learner ever emitted an illegal
action — `max_executed_eirp_dbm = 33.0`, `executed_out_of_band_steps = 0` — under
every reward function, seed, and credit rule tested. The hard safety guarantee is
not overstated. Nearly everything else below is.

## Apportionment

Roughly **80–85 % of the reported "bad learning" is our experiment design**;
15–20 % is inherent to learning behind a projection operator. The environment
determines whether the pathology is *visible*; the learner architecture
determines whether it *costs* anything. Measured 2×2 on the same 4096 receivers:

| | tabular value table | smooth (poly) value model |
| --- | --- | --- |
| **boundary-optimal reward (shipped)** | tie = 20, projQ4 = 0.9333, **realised loss 0.000000** | **loss 0.000000** |
| **interior-optimal reward** | no pathology | **loss 0.055420 = 29.2 % of optimum** |

The shipped experiment sits in the top-left — the one cell where the pathology is
maximally visible and costs exactly nothing. The mechanism that *does* cost
utility (the projection makes the proposal-indexed regression target constant
across the infeasible region, so a limited-capacity value model fits that plateau
and drags its argmax out of the feasible optimum) is excluded by our design twice
over, and appears in no proof doc or result JSON.

## Corrections

1. **"~0.3 dB subband spread" — wrong.** The per-subband *mean gain* spread on the
   canonical 4096-receiver build is **0.0850 dB**. The 0.3 dB figure came from the
   earlier 512-receiver cache and was carried forward incorrectly, including into
   `FEDERATED_COVERAGE_PROOF.md` and the federated commit message. (The 3.7020 dB
   figure that also circulates is the *per-receiver across-subband* spread — a
   different quantity, and the one that matches `worst_subband.mean_regret_db`.)

2. **"Correction-aware learning graduates because it is penalised for being
   corrected" — false.** `CORRECTION_PENALTY` is **inert**: sweeping it 0.0 → 5.0
   changes only `argmax_tie_count` (a reporting field). `illegal_proposals = 32`,
   `projection_rate_last_quarter = 0.016667` and `realised_mean_reward = 0.335835`
   are identical at every value including zero. Graduation comes from indexing the
   value table by the **executed** bin, which makes infeasible bins permanently
   unreachable by the greedy rule — not from the correction signal. In the credit
   loop `projection_aware` and `correction_aware` are bit-identical at equal RNG
   (6/6 seeds); the published B-vs-C difference is 100 % RNG offset.

3. **`value_mae_infeasible = 0.1557` is a free parameter of the grid, not a
   measured failure.** It equals `mean_{b>33}|R(33) − R(b)|` and scales purely with
   where we put the top of the EIRP grid: 0.0183 at grid top 34 dBm, 0.1557 at the
   chosen 52 dBm, 0.4610 at 110 dBm. It has **zero** effect on realised reward.

4. **"A flat plateau with zero gradient above the cap" — false of the objective.**
   The true reward slope is 0.016339/dB below the cap and **0.013595/dB above** it
   (83 % of the sub-cap slope); it does not saturate anywhere in the action space.
   The plateau is manufactured entirely by the projection clipping every over-cap
   proposal onto one executed action. `exploration_truncation_loop.py:16-18` states
   it as a property of the reward; it is a property of the Shield.

5. **"213/213 unobserved infeasible bins" is not a finding.** It is a definitional
   restatement of "the projection never executes an infeasible action."

6. **The 0.9333 illegal-proposal rate is a tie-break convention.** `naive` breaks
   ties by seeded random choice while the other regimes use `tied[0]` (lowest EIRP,
   a safety-favouring rule). Making all regimes use `tied[0]` moves naive's illegal
   proposals 201 → 33 and its Q4 rate 0.9333 → **0.0000**, with `MAE_inf` unchanged.
   Using `tied[-1]` gives 0.9833. No learned quantity changes.

7. **Illegal/graduation rates in `exploration_truncation.json` are measured against
   the wrong feasible set.** `_arm_illegal` tests centre-frequency-in-band; the
   Shield tests occupied-bandwidth-in-band. They disagree on **28 of 84** arms. The
   `projection_rate 0.909` vs `illegal_proposal_rate 0.878` gap is the visible
   residue. These rates must be recomputed against the Shield's own feasibility check.

8. **DP's "350 dB is an honest privacy/utility trade-off" — it is a defect, not a
   trade-off.** At the *same* ε = 4.1447, same δ, same K, same accountant, tuned
   clipping and round count give **15.86 dB** — below the 20.11 dB baseline. A 22×
   utility difference at identical certified privacy. Also, the comment
   `DP_NOISE = 8.0 # certified eps ~ 3.2` understates the shipped accountant's own
   output (4.1447) by 30 %, and is contradicted by the JSON in the same repo.

9. **Krum's "7.6 dB poison tax" is 75 % not a poison tax.** 5.653 of the 7.586 dB
   is present with **zero** adversaries (clean Krum 17.527 vs clean FedAvg 11.874) —
   it is the cost of discarding K−1 updates under a non-IID split. FLTrust reaches
   11.904 and survives 7/10 malicious where Krum fails at 4/10.

10. **median 22.16 / trimmed-mean 35.81 are not intrinsic robustness costs.** With
    no attack they cost 0.256 dB and 0.119 dB. 35.814 is the **max of 12 seeds**
    (mean 24.658); both beat the baseline in 3/12 seeds. The shipped figures reflect
    `beta = f = 3` at K = 10 trimming asymmetrically against 3 identical colluders.

11. **"Krum stays below the 20.11 dB baseline" holds by only 0.36σ** and fails
    outright at 1/12 partition seeds (20.655 > 20.110). `robust_target_receiver`
    reproduces at 8/12 seeds, so the exact-match CI assertion is seed-fragile.

12. **`safety_utility_frontier`'s `free_constraint_case` is worth 12 receivers.**
    The per-subband reward spread at the cap is 0.002930 of 4096, and the argmax
    subband is not stable across EIRP (1 at 26 dBm, 5 at 32, 3 at 33 and 46). The
    EIRP axis spans 0.4707 — **161×** the frequency axis.

**Retracted from the post-mortem itself:** a claimed `features_sha256` mismatch —
`compute_dataset_sha256` returns the committed `ab4414a1…` exactly.

## What this does not change

The safety results stand unmodified: the Shield projected every unsafe proposal
(over-EIRP → 33 dBm, out-of-band → in-band), the guard chain refused every
uncorrected emit, no illegal action ever reached the RAN, every evidence chain
verified intact and every injected tamper was caught at its exact index, and
`load_transitions()` refused to build training data from a tampered chain. Those
held under every perturbation the post-mortem could construct.

## Follow-up

**P4 and P5 are DONE** (see `deploy/federated-coverage/FEDERATED_COVERAGE_PROOF.md`).
Correction 8 above said DP's 350.5 dB was a defect rather than a trade-off: at the
identical certified epsilon = 4.1447 it is now **14.34 dB**, below the 20.11 dB
baseline, achieved by calibrating the clip on public server-root receivers and
cutting 40 rounds to 12. The adjacency convention was deliberately NOT relaxed —
halving sigma at the same numeric epsilon would be a weaker guarantee, not a fix.
Correction 9 said most of Krum's tax was not a poison tax; measured on the current
tree its poison tax is **exactly 0.00 dB** (poisoned = clean = 14.4572), so all of
it is non-IID aggregation cost. FLTrust replaces it at **11.97 dB** poisoned
against clean FedAvg's 11.92, holding **13.46 dB at 7 of 10 malicious clients**.
Correction 11's fragile gates are retired: 17 gates now hold 24/24 seeds and each
is proven to fail under an injected regression.

Two figures quoted in this errata did not survive re-measurement and are corrected
here: the clip-never-binds observation (0/400 clipped) holds only on the *clean*
trajectory — on the actual DP trajectory 243/400 updates clip, because the noise
inflates the iterate — and the one-shot release reaching clean-FedAvg parity at
K=200 required an oracle clip read off the private data. Under a legitimate public
bound it is 12.84 dB at K=200 and 12.21 dB at K=400.

**Still outstanding: P1, P2, P3.** Mask the action space with the Shield's own
`is_feasible()` (which also fixes `_arm_illegal` disagreeing with the Shield on
28/84 arms); delete or fix regime C in `credit_assignment_loop.py`; and re-gate the
shield-learning benchmarks on realised regret against the zero-data constant-33 dBm
policy, which scores 0.347412 — identical to the optimum.
