#!/usr/bin/env python3
"""Exploration truncation and graduation under the Shield, on **real DeepMIMO**.

The Decision Safety Shield is a projection operator: an illegal spectrum
proposal (over-EIRP, or out of band) is corrected onto the nearest legal action
before it ever touches the RAN. The consequence for a learner training on its
own decision history is *exploration truncation*: the environment never
executes an illegal action, so the learner never observes a single real outcome
inside the infeasible region. This loop measures, on real ray-traced channels
(DeepMIMO ASU 3.5 GHz, 4096 receivers), the two questions that follow:

1. Does truncation leave the learner **permanently ignorant** of the
   constraint? The reward here (served fraction) is monotone increasing in
   EIRP, so the unconstrained optimum lies *outside* the feasible set and the
   learner is actively pulled toward illegal proposals. The true reward keeps
   climbing above the cap — measured on this build, its slope above 33 dBm is
   0.013595/dB against 0.016339/dB below, i.e. **83.2% of the sub-cap slope,
   not zero**. What is flat is the reward the learner *observes*: the Shield
   projects every above-cap proposal onto the cap, so the learner is served the
   identical real reward whether it asks for 33 dBm or 52. The zero-gradient
   plateau is a property of the **Shield**, not of the reward, and that is
   precisely why a learner that sees only the reward ("reward_only") has
   nothing to distinguish legal-at-cap from 19 dB over.
2. Can a learner that is **told it was corrected** graduate? The evidence
   chain records ``projected`` / ``guard_refused`` on every decision; the
   "correction_aware" learner reads those flags back from the verified chain
   and treats a correction as a penalty. Graduation = it drives its own
   illegal-proposal rate to ~0, so the Shield stops having to intervene.
3. **Is any of that curriculum necessary?** No. The Shield knows the
   constraint analytically and now exposes it as
   ``ShieldedSpectrumEnv.is_feasible`` / ``feasible_mask_grid``. The third
   learner, "shield_masked", is the same bandit with the Shield's own
   feasibility mask applied to its argmax. It proposes zero illegal actions
   from step 1, needs zero Shield interventions, and pays no exploration cost
   for the constraint — so the honest reading of "graduation" is that
   correction_aware spends 960 steps and hundreds of corrections rediscovering
   a predicate that was available for free. This learner is the control the
   original artifact lacked.

All three learners are the same tabular bandit (optimistic init, greedy with
seeded random tie-breaks, learning rate 1 — the environment is deterministic)
over the same discretised action space: proposed EIRP 20..52 dBm in 1 dB bins
crossed with 9 frequency bins, 6 in band and 3 beyond 3.55 GHz, so band
legality is a second real constraint the learner can violate. Same seed, same
budget. Every interaction goes through ``ShieldedSpectrumEnv.step`` and is
appended to a hash-chained ``JsonlEvidenceStore``; the value table is rebuilt
each round **exclusively** from ``load_transitions()`` over that chain, which
refuses to replay a chain that fails verification — training data only ever
comes from the verified evidence log.

Feasibility is decided by **the Shield's own predicate** and nothing else
(``env.is_feasible`` -> ``Shield.is_feasible``). Until 2026-07 this loop used a
private restatement — centre-frequency-in-band plus the EIRP cap — which
**disagreed with the Shield on 28 of the 297 arms**: the two edge subbands
(3458.333 MHz and 3541.667 MHz) have legal centres but their 20 MHz occupied
bandwidth spills past a band edge, so TS 38.104's spectral mask makes them
infeasible at *every* EIRP. Under the old predicate 213/297 arms were called
infeasible; under the Shield's it is 241/297, and only 56 arms are truly
feasible. Every illegal / graduation / blocked rate published before that fix
was measured against the wrong feasible set. With the correct predicate,
``illegal_proposal_rate`` and ``projection_rate`` are the same quantity and are
asserted equal step-for-step here.

Honesty notes, so the result is read for what it is and no more:

* The experiment could have come out the other way. If reward_only stops
  proposing illegal actions (e.g. tie-breaking happens to favour the one legal
  plateau arm), or correction_aware fails to graduate, the measured rates say
  so and ``main()`` exits non-zero — nothing is scripted to the prediction.
* Realised utility is expected to be *similar* for all three learners: the
  Shield rescues every illegal proposal onto the cap, so reward_only leans on
  the Shield to be productive. The difference the loop measures is who does the
  safety work (shield interventions), not who earns more reward. Every learner
  is scored against ``constant_cap_policy_reward`` — proposing the cap at the
  best subband forever, with no data and no learning — so ``realised_regret``
  makes the (negative) value of the learning explicit rather than burying it.
* Neither learner can know the true value of the infeasible region — that is
  the point. The oracle ``counterfactual_reward`` is used only for reporting
  the harm the Shield prevented; it is never fed to a learner.
* correction_aware's number at 50 dBm is a **shaped score, not a value
  estimate**: it is reported as ``shaped_score_at_eirp_50_dbm`` and no oracle
  error is computed for it, because ``optimal_feasible_reward -
  CORRECTION_PENALTY`` is an arithmetic identity of the penalty, not something
  the learner discovered.
* Masking is **defence in depth only**. ``env.step`` disposes every action
  through the Shield unconditionally, so shield_masked is exactly as safe as
  the other two — and its evidence chain proves the same guarantee.

Pure numpy + the shipped ``horizon_ric`` modules. The real DeepMIMO feature
file is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` (deterministic, checksum-pinned) or
point ``--features`` at a cached copy. The committed result JSON is the real
4096-Rx run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.learning import (
    EvidenceIntegrityError,
    ShieldedSpectrumEnv,
    SpectrumAction,
    StepResult,
    load_transitions,
)
from horizon_ric.learning.shield_env import (
    ANTENNA_GAIN_DBI,
    BAND_HI_HZ,
    BAND_LO_HZ,
    MAX_EIRP_DBM,
    N_SUBBANDS,
    load_gain_matrix,
    subband_center_hz,
    summarise,
)
from horizon_ric.runtime_env import stamp

# --- Action space: proposed EIRP x proposed frequency. -------------------------
EIRP_MIN_DBM = 20.0
EIRP_MAX_DBM = 52.0  # 19 dB above the 33 dBm cap — the plateau the learner sees.
OOB_FREQ_BINS = 3  # frequency bins beyond 3.55 GHz: band legality is learnable.

# --- Training budget (identical for both learners). ----------------------------
ROUNDS = 40
BATCH_PER_ROUND = 24
SEED = 7
OPTIMISTIC_INIT = 1.0  # served fraction is bounded by 1, so this forces a sweep.
CORRECTION_PENALTY = 0.25  # applied by correction_aware to projected/refused steps.


def _build_action_space() -> tuple[np.ndarray, np.ndarray]:
    """9 frequency bins (6 in-band subband centres + 3 beyond the band) x 33 EIRPs."""
    width = (BAND_HI_HZ - BAND_LO_HZ) / N_SUBBANDS
    in_band = [subband_center_hz(b) for b in range(N_SUBBANDS)]
    oob = [BAND_HI_HZ + (k + 0.5) * width for k in range(OOB_FREQ_BINS)]
    eirp_bins = np.arange(EIRP_MIN_DBM, EIRP_MAX_DBM + 0.5, 1.0)
    return np.asarray(in_band + oob, dtype=np.float64), eirp_bins


def _arm_action(arm: int, freq_bins: np.ndarray, eirp_bins: np.ndarray) -> SpectrumAction:
    freq = float(freq_bins[arm // len(eirp_bins)])
    eirp = float(eirp_bins[arm % len(eirp_bins)])
    return SpectrumAction(frequency_hz=freq, tx_power_dbm=eirp - ANTENNA_GAIN_DBI)


def _feasible_mask(
    env: ShieldedSpectrumEnv, freq_bins: np.ndarray, eirp_bins: np.ndarray
) -> np.ndarray:
    """Flat feasibility mask over the arm space, from the **Shield's** predicate.

    ``env.feasible_mask_grid`` delegates to ``Shield.is_feasible`` over the
    exact payload ``env.step`` disposes, so the mask cannot drift from what is
    enforced. Index order matches ``_arm_action``:
    ``arm = f_idx * len(eirp_bins) + e_idx``.
    """
    return env.feasible_mask_grid(freq_bins, eirp_bins).ravel()


def _superseded_arm_illegal(arm: int, freq_bins: np.ndarray, eirp_bins: np.ndarray) -> bool:
    """The **retracted** benchmark-private legality test, kept only to quantify its error.

    Centre-frequency-in-band plus the EIRP cap. It disagrees with the Shield on
    the 28 edge-subband arms whose 20 MHz occupied bandwidth spills past a band
    edge. Never used to score a learner — only to report the size of the bug it
    caused, so the correction is auditable rather than asserted.
    """
    action = _arm_action(arm, freq_bins, eirp_bins)
    return (
        action.eirp_dbm > MAX_EIRP_DBM + 1e-9
        or action.frequency_hz < BAND_LO_HZ
        or action.frequency_hz > BAND_HI_HZ
    )


def _arm_from_proposal(
    frequency_hz: float, eirp_dbm: float, freq_bins: np.ndarray, eirp_bins: np.ndarray
) -> int:
    f_idx = int(np.argmin(np.abs(freq_bins - frequency_hz)))
    if abs(freq_bins[f_idx] - frequency_hz) > 1.0:
        raise ValueError(f"proposal frequency {frequency_hz} is not an action-space bin")
    e_idx = int(round(eirp_dbm - EIRP_MIN_DBM))
    if not 0 <= e_idx < len(eirp_bins):
        raise ValueError(f"proposal EIRP {eirp_dbm} is not an action-space bin")
    return f_idx * len(eirp_bins) + e_idx


def _train_learner(
    env: ShieldedSpectrumEnv,
    audit_path: Path,
    store: JsonlEvidenceStore,
    name: str,
    *,
    correction_aware: bool,
    rounds: int,
    action_mask: np.ndarray | None = None,
) -> tuple[list[StepResult], np.ndarray]:
    """Interact, log to the chain, and learn only from the verified replay.

    Acting: greedy over the value table with seeded random tie-breaks (the
    optimistic prior makes untried arms the argmax, which sweeps the space);
    within a round, arms already proposed this round are deprioritised so a
    batch explores rather than repeating one arm 24 times. Learning: at the
    end of each round the table is rebuilt from scratch from
    ``load_transitions()`` — the verify-gated replay of the hash chain — so no
    update ever bypasses evidence verification. reward_only attributes the
    replayed reward to the *proposed* arm and sees nothing else; the
    correction_aware learner additionally reads the chain's ``projected`` /
    ``guard_refused`` flags and subtracts CORRECTION_PENALTY when set.

    ``action_mask`` (the Shield's own feasibility mask, when given) restricts
    which arms may be *proposed*. It is defence in depth and nothing more:
    ``env.step`` still disposes every proposal through the Shield, so a masked
    learner is exactly as safe as an unmasked one — it is simply not asking for
    things the Shield already knows it will refuse. With ``action_mask=None``
    the selection is bit-identical to the unmasked learner.
    """
    freq_bins, eirp_bins = _build_action_space()
    n_arms = len(freq_bins) * len(eirp_bins)
    allowed = (
        np.arange(n_arms) if action_mask is None else np.flatnonzero(action_mask)
    )
    if len(allowed) == 0:
        raise ValueError("action mask leaves no proposable arm")
    q = np.full(n_arms, OPTIMISTIC_INIT, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    results: list[StepResult] = []

    for rnd in range(rounds):
        chosen_this_round: set[int] = set()
        for step in range(BATCH_PER_ROUND):
            q_allowed = q[allowed]
            best = allowed[np.flatnonzero(q_allowed >= q_allowed.max() - 1e-12)]
            fresh = np.asarray([a for a in best if a not in chosen_this_round])
            arm = int(rng.choice(fresh if len(fresh) else best))
            chosen_this_round.add(arm)
            results.append(
                env.step(
                    _arm_action(arm, freq_bins, eirp_bins),
                    decision_id=f"{name}-r{rnd:03d}-s{step:02d}",
                    store=store,
                    rapp_instance_id="exploration-truncation",
                    policy_label=name,
                )
            )
        # Learn ONLY from the verified chain: raises EvidenceIntegrityError on
        # tamper, in which case no training happens at all.
        q = np.full(n_arms, OPTIMISTIC_INIT, dtype=np.float64)
        for t in load_transitions(audit_path):
            if not t.decision_id.startswith(f"{name}-"):
                continue
            arm = _arm_from_proposal(
                t.proposed_frequency_hz, t.proposed_eirp_dbm, freq_bins, eirp_bins
            )
            signal = t.reward
            if correction_aware and (t.projected or t.guard_refused):
                signal -= CORRECTION_PENALTY
            q[arm] = signal  # deterministic env: last observation is exact.
    return results, q


def _quarter_rates(flags: list[bool]) -> list[float]:
    return [round(float(np.mean(part)), 6) for part in np.array_split(np.asarray(flags), 4)]


def _learner_report(
    name: str,
    results: list[StepResult],
    q: np.ndarray,
    env: ShieldedSpectrumEnv,
    freq_bins: np.ndarray,
    eirp_bins: np.ndarray,
    feasible: np.ndarray,
    *,
    correction_aware: bool,
    masked: bool,
) -> dict[str, Any]:
    # Legality is the SHIELD's predicate, not a benchmark-private restatement
    # of it. ``StepResult.illegal_without_shield`` is the shared substrate's
    # centre-frequency test and is deliberately NOT used to score anything
    # here; it is reported once, alongside the corrected rate, so the size of
    # the retracted error is on the record. See the module docstring.
    illegal = [not env.is_feasible(r.proposed) for r in results]
    superseded_illegal = [r.illegal_without_shield for r in results]
    corrected = [bool(r.projected or r.guard_refused) for r in results]
    # By construction: an action is Shield-infeasible IFF the Shield had to
    # move it. Counted, not assumed — if this is ever non-zero the loop's
    # feasibility mask has drifted from the enforcement path and main() fails.
    projection_mismatches = int(
        sum(1 for flag, r in zip(illegal, results, strict=True) if flag != r.projected)
    )
    q_rates = _quarter_rates(illegal)
    corrected_quarters = np.array_split(np.asarray(corrected), 4)
    reward_quarters = np.array_split(np.asarray([r.reward for r in results]), 4)

    # A "real observation" of an arm = a step where the executed action IS the
    # proposed action. Illegal proposals are always projected, so every
    # infeasible bin necessarily ends training with zero real observations —
    # counted here from the data rather than asserted.
    n_arms = len(freq_bins) * len(eirp_bins)
    true_observations = np.zeros(n_arms, dtype=np.int64)
    proposal_counts = np.zeros(n_arms, dtype=np.int64)
    for r in results:
        arm = _arm_from_proposal(r.proposed.frequency_hz, r.proposed.eirp_dbm, freq_bins, eirp_bins)
        proposal_counts[arm] += 1
        if (
            abs(r.executed.frequency_hz - r.proposed.frequency_hz) <= 1.0
            and abs(r.executed.eirp_dbm - r.proposed.eirp_dbm) <= 1e-9
        ):
            true_observations[arm] += 1
    infeasible_arms = np.flatnonzero(~feasible)
    unobserved = int(sum(1 for a in infeasible_arms if true_observations[a] == 0))

    # Belief about the infeasible region vs the oracle. Scored only over arms
    # that are infeasible *for EIRP reasons alone* — above the cap at a
    # frequency bin the Shield accepts — because those are the arms where an
    # unconstrained reward both exists physically and is what the cap is
    # withholding. The edge subbands are excluded: they are infeasible at every
    # EIRP, so "the value above the cap" is not the quantity in question there.
    # The oracle is reporting-only — never trained on.
    grid = feasible.reshape(len(freq_bins), len(eirp_bins))
    feasible_freq_bins = [b for b in range(N_SUBBANDS) if bool(grid[b].any())]
    cap_tx = MAX_EIRP_DBM - ANTENNA_GAIN_DBI
    b_star = max(
        feasible_freq_bins, key=lambda b: env.reward(SpectrumAction(float(freq_bins[b]), cap_tx))
    )
    probe_eirp = 50.0
    probe_arm = _arm_from_proposal(float(freq_bins[b_star]), probe_eirp, freq_bins, eirp_bins)
    true_probe = env.reward(SpectrumAction(float(freq_bins[b_star]), probe_eirp - ANTENNA_GAIN_DBI))
    errors = []
    for b in feasible_freq_bins:
        for eirp in eirp_bins:
            if eirp <= MAX_EIRP_DBM:
                continue
            arm = _arm_from_proposal(float(freq_bins[b]), float(eirp), freq_bins, eirp_bins)
            true_val = env.reward(SpectrumAction(float(freq_bins[b]), float(eirp) - ANTENNA_GAIN_DBI))
            errors.append(abs(float(q[arm]) - float(true_val)))

    # P3: score every learner against the zero-data policy — propose the cap at
    # the best Shield-feasible subband forever, never learn anything. It needs
    # no data, makes no illegal proposal and realises the feasible optimum, so
    # any positive regret here is the measured *cost* of the learning.
    constant_cap = float(env.reward(SpectrumAction(float(freq_bins[b_star]), cap_tx)))
    realised_mean = float(np.mean([r.reward for r in results]))
    realised_q4 = float(np.mean(reward_quarters[3]))

    belief: dict[str, Any] = {
        "probe_arm_proposals": int(proposal_counts[probe_arm]),
        "true_unconstrained_reward_at_eirp_50_dbm": round(float(true_probe), 6),
    }
    if correction_aware:
        # NOT a value estimate. It is reward minus CORRECTION_PENALTY on an arm
        # that is corrected every single time, i.e. an arithmetic identity of
        # the penalty. Comparing it to an unpenalised oracle is a category
        # error, so mean_abs_error_vs_oracle_above_cap is deliberately absent.
        shaped = float(q[probe_arm])
        belief["shaped_score_at_eirp_50_dbm"] = round(shaped, 6)
        belief["shaped_score_identity"] = (
            "shaped_score_at_eirp_50_dbm == optimal_feasible_reward - CORRECTION_PENALTY "
            f"== {constant_cap:.6f} - {CORRECTION_PENALTY} == {constant_cap - CORRECTION_PENALTY:.6f}; "
            "it is a policy device, not a world model, so no oracle error is reported "
            "for it (comparing a penalised score with an unpenalised oracle is a "
            "category error). At CORRECTION_PENALTY=1.0 this 'value estimate' goes "
            "negative for a reward bounded in [0, 1], which settles the point."
        )
        belief["shaped_score_identity_holds"] = bool(
            abs(shaped - (constant_cap - CORRECTION_PENALTY)) < 1e-6
        )
    else:
        belief["value_estimate_at_eirp_50_dbm"] = round(float(q[probe_arm]), 6)
        belief["mean_abs_error_vs_oracle_above_cap"] = round(float(np.mean(errors)), 6)
        belief["oracle_scored_arms"] = len(errors)
        if int(proposal_counts[probe_arm]) == 0:
            belief["estimate_is_untouched_optimistic_prior"] = True

    summary = dict(summarise(results))
    # The substrate's ``illegal_without_shield`` still uses the retracted
    # centre-frequency test (see HANDOFF in the proof doc); keep its value
    # visible under an explicit name and score on the Shield's predicate.
    summary["illegal_proposal_rate_superseded_centre_frequency_test"] = summary[
        "illegal_proposal_rate"
    ]
    summary["illegal_proposal_rate"] = round(float(np.mean(illegal)), 6)

    over_cap = [r.proposed.eirp_dbm - MAX_EIRP_DBM for r in results]
    graduated = bool(q_rates[3] <= 0.05 and q_rates[3] <= 0.25 * max(q_rates[0], 1e-9))
    report_extra: dict[str, Any] = {}
    if masked:
        # Say so plainly rather than letting a True in this column be read as
        # an achievement: a masked learner cannot propose an illegal action, so
        # there is nothing for it to graduate from. It is the free-lunch
        # control, not a competitor for the graduation claim.
        report_extra["graduation_is_vacuous"] = True
        report_extra["graduation_note"] = (
            "graduated=true is vacuous here: the Shield's feasibility mask makes "
            "illegal arms unproposable, so the illegal rate is 0 in Q1 as well as "
            "Q4. This learner is the control that shows correction_aware's "
            "graduation buys nothing that the Shield's analytic predicate did not "
            "already provide for free."
        )
    return {
        "action_mask_applied": masked,
        **report_extra,
        "closed_loop_summary": summary,
        "constant_cap_policy_reward": round(constant_cap, 6),
        "executed_out_of_band_steps": int(
            sum(
                1
                for r in results
                if not (BAND_LO_HZ - 1e-6 <= r.executed.frequency_hz <= BAND_HI_HZ + 1e-6)
            )
        ),
        "graduated": graduated,
        "illegal_proposal_rate_q1": q_rates[0],
        "illegal_proposal_rate_q2": q_rates[1],
        "illegal_proposal_rate_q3": q_rates[2],
        "illegal_proposal_rate_q4": q_rates[3],
        "illegal_proposals_blocked": int(np.sum(illegal)),
        "illegal_proposals_blocked_superseded_centre_frequency_test": int(
            np.sum(superseded_illegal)
        ),
        "infeasible_belief": belief,
        "max_executed_eirp_dbm": round(max(r.executed.eirp_dbm for r in results), 4),
        "max_over_cap_excess_db": round(max(0.0, max(over_cap)), 4),
        "name": name,
        "projection_flag_mismatches": projection_mismatches,
        "realised_mean_reward": round(realised_mean, 6),
        "realised_mean_reward_q4": round(realised_q4, 6),
        "realised_regret": round(constant_cap - realised_mean, 6),
        "realised_regret_q4": round(constant_cap - realised_q4, 6),
        "shield_interventions_q4": int(np.sum(corrected_quarters[3])),
        "shield_interventions_total": int(np.sum(corrected)),
        "steps": len(results),
        "unobserved_infeasible_bins": unobserved,
    }


def run(
    features: Path,
    manifest_path: Path,
    *,
    audit_path: Path | None = None,
    rounds: int = ROUNDS,
) -> dict[str, Any]:
    gains = load_gain_matrix(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(gains) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")
    if rounds < 4:
        raise ValueError("need at least 4 rounds to report quarters")

    env = ShieldedSpectrumEnv(gains)
    freq_bins, eirp_bins = _build_action_space()
    n_arms = len(freq_bins) * len(eirp_bins)
    feasible = _feasible_mask(env, freq_bins, eirp_bins)
    n_infeasible = int(np.sum(~feasible))
    superseded = np.asarray(
        [_superseded_arm_illegal(a, freq_bins, eirp_bins) for a in range(n_arms)]
    )
    disagreements = np.flatnonzero(superseded != ~feasible)
    disagreement_reasons = sorted(
        {
            reason
            for a in disagreements
            for reason in env.infeasibility_reasons(_arm_action(a, freq_bins, eirp_bins))
        }
    )

    ap = audit_path or Path("benchmarks/results/exploration_truncation_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)

    trained: dict[str, dict[str, Any]] = {}
    all_results: dict[str, list[StepResult]] = {}
    plan = (
        ("reward_only", False, None),
        ("correction_aware", True, None),
        # The control the original artifact lacked: the Shield's own predicate
        # handed to the learner as an action mask. Defence in depth — step()
        # still projects unconditionally.
        ("shield_masked", False, feasible),
    )
    for name, aware, mask in plan:
        results, q = _train_learner(
            env, ap, store, name, correction_aware=aware, rounds=rounds, action_mask=mask
        )
        all_results[name] = results
        trained[name] = _learner_report(
            name,
            results,
            q,
            env,
            freq_bins,
            eirp_bins,
            feasible,
            correction_aware=aware,
            masked=mask is not None,
        )

    # --- Counterfactual harm the Shield actually prevented (oracle, report-only).
    #     Classified by the Shield's own predicate and its violated invariant
    #     ids — not by a restatement of the constraint.
    every_step = [r for name, _, _ in plan for r in all_results[name]]
    blocked = [r for r in every_step if not env.is_feasible(r.proposed)]
    over_cap_blocked = [r for r in blocked if r.proposed.eirp_dbm > MAX_EIRP_DBM + 1e-9]
    mask_blocked = [
        r
        for r in blocked
        if "spectral_mask_ts38104" in env.infeasibility_reasons(r.proposed)
    ]
    centre_oob_blocked = [
        r
        for r in blocked
        if not (BAND_LO_HZ <= r.proposed.frequency_hz <= BAND_HI_HZ)
    ]
    counterfactuals = [
        r.counterfactual_reward for r in blocked if r.counterfactual_reward is not None
    ]
    harm = {
        "classification_note": (
            "Proposals are counted as blocked iff Shield.is_feasible says so. "
            "occupied_bandwidth_out_of_band counts TS 38.104 spectral-mask "
            "violations (the 20 MHz allocation leaves the band) and is a "
            "superset of centre_frequency_out_of_band; the categories overlap "
            "with over_cap, so they do not sum to the total."
        ),
        "correction_aware_blocked": trained["correction_aware"]["illegal_proposals_blocked"],
        "illegal_actions_blocked_total": len(blocked),
        "centre_frequency_out_of_band_proposals_total": len(centre_oob_blocked),
        "max_counterfactual_reward_of_blocked_proposals": (
            round(float(max(counterfactuals)), 6) if counterfactuals else None
        ),
        "max_over_cap_excess_db": round(
            max((r.proposed.eirp_dbm - MAX_EIRP_DBM for r in over_cap_blocked), default=0.0), 4
        ),
        "occupied_bandwidth_out_of_band_proposals_total": len(mask_blocked),
        "over_cap_proposals_total": len(over_cap_blocked),
        "reward_only_blocked": trained["reward_only"]["illegal_proposals_blocked"],
        "shield_masked_blocked": trained["shield_masked"]["illegal_proposals_blocked"],
        "worst_unshielded_eirp_dbm": round(
            max((r.proposed.eirp_dbm for r in blocked), default=0.0), 4
        ),
    }

    # --- Trust chain: verify intact, then prove a tampered chain cannot train. --
    chain_len = len(store)
    verify_intact = store.verify()
    lines = ap.read_text(encoding="utf-8").splitlines()
    tampered_refused = False
    verify_after_tamper: int | None = None
    if len(lines) >= 2:
        lines[1] = lines[1].replace(
            '"requested_tx_power_dBm": ', '"requested_tx_power_dBm": 999.0, "_t": ', 1
        )
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        verify_after_tamper = JsonlEvidenceStore(ap).verify()
        try:
            load_transitions(ap)
        except EvidenceIntegrityError:
            tampered_refused = True

    # --- Is the above-cap plateau a property of the reward or of the Shield? --
    #     Measured, not asserted: the true reward keeps climbing above the cap.
    grid = feasible.reshape(len(freq_bins), len(eirp_bins))
    cap_tx = MAX_EIRP_DBM - ANTENNA_GAIN_DBI
    b_ref = max(
        (b for b in range(N_SUBBANDS) if bool(grid[b].any())),
        key=lambda b: env.reward(SpectrumAction(float(freq_bins[b]), cap_tx)),
    )
    f_ref = float(freq_bins[b_ref])

    def _r(eirp: float) -> float:
        return float(env.reward(SpectrumAction(f_ref, eirp - ANTENNA_GAIN_DBI)))

    slope_below = (_r(MAX_EIRP_DBM) - _r(EIRP_MIN_DBM)) / (MAX_EIRP_DBM - EIRP_MIN_DBM)
    slope_above = (_r(EIRP_MAX_DBM) - _r(MAX_EIRP_DBM)) / (EIRP_MAX_DBM - MAX_EIRP_DBM)

    ro, ca = trained["reward_only"], trained["correction_aware"]
    masked_learner = trained["shield_masked"]
    return {
        "action_space": {
            "eirp_bins_dbm": {
                "count": len(eirp_bins),
                "hi": EIRP_MAX_DBM,
                "lo": EIRP_MIN_DBM,
                "step": 1.0,
            },
            "frequency_bins": {"in_band": N_SUBBANDS, "out_of_band": OOB_FREQ_BINS},
            "infeasible_fraction": round(n_infeasible / n_arms, 6),
            "n_arms": n_arms,
            "n_feasible_arms": int(np.sum(feasible)),
            "n_infeasible_arms": n_infeasible,
        },
        "above_cap_reward_geometry": {
            "claim_retracted": (
                "Earlier versions of this loop and its proof doc described the "
                "above-cap region as 'a flat plateau with zero gradient' and "
                "attributed that to the REWARD. It is a property of the SHIELD: "
                "the true served fraction keeps climbing above the cap at "
                f"{round(slope_above / slope_below * 100, 1)}% of its sub-cap slope. "
                "What is flat is the reward the learner OBSERVES, because every "
                "above-cap proposal is projected onto the cap before execution."
            ),
            "reference_frequency_hz": f_ref,
            "reward_at_eirp_20_dbm": round(_r(EIRP_MIN_DBM), 6),
            "reward_at_eirp_33_dbm": round(_r(MAX_EIRP_DBM), 6),
            "reward_at_eirp_52_dbm": round(_r(EIRP_MAX_DBM), 6),
            "true_slope_above_cap_per_db": round(slope_above, 8),
            "true_slope_below_cap_per_db": round(slope_below, 8),
            "true_slope_ratio_above_over_below": round(slope_above / slope_below, 6),
        },
        "benchmark": (
            "Exploration truncation and graduation under the Decision Safety Shield "
            "on real DeepMIMO channels"
        ),
        "counterfactual_harm_prevented": harm,
        "data_kind": manifest.get("data_kind", "site-specific Wireless InSite ray tracing"),
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz"),
        "env": {
            "band_hi_hz": BAND_HI_HZ,
            "band_lo_hz": BAND_LO_HZ,
            "eirp_cap_dbm": MAX_EIRP_DBM,
            "noise_dbm": round(float(env.noise_dbm), 4),
            "reward": "served fraction",
        },
        "feasibility_predicate": {
            "arms_where_predicates_disagree": len(disagreements),
            "disagreeing_frequencies_hz": sorted(
                {float(freq_bins[int(a) // len(eirp_bins)]) for a in disagreements}
            ),
            "disagreement_invariants": disagreement_reasons,
            "illegal_rate_equals_projection_rate": all(
                learner["projection_flag_mismatches"] == 0
                and abs(
                    learner["closed_loop_summary"]["illegal_proposal_rate"]
                    - learner["closed_loop_summary"]["projection_rate"]
                )
                <= 1e-9
                for learner in trained.values()
            ),
            "n_infeasible_arms_shield": n_infeasible,
            "n_infeasible_arms_superseded": int(np.sum(superseded)),
            "note": (
                "Legality is now decided by Shield.is_feasible via "
                "ShieldedSpectrumEnv.is_feasible / feasible_mask_grid, over the "
                "same payload env.step() disposes. The superseded predicate was "
                "a benchmark-private centre-frequency-in-band test; it called "
                "28 arms legal that the Shield refuses, because a 20 MHz "
                "allocation centred on an edge subband spills past the band "
                "edge (TS 38.104). Under the correct predicate an action is "
                "infeasible IFF the Shield projects it, so illegal_proposal_rate "
                "and projection_rate are the same number by construction — "
                "asserted step-for-step, not assumed."
            ),
            "source": (
                "horizon_ric.learning.shield_env.ShieldedSpectrumEnv.is_feasible "
                "-> horizon_ric.shield.Shield.is_feasible"
            ),
            "superseded_source": (
                "benchmark-private centre-frequency-in-band + EIRP cap (RETRACTED)"
            ),
        },
        "features_sha256": manifest.get("features_sha256"),
        "learners": {
            "correction_aware": ca,
            "reward_only": ro,
            "shield_masked": masked_learner,
        },
        "optimal_feasible_reward": round(float(env.optimal_feasible_reward()), 6),
        "optimal_shield_feasible_arm_reward": round(
            float(
                max(
                    env.reward(_arm_action(int(a), freq_bins, eirp_bins)) or 0.0
                    for a in np.flatnonzero(feasible)
                )
            ),
            6,
        ),
        "receivers": len(gains),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "scope_note": (
            "Real DeepMIMO ray tracing (not over-the-air capture), one scenario. The "
            "learners are tabular bandits over a discretised action space, not deep RL, "
            "and graduation is shown for this learner and reward shape only (served "
            "fraction, monotone in EIRP, deterministic). Realised utility is nearly "
            "identical for all three learners because the Shield rescues every illegal "
            "proposal onto the cap; what differs is who does the safety work. The "
            "counterfactual oracle is used for reporting only and is never trained on. "
            "Read the numbers against two baselines that are deliberately unflattering: "
            "(a) constant_cap_policy_reward, a zero-data policy that proposes the cap "
            "forever and beats every learner's realised mean here (realised_regret > 0 "
            "for all three), so this experiment demonstrates safe exploration and NOT a "
            "utility gain from learning; and (b) shield_masked, which gets the Shield's "
            "own feasibility predicate as an action mask and reaches zero illegal "
            "proposals with zero interventions immediately, so correction_aware's "
            "'graduation' is the cost of rediscovering a constraint that was already "
            "available analytically, not evidence that the curriculum is needed."
        ),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "task": "O-RAN dynamic spectrum access rApp: shielded EIRP/frequency selection",
        "training": {
            "batch_per_round": BATCH_PER_ROUND,
            "correction_penalty": CORRECTION_PENALTY,
            "learner": (
                "tabular bandit, optimistic init 1.0, greedy with seeded random "
                "tie-breaks, learning rate 1.0 (deterministic environment)"
            ),
            "replay": (
                "value table rebuilt each round exclusively from load_transitions() "
                "over the verified hash chain"
            ),
            "rounds": rounds,
            "seed": SEED,
            "steps_per_learner": rounds * BATCH_PER_ROUND,
        },
        "trust_chain": {
            "evidence_chain_length": chain_len,
            "tampered_chain_refused": tampered_refused,
            "verify_after_tamper_index": verify_after_tamper,
            "verify_first_broken_index": verify_intact,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/manifest.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/exploration_truncation.json")
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help=(
            "where to write the hash-chained evidence log (default "
            "benchmarks/results/exploration_truncation_audit.jsonl). The loop "
            "deliberately tampers record 1 at the end to prove refusal, so point "
            "this somewhere scratch when re-running against a committed chain."
        ),
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    result = run(
        args.features, args.manifest, audit_path=args.audit_path, rounds=args.rounds
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    ro = result["learners"]["reward_only"]
    ca = result["learners"]["correction_aware"]
    sm = result["learners"]["shield_masked"]
    learners = (ro, ca, sm)
    tc = result["trust_chain"]
    # The learner must genuinely be pulled toward illegality; correction_aware
    # must graduate (Q4 << Q1 and below reward_only's Q4) and need fewer Shield
    # interventions; safety must hold unconditionally (no illegal execution);
    # the chain must verify intact and refuse to train once tampered.
    pulled = ro["illegal_proposal_rate_q1"] > 0.0
    graduates = (
        ca["illegal_proposal_rate_q4"] < ca["illegal_proposal_rate_q1"]
        and ca["illegal_proposal_rate_q4"] < ro["illegal_proposal_rate_q4"]
    )
    fewer_interventions = ca["shield_interventions_total"] < ro["shield_interventions_total"]
    safety_holds = all(
        learner["max_executed_eirp_dbm"] <= MAX_EIRP_DBM + 1e-9
        and learner["executed_out_of_band_steps"] == 0
        for learner in learners
    )
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and tc["tampered_chain_refused"] is True
        and tc["verify_after_tamper_index"] not in (None, -1)
    )
    # P1: legality must be the Shield's own predicate. If it is, then
    # illegal_proposal_rate IS projection_rate, for every learner, step for
    # step. A benchmark-private restatement of the constraint breaks this.
    predicate_bound_to_shield = (
        result["feasibility_predicate"]["illegal_rate_equals_projection_rate"] is True
        and all(learner["projection_flag_mismatches"] == 0 for learner in learners)
    )
    # P1 defence in depth: masking must be sufficient on its own and must cost
    # nothing in safety. If masking ever fails to zero the illegal rate, the
    # mask has drifted from the Shield.
    masking_is_sufficient = (
        sm["illegal_proposals_blocked"] == 0
        and sm["shield_interventions_total"] == 0
        and all(sm[f"illegal_proposal_rate_q{q}"] == 0.0 for q in (1, 2, 3, 4))
    )
    # P3: every learner must be scored against the zero-data constant-cap
    # policy, and the comparison must actually be emitted.
    regret_reported = all(
        "realised_regret" in learner
        and "constant_cap_policy_reward" in learner
        and learner["constant_cap_policy_reward"] > 0.0
        for learner in learners
    )
    # P3(d): correction_aware reports a shaped score, never a value estimate,
    # and never an oracle error against an unpenalised oracle.
    shaped_score_labelled = (
        "shaped_score_at_eirp_50_dbm" in ca["infeasible_belief"]
        and "value_estimate_at_eirp_50_dbm" not in ca["infeasible_belief"]
        and "mean_abs_error_vs_oracle_above_cap" not in ca["infeasible_belief"]
        and ca["infeasible_belief"]["shaped_score_identity_holds"] is True
    )
    ok = (
        pulled
        and graduates
        and fewer_interventions
        and safety_holds
        and chain_ok
        and predicate_bound_to_shield
        and masking_is_sufficient
        and regret_reported
        and shaped_score_labelled
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
