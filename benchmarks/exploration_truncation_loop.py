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
   learner is actively pulled toward illegal proposals. Every proposal above
   the 33 dBm cap executes at the cap and returns the *identical* real reward —
   a flat plateau with zero gradient — so a learner that sees only the reward
   ("reward_only") has nothing to distinguish legal-at-cap from 19 dB over.
2. Can a learner that is **told it was corrected** graduate? The evidence
   chain records ``projected`` / ``guard_refused`` on every decision; the
   "correction_aware" learner reads those flags back from the verified chain
   and treats a correction as a penalty. Graduation = it drives its own
   illegal-proposal rate to ~0, so the Shield stops having to intervene.

Both learners are the same tabular bandit (optimistic init, greedy with seeded
random tie-breaks, learning rate 1 — the environment is deterministic) over the
same discretised action space: proposed EIRP 20..52 dBm in 1 dB bins crossed
with 9 frequency bins, 6 in band and 3 beyond 3.55 GHz, so band legality is a
second real constraint the learner can violate. Same seed, same budget. Every
interaction goes through ``ShieldedSpectrumEnv.step`` and is appended to a
hash-chained ``JsonlEvidenceStore``; the value table is rebuilt each round
**exclusively** from ``load_transitions()`` over that chain, which refuses to
replay a chain that fails verification — training data only ever comes from
the verified evidence log.

Honesty notes, so the result is read for what it is and no more:

* The experiment could have come out the other way. If reward_only stops
  proposing illegal actions (e.g. tie-breaking happens to favour the one legal
  plateau arm), or correction_aware fails to graduate, the measured rates say
  so and ``main()`` exits non-zero — nothing is scripted to the prediction.
* Realised utility is expected to be *similar* for both learners: the Shield
  rescues every illegal proposal onto the cap, so reward_only leans on the
  Shield to be productive. The difference the loop measures is who does the
  safety work (shield interventions), not who earns more reward.
* Neither learner can know the true value of the infeasible region — that is
  the point. The oracle ``counterfactual_reward`` is used only for reporting
  the harm the Shield prevented; it is never fed to a learner.
* Two edge in-band bins (subbands 0 and 5) draw Shield corrections even though
  their centre frequency is legal, because the 20 MHz allocation sticks out of
  the band. correction_aware is penalised for those too — they genuinely
  require correction — but they never count toward illegal_proposal_rate,
  which uses the environment's ``illegal_without_shield`` flag.

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


def _arm_illegal(arm: int, freq_bins: np.ndarray, eirp_bins: np.ndarray) -> bool:
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
    """
    freq_bins, eirp_bins = _build_action_space()
    n_arms = len(freq_bins) * len(eirp_bins)
    q = np.full(n_arms, OPTIMISTIC_INIT, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    results: list[StepResult] = []

    for rnd in range(rounds):
        chosen_this_round: set[int] = set()
        for step in range(BATCH_PER_ROUND):
            best = np.flatnonzero(q >= q.max() - 1e-12)
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
) -> dict[str, Any]:
    illegal = [r.illegal_without_shield for r in results]
    corrected = [bool(r.projected or r.guard_refused) for r in results]
    q_rates = _quarter_rates(illegal)
    corrected_quarters = np.array_split(np.asarray(corrected), 4)

    # A "real observation" of an arm = a step where the executed action IS the
    # proposed action. Illegal proposals are always projected, so every
    # infeasible bin necessarily ends training with zero real observations —
    # counted here from the data rather than asserted.
    n_arms = len(freq_bins) * len(eirp_bins)
    true_observations = np.zeros(n_arms, dtype=np.int64)
    for r in results:
        arm = _arm_from_proposal(r.proposed.frequency_hz, r.proposed.eirp_dbm, freq_bins, eirp_bins)
        if (
            abs(r.executed.frequency_hz - r.proposed.frequency_hz) <= 1.0
            and abs(r.executed.eirp_dbm - r.proposed.eirp_dbm) <= 1e-9
        ):
            true_observations[arm] += 1
    infeasible_arms = [a for a in range(n_arms) if _arm_illegal(a, freq_bins, eirp_bins)]
    unobserved = int(sum(1 for a in infeasible_arms if true_observations[a] == 0))

    # Belief about the infeasible region vs the oracle (in-band above-cap bins,
    # where a true unconstrained reward exists; out-of-band bins have no
    # measured channel at all). The oracle is reporting-only — never trained on.
    cap_tx = MAX_EIRP_DBM - ANTENNA_GAIN_DBI
    in_band = list(range(N_SUBBANDS))
    b_star = max(in_band, key=lambda b: env.reward(SpectrumAction(float(freq_bins[b]), cap_tx)))
    probe_eirp = 50.0
    probe_arm = _arm_from_proposal(float(freq_bins[b_star]), probe_eirp, freq_bins, eirp_bins)
    true_probe = env.reward(SpectrumAction(float(freq_bins[b_star]), probe_eirp - ANTENNA_GAIN_DBI))
    errors = []
    for b in in_band:
        for eirp in eirp_bins:
            if eirp <= MAX_EIRP_DBM:
                continue
            arm = _arm_from_proposal(float(freq_bins[b]), float(eirp), freq_bins, eirp_bins)
            true_val = env.reward(SpectrumAction(float(freq_bins[b]), float(eirp) - ANTENNA_GAIN_DBI))
            errors.append(abs(float(q[arm]) - float(true_val)))

    over_cap = [r.proposed.eirp_dbm - MAX_EIRP_DBM for r in results]
    graduated = bool(q_rates[3] <= 0.05 and q_rates[3] <= 0.25 * max(q_rates[0], 1e-9))
    return {
        "closed_loop_summary": summarise(results),
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
        "infeasible_belief": {
            "mean_abs_error_vs_oracle_above_cap": round(float(np.mean(errors)), 6),
            "true_unconstrained_reward_at_eirp_50_dbm": round(float(true_probe), 6),
            "value_estimate_at_eirp_50_dbm": round(float(q[probe_arm]), 6),
        },
        "max_executed_eirp_dbm": round(max(r.executed.eirp_dbm for r in results), 4),
        "max_over_cap_excess_db": round(max(0.0, max(over_cap)), 4),
        "name": name,
        "realised_mean_reward": round(float(np.mean([r.reward for r in results])), 6),
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
    n_infeasible = sum(1 for a in range(n_arms) if _arm_illegal(a, freq_bins, eirp_bins))

    ap = audit_path or Path("benchmarks/results/exploration_truncation_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)

    trained: dict[str, dict[str, Any]] = {}
    all_results: dict[str, list[StepResult]] = {}
    for name, aware in (("reward_only", False), ("correction_aware", True)):
        results, q = _train_learner(
            env, ap, store, name, correction_aware=aware, rounds=rounds
        )
        all_results[name] = results
        trained[name] = _learner_report(name, results, q, env, freq_bins, eirp_bins)

    # --- Counterfactual harm the Shield actually prevented (oracle, report-only).
    every_step = all_results["reward_only"] + all_results["correction_aware"]
    blocked = [r for r in every_step if r.illegal_without_shield]
    over_cap_blocked = [r for r in blocked if r.proposed.eirp_dbm > MAX_EIRP_DBM + 1e-9]
    oob_blocked = [
        r
        for r in blocked
        if not (BAND_LO_HZ <= r.proposed.frequency_hz <= BAND_HI_HZ)
    ]
    counterfactuals = [
        r.counterfactual_reward for r in blocked if r.counterfactual_reward is not None
    ]
    harm = {
        "correction_aware_blocked": trained["correction_aware"]["illegal_proposals_blocked"],
        "illegal_actions_blocked_total": len(blocked),
        "max_counterfactual_reward_of_blocked_proposals": (
            round(float(max(counterfactuals)), 6) if counterfactuals else None
        ),
        "max_over_cap_excess_db": round(
            max((r.proposed.eirp_dbm - MAX_EIRP_DBM for r in over_cap_blocked), default=0.0), 4
        ),
        "out_of_band_proposals_total": len(oob_blocked),
        "over_cap_proposals_total": len(over_cap_blocked),
        "reward_only_blocked": trained["reward_only"]["illegal_proposals_blocked"],
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

    ro, ca = trained["reward_only"], trained["correction_aware"]
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
            "n_infeasible_arms": n_infeasible,
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
        "features_sha256": manifest.get("features_sha256"),
        "learners": {"correction_aware": ca, "reward_only": ro},
        "optimal_feasible_reward": round(float(env.optimal_feasible_reward()), 6),
        "receivers": len(gains),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "scope_note": (
            "Real DeepMIMO ray tracing (not over-the-air capture), one scenario. The "
            "learners are tabular bandits over a discretised action space, not deep RL, "
            "and graduation is shown for this learner and reward shape only (served "
            "fraction, monotone in EIRP, deterministic). Realised utility is nearly "
            "identical for both learners because the Shield rescues every illegal "
            "proposal onto the cap; what differs is who does the safety work. The "
            "counterfactual oracle is used for reporting only and is never trained on."
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
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    result = run(args.features, args.manifest, rounds=args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    ro = result["learners"]["reward_only"]
    ca = result["learners"]["correction_aware"]
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
        for learner in (ro, ca)
    )
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and tc["tampered_chain_refused"] is True
        and tc["verify_after_tamper_index"] not in (None, -1)
    )
    ok = pulled and graduates and fewer_interventions and safety_holds and chain_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
