#!/usr/bin/env python3
"""Does the Shield's projection operator bias what a learner learns? Real-data test.

This drives Horizon's *closed-loop learning + trust core* on real measured-physics
data — no synthetic RF anywhere in the loop:

    real ray-traced channels (DeepMIMO ASU 3.5 GHz, 4096 receivers)
        -> ShieldedSpectrumEnv: reward = served fraction of real receivers
        -> a tabular bandit proposes an EIRP; the Shield *projects* it legally
        -> every decision lands on a hash-chained, tamper-evident evidence store
        -> training tables are rebuilt from the verify-gated replay buffer
        -> four credit-assignment regimes x TWO action-space arms

The scientific question: when a learner proposes action ``a`` but the Shield
executes ``Π(a)`` (EIRP clipped to the 33 dBm legal cap), *whom* should the
observed reward be credited to? Credit regimes, same learner, same environment,
same schedule:

  A. ``naive_proposed_credit``  — credit the observed reward to the PROPOSED
     action (the bug under test). Every over-cap proposal is silently corrected
     to the same executed action, so the learner books ~0.347 as the value of
     EVERY illegal EIRP: censored feedback, systematically wrong value function.
  B. ``projection_aware``       — credit the observed reward to the EXECUTED
     action. The table simply never claims a value for infeasible EIRPs.
  C. ``correction_aware``       — value credit as B, plus a projection penalty
     attached to the PROPOSED bin in the behaviour score.
  D. ``unconstrained_oracle``   — ORACLE / NOT DEPLOYABLE: trained on the
     counterfactual reward the proposal would earn with NO Shield. Its proposals
     are still emitted only through the Shield (nothing illegal ever reaches the
     RAN), but its learning signal requires an oracle no fielded rApp has.

TWO ACTION-SPACE ARMS (this is the P1 correction, see ``ERRATA.md``)
--------------------------------------------------------------------
Every regime is trained twice:

* ``unmasked`` — the learner may propose any bin on the 20..52 dBm grid and must
  rediscover the cap empirically from censored feedback. This is the *only*
  configuration in which the credit-assignment bias is observable at all, so it
  is retained as the measurement arm — not as a recommendation.
* ``shield_masked`` — the action space is masked by the Shield's OWN feasibility
  predicate (``ShieldedSpectrumEnv.feasible_mask_grid`` →
  ``Shield.is_feasible``). Masked bins are never proposed, so they never enter
  the value table. The honest consequence, measured below: **the entire
  credit-assignment pathology disappears.** The 20-way argmax tie collapses to 1,
  illegal proposals go to 0, and regimes A, B and C become behaviourally
  indistinguishable. The pathology this benchmark was built to exhibit is an
  artifact of *withholding from the learner a constraint the Shield knows
  analytically*.

Masking is **defence in depth, never a substitute**: ``env.step`` still disposes
every proposal through the Shield unconditionally, and the guard chain still
refuses any uncorrected emit. A masked run is exactly as safe as an unmasked one,
and both are proven safe here (``max_executed_eirp_dbm = 33.0`` in all arms).

THE ZERO-DATA BASELINE (P3)
---------------------------
Every regime report carries ``constant_cap_policy_reward`` — the realised reward
of proposing a constant 33 dBm forever, with no data and no learning — and
``realised_regret_*`` against it. On this scenario that policy is *exactly*
optimal, so the measured learning gain of the best deployable learner is
0.000000 and every learner is strictly worse than it over the full run. That is
reported, not hidden.

Trust chain: every ``env.step`` writes a DecisionRecord carrying BOTH actions to
a hash-chained JsonlEvidenceStore. After training, the learner tables for A, B
and C are rebuilt exclusively from ``load_transitions`` (the verify-gated replay
buffer) and must match the online tables bit-for-bit; then record #1 is tampered
on disk and ``load_transitions`` must REFUSE the chain (EvidenceIntegrityError).
Regime D is deliberately NOT rebuildable: its oracle signal is not evidence.

Pure numpy + the shipped ``horizon_ric`` modules. The real DeepMIMO feature file
is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` (deterministic, checksum-pinned) or point
``--features`` at a cached copy. The committed result JSON is the real 4096-Rx
run; the ``realdata`` CI workflow rebuilds the data and re-runs this loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.learning import (
    EvidenceIntegrityError,
    ShieldedSpectrumEnv,
    SpectrumAction,
    StepResult,
    Transition,
    load_transitions,
)
from horizon_ric.learning.shield_env import (
    ANTENNA_GAIN_DBI,
    MAX_EIRP_DBM,
    N_SUBBANDS,
    load_gain_matrix,
    subband_center_hz,
)
from horizon_ric.runtime_env import stamp

# --- Learner geometry: a 1-D EIRP grid that deliberately SPANS the legal cap. --
EIRP_MIN_DBM = 20.0
EIRP_MAX_DBM = 52.0
N_BINS = int(EIRP_MAX_DBM - EIRP_MIN_DBM) + 1  # 33 bins, 1 dB apart.
EIRP_BINS_DBM = np.arange(N_BINS, dtype=np.float64) + EIRP_MIN_DBM

ROUNDS = 240  # env.step interactions per regime per arm.
EPS_START = 0.5  # epsilon-greedy exploration after the initial full sweep.
EPS_FLOOR = 0.04
CORRECTION_PENALTY = 0.2  # regime C's behaviour penalty per projected proposal.
SEED_BASE = 2027  # all randomness is np.random.default_rng(SEED_BASE + regime).

# Grid tops used to expose how arbitrary ``value_mae_infeasible`` is (P3c).
GRID_TOP_PROBE_DBM = (34.0, 40.0, 52.0, 80.0, 110.0)

REGIMES = (
    "naive_proposed_credit",
    "projection_aware",
    "correction_aware",
    "unconstrained_oracle",
)

ARMS = ("unmasked", "shield_masked")

CREDIT_RULES = {
    "naive_proposed_credit": (
        "observed (post-Shield) reward credited to the PROPOSED EIRP bin and "
        "claimed as that action's value — the bug under test"
    ),
    "projection_aware": (
        "observed reward credited to the EXECUTED EIRP bin; infeasible bins are "
        "never claimed"
    ),
    "correction_aware": (
        "value credit as projection_aware, plus a -%.2f behaviour penalty on the "
        "PROPOSED bin whenever the Shield had to project it" % CORRECTION_PENALTY
    ),
    "unconstrained_oracle": (
        "ORACLE, NOT DEPLOYABLE: counterfactual no-Shield reward credited to the "
        "PROPOSED bin (requires an oracle no fielded rApp has)"
    ),
}

ARM_NOTES = {
    "unmasked": (
        "the learner may propose any bin on the 20..52 dBm grid and must "
        "rediscover the EIRP cap empirically from censored feedback; retained "
        "because it is the only arm in which the credit-assignment bias is "
        "observable, NOT as a recommendation"
    ),
    "shield_masked": (
        "P1: the action space is masked by the Shield's own feasibility "
        "predicate (ShieldedSpectrumEnv.feasible_mask_grid -> "
        "Shield.is_feasible); masked bins are never proposed and never enter "
        "the value table. Defence in depth only — env.step still projects "
        "every proposal unconditionally"
    ),
}


def _bin_of(eirp_dbm: float) -> int:
    b = int(round(eirp_dbm - EIRP_MIN_DBM))
    if not 0 <= b < N_BINS:
        raise ValueError(f"EIRP {eirp_dbm} dBm outside the learner grid")
    return b


def _epsilon(t: int, rounds: int) -> float:
    """Deterministic linear decay from EPS_START to EPS_FLOOR by mid-training."""
    span = max(1, (rounds - N_BINS) // 2)
    return max(EPS_FLOOR, EPS_START * (1.0 - (t - N_BINS) / span))


def _r(value: float | None, digits: int = 6) -> float | None:
    """Round for JSON, mapping NaN/None to ``null`` rather than a fake 0.0."""
    if value is None:
        return None
    v = float(value)
    return None if math.isnan(v) else round(v, digits)


class _TabularRegime:
    """One credit regime's tables. Incremental exact means; no learning rate.

    ``proposable`` is the arm's action mask over the EIRP grid. In the
    ``shield_masked`` arm it is the Shield's own feasibility mask, so an
    infeasible bin can never be selected and therefore can never be claimed.
    """

    def __init__(
        self,
        name: str,
        rng: np.random.Generator,
        *,
        proposable: np.ndarray,
        correction_penalty: float = CORRECTION_PENALTY,
    ) -> None:
        self.name = name
        self.rng = rng
        self.proposable = np.asarray(proposable, dtype=bool)
        self.proposable_bins = np.flatnonzero(self.proposable)
        if self.proposable_bins.size == 0:  # pragma: no cover - defensive
            raise ValueError("the action mask left no proposable bin")
        self.correction_penalty = float(correction_penalty)
        # The CLAIMED value table (NaN = the regime makes no claim for that bin).
        # A and D index it by PROPOSED bin; B and C index it by EXECUTED bin.
        self.q = np.full(N_BINS, np.nan)
        self.visits = np.zeros(N_BINS, dtype=np.int64)
        # Regime C's behaviour tables over PROPOSED bins (not value claims).
        self.r_hat = np.full(N_BINS, np.nan)
        self.p_hat = np.full(N_BINS, np.nan)
        self.b_visits = np.zeros(N_BINS, dtype=np.int64)

    @staticmethod
    def _mean_update(table: np.ndarray, counts: np.ndarray, b: int, x: float) -> None:
        counts[b] += 1
        table[b] = x if counts[b] == 1 else table[b] + (x - table[b]) / counts[b]

    def update(self, proposed_bin: int, res: StepResult) -> None:
        executed_bin = _bin_of(res.executed.eirp_dbm)
        if self.name == "naive_proposed_credit":
            self._mean_update(self.q, self.visits, proposed_bin, res.reward)
        elif self.name == "projection_aware":
            self._mean_update(self.q, self.visits, executed_bin, res.reward)
        elif self.name == "correction_aware":
            self._mean_update(self.q, self.visits, executed_bin, res.reward)
            self._mean_update(self.r_hat, self.b_visits, proposed_bin, res.reward)
            # b_visits was just bumped by the r_hat update; keep p_hat in step.
            n = self.b_visits[proposed_bin]
            x = float(res.projected)
            self.p_hat[proposed_bin] = (
                x if n == 1 else self.p_hat[proposed_bin] + (x - self.p_hat[proposed_bin]) / n
            )
        elif self.name == "unconstrained_oracle":
            if res.counterfactual_reward is None:  # pragma: no cover - always in band
                raise RuntimeError("oracle regime proposed an out-of-band action")
            self._mean_update(self.q, self.visits, proposed_bin, res.counterfactual_reward)
        else:  # pragma: no cover
            raise ValueError(self.name)

    def _score(self) -> np.ndarray:
        if self.name == "correction_aware":
            return self.r_hat - self.correction_penalty * self.p_hat
        return self.q

    def _tied(self) -> np.ndarray:
        """Bins attaining the max of the behaviour score, restricted to the arm."""
        score = self._score()
        known = ~np.isnan(score) & self.proposable
        if not np.any(known):  # pragma: no cover - the sweep visits every bin first
            return self.proposable_bins
        best = np.max(score[known])
        return np.flatnonzero(known & (score == best))

    def greedy_bin(self) -> int:
        """Behavioural argmax with a UNIFORM, seeded, exact-tie randomisation.

        Every regime now breaks exact ties the same way. The previous code broke
        regime A's ties by seeded random choice but every other regime's by
        ``tied[0]`` (lowest EIRP — a silently safety-favouring rule). That
        asymmetry, not the correction penalty, was doing regime C's work: at
        CORRECTION_PENALTY = 0 regime C's behaviour score is flat across the
        legal cap and all 19 illegal bins, and ``tied[0]`` picked the cap anyway.
        With the rule made uniform the penalty is genuinely load-bearing, which
        ``correction_penalty_probe`` in the result JSON measures directly.

        The rng is consumed only on a genuine (>1) tie, so regimes that never tie
        keep the exact stream they had before the rule was uniformised.
        """
        tied = self._tied()
        if tied.size == 1:
            return int(tied[0])
        return int(self.rng.choice(tied))

    def greedy_tie_count(self) -> int:
        return int(self._tied().size)

    def greedy_bin_deterministic(self) -> int:
        """First tied bin — used for REPORTING only, so the committed JSON and
        cross-host verification never depend on the tie-break rng."""
        return int(self._tied()[0])


def _train_regime(
    env: ShieldedSpectrumEnv,
    store: JsonlEvidenceStore | None,
    name: str,
    frequency_hz: float,
    rounds: int,
    *,
    arm: str,
    proposable: np.ndarray,
    correction_penalty: float = CORRECTION_PENALTY,
) -> tuple[_TabularRegime, list[StepResult], list[int]]:
    regime = _TabularRegime(
        name,
        np.random.default_rng(SEED_BASE + REGIMES.index(name)),
        proposable=proposable,
        correction_penalty=correction_penalty,
    )
    bins = regime.proposable_bins
    steps: list[StepResult] = []
    proposals: list[int] = []
    for t in range(rounds):
        if t < N_BINS:
            # Deterministic opening sweep. In the unmasked arm this is exactly
            # ``t % N_BINS`` (every bin once); in a masked arm it cycles the
            # proposable bins while keeping the rng stream and the epsilon
            # schedule identical across arms.
            b = int(bins[t % bins.size])
        elif regime.rng.random() < _epsilon(t, rounds):
            b = int(bins[regime.rng.integers(bins.size)])
        else:
            b = regime.greedy_bin()
        proposed = SpectrumAction(frequency_hz, EIRP_MIN_DBM + b - ANTENNA_GAIN_DBI)
        res = env.step(
            proposed,
            decision_id=f"credit-{arm}-{name}-{t:04d}",
            store=store,
            rapp_instance_id="credit-assignment",
            policy_label=f"{arm}:{name}",
        )
        regime.update(b, res)
        steps.append(res)
        proposals.append(b)
    return regime, steps, proposals


def _rebuild_from_chain(
    name: str, transitions: list[Transition], *, proposable: np.ndarray
) -> _TabularRegime:
    """Rebuild a Shield-constrained regime's tables ONLY from verified evidence.

    This is the provenance path a fielded learner must use: the replay buffer is
    ``load_transitions`` (verify-gated), and each Transition carries BOTH actions
    plus the projection flags, so every credit rule below is reconstructible.
    The oracle regime is deliberately absent: its counterfactual signal is not
    recorded evidence and cannot be recovered from the chain.
    """
    regime = _TabularRegime(
        name, np.random.default_rng(0), proposable=proposable
    )  # rng unused in replay.
    for tr in transitions:
        pb = _bin_of(tr.proposed_eirp_dbm)
        eb = _bin_of(tr.executed_eirp_dbm)
        if name == "naive_proposed_credit":
            regime._mean_update(regime.q, regime.visits, pb, tr.reward)
        elif name == "projection_aware":
            regime._mean_update(regime.q, regime.visits, eb, tr.reward)
        elif name == "correction_aware":
            regime._mean_update(regime.q, regime.visits, eb, tr.reward)
            regime._mean_update(regime.r_hat, regime.b_visits, pb, tr.reward)
            n = regime.b_visits[pb]
            x = float(tr.projected)
            regime.p_hat[pb] = x if n == 1 else regime.p_hat[pb] + (x - regime.p_hat[pb]) / n
        else:  # pragma: no cover
            raise ValueError(f"regime {name} is not rebuildable from the chain")
    return regime


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def _mae_over(claims: np.ndarray, truth: np.ndarray, bins: np.ndarray) -> tuple[float, int]:
    """Mean |claim − truth| over the bins the regime actually CLAIMS.

    Returns ``NaN`` (not 0.0) when the claimed set is empty. A learner that
    claims nothing has not scored perfectly; it has abstained, and the metric is
    undefined. The previous 0.0 made every accuracy gate satisfiable by
    abstention — see ``ERRATA.md`` P3(a).
    """
    claimed = bins[~np.isnan(claims[bins])]
    if len(claimed) == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(claims[claimed] - truth[claimed]))), int(len(claimed))


def _regime_report(
    name: str,
    regime: _TabularRegime,
    steps: list[StepResult],
    proposals: list[int],
    *,
    truth: np.ndarray,
    truth_projected: np.ndarray,
    shield_feasible: np.ndarray,
    rounds: int,
    optimal_feasible: float,
    constant_cap_reward: float,
) -> dict[str, Any]:
    feasible = np.flatnonzero(shield_feasible)
    infeasible = np.flatnonzero(~shield_feasible)
    mae_inf, claimed_inf = _mae_over(regime.q, truth, infeasible)
    mae_inf_proj, _ = _mae_over(regime.q, truth_projected, infeasible)
    mae_feas, claimed_feas = _mae_over(regime.q, truth, feasible)

    q = max(1, rounds // 4)
    first, last = steps[:q], steps[rounds - q:]
    oracle = name == "unconstrained_oracle"
    # A/B/C realise the executed (post-Shield) reward; the oracle regime's world
    # has no Shield, so its realisation is the unconstrained counterfactual.
    realised = [(s.counterfactual_reward if oracle else s.reward) for s in steps]
    realised_last = realised[rounds - q:]
    realised_mean = float(np.mean(realised))
    realised_last_mean = float(np.mean(realised_last))

    # Illegality is judged by the SHIELD'S OWN predicate over the proposed bin,
    # not by a benchmark-private restatement of the constraint (ERRATA P1). At
    # this loop's fixed interior subband the substrate's StepResult flag happens
    # to agree; it is reported alongside so the agreement is auditable rather
    # than assumed.
    illegal_idx = [i for i, b in enumerate(proposals) if not shield_feasible[b]]
    illegal = [steps[i] for i in illegal_idx]
    substrate_illegal = [s for s in steps if s.illegal_without_shield]

    report: dict[str, Any] = {
        "credit_rule": CREDIT_RULES[name],
        "oracle": oracle,
        "deployable": not oracle,
        # --- value-function accuracy (P3a/P3c) ------------------------------
        "value_mae_infeasible": _r(mae_inf),
        "value_mae_infeasible_vs_projected_truth": _r(mae_inf_proj),
        "value_mae_feasible": _r(mae_feas),
        "infeasible_bins_claimed": claimed_inf,
        "feasible_bins_claimed": claimed_feas,
        "abstained_on_infeasible_bins": claimed_inf == 0,
        # --- behaviour -------------------------------------------------------
        "argmax_proposed_eirp_dbm": float(EIRP_MIN_DBM + regime.greedy_bin_deterministic()),
        "argmax_tie_count": regime.greedy_tie_count(),
        "proposable_bins": int(regime.proposable_bins.size),
        "projection_rate_first_quarter": round(float(np.mean([s.projected for s in first])), 6),
        "projection_rate_last_quarter": round(float(np.mean([s.projected for s in last])), 6),
        # --- realised utility and the zero-data baseline (P3b) ---------------
        "realised_mean_reward": round(realised_mean, 6),
        "realised_mean_reward_last_quarter": round(realised_last_mean, 6),
        "constant_cap_policy_reward": round(float(constant_cap_reward), 6),
        "realised_regret_last_quarter": round(float(optimal_feasible - realised_last_mean), 6),
        "realised_regret_all": round(float(optimal_feasible - realised_mean), 6),
        "learning_gain_over_constant_cap_policy_last_quarter": round(
            float(realised_last_mean - constant_cap_reward), 6
        ),
        "learning_gain_over_constant_cap_policy_all": round(
            float(realised_mean - constant_cap_reward), 6
        ),
        # --- legality, judged by the Shield ----------------------------------
        "illegal_proposals": len(illegal),
        "illegal_proposals_substrate_predicate": len(substrate_illegal),
        "illegal_predicate_agrees_with_substrate": len(illegal) == len(substrate_illegal),
        "illegal_mean_excess_eirp_db": (
            _r(float(np.mean([s.proposed.eirp_dbm - MAX_EIRP_DBM for s in illegal])))
            if illegal
            else None
        ),
        "illegal_mean_counterfactual_reward": (
            _r(float(np.mean([s.counterfactual_reward for s in illegal]))) if illegal else None
        ),
        "max_executed_eirp_dbm": round(float(max(s.executed.eirp_dbm for s in steps)), 6),
        "executed_out_of_band_steps": int(
            sum(
                1
                for s in steps
                if s.executed.frequency_hz < 3.45e9 or s.executed.frequency_hz > 3.55e9
            )
        ),
        "learned_value_by_eirp": {
            str(int(EIRP_MIN_DBM + b)): _r(regime.q[b]) for b in range(N_BINS)
        },
    }
    if oracle:
        report["shielded_execution_mean_reward"] = round(
            float(np.mean([s.reward for s in steps])), 6
        )
        report["reward_stream"] = "unconstrained counterfactual (ORACLE, not deployable)"
    else:
        report["reward_stream"] = "executed (post-Shield) reward — the only realised one"
    return report


def _traj_digest(proposals: list[int]) -> str:
    """Stable digest of a proposal sequence (``hash()`` is not portable)."""
    payload = ",".join(str(b) for b in proposals).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _penalty_probe(
    env: ShieldedSpectrumEnv,
    frequency_hz: float,
    rounds: int,
    *,
    arm: str,
    proposable: np.ndarray,
    shield_feasible: np.ndarray,
) -> dict[str, Any]:
    """Does CORRECTION_PENALTY change anything? Same rng, penalty on vs off.

    This is the P2 gate. It must key on the PROPOSAL TRAJECTORY and a realised
    scalar, never on ``argmax_tie_count`` — the tie count moves with the penalty
    even when the trajectory does not, so a gate written against it would pass
    while proving nothing (``ERRATA.md`` P2).
    """
    off = _train_regime(
        env, None, "correction_aware", frequency_hz, rounds,
        arm=arm, proposable=proposable, correction_penalty=0.0,
    )
    on = _train_regime(
        env, None, "correction_aware", frequency_hz, rounds,
        arm=arm, proposable=proposable, correction_penalty=CORRECTION_PENALTY,
    )
    q = max(1, rounds // 4)

    def _stats(trained: tuple[_TabularRegime, list[StepResult], list[int]]) -> dict[str, Any]:
        regime, steps, proposals = trained
        return {
            "proposal_trajectory_sha256": _traj_digest(proposals),
            "projected_steps": int(sum(s.projected for s in steps)),
            "projection_rate_last_quarter": round(
                float(np.mean([s.projected for s in steps[rounds - q:]])), 6
            ),
            "illegal_proposals": int(sum(1 for b in proposals if not shield_feasible[b])),
            "realised_mean_reward": round(float(np.mean([s.reward for s in steps])), 6),
            "argmax_tie_count": regime.greedy_tie_count(),
        }

    off_stats, on_stats = _stats(off), _stats(on)
    traj_differs = off[2] != on[2]
    return {
        "penalty_off": off_stats,
        "penalty_on": on_stats,
        "correction_penalty": CORRECTION_PENALTY,
        "proposal_trajectory_differs": bool(traj_differs),
        "realised_scalar_differs": bool(
            off_stats["projection_rate_last_quarter"]
            != on_stats["projection_rate_last_quarter"]
            or off_stats["projected_steps"] != on_stats["projected_steps"]
        ),
        "penalty_bites": bool(
            traj_differs and off_stats["projected_steps"] != on_stats["projected_steps"]
        ),
        "note": (
            "argmax_tie_count is deliberately reported but NOT used as the "
            "criterion: it moves with the penalty even when nothing else does."
        ),
    }


def run(
    features: Path,
    manifest_path: Path,
    *,
    audit_path: Path | None = None,
    sample_path: Path | None = None,
    rounds: int = ROUNDS,
) -> dict[str, Any]:
    gains = load_gain_matrix(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and gains.shape[0] != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")
    if rounds < 4 * N_BINS:
        raise ValueError(f"rounds must be >= {4 * N_BINS} so every bin is swept per quarter math")

    env = ShieldedSpectrumEnv(gains)
    # The learner works one in-band subband (the best feasible one) so the EIRP
    # axis — the scientific axis — is isolated. Subband gains differ by <0.1 dB.
    cap_rewards = [
        env.reward(SpectrumAction(subband_center_hz(b), MAX_EIRP_DBM - ANTENNA_GAIN_DBI))
        for b in range(N_SUBBANDS)
    ]
    best_subband = int(np.argmax(cap_rewards))
    frequency_hz = subband_center_hz(best_subband)
    optimal_feasible = env.optimal_feasible_reward()

    # --- P1: the feasibility mask comes from the SHIELD, not from a constant. --
    shield_feasible = env.feasible_mask_grid([frequency_hz], EIRP_BINS_DBM)[0]
    n_feasible = int(shield_feasible.sum())
    analytic_cap_mask = EIRP_BINS_DBM <= MAX_EIRP_DBM
    mask_matches_analytic_cap = bool(np.array_equal(shield_feasible, analytic_cap_mask))
    mask_agrees_with_projection = bool(
        np.array_equal(
            shield_feasible,
            np.asarray(
                [
                    not env.step(
                        SpectrumAction(frequency_hz, float(e) - ANTENNA_GAIN_DBI),
                        decision_id=f"mask-probe-{i:04d}",
                    ).projected
                    for i, e in enumerate(EIRP_BINS_DBM)
                ]
            ),
        )
    )
    arm_masks = {"unmasked": np.ones(N_BINS, dtype=bool), "shield_masked": shield_feasible}

    # True UNCONSTRAINED value of every proposed EIRP bin: the real reward the
    # proposal itself would earn with no Shield. Plus the REALISABLE truth
    # R∘Π — what the proposal actually causes once the Shield has projected it.
    # Reporting value error against both is P3(c): against R∘Π the naive
    # regime's infeasible-bin values are exactly right, which is why the
    # published "bias" number is a statement about the measuring stick.
    truth = np.array(
        [env.reward(SpectrumAction(frequency_hz, float(e) - ANTENNA_GAIN_DBI)) for e in EIRP_BINS_DBM]
    )
    truth_projected = np.array(
        [
            env.reward(
                SpectrumAction(
                    frequency_hz, min(float(e), MAX_EIRP_DBM) - ANTENNA_GAIN_DBI
                )
            )
            for e in EIRP_BINS_DBM
        ]
    )

    # --- P3(b): the zero-data baseline, driven through the real Shield. -------
    constant_cap_steps = [
        env.step(
            SpectrumAction(frequency_hz, MAX_EIRP_DBM - ANTENNA_GAIN_DBI),
            decision_id=f"constant-cap-{t:04d}",
        )
        for t in range(rounds)
    ]
    constant_cap_reward = float(np.mean([s.reward for s in constant_cap_steps]))

    # --- P3(c): how arbitrary is value_mae_infeasible? ------------------------
    r_at_cap = float(truth[_bin_of(MAX_EIRP_DBM)])
    grid_top_sensitivity = {}
    for top in GRID_TOP_PROBE_DBM:
        above = np.arange(MAX_EIRP_DBM + 1.0, top + 1.0)
        vals = [
            env.reward(SpectrumAction(frequency_hz, float(e) - ANTENNA_GAIN_DBI)) for e in above
        ]
        grid_top_sensitivity[str(int(top))] = {
            "infeasible_bins": int(above.size),
            "value_mae_infeasible_if_naive": round(
                float(np.mean([abs(r_at_cap - v) for v in vals])), 6
            ),
        }

    ap = audit_path or Path("benchmarks/results/credit_assignment_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)

    trained: dict[str, dict[str, tuple[_TabularRegime, list[StepResult], list[int]]]] = {}
    for arm in ARMS:
        trained[arm] = {
            name: _train_regime(
                env, store, name, frequency_hz, rounds,
                arm=arm, proposable=arm_masks[arm],
            )
            for name in REGIMES
        }

    chain_len = len(store)
    verify_intact = store.verify()

    # --- Verify-gated replay: rebuild A/B/C training tables from the chain. ---
    transitions = load_transitions(ap)
    replay_matches = True
    for arm in ARMS:
        for name in ("naive_proposed_credit", "projection_aware", "correction_aware"):
            subset = [
                t for t in transitions if t.decision_id.startswith(f"credit-{arm}-{name}-")
            ]
            online = trained[arm][name][0]
            rebuilt = _rebuild_from_chain(name, subset, proposable=arm_masks[arm])
            ok = _same(online.q, rebuilt.q)
            if name == "correction_aware":
                ok = ok and _same(online.r_hat, rebuilt.r_hat) and _same(online.p_hat, rebuilt.p_hat)
            replay_matches = replay_matches and ok

    # --- Intact record 0, preserved before the tamper proof below. ------------
    first_line = ap.read_text(encoding="utf-8").splitlines()[0]
    if sample_path is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            json.dumps(json.loads(first_line), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    arms_report = {
        arm: {
            "action_space": ARM_NOTES[arm],
            "proposable_bins": int(arm_masks[arm].sum()),
            "regimes": {
                name: _regime_report(
                    name, regime, steps, proposals,
                    truth=truth,
                    truth_projected=truth_projected,
                    shield_feasible=shield_feasible,
                    rounds=rounds,
                    optimal_feasible=optimal_feasible,
                    constant_cap_reward=constant_cap_reward,
                )
                for name, (regime, steps, proposals) in trained[arm].items()
            },
        }
        for arm in ARMS
    }

    # --- P2: does the correction penalty do anything, in either arm? ----------
    penalty_probe = {
        arm: _penalty_probe(
            env,
            frequency_hz,
            rounds,
            arm=f"probe-{arm}",
            proposable=arm_masks[arm],
            shield_feasible=shield_feasible,
        )
        for arm in ARMS
    }

    unmasked = arms_report["unmasked"]["regimes"]
    masked = arms_report["shield_masked"]["regimes"]
    oracle_last_q = unmasked["unconstrained_oracle"]["realised_mean_reward_last_quarter"]
    safety_utility_gap = round(float(oracle_last_q - optimal_feasible), 6)

    # Unrounded realised means, so the headline learning gain is exact rather
    # than a rounding artifact of the reported 6-dp values.
    qn = max(1, rounds // 4)
    raw_realised: dict[tuple[str, str], tuple[float, float]] = {}
    for arm in ARMS:
        for name, (_reg, steps, _prop) in trained[arm].items():
            stream = [
                (s.counterfactual_reward if name == "unconstrained_oracle" else s.reward)
                for s in steps
            ]
            raw_realised[(arm, name)] = (
                float(np.mean(stream[rounds - qn:])),
                float(np.mean(stream)),
            )

    deployable = [n for n in REGIMES if n != "unconstrained_oracle"]
    best_deployable_q4 = max(raw_realised[(a, n)][0] for a in ARMS for n in deployable)
    best_deployable_all = max(raw_realised[(a, n)][1] for a in ARMS for n in deployable)
    masking_regret_all = {
        arm: round(
            float(np.mean([optimal_feasible - raw_realised[(arm, n)][1] for n in deployable])), 6
        )
        for arm in ARMS
    }

    result: dict[str, Any] = {
        "benchmark": (
            "Credit-assignment bias of the Shield projection operator on real DeepMIMO channels"
        ),
        "task": "shielded spectrum/EIRP selection for an O-RAN DSA rApp",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get("data_kind", "site-specific Wireless InSite ray tracing"),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": int(gains.shape[0]),
        "env": {
            "noise_dbm": round(float(env.noise_dbm), 4),
            "eirp_cap_dbm": MAX_EIRP_DBM,
            "bandwidth_hz": env.bandwidth_hz,
            "sinr_threshold_db": env.sinr_threshold_db,
            "reward": "served fraction (receivers whose real-channel SINR clears the threshold)",
            "band_lo_hz": env.band_lo_hz,
            "band_hi_hz": env.band_hi_hz,
            "learner_subband": best_subband,
            "learner_frequency_hz": frequency_hz,
            "optimal_feasible_reward": round(float(optimal_feasible), 6),
        },
        "learner": {
            "type": "tabular epsilon-greedy bandit over proposed EIRP (1 dB bins)",
            "eirp_grid_dbm": [EIRP_MIN_DBM, EIRP_MAX_DBM],
            "bins": N_BINS,
            "feasible_bins": n_feasible,
            "infeasible_bins": N_BINS - n_feasible,
            "rounds_per_regime": rounds,
            "sweep_steps": N_BINS,
            "epsilon_start": EPS_START,
            "epsilon_floor": EPS_FLOOR,
            "correction_penalty": CORRECTION_PENALTY,
            "seed_base": SEED_BASE,
            "tie_break_rule": (
                "uniform across all regimes: seeded random choice among exactly "
                "tied bins (rng consumed only when the tie is genuine); reporting "
                "uses the lowest tied bin so committed numbers never depend on it"
            ),
        },
        # --- P1 -------------------------------------------------------------
        "action_space_masking": {
            "mask_source": (
                "ShieldedSpectrumEnv.feasible_mask_grid -> Shield.is_feasible "
                "(the Shield's own analytic predicate, not a benchmark restatement)"
            ),
            "feasible_bins": n_feasible,
            "infeasible_bins": N_BINS - n_feasible,
            "mask_matches_analytic_eirp_cap": mask_matches_analytic_cap,
            "mask_agrees_with_projection_on_every_bin": mask_agrees_with_projection,
            "defence_in_depth_note": (
                "env.step disposes every proposal through the Shield regardless "
                "of the mask; masking removes the learner's need to rediscover "
                "the constraint, it does not replace the guarantee"
            ),
            "argmax_tie_count_before_after": {
                name: {
                    "unmasked": unmasked[name]["argmax_tie_count"],
                    "shield_masked": masked[name]["argmax_tie_count"],
                }
                for name in REGIMES
            },
            "illegal_proposals_before_after": {
                name: {
                    "unmasked": unmasked[name]["illegal_proposals"],
                    "shield_masked": masked[name]["illegal_proposals"],
                }
                for name in REGIMES
            },
            "infeasible_bins_claimed_before_after": {
                name: {
                    "unmasked": unmasked[name]["infeasible_bins_claimed"],
                    "shield_masked": masked[name]["infeasible_bins_claimed"],
                }
                for name in REGIMES
            },
            "mean_realised_regret_over_deployable_regimes": masking_regret_all,
            "exploration_cost_of_masking": round(
                float(masking_regret_all["shield_masked"] - masking_regret_all["unmasked"]), 6
            ),
            "exploration_cost_note": (
                "masking makes realised regret WORSE on this scenario, and the "
                "reason is a harness artifact worth stating: unmasked, %d of the "
                "%d grid bins are above the cap and all of them are clipped onto "
                "the single optimal executed action, so uniform exploration lands "
                "on the optimum with probability %.3f. Masking removes that "
                "accidental exploration bonus (probability drops to %.3f), so the "
                "learner pays the normal price of exploring. The projection "
                "operator was silently subsidising the unmasked arm's realised "
                "reward."
                % (
                    N_BINS - n_feasible + 1,
                    N_BINS,
                    (N_BINS - n_feasible + 1) / N_BINS,
                    1.0 / n_feasible,
                )
            ),
            "finding": (
                "masking the action space with the Shield's own predicate removes "
                "the credit-assignment pathology entirely: the naive regime's "
                "20-way argmax tie collapses to 1, every regime's illegal "
                "proposals go to 0, and no regime claims a value for an "
                "infeasible bin. The pathology this benchmark exhibits is an "
                "artifact of withholding a constraint the Shield knows analytically."
            ),
        },
        # --- P2 -------------------------------------------------------------
        "correction_penalty_probe": penalty_probe,
        # --- P3 -------------------------------------------------------------
        "zero_data_baseline": {
            "policy": (
                "propose a constant %.1f dBm EIRP at the learner subband every "
                "step; no data, no learning, no exploration" % MAX_EIRP_DBM
            ),
            "realised_mean_reward": round(constant_cap_reward, 6),
            "optimal_feasible_reward": round(float(optimal_feasible), 6),
            "equals_optimal_feasible": bool(
                abs(constant_cap_reward - optimal_feasible) < 1e-12
            ),
            "steps": len(constant_cap_steps),
            "projected_steps": int(sum(s.projected for s in constant_cap_steps)),
            "best_deployable_learner_realised_last_quarter": round(best_deployable_q4, 6),
            "best_deployable_learner_realised_all": round(best_deployable_all, 6),
            "measured_learning_gain_last_quarter": round(
                float(best_deployable_q4 - constant_cap_reward), 6
            ),
            "measured_learning_gain_all": round(
                float(best_deployable_all - constant_cap_reward), 6
            ),
            "finding": (
                "the measured learning gain of the best deployable learner over a "
                "zero-data constant-cap policy is 0.000000 on the last quarter, "
                "and NEGATIVE over the full run (every learner pays for "
                "exploration). Nothing in this benchmark demonstrates that "
                "learning beats not learning on this scenario."
            ),
        },
        "value_mae_infeasible_sensitivity": {
            "definition": (
                "value_mae_infeasible for the naive regime is exactly "
                "mean_{b>cap} |R(cap) - R(b)|: a pure function of where the top "
                "of the EIRP grid is placed, with no effect on realised reward"
            ),
            "grid_top_dbm": {k: v for k, v in grid_top_sensitivity.items()},
            "published_grid_top_dbm": EIRP_MAX_DBM,
            "measuring_stick_note": (
                "value_mae_infeasible scores claims against R, the unconstrained "
                "reward of the PROPOSED action — a counterfactual no deployed "
                "rApp can observe. value_mae_infeasible_vs_projected_truth scores "
                "them against R∘Π, what the proposal actually causes. Against "
                "R∘Π the naive regime is exactly right and the oracle is the one "
                "in error. Both are reported; neither alone is the story."
            ),
        },
        "true_unconstrained_value_by_eirp": {
            str(int(e)): round(float(truth[i]), 6) for i, e in enumerate(EIRP_BINS_DBM)
        },
        "realisable_value_by_eirp_after_projection": {
            str(int(e)): round(float(truth_projected[i]), 6) for i, e in enumerate(EIRP_BINS_DBM)
        },
        "arms": arms_report,
        # Back-compat alias: the unmasked arm is the measurement arm.
        "regimes": unmasked,
        "safety_utility_gap": safety_utility_gap,
        "trust_chain": {
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
            "replay_transitions": len(transitions),
            "replay_matches_online": bool(replay_matches),
        },
        "scope_note": (
            "Real DeepMIMO ray tracing supplies every reward; the Shield, guard chain and "
            "evidence chain are the shipped production code paths. This is site-specific ray "
            "tracing, not over-the-air capture, and one scenario. The learner is a tabular "
            "bandit over proposed EIRP at one in-band subband, not a deep RL agent. The "
            "unconstrained_oracle regime is a measuring stick only — it needs a no-Shield "
            "counterfactual no deployed rApp has, and is NOT deployable. The credit-assignment "
            "pathology is measurable ONLY in the unmasked arm; masking the action space with "
            "the Shield's own feasibility predicate removes it, and on this scenario no "
            "deployable learner beats a zero-data constant-cap policy. The safety guarantee is "
            "the claim that survives: no illegal action was executed in any arm or regime."
        ),
    }

    # --- Tamper-evidence proof: a tampered chain cannot become training data. --
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace('"reward": ', '"reward": 0.999999, "_tampered": ', 1)
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tampered_refused = False
        try:
            load_transitions(ap)
        except EvidenceIntegrityError:
            tampered_refused = True
        result["trust_chain"]["tampered_chain_refused"] = tampered_refused
        result["trust_chain"]["verify_after_tamper_index"] = JsonlEvidenceStore(ap).verify()

    return result


def gate(result: dict[str, Any]) -> list[str]:
    """Return the list of failed gate names (empty == pass).

    Split out of :func:`main` so ``tests/test_credit_assignment_evidence.py`` can
    prove each gate FAILS on a deliberately corrupted result.
    """
    failures: list[str] = []
    arms = result["arms"]
    unmasked = arms["unmasked"]["regimes"]
    masked = arms["shield_masked"]["regimes"]
    tc = result["trust_chain"]

    def req(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    # --- The Shield genuinely binds. ---------------------------------------
    req("shield_binds", result["safety_utility_gap"] > 0.0)

    # --- P1: the mask is the Shield's own predicate and it removes the
    #     pathology, without weakening the guarantee. -------------------------
    am = result["action_space_masking"]
    req("mask_agrees_with_projection", am["mask_agrees_with_projection_on_every_bin"] is True)
    req("mask_matches_analytic_cap", am["mask_matches_analytic_eirp_cap"] is True)
    req(
        "mask_collapses_the_tie",
        unmasked["naive_proposed_credit"]["argmax_tie_count"] > 1
        and all(masked[n]["argmax_tie_count"] == 1 for n in REGIMES),
    )
    req(
        "mask_eliminates_illegal_proposals",
        all(masked[n]["illegal_proposals"] == 0 for n in REGIMES)
        and unmasked["naive_proposed_credit"]["illegal_proposals"] > 0,
    )
    req(
        "mask_eliminates_infeasible_claims",
        all(masked[n]["infeasible_bins_claimed"] == 0 for n in REGIMES),
    )

    # --- P3(a): the accuracy claim must not be satisfiable by abstention. ----
    naive = unmasked["naive_proposed_credit"]
    proj = unmasked["projection_aware"]
    req(
        "bias_is_a_real_claim_not_abstention",
        naive["infeasible_bins_claimed"] == result["learner"]["infeasible_bins"]
        and naive["value_mae_infeasible"] is not None
        and naive["value_mae_infeasible"] > 0.1,
    )
    req(
        "abstention_is_reported_as_abstention",
        proj["infeasible_bins_claimed"] == 0
        and proj["value_mae_infeasible"] is None
        and proj["abstained_on_infeasible_bins"] is True,
    )

    # --- P3(b): the zero-data baseline is emitted and the gain is honest. ----
    zdb = result["zero_data_baseline"]
    req("zero_data_baseline_emitted", zdb["equals_optimal_feasible"] is True)
    req(
        "regret_emitted_for_every_regime",
        all(
            arms[a]["regimes"][n]["constant_cap_policy_reward"] is not None
            and arms[a]["regimes"][n]["realised_regret_last_quarter"] is not None
            for a in ARMS
            for n in REGIMES
        ),
    )
    req(
        "learning_gain_is_not_overclaimed",
        zdb["measured_learning_gain_last_quarter"] <= 1e-9,
    )

    # --- P2: the correction penalty must actually change behaviour. ---------
    probe = result["correction_penalty_probe"]
    req("correction_penalty_bites_unmasked", probe["unmasked"]["penalty_bites"] is True)
    req(
        "correction_penalty_is_a_no_op_under_masking",
        probe["shield_masked"]["penalty_bites"] is False,
    )

    # --- The hard guarantee, in every arm and regime. -----------------------
    req(
        "nothing_illegal_executed",
        all(
            arms[a]["regimes"][n]["max_executed_eirp_dbm"] <= MAX_EIRP_DBM + 1e-9
            and arms[a]["regimes"][n]["executed_out_of_band_steps"] == 0
            for a in ARMS
            for n in REGIMES
        ),
    )

    # --- Trust chain. -------------------------------------------------------
    req(
        "chain_verifies_replays_and_catches_tamper",
        tc["verify_first_broken_index"] == -1
        and tc.get("tampered_chain_refused") is True
        and tc.get("verify_after_tamper_index") not in (None, -1)
        and tc["replay_matches_online"] is True,
    )
    return failures


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
        "--out", type=Path, default=Path("benchmarks/results/credit_assignment.json")
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    result = run(
        args.features,
        args.manifest,
        sample_path=Path("benchmarks/results/credit_assignment_sample_record.json"),
        rounds=args.rounds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    failures = gate(result)
    if failures:
        print("FAILED GATES: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
