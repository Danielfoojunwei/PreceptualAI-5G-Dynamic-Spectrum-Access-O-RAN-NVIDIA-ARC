#!/usr/bin/env python3
"""Does the Shield's projection operator bias what a learner learns? Real-data test.

This drives Horizon's *closed-loop learning + trust core* on real measured-physics
data — no synthetic RF anywhere in the loop:

    real ray-traced channels (DeepMIMO ASU 3.5 GHz, 4096 receivers)
        -> ShieldedSpectrumEnv: reward = served fraction of real receivers
        -> a tabular bandit proposes an EIRP; the Shield *projects* it legally
        -> every decision lands on a hash-chained, tamper-evident evidence store
        -> training tables are rebuilt from the verify-gated replay buffer
        -> four credit-assignment regimes, same learner, same env, same seeds

The scientific question: when a learner proposes action ``a`` but the Shield
executes ``Π(a)`` (EIRP clipped to the 33 dBm legal cap), *whom* should the
observed reward be credited to? The environment genuinely makes this matter:
reward (served fraction) is monotone increasing in EIRP, so the unconstrained
optimum lies OUTSIDE the feasible set and the projection binds — proposing
EIRP 50 dBm executes at 33 dBm and earns 0.347, while its true unconstrained
value is 0.592. Credit regimes, same learner, same environment, same schedule:

  A. ``naive_proposed_credit``  — credit the observed reward to the PROPOSED
     action (the bug under test). Every over-cap proposal is silently corrected
     to the same executed action, so the learner books ~0.347 as the value of
     EVERY illegal EIRP: censored feedback, systematically wrong value function.
  B. ``projection_aware``       — credit the observed reward to the EXECUTED
     action. The table simply never claims a value for infeasible EIRPs.
  C. ``correction_aware``       — value credit as B, plus a projection penalty
     attached to the PROPOSED bin in the behaviour score, so the learner learns
     to stop proposing actions the Shield must fix (it "graduates").
  D. ``unconstrained_oracle``   — ORACLE / NOT DEPLOYABLE: trained on the
     counterfactual reward the proposal would earn with NO Shield. Its proposals
     are still emitted only through the Shield (nothing illegal ever reaches the
     RAN), but its learning signal requires an oracle no fielded rApp has. It is
     the safety/utility upper bound: oracle reward minus best feasible reward is
     what the EIRP cap costs on this scenario.

The headline measurement is the value-function error over the INFEASIBLE bins
(EIRP 34..52 dBm): |learned value − true unconstrained value| per bin. Because
the environment is deterministic given (frequency, EIRP), there is no estimation
noise — any error in a claimed value is pure mis-attribution, so the measurement
isolates the credit-assignment bias exactly.

Honest accounting, stated up front: with an exact projection operator the
*realised* reward of regimes A, B and C is nearly identical — the Shield fixes
A's illegal proposals before they cost anything. The naive-credit bias shows up
in (1) the learned value function (A claims ~0.347 for actions worth up to
~0.606), (2) A's terminal indifference between the legal cap and every illegal
EIRP (a 20-way argmax tie, so it keeps proposing illegal actions forever), and
(3) the sustained projection load A imposes on the Shield. If a regime shows no
measurable bias the numbers will say so.

Trust chain: every ``env.step`` writes a DecisionRecord carrying BOTH actions to
a hash-chained JsonlEvidenceStore. After training, the learner tables for A, B
and C are rebuilt exclusively from ``load_transitions`` (the verify-gated replay
buffer) and must match the online tables bit-for-bit; then record #1 is tampered
on disk and ``load_transitions`` must REFUSE the chain (EvidenceIntegrityError) —
a tampered decision log cannot become training data. Regime D is deliberately
NOT rebuildable from the chain: its oracle signal is not recorded evidence.

Pure numpy + the shipped ``horizon_ric`` modules. The real DeepMIMO feature file
is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` (deterministic, checksum-pinned) or point
``--features`` at a cached copy. The committed result JSON is the real 4096-Rx
run; the ``realdata`` CI workflow rebuilds the data and re-runs this loop.
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
N_FEASIBLE_BINS = int(MAX_EIRP_DBM - EIRP_MIN_DBM) + 1  # EIRP 20..33 -> 14 bins.

ROUNDS = 240  # env.step interactions per regime.
EPS_START = 0.5  # epsilon-greedy exploration after the initial full sweep.
EPS_FLOOR = 0.04
CORRECTION_PENALTY = 0.2  # regime C's behaviour penalty per projected proposal.
SEED_BASE = 2027  # all randomness is np.random.default_rng(SEED_BASE + regime).

REGIMES = (
    "naive_proposed_credit",
    "projection_aware",
    "correction_aware",
    "unconstrained_oracle",
)

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


def _bin_of(eirp_dbm: float) -> int:
    b = int(round(eirp_dbm - EIRP_MIN_DBM))
    if not 0 <= b < N_BINS:
        raise ValueError(f"EIRP {eirp_dbm} dBm outside the learner grid")
    return b


def _epsilon(t: int, rounds: int) -> float:
    """Deterministic linear decay from EPS_START to EPS_FLOOR by mid-training."""
    span = max(1, (rounds - N_BINS) // 2)
    return max(EPS_FLOOR, EPS_START * (1.0 - (t - N_BINS) / span))


class _TabularRegime:
    """One credit regime's tables. Incremental exact means; no learning rate."""

    def __init__(self, name: str, rng: np.random.Generator) -> None:
        self.name = name
        self.rng = rng
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
            return self.r_hat - CORRECTION_PENALTY * self.p_hat
        return self.q

    def greedy_bin(self) -> int:
        score = self._score()
        known = ~np.isnan(score)
        best = np.nanmax(score)
        tied = np.flatnonzero(known & (score == best))
        if self.name == "naive_proposed_credit":
            # Exact-tie randomisation (seeded): the naive table CANNOT distinguish
            # the legal cap from any illegal EIRP, and this exposes it honestly.
            return int(self.rng.choice(tied))
        return int(tied[0])

    def greedy_tie_count(self) -> int:
        score = self._score()
        return int(np.sum(~np.isnan(score) & (score == np.nanmax(score))))

    def greedy_bin_deterministic(self) -> int:
        """First tied bin — used for REPORTING only, so the committed JSON and
        cross-host verification never depend on the tie-break rng. The
        behavioural tie-randomisation in :meth:`greedy_bin` is separate and only
        drives regime A's proposals."""
        score = self._score()
        return int(np.flatnonzero(~np.isnan(score) & (score == np.nanmax(score)))[0])


def _train_regime(
    env: ShieldedSpectrumEnv,
    store: JsonlEvidenceStore,
    name: str,
    frequency_hz: float,
    rounds: int,
) -> tuple[_TabularRegime, list[StepResult]]:
    regime = _TabularRegime(name, np.random.default_rng(SEED_BASE + REGIMES.index(name)))
    steps: list[StepResult] = []
    for t in range(rounds):
        if t < N_BINS:
            b = t % N_BINS  # deterministic full sweep: every bin is visited once.
        elif regime.rng.random() < _epsilon(t, rounds):
            b = int(regime.rng.integers(N_BINS))
        else:
            b = regime.greedy_bin()
        proposed = SpectrumAction(frequency_hz, EIRP_MIN_DBM + b - ANTENNA_GAIN_DBI)
        res = env.step(
            proposed,
            decision_id=f"credit-{name}-{t:04d}",
            store=store,
            rapp_instance_id="credit-assignment",
            policy_label=name,
        )
        regime.update(b, res)
        steps.append(res)
    return regime, steps


def _rebuild_from_chain(name: str, transitions: list[Transition]) -> _TabularRegime:
    """Rebuild a Shield-constrained regime's tables ONLY from verified evidence.

    This is the provenance path a fielded learner must use: the replay buffer is
    ``load_transitions`` (verify-gated), and each Transition carries BOTH actions
    plus the projection flags, so every credit rule below is reconstructible.
    The oracle regime is deliberately absent: its counterfactual signal is not
    recorded evidence and cannot be recovered from the chain.
    """
    regime = _TabularRegime(name, np.random.default_rng(0))  # rng unused in replay.
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
    claimed = bins[~np.isnan(claims[bins])]
    if len(claimed) == 0:
        return 0.0, 0
    return float(np.mean(np.abs(claims[claimed] - truth[claimed]))), int(len(claimed))


def _regime_report(
    name: str,
    regime: _TabularRegime,
    steps: list[StepResult],
    truth: np.ndarray,
    rounds: int,
) -> dict[str, Any]:
    feasible = np.arange(N_FEASIBLE_BINS)
    infeasible = np.arange(N_FEASIBLE_BINS, N_BINS)
    mae_inf, claimed_inf = _mae_over(regime.q, truth, infeasible)
    mae_feas, claimed_feas = _mae_over(regime.q, truth, feasible)

    q = max(1, rounds // 4)
    first, last = steps[:q], steps[rounds - q:]
    oracle = name == "unconstrained_oracle"
    # A/B/C realise the executed (post-Shield) reward; the oracle regime's world
    # has no Shield, so its realisation is the unconstrained counterfactual.
    realised = [
        (s.counterfactual_reward if oracle else s.reward) for s in steps
    ]
    realised_last = realised[rounds - q:]
    illegal = [s for s in steps if s.illegal_without_shield]

    report: dict[str, Any] = {
        "credit_rule": CREDIT_RULES[name],
        "oracle": oracle,
        "deployable": not oracle,
        "value_mae_infeasible": round(mae_inf, 6),
        "value_mae_feasible": round(mae_feas, 6),
        "infeasible_bins_claimed": claimed_inf,
        "feasible_bins_claimed": claimed_feas,
        "argmax_proposed_eirp_dbm": float(EIRP_MIN_DBM + regime.greedy_bin_deterministic()),
        "argmax_tie_count": regime.greedy_tie_count(),
        "projection_rate_first_quarter": round(float(np.mean([s.projected for s in first])), 6),
        "projection_rate_last_quarter": round(float(np.mean([s.projected for s in last])), 6),
        "realised_mean_reward": round(float(np.mean(realised)), 6),
        "realised_mean_reward_last_quarter": round(float(np.mean(realised_last)), 6),
        "illegal_proposals": len(illegal),
        "illegal_mean_excess_eirp_db": (
            round(float(np.mean([s.proposed.eirp_dbm - MAX_EIRP_DBM for s in illegal])), 6)
            if illegal
            else 0.0
        ),
        "illegal_mean_counterfactual_reward": (
            round(float(np.mean([s.counterfactual_reward for s in illegal])), 6)
            if illegal
            else 0.0
        ),
        "max_executed_eirp_dbm": round(float(max(s.executed.eirp_dbm for s in steps)), 6),
        "learned_value_by_eirp": {
            str(int(EIRP_MIN_DBM + b)): (None if np.isnan(regime.q[b]) else round(float(regime.q[b]), 6))
            for b in range(N_BINS)
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
    # axis — the scientific axis — is isolated. Subband gains differ by <0.5 dB.
    cap_rewards = [
        env.reward(SpectrumAction(subband_center_hz(b), MAX_EIRP_DBM - ANTENNA_GAIN_DBI))
        for b in range(N_SUBBANDS)
    ]
    best_subband = int(np.argmax(cap_rewards))
    frequency_hz = subband_center_hz(best_subband)
    optimal_feasible = env.optimal_feasible_reward()

    # True UNCONSTRAINED value of every proposed EIRP bin: the real reward the
    # proposal itself would earn with no Shield. This is the measuring stick for
    # the credit-assignment bias; it is real DeepMIMO physics, not a model.
    truth = np.array(
        [
            env.reward(SpectrumAction(frequency_hz, EIRP_MIN_DBM + b - ANTENNA_GAIN_DBI))
            for b in range(N_BINS)
        ]
    )

    ap = audit_path or Path("benchmarks/results/credit_assignment_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)

    trained: dict[str, tuple[_TabularRegime, list[StepResult]]] = {}
    for name in REGIMES:
        trained[name] = _train_regime(env, store, name, frequency_hz, rounds)

    chain_len = len(store)
    verify_intact = store.verify()

    # --- Verify-gated replay: rebuild A/B/C training tables from the chain. ---
    transitions = load_transitions(ap)
    by_regime = {
        name: [t for t in transitions if t.decision_id.startswith(f"credit-{name}-")]
        for name in REGIMES
    }
    replay_matches = True
    for name in ("naive_proposed_credit", "projection_aware", "correction_aware"):
        online = trained[name][0]
        rebuilt = _rebuild_from_chain(name, by_regime[name])
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

    regimes_report = {
        name: _regime_report(name, regime, steps, truth, rounds)
        for name, (regime, steps) in trained.items()
    }
    oracle_last_q = regimes_report["unconstrained_oracle"]["realised_mean_reward_last_quarter"]
    safety_utility_gap = round(float(oracle_last_q - optimal_feasible), 6)

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
            "feasible_bins": N_FEASIBLE_BINS,
            "infeasible_bins": N_BINS - N_FEASIBLE_BINS,
            "rounds_per_regime": rounds,
            "sweep_steps": N_BINS,
            "epsilon_start": EPS_START,
            "epsilon_floor": EPS_FLOOR,
            "correction_penalty": CORRECTION_PENALTY,
            "seed_base": SEED_BASE,
        },
        "true_unconstrained_value_by_eirp": {
            str(int(EIRP_MIN_DBM + b)): round(float(truth[b]), 6) for b in range(N_BINS)
        },
        "regimes": regimes_report,
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
            "counterfactual no deployed rApp has, and is NOT deployable. Because the "
            "projection is exact, realised rewards of the shielded regimes are near-identical; "
            "the measured bias lives in the value function, the terminal argmax tie, and the "
            "sustained projection load."
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

    r = result["regimes"]
    tc = result["trust_chain"]
    # The Shield must genuinely bind (safety has a measurable utility cost here).
    shield_binds = result["safety_utility_gap"] > 0.0
    # The headline: naive proposed-action credit is measurably biased over the
    # infeasible bins, strictly worse than projection-aware credit.
    bias_real = (
        r["naive_proposed_credit"]["value_mae_infeasible"]
        > r["projection_aware"]["value_mae_infeasible"]
    )
    # Correction-aware learning graduates: the Shield corrects it less over time.
    graduates = (
        r["correction_aware"]["projection_rate_last_quarter"]
        < r["correction_aware"]["projection_rate_first_quarter"]
    )
    # No regime ever emitted an illegal action to the RAN.
    all_legal = all(
        r[name]["max_executed_eirp_dbm"] <= MAX_EIRP_DBM + 1e-9 for name in REGIMES
    )
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and tc.get("tampered_chain_refused") is True
        and tc.get("verify_after_tamper_index") not in (None, -1)
        and tc["replay_matches_online"] is True
    )
    return 0 if (shield_binds and bias_real and graduates and all_legal and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
