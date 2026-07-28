#!/usr/bin/env python3
"""The study we should have run: when does learning behind a projection cost anything?

Every committed shield-learning benchmark measures a task whose optimum sits ON
the constraint boundary, learned by a TABULAR value table. The post-mortem in
``deploy/shield-learning/ERRATA.md`` found that this is precisely the one
configuration where the projection's distortion is maximally *visible* and
exactly *free* — realised loss 0.000000 — and that the mechanism which actually
costs utility is excluded by our design twice over.

That mechanism: the Shield executes ``project(a)``, so above the cap every
distinct proposal yields the identical reward. Credited to the PROPOSED action,
the regression target is therefore CONSTANT across the whole infeasible region.
A tabular table is immune — each bin is independent, and the constant simply
fills the unreachable bins. A limited-capacity value model is not: it must fit
that plateau, the fit bleeds into the feasible region, and its argmax is dragged
off the feasible optimum. A boundary-optimal objective hides this too, because
the argmax is at the cap regardless.

This benchmark measures the 2x2 directly, on the real measured DeepMIMO gains:

    reward shape        x   value model    ->   realised loss
    boundary-optimal        tabular             (shipped configuration)
    boundary-optimal        smooth
    interior-optimal        tabular
    interior-optimal        smooth              <- the only cell that can bleed

WHAT THIS DOES NOT CLAIM
------------------------
The interior-optimal objective is a MODELLED coexistence penalty applied to real
measured gains, not a measured incumbent. It exists to move the unconstrained
optimum inside the feasible set, which is the single structural difference
between the two rows. The propagation is real; the incumbent is not. The loss
magnitude is a strong function of how far the interior optimum sits below the
cap, so this reports a SWEEP rather than a headline number — a single figure
would be meaningless without its calibration.

The penalty is deliberately superlinear. Served fraction rises only 0.014694/dB
on this campus, so a linear harm term is either negligible or dominant and can
never produce an interior peak; a linear first attempt put the argmax at the
grid floor for every weight tried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.runtime_env import stamp

NOISE_DBM = -94.0
EIRP_MIN, EIRP_MAX = 20.0, 52.0
CAP_DBM = 33.0
N_BINS = 33
BINS = np.linspace(EIRP_MIN, EIRP_MAX, N_BINS)
PROTECTION_DBW = -105.0
STEPS = 600
SEEDS = 24

# Harm weights chosen to walk the interior optimum from just below the cap down
# to the grid floor. Reported as a curve because the loss depends on that depth.
HARM_WEIGHTS = (2.6e-4, 3.0e-4, 3.6e-4, 4.5e-4, 6.0e-4, 9.0e-4)


def load_gains(features: Path) -> np.ndarray:
    rows = [json.loads(line) for line in features.open() if line.strip()]
    return np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)


def project(eirp: float) -> float:
    """The Shield, reduced to its essential operator: clamp to the licensed cap."""
    return min(eirp, CAP_DBM)


class Env:
    def __init__(self, gains: np.ndarray, subband: int, harm_w: float) -> None:
        self.g = gains[:, subband]
        self.harm_w = harm_w  # 0.0 => the shipped boundary-optimal served fraction
        order = np.argsort(self.g)[::-1]
        self.victims = order[: max(1, len(order) // 10)]

    def reward(self, eirp: float) -> float:
        served = float(np.mean(eirp + self.g - NOISE_DBM >= 0.0))
        if self.harm_w == 0.0:
            return served
        excess = float(
            np.mean(np.maximum(0.0, eirp + self.g[self.victims] - PROTECTION_DBW))
        )
        return served - self.harm_w * (excess**2)

    def true_argmax_dbm(self) -> float:
        return float(BINS[int(np.argmax([self.reward(b) for b in BINS]))])

    def optimal_feasible(self) -> float:
        return float(max(self.reward(b) for b in BINS if b <= CAP_DBM))


def _value(sums: np.ndarray, counts: np.ndarray, degree: int | None) -> np.ndarray:
    seen = counts > 0
    mean = np.where(seen, sums / np.maximum(counts, 1), -np.inf)
    if degree is None or seen.sum() < degree + 2:
        return mean  # tabular: bins independent, immune to the plateau
    coef = np.polyfit(BINS[seen], (sums / np.maximum(counts, 1))[seen], degree)
    return np.polyval(coef, BINS)


def run_learner(env: Env, degree: int | None, seed: int) -> dict[str, float]:
    """Epsilon-greedy over PROPOSED EIRP, credited with the REALISED reward."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(N_BINS)
    counts = np.zeros(N_BINS)
    for t in range(STEPS):
        eps = max(0.05, 1.0 - t / (0.7 * STEPS))
        if rng.random() < eps or counts.sum() == 0:
            b = int(rng.integers(N_BINS))
        else:
            b = int(np.argmax(_value(sums, counts, degree)))
        sums[b] += env.reward(project(float(BINS[b])))
        counts[b] += 1
    chosen = int(np.argmax(_value(sums, counts, degree)))
    executed = project(float(BINS[chosen]))
    realised = env.reward(executed)
    best = env.optimal_feasible()
    return {
        "proposed_eirp_dbm": float(BINS[chosen]),
        "executed_eirp_dbm": executed,
        "realised": realised,
        "loss": best - realised,
        "loss_frac": (best - realised) / best if best > 0 else float("nan"),
    }


def _cell(env: Env, degree: int | None) -> dict[str, Any]:
    runs = [run_learner(env, degree, s) for s in range(SEEDS)]
    loss = np.array([r["loss_frac"] for r in runs])
    return {
        "mean_loss_fraction": round(float(loss.mean()), 6),
        "max_loss_fraction": round(float(loss.max()), 6),
        "std_loss_fraction": round(float(loss.std()), 6),
        "max_executed_eirp_dbm": round(max(r["executed_eirp_dbm"] for r in runs), 4),
    }


def run(features: Path, manifest: Path) -> dict[str, Any]:
    gains = load_gains(features)
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    boundary = Env(gains, 3, 0.0)
    result: dict[str, Any] = {
        "benchmark": "projection distortion vs value-model capacity (2x2 + depth sweep)",
        "dataset": meta.get("dataset"),
        "scenario": meta.get("scenario"),
        "features_sha256": meta.get("features_sha256"),
        "data_kind": meta.get("data_kind")
        or "site-specific Wireless InSite ray tracing; not over-the-air capture",
        "receivers": int(gains.shape[0]),
        "eirp_cap_dbm": CAP_DBM,
        "scope_note": (
            "Real measured propagation; the incumbent coexistence penalty is MODELLED, "
            "not measured, and exists only to move the unconstrained optimum inside the "
            "feasible set. Loss magnitude depends strongly on how far that optimum sits "
            "below the cap, so a sweep is reported rather than a single headline."
        ),
        "boundary_optimal": {
            "true_argmax_dbm": boundary.true_argmax_dbm(),
            "optimal_feasible_reward": round(boundary.optimal_feasible(), 6),
            "tabular": _cell(boundary, None),
            "smooth_poly2": _cell(boundary, 2),
        },
        "interior_optimal_sweep": [],
    }

    for w in HARM_WEIGHTS:
        env = Env(gains, 3, w)
        if env.optimal_feasible() <= 0:
            continue
        argmax = env.true_argmax_dbm()
        if argmax >= CAP_DBM:
            continue  # not actually interior; the row would not test anything
        result["interior_optimal_sweep"].append(
            {
                "harm_weight": w,
                "true_argmax_dbm": argmax,
                "gap_below_cap_db": round(CAP_DBM - argmax, 4),
                "optimal_feasible_reward": round(env.optimal_feasible(), 6),
                "tabular": _cell(env, None),
                "smooth_poly2": _cell(env, 2),
                "smooth_poly3": _cell(env, 3),
            }
        )

    sweep = result["interior_optimal_sweep"]
    smooth_losses = [c["smooth_poly2"]["mean_loss_fraction"] for c in sweep]
    result["findings"] = {
        "tabular_loss_is_zero_everywhere": all(
            c["tabular"]["mean_loss_fraction"] == 0.0 for c in sweep
        )
        and result["boundary_optimal"]["tabular"]["mean_loss_fraction"] == 0.0,
        "boundary_optimal_hides_the_defect": (
            result["boundary_optimal"]["smooth_poly2"]["mean_loss_fraction"] == 0.0
        ),
        "smooth_loss_min_fraction": min(smooth_losses) if smooth_losses else None,
        "smooth_loss_max_fraction": max(smooth_losses) if smooth_losses else None,
        "no_illegal_action_executed": all(
            c["tabular"]["max_executed_eirp_dbm"] <= CAP_DBM
            and c["smooth_poly2"]["max_executed_eirp_dbm"] <= CAP_DBM
            for c in sweep
        ),
    }
    return stamp(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    ap.add_argument(
        "--manifest", type=Path, default=Path("datasets/deepmimo_asu_3p5/manifest.json")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/projection_capacity_2x2.json")
    )
    args = ap.parse_args()

    result = run(args.features, args.manifest)
    f = result["findings"]
    if not f["tabular_loss_is_zero_everywhere"]:
        raise SystemExit("tabular learner lost utility; the 2x2 premise does not hold")
    if not f["no_illegal_action_executed"]:
        raise SystemExit("an executed action exceeded the cap — safety guarantee broken")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
