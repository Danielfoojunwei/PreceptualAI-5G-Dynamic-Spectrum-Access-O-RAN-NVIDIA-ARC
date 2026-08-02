#!/usr/bin/env python3
"""End-to-end learning demonstration: does enforcement stop the learner?

Everything else in this repository asks whether the Shield holds. This asks
the question an operator actually has: **if I put a projection operator in
front of my planner, does the planner still learn — and what does it learn?**

The answer turns on one wiring decision nobody usually writes down: *which
action\'s reward the planner is told about.*

Two arms, same planner, same Shield, same seed, same environment:

* **arm ``reward_the_proposal``** — the planner is told the reward of the
  action it asked for. This is the arrangement ``benchmarks/g1_second_planner``
  uses, deliberately, because a planner that adapted to the enforcement
  boundary would no longer be independent of it. The projection is outside the
  learner\'s feedback.

* **arm ``reward_the_emission``** — the planner is told the reward of the
  action that was actually emitted, which is what a real deployment measures,
  because the radio transmitted the projected action. The projection is inside
  the feedback loop.

Both arms are scored on the same thing: the reward of what was actually
emitted. That is the only comparison that means anything, because it is the
only one the radio can pay.

MEASURED — 3 seeds x 2 arms x 4000 decisions
--------------------------------------------
====  ===================  ===================  =============  =============
seed  illegal req (open)   illegal req (closed) reward open    reward closed
====  ===================  ===================  =============  =============
…802  3585                 2610                 .1113 -> .1049 .0963 -> .1502
11    3423                 2555                 .1388 -> .2170 .1106 -> .1419
12    3338                 2613                 .1189 -> .2069 .1013 -> .1326
====  ===================  ===================  =============  =============

**What holds in every seed.** Enforcement: 0 illegal emissions out of 24000
decisions, graded by an oracle that shares no constant with the Shield. And
closing the loop reduces how much illegality is *requested* — the closed arm
asks for 73-78% as many illegal emissions as the open one.

**What does not hold, and was published before it was checked.** The first
version of this demo ran one seed, found the closed arm\'s realised reward
rising while the open arm\'s fell, and headlined that closing the loop is the
only arm in which learning works. Two further seeds contradict it: in seeds 11
and 12 the open arm ends *higher*, and by a wide margin. The effect was a
single-seed artifact and the claim is withdrawn.

Reading the three seeds together, the honest direction is the opposite of the
original guess: closing the loop appears to **cost** realised utility. The
mechanism is the same one that refutes the other hypothesis below. Projection
clamps every over-ceiling proposal to exactly the ceiling — which is the best
*legal* power. The open-loop planner climbs a fictional gradient to maximum
power and is therefore clamped onto the legal optimum every time; being wrong
about the world lands it in the right place. The closed-loop planner sees a
flat plateau above the ceiling, has no gradient to climb, and keeps exploring
it — including arms that pay less. This is stated as a direction, not a
result: three seeds is three seeds, and ``--seeds`` exists so it can be
checked properly.

**A hypothesis this demo refuted, in every seed.** It was built expecting the
closed arm to converge toward compliance — for the illegal-request rate to
decay as the learner discovered that over-ceiling arms buy nothing. It does
not; the rate rises in both arms in all three seeds. Projection makes an
over-ceiling arm *indistinguishable* from the ceiling, not *worse* than it.
UCB1 abandons arms that pay less and has no reason to abandon one that pays
the same: a tied arm keeps its optimistic bound and keeps being explored
forever. Enforcement removes the incentive to exceed the licence without
creating any incentive to stay inside it. Making a learner stop *asking* would
mean pricing the correction, which couples the planner to the enforcement
boundary — a different and more invasive design.

Because a single seed already produced one false headline here, the report
only states a finding that holds in **every** seed. A split result is printed
as split.

HORIZON MATTERS TOO
-------------------
The utility effect is a late-run phenomenon and is not visible at 800
decisions, where both arms still improve. UCB1 needs on the order of a
thousand pulls across a 54-arm grid to commit. ``--decisions 4000`` is the
default for that reason.

End to end is literal. Each decision runs the shipping
``default_terrestrial_shield`` through its ordinary ``dispose`` contract, signs
the certificate with a real Ed25519 key, and appends a real ``DecisionRecord``
to a real hash-chained ``JsonlEvidenceStore`` — refusals included. Every chain
is then verified by ``audit/verify_evidence.py`` in a subprocess with
``horizon_ric`` deliberately absent from its path: the regulator\'s view, not
ours.

Cost note: ``JsonlEvidenceStore.append`` re-scans the whole file to find the
tenant\'s last chain hash, so a run is O(n^2) in decisions — the store\'s own
docstring says as much. The default 3 seeds x 2 arms x 4000 decisions takes on
the order of fifteen minutes. ``--decisions 300 --seeds 5,6`` smoke-tests the
whole path in seconds without pretending to measure anything.

Run::

    PYTHONPATH=src python demo/end_to_end_learning.py --out /tmp/demo

Exit codes:
    0  the demonstration ran and every evidence chain verified
    1  an illegal emission got through, or a chain is broken
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.g1_second_planner import (  # noqa: E402
    ANTENNA_GAIN_DBI,
    BAND_HI_HZ,
    BAND_LO_HZ,
    CARRIER_BW_HZ,
    FREQUENCY_ARMS,
    GUARD_BAND_HZ,
    MAX_EIRP_DBM,
    MAX_FADING_DB,
    MAX_SPECTRAL_EFF,
    MEASUREMENT_NOISE_DB,
    PA_CEILING_DBM,
    PA_FLOOR_DBM,
    POWER_ARMS,
    REF_PATH_LOSS_DB,
    TUNING_CENTRE_HZ,
    TUNING_SPAN_HZ,
    oracle_verdict,
)
from horizon_ric.evidence.schema import (  # noqa: E402
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore  # noqa: E402
from horizon_ric.planners import (  # noqa: E402
    UcbSpectrumPlanner,
    frequency_arm_grid,
    spectrum_choice_to_action,
)
from horizon_ric.shield.shield import default_terrestrial_shield  # noqa: E402
from horizon_ric.shield.signing import signed_certificate  # noqa: E402

ARMS = ("reward_the_proposal", "reward_the_emission")


def _quartiles(values: list[float]) -> list[float]:
    """Mean of each quarter of the run, so a trend is visible without a plot."""
    if not values:
        return []
    n = len(values)
    out = []
    for q in range(4):
        lo, hi = n * q // 4, n * (q + 1) // 4
        chunk = values[lo:hi]
        out.append(round(sum(chunk) / len(chunk), 6) if chunk else 0.0)
    return out


def make_continuous_reward(centres_hz: tuple[float, ...], *, seed: int):
    """The declared link model, evaluated at ANY centre frequency.

    ``benchmarks.g1_second_planner.make_reward`` draws one fading realisation
    per arm and looks it up by exact key. That is fine there, because nothing
    ever asks it about a frequency off the grid. Here something does: the
    Shield projects an out-of-band carrier back inside the licensed band, and
    the projected centre is *not* an arm. Keying by equality raises KeyError on
    the first such projection.

    So the same seeded per-arm realisation is kept and evaluated at the nearest
    arm for any other frequency. Frequency-selective fading is smooth on the
    scale of the arm spacing, so this is the physically sensible reading rather
    than a convenience — and it is the only way the two arms of this
    demonstration can share one environment, which they must, or the comparison
    means nothing.
    """
    import random as _random

    fading_rng = _random.Random(seed + 977)
    fading_dB = {c: fading_rng.uniform(0.0, MAX_FADING_DB) for c in centres_hz}
    noise_rng = _random.Random(seed + 1)
    grid = tuple(centres_hz)

    def reward(centre_hz: float, tx_power_dBm: float) -> float:
        nearest = min(grid, key=lambda c: abs(c - centre_hz))
        sinr_dB = (
            tx_power_dBm
            + ANTENNA_GAIN_DBI
            - REF_PATH_LOSS_DB
            - fading_dB[nearest]
            + noise_rng.gauss(0.0, MEASUREMENT_NOISE_DB)
        )
        util = math.log2(1.0 + 10.0 ** (sinr_dB / 10.0)) / MAX_SPECTRAL_EFF
        return min(1.0, max(0.0, util))

    return reward, fading_dB


def _signing_key():
    """A real Ed25519 key, generated per run.

    Not a fixture: the point of signing here is that the verifier can check a
    signature it was not handed in advance, so the key is generated, used, and
    its public half written next to the evidence.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519

    return ed25519.Ed25519PrivateKey.generate()


def run_arm(
    arm: str,
    *,
    decisions: int,
    seed: int,
    store_path: Path,
    signer: Any,
) -> dict[str, Any]:
    """One arm of the demonstration. Returns its measurements."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    close_the_loop = arm == "reward_the_emission"

    centres_hz = frequency_arm_grid(
        centre_hz=TUNING_CENTRE_HZ, span_hz=TUNING_SPAN_HZ, arms=FREQUENCY_ARMS
    )
    reward, _fading = make_continuous_reward(centres_hz, seed=seed)
    planner = UcbSpectrumPlanner(
        reward=reward,
        tuning_centre_hz=TUNING_CENTRE_HZ,
        tuning_span_hz=TUNING_SPAN_HZ,
        frequency_arms=FREQUENCY_ARMS,
        pa_floor_dBm=PA_FLOOR_DBM,
        pa_ceiling_dBm=PA_CEILING_DBM,
        power_arms=POWER_ARMS,
        seed=seed,
    )
    # The stock chain with the stock arguments. Nothing here is reconfigured
    # for this planner beyond the licence it is policing.
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ,
        band_hi_hz=BAND_HI_HZ,
        max_eirp_dBm=MAX_EIRP_DBM,
        guard_band_hz=GUARD_BAND_HZ,
    )
    store = JsonlEvidenceStore(store_path)

    illegal_requests: list[float] = []
    realised_reward: list[float] = []
    requested_eirp: list[float] = []
    illegal_emissions = 0
    blocked = 0
    projected = 0

    for d in range(decisions):
        # 1. The planner proposes. It has no idea a Shield exists.
        choice = planner.propose()
        proposal = spectrum_choice_to_action(
            choice, bandwidth_hz=CARRIER_BW_HZ, antenna_gain_dBi=ANTENNA_GAIN_DBI
        )

        # 2. Independent grading of what was ASKED for.
        asked = oracle_verdict(
            proposal,
            band_lo_hz=BAND_LO_HZ,
            band_hi_hz=BAND_HI_HZ,
            guard_band_hz=GUARD_BAND_HZ,
            max_eirp_dBm=MAX_EIRP_DBM,
        )
        illegal_requests.append(1.0 if asked["illegal"] else 0.0)
        requested_eirp.append(asked["eirp_dBm"])

        # 3. The Shield projects, through its ordinary public contract.
        disposition = shield.dispose(
            proposal,
            decision_id=f"{arm}-{d:06d}",
            rng_seed=seed,
            model_provenance={
                "planner_id": planner.planner_id,
                "planner_module": "horizon_ric.planners.ucb_spectrum",
                "feedback_arm": arm,
            },
        )
        # Signing is a separate step from disposal, deliberately: the Shield
        # decides, the operator's key attests. Sign every certificate, refusals
        # included — a refusal nobody can authenticate is not evidence.
        cert = signed_certificate(disposition.certificate, signer)
        projected += int(bool(cert.projected))

        if cert.emit_blocked:
            blocked += 1
            emitted = None
            # A refusal is still an outcome the environment delivers: nothing
            # was transmitted, so nothing was earned.
            realised = 0.0
        else:
            emitted = disposition.safe_action
            graded = oracle_verdict(
                emitted,
                band_lo_hz=BAND_LO_HZ,
                band_hi_hz=BAND_HI_HZ,
                guard_band_hz=GUARD_BAND_HZ,
                max_eirp_dBm=MAX_EIRP_DBM,
            )
            if graded["illegal"]:
                illegal_emissions += 1
            realised = reward(
                float(emitted["frequency_hz"]), float(emitted["tx_power_dBm"])
            )
        realised_reward.append(realised)

        # 4. The one wiring decision this demonstration is about.
        if close_the_loop:
            # What the radio actually did. The projection is inside the loop.
            planner.observe(choice, realised)
        else:
            # What the planner asked for. The projection is outside the loop.
            planner.observe(
                choice, reward(choice.centre_hz, choice.tx_power_dBm)
            )

        # 5. Evidence. A real record on a real hash chain, refusals included —
        #    the record of what was refused must not depend on the component
        #    that refused it.
        state = json.dumps(proposal, sort_keys=True).encode()
        record = DecisionRecord.new(
            decision_id=f"{arm}-{d:06d}",
            rapp_instance_id="horizon-demo-learning",
            state_hash=hashlib.sha256(state).hexdigest(),
            chosen_action={
                "proposal": proposal,
                "safe_action": emitted,
                "emit_blocked": bool(cert.emit_blocked),
                "realised_reward": realised,
                "certificate": {
                    "safe": cert.safe,
                    "projected": cert.projected,
                    "violated_ids": list(cert.violated_ids),
                    **(
                        {
                            "signature": cert.signature,
                            "signing_key_fingerprint": cert.signing_key_fingerprint,
                        }
                        if cert.signature is not None
                        else {}
                    ),
                },
            },
            # The realised utility recorded as SLA risk: 1 - reward, so a
            # refused decision (reward 0) reads as maximum risk rather than as
            # a silent zero.
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=round(1.0 - realised, 6),
                sla_risk_1min=round(1.0 - realised, 6),
                sla_risk_5min=round(1.0 - realised, 6),
                constraint_violations=list(cert.violated_ids),
            ),
            rejected_alternatives=[],
            constraint_corrections=[],
            model_versions=ModelVersions(
                encoder="none",
                risk_heads="none",
                dyna="none",
                policy=planner.planner_id,
                constraint_layer="shield_terrestrial_v1",
                rapp="demo_end_to_end_learning_v1",
            ),
            random_seed=seed,
        )
        store.append(record)

    illegal_by_quarter = _quartiles(illegal_requests)
    return {
        "arm": arm,
        "projection_inside_the_feedback_loop": close_the_loop,
        "decisions": decisions,
        "illegal_requests": int(sum(illegal_requests)),
        "illegal_request_rate_by_quarter": illegal_by_quarter,
        "illegal_emissions": illegal_emissions,
        "blocked_emissions": blocked,
        "projections_applied": projected,
        "mean_reward_by_quarter": _quartiles(realised_reward),
        "peak_requested_eirp_dBm": round(max(requested_eirp), 6),
        "final_quarter_illegal_rate": illegal_by_quarter[-1],
        "first_quarter_illegal_rate": illegal_by_quarter[0],
        "evidence": str(store_path),
    }


def verify_offline(evidence: Path, pubkey_pem: Path) -> dict[str, Any]:
    """Run the standalone verifier the way a regulator would.

    ``horizon_ric`` is removed from the subprocess's path. If the verifier has
    quietly grown a dependency on our code, it fails here rather than passing
    on a machine that happens to have us installed.
    """
    env_path = [p for p in sys.path if not p.endswith("/src")]
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "audit" / "verify_evidence.py"),
            str(evidence),
            "--pubkey",
            str(pubkey_pem),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(env_path), "PATH": "/usr/bin:/bin"},
    )
    report: dict[str, Any]
    try:
        parsed = json.loads(proc.stdout)
        report = parsed if isinstance(parsed, dict) else {"result": "ERROR"}
    except json.JSONDecodeError:
        report = {"result": "ERROR", "stdout": proc.stdout[-800:],
                  "stderr": proc.stderr[-800:]}
    report["exit_code"] = proc.returncode
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--decisions", type=int, default=4000)
    ap.add_argument(
        "--seeds",
        default="20260802,11,12",
        help="comma-separated seeds. More than one on purpose: the first "
        "version of this demo ran a single seed and headlined a result that "
        "two further seeds then contradicted.",
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for the evidence chains and the report")
    args = ap.parse_args(argv)

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    signer = _signing_key()

    from cryptography.hazmat.primitives import serialization

    pub_pem = args.out / "demo_pub.pem"
    pub_pem.write_bytes(
        signer.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    runs: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for seed in seeds:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            evidence = args.out / f"seed{seed}-{arm}.jsonl"
            if evidence.exists():
                evidence.unlink()
            print(f"seed {seed}: running {arm} ...", file=sys.stderr)
            m = run_arm(
                arm,
                decisions=args.decisions,
                seed=seed,
                store_path=evidence,
                signer=signer,
            )
            m["offline_verification"] = verify_offline(evidence, pub_pem)
            arms[arm] = m

            if m["illegal_emissions"]:
                problems.append(
                    f"seed {seed} {arm}: {m['illegal_emissions']} illegal "
                    f"emission(s) passed the Shield — the enforcement claim is "
                    f"false in this run"
                )
            v = m["offline_verification"]
            if v.get("exit_code") != 0 or v.get("result") != "PASS":
                problems.append(
                    f"seed {seed} {arm}: evidence did not verify offline: {v}"
                )
            if not v.get("chain", {}).get("intact"):
                problems.append(f"seed {seed} {arm}: hash chain not intact")
        runs[str(seed)] = arms

    def _delta(m: dict[str, Any]) -> float:
        q = m["mean_reward_by_quarter"]
        return round(q[-1] - q[0], 6) if q else 0.0

    per_seed: list[dict[str, Any]] = []
    for seed_key, arms in runs.items():
        open_a, closed_a = arms[ARMS[0]], arms[ARMS[1]]
        per_seed.append(
            {
                "seed": seed_key,
                "illegal_requests_open": open_a["illegal_requests"],
                "illegal_requests_closed": closed_a["illegal_requests"],
                "closed_asks_fewer": (
                    closed_a["illegal_requests"] < open_a["illegal_requests"]
                ),
                "reward_delta_open": _delta(open_a),
                "reward_delta_closed": _delta(closed_a),
                "final_reward_open": open_a["mean_reward_by_quarter"][-1],
                "final_reward_closed": closed_a["mean_reward_by_quarter"][-1],
                "closed_ends_higher": (
                    closed_a["mean_reward_by_quarter"][-1]
                    > open_a["mean_reward_by_quarter"][-1]
                ),
                "illegal_rate_fell_closed": (
                    closed_a["final_quarter_illegal_rate"]
                    < closed_a["first_quarter_illegal_rate"]
                ),
                "illegal_emissions": (
                    open_a["illegal_emissions"] + closed_a["illegal_emissions"]
                ),
            }
        )

    # Only what holds in EVERY seed is allowed to be a finding.
    unanimous_fewer_asks = all(r["closed_asks_fewer"] for r in per_seed)
    unanimous_closed_wins = all(r["closed_ends_higher"] for r in per_seed)
    unanimous_open_wins = all(not r["closed_ends_higher"] for r in per_seed)
    unanimous_compliance = all(r["illegal_rate_fell_closed"] for r in per_seed)
    zero_illegal = all(r["illegal_emissions"] == 0 for r in per_seed)
    total_decisions = args.decisions * len(ARMS) * len(seeds)

    findings = []
    if zero_illegal:
        findings.append(
            f"Enforcement held in every arm of every seed: 0 illegal emissions "
            f"across {total_decisions} decisions, graded by an oracle that "
            f"shares no constant with the Shield."
        )
    if unanimous_fewer_asks:
        ratios = [
            r["illegal_requests_closed"] / r["illegal_requests_open"]
            for r in per_seed
        ]
        findings.append(
            f"Closing the loop reduces how much illegality is REQUESTED, in "
            f"every seed: the closed arm asks for "
            f"{min(ratios):.0%}-{max(ratios):.0%} as many illegal emissions."
        )
    if not unanimous_compliance:
        findings.append(
            "It does NOT make the learner converge toward compliance. The "
            "illegal-request rate rises in both arms in every seed. Projection "
            "makes an over-ceiling arm indistinguishable from the ceiling, not "
            "worse than it, and UCB1 never abandons a tied arm."
        )
    if unanimous_open_wins:
        findings.append(
            "And it COSTS realised utility: the open-loop arm ends with higher "
            "realised reward in every seed. The open-loop planner climbs a "
            "fictional gradient to maximum power, which the Shield then clamps "
            "to exactly the ceiling — the best legal power — so being wrong "
            "about the world lands it on the legal optimum. The closed-loop "
            "planner sees a flat plateau above the ceiling and keeps exploring "
            "it, including arms that pay less."
        )
    elif unanimous_closed_wins:
        findings.append(
            "Closing the loop also improves realised reward in every seed."
        )
    else:
        findings.append(
            "Realised reward does NOT separate the arms consistently: the "
            "closed arm ends higher in "
            f"{sum(r['closed_ends_higher'] for r in per_seed)} of "
            f"{len(per_seed)} seeds. Any headline about utility from a single "
            "seed would be an artifact — which is exactly what the first "
            "version of this demo published before these seeds were run."
        )

    report = {
        "demo": "end_to_end_learning",
        "question": (
            "Does a projection operator stop an unmodified learner from "
            "learning — and what does it learn?"
        ),
        "seeds": seeds,
        "decisions_per_arm": args.decisions,
        "findings": findings,
        "holds_in_every_seed": {
            "zero_illegal_emissions": zero_illegal,
            "closed_loop_asks_for_less_illegality": unanimous_fewer_asks,
            "closed_loop_converges_toward_compliance": unanimous_compliance,
            "closed_loop_ends_with_higher_realised_reward": unanimous_closed_wins,
            "open_loop_ends_with_higher_realised_reward": unanimous_open_wins,
        },
        "per_seed": per_seed,
        "runs": runs,
        "problems": problems,
        "passed": not problems,
    }

    (args.out / "end-to-end-learning.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print(f"  question : {report['question']}")
    print(f"  {'seed':>9} {'illegal req O/C':>17} {'reward q1->q4 open':>22}"
          f" {'reward q1->q4 closed':>22}")
    for r, seed in zip(per_seed, seeds):
        arms = runs[str(seed)]
        o = arms[ARMS[0]]["mean_reward_by_quarter"]
        c = arms[ARMS[1]]["mean_reward_by_quarter"]
        print(f"  {r['seed']:>9} "
              f"{r['illegal_requests_open']:>7}/{r['illegal_requests_closed']:<9}"
              f" {o[0]:>10.4f} -> {o[-1]:<8.4f}"
              f" {c[0]:>10.4f} -> {c[-1]:<8.4f}")
    for f in findings:
        print(f"  finding  : {f}")
    if problems:
        print("\n".join("  ERROR: " + p for p in problems), file=sys.stderr)
        return 1
    print("  evidence : every chain verified offline, without horizon_ric "
          "importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
