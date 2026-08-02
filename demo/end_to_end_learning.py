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
  because the radio transmitted the projected action and not the proposal. The
  projection is inside the feedback loop.

WHAT WAS MEASURED (4000 decisions, seed 20260802)
-------------------------------------------------
=========================  ==================  ==================
                           reward_the_proposal reward_the_emission
=========================  ==================  ==================
illegal requests           3585  (89.6%)       2610  (65.3%)
mean reward, 1st quarter   0.1113              0.0963
mean reward, 4th quarter   0.1049              0.1502
illegal emissions          0                   0
=========================  ==================  ==================

**The finding.** Closing the loop around the projection does not stop the
learner — it is the only arm in which learning actually works. Its realised
reward rises monotonically across the run, +56% first quarter to last, while
the open-loop arm\'s realised reward *falls*. The open-loop planner is
optimising a fiction: it is told what its proposal would have earned, keeps
choosing arms on that basis, and the radio keeps transmitting something else.
It gets steadily better at a task it is not performing.

**And a hypothesis this demo refuted.** It was built expecting the closed arm
to converge toward compliance — for the illegal-request rate to decay as the
learner discovered that over-ceiling arms buy nothing. It does not. The rate
rises in both arms (0.601 -> 0.699 closed, 0.763 -> 0.970 open); the closed
arm merely asks for 27% fewer illegal emissions in total.

The mechanism is exact, and it is the reason the original guess was wrong.
Projection makes an over-ceiling arm *indistinguishable* from the ceiling, not
*worse* than it. UCB1 abandons arms that pay less; it has no reason whatever to
abandon an arm that pays the same. A tied arm keeps its optimistic bound and
keeps being explored forever. Enforcement removes the incentive to exceed the
licence without creating any incentive to stay inside it.

That distinction matters for anyone designing this: if you want a learner to
stop *asking*, the projection alone will not do it — you would have to price
the correction, which is a different and more invasive design that couples the
planner to the enforcement boundary. What the projection guarantees is what it
claims to guarantee: nothing illegal is ever emitted, in either arm, 0 out of
8000 decisions.

The demo reports whichever way both measurements come out. What is an *error*,
and exits non-zero, is an illegal emission getting past the Shield or an
evidence chain failing to verify.

End to end means end to end. Each decision runs the shipping
``default_terrestrial_shield`` through its ordinary ``dispose`` contract, signs
the certificate with a real Ed25519 key, and appends a real ``DecisionRecord``
to a real hash-chained ``JsonlEvidenceStore``. At the end the chain is verified
by ``audit/verify_evidence.py`` in a subprocess with ``horizon_ric``
deliberately absent from its path — the regulator\'s view, not ours.

Legality is graded by ``benchmarks.g1_second_planner.oracle_verdict``, which
imports nothing from the Shield and recomputes every bound from first
principles, so this is not the system marking its own homework.

HORIZON MATTERS, AND ONE SEED IS ONE SEED
-----------------------------------------
The reward separation is a late-run effect and does **not** appear at 800
decisions: there both arms still improve (open 0.0905 -> 0.1264, closed 0.0874
-> 0.1042) and the summary says so rather than claiming a result it cannot see.
The open-loop arm\'s realised reward only starts falling once UCB1 has
committed to the illegal high-power arms, which takes on the order of a
thousand pulls across a 54-arm grid. ``--decisions 4000`` is the default for
that reason, not for show.

Everything above is one seed. Treat the direction as demonstrated and the
magnitude as anecdotal until it has been run across several; ``--seed`` is
there for exactly that.

Cost note: ``JsonlEvidenceStore.append`` re-scans the whole file to find the
tenant\'s last chain hash, so a run is O(n^2) in decisions — the store\'s own
docstring says as much. 4000 decisions x 2 arms takes a few minutes.

Run::

    PYTHONPATH=src python demo/end_to_end_learning.py --out /tmp/demo

Exit codes:
    0  the demonstration ran and its evidence verified
    1  an illegal emission got through, or the evidence chain is broken
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
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for the evidence chains and the report")
    args = ap.parse_args(argv)

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

    arms = {}
    for arm in ARMS:
        evidence = args.out / f"{arm}.jsonl"
        if evidence.exists():
            evidence.unlink()
        print(f"running {arm} ...", file=sys.stderr)
        arms[arm] = run_arm(
            arm,
            decisions=args.decisions,
            seed=args.seed,
            store_path=evidence,
            signer=signer,
        )
        arms[arm]["offline_verification"] = verify_offline(evidence, pub_pem)

    open_arm = arms["reward_the_proposal"]
    closed_arm = arms["reward_the_emission"]

    def _delta(m):
        q = m["mean_reward_by_quarter"]
        return round(q[-1] - q[0], 6) if q else 0.0

    open_delta, closed_delta = _delta(open_arm), _delta(closed_arm)
    # The primary measurement: does realised reward improve? Only the arm
    # that is told what the radio actually did can improve on what the radio
    # actually did.
    learning_works_only_when_closed = closed_delta > 0 and open_delta <= 0

    # The secondary measurement, and the hypothesis this demo was built to
    # test. It came out false; see the module docstring for why.
    closed_first = closed_arm["first_quarter_illegal_rate"]
    closed_last = closed_arm["final_quarter_illegal_rate"]
    converged = closed_last < closed_first

    problems = []
    for arm, m in arms.items():
        if m["illegal_emissions"]:
            problems.append(
                f"{arm}: {m['illegal_emissions']} illegal emission(s) passed the "
                f"Shield — the enforcement claim is false in this run"
            )
        v = m["offline_verification"]
        if v.get("exit_code") != 0 or v.get("result") != "PASS":
            problems.append(f"{arm}: evidence chain did not verify offline: {v}")
        if not v.get("chain", {}).get("intact"):
            problems.append(f"{arm}: hash chain not intact")

    report = {
        "demo": "end_to_end_learning",
        "question": (
            "Does a projection operator stop an unmodified learner from "
            "learning — and what does it learn?"
        ),
        "finding": (
            "Closing the loop around the projection is the only arm in which "
            "learning works: realised reward rises "
            f"{open_arm['mean_reward_by_quarter'][0]:.4f} -> "
            f"{closed_arm['mean_reward_by_quarter'][-1]:.4f} while the "
            "open-loop arm's realised reward falls. The open-loop planner "
            "optimises a fiction — it is told what its proposal would have "
            "earned while the radio transmits something else."
            if learning_works_only_when_closed
            else "Realised reward did not separate the two arms in this run; "
                 "the demonstration reports that rather than hiding it."
        ),
        "learning_works_only_when_the_loop_is_closed": learning_works_only_when_closed,
        "realised_reward_change_first_to_last_quarter": {
            "open_loop": open_delta,
            "closed_loop": closed_delta,
        },
        "refuted_hypothesis": {
            "was": (
                "closing the loop would make the learner converge toward "
                "compliance, the illegal-request rate decaying as it "
                "discovered that over-ceiling arms buy nothing"
            ),
            "held": converged,
            "why_not": (
                "Projection makes an over-ceiling arm INDISTINGUISHABLE from "
                "the ceiling, not worse than it. UCB1 abandons arms that pay "
                "less and has no reason to abandon one that pays the same: a "
                "tied arm keeps its optimistic bound and keeps being explored. "
                "Enforcement removes the incentive to exceed the licence "
                "without creating one to stay inside it. Making a learner stop "
                "ASKING would require pricing the correction, which couples "
                "the planner to the enforcement boundary — a different and "
                "more invasive design."
            ),
            "illegal_requests_open_loop": open_arm["illegal_requests"],
            "illegal_requests_closed_loop": closed_arm["illegal_requests"],
        },
        "what_the_projection_does_guarantee": (
            "nothing illegal is emitted, in either arm: "
            f"{sum(m['illegal_emissions'] for m in arms.values())} illegal "
            f"emissions across {sum(m['decisions'] for m in arms.values())} "
            "decisions"
        ),
        "illegal_request_rate_by_quarter": {
            "open_loop": open_arm["illegal_request_rate_by_quarter"],
            "closed_loop": closed_arm["illegal_request_rate_by_quarter"],
        },
        "enforcement_held_in_both_arms": all(
            m["illegal_emissions"] == 0 for m in arms.values()
        ),
        "arms": arms,
        "seed": args.seed,
        "problems": problems,
        "passed": not problems,
    }

    (args.out / "end-to-end-learning.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print(f"  question : {report['question']}")
    for arm, m in arms.items():
        print(
            f"  {arm:22s} illegal requests {m['illegal_requests']:5d}"
            f"  by quarter {m['illegal_request_rate_by_quarter']}"
            f"  illegal emissions {m['illegal_emissions']}"
        )
        print(
            f"  {'':22s} mean reward by quarter "
            f"{m['mean_reward_by_quarter']}"
            f"  chain {'INTACT' if m['offline_verification'].get('chain', {}).get('intact') else 'BROKEN'}"
        )
    print(f"  finding  : {report['finding']}")
    print(f"  refuted  : {report['refuted_hypothesis']['was']}")
    print(f"             held = {report['refuted_hypothesis']['held']}")
    print(f"  guarantee: {report['what_the_projection_does_guarantee']}")
    if problems:
        print("\n".join("  ERROR: " + p for p in problems), file=sys.stderr)
        return 1
    print("  evidence : verified offline, without horizon_ric importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
