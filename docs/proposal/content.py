#!/usr/bin/env python3
"""Single source of truth for the AI-RAN Alliance Call-for-Innovations proposal.

Both the .docx builder and the HTML preview render from this module, so the two
artefacts cannot drift apart.

Layout contract: **two pages of body, references on page 3.** A hard page break
precedes the reference list and ``validate_docx.py`` asserts the page count and
the page the references start on, so a section that grows past the limit fails
the build rather than silently making a three-page document four.

Structure follows the six content items the Call requires — executive summary;
problem statement and market relevance; innovative solution and technical
approach; deployment feasibility; twelve-month timeline with milestones;
expected deliverables and impact — and Table II maps the document onto the five
official evaluation criteria.

Every quantitative statement is traceable to a row of docs/CLAIM_LEDGER.md.
Seven statements an earlier revision made are corrected here rather than
carried forward; ``audit/verify_proposal_claims.py`` gates each correction
against the result file that established it.
"""

# Shared type scale, so the .docx and the HTML preview paginate the same way.
BODY_PT = 7.2
PARA_AFTER_PT = 1.8
REF_PT = 6.6

TITLE = ("Horizon-RIC: Projection-Enforced Safety for "
         "AI-Controlled Radio Access Networks")

AUTHOR_LINE = "Daniel Foo¹, Bowen Shen², Feng Li²"
AFFIL_LINE = ("¹PreceptualAI, Singapore · "
              "²Nanyang Technological University, Singapore")

EXEC_SUMMARY = (
    "Learned control has crossed into live actuation of licensed spectrum, and "
    "vendors have opened their rApp ecosystems to one another, so operators "
    "will soon actuate spectrum through planners they neither wrote nor can "
    "inspect while keeping sole liability. Constrained reinforcement learning "
    "lowers the probability of a violation without bounding it. Horizon-RIC "
    "moves the constraint out of the objective and into the topology: a "
    "projection operator, the Decision Safety Shield, sits between the planner "
    "and the southbound interface, so the planner emits a proposal, never an "
    "action, and only the projected action exists downstream. An out-of-licence "
    "emission is unrepresentable rather than unlikely, every decision and "
    "refusal is appended to a hash-chained Ed25519-signed store, and the "
    "property survives retraining, vendor substitution and a hostile planner. "
    "Two independent results: 12 of 12 policies ENFORCED by the production "
    "ric-plt/a1 mediator on real sockets; and, on ray-traced propagation for "
    "4096 receivers, 4658 of 8000 requested actions out of licence with none "
    "passing the emission boundary — 2442 demanded by the propagation itself, "
    "not by any attack. We ask for funding against the work packages below, and "
    "for what funding cannot buy: measured environments, a member's planner to "
    "shield, and a working-group route for the profile."
)

KEYWORDS = ("AI-RAN assurance, O-RAN, action projection, safe reinforcement "
            "learning, spectrum compliance, tamper-evident evidence, "
            "regulatory liability")

# One figure, in the full-width band between III and IV.
FIG1_CAPTION = (
    "Fig. 1.  Measured outcomes on ray-traced propagation, every panel plotted "
    "from a CI-gated result file. (a) Of 4658 out-of-licence requests over 8000 "
    "decisions, 2442 are demanded by the measured geometry alone and only 2216 "
    "by a poisoned planner. (b) Actions inside the emission mask but harmful by "
    "adjacent-channel leakage, adjudicated by an oracle that does not share the "
    "Shield's constants. (c) Peak requested EIRP against the 33 dBm licence: "
    "52.0 dBm is the 46.0 dBm transmit-power request plus the declared 6.0 dBi "
    "antenna gain, which is how the benchmark defines EIRP. (d) Worst-case "
    "neural-receiver symbol-error penalty against the certified classical "
    "baseline, five attacks in three regimes, held inside 1 dB."
)

SECTIONS = [
    ('I.  Problem Statement and Market Relevance', [
        'A transmission that exceeds a licensed EIRP ceiling, or falls outside an assigned block, is a regulatory breach rather than a degraded KPI. An assurance argument that terminates in a probability is the wrong shape for the liability the operator carries — that mismatch, not model accuracy, is why AI is still advisory rather than actuating. The industry has crossed that line anyway: in Q1 2026 an AI rApp was field-trialled across roughly 2800 live cells at autonomy Level 3.86 [9], and in March 2026 the two largest RAN vendors opened their rApp ecosystems to one another over R1 [10]. Operators will increasingly actuate spectrum through planners they did not write and cannot inspect, while retaining sole liability.',

        "Market relevance. The buyer is the operator function owning spectrum compliance, and the vendor needing its rApp accepted elsewhere. On real campus geometry 46% of measured receivers demand more than the licensed EIRP to close their link, so an unshielded planner breaches the licence on an ordinary day. Where the licence does not bind, enforcement surrenders nothing; where it binds, the cost is measured rather than hidden — 0.20 of served fraction against a 40 W small-cell anchor. We have no operator pilot and no revenue, and name that as this proposal's weakest part.",

    ]),

    ('II.  Innovative Solution and Technical Approach', [
        "The Decision Safety Shield is a projection operator interposed between the planner and the southbound interface. The planner's output is never an action; it is a proposal a, and only P(a) exists downstream, so an out-of-licence action is not penalised — it is unrepresentable at the emission point.",

        'Three consequences are architectural rather than empirical. The bound is worst-case rather than distributional and holds for an arbitrary, including hostile, planner, because nothing in the argument refers to how the planner was trained. It is auditable per decision: each emission carries a certificate naming the invariants violated, the corrections applied and the residual margins, appended to a hash-chained Ed25519-signed store in which one altered record is localised to its index. And it is falsifiable: a gate that cannot fail is no gate.',

        "The novelty is not shielding as an idea — action projection is prior art [4], [5] — but its realisation as a standards-facing enforcement plane for licensed spectrum, with invariants in the operator's regulatory vocabulary rather than in a reward function — EIRP ceiling, occupied bandwidth, spectral mask, power-flux density, lawful-intercept reachability, protected-slice capacity floor — each supplying an evaluate and a project method, composed to a fixed point so satisfying one cannot silently violate another. Where the slice floor is unreachable the invariant stays unsatisfied so the guard chain refuses rather than emitting an allocation that merely appears compliant. This complements the dynamic spectrum access line of two of the present authors with Lam [1], [2], [3]: those agents choose the action, this one ensures the operator may emit it.",

    ]),

    ('III.  Evidence Achieved to Date', [
        "O-RAN interoperability. Policy emission was exercised against the O-RAN Software Community A1 simulator (12/12 accepted) and the production Go A1 mediator ric-plt/a1 with RMR 4.9.4 driving the official ric-app/hw-python xApp: 12/12 enforceStatus = ENFORCED on a three-witness proof across emitter, mediator and xApp logs [13]. That run replays authored telemetry; the ray-traced measurements below are a separate evidence base. Both southbound directions are gated: the certificate rides on the A1 policy as an optional envelope carrying its digest and Ed25519 signature, while E2SM-RC payloads built from the O-RAN v1.03 standard ASN.1 are independently decoded by FlexRIC's C codec and cannot be encoded from a refused decision — construction and verified decode, not delivery.",

        "Enforcement under real load. Propagation benchmarks are driven by DeepMIMO [6] ray tracing of the ASU campus at 3.5 GHz — 4096 receivers, six subbands across 100 MHz — bound to a checksum-pinned archive; perturbing the measurements must move a benchmark's outputs, and three instead reject physically inconsistent input outright. The decisive question is not whether zero out-of-licence actions were emitted but how many were attempted. Over 8000 decisions, each serving a measured receiver on its best subband at closed-loop power against its measured path gain, 4658 requests were out of licence and 0 passed the emission boundary; 2216 came from a poisoned planner, 2442 from the real geometry alone. A further 461 actions sat inside the emission mask yet were harmful by adjacent-channel leakage; after projection, 0. Legality is adjudicated by an independent emission-mask and ACLR integral that does not share the Shield's constants.",

        "Utility, stated conditionally. Because the projection maps every over-ceiling proposal onto the same executed action, reward above the ceiling is constant, so a zero-data constant-power policy is already optimal there and measured learning gain is +0.000000; where an interior optimum exists a tabular learner gains +0.0096 to +0.0696, while a limited-capacity value model can forfeit up to 52% of reward, which the same study reports rather than suppresses. Both directions are gates, so no claim of AI benefit can be registered on a task incapable of showing one. The same structure bounds unrelated threats without having been coded against them: adversarial perturbation drives the neural receiver's symbol error to 85 times the classical baseline, yet realised error stays inside a certified 1 dB envelope, worst case 0.370 dB; captured traffic from ten real attacks on a live free5GC core [8] is caught 11 of 11; and FLTrust [7] holds coverage RMSE at parity with clean FedAvg.",

    ]),

    ('IV.  Deployment Feasibility', [
        'Enforcement is transparent to the standard interface: the 12/12 result needed no patch to the mediator or the xApp, because the layer changes what may be emitted, not how. It ships as an ordinary non-real-time rApp — container images, a Helm chart with template-rendering tests, EIAP and MantaRay onboarding descriptors, OAuth2 and mTLS, a TimescaleDB evidence backend live-tested to SQL-level tamper detection — is torch-free, and costs 27 µs per candidate action on average. Eleven verification workflows run on every pull request.',

        'What is not proven. No vendor platform has onboarded this; the RAN software is open-source rather than vendor equipment; the propagation is one scenario; and the OCUDU CU/DU E2 join — all three of CU-CP, CU-UP and DU, with RAN function 3 accepted — runs over a disclosed UDP transport substitution because the host kernel has no SCTP.',

    ]),

    ('V.  Twelve-Month Timeline and Milestones', [
        'Five work packages, each with a named owner and a falsifiable exit gate; where a gate fails, the failure is published. WP1, months 1–4, conformance profile (Foo). G1: a second, independently written planner is shielded without modifying either side, with the SHA-256 of both pinned, so editing either to make the numbers agree fails the gate rather than passing it. WP2, months 3–6, second ray-traced scenario (Shen). G2: every published gate reproduces at a different carrier without retuning any constant.',

        'WP3, months 6–9, measured coexistence (Li). G3: with the modelled incumbent replaced by a bench-measured source, learning gain over the zero-data constant-power policy is positive on measured data. WP4, months 8–12, audit-grade export and a delivered E2 control action (Foo). G4: a RIC Control Request derived from a signed certificate is accepted by an E2 node, and one derived from a refused decision cannot be constructed at all. WP5, months 10–12, replication and standardisation (Foo, Li). G5: an external party reproduces the headline result without contacting us, and the profile goes to the Alliance working group and O-RAN WG11.',

    ]),

    ('VI.  Team, and What We Ask For', [
        'Daniel Foo (PreceptualAI) is the architect and implementer of the enforcement layer, the evidence chain and the O-RAN integration. Bowen Shen (Nanyang Technological University) contributes the federated DSA and satellite-IoT spectrum work this layer sits on [1], [3]. Dr Feng Li (Nanyang Technological University) contributes the dynamic spectrum access and secure-learning programme [1], [2], [3], with oversight of measurement methodology.',

        "Table I separates the two kinds of support, because they are not interchangeable. The first is funding for WP2 through WP5; no figure is quoted on purpose, because the right amount depends on which lab the Alliance can open. The second is access: the endorsed labs with Keysight Technologies, Northeastern University, Singapore University of Technology and Design and VIAVI Solutions, Data-for-AI, Test Methodology [15], and an AI-for-RAN working group [14]. The dominant risk is WP3 — without a measured interference source G3 cannot be attempted. WP1, WP2, WP4 and WP5 carry no external dependency, and the fallback is a multi-cell ray-traced incumbent, published as the weaker result it is.",

    ]),

    ('VII.  Expected Deliverables and Impact', [
        "Deliverables map onto the output types the Call anticipates. Prototype: the enforcement layer as a non-real-time rApp with a live demonstration — policy emission through the production A1 mediator into a real xApp, with an over-power proposal corrected to the EIRP ceiling before emission and a non-physical proposal refused outright — running unattended. Algorithms and models: the invariant algebra, the protected-slice floor and the certificate schema. Benchmarking-ready code: the gate harness, with the injected-regression discipline extended across it under WP5. Datasets: the pinned manifests and per-decision evidence chains, with WP4's export format. Standardization: the schema offered as a candidate assurance profile. The impact: a deployment blocker moves from statistical argument to architectural guarantee, so an operator can adopt a stronger, less interpretable planner without enlarging regulatory exposure.",

    ]),

    ('VIII.  Alignment with AI-RAN Alliance Priorities', [
        "AI-for-RAN's remit is spectral and energy efficiency [14] — precisely where a learned planner pushes against a licence, and therefore where an enforcement layer earns its place. Our falsification rule is offered to Test Methodology, because a metric no one can fail is not a metric, and the pinned-manifest tooling to Data-for-AI [15]. ITU-R M.2160 places security, privacy and resilience among the design principles for every IMT-2030 usage scenario [11], and 3GPP opened 6G study in Release 20 with Release 21 expected around end-2028 [12]: a profile intended to be normative must exist, be implemented and have survived falsification before that window closes.",

    ]),

    ('IX.  Evidence Boundary and What Is Not Claimed', [
        "The propagation is site-specific ray tracing, not over-the-air capture; it is a single scenario; the incumbent is modelled. No E2 control request has been delivered to a node and executed, and nothing has yet been submitted to a standards body. Two of twenty gates carry a re-runnable injected-regression proof today; extending that across the harness is WP5 work, scheduled rather than claimed. “No action reached the radio” would overstate what was tested: the correct form is that no non-compliant action passed the Shield's software emission boundary in the tested control path, within the enumerated invariants and configured limits.",

    ]),

]

# Table II — the five official evaluation criteria, mapped.
CRITERIA = [
    ("1.  Innovation and technical merit",
     "§II, §III — enforcement as a projection on the action space, prior art "
     "[4], [5] conceded, every headline number re-executed by CI."),
    ("2.  Relevance to AI-RAN challenges",
     "§I — the blocker that keeps AI advisory, as autonomy Level 4 [9] and R1 "
     "reciprocity [10] move planner authorship away from the operator."),
    ("3.  Commercial impact and feasibility",
     "§I, §IV — no utility cost where the licence does not bind, a measured "
     "cost where it does. Weakest criterion: no operator pilot, no revenue."),
    ("4.  Strength of execution plan",
     "§V, §VI — five packages, named owners, falsifiable gates, four free of "
     "external dependency, dominant risk and fallback stated."),
    ("5.  Alignment with AI-RAN priorities",
     "§VII, §VIII — open and pre-competitive; schema to AI-for-RAN [14], "
     "falsification rule to Test Methodology, provenance tooling to "
     "Data-for-AI [15]."),
]

# Table I — what we ask for, split into funding and access, with what goes
# back in return. Ordered by how much each changes the evidence, not by how
# easy it is to grant. No monetary figure appears here by design: see VI.
ALLIANCE_ASKS = [
    ("FUND  Measured-coexistence campaign (WP3)",
     "Person-months and instrumentation to replace the modelled incumbent with "
     "a bench-measured interference source — our weakest input, and the only "
     "gate we cannot attempt alone. Return: scripts, manifests and raw results, "
     "published whether or not they favour us."),

    ("FUND  Second ray-traced scenario (WP2)",
     "Person-months, storage and compute for a scenario with genuine frequency "
     "selectivity; the band half is done — n78 to n257, 24.4 GHz apart, no "
     "constant retuned. Return: a report naming any constant that proves to "
     "have been tuned rather than derived."),

    ("FUND  A delivered E2 control action (WP4)",
     "Person-months to carry the E2SM-RC payloads over a real E2 termination to "
     "a node with a UE attached; the offline-verifiable export half is built "
     "and passing. Return: the E2 binding, offered to the profile."),

    ("FUND  Independent replication and security review (WP5)",
     "An externally commissioned re-execution of every gate and a review of the "
     "guard chain by someone with no stake in the answer. The harness is built, "
     "so this funds a reviewer, not a harness."),

    ("ACCESS  Bench time in an endorsed lab",
     "Funding alone does not buy this. The endorsed labs exist for data "
     "creation and benchmarking [15]; WP3 needs one, with a real interference "
     "source. Everything downstream is built and gated, so the grant converts "
     "directly into a result. Return: every measurement artefact, and "
     "co-authorship."),

    ("ACCESS  A member's planner to shield, under NDA or in the open",
     "Turns G1 from a demonstration on a planner we wrote ourselves into an "
     "industrial result. The layer never reads the planner, so a member "
     "discloses nothing, and a failure is the most useful outcome available: it "
     "names an invariant the profile is missing. Return: a per-decision "
     "compliance record, at no change to the planner."),

    ("ACCESS  An operator willing to run the layer in shadow mode",
     "Certificates and refusals emitted, nothing actuated. Closes the one "
     "criterion on which we are honestly weak. Return: an audit-grade evidence "
     "chain over the operator's own control path."),

    ("ACCESS  A host with SCTP and an E2 node with a UE attached",
     "G4's delivery half is blocked on a host, not person-months: our E2 join "
     "runs over a disclosed UDP substitution because the kernel has no SCTP. "
     "Return: the delivered control action, and the substitution retired."),
]

REFERENCES = [
    "F. Li, B. Shen, J. Guo, K.-Y. Lam, G. Wei, and L. Wang, “Dynamic "
    "spectrum access for Internet-of-Things based on federated deep "
    "reinforcement learning,” IEEE Trans. Veh. Technol., vol. 71, no. 7, "
    "pp. 7952–7956, 2022.",

    "S. Zhang, K.-Y. Lam, B. Shen, L. Wang, and F. Li, “Dynamic spectrum "
    "access for Internet-of-Things with hierarchical federated deep "
    "reinforcement learning,” Ad Hoc Networks, vol. 149, 103257, 2023.",

    "B. Shen, K.-Y. Lam, F. Li, and L. Wang, “Privacy-aware spectrum "
    "pricing and power control optimization for LEO satellite "
    "Internet-of-Things,” IEEE Trans. Wireless Commun., 2025.",

    "M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. "
    "Topcu, “Safe reinforcement learning via shielding,” in Proc. "
    "AAAI, 2018.",

    "G. Dalal, K. Dvijotham, M. Vecerik, T. Hester, C. Paduraru, and Y. Tassa, "
    "“Safe exploration in continuous action spaces,” "
    "arXiv:1801.08757, 2018.",

    "A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for "
    "millimeter wave and massive MIMO applications,” in Proc. Inf. Theory "
    "and Applications Workshop, 2019.",

    "X. Cao, M. Fang, J. Liu, and N. Z. Gong, “FLTrust: Byzantine-robust "
    "federated learning via trust bootstrapping,” in Proc. NDSS, 2021.",

    "C. Coldwell et al., “Machine learning 5G attack detection in "
    "programmable logic,” in Proc. IEEE Globecom Workshops, 2022. Dataset: "
    "5GAD-2022, Idaho National Laboratory, doi: 10.11578/dc.20220811.1.",

    "Ericsson, “Ericsson and KDDI succeed in AI uplink optimization field "
    "trial, advancing toward Autonomous Networks Level 4,” press release, "
    "Feb. 2026.",

    "Nokia, “Nokia and Ericsson strengthen cooperation to accelerate "
    "towards Autonomous Networks,” press release, 1 Mar. 2026. Reciprocal "
    "SMO Marketplace and rApp Ecosystem membership over R1.",

    "Recommendation ITU-R M.2160-0, “Framework and overall objectives of "
    "the future development of IMT for 2030 and beyond,” ITU-R, Nov. 2023.",

    "3GPP, “Release 20,” www.3gpp.org/specifications-technologies/"
    "releases/release-20.",

    "O-RAN Software Community, “RIC platform A1 mediator (ric-plt/a1) and "
    "ric-app/hw-python,” gerrit.o-ran-sc.org.",

    "AI-RAN Alliance working groups: AI-for-RAN, AI-and-RAN and AI-on-RAN. "
    "[Online]. Available: ai-ran.org",

    "AI-RAN Alliance, “Data-for-AI” and “Test Methodology” "
    "initiatives, and four AI-RAN Alliance-endorsed labs with Keysight "
    "Technologies, Northeastern University, Singapore University of Technology "
    "and Design, and VIAVI Solutions, for data creation and benchmarking. "
    "Announced at MWC 2025; 75 members across 17 countries at that date.",
]

# Repository the reviewers can check every number against.
REPO_URL = "https://github.com/danielfoojunwei/preceptualai-horizon-ric"
REPO_NOTE = (
    "All code, gates and result files are public: %s. The claim ledger at "
    "docs/CLAIM_LEDGER.md states, for every number in this proposal, which CI "
    "job re-executes it and what is not yet gated." % REPO_URL
)
