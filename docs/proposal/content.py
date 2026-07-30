#!/usr/bin/env python3
"""Single source of truth for the AI-RAN Alliance Call-for-Innovations proposal.

Both the .docx builder and the HTML preview render from this module, so the two
artefacts cannot drift apart.

Structure follows the six content items the Call requires — executive summary;
problem statement and market relevance; innovative solution and technical
approach; deployment feasibility; twelve-month timeline with milestones;
expected deliverables and impact — and Table II maps the document onto the five
official evaluation criteria.

Every quantitative statement is traceable to a row of docs/CLAIM_LEDGER.md on
main. Claims the ledger marks red appear in their corrected form or not at all;
the qualifications it requires are in the final section.
"""

# Shared type scale, so the .docx and the HTML preview paginate the same way.
BODY_PT = 9.0
PARA_AFTER_PT = 2.5
REF_PT = 7.0

TITLE = ("Horizon-RIC: Projection-Enforced Safety for "
         "AI-Controlled Radio Access Networks")

AUTHOR_LINE = "Daniel Foo¹, Bowen Shen², Feng Li²"
AFFIL_LINE = ("¹PreceptualAI, Singapore · "
              "²Nanyang Technological University, Singapore")

EXEC_SUMMARY = (
    "Learned control has crossed into live actuation of licensed spectrum, and "
    "the two largest RAN vendors have opened their rApp ecosystems to one "
    "another, so operators will soon actuate spectrum through planners they "
    "neither wrote nor can inspect while retaining sole liability for the "
    "licence. Every existing assurance method — constrained reinforcement "
    "learning, Lagrangian penalties, confidence gating — lowers the "
    "probability of a violation without bounding it, and binds the argument to "
    "one trained artefact. Horizon-RIC removes the constraint from the "
    "objective and places it in the topology. A projection operator, the "
    "Decision Safety Shield, sits between the planner and the southbound "
    "interface: the planner emits a proposal, never an action, and only the "
    "projected action exists downstream, so an out-of-licence emission is "
    "unrepresentable rather than unlikely. A fail-closed guard chain refuses "
    "any emission whose corrections were not audited, and every decision and "
    "refusal is appended to a hash-chained, Ed25519-signed evidence store. The "
    "property therefore survives retraining, vendor substitution and an "
    "adversarial planner. The layer is built and exercised against third-party "
    "open-source O-RAN software on real sockets — 12 of 12 policies "
    "acknowledged ENFORCED by the production ric-plt/a1 mediator driving the "
    "ric-app/hw-python xApp — and driven by ray-traced propagation for 4096 "
    "receivers. Across 8000 decisions on that geometry, 4658 requested actions "
    "were out of licence and none passed the emission boundary; 2442 were "
    "demanded by the propagation itself rather than by any attack. We request "
    "US$150,000 over twelve months to turn this into a conformance profile the "
    "Alliance can adopt."
)

KEYWORDS = ("AI-RAN assurance, O-RAN, action projection, safe reinforcement "
            "learning, spectrum compliance, tamper-evident evidence, "
            "regulatory liability")

FIG1_CAPTION = (
    "Fig. 1.  Enforcement topology. The planner is untrusted: its output is a "
    "proposal, and only the projected action P(a) leaves the non-real-time "
    "plane. The guard chain fails closed — an unaudited correction produces "
    "a refusal that never reaches the southbound interface. Southbound, the "
    "policy traverses third-party open-source software on real sockets. Every "
    "certificate and every refusal is appended to the assurance plane, where a "
    "single altered record is localised to its exact chain index."
)

FIG2_CAPTION = (
    "Fig. 2.  Measured outcomes on ray-traced propagation, all four panels from "
    "the same CI-gated runs. (a) Of 4658 out-of-licence action requests over "
    "8000 decisions, 2442 are demanded by the measured geometry alone and only "
    "2216 by a poisoned planner — the exposure is an everyday one, not an "
    "attack artefact. (b) Actions inside the emission mask but harmful by "
    "adjacent-channel leakage, adjudicated by an oracle that does not share the "
    "Shield's constants. (c) Worst emission over the same run against a 33 dBm "
    "licence. (d) Worst-case symbol-error penalty of the neural receiver "
    "relative to the certified classical baseline, over five attacks in three "
    "propagation regimes; the Shield holds it inside a 1 dB envelope. Scales are "
    "per-panel: counts and dBm never share an axis."
)

FIG3_CAPTION = (
    "Fig. 3.  Work packages, owners (DF Daniel Foo, BS Bowen Shen, FL Feng Li) "
    "and the four exit gates. Each gate is a falsifiable criterion, not a "
    "report; a gate that cannot fail is treated as no gate."
)

SECTIONS = [
    ("I.  Problem Statement and Market Relevance", [
        "A transmission that exceeds a licensed EIRP ceiling, or falls outside "
        "an assigned block, is a regulatory breach rather than a degraded KPI. "
        "Any assurance argument for AI in the RAN that terminates in a "
        "probability is therefore the wrong shape for the liability the "
        "operator carries. That mismatch, not model accuracy, is the practical "
        "reason AI is still deployed in advisory rather than actuating roles.",

        "The industry has now crossed that line anyway. In Q1 2026 an AI "
        "uplink-interference rApp was field-trialled across roughly 1500 5G and "
        "1300 4G live cells, lifting 5G SINR 27% under 10% higher uplink "
        "traffic, and was assessed at autonomy Level 3.86 on the path to Level "
        "4 [9]. Level 4 removes the human from the per-decision loop. The same "
        "trial validated a third-party rApp over R1, and in March 2026 the two "
        "largest RAN vendors opened their SMO marketplace and rApp ecosystem to "
        "one another over the same interface [10]. The consequence is "
        "structural: operators will increasingly actuate licensed spectrum "
        "through planners they did not write and cannot inspect, while "
        "retaining sole liability for the licence.",

        "Market relevance. The buyer is the operator function that owns "
        "spectrum compliance and network assurance, and — symmetrically — "
        "the vendor that needs its rApp accepted into someone else's network. "
        "Two commercial facts fall out of our own measurements rather than from "
        "market assertion. First, the exposure is routine, not exceptional: on "
        "real campus geometry 46% of measured receivers require more than the "
        "licensed EIRP to close their link, so an unshielded planner breaches "
        "the licence on an ordinary day. Second, the control is not a "
        "performance tax: wherever the optimum is feasible, enforcement "
        "surrenders 0.0 dB, which is what makes it adoptable by the RAN team "
        "rather than merely mandated at them. We have no operator pilot and no "
        "revenue; converting technical readiness into commercial validation is "
        "precisely what WP1 and WP3 are funded to do, and we name it as the "
        "weakest part of this proposal rather than dress it up.",

        "What is missing today is an assurance mechanism indifferent to which "
        "planner is installed. Constrained reinforcement learning, Lagrangian "
        "penalties and confidence gating all reduce the likelihood of a "
        "violation without bounding it, and each binds the argument to one "
        "trained artefact: retrain, re-reward or substitute a vendor and the "
        "argument must be rebuilt from scratch.",
    ]),

    ("II.  Innovative Solution and Technical Approach", [
        "Horizon-RIC takes the constraint out of the objective and puts it in "
        "the topology. A projection operator — the Decision Safety Shield "
        "— is interposed between the planner and the southbound "
        "interface. The planner's output is never an action; it is a proposal "
        "a, and only P(a) exists downstream. An out-of-licence action is not "
        "penalised, it is unrepresentable at the emission point.",

        "Three consequences follow that are architectural rather than "
        "empirical. First, the bound is worst-case rather than distributional "
        "and holds for an arbitrary planner, including a hostile one, because "
        "nothing in the argument refers to how the planner was trained. "
        "Second, it is auditable per decision: each emission carries a "
        "certificate naming the invariants that were violated, the corrections "
        "applied and the residual margins, appended to a hash-chained, "
        "Ed25519-signed store in which one altered record is localised to its "
        "exact index. Third, it is falsifiable: every published gate must be "
        "shown to fail under an injected regression, and a gate that cannot "
        "fail is treated as no gate.",

        "The novelty is not shielding as an idea. Action projection for safe "
        "reinforcement learning is established prior art [4], [5], and we claim "
        "no priority over it. What is new is its realisation as a "
        "standards-facing enforcement plane for licensed spectrum: invariants "
        "written in the operator's regulatory vocabulary rather than in a "
        "reward function — EIRP ceiling, occupied bandwidth, spectral "
        "mask, power-flux density, lawful-intercept reachability, "
        "protected-slice capacity floor — composed to a fixed point, "
        "wrapped in a fail-closed guard chain, emitted over O-RAN A1/R1 as an "
        "ordinary rApp, and evidenced in a form a third party can replay "
        "without trusting us.",

        "Enforcement path. Fig. 1 shows the decomposition. Telemetry enters the "
        "non-real-time plane; the planner proposes; the Shield projects; the "
        "guard chain refuses anything whose corrections were not audited; the "
        "A1 adapter emits the policy together with its certificate under OAuth2 "
        "or mTLS. Southbound, the policy crosses a near-real-time RIC into an "
        "xApp and onto the RAN. Certificates and refusals land independently in "
        "the assurance plane, so the record of what was refused does not depend "
        "on the component that refused it.",

        "The Shield composes invariants, each supplying an evaluate and a "
        "project method, applied until a fixed point is reached so that "
        "satisfying one cannot silently violate another. They span regulatory "
        "limits and service commitments alike. A protected-slice floor, for "
        "example, reserves capacity that no planner can reallocate, restoring "
        "it pro rata so the allocation remains a valid simplex rather than a "
        "clipped vector; where the floor is unreachable the invariant "
        "deliberately stays unsatisfied so the guard chain refuses, rather than "
        "emitting an allocation that merely appears compliant.",

        "Three distinct cryptographic mechanisms carry three distinct claims, "
        "and we keep them separate rather than calling them all signing: "
        "Ed25519 signs each per-decision safety certificate; RSA-PSS, "
        "optionally in an HSM over PKCS#11, signs model and dataset artefacts "
        "for provenance; RFC-3161 timestamps anchor the SHA-256 evidence chain "
        "to an external time authority.",

        "Relation to the learning literature. This work sits on the dynamic "
        "spectrum access line developed by two of the present authors with Lam "
        "— federated deep reinforcement learning for IoT DSA [1], its "
        "hierarchical extension [2], and privacy-preserving spectrum pricing "
        "and power control for LEO satellite IoT [3]. That trajectory answers "
        "how well, and how safely, the learning itself can be conducted. What "
        "it does not address is the emission: an agent that selects a channel "
        "and a transmit power has produced an action some component must "
        "actuate, and securing or privatising the training pipeline does not "
        "constrain it. The relationship is complementary, not competitive — "
        "those agents decide which spectrum action is best; this layer ensures "
        "whichever is chosen is one the operator is licensed to emit.",
    ]),

    ("III.  Evidence Achieved to Date", [
        "Every result below is produced against software or data Horizon-RIC "
        "does not control, which is what makes it checkable, and each is "
        "re-executed by continuous integration on a clean runner — a "
        "committed result file is an assertion about a file, and only a gate "
        "makes it evidence about the system.",

        "O-RAN interoperability. Policy emission was exercised against the "
        "O-RAN Software Community A1 simulator (12/12 accepted) and against the "
        "production Go A1 mediator ric-plt/a1 with RMR 4.9.4 driving the "
        "official ric-app/hw-python xApp: 12/12 enforceStatus = ENFORCED, "
        "confirmed by a three-witness proof across emitter, mediator and xApp "
        "logs [13]. A workflow rebuilds RMR, the mediator and the xApp from "
        "source on a clean runner and re-asserts the result.",

        "Measured propagation. All benchmarks are driven by DeepMIMO [6] ray "
        "tracing of the ASU campus at 3.5 GHz — 4096 receivers, six 100 "
        "MHz subbands, per-path angular data — bound to a checksum-pinned "
        "archive that CI rebuilds and compares field by field. Provenance is "
        "enforced rather than asserted: perturbing the measurements must move a "
        "benchmark's outputs, and four benchmarks instead reject physically "
        "inconsistent input, having independently reconstructed gains from ray "
        "geometry.",

        "Enforcement under real load. The decisive question is not whether zero "
        "out-of-licence actions were emitted but how many were attempted. Over "
        "8000 decisions, each serving a measured receiver on its measured best "
        "subband at closed-loop power against its measured path gain, 4658 "
        "requests were out of licence and 0 passed the emission boundary "
        "(Fig. 2a). The composition matters more than the total: 2216 came from "
        "a poisoned planner, but 2442 came from the real geometry alone. A "
        "further 461 actions were inside the emission mask yet harmful by "
        "adjacent-channel leakage; after projection, 0 (Fig. 2b). Legality is "
        "adjudicated by an independent emission-mask and ACLR integral that "
        "does not share the Shield's constants, so the result cannot be an "
        "artefact of the system grading its own work.",

        "Utility is not the price. Where the optimum is feasible, enforcement "
        "surrenders 0.0 dB. Because the projection maps every over-ceiling "
        "proposal onto the same executed action, the reward observed above the "
        "ceiling is constant, so a zero-data constant-power policy is already "
        "optimal there and measured learning gain is +0.000000; where an "
        "interior optimum exists the same learner gains +0.0096 to +0.0696. "
        "Both directions are enforced as gates, so no claim of AI benefit can "
        "be registered on a task incapable of showing one.",

        "The same structure bounds an unrelated threat. White-box adversarial "
        "perturbation drives the neural receiver's symbol error up to 85 times "
        "the classical baseline, yet realised error stays inside a certified "
        "1 dB envelope of that baseline for every attack in every regime, the "
        "worst case being 0.370 dB — without the layer having been coded "
        "against any of these attacks (Fig. 2d).",

        "Integrity and robust aggregation. Traffic captured over the wire while "
        "ten real attacks were executed against a live free5GC standalone core "
        "[8] is replayed through the artefact gate byte for byte: 11 of 11 "
        "probes detected or blocked, none bypassing. Against a "
        "model-replacement attack from 3 of 10 clients, FLTrust [7] attains "
        "11.97 dB coverage RMSE against clean FedAvg's 11.92 dB — parity "
        "— where Krum costs 14.46 dB. Under a Rényi accountant at a "
        "fixed certified ε = 4.1447, calibrating the clip bound on public "
        "data moves utility from 340.12 dB to 14.34 dB without altering the "
        "guarantee: the same privacy certificate at 24 times less error.",
    ]),

    ("IV.  Deployment Feasibility", [
        "Enforcement is transparent to the standard interface. The 12/12 "
        "ENFORCED result above was obtained without patching the mediator or "
        "the xApp: the layer changes what may be emitted, not how it is "
        "emitted, so an operator adopts it without renegotiating its "
        "southbound contracts.",

        "The system ships as an ordinary non-real-time rApp: container images; "
        "a Helm chart whose rendered templates are asserted by tests; EIAP and "
        "MantaRay onboarding descriptors written against the publicly "
        "documented surface; OAuth2 and mTLS credential paths; and a "
        "TimescaleDB evidence backend live-tested to SQL-level tamper "
        "detection. It is torch-free and installs on stock Python with no "
        "accelerator. Enforcement itself costs tens of microseconds per "
        "candidate action, so it sits inside a non-real-time control budget "
        "with room to spare.",

        "It is also exercised against the open stack it must coexist with: the "
        "O-RAN SC A1 mediator and hw-python xApp [13], FlexRIC for E2SM-KPM, "
        "and the Linux Foundation's OCUDU CU/DU built from source with its E2 "
        "agent confirmed. Eight verification workflows gate every change; one "
        "rebuilds the licence-gated measurements from the pinned archive and "
        "re-verifies every published figure field by field.",

        "What is not proven. No vendor platform has onboarded this — the "
        "EIAP and MantaRay packages are wire-contract tests against documented "
        "surfaces, not interoperability results. The RAN software is "
        "open-source rather than vendor equipment, the propagation is one "
        "scenario, and the OCUDU E2 join runs over a disclosed UDP transport "
        "substitution because the host kernel has no SCTP. The single external "
        "dependency this programme cannot supply for itself is access to a "
        "measured environment, which is why we ask the Alliance for it "
        "directly.",
    ]),

    ("V.  Twelve-Month Timeline and Milestones", [
        "Fig. 3 gives the schedule, owners and exit gates. Each gate is a "
        "falsifiable criterion; where a gate fails, the failure is published.",

        "WP1, months 1–4 — conformance profile. The invariant and "
        "safety-certificate schema are mapped onto O-RAN A1 and E2 and released "
        "as a draft profile. Gate G1: a second, independently written planner is "
        "shielded without modification to either the planner or the Shield.",

        "WP2, months 3–6 — second scenario and band. An independent "
        "ray-traced scenario with genuine frequency selectivity, at a different "
        "carrier. Gate G2: every published gate reproduces without retuning any "
        "constant, or the discrepancy is published.",

        "WP3, months 6–9 — measured coexistence. The modelled "
        "incumbent is replaced by a multi-cell or bench-measured interference "
        "source. Gate G3: learning gain over the zero-data constant-power "
        "policy is positive on measured data — the honest test of whether "
        "AI contributes anything in this regime.",

        "WP4, months 8–12 — audit-grade export and E2 closed loop. "
        "Evidence export in a form a spectrum authority can verify offline, and "
        "near-real-time enforcement via E2SM-RC. Gate G4: an external party "
        "replicates the headline enforcement result from the published archive "
        "without contacting us.",
    ]),

    ("VI.  Team, Budget and Execution Risk", [
        "Daniel Foo (PreceptualAI, Singapore) is the architect and implementer "
        "of the enforcement layer, the evidence chain and the O-RAN "
        "integration, and leads WP1 and WP4. Bowen Shen (Nanyang Technological "
        "University) contributes the federated DSA and satellite-IoT spectrum "
        "work this layer sits on [1], [3] and leads WP2. Dr Feng Li (Nanyang "
        "Technological University) contributes the dynamic spectrum access and "
        "secure-learning programme [1], [2], [3] and leads WP3, with oversight "
        "of the measurement methodology across all packages.",

        "We request US$150,000 over twelve months (Table I). The allocation is "
        "weighted towards the two things that convert current evidence into an "
        "adoptable profile: engineering time on the conformance and export "
        "work, and measured environments to replace the modelled ones. "
        "Independent replication is funded explicitly, because a result nobody "
        "outside the project has reproduced is not yet evidence.",

        "Execution risk. The dominant risk is WP3: if no measured interference "
        "source or bench slot can be secured, gate G3 cannot be attempted. The "
        "mitigation is sequencing — WP1, WP2 and WP4 carry no external "
        "dependency and deliver the profile, the second scenario and the export "
        "format regardless — plus a held contingency and a documented "
        "fallback to a multi-cell ray-traced incumbent, which we would publish "
        "as a weaker result rather than present as measured. The second risk is "
        "that gate G1 exposes a planner our invariant set cannot express; that "
        "outcome is a publishable finding about the profile's coverage, and the "
        "schema is versioned so it can absorb it.",
    ]),

    ("VII.  Expected Deliverables and Impact", [
        "Deliverables map onto the output types the Call anticipates. "
        "Prototype: the enforcement layer running as a non-real-time rApp with "
        "a live demonstration — policy emission through the production A1 "
        "mediator into a real xApp, including a Shield-corrected over-power "
        "proposal refused rather than emitted — which runs unattended and "
        "is available to the Alliance on request. Algorithms and models: the "
        "invariant algebra with its fixed-point composition, the protected-slice "
        "floor, and the certificate schema. Benchmarking-ready code: the harness "
        "in which every gate is proven to fail under injected regression, "
        "runnable by a third party from a pinned archive. Datasets: the pinned "
        "measurement manifests and the per-decision evidence chains, with the "
        "audit-grade export format from WP4. Standardization contributions: the "
        "invariant and certificate schema offered as a candidate assurance "
        "profile for enforceable service objectives, and the protected-slice "
        "floor as a concrete realisation of capacity an AI cannot reallocate.",

        "Impact. The effect is to move a class of deployment blocker from "
        "statistical argument to architectural guarantee. An operator can adopt "
        "a stronger and less interpretable planner without enlarging regulatory "
        "exposure, because exposure is bounded by its own control path rather "
        "than by a supplier's model — and can demonstrate that bound to a "
        "third party from the evidence chain rather than from validation "
        "statistics. A vendor gains a way to have its planner accepted into a "
        "network that will not inspect it. Published negative results are part "
        "of the deliverable, not an accident of it: the layer, schemas, evidence "
        "store and harness are developed in the open under a permissive licence "
        "with public CI, and every figure here is reproducible from a "
        "checksum-pinned archive.",
    ]),

    ("VIII.  Alignment with AI-RAN Alliance Priorities", [
        "The Alliance's problem is not whether AI can improve the RAN — the "
        "field trials have answered that — but what has to exist before an "
        "operator can let it actuate. This proposal supplies the missing "
        "artefact on the assurance side, and is deliberately shaped to be "
        "adoptable rather than merely publishable: vendor-neutral by "
        "construction, because the layer never reads the planner and so binds a "
        "member's model without that member disclosing anything about it; "
        "standards-facing, because the schema is offered as a profile rather "
        "than a product; and packaged for the exact integration path the March "
        "2026 marketplace reciprocity created [10].",

        "The timing is set by the standards calendar. ITU-R M.2160 places "
        "security, privacy and resilience among the four design principles that "
        "apply to every IMT-2030 usage scenario and admits AI as a first-class "
        "scenario driver [11]; 3GPP opened 6G study in Release 20 with "
        "normative Release 21 work expected around end-2028 [12]. An assurance "
        "profile intended to be normative must exist, be implemented, and have "
        "survived falsification before that window closes. Table II maps this "
        "document onto the five official evaluation criteria.",
    ]),

    ("IX.  Evidence Boundary and What Is Not Claimed", [
        "We state the boundary precisely, because a proposal that overstates it "
        "cannot be checked. The propagation is site-specific ray tracing, not "
        "over-the-air capture; it is a single scenario; the RAN software is "
        "open-source rather than vendor equipment; the incumbent in the "
        "coexistence analysis is modelled. Live-network, "
        "vendor-interoperability, RF-conformance and carrier-scale properties "
        "are not claimed. “No action reached the radio” would "
        "overstate what was tested; the correct form is that no non-compliant "
        "action passed the Shield's software emission boundary in the tested "
        "O-RAN control path. The structural claim holds within the enumerated "
        "invariants, correctly configured limits, trusted telemetry and the "
        "guarded adapter path — it does not cover a missing invariant, a "
        "misconfigured licence limit, compromised telemetry, or a bypass around "
        "the adapter. No 6G or IMT-2030 compliance is claimed or claimable: no "
        "such specification yet exists. Closing these gaps is what WP2 and WP3 "
        "are for.",

        "What the work most needs from members is measured environments: "
        "multi-cell measurements, a real incumbent, or bench time. Results, "
        "including failures, would be co-published.",
    ]),
]

# Table II — the five official evaluation criteria, mapped.
CRITERIA = [
    ("1.  Innovation and technical merit",
     "§II, §III — enforcement as a projection on the action space, with "
     "prior art [4], [5] conceded; every headline number re-executed by CI, "
     "and each gate proven to fail under injected regression."),
    ("2.  Relevance to AI-RAN challenges",
     "§I, §VIII — addresses the blocker that keeps AI advisory rather than "
     "actuating, at the moment autonomy Level 4 [9] and R1 marketplace "
     "reciprocity [10] transfer planner authorship away from the operator."),
    ("3.  Commercial impact and feasibility",
     "§I, §IV — 0.0 dB utility cost where the optimum is feasible; ships "
     "as an ordinary rApp with Helm, OAuth2/mTLS and onboarding descriptors. "
     "Weakest criterion: no operator pilot and no revenue yet."),
    ("4.  Strength of execution plan",
     "§V, §VI — four work packages with named owners, falsifiable exit "
     "gates, a costed budget summing to the request, and a stated dominant "
     "risk with its mitigation and fallback."),
    ("5.  Alignment with AI-RAN priorities",
     "§VII, §VIII — open, permissively licensed, pre-competitive; the "
     "invariant and certificate schema offered as a candidate assurance "
     "profile ahead of the Release 21 window [11], [12]."),
]

BUDGET = [
    ("WP1  Conformance profile, second planner shielded (D. Foo)", 30_000,
     "6 person-months engineering; draft A1/E2 profile, schema release"),
    ("WP2  Second ray-traced scenario and band (B. Shen)", 22_000,
     "4 person-months; scenario generation, gate re-execution"),
    ("WP3  Measured coexistence, real interference source (F. Li)", 18_000,
     "3 person-months; measurement design and supervision"),
    ("WP4  Audit-grade export and E2SM-RC closed loop (D. Foo)", 26_000,
     "5 person-months; export format, near-real-time enforcement"),
    ("Compute and data", 16_000,
     "GPU and CPU hours, CI minutes, ray-tracing scenario generation"),
    ("RF bench and measurement", 20_000,
     "SDR hardware, spectrum-analyser hire, chamber and bench time"),
    ("Independent replication and security review", 9_000,
     "Third-party re-execution of every gate; external review of the guard chain"),
    ("Standards work and dissemination", 5_000,
     "O-RAN and Alliance contributions, travel, open-access fees"),
    ("Contingency", 4_000,
     "2.7%; held against the WP3 measurement-access risk"),
]
BUDGET_TOTAL = 150_000

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
]

# Repository the reviewers can check every number against.
REPO_URL = "https://github.com/danielfoojunwei/preceptualai-horizon-ric"
REPO_NOTE = (
    "All code, gates and result files are public: %s. The claim ledger at "
    "docs/CLAIM_LEDGER.md states, for every number in this proposal, which CI "
    "job re-executes it and what is not yet gated." % REPO_URL
)
