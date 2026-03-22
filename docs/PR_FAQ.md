# SpectrAI PR/FAQ

*Amazon-style Press Release / Frequently Asked Questions*

---

## PRESS RELEASE

### SpectrAI Launches AI-Powered Spectrum Engine That Adapts to Radio Conditions in Real Time

**Open-source O-RAN xApp uses biologically-inspired neural networks and federated learning to deliver 63% successful transmission rates on real 5G data — with sub-4ms latency on NVIDIA ARC hardware**

**San Jose, CA — 2026** — SpectrAI today announced the general availability of its intelligent spectrum management engine, a production-grade O-RAN xApp that uses a novel class of neural networks called Liquid Time-Constant (LTC) networks to make real-time dynamic spectrum access decisions. The software is available under the Apache 2.0 open-source license, with enterprise tiers for operators seeking federated learning, dedicated support, and NVIDIA ARC optimization.

Mobile operators worldwide manage over $500 billion in cumulative spectrum assets, yet current RAN intelligent controllers rely on static rules or fixed-timescale AI models that cannot adapt their decision-making speed to changing radio conditions. When a sports stadium fills with 80,000 subscribers, or when interference patterns shift due to weather, these systems react too slowly — wasting spectrum and degrading subscriber experience.

SpectrAI solves this with Liquid Time-Constant neural networks, a continuous-time neural ODE architecture whose time constants are input-dependent. When the spectrum is volatile, SpectrAI's inference speed increases automatically. When conditions are stable, it retains longer-term memory for better predictions. No manual tuning is required.

"For the first time, we have an AI system that matches its temporal reasoning to the actual dynamics of the radio environment," said the SpectrAI founding team. "Traditional LSTM and Transformer models process spectrum data at a fixed clock rate regardless of what's happening in the air. That's like driving with a fixed reaction time whether you're on an empty highway or in rush-hour traffic."

Trained and validated on 188,000 real 5G measurements from the UCC MISL dataset (collected from an operational Irish mobile network), SpectrAI achieves:

- **63.4% successful transmission rate** — best among all tested architectures including SAC-LSTM, SAC-LFM, and PPO-LSTM
- **36.6% collision rate** — lowest across all baselines, reducing interference with primary users
- **3.14ms mean inference latency** (3.96ms P99) — well within the 10ms Near-RT RIC control loop budget
- **0.634 spectral efficiency** — 1.4% improvement over LSTM baselines

SpectrAI also introduces Hybrid Federated Aggregation, a novel federated learning protocol designed specifically for LTC networks. The system separates model parameters into structural weights (shared globally across all cell sites) and time-constant weights (personalized per site). This means an operator can train a single federated model across thousands of cell sites, with each site retaining its own temporal adaptation characteristics — a dense urban site in Manhattan does not inherit the same dynamics as a rural highway cell in Kansas.

"Privacy-preserving AI for telecom is not optional — it's regulatory," said the team. "Hybrid Federated Aggregation means raw spectrum data never leaves the cell site. Only model weight updates are transmitted. And because we personalize the time constants, we don't sacrifice per-site performance for the sake of a global average."

The software runs natively on NVIDIA Aerial RAN CoProcessors (ARC), supporting both ARC-Compact (L4) for cell-site inference and ARC-Pro (Blackwell RTX PRO) for training and federated aggregation. It integrates with any O-RAN-compliant Near-RT RIC via standard E2 interfaces, and exposes a gRPC API with Prometheus metrics for operational monitoring.

SpectrAI is available today at [github.com/spectrai-project/spectrai](https://github.com/spectrai-project/spectrai). Enterprise licenses with SLA-backed support, federated learning orchestration, and NVIDIA ARC optimization are available by contacting sales@spectrai.ai.

---

## FREQUENTLY ASKED QUESTIONS

### External FAQ (Customers & Partners)

**Q: What is SpectrAI?**

SpectrAI is an AI-powered O-RAN xApp (application for the RAN Intelligent Controller) that performs dynamic spectrum access — deciding which radio channels to use and when — using a novel neural network architecture called Liquid Time-Constant networks. It runs as a standard xApp on any O-RAN-compliant Near-RT RIC and makes spectrum decisions in under 4 milliseconds.

**Q: Who is SpectrAI for?**

SpectrAI is designed for:
- **Mobile network operators** (MNOs) managing spectrum across thousands of cell sites
- **Neutral host providers** sharing spectrum across multiple operators in venues
- **Private 5G network operators** (enterprise, defense, mining, manufacturing)
- **NVIDIA ARC ecosystem partners** building AI-native RAN applications
- **Telecom system integrators** deploying O-RAN solutions

**Q: How is SpectrAI different from existing spectrum management solutions?**

Three architectural differentiators that no competitor offers:

1. **Continuous-time adaptation:** LTC networks adjust their temporal reasoning speed based on current spectrum conditions. Fixed-timescale models (LSTM, Transformer) cannot do this.
2. **Hybrid Federated Aggregation:** Our novel FL protocol shares feature-extraction weights globally while keeping time-constant weights personalized per site. This is the only production system that preserves per-site temporal dynamics during federated learning.
3. **NVIDIA ARC native:** Purpose-built for the Aerial RAN CoProcessor, with TensorRT optimization and sub-10ms inference. As NVIDIA's dApp ecosystem matures, SpectrAI will access raw L1 PHY data for even faster decisions.

**Q: What data does SpectrAI need?**

SpectrAI processes standard 3GPP KPM (Key Performance Measurements) delivered via O-RAN E2 interface:
- RSRP (Reference Signal Received Power)
- RSRQ (Reference Signal Received Quality)
- SNR (Signal-to-Noise Ratio)
- CQI (Channel Quality Indicator)
- RSSI (Received Signal Strength Indicator)

No proprietary data feeds are required. If your RAN exports E2 indications, SpectrAI can consume them.

**Q: How much does it cost?**

SpectrAI uses an open-core model:
- **Community Edition:** Free and open source under Apache 2.0. Includes the full LTC encoder, SAC agent, training pipeline, ONNX export, and gRPC server.
- **Pro:** Starting at $120/cell-site/year. Adds enterprise support (4-hour SLA), pre-trained models, NVIDIA ARC optimization, and Grafana dashboards.
- **Enterprise:** Starting at $300/cell-site/year. Adds Hybrid Federated Learning orchestration, custom model training, dedicated support, and on-premise deployment.

Volume discounts are available for deployments over 5,000 cell sites. See [PRICING.md](PRICING.md) for details.

**Q: What about data privacy? Does SpectrAI send spectrum data to the cloud?**

No. SpectrAI's federated learning architecture is privacy-preserving by design:
- Raw spectrum data never leaves the cell site
- Only model weight updates (tensors, not data) are transmitted to the FL aggregator
- The FL aggregator can run on-premise within the operator's network
- Time-constant personalization means the system works well without centralizing data
- Compliant with GDPR, telecom data sovereignty requirements, and ORAN Alliance security specifications

**Q: What hardware do I need?**

For inference only:
- Any x86 server with a CPU can run SpectrAI via ONNX Runtime
- NVIDIA L4 GPU (ARC-Compact) recommended for production latency targets

For training and federated learning:
- NVIDIA GPU with 8GB+ VRAM (RTX 4070 or better, ARC-Pro for production)
- Federated aggregator: any server with 16GB+ RAM

**Q: How long does deployment take?**

- **Simulated evaluation:** 5 minutes (pip install, run training script)
- **Lab POC with real data:** 1-2 weeks (data integration, model training, validation)
- **Production deployment on RIC:** 4-8 weeks (RIC integration, E2 subscription, monitoring setup)
- **Multi-site federated deployment:** 8-12 weeks (FL infrastructure, per-site validation)

**Q: Is SpectrAI O-RAN compliant?**

Yes. SpectrAI implements:
- Standard xApp registration via ricxappframe
- E2SM-KPM for receiving spectrum measurements
- E2SM-RC for sending control decisions to gNodeBs
- A1 policy interface for non-RT RIC integration (roadmap)
- Standard gRPC health checks and Prometheus metrics

**Q: What is the performance impact on the RAN?**

SpectrAI's inference path adds 3.14ms mean latency (3.96ms P99) to the control loop. The Near-RT RIC specification allows up to 10ms for xApp processing. SpectrAI uses less than 40% of this budget, leaving headroom for other xApps in the pipeline.

---

### Internal FAQ (Investors & Team)

**Q: What is the total addressable market (TAM)?**

The market opportunity spans multiple segments:

| Market | 2025 | 2030 | CAGR |
|---|---|---|---|
| RIC Platform & xApp | $0.67B | $7.09B | 60% |
| AI-RAN (broader) | $2.96B | $37.19B (2035) | 29% |
| Spectrum Management Software | $8.2B | $14.1B | 11.5% |

Our serviceable addressable market (SAM) is the AI-RAN xApp segment: operators deploying AI-driven xApps on O-RAN RICs. We estimate $2.1B SAM by 2028.

Our serviceable obtainable market (SOM) targets 50 operator deployments by Year 3, with an average contract value of $1.2M/year, yielding $60M ARR.

**Q: What are the unit economics?**

- **Cost to serve per cell site:** ~$8-15/year (cloud compute for FL aggregation, model updates, support infrastructure)
- **Revenue per cell site:** $120-300/year (Pro and Enterprise tiers)
- **Gross margin:** 88-95% at scale
- **Customer acquisition cost:** Estimated $200K-500K per operator (long enterprise sales cycle, POC-driven)
- **Lifetime value:** $3.6M+ per Tier-1 operator (3-year contract, 10K+ sites)
- **LTV/CAC ratio:** 7-18x

**Q: What is the go-to-market strategy?**

Three-phase approach:

1. **Phase 1 (Now - Q4 2026): Open Source + Ecosystem**
   - Release core engine under Apache 2.0
   - Join NVIDIA AI-RAN Alliance as a partner
   - Build developer community via tutorials, conference talks, academic partnerships
   - Target: 500+ GitHub stars, 50+ contributors, 10+ academic citations

2. **Phase 2 (2027): Operator POCs**
   - Partner with NVIDIA to offer joint POCs to their Aerial customers
   - Target 5-10 Tier-1/Tier-2 operator POCs
   - Convert POCs to paid Pro/Enterprise contracts
   - Pricing validated through POC value demonstration

3. **Phase 3 (2028+): Enterprise Scale**
   - Direct enterprise sales to operators with 10K+ cell sites
   - Channel partnerships with system integrators (Accenture, TCS, Infosys)
   - OEM licensing to RIC platform vendors
   - Geographic expansion (NA, EMEA, APAC)

**Q: Why now? What has changed?**

Four structural shifts make this the right time:

1. **NVIDIA ARC creates a new market.** By open-sourcing Aerial and releasing ARC hardware, NVIDIA is creating a GPU-accelerated RAN ecosystem that needs AI-native software. SpectrAI is purpose-built for this platform.

2. **O-RAN deployments are real.** Tier-1 operators (Vodafone, DT, Rakuten, Dish) have deployed RICs in production. The xApp market is no longer theoretical.

3. **Spectrum economics are shifting.** CBRS, AFC, and the FCC's National Spectrum Strategy are driving dynamic spectrum sharing. Static allocation is ending.

4. **Federated learning is production-ready.** Libraries like Flower, PySyft, and FedML have matured. The infrastructure to deploy FL at scale exists.

**Q: Why are you the right team?**

- Deep expertise in both reinforcement learning and telecommunications
- Published research on LTC networks for spectrum management (first in the field)
- Working system trained on real 5G data (not just simulations)
- Production engineering experience: gRPC, Docker, Prometheus, ONNX, TensorRT
- Understanding of operator procurement and telecom go-to-market

**Q: What are the biggest risks?**

| Risk | Mitigation |
|---|---|
| **Operator sales cycles (12-18 months)** | Open-source-first GTM reduces friction; NVIDIA partnership provides channel access |
| **RIC platform fragmentation** | Standard E2/A1 interfaces; ricxappframe compatibility; vendor-agnostic design |
| **Performance delta is small (1-2%)** | Compound effect at scale (10K+ sites); federated learning benefit grows with fleet size; continuous-time advantage increases with deployment heterogeneity |
| **NVIDIA dependency** | Core engine is hardware-agnostic (runs on CPU/any GPU); ARC is an optimization, not a requirement |
| **Competitor response (Mavenir, Nokia)** | Patent-pending hybrid FL; 18+ month head start on LTC for spectrum; open-source community moat |
| **Regulatory uncertainty** | Privacy-preserving FL design is regulatory-friendly; on-premise deployment option |

**Q: What are the key milestones for the next 12 months?**

1. **Q2 2026:** Open-source release, 100+ GitHub stars, NVIDIA AI-RAN Alliance membership
2. **Q3 2026:** TensorRT optimization complete, ARC benchmark published, first academic paper accepted
3. **Q4 2026:** First operator POC signed (target: European Tier-1), 500+ GitHub stars
4. **Q1 2027:** POC results published, second operator POC, seed round ($3-5M) if venture-backed
5. **Q2 2027:** First paid Pro contract, 3+ operator POCs active

**Q: What funding do you need?**

Pre-seed / bootstrapping phase:
- NVIDIA Inception program (compute credits, go-to-market support)
- Academic grants (NSF, Horizon Europe — spectrum management is a funded research area)
- Revenue from early Pro/Enterprise contracts

If venture-backed (Seed round, $3-5M):
- 18-month runway
- Hire: 2 ML engineers, 1 telecom systems engineer, 1 DevRel, 1 enterprise sales
- Infrastructure: NVIDIA DGX access for training, multi-site FL testbed
- Target: 3+ paying operator customers, $500K+ ARR by end of runway

**Q: What is the exit potential?**

Comparable transactions:
- **Cellwize acquired by Qualcomm for $350M** (2022) — SON/RAN optimization, no AI
- **Cohere Technologies raised $46M Series D** — massive MIMO, spectral efficiency
- **AirHopAI** — 1.5M cells managed, acquired by Amdocs

Potential acquirers: NVIDIA (ARC ecosystem), Qualcomm, Samsung, VMware/Broadcom (RIC), Mavenir, Ericsson, Nokia.

At $50-100M ARR, a 10-20x revenue multiple implies $500M-2B valuation, consistent with telecom infrastructure software transactions.
