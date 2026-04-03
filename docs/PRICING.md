# PreceptualAI Pricing Model

---

## Pricing Philosophy

PreceptualAI follows an **open-core** model. The complete SAC-LTC agent, training pipeline, ONNX export, gRPC inference server, and Prometheus monitoring are free and open source under the Apache 2.0 license. This is not a crippled demo — it is a production-grade xApp that can run on a single cell site or in a lab environment with zero cost.

Paid tiers add the capabilities that matter at fleet scale: federated learning across thousands of cell sites, per-site model personalization, enterprise support, and NVIDIA ARC optimization. The pricing is designed so that the free tier gets operators to evaluate and validate PreceptualAI, and the paid tiers unlock the compounding value that comes from multi-site deployment.

**Guiding principles:**
1. The free tier must be genuinely useful, not a marketing ploy
2. Pricing must be simple: per cell site, per year, no hidden fees
3. The ROI must be obvious: spectrum efficiency gains should exceed the cost by 10x or more
4. Volume discounts reward commitment and reduce per-unit cost at scale

---

## Tier Breakdown

### Community Edition — Free (Apache 2.0)

**For:** Researchers, individual developers, operators evaluating PreceptualAI, small private 5G deployments.

| Capability | Included |
|---|---|
| LTC Encoder + SAC Agent | Yes |
| Simulated DSA Environment | Yes |
| Real 5G Data Training (bring your own data) | Yes |
| ONNX Model Export | Yes |
| gRPC Inference Server | Yes |
| Prometheus Metrics | Yes |
| Docker Packaging | Yes |
| PyTorch / ONNX Runtime Inference | Yes |
| Community Support (GitHub Issues) | Yes |
| **Max cell sites** | **Unlimited (single-site inference only)** |

**What's not included:** Federated learning, TensorRT optimization, pre-trained models, enterprise support, multi-site management.

---

### Pro — $120 / cell site / year

**For:** Operators running PreceptualAI at 100-10,000 cell sites who need production support and GPU optimization but can manage their own FL infrastructure.

| Capability | Included |
|---|---|
| Everything in Community | Yes |
| Pre-trained Spectrum Models (5G bands) | Yes |
| TensorRT Optimization for NVIDIA ARC | Yes |
| Grafana Dashboard Templates | Yes |
| Priority Security Patches | Yes |
| Enterprise Support (4-hour response SLA, business hours) | Yes |
| Email + Slack Support Channel | Yes |
| Quarterly Model Updates | Yes |
| License | Commercial (per-site) |

**Pricing:**

| Cell Sites | Per-Site/Year | Annual Total |
|---|---|---|
| 100-999 | $120 | $12K-120K |
| 1,000-4,999 | $120 | $120K-600K |
| 5,000-9,999 | $108 (10% discount) | $540K-1.08M |
| 10,000-19,999 | $96 (20% discount) | $960K-1.92M |
| 20,000+ | Custom | Contact sales |

**Minimum commitment:** 100 cell sites, 1-year term.

---

### Enterprise — $300 / cell site / year

**For:** Tier-1 and Tier-2 operators deploying PreceptualAI across 10,000+ cell sites who need federated learning, per-site personalization, and dedicated support.

| Capability | Included |
|---|---|
| Everything in Pro | Yes |
| Hybrid Federated Learning Orchestration | Yes |
| Per-site Tau Personalization | Yes |
| FL Aggregation Server (on-premise) | Yes |
| FL Management Console (web UI) | Yes |
| Custom Model Training (operator data) | Yes |
| Custom KPM Integration | Yes |
| Multi-site Deployment Tooling (Helm charts, Ansible) | Yes |
| Dedicated Support Engineer | Yes |
| 24/7 Support with 1-hour P1 SLA | Yes |
| Quarterly Business Reviews | Yes |
| Model Performance Guarantees (SLA) | Yes |
| On-premise Deployment Support | Yes |
| License | Commercial (per-site, annual) |

**Pricing:**

| Cell Sites | Per-Site/Year | Annual Total |
|---|---|---|
| 1,000-4,999 | $300 | $300K-1.5M |
| 5,000-9,999 | $270 (10% discount) | $1.35M-2.7M |
| 10,000-19,999 | $240 (20% discount) | $2.4M-4.8M |
| 20,000-49,999 | $210 (30% discount) | $4.2M-10.5M |
| 50,000+ | Custom (est. $180-200) | Contact sales |

**Minimum commitment:** 1,000 cell sites, 2-year term.

---

## Per-Cell-Site Economics

### Cost to Serve

| Component | Cost per Site/Year | Notes |
|---|---|---|
| Cloud compute (FL aggregation) | $3.00 | Amortized across fleet; decreases with scale |
| Model update delivery (bandwidth) | $0.50 | Weight deltas are small (~5MB per round) |
| Support infrastructure (ticketing, monitoring) | $2.00 | Shared infrastructure, amortized |
| Support labor (per-site allocation) | $4.00 | Assumes 1 CSE per 2,500 sites |
| Software maintenance (per-site allocation) | $2.50 | Engineering cost amortized across customer base |
| **Total cost to serve** | **$12.00** | |

### Margin Analysis

| Tier | Revenue/Site | Cost/Site | Gross Margin/Site | Gross Margin % |
|---|---|---|---|---|
| Pro | $120 | $10 | $110 | 91.7% |
| Enterprise | $300 | $15 | $285 | 95.0% |
| Enterprise (50K+ sites, discounted) | $200 | $8 | $192 | 96.0% |

At scale, the cost per site decreases because FL aggregation compute, support infrastructure, and engineering are amortized across more sites. Gross margins improve from 91% at small scale to 96% at large scale.

---

## ROI Calculator for Operators

### Inputs

| Parameter | Conservative | Moderate | Optimistic |
|---|---|---|---|
| Number of cell sites | 10,000 | 25,000 | 50,000 |
| Spectrum asset value (per site) | $100,000 | $100,000 | $100,000 |
| Spectral efficiency improvement | 0.5% | 1.0% | 1.4% |
| PreceptualAI tier | Pro | Enterprise | Enterprise |
| Per-site price (with volume discount) | $108 | $210 | $200 |

### Calculations

**Conservative (10K sites, Pro, 0.5% improvement):**

| Item | Value |
|---|---|
| Annual spectrum value at risk | $1.0B (10K sites x $100K/site) |
| Value of 0.5% improvement | $5.0M/year |
| PreceptualAI cost | $1.08M/year (10K x $108) |
| **Net annual value** | **$3.92M** |
| **ROI** | **363%** |
| **Payback period** | **2.6 months** |

**Moderate (25K sites, Enterprise, 1.0% improvement):**

| Item | Value |
|---|---|
| Annual spectrum value at risk | $2.5B |
| Value of 1.0% improvement | $25.0M/year |
| PreceptualAI cost | $5.25M/year (25K x $210) |
| **Net annual value** | **$19.75M** |
| **ROI** | **376%** |
| **Payback period** | **2.5 months** |

**Optimistic (50K sites, Enterprise, 1.4% improvement):**

| Item | Value |
|---|---|
| Annual spectrum value at risk | $5.0B |
| Value of 1.4% improvement | $70.0M/year |
| PreceptualAI cost | $10.0M/year (50K x $200) |
| **Net annual value** | **$60.0M** |
| **ROI** | **600%** |
| **Payback period** | **1.7 months** |

### Sensitivity Analysis

How ROI varies with efficiency improvement and deployment size:

| Efficiency Gain | 5K Sites (Pro) | 10K Sites (Pro) | 25K Sites (Ent.) | 50K Sites (Ent.) |
|---|---|---|---|---|
| 0.25% | 79% | 131% | 138% | 225% |
| 0.50% | 258% | 363% | 376% | 600% |
| 1.00% | 617% | 826% | 852% | 1350% |
| 1.40% | 898% | 1196% | 1233% | 1950% |

**Breakeven efficiency improvement** (where PreceptualAI cost equals the value delivered):

| Deployment | PreceptualAI Cost | Breakeven Improvement |
|---|---|---|
| 5K sites, Pro | $540K | 0.11% |
| 10K sites, Pro | $960K | 0.10% |
| 25K sites, Enterprise | $5.25M | 0.21% |
| 50K sites, Enterprise | $10.0M | 0.20% |

PreceptualAI's measured improvement of 1.4% is 7-14x the breakeven threshold, providing substantial margin of safety.

---

## Comparison to Alternatives

### Build In-House

| Factor | Build In-House | Buy PreceptualAI (Enterprise) |
|---|---|---|
| Initial investment | $2-5M (team of 5-8 engineers, 18-24 months) | $0 (evaluate Community free) |
| Time to first deployment | 18-24 months | 4-8 weeks (with POC support) |
| Ongoing engineering cost | $1-2M/year (maintenance, improvements) | Included in license |
| Federated learning | Build from scratch (6-12 months additional) | Included |
| NVIDIA ARC optimization | Requires GPU engineering expertise | Included |
| Risk of failure | High (novel architecture, rare expertise) | Low (proven on real 5G data) |
| Community contributions | None | Open-source community improvements |
| **Total 3-year cost (50K sites)** | **$8-16M** | **$30M** (but with proven ROI of $60M+/yr) |

**Verdict:** Building in-house is cheaper in direct cost but carries high technical risk and an 18-24 month delay. The opportunity cost of delayed spectrum optimization (at $70M/year potential value) far exceeds the price difference.

### Legacy SON (AirHopAI, Nokia MantaRay)

| Factor | Legacy SON | PreceptualAI Enterprise |
|---|---|---|
| Architecture | Statistical ML, batch optimization | Continuous-time neural ODE, real-time RL |
| Optimization interval | Minutes to hours | Milliseconds (3.14ms) |
| Federated learning | No | Yes (hybrid LTC-aware) |
| Per-site personalization | No (cluster-based) | Yes (tau weights per site) |
| Open source evaluation | No | Yes (Apache 2.0) |
| O-RAN native | Partially (retrofitted) | Yes (built for Near-RT RIC) |
| NVIDIA ARC support | No | Yes |
| Vendor lock-in | High (proprietary platform) | Low (open standards, portable) |
| Estimated cost | $200-500/site/year | $200-300/site/year |

**Verdict:** PreceptualAI is architecturally superior and competitively priced compared to legacy SON solutions. The migration risk is mitigated by the open-source evaluation path.

### SD-RAN Platform (ONF, Mavenir)

| Factor | SD-RAN Platform | PreceptualAI Enterprise |
|---|---|---|
| Scope | Full RIC + multiple xApps | Spectrum management xApp only |
| Cost | $500K-2M/year (platform license) | $2.4-10M/year (depends on site count) |
| Spectrum management quality | Generic (one of many xApps) | Best-in-class (sole focus) |
| Federated learning | No | Yes |
| Open source | Partially (ONF) | Yes (Apache 2.0) |
| Deployment complexity | High (full platform) | Low (single xApp on existing RIC) |

**Verdict:** Not directly comparable. SD-RAN is a platform; PreceptualAI is a specialized xApp. PreceptualAI can run on top of SD-RAN platforms, complementing rather than replacing them.

---

## Pricing FAQ

**Q: Why per-cell-site pricing instead of per-CPU/GPU?**
Per-site pricing aligns PreceptualAI's revenue with the operator's value. An operator with 50,000 cell sites gets 50,000x the value of an operator with one site, so they should pay proportionally. Infrastructure-based pricing (per-CPU) would penalize efficient deployments and misalign incentives.

**Q: What counts as a "cell site"?**
A cell site is a unique gNodeB or eNodeB that PreceptualAI manages. If a physical tower has three sectors, it counts as one cell site. Small cells and DAS (Distributed Antenna Systems) each count as one site. Private 5G deployments count each base station.

**Q: Is there a free trial for Pro/Enterprise?**
Yes. We offer a 90-day free POC for operators evaluating Pro or Enterprise. The POC includes full feature access, onboarding support, and a dedicated Slack channel. No credit card required.

**Q: Can I use the Community Edition in production?**
Yes. The Apache 2.0 license permits commercial use without restrictions. Many operators will start with Community in a lab or staging environment, then upgrade to Pro/Enterprise for production deployment with support and FL capabilities.

**Q: Are there multi-year discounts?**
Yes. 2-year commitments receive an additional 5% discount. 3-year commitments receive 10%. These stack with volume discounts.

| Term | Additional Discount | Example: 50K Sites Enterprise (base $200/site) |
|---|---|---|
| 1 year | 0% | $10.0M/year |
| 2 years | 5% | $9.5M/year |
| 3 years | 10% | $9.0M/year |

**Q: What about government/defense pricing?**
Government and defense customers receive custom pricing based on mission requirements. Contact sales for GSA schedule or SEWP pricing.

**Q: Do you offer a managed service?**
Not currently. PreceptualAI is deployed on-premise within the operator's infrastructure. A managed cloud option for smaller operators is on the roadmap for 2028.

**Q: What happens if we reduce cell sites?**
Enterprise contracts are based on committed site counts. If your deployment shrinks below the committed count, the per-site rate remains the same for the remainder of the contract term. Expansion above the committed count is billed at the same rate or better (if the new total qualifies for a higher volume tier).

---

## Summary Pricing Table

| Tier | Per-Site/Year | Min Sites | Min Term | Support | FL Included |
|---|---|---|---|---|---|
| **Community** | Free | None | None | GitHub | No |
| **Pro** | $120 | 100 | 1 year | 4h SLA, biz hrs | No |
| **Enterprise** | $300 | 1,000 | 2 years | 1h P1 SLA, 24/7 | Yes |

Volume discounts: 10% at 5K+ sites, 20% at 20K+, 30% at 50K+. Multi-year discounts: 5% for 2-year, 10% for 3-year.

For custom pricing, enterprise POCs, or government contracts: **sales@preceptualai.ai**
