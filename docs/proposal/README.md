# AI-RAN Alliance Call for Innovations — proposal build

The submission document and everything needed to rebuild it. Nothing here is
hand-edited in Word: the text lives in `content.py`, the figures are generated,
and the `.docx` is assembled and then validated by a script. That is deliberate
— a proposal whose numbers are retyped by hand drifts from the repository it
describes, and this one has to stay checkable against
[`../CLAIM_LEDGER.md`](../CLAIM_LEDGER.md).

## Deliverables

| File | What it is |
|---|---|
| `Horizon-RIC_AI-RAN_Call-for-Innovation_Proposal.docx` | The submission. Five pages, US Letter, IEEE two-column. |
| `Horizon-RIC_Proposal_PREVIEW.pdf` | Page-accurate preview, for reading and page counting without Word. |

## Rebuilding

```sh
python docs/proposal/mkfigs.py        # regenerate all three figures
python docs/proposal/build_docx.py    # assemble the .docx
python docs/proposal/validate_docx.py # 89 checks; exits non-zero on any failure
python docs/proposal/preview.py       # render the preview PDF
```

Needs `python-docx`, `matplotlib`, `cairosvg`, `weasyprint` and `pypdfium2`.
`mkfigs.py` reads `benchmarks/results/*.json` from the repository root, so run it
from a full checkout.

## What each file does

- **`content.py`** — every word of the proposal, plus the shared type scale.
  Both renderers import it, so the `.docx` and the preview cannot disagree.
- **`mkfigs.py`** — regenerates the figures. `fig-enforcement.png` is plotted
  directly from `benchmarks/results/poisoning_shield.json` and
  `evasion_suite.json`; no number in it is typed in. `fig-plan.png` is drawn from
  the work-package table in the script. `fig-arch.png` is rasterised from
  `fig-arch.svg`, which is the hand-authored source for that diagram and the one
  file here you would edit in a vector editor.
- **`build_docx.py`** — assembles the OOXML. See below.
- **`validate_docx.py`** — stands in for Word's validator, which is not
  available in this environment.
- **`preview.py`** — mirrors the `.docx` layout contract in HTML and renders it
  with weasyprint. A proxy for Word's pagination, not Word itself: trust it for
  page count and column balance, not for kerning.

## Layout contract

The Alliance's template is an IEEE two-column Word document. The build reproduces
its geometry rather than editing a copy of it:

- US Letter, 12240 × 15840 twips; margins 1440 top/bottom, 1080 left/right, so
  the text block is 7.0 × 9.0 in and each column is 3.375 in.
- `IEEETitle` (24 pt centred), `IEEEAuthors`, `AbstractHeading` styles.
- Four sections joined by **continuous** breaks: single-column masthead →
  two columns → single-column band for the wide results figure → two
  columns. Continuous is what keeps each band on the same page instead of
  starting a new one; it is the standard idiom for a full-width figure in a
  two-column paper.

Two OOXML details the validator guards, because both broke the first build:
paragraph-property children must be emitted in `CT_PPr` schema order (`pStyle`,
`keepNext`, `pBdr`, `spacing`, `ind`, `jc`, `rPr`, `sectPr` — Word rejects the
file otherwise), and `w:cols` must sit in its schema position inside `w:sectPr`.

## Structure, and why

The Call requires six content items and scores against five criteria. Sections
are named after the required items so a reviewer can find each one, and Table II
maps the document onto the criteria explicitly. `validate_docx.py` asserts that
all six items and all five criteria appear, so the structure cannot quietly drift
back into a research-paper shape.

## Honesty checks in the validator

Beyond layout, the validator enforces things a reviewer would otherwise have to
take on trust:

- Every reference is cited, and no citation points past the list.
- **Funding is requested; no amount is named.** The proposal asks to be funded
  for named work (WP2–WP5) and separately asks for access that funding cannot
  buy. The validator therefore enforces both halves: no currency figure and no
  amount spelled out in words, *and* that the document still says what it wants
  funded — at least three `FUND` lines, at least three `ACCESS` lines, every
  ask present in the rendered text, and the Alliance capabilities it draws on
  named (Data-for-AI, Test Methodology, AI-for-RAN, the endorsed labs).
- **WP1 is presented as scheduled work, deliberately.** Much of it is in fact
  already built and CI-gated in this repository — the schemas, the draft
  profile, the G1 gate. How far along it is is not disclosed in the submitted
  document, and that is an authorial decision rather than an oversight. The
  validator enforces the decision in both directions: WP1 must appear with its
  months, and no completion claim ("already delivered", "delivered baseline")
  may leak back in through a later edit.

  Two leaks did survive the first pass, and are worth knowing about if you edit
  the text: a §V line promising "the E2SM-RC payloads that **WP1 already
  constructs**", and a Table I line calling G1 "a **delivered
  demonstration**". Neither matched the phrase list, and neither reads as being
  about WP1's status until you notice it contradicts §V scheduling WP1 for
  months 1–4. Both phrases are now in the list. The general shape to watch for
  is a sentence about some *other* work package that quietly asserts WP1's
  output already exists.

  Worth knowing if you revisit this: the repository URL is in the proposal's
  masthead, so a reviewer who follows it can see the state of WP1 for
  themselves. The choice being made here is not to conceal, only not to
  advertise.
- The gated figures (4658, 2442, 461, 0.370, 12/12) are present.
- Claims the ledger marks **red** are absent in their withdrawn form — "none
  reached the radio", "never exceeds the classical baseline", and any assertion
  of 6G compliance will fail the build if they reappear.

That last group is the point of running a validator on a document at all. The
worst-case adversarial penalty in Fig. 2(d) reads 0.370 dB because that is what
`evasion_suite.json` says; an earlier draft of this proposal claimed the Shield
was never worse than the classical baseline, and the gate that now guards this
number is what caught it.

## Known open items

- **Page limit.** This builds to five pages, matching the structure the review
  recommended. Some readings of the call text suggest a shorter limit; if a hard
  1–2 page limit applies, the condensation to write is the executive summary
  plus Table II plus Fig. 2, and the rest becomes an appendix.
- **Authorship sign-off.** The document names Bowen Shen and Dr Feng Li as work
  package owners. Their written approval of the byline,
  affiliation and role must precede any submission.
