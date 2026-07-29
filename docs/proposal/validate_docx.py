#!/usr/bin/env python3
"""Validate the built proposal against the template's layout contract.

soffice cannot load any file in this sandbox, so Word's own validator is not
available; these checks stand in for it. They are the failure modes that
actually bit during the first build: unreferenced parts left in the package,
paragraph-property children emitted out of schema order, a section that lost
its column count, an executive summary outside the template's word band, a
budget that does not add up, and an uncited reference.
"""
from __future__ import annotations

import posixpath
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

DOCX = HERE / "Horizon-RIC_AI-RAN_Call-for-Innovation_Proposal.docx"

# CT_PPr's element sequence. A child appearing out of this order makes Word
# refuse the file.
PPR_ORDER = [
    "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr",
    "widowControl", "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs",
    "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct",
    "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
    "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
    "suppressOverlap", "jc", "textDirection", "textAlignment",
    "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr",
    "pPrChange",
]
SECTPR_ORDER = [
    "headerReference", "footerReference", "footnotePr", "endnotePr", "type",
    "pgSz", "pgMar", "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols",
    "formProt", "vAlign", "noEndnote", "titlePg", "textDirection", "bidi",
    "rtlGutter", "docGrid", "printerSettings", "sectPrChange",
]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

problems: list[str] = []
notes: list[str] = []


def check(cond, msg, detail=""):
    if cond:
        notes.append("  ok    " + msg + (("  (%s)" % detail) if detail else ""))
    else:
        problems.append("  FAIL  " + msg + (("  — %s" % detail) if detail else ""))


def check_package():
    with zipfile.ZipFile(DOCX) as z:
        names = set(z.namelist())
        bad = [n for n in names if "[trash]" in n or n.startswith("[trash]")]
        check(not bad, "no orphaned [trash] parts in the package",
              ",".join(bad))

        rel_targets = set()
        for n in names:
            if n.endswith(".rels"):
                body = z.read(n).decode("utf-8", "replace")
                base = "/".join(n.split("/")[:-2])
                for m in re.finditer(r'Target="([^"]+)"[^>]*?(/>|>)', body):
                    t = m.group(1)
                    if t.startswith(("http", "mailto", "#")):
                        continue
                    t = t.lstrip("/")
                    joined = (base + "/" + t) if base else t
                    rel_targets.add(posixpath.normpath(joined))
        missing = sorted(t for t in rel_targets if t not in names)
        check(not missing, "every relationship target exists in the package",
              ",".join(missing[:4]))

        media = sorted(n for n in names if n.startswith("word/media/"))
        check(len(media) == 3, "exactly three embedded images",
              "%d: %s" % (len(media), ", ".join(p.split("/")[-1] for p in media)))
        return names


def check_schema_order():
    doc = Document(DOCX)
    tree = doc.element
    bad_ppr, bad_sect = [], []
    for i, pPr in enumerate(tree.iter(W + "pPr")):
        seen = [c.tag.replace(W, "") for c in pPr
                if c.tag.startswith(W) and c.tag.replace(W, "") in PPR_ORDER]
        idx = [PPR_ORDER.index(t) for t in seen]
        if idx != sorted(idx):
            bad_ppr.append("pPr#%d %s" % (i, seen))
    for i, sectPr in enumerate(tree.iter(W + "sectPr")):
        seen = [c.tag.replace(W, "") for c in sectPr
                if c.tag.startswith(W) and c.tag.replace(W, "") in SECTPR_ORDER]
        idx = [SECTPR_ORDER.index(t) for t in seen]
        if idx != sorted(idx):
            bad_sect.append("sectPr#%d %s" % (i, seen))
    check(not bad_ppr, "every w:pPr emits its children in schema order",
          "; ".join(bad_ppr[:3]))
    check(not bad_sect, "every w:sectPr emits its children in schema order",
          "; ".join(bad_sect[:3]))
    return doc


def check_layout(doc):
    # Single-column masthead band, two-column body, a full-width figure band,
    # then two columns again — the IEEE wide-figure idiom. The first three
    # sections must be continuous breaks or each band starts a fresh page.
    WANT_COLS = ("1", "2", "1", "2")
    secs = doc.sections
    check(len(secs) == len(WANT_COLS),
          "four sections: band, two columns, wide-figure band, two columns",
          "found %d" % len(secs))
    for i, s in enumerate(secs):
        tag = "section %d" % (i + 1)
        check(s.page_width.twips == 12240 and s.page_height.twips == 15840,
              tag + ": US Letter 12240x15840 twips",
              "%dx%d" % (s.page_width.twips, s.page_height.twips))
        check(s.top_margin.twips == 1440 and s.bottom_margin.twips == 1440
              and s.left_margin.twips == 1080 and s.right_margin.twips == 1080,
              tag + ": margins 1440/1080",
              "t%d b%d l%d r%d" % (s.top_margin.twips, s.bottom_margin.twips,
                                   s.left_margin.twips, s.right_margin.twips))
        cols = s._sectPr.find(qn("w:cols"))
        num = cols.get(qn("w:num")) if cols is not None else None
        space = cols.get(qn("w:space")) if cols is not None else None
        want_num = WANT_COLS[i]
        want = (want_num, "720" if want_num == "1" else "360")
        check((num, space) == want,
              tag + ": w:cols num=%s space=%s" % want,
              "got num=%s space=%s" % (num, space))
        if i < len(secs) - 1:
            st = s._sectPr.find(qn("w:type"))
            val = st.get(qn("w:val")) if st is not None else None
            check(val == "continuous",
                  tag + ": continuous break, so the next band shares the page",
                  "w:type=%s" % val)

    have = {s.name for s in doc.styles}
    for want in ("IEEETitle", "IEEEAuthors", "AbstractHeading"):
        check(want in have, "style %s defined" % want)

    check(len(doc.inline_shapes) == 3, "three inline figures placed",
          "found %d" % len(doc.inline_shapes))
    for sh, name in zip(doc.inline_shapes,
                        ("fig-arch", "fig-enforcement", "fig-plan")):
        w_in = sh.width.inches
        h_in = sh.height.inches
        ok = 3.0 <= w_in <= 6.5 and 1.0 <= h_in <= 3.0
        check(ok, "%s sized sanely" % name, "%.2f x %.2f in" % (w_in, h_in))

    check(len(doc.tables) == 2, "two tables (budget, criteria map)",
          "found %d" % len(doc.tables))


def check_content(doc):
    text = "\n".join(p.text for p in doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                text += "\n" + cell.text

    check(C.TITLE in text, "title present")
    check("Daniel Foo" in text and "Bowen Shen" in text and "Feng Li" in text,
          "all three team members named in the byline")
    check("Zhejiang" not in text and "Jun wei" not in text
          and "Jun Wei" not in text, "removed affiliations stay removed")

    # Conservative: str.split() counts a spaced em dash as a word and Word does
    # not, so this over-counts by roughly the number of dashes. Passing here
    # means Word's own count is inside the band too.
    words = len(C.EXEC_SUMMARY.split())
    check(150 <= words <= 250,
          "executive summary within the template's 150-250 word band",
          "%d words" % words)

    total = sum(a for _, a, _ in C.BUDGET)
    check(total == C.BUDGET_TOTAL == 150_000,
          "budget lines sum to the requested total",
          "%s vs %s" % (format(total, ","), format(C.BUDGET_TOTAL, ",")))
    check("150,000" in text, "the total appears in the document body")

    body_only = text.split("References")[0]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body_only)}
    n_refs = len(C.REFERENCES)
    uncited = sorted(set(range(1, n_refs + 1)) - cited)
    dangling = sorted(n for n in cited if n > n_refs)
    check(not uncited, "every reference is cited in the body",
          "uncited: %s" % uncited)
    check(not dangling, "no citation points past the reference list",
          "dangling: %s" % dangling)

    numbered = [p.text for p in doc.paragraphs
                if re.match(r"^\[\d+\]\s", p.text.strip())]
    check(len(numbered) == n_refs, "reference list is complete",
          "%d of %d" % (len(numbered), n_refs))

    # Claims the ledger marks red must not appear in their withdrawn form.
    for forbidden, why in (
        ("none reached the radio", "withdrawn: no over-the-air capture"),
        ("never exceeds the classical", "corrected to a 1 dB envelope"),
        ("6G compliant", "not claimable, no specification exists"),
    ):
        check(forbidden.lower() not in text.lower(),
              "withdrawn claim absent: %r" % forbidden, why)

    for want in ("0.370", "4658", "2442", "461", "12/12", "150,000",
                 "1 dB envelope", "US$150,000"):
        check(want in text, "gated figure present: %s" % want)

    # The Call requires six content items and scores against five criteria.
    for want in ("Problem Statement and Market Relevance",
                 "Innovative Solution and Technical Approach",
                 "Deployment Feasibility",
                 "Twelve-Month Timeline and Milestones",
                 "Expected Deliverables and Impact",
                 "Executive Summary"):
        check(want in text, "required content item present: %s" % want)
    for crit, _ in C.CRITERIA:
        label = crit.split("  ", 1)[1]
        check(label in text, "official criterion mapped: %s" % label)

    for sec, _ in C.SECTIONS:
        num = sec.split(".")[0]
        check(any(p.text.strip().startswith(num + ".") for p in doc.paragraphs),
              "section %s present" % num)

    stats = {
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "inline_shapes": len(doc.inline_shapes),
        "body_words": len(text.split()),
        "exec_summary_words": words,
        "references": n_refs,
    }
    return stats


def main() -> int:
    if not DOCX.exists():
        print("missing", DOCX)
        return 2
    check_package()
    doc = check_schema_order()
    check_layout(doc)
    stats = check_content(doc)

    print("\n".join(notes))
    if problems:
        print("\n".join(problems))
        print("\n%d validation FAILURE(S)" % len(problems))
        return 1
    print("\nAll validations PASSED  (%d checks)" % len(notes))
    for k, v in stats.items():
        print("  %-20s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
