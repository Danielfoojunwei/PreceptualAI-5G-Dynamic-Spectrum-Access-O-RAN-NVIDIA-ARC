#!/usr/bin/env python3
"""Build the AI-RAN Alliance Call-for-Innovations proposal as a .docx.

The original template upload is no longer on disk, so this script reconstructs
its layout contract exactly rather than editing it in place:

  US Letter, 12240 x 15840 twips
  margins   top/bottom 1440, left/right 1080  (text block 7.0in x 9.0in)
  section 1 single column, <w:cols w:space="720"/>
      IEEETitle -> centred title, sz 48 (24pt)
      three centred author/affiliation/contact lines
      Executive Summary heading + body, Keywords, bottom-bordered rule
      the two full-width figures
  section 2 two columns, <w:cols w:num="2" w:space="360"/>
      sections I-IX, in-column figure and budget table, REFERENCES

Paragraph-property children are emitted in schema order
(pStyle, keepNext, pBdr, spacing, ind, jc, rPr, sectPr) — Word rejects the
file otherwise.

Content comes from content.py, which the HTML preview also renders, so the two
artefacts cannot drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

OUT = HERE / "Horizon-RIC_AI-RAN_Call-for-Innovation_Proposal.docx"

SERIF = "Times New Roman"
BODY_PT = C.BODY_PT
PARA_AFTER = C.PARA_AFTER_PT
TEXT_WIDTH_IN = 7.0
WIDE_FIG_IN = 6.1
COL_WIDTH_IN = (TEXT_WIDTH_IN - 0.25) / 2  # 3.375in

# Elements that must follow <w:cols> inside <w:sectPr>, per CT_SectPr's sequence.
_AFTER_COLS = [
    "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection",
    "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings", "w:sectPrChange",
]


def set_cols(section, num: int, space_twips: int) -> None:
    """Replace <w:cols> on this section, inserting it in schema position."""
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    if num > 1:
        cols.set(qn("w:equalWidth"), "1")
    anchor = None
    for tag in _AFTER_COLS:
        found = sectPr.find(qn(tag))
        if found is not None:
            anchor = found
            break
    if anchor is None:
        sectPr.append(cols)
    else:
        anchor.addprevious(cols)


def page_setup(section) -> None:
    section.page_width = Twips(12240)
    section.page_height = Twips(15840)
    section.top_margin = Twips(1440)
    section.bottom_margin = Twips(1440)
    section.left_margin = Twips(1080)
    section.right_margin = Twips(1080)


def define_styles(doc) -> None:
    from docx.enum.style import WD_STYLE_TYPE

    normal = doc.styles["Normal"]
    normal.font.name = SERIF
    normal.font.size = Pt(BODY_PT)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), SERIF)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    title = doc.styles.add_style("IEEETitle", WD_STYLE_TYPE.PARAGRAPH)
    title.base_style = normal
    title.font.name = SERIF
    title.font.size = Pt(24)          # sz 48 half-points, as in the template
    title.font.bold = False
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)

    authors = doc.styles.add_style("IEEEAuthors", WD_STYLE_TYPE.PARAGRAPH)
    authors.base_style = normal
    authors.font.name = SERIF
    authors.font.size = Pt(11)
    authors.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    abs_h = doc.styles.add_style("AbstractHeading", WD_STYLE_TYPE.PARAGRAPH)
    abs_h.base_style = normal
    abs_h.font.name = SERIF
    abs_h.font.size = Pt(BODY_PT)
    abs_h.font.bold = True
    abs_h.paragraph_format.space_before = Pt(6)
    abs_h.paragraph_format.space_after = Pt(2)


def body_para(doc, text, *, first=False, size=BODY_PT, justify=True,
              space_after=PARA_AFTER, italic_lead=None):
    """One justified body paragraph, IEEE first-line indent except the first."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.first_line_indent = Inches(0.0 if first else 0.22)
    pf.alignment = (WD_ALIGN_PARAGRAPH.JUSTIFY if justify
                    else WD_ALIGN_PARAGRAPH.LEFT)
    if italic_lead:
        r = p.add_run(italic_lead)
        r.font.name = SERIF
        r.font.size = Pt(size)
        r.italic = True
    r = p.add_run(text)
    r.font.name = SERIF
    r.font.size = Pt(size)
    return p


def heading(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)   # w:spacing w:before="120"
    pf.space_after = Pt(3)    # w:after="60"
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.keep_with_next = True
    r = p.add_run(text)
    r.font.name = SERIF
    r.font.size = Pt(BODY_PT)
    r.bold = True
    return p


def rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "auto")
    pBdr.append(bottom)
    # pBdr precedes w:spacing in CT_PPr's sequence.
    spacing = pPr.find(qn("w:spacing"))
    if spacing is not None:
        spacing.addprevious(pBdr)
    else:
        pPr.append(pBdr)
    return p


def figure(doc, png: Path, width_in: float, caption: str, *,
           caption_indent: float = 0.0, caption_pt: float = 8.0):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(png), width=Inches(width_in))

    cap = doc.add_paragraph()
    cf = cap.paragraph_format
    cf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cf.space_after = Pt(5)
    cf.left_indent = Inches(caption_indent)
    cf.right_indent = Inches(caption_indent)
    label, _, rest = caption.partition("  ")
    r = cap.add_run(label + "  ")
    r.font.name = SERIF
    r.font.size = Pt(caption_pt)
    r.bold = True
    r = cap.add_run(rest)
    r.font.name = SERIF
    r.font.size = Pt(caption_pt)
    return cap


def budget_table(doc):
    rows = len(C.BUDGET) + 2
    t = doc.add_table(rows=rows, cols=2)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = (Inches(COL_WIDTH_IN - 0.72), Inches(0.72))

    def cell_text(cell, text, *, bold=False, italic=False, pt=7.5, right=False):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0.5)
        p.paragraph_format.space_before = Pt(0.5)
        if right:
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(text)
        r.font.name = SERIF
        r.font.size = Pt(pt)
        r.bold = bold
        r.italic = italic
        return p

    cell_text(t.cell(0, 0), "Item", bold=True)
    cell_text(t.cell(0, 1), "US$", bold=True, right=True)
    for i, (item, amount, why) in enumerate(C.BUDGET, start=1):
        p = cell_text(t.cell(i, 0), item)
        r = p.add_run("  — " + why)
        r.font.name = SERIF
        r.font.size = Pt(6.8)
        r.italic = True
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        cell_text(t.cell(i, 1), "{:,}".format(amount), right=True)
    last = len(C.BUDGET) + 1
    cell_text(t.cell(last, 0), "Total requested", bold=True)
    cell_text(t.cell(last, 1), "{:,}".format(C.BUDGET_TOTAL), bold=True,
              right=True)

    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w
    return no_split(t)


def caption_only(doc, text, pt=7.5, keep_with_next=False):
    """A table caption. IEEE puts these above the table, so the default is to
    bind it to what follows."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(1.5 if keep_with_next else 5)
    p.paragraph_format.keep_with_next = keep_with_next
    label, _, rest = text.partition("  ")
    r = p.add_run(label + "  ")
    r.font.name = SERIF
    r.font.size = Pt(pt)
    r.bold = True
    r = p.add_run(rest)
    r.font.name = SERIF
    r.font.size = Pt(pt)
    return p


def no_split(table):
    """Keep each row intact — a row split across a column reads as corruption."""
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    return table


def criteria_table(doc):
    """Table II — the five official criteria, one row each."""
    t = doc.add_table(rows=len(C.CRITERIA), cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for row, (crit, where) in zip(t.rows, C.CRITERIA):
        cell = row.cells[0]
        cell.width = Inches(COL_WIDTH_IN)
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0.5)
        p.paragraph_format.space_after = Pt(0.5)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(crit + "  ")
        r.font.name = SERIF
        r.font.size = Pt(7.2)
        r.bold = True
        r = p.add_run(where)
        r.font.name = SERIF
        r.font.size = Pt(7.0)
    return no_split(t)


def emit_sections(doc, groups) -> None:
    """Render numbered proposal sections into the current column flow."""
    for head, paras in groups:
        heading(doc, head)
        for j, text in enumerate(paras):
            lead, body = None, text
            # Pull a leading "Sentence fragment." run into italics, IEEE style.
            head_frag, sep, rest = text.partition(". ")
            if (sep and j > 0 and len(head_frag) <= 40
                    and head_frag[:1].isupper() and "," not in head_frag):
                lead, body = head_frag + ".  ", rest
            body_para(doc, body, first=(j == 0), italic_lead=lead)

        # "V." does not match "VI."/"VII."/"VIII." — the char after the
        # numeral differs — so these prefixes select exactly one section each.
        if head.startswith("V."):
            figure(doc, HERE / "fig-plan.png", COL_WIDTH_IN - 0.02,
                   C.FIG3_CAPTION, caption_pt=7.2)
        if head.startswith("VI."):
            caption_only(doc, "Table I.  Requested budget, "
                              "US$150,000 over twelve months.",
                         keep_with_next=True)
            budget_table(doc)
        if head.startswith("VIII."):
            caption_only(doc, "Table II.  The five official evaluation "
                              "criteria, and where this proposal answers them.",
                         keep_with_next=True)
            criteria_table(doc)


def build() -> Path:
    doc = Document()
    define_styles(doc)

    # ---------------- section 1: single column ---------------------------
    p = doc.add_paragraph(style="IEEETitle")
    p.add_run(C.TITLE)

    for line, pt, italic in (
        (C.AUTHOR_LINE, 11.0, False),
        (C.AFFIL_LINE, 9.5, True),
        (C.CONTACT_LINE, 8.5, False),
    ):
        ap = doc.add_paragraph(style="IEEEAuthors")
        ap.paragraph_format.space_after = Pt(1)
        r = ap.add_run(line)
        r.font.name = SERIF
        r.font.size = Pt(pt)
        r.italic = italic

    h = doc.add_paragraph(style="AbstractHeading")
    h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h.add_run("Executive Summary")
    body_para(doc, C.EXEC_SUMMARY, first=True)

    kw = doc.add_paragraph()
    kw.paragraph_format.space_after = Pt(2)
    kw.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = kw.add_run("Keywords — ")
    r.font.name = SERIF
    r.font.size = Pt(BODY_PT)
    r.bold = True
    r.italic = True
    r = kw.add_run(C.KEYWORDS)
    r.font.name = SERIF
    r.font.size = Pt(BODY_PT)

    nb = doc.add_paragraph()
    nb.paragraph_format.space_after = Pt(2)
    nb.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = nb.add_run(C.REPO_NOTE)
    r.font.name = SERIF
    r.font.size = Pt(8.0)
    r.italic = True

    rule(doc)

    pad = (TEXT_WIDTH_IN - WIDE_FIG_IN) / 2
    figure(doc, HERE / "fig-arch.png", WIDE_FIG_IN, C.FIG1_CAPTION,
           caption_indent=pad)

    # ---- continuous break: two columns resume on the same page ----------
    doc.add_section(WD_SECTION.CONTINUOUS)
    # The wide results figure follows the evidence section it belongs to,
    # so the band splits the flow after III rather than at a fixed page.
    emit_sections(doc, C.SECTIONS[:3])

    # ---- continuous break: full-width band for the wide results figure --
    doc.add_section(WD_SECTION.CONTINUOUS)
    figure(doc, HERE / "fig-enforcement.png", WIDE_FIG_IN, C.FIG2_CAPTION,
           caption_indent=pad)

    # ---- continuous break: back to two columns --------------------------
    doc.add_section(WD_SECTION.CONTINUOUS)
    emit_sections(doc, C.SECTIONS[3:])

    heading(doc, "References")
    for i, ref in enumerate(C.REFERENCES, start=1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(1.0)
        pf.left_indent = Inches(0.16)
        pf.first_line_indent = Inches(-0.16)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run("[%d]  %s" % (i, ref))
        r.font.name = SERIF
        r.font.size = Pt(C.REF_PT)

    # ---------------- section geometry, set once at the end ---------------
    # sections[0..2] are paragraph-level and terminate their own section, so
    # they carry the break type; sections[3] is the body sentinel.
    cols_per_section = (1, 2, 1, 2)
    secs = doc.sections
    if len(secs) != len(cols_per_section):
        raise AssertionError("expected %d sections, built %d"
                             % (len(cols_per_section), len(secs)))
    for i, (s, num) in enumerate(zip(secs, cols_per_section)):
        page_setup(s)
        set_cols(s, num, 720 if num == 1 else 360)
        if i < len(secs) - 1:
            s.start_type = WD_SECTION.CONTINUOUS

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print("wrote", path, path.stat().st_size, "bytes")
