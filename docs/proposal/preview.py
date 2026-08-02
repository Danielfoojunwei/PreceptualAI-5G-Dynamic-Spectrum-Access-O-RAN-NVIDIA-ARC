#!/usr/bin/env python3
"""Render a page-accurate preview of the proposal, from the same content module.

soffice cannot open any file in this sandbox, so the .docx cannot be converted
directly. This mirrors the .docx's layout contract in HTML — US Letter, 0.75in
side margins and 1in top/bottom, two 3.375in columns with a 0.25in gutter, and
`column-span: all` bands standing in for the .docx's continuous section breaks.
The type scale comes from content.py, so the two artefacts paginate alike. It is
a proxy for Word's pagination, not Word itself: use it to judge page count and
balance, not to certify kerning.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

HTML_OUT = HERE / "proposal_preview.html"
PDF_OUT = HERE / "Horizon-RIC_Proposal_PREVIEW.pdf"

CSS = """
@page { size: Letter; margin: 1in 0.75in; }
.refpage { break-before: page; }
h2.reftop { margin-top: 0; }
html { font-family: "Times New Roman", Times, serif;
       font-size: %(body)spt; }
body { margin: 0; }
.title { font-size: 24pt; text-align: center; margin: 0 0 6pt;
         line-height: 1.12; }
.authors { text-align: center; font-size: 11pt; margin: 0 0 1pt; }
.affil { text-align: center; font-size: %(body)spt; font-style: italic;
         margin: 0 0 1pt; }
h3.absh { font-size: %(body)spt; font-weight: bold; margin: 6pt 0 2pt; }
p { margin: 0 0 %(after)spt; text-indent: 0.22in; text-align: justify;
    orphans: 1; widows: 1; }
p.first { text-indent: 0; }
p.kw { font-size: %(body)spt; text-indent: 0; }
p.kw b { font-style: italic; }
p.repo { font-size: 8pt; font-style: italic; text-indent: 0; }
hr { border: none; border-top: 0.5pt solid #000; margin: 4pt 0; }
.band { column-span: all; }
.band figure { margin: 4pt auto 5pt; break-inside: avoid; width: 5.2in; }
.band figure img { width: 5.2in; display: block; margin: 0 auto; }
figcaption { font-size: 8pt; text-align: justify; margin-top: 2pt;
             line-height: 1.2; }
.cols { column-count: 2; column-gap: 0.25in; }
h2 { font-size: %(body)spt; font-weight: bold; text-align: center;
     margin: 6pt 0 3pt; break-after: avoid; }
.cols figure { margin: 4pt 0 5pt; break-inside: avoid; }
.cols figure img { width: 3.35in; display: block; margin: 0 auto; }
.cols figcaption { font-size: 7.2pt; }
/* The band rules must outrank the in-column ones: a .band lives inside .cols,
   so equal-specificity selectors would otherwise clamp a wide figure to a
   single column. */
.cols .band figure { width: 5.2in; margin: 4pt auto 5pt; }
.cols .band figure img { width: 5.2in; }
.cols .band figcaption { font-size: 8pt; }
table.budget { width: 100%%; border-collapse: collapse; font-size: 7.0pt;
               margin: 2pt 0 3pt; }
table.budget th, table.budget td { border: 0.5pt solid #000; padding: 0.5pt 2pt;
                                   text-align: left; vertical-align: top; }
table.budget td.n, table.budget th.n { text-align: right;
                                       white-space: nowrap; }
table.budget span.why { font-size: 6.8pt; font-style: italic; color: #444; }
p.tabcap { font-size: 7.5pt; text-indent: 0; }
table.crit td { font-size: 6.6pt; text-align: justify; }
/* A table that cannot split is a table that jumps a whole page when it
   does not fit — that is what pushed the document past two pages. Rows
   stay unsplit; the block may break between them, and the caption is
   held to the first row. */
.tblock p.tabcap { break-after: avoid; }
.tblock p.tabcap { margin-bottom: 1.5pt; }
table.budget tr { break-inside: avoid; }
ol.refs { font-size: %(ref)spt; padding-left: 0.16in; margin: 1pt 0 0; }
ol.refs li { margin-bottom: 1.0pt; text-align: left; break-inside: avoid; }
.pagebreak { break-after: page; }
""" % {"body": C.BODY_PT, "after": C.PARA_AFTER_PT,
       "ref": C.REF_PT}


def esc(t: str) -> str:
    return html.escape(t, quote=False)


def wide_figure(png: str, caption: str) -> str:
    label, _, rest = caption.partition("  ")
    return ('<div class="band"><figure><img src="%s">'
            '<figcaption><b>%s</b> %s</figcaption></figure></div>'
            % (png, esc(label), esc(rest)))


def build_html() -> str:
    out = ['<meta charset="utf-8"><style>%s</style>' % CSS]
    out.append('<div class="cols">')
    out.append('<div class="band">')
    out.append('<div class="title">%s</div>' % esc(C.TITLE))
    out.append('<div class="authors">%s</div>' % esc(C.AUTHOR_LINE))
    out.append('<div class="affil">%s</div>' % esc(C.AFFIL_LINE))
    out.append('<h3 class="absh">Executive Summary</h3>')
    out.append('<p class="first">%s</p>' % esc(C.EXEC_SUMMARY))
    out.append('<p class="kw"><b>Keywords &mdash;</b> %s</p>' % esc(C.KEYWORDS))
    out.append('<p class="repo">%s</p>' % esc(C.REPO_NOTE))
    out.append("<hr>")
    out.append(wide_figure("fig-enforcement.png", C.FIG1_CAPTION))
    out.append("</div>")

    for head, paras in C.SECTIONS:
        out.append("<h2>%s</h2>" % esc(head))
        for j, text in enumerate(paras):
            cls = ' class="first"' if j == 0 else ""
            out.append("<p%s>%s</p>" % (cls, esc(text)))
        if head.startswith("VI."):
            out.append('<div class="tblock">')
            out.append('<p class="tabcap"><b>Table I.</b> What we ask for '
                       '&mdash; funding against named work, and the access '
                       'funding cannot buy &mdash; with what goes back in '
                       'return. No amount is named: see &sect;VI.</p>')
            out.append('<table class="budget crit">')
            for ask, gives in C.ALLIANCE_ASKS:
                out.append('<tr><td><b>%s</b> %s</td></tr>'
                           % (esc(ask), esc(gives)))
            out.append('</table></div>')
        if head.startswith("VIII."):
            out.append('<div class="tblock">')
            out.append('<p class="tabcap"><b>Table II.</b> The five official '
                       'evaluation criteria, and where this proposal answers '
                       'them.</p>')
            out.append('<table class="budget crit">')
            for crit, where in C.CRITERIA:
                out.append('<tr><td><b>%s</b> %s</td></tr>'
                           % (esc(crit), esc(where)))
            out.append('</table></div>')
    # References start on their own page, matching the .docx page break.
    out.append("</div>")
    out.append('<div class="cols refpage">')
    out.append('<h2 class="reftop">References</h2><ol class="refs">')
    for ref in C.REFERENCES:
        out.append("<li>%s</li>" % esc(ref))
    out.append("</ol></div>")
    return "\n".join(out)


def main() -> int:
    from weasyprint import HTML

    HTML_OUT.write_text(build_html(), encoding="utf-8")
    HTML(filename=str(HTML_OUT), base_url=str(HERE)).write_pdf(str(PDF_OUT))

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(PDF_OUT))
    n = len(pdf)
    print("preview:", PDF_OUT.name, n, "pages", PDF_OUT.stat().st_size, "bytes")
    for i in range(n):
        img = pdf[i].render(scale=1.55).to_pil()
        img.save(HERE / ("prev%d.png" % (i + 1)))
        print("  page %d -> prev%d.png  %s" % (i + 1, i + 1, img.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
