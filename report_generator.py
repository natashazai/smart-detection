"""
SENTINEL -- Clinical Report Generator
======================================
Asks Nemotron 120B to write a structured clinical report
and saves it as a polished PDF the patient can bring to a doctor.

Layout rules
------------
* All text goes through _safe() before entering a Paragraph -- strips every
  non-Latin-1 character so Helvetica never produces solid black boxes.
* Badge uses a proper multi-row Table (not a list inside one cell) so
  ReportLab can size each row independently without overlap.
* No Unicode decorators: x (not x), - (not em/en dash), [!] (not warning sign).
"""

import json
import time
from datetime import datetime
from io import BytesIO
from openai import OpenAI

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY

from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"


# ---------------------------------------------------------------------------
# Text sanitiser -- call on ALL user/LLM strings before Paragraph()
# ---------------------------------------------------------------------------

def _safe(text) -> str:
    """Replace non-Latin-1 chars that Helvetica renders as black boxes."""
    if not isinstance(text, str):
        text = str(text)
    subs = [
        ("\u2014", " - "), ("\u2013", " - "),   # em dash, en dash
        ("\u2018", "'"),   ("\u2019", "'"),       # smart single quotes
        ("\u201c", '"'),   ("\u201d", '"'),       # smart double quotes
        ("\u00d7", "x"),                          # multiplication sign
        ("\u00b7", "-"),                          # middle dot
        ("\u25a0", ""),    ("\u25cf", ""),        # black square / circle
        ("\u2022", "-"),                          # bullet
        ("\u2026", "..."),                        # ellipsis
        ("\u00ae", "(R)"), ("\u00a9", "(C)"),
    ]
    for bad, good in subs:
        text = text.replace(bad, good)
    # Drop anything still outside Latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_INK     = colors.HexColor("#111827")
_SUBTEXT = colors.HexColor("#6b7280")
_RULE    = colors.HexColor("#e5e7eb")
_STRIPE  = colors.HexColor("#f9fafb")
_WHITE   = colors.white

_SEV_PALETTE = {
    "none":     ("#dcfce7", "#166534", "#15803d"),
    "mild":     ("#fef9c3", "#854d0e", "#ca8a04"),
    "moderate": ("#fff7ed", "#9a3412", "#ea580c"),
    "marked":   ("#fef2f2", "#991b1b", "#dc2626"),
    "severe":   ("#fdf2f8", "#701a75", "#a21caf"),
}


def _sev_colors(severity: str):
    bg, fg, bar = _SEV_PALETTE.get(severity.lower(), _SEV_PALETTE["moderate"])
    return colors.HexColor(bg), colors.HexColor(fg), colors.HexColor(bar)


# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------

def _section_header(title: str, W: float, bar_color) -> list:
    """Left-accent section title: coloured bar | bold label."""
    tbl = Table(
        [[
            Paragraph(" ", ParagraphStyle("_bar", fontSize=11, leading=14)),
            Paragraph(
                f"<b>{_safe(title)}</b>",
                ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=11,
                               textColor=_INK, leading=14),
            ),
        ]],
        colWidths=[6, W - 6],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), bar_color),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (1, 0), (1, -1), 10),
    ]))
    return [tbl, Spacer(1, 8)]


# ---------------------------------------------------------------------------
# LLM report text
# ---------------------------------------------------------------------------

def generate_report_text(features, severity: str, ftm_score: int) -> dict:
    prompt = (
        "You are a clinical neurologist AI writing a structured screening report.\n"
        "A patient completed a 30-second hand tremor assessment.\n\n"
        "Measurements:\n"
        f"  Amplitude         : {features.amplitude_mm} mm\n"
        f"  Dominant Frequency: {features.dominant_frequency_hz} Hz\n"
        f"  Symmetry Score    : {features.symmetry_score} (1.0 = symmetric)\n"
        f"  Right Hand        : {features.right_hand_frequency} Hz, {features.right_hand_amplitude} mm\n"
        f"  Left Hand         : {features.left_hand_frequency} Hz, {features.left_hand_amplitude} mm\n"
        f"  Preliminary Risk  : {features.risk_level}\n"
        f"  FTM Severity      : {severity.upper()} (Grade {ftm_score}/4)\n"
        f"  Notes             : {features.notes}\n\n"
        "Respond ONLY with JSON. Use ASCII characters only -- no dashes, no smart quotes, no special symbols.\n"
        "Start immediately with {:\n"
        "{\n"
        '  "summary": "2-3 sentence plain English summary of findings",\n'
        '  "findings": "3-4 sentences describing what the measurements show clinically",\n'
        '  "risk_assessment": "1-2 sentences on the risk level and what it means",\n'
        '  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],\n'
        '  "disclaimer": "one sentence screening tool disclaimer"\n'
        "}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content":
                    "You are a clinical neurologist AI. "
                    "Output only valid JSON. ASCII characters only. "
                    "No em dashes, no smart quotes, no Unicode symbols. "
                    "Start your response with { immediately."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        raw   = content.strip()
        start = raw.rfind("{")   # rfind skips chain-of-thought preamble
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")
        return json.loads(raw[start:end + 1])
    except Exception as e:
        return {
            "summary":         "Report generation encountered an error.",
            "findings":        str(e),
            "risk_assessment": "Unable to assess.",
            "recommendations": ["Please re-run the analysis."],
            "disclaimer":      "This is a screening tool only. Not a medical diagnosis.",
        }


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_pdf(features, severity: str, ftm_score: int, report: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.65 * inch,  bottomMargin=0.75 * inch,
    )

    W = 7.0 * inch
    sev_bg, sev_fg, sev_bar = _sev_colors(severity)
    story = []

    # -- shared styles --------------------------------------------------------
    body_j   = ParagraphStyle("body_j",  fontName="Helvetica",      fontSize=10,
                              textColor=_INK,   leading=17, alignment=TA_JUSTIFY)
    tbl_hdr  = ParagraphStyle("tbl_hdr", fontName="Helvetica-Bold", fontSize=8,
                              textColor=_WHITE)
    tbl_cell = ParagraphStyle("tbl_cell",fontName="Helvetica",      fontSize=9,
                              textColor=_INK,  leading=14)
    tbl_bold = ParagraphStyle("tbl_bold",fontName="Helvetica-Bold", fontSize=9,
                              textColor=_INK,  leading=14)

    _no_pad = TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ])

    # =========================================================================
    # Header
    # =========================================================================
    hdr = Table([[
        Paragraph("<b>SENTINEL</b>",
            ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=22,
                           textColor=_INK, leading=26)),
        Paragraph(
            f"Tremor Screening Report<br/>"
            f"<font color='#9ca3af'>"
            f"{datetime.now().strftime('%B %d, %Y  -  %I:%M %p')}"
            f"</font>",
            ParagraphStyle("dr", fontName="Helvetica", fontSize=9,
                           textColor=_SUBTEXT, leading=14, alignment=TA_RIGHT)),
    ]], colWidths=[W * 0.55, W * 0.45])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=W, thickness=1.5, color=sev_bar))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Badge  -- EACH text is its OWN row so ReportLab sizes them correctly.
    # Putting [Para, Spacer, Para, Spacer, Para] as a LIST in ONE cell caused
    # overlap because Spacers are not processed inside cell lists.
    # =========================================================================
    badge = Table(
        [
            [Paragraph(
                severity.upper(),
                ParagraphStyle("bsev", fontName="Helvetica-Bold", fontSize=30,
                               textColor=sev_fg, leading=34, alignment=TA_CENTER),
            )],
            [Paragraph(
                f"Fahn-Tolosa-Marin Grade <b>{ftm_score}</b> / 4",
                ParagraphStyle("bftm", fontName="Helvetica", fontSize=10,
                               textColor=sev_fg, leading=14, alignment=TA_CENTER),
            )],
            [Spacer(1, 6)],
            [Paragraph(
                "ASSESSMENT RESULT",
                ParagraphStyle("basr", fontName="Helvetica-Bold", fontSize=7,
                               textColor=sev_fg, leading=10, alignment=TA_CENTER),
            )],
        ],
        colWidths=[W * 0.36],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), sev_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, sev_bar),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        # Extra breathing room for the big severity label row
        ("TOPPADDING",    (0, 0), (0, 0),   16),
        ("BOTTOMPADDING", (0, 3), (0, 3),   14),
    ]))

    # -- metric mini-cards ----------------------------------------------------
    GAP = 8
    CW  = (W * 0.60 - GAP) / 2

    def mini(label: str, val: str, unit: str = "") -> Table:
        rows = [
            [Paragraph(f"<b>{_safe(val)}</b>",
                ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=16,
                               textColor=_INK, leading=20))],
        ]
        if unit:
            rows.append([Paragraph(
                _safe(unit),
                ParagraphStyle("mu", fontName="Helvetica", fontSize=8,
                               textColor=_SUBTEXT, leading=11))])
        rows.append([Spacer(1, 3)])
        rows.append([Paragraph(
            _safe(label).upper(),
            ParagraphStyle("ml", fontName="Helvetica-Bold", fontSize=7,
                           textColor=_SUBTEXT, leading=9))])
        t = Table(rows, colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _STRIPE),
            ("BOX",           (0, 0), (-1, -1), 0.5, _RULE),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (0, 0),   10),
            ("BOTTOMPADDING", (0, -1), (0, -1), 10),
        ]))
        return t

    row1 = Table(
        [[mini("Amplitude", f"{features.amplitude_mm}", "mm"),
          Spacer(GAP, 1),
          mini("Frequency", f"{features.dominant_frequency_hz}", "Hz")]],
        colWidths=[CW, GAP, CW],
    )
    row1.setStyle(_no_pad)

    row2 = Table(
        [[mini("Tremor Type", features.tremor_type.capitalize()),
          Spacer(GAP, 1),
          mini("Symmetry", f"{features.symmetry_score}", "/ 1.0")]],
        colWidths=[CW, GAP, CW],
    )
    row2.setStyle(_no_pad)

    metrics_col = Table([[row1], [Spacer(1, GAP)], [row2]], colWidths=[W * 0.60])
    metrics_col.setStyle(_no_pad)

    top = Table(
        [[badge, Spacer(14, 1), metrics_col]],
        colWidths=[W * 0.36, 14, W * 0.60],
    )
    top.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(top)
    story.append(Spacer(1, 22))

    # =========================================================================
    # Summary
    # =========================================================================
    story.extend(_section_header("Summary", W, sev_bar))
    story.append(Paragraph(_safe(report.get("summary", "")), body_j))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Measurements table
    # =========================================================================
    story.extend(_section_header("Measurements", W, sev_bar))
    mrows = [
        [Paragraph("PARAMETER", tbl_hdr), Paragraph("VALUE", tbl_hdr)],
        [Paragraph("Amplitude",          tbl_cell), Paragraph(f"{features.amplitude_mm} mm",                                                       tbl_bold)],
        [Paragraph("Dominant Frequency", tbl_cell), Paragraph(f"{features.dominant_frequency_hz} Hz",                                              tbl_bold)],
        [Paragraph("Tremor Type",        tbl_cell), Paragraph(_safe(features.tremor_type.capitalize()),                                             tbl_bold)],
        [Paragraph("Symmetry Score",     tbl_cell), Paragraph(f"{features.symmetry_score} / 1.0",                                                  tbl_bold)],
        [Paragraph("Right Hand",         tbl_cell), Paragraph(f"{features.right_hand_frequency} Hz  |  {features.right_hand_amplitude} mm",        tbl_bold)],
        [Paragraph("Left Hand",          tbl_cell), Paragraph(f"{features.left_hand_frequency} Hz  |  {features.left_hand_amplitude} mm",          tbl_bold)],
        [Paragraph("Preliminary Risk",   tbl_cell), Paragraph(_safe(features.risk_level.upper()),                                                   tbl_bold)],
    ]
    mt = Table(mrows, colWidths=[W * 0.38, W * 0.62])
    mt.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0),  (-1, 0),  _INK),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1), [_STRIPE, _WHITE]),
        ("GRID",           (0, 0),  (-1, -1), 0.4, _RULE),
        ("TOPPADDING",     (0, 0),  (-1, -1), 8),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 8),
        ("LEFTPADDING",    (0, 0),  (-1, -1), 14),
        ("RIGHTPADDING",   (0, 0),  (-1, -1), 10),
        ("VALIGN",         (0, 0),  (-1, -1), "MIDDLE"),
    ]))
    story.append(mt)
    story.append(Spacer(1, 18))

    # =========================================================================
    # Clinical Findings
    # =========================================================================
    story.extend(_section_header("Clinical Findings", W, sev_bar))
    story.append(Paragraph(_safe(report.get("findings", "")), body_j))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Risk Assessment
    # =========================================================================
    story.extend(_section_header("Risk Assessment", W, sev_bar))
    risk_box = Table([[
        Paragraph(
            _safe(report.get("risk_assessment", "")),
            ParagraphStyle("rb", fontName="Helvetica", fontSize=10,
                           textColor=sev_fg, leading=17),
        ),
    ]], colWidths=[W])
    risk_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), sev_bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("LINEBEFORE",    (0, 0), (0, -1),  3, sev_bar),
    ]))
    story.append(risk_box)
    story.append(Spacer(1, 18))

    # =========================================================================
    # Recommendations
    # =========================================================================
    story.extend(_section_header("Recommendations", W, sev_bar))
    for i, rec in enumerate(report.get("recommendations", []), 1):
        bullet = Table([[
            Paragraph(f"<b>{i}</b>",
                ParagraphStyle("bn", fontName="Helvetica-Bold", fontSize=10,
                               textColor=_WHITE, leading=14, alignment=TA_CENTER)),
            Paragraph(_safe(rec),
                ParagraphStyle("rt", fontName="Helvetica", fontSize=10,
                               textColor=_INK, leading=16)),
        ]], colWidths=[22, W - 22])
        bullet.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0),  sev_bar),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (0, 0),  0),
            ("RIGHTPADDING",  (0, 0), (0, 0),  0),
            ("LEFTPADDING",   (1, 0), (1, -1), 12),
            ("RIGHTPADDING",  (1, 0), (1, -1), 8),
        ]))
        story.append(bullet)
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))

    # =========================================================================
    # Footer  -- ASCII only: [!] not warning sign, x not x, - not dot
    # =========================================================================
    story.append(HRFlowable(width=W, thickness=0.5, color=_RULE))
    story.append(Spacer(1, 8))
    disclaimer = _safe(report.get(
        "disclaimer",
        "This report is generated by an AI screening tool and does not constitute a medical diagnosis.",
    ))
    story.append(Paragraph(
        f"[!]  {disclaimer}",
        ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=_SUBTEXT, leading=12, alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "SENTINEL  x  Nemotron 120B  -  Always consult a qualified neurologist.",
        ParagraphStyle("foot", fontName="Helvetica", fontSize=7,
                       textColor=colors.HexColor("#9ca3af"), leading=11,
                       alignment=TA_CENTER),
    ))

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(features, severity: str, ftm_score: int) -> bytes:
    report_text = generate_report_text(features, severity, ftm_score)
    return build_pdf(features, severity, ftm_score, report_text)