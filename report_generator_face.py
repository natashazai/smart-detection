"""
SENTINEL -- Facial Expressivity Clinical Report Generator
==========================================================
Asks Nemotron 120B to write a structured clinical report for the
hypomimia (smile-video) screening mode and saves it as a PDF.

Mirrors report_generator.py layout rules:
  * All text through _safe() -- strips non-Latin-1 chars.
  * No Unicode decorators: x (not x), - (not em/en dash), [!] (not warning sign).
  * Badge uses a multi-row Table so ReportLab sizes rows independently.
"""

import json
import os
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

load_dotenv()

MODEL = "nvidia/nemotron-3-super-120b-a12b"


# ---------------------------------------------------------------------------
# Text sanitiser
# ---------------------------------------------------------------------------

def _safe(text) -> str:
    """Replace non-Latin-1 chars that Helvetica renders as black boxes."""
    if not isinstance(text, str):
        text = str(text)
    subs = [
        ("\u2014", " - "), ("\u2013", " - "),
        ("\u2018", "'"),   ("\u2019", "'"),
        ("\u201c", '"'),   ("\u201d", '"'),
        ("\u00d7", "x"),   ("\u00b7", "-"),
        ("\u25a0", ""),    ("\u25cf", ""),
        ("\u2022", "-"),   ("\u2026", "..."),
        ("\u00ae", "(R)"), ("\u00a9", "(C)"),
    ]
    for bad, good in subs:
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_INK     = colors.HexColor("#111827")
_SUBTEXT = colors.HexColor("#6b7280")
_RULE    = colors.HexColor("#e5e7eb")
_STRIPE  = colors.HexColor("#f9fafb")
_WHITE   = colors.white

_RISK_PALETTE = {
    "low":      ("#dcfce7", "#166534", "#15803d"),
    "moderate": ("#fff7ed", "#9a3412", "#ea580c"),
    "high":     ("#fef2f2", "#991b1b", "#dc2626"),
    "unknown":  ("#f3f4f6", "#374151", "#6b7280"),
}


def _risk_colors(risk: str):
    bg, fg, bar = _RISK_PALETTE.get(risk.lower(), _RISK_PALETTE["unknown"])
    return colors.HexColor(bg), colors.HexColor(fg), colors.HexColor(bar)


# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------

def _section_header(title: str, W: float, bar_color) -> list:
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

def generate_report_text(features, result: dict) -> dict:
    """Ask Nemotron to write the narrative sections of the report."""
    prompt = (
        "You are a clinical neurologist AI writing a structured screening report.\n"
        "A patient completed a facial expressivity assessment (smile video task) "
        "for Parkinson's disease hypomimia screening.\n\n"
        "Measurements:\n"
        f"  Hypomimia Score   : {features.hypomimia_score:.2f} / 1.0 "
        f"(0 = no hypomimia, 1 = severe)\n"
        f"  Smile Amplitude   : {features.smile_amplitude:.3f} "
        f"(healthy: 0.18-0.26 normalized)\n"
        f"  Smile Variability : {features.smile_variability:.3f} "
        f"(healthy: 0.02-0.06; low = mask-like)\n"
        f"  Cheek Elevation   : {features.cheek_elevation:.3f} "
        f"(AU6 proxy; healthy: 0.05-0.15)\n"
        f"  Facial Movement   : {features.facial_movement:.5f} "
        f"(kinetic energy; healthy: 0.002-0.008)\n"
        f"  Signal Confidence : {int(features.confidence * 100)}%\n"
        f"  Preliminary Risk  : {features.risk_level.upper()}\n"
        f"  Analysis Notes    : {features.notes}\n\n"
        "Clinical context: Hypomimia is reduced facial expressivity, a prominent motor "
        "symptom of Parkinson's disease. AU12 (lip-corner pull) and AU6 (cheek raise) "
        "are the most discriminative features. This tool is based on methods from "
        "Adnan et al., NEJM AI 2025.\n\n"
        "Respond ONLY with JSON. Use ASCII characters only -- no dashes, no smart quotes.\n"
        "Start immediately with {:\n"
        "{\n"
        '  "summary": "2-3 sentence plain English summary of the facial expressivity findings",\n'
        '  "findings": "3-4 sentences describing what the measurements show clinically",\n'
        '  "risk_assessment": "1-2 sentences on what the hypomimia risk level means",\n'
        '  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],\n'
        '  "disclaimer": "one sentence screening tool disclaimer"\n'
        "}"
    )
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
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
        raw   = (response.choices[0].message.content or "").strip()
        start = raw.rfind("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")
        return json.loads(raw[start : end + 1])
    except Exception as exc:
        # Graceful fallback — use what the UI already computed
        return {
            "summary":         result.get("interpretation", "Report generation encountered an error."),
            "findings":        features.notes,
            "risk_assessment": f"Hypomimia risk is {features.risk_level.upper()} "
                               f"(score {features.hypomimia_score:.2f}/1.0).",
            "recommendations": [
                result.get("recommendation", "Consult a qualified neurologist."),
                "Repeat the assessment in a well-lit environment for higher confidence.",
                "Bring this report to your next medical appointment.",
            ],
            "disclaimer": "This report is generated by an AI screening tool and does not constitute a medical diagnosis.",
        }


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_face_pdf(features, result: dict, report: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.65 * inch,  bottomMargin=0.75 * inch,
    )

    W = 7.0 * inch
    risk      = features.risk_level
    score_pct = int(features.hypomimia_score * 100)
    risk_bg, risk_fg, risk_bar = _risk_colors(risk)
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
            f"Facial Expressivity Report<br/>"
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
    story.append(HRFlowable(width=W, thickness=1.5, color=risk_bar))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Badge
    # =========================================================================
    badge = Table(
        [
            [Paragraph(
                risk.upper(),
                ParagraphStyle("brisk", fontName="Helvetica-Bold", fontSize=30,
                               textColor=risk_fg, leading=34, alignment=TA_CENTER),
            )],
            [Paragraph(
                f"Hypomimia Score  <b>{score_pct}</b> / 100",
                ParagraphStyle("bscr", fontName="Helvetica", fontSize=10,
                               textColor=risk_fg, leading=14, alignment=TA_CENTER),
            )],
            [Spacer(1, 6)],
            [Paragraph(
                "HYPOMIMIA RISK",
                ParagraphStyle("blbl", fontName="Helvetica-Bold", fontSize=7,
                               textColor=risk_fg, leading=10, alignment=TA_CENTER),
            )],
        ],
        colWidths=[W * 0.36],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), risk_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, risk_bar),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
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
        [[mini("Smile Amplitude",   f"{features.smile_amplitude:.3f}", "normalized"),
          Spacer(GAP, 1),
          mini("Smile Variability", f"{features.smile_variability:.3f}", "std dev")]],
        colWidths=[CW, GAP, CW],
    )
    row1.setStyle(_no_pad)

    row2 = Table(
        [[mini("Cheek Elevation", f"{features.cheek_elevation:.3f}", "AU6 proxy"),
          Spacer(GAP, 1),
          mini("Confidence",      f"{int(features.confidence * 100)}%")]],
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
    story.extend(_section_header("Summary", W, risk_bar))
    story.append(Paragraph(_safe(report.get("summary", "")), body_j))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Measurements table
    # =========================================================================
    story.extend(_section_header("Measurements", W, risk_bar))
    mrows = [
        [Paragraph("PARAMETER",         tbl_hdr), Paragraph("VALUE",                                                tbl_hdr)],
        [Paragraph("Hypomimia Score",    tbl_cell), Paragraph(f"{score_pct} / 100",                                 tbl_bold)],
        [Paragraph("Smile Amplitude",    tbl_cell), Paragraph(f"{features.smile_amplitude:.3f}  (healthy 0.18-0.26)", tbl_bold)],
        [Paragraph("Smile Variability",  tbl_cell), Paragraph(f"{features.smile_variability:.3f}  (healthy 0.02-0.06)", tbl_bold)],
        [Paragraph("Cheek Elevation",    tbl_cell), Paragraph(f"{features.cheek_elevation:.3f}  (AU6 proxy; healthy 0.05-0.15)", tbl_bold)],
        [Paragraph("Facial Movement",    tbl_cell), Paragraph(f"{features.facial_movement:.5f}  (kinetic energy)", tbl_bold)],
        [Paragraph("Signal Confidence",  tbl_cell), Paragraph(f"{int(features.confidence * 100)}%",                tbl_bold)],
        [Paragraph("Preliminary Risk",   tbl_cell), Paragraph(_safe(risk.upper()),                                  tbl_bold)],
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
    story.extend(_section_header("Clinical Findings", W, risk_bar))
    story.append(Paragraph(_safe(report.get("findings", "")), body_j))
    story.append(Spacer(1, 18))

    # =========================================================================
    # Risk Assessment
    # =========================================================================
    story.extend(_section_header("Risk Assessment", W, risk_bar))
    risk_box = Table([[
        Paragraph(
            _safe(report.get("risk_assessment", "")),
            ParagraphStyle("rb", fontName="Helvetica", fontSize=10,
                           textColor=risk_fg, leading=17),
        ),
    ]], colWidths=[W])
    risk_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), risk_bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("LINEBEFORE",    (0, 0), (0, -1),  3, risk_bar),
    ]))
    story.append(risk_box)
    story.append(Spacer(1, 18))

    # =========================================================================
    # Recommendations
    # =========================================================================
    story.extend(_section_header("Recommendations", W, risk_bar))
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
            ("BACKGROUND",    (0, 0), (0, 0),  risk_bar),
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
    # Method note
    # =========================================================================
    story.extend(_section_header("Methodology", W, risk_bar))
    story.append(Paragraph(
        _safe(
            "This assessment uses computer vision to detect hypomimia (reduced facial "
            "expressivity), a motor symptom of Parkinson's disease. MediaPipe FaceLandmarker "
            "extracts 478 facial landmarks per frame. Key features -- smile amplitude (AU12 "
            "proxy), smile variability, cheek elevation (AU6 proxy), and overall facial "
            "kinetic energy -- are computed from a short smile-task recording. The method "
            "is based on: Adnan et al., AI-Enabled Parkinson's Disease Screening Using "
            "Smile Videos, NEJM AI 2025 (doi:10.1056/AIoa2400950)."
        ),
        ParagraphStyle("method", fontName="Helvetica", fontSize=9,
                       textColor=_SUBTEXT, leading=15, alignment=TA_JUSTIFY),
    ))
    story.append(Spacer(1, 20))

    # =========================================================================
    # Footer
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

def generate_face_report(features, result: dict) -> bytes:
    """
    Generate a facial expressivity PDF report.

    Args:
        features: FacialFeatures dataclass from facial_analysis.analyze_facial_expression
        result:   dict from facial_analysis.classify_hypomimia_with_nemotron
                  (already contains interpretation + recommendation)

    Returns:
        PDF as bytes, ready for st.download_button.
    """
    report_text = generate_report_text(features, result)
    return build_face_pdf(features, result, report_text)