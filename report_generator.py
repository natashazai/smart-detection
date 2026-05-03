"""
SENTINEL — Clinical Report Generator
======================================
Asks Nemotron 120B to write a structured clinical report
and saves it as a polished PDF the patient can bring to a doctor.
"""

import json
import time
from datetime import datetime
from io import BytesIO
from openai import OpenAI

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

MODEL = "nvidia/nemotron-3-super-120b-a12b"


def generate_report_text(features, severity: str, ftm_score: int) -> dict:
    prompt = f"""You are a clinical neurologist AI writing a structured screening report.
A patient completed a 30-second hand tremor assessment. Write a professional report.

Measurements:
  Amplitude         : {features.amplitude_mm} mm
  Dominant Frequency: {features.dominant_frequency_hz} Hz
  Tremor Type       : {features.tremor_type}
  Symmetry Score    : {features.symmetry_score} (1.0 = symmetric)
  Right Hand        : {features.right_hand_frequency} Hz, {features.right_hand_amplitude} mm
  Left Hand         : {features.left_hand_frequency} Hz, {features.left_hand_amplitude} mm
  Preliminary Risk  : {features.risk_level}
  FTM Severity      : {severity.upper()} (Grade {ftm_score}/4)
  Notes             : {features.notes}

Respond ONLY with JSON, no markdown, start with {{:
{{
  "summary": "2-3 sentence plain English summary of findings",
  "findings": "3-4 sentences describing what the measurements show clinically",
  "risk_assessment": "1-2 sentences on the risk level and what it means",
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "disclaimer": "one sentence screening tool disclaimer"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical neurologist AI. Output only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response")
        raw   = content.strip()
        start = raw.find('{')
        end   = raw.rfind('}')
        return json.loads(raw[start:end+1])
    except Exception as e:
        return {
            "summary":         "Report generation encountered an error.",
            "findings":        str(e),
            "risk_assessment": "Unable to assess.",
            "recommendations": ["Please re-run the analysis."],
            "disclaimer":      "This is a screening tool only. Not a medical diagnosis.",
        }


def build_pdf(features, severity: str, ftm_score: int, report: dict) -> bytes:
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.6*inch,   bottomMargin=0.75*inch,
    )

    W = 7.0 * inch   # usable width

    severity_colors = {
        "none":     "#22c55e",
        "mild":     "#84cc16",
        "moderate": "#f59e0b",
        "marked":   "#f97316",
        "severe":   "#ef4444",
    }
    sev_hex   = severity_colors.get(severity.lower(), "#6b7280")
    sev_color = colors.HexColor(sev_hex)

    DARK      = colors.HexColor("#0f172a")
    MID       = colors.HexColor("#1e293b")
    SLATE     = colors.HexColor("#334155")
    MUTED     = colors.HexColor("#64748b")
    LIGHT     = colors.HexColor("#e2e8f0")
    WHITE     = colors.white

    story = []

    # ── Top accent bar ────────────────────────────────────────────
    accent = Table([["  "]], colWidths=[W])
    accent.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), sev_color),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(accent)
    story.append(Spacer(1, 14))

    # ── Header: logo left, date right ────────────────────────────
    header_data = [[
        Paragraph("<font size=20><b>SENTINEL</b></font>", ParagraphStyle("hl",
            fontName="Helvetica-Bold", fontSize=20,
            textColor=DARK)),
        Paragraph(
            f"<font size=8 color='#64748b'>Tremor Screening Report<br/>"
            f"{datetime.now().strftime('%B %d, %Y  ·  %I:%M %p')}</font>",
            ParagraphStyle("hr", fontName="Helvetica", fontSize=8,
                textColor=MUTED, alignment=TA_RIGHT)),
    ]]
    ht = Table(header_data, colWidths=[W*0.5, W*0.5])
    ht.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(ht)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=W, thickness=0.5, color=LIGHT))
    story.append(Spacer(1, 16))

    # ── Severity badge — single cell, all content in one paragraph ─
    badge_data = [[
        Paragraph(
            f"<para align='center'>"
            f"<font size=9 color='#ffffff'>ASSESSMENT RESULT</font><br/><br/>"
            f"<font size=40 color='#ffffff'><b>{severity.upper()}</b></font><br/><br/>"
            f"<font size=10 color='#ffffff'>Fahn-Tolosa-Marin Grade {ftm_score} / 4</font>"
            f"</para>",
            ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=40,
                textColor=WHITE, alignment=TA_CENTER)),
    ]]
    bt = Table(badge_data, colWidths=[W])
    bt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), sev_color),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(bt)
    story.append(Spacer(1, 22))

    # ── Helper: section heading ───────────────────────────────────
    def section(title):
        t = Table([[
            Paragraph(f"<font size=11><b>{title}</b></font>",
                ParagraphStyle("sh", fontName="Helvetica-Bold",
                    fontSize=11, textColor=WHITE)),
        ]], colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), MID),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    body_style = ParagraphStyle("body",
        fontName="Helvetica", fontSize=10,
        textColor=SLATE, leading=17, spaceAfter=10)

    # ── Summary ───────────────────────────────────────────────────
    section("Summary")
    story.append(Paragraph(report.get("summary", ""), body_style))
    story.append(Spacer(1, 8))

    # ── Measurements table ────────────────────────────────────────
    section("Measurements")
    rows = [
        ["Parameter",          "Value"],
        ["Amplitude",          f"{features.amplitude_mm} mm"],
        ["Dominant Frequency", f"{features.dominant_frequency_hz} Hz"],
        ["Tremor Type",        features.tremor_type.capitalize()],
        ["Symmetry Score",     f"{features.symmetry_score} / 1.0"],
        ["Right Hand",         f"{features.right_hand_frequency} Hz   |   {features.right_hand_amplitude} mm"],
        ["Left Hand",          f"{features.left_hand_frequency} Hz   |   {features.left_hand_amplitude} mm"],
        ["Preliminary Risk",   features.risk_level.upper()],
    ]
    mt = Table(rows, colWidths=[W*0.35, W*0.65])
    mt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  DARK),
        ("TEXTCOLOR",      (0,0), (-1,0),  WHITE),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), WHITE]),
        ("TEXTCOLOR",      (0,1), (-1,-1), SLATE),
        ("GRID",           (0,0), (-1,-1), 0.4, LIGHT),
        ("TOPPADDING",     (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 7),
        ("LEFTPADDING",    (0,0), (-1,-1), 12),
    ]))
    story.append(mt)
    story.append(Spacer(1, 14))

    # ── Clinical findings ─────────────────────────────────────────
    section("Clinical Findings")
    story.append(Paragraph(report.get("findings", ""), body_style))
    story.append(Spacer(1, 8))

    # ── Risk assessment ───────────────────────────────────────────
    section("Risk Assessment")
    story.append(Paragraph(report.get("risk_assessment", ""), body_style))
    story.append(Spacer(1, 8))

    # ── Recommendations ───────────────────────────────────────────
    section("Recommendations")
    for i, rec in enumerate(report.get("recommendations", []), 1):
        story.append(Paragraph(
            f"<b>{i}.</b>  {rec}",
            ParagraphStyle("rec", fontName="Helvetica", fontSize=10,
                textColor=SLATE, leading=17, leftIndent=8, spaceAfter=6)
        ))
    story.append(Spacer(1, 20))

    # ── Footer ────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=LIGHT))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<font size=8 color='#94a3b8'>"
        f"&#9888;  {report.get('disclaimer', 'This report is generated by an AI screening tool and does not constitute a medical diagnosis.')}"
        f"</font>",
        ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=8,
            textColor=colors.HexColor("#94a3b8"), leading=12, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<font size=7 color='#cbd5e1'>SENTINEL  ×  Nemotron 120B  ·  Always consult a qualified neurologist.</font>",
        ParagraphStyle("foot", fontName="Helvetica", fontSize=7,
            textColor=LIGHT, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buffer.getvalue()


def generate_report(features, severity: str, ftm_score: int) -> bytes:
    report_text = generate_report_text(features, severity, ftm_score)
    return build_pdf(features, severity, ftm_score, report_text)