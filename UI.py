"""Streamlit dashboard for SENTINEL tremor analysis.

Run with:
    streamlit run dashboard.py

Make sure tremor_analysis.py and report_generator.py are in the same folder.
"""

from __future__ import annotations

import json
import time
import streamlit as st
from openai import OpenAI

from tremor_analysis import (
    generate_mock_hand_data,
    analyze_tremor,
    classify_with_nemotron,
)
from report_generator import generate_report

from dotenv import load_dotenv
import os
load_dotenv()

nemotron_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"

SEVERITY_COLOR = {
    "none":     "#22c55e",
    "mild":     "#84cc16",
    "moderate": "#f59e0b",
    "marked":   "#f97316",
    "severe":   "#ef4444",
    "unknown":  "#6b7280",
    "error":    "#6b7280",
}

SCENARIOS = [
    {"name": "Healthy baseline",      "frequency": 10.0, "amplitude": 0.05, "noise": 0.3},
    {"name": "Mild essential tremor", "frequency":  7.0, "amplitude": 2.0,  "noise": 0.5},
    {"name": "Moderate Parkinson's",  "frequency":  5.2, "amplitude": 4.0,  "noise": 0.5},
    {"name": "Severe tremor",         "frequency":  4.5, "amplitude": 15.0, "noise": 1.0},
]



def get_explanation(features, severity: str, ftm: int) -> str:
    """Ask Nemotron to explain the result in plain English for the patient."""
    try:
        response = nemotron_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly clinical AI explaining tremor results to a patient. Be warm, clear, and concise. Never diagnose. Always recommend seeing a doctor."
                },
                {
                    "role": "user",
                    "content": (
                        f"The patient's tremor was classified as {severity.upper()} (FTM grade {ftm}/4).\n"
                        f"Amplitude: {features.amplitude_mm} mm, Frequency: {features.dominant_frequency_hz} Hz, "
                        f"Symmetry: {features.symmetry_score}, Risk: {features.risk_level}.\n\n"
                        f"Write 2-3 sentences explaining what this means in plain English. "
                        f"What should the patient know? What should they do next? "
                        f"Be reassuring but honest. No markdown, no bullet points, just plain text."
                    )
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Unable to generate explanation."
    except Exception as e:
        return f"Could not generate explanation: {e}"


def main() -> None:
    st.set_page_config(page_title="SENTINEL — Tremor Monitor", layout="wide")

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #0a0f1e; }
    [data-testid="stSidebar"] { background: #0f172a; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, label { color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Tremor Monitor")
    st.caption("Nemotron 120B severity classification · Not a medical diagnostic device")
    st.divider()

    with st.sidebar:
        st.subheader("Test Scenario")
        scenario_name = st.selectbox(
            "Select a tremor profile",
            [s["name"] for s in SCENARIOS],
        )
        scenario = next(s for s in SCENARIOS if s["name"] == scenario_name)

        st.subheader("Mock Data Parameters")
        st.caption(f"Frequency : {scenario['frequency']} Hz")
        st.caption(f"Amplitude : {scenario['amplitude']} mm")
        st.caption(f"Noise     : {scenario['noise']}")

        run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    # store results in session
    for key in ["last_features", "last_severity", "last_ftm", "last_explanation"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if run:
        with st.spinner("Running analysis and calling Nemotron 120B..."):
            hand_data = generate_mock_hand_data(
                duration_seconds=30, sample_rate=30,
                tremor_frequency=scenario["frequency"],
                tremor_amplitude=scenario["amplitude"],
                noise_level=scenario["noise"],
            )
            features = analyze_tremor(hand_data)
            result   = classify_with_nemotron(features.amplitude_mm)
            severity = result.get("severity", "unknown")
            ftm      = result.get("ftm_score", "?")
            explanation = get_explanation(features, severity, ftm)

        st.session_state.last_features    = features
        st.session_state.last_severity    = severity
        st.session_state.last_ftm         = ftm
        st.session_state.last_explanation = explanation

    if st.session_state.last_features:
        features    = st.session_state.last_features
        severity    = st.session_state.last_severity
        ftm         = st.session_state.last_ftm
        explanation = st.session_state.last_explanation
        color       = SEVERITY_COLOR.get(severity, "#6b7280")

        # severity badge
        st.markdown(
            f"""<div style='background:#0f172a;border:2px solid {color};border-radius:16px;
                            padding:40px;text-align:center;margin-bottom:24px;'>
                <p style='color:#94a3b8;font-size:13px;letter-spacing:3px;
                          text-transform:uppercase;margin:0 0 12px 0;'>
                    Nemotron 120B Assessment
                </p>
                <p style='color:{color};font-size:72px;font-weight:900;
                          letter-spacing:6px;margin:0;line-height:1;'>
                    {severity.upper()}
                </p>
                <p style='color:#64748b;font-size:16px;margin:14px 0 0 0;'>
                    FTM Grade {ftm} / 4
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

        # metrics row
        st.subheader("Signal Features")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amplitude",  f"{features.amplitude_mm} mm")
        col2.metric("Frequency",  f"{features.dominant_frequency_hz} Hz")
        col3.metric("Symmetry",   f"{features.symmetry_score}")
        col4.metric("Risk",       features.risk_level.upper())

        # Nemotron explanation
        st.subheader("What This Means")
        st.markdown(
            f"""<div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                            padding:20px 24px;'>
                <p style='color:#cbd5e1;font-size:15px;line-height:1.7;margin:0;'>
                    {explanation}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # report section
        st.subheader("Clinical Report")
        st.caption("Generate a PDF report you can bring to your doctor.")

        if st.button("Generate Report", type="primary"):
            with st.spinner("Nemotron is writing your clinical report..."):
                pdf_bytes = generate_report(features, severity, ftm)

            st.success("Report ready!")
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"sentinel_tremor_report_{severity}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    else:
        st.markdown(
            """<div style='background:#0f172a;border:1px solid #1e293b;border-radius:16px;
                           padding:60px;text-align:center;'>
                <p style='color:#334155;font-size:18px;margin:0;'>
                    Select a scenario and press Run Analysis
                </p>
            </div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()