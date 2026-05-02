"""Streamlit dashboard for SENTINEL tremor analysis.

Run with:
    streamlit run dashboard.py

Make sure tremor_analysis.py is in the same folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

from tremor_analysis import (
    generate_mock_hand_data,
    analyze_tremor,
    classify_with_nemotron,
)

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


def main() -> None:
    st.set_page_config(page_title="Tremor Monitor", layout="wide")

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #0a0f1e; }
    [data-testid="stSidebar"] { background: #0f172a; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, label { color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Tremor Monitor")
    st.caption("Nemotron 120B severity classification | Not a medical diagnostic device")
    st.divider()

    # sidebar — pick scenario
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

        run = st.button("Run Analysis", type="primary", use_container_width=True)

    if run:
        with st.spinner("Generating mock data and calling Nemotron 120B..."):
            # Person 2 — generate and analyze
            hand_data = generate_mock_hand_data(
                duration_seconds=30,
                sample_rate=30,
                tremor_frequency=scenario["frequency"],
                tremor_amplitude=scenario["amplitude"],
                noise_level=scenario["noise"],
            )
            features = analyze_tremor(hand_data)

            # Natasha — Nemotron classification
            result   = classify_with_nemotron(features.amplitude_mm)
            severity = result.get("severity", "unknown")
            ftm      = result.get("ftm_score", "?")
            color    = SEVERITY_COLOR.get(severity, "#6b7280")

        # big severity badge
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

        # feature breakdown
        st.subheader("Signal Features")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amplitude",  f"{features.amplitude_mm} mm")
        col2.metric("Frequency",  f"{features.dominant_frequency_hz} Hz")
        col3.metric("Symmetry",   f"{features.symmetry_score}")
        col4.metric("Risk",       features.risk_level.upper())

        st.subheader("Analysis Notes")
        st.info(features.notes if features.notes else "No significant tremor indicators detected.")

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