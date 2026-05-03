"""Streamlit dashboard for SENTINEL tremor analysis.

Run with:
    streamlit run UI.py
"""

from __future__ import annotations

import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from tremor_analysis import (
    analyze_tremor,
    classify_with_nemotron,
)
from facial_analysis import (
    analyze_facial_expression,
    classify_hypomimia_with_nemotron,
)
from pipeline import capture_hand_data_streaming, capture_face_data_streaming
from report_generator import generate_report

load_dotenv()

nemotron_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"

SEVERITY_COLOR = {
    "none":     "#16a34a",
    "mild":     "#2563eb",
    "moderate": "#d97706",
    "marked":   "#dc2626",
    "severe":   "#7f1d1d",
    "unknown":  "#6b7280",
    "error":    "#6b7280",
}

SEVERITY_BG = {
    "none":     "#f0fdf4",
    "mild":     "#eff6ff",
    "moderate": "#fffbeb",
    "marked":   "#fef2f2",
    "severe":   "#fef2f2",
    "unknown":  "#f9fafb",
    "error":    "#f9fafb",
}

# Capture options offered in the sidebar. Order matters: OAK is the default.
CAMERA_SOURCES = [
    {"id": "oak",     "label": "OAK-D Lite (RGB + depth)"},
    {"id": "oak-rgb", "label": "OAK-D Lite (RGB only)"},
    {"id": "webcam",  "label": "Webcam"},
]

HAND_OPTIONS = [
    {"id": "both",  "label": "Both hands"},
    {"id": "right", "label": "Left hand only"}, #These are reversed because it's from the camera's perspective which mirrors left and right
    {"id": "left",  "label": "Right hand only"},
    {"id": "auto",  "label": "Most confident hand"},
]


def get_explanation(features, severity: str, ftm: int) -> str:
    try:
        response = nemotron_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical AI. Write exactly 2-3 sentences explaining tremor results to a patient in plain English. Output only those sentences, nothing else."
                },
                {
                    "role": "user",
                    "content": (
                        f"Tremor severity: {severity.upper()} (FTM grade {ftm}/4). "
                        f"Amplitude: {features.amplitude_mm} mm, "
                        f"Frequency: {features.dominant_frequency_hz} Hz. "
                        f"Write 2-3 plain English sentences for the patient."
                    )
                }
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            return "Analysis complete. Please consult a neurologist for interpretation."
        sentences = content.strip().split(". ")
        return ". ".join(sentences[:3]).strip()
    except Exception as e:
        return f"Could not generate explanation: {e}"


# Preview rendering knobs. These do NOT affect capture FPS or sample quality —
# capture still runs as fast as the camera+MediaPipe loop allows; we just throttle
# how often the browser is asked to repaint.
PREVIEW_FPS         = 10           # max preview repaints per second
PROGRESS_UPDATE_HZ  = 4            # max progress-bar updates per second
PREVIEW_MAX_WIDTH   = 640          # downscale frames wider than this before sending
PREVIEW_JPEG_QUALITY = 70          # JPEG quality for the preview stream


def run_capture(
    *,
    source: str,
    hand: str,
    duration: float,
    fps: int,
    preview_slot,
    progress_slot,
    status_slot,
    sample_count_slot,
) -> dict | None:
    """Drive the streaming capture generator and update UI placeholders.

    Capture runs at full speed inside the generator. UI repaints are throttled
    so Streamlit's websocket isn't flooded with full-resolution frames.

    Returns the final hand_data dict, or None on failure.
    """
    import cv2

    final_hand_data: dict | None = None
    try:
        gen = capture_hand_data_streaming(
            duration_seconds=duration,
            source=source,
            hand=hand,
            fps=fps,
        )
    except Exception as exc:
        status_slot.error(f"Could not start camera: {exc}")
        return None

    preview_interval  = 1.0 / PREVIEW_FPS
    progress_interval = 1.0 / PROGRESS_UPDATE_HZ
    last_preview_t    = 0.0
    last_progress_t   = 0.0
    jpeg_params       = [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY]

    try:
        for preview_frame, elapsed, maybe_final in gen:
            if maybe_final is not None:
                final_hand_data = maybe_final
                break

            now = time.perf_counter()

            # Throttled preview repaint. Downscale, then JPEG-encode so we ship
            # ~50 KB per frame over the websocket instead of ~900 KB raw.
            if (now - last_preview_t) >= preview_interval:
                small = preview_frame
                h, w = small.shape[:2]
                if w > PREVIEW_MAX_WIDTH:
                    scale = PREVIEW_MAX_WIDTH / w
                    small = cv2.resize(
                        small,
                        (PREVIEW_MAX_WIDTH, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buf = cv2.imencode(".jpg", small, jpeg_params)
                if ok:
                    preview_slot.image(buf.tobytes(), width="stretch")
                last_preview_t = now

            # Throttled progress repaint.
            if (now - last_progress_t) >= progress_interval:
                pct = min(elapsed / duration, 1.0)
                progress_slot.progress(
                    pct,
                    text=f"Recording... {elapsed:0.1f}s / {duration:0.0f}s",
                )
                last_progress_t = now
    except Exception as exc:
        status_slot.error(f"Capture failed: {exc}")
        return None

    return final_hand_data


def render_processing_screen(slot, stage: str, detail: str) -> None:
    """Render the post-recording processing state in a stable placeholder."""
    slot.markdown(
        f"""
        <div class="processing-screen" role="status" aria-live="polite">
            <div class="processing-spinner"></div>
            <p class="processing-kicker">Recording complete</p>
            <p class="processing-title">{stage}</p>
            <p class="processing-detail">{detail}</p>
            <div class="processing-steps" aria-hidden="true">
                <span></span><span></span><span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


HYPO_RISK_COLOR = {
    "low":      "#16a34a",
    "moderate": "#d97706",
    "high":     "#dc2626",
    "unknown":  "#6b7280",
}
HYPO_RISK_BG = {
    "low":      "#f0fdf4",
    "moderate": "#fffbeb",
    "high":     "#fef2f2",
    "unknown":  "#f9fafb",
}


def run_face_capture(
    *,
    source: str,
    duration: float,
    preview_slot,
    progress_slot,
    status_slot,
):
    """Drive capture_face_data_streaming and update UI placeholders.
    Returns (N, 478, 3) landmark array, or None on failure.
    """
    import cv2

    final_landmarks = None
    try:
        gen = capture_face_data_streaming(
            duration_seconds=duration,
            source=source,
            fps=30,
        )
    except Exception as exc:
        status_slot.error(f"Could not start camera: {exc}")
        return None

    preview_interval  = 1.0 / PREVIEW_FPS
    progress_interval = 1.0 / PROGRESS_UPDATE_HZ
    last_preview_t    = 0.0
    last_progress_t   = 0.0
    jpeg_params       = [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY]

    try:
        for preview_frame, elapsed, maybe_final in gen:
            if maybe_final is not None:
                final_landmarks = maybe_final
                break

            now = time.perf_counter()

            if (now - last_preview_t) >= preview_interval:
                small = preview_frame
                h, w  = small.shape[:2]
                if w > PREVIEW_MAX_WIDTH:
                    scale = PREVIEW_MAX_WIDTH / w
                    small = cv2.resize(
                        small, (PREVIEW_MAX_WIDTH, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buf = cv2.imencode(".jpg", small, jpeg_params)
                if ok:
                    preview_slot.image(buf.tobytes(), width="stretch")
                last_preview_t = now

            if (now - last_progress_t) >= progress_interval:
                pct = min(elapsed / duration, 1.0)
                progress_slot.progress(
                    pct,
                    text=f"Recording... {elapsed:0.1f}s / {duration:0.0f}s — Keep smiling 😊",
                )
                last_progress_t = now

    except Exception as exc:
        status_slot.error(f"Capture failed: {exc}")
        return None

    return final_landmarks


def main() -> None:
    st.set_page_config(
        page_title="SENTINEL -- Neurological Screening",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"] { background-color: #f0f4f8; }

    [data-testid="stSidebar"] {
        background-color: #1a3a5c;
        border-right: none;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] * { color: #374151 !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #93c5fd !important;
        font-size: 15px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background-color: white !important;
        color: #374151 !important;
    }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100%; padding-left: 2rem; padding-right: 2rem; }

    h1, h2, h3, h4 { color: #1e3a5f !important; font-weight: 600 !important; }

    .stMetric { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; }
    .stMetric label { color: #64748b !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1e3a5f !important; font-size: 24px !important; font-weight: 600 !important; }

    [data-testid="stSidebar"] div[data-testid="stButton"] button {
        background-color: #2563eb !important; border: none !important;
        border-radius: 6px !important; font-weight: 600 !important;
        color: white !important; padding: 10px 20px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button:hover { background-color: #1d4ed8 !important; }

    .block-container div[data-testid="stButton"] button {
        background-color: #ffffff !important; border: 1.5px solid #1a3a5c !important;
        border-radius: 6px !important; font-weight: 600 !important;
        color: #1a3a5c !important; padding: 10px 20px !important;
    }
    .block-container div[data-testid="stButton"] button:hover { background-color: #eff6ff !important; }

    .stDownloadButton button {
        background-color: #ffffff !important; color: #1a3a5c !important;
        border: 1.5px solid #1a3a5c !important; border-radius: 6px !important; font-weight: 500 !important;
    }
    .stDownloadButton button:hover { background-color: #eff6ff !important; }

    hr { border-color: #e2e8f0; }

    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    footer { display: none; }

    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: 0 !important;
        transform: none !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
    }

    /* Camera preview frame */
    .preview-frame img {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: #0f172a;
    }

    .processing-screen {
        min-height: 360px;
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 56px 40px;
        margin: 24px 0;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .processing-spinner {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        border: 6px solid #dbeafe;
        border-top-color: #2563eb;
        animation: sentinel-spin 0.85s linear infinite;
        margin-bottom: 24px;
    }
    .processing-kicker {
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin: 0 0 8px 0;
        text-transform: uppercase;
    }
    .processing-title {
        color: #1e3a5f;
        font-size: 24px;
        font-weight: 700;
        line-height: 1.25;
        margin: 0 0 10px 0;
    }
    .processing-detail {
        color: #64748b;
        font-size: 14px;
        line-height: 1.7;
        max-width: 560px;
        margin: 0;
    }
    .processing-steps {
        display: flex;
        gap: 8px;
        margin-top: 24px;
    }
    .processing-steps span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2563eb;
        opacity: 0.35;
        animation: sentinel-pulse 1.2s ease-in-out infinite;
    }
    .processing-steps span:nth-child(2) { animation-delay: 0.16s; }
    .processing-steps span:nth-child(3) { animation-delay: 0.32s; }
    @keyframes sentinel-spin {
        to { transform: rotate(360deg); }
    }
    @keyframes sentinel-pulse {
        0%, 100% { opacity: 0.25; transform: scale(0.85); }
        50% { opacity: 1; transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Force sidebar open via JS
    st.markdown("""
    <script>
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        sidebar.setAttribute('aria-expanded', 'true');
        sidebar.style.transform = 'none';
        sidebar.style.visibility = 'visible';
    }
    const btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (btn) btn.style.display = 'none';
    </script>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style='background:#1a3a5c;padding:16px 32px;margin:0 0 32px 0;border-radius:8px;
                display:flex;align-items:center;justify-content:space-between;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <span style='font-size:22px;font-weight:700;color:white;letter-spacing:0.05em;'>SENTINEL</span>
            <span style='background:#2563eb;color:white;font-size:10px;font-weight:600;
                         padding:3px 8px;border-radius:4px;letter-spacing:0.08em;'>BETA</span>
        </div>
        <span style='color:#93c5fd;font-size:13px;'>Neurological Screening System - Powered by Nemotron 120B</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='background:#1a3a5c;padding:20px 16px 16px 16px;
                    margin:-60px -16px 20px -16px;border-bottom:1px solid #2d5a8e;'>
            <p style='color:#93c5fd;font-size:13px;letter-spacing:0.1em;
                      text-transform:uppercase;margin:0 0 4px 0;font-weight:700;'>SENTINEL</p>
            <p style='color:#4a7aaa;font-size:14px;margin:0;'>Patient Assessment Panel</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Screening Mode**")
        mode = st.radio(
            "Screening mode",
            ["🤚  Hand — Tremor", "😊  Face — Hypomimia"],
            label_visibility="collapsed",
        )
        is_face_mode = mode.startswith("😊")

        st.markdown("<hr style='border-color:#2d5a8e;margin:12px 0;'/>", unsafe_allow_html=True)
        st.markdown("**Camera Source**")

        if is_face_mode:
            # Webcam makes most sense for smile task; OAK-D RGB also works
            face_sources = [
                {"id": "webcam",  "label": "Webcam"},
                {"id": "oak-rgb", "label": "OAK-D Lite (RGB only)"},
                {"id": "oak",     "label": "OAK-D Lite (RGB + depth)"},
            ]
            face_source_label = st.selectbox(
                "Camera source",
                [s["label"] for s in face_sources],
                label_visibility="collapsed",
            )
            face_source_id = next(s["id"] for s in face_sources if s["label"] == face_source_label)

            st.markdown("**Recording Duration**")
            face_duration = st.slider(
                "Duration (seconds)",
                min_value=8, max_value=30, value=10, step=2,
                label_visibility="collapsed",
            )
            st.markdown(f"<p style='margin-top:-8px;'>{face_duration} seconds</p>", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:12px;line-height:1.6;'>"
                "Face the camera directly. When recording starts, hold a natural smile "
                "for the full duration. Good lighting and a neutral background help.</p>",
                unsafe_allow_html=True,
            )
            st.markdown("<br/>", unsafe_allow_html=True)
            run_face = st.button("Start Recording", type="primary", use_container_width=True)
            run = False  # hand mode not triggered

        else:
            source_label = st.selectbox(
                "Camera source",
                [s["label"] for s in CAMERA_SOURCES],
                label_visibility="collapsed",
            )
            source_id = next(s["id"] for s in CAMERA_SOURCES if s["label"] == source_label)

            st.markdown("**Hand Selection**")
            hand_label = st.selectbox(
                "Hand selection",
                [h["label"] for h in HAND_OPTIONS],
                label_visibility="collapsed",
            )
            hand_id = next(h["id"] for h in HAND_OPTIONS if h["label"] == hand_label)

            st.markdown("**Recording Duration**")
            duration = st.slider(
                "Duration (seconds)",
                min_value=10, max_value=60, value=30, step=5,
                label_visibility="collapsed",
            )
            st.markdown(f"<p style='margin-top:-8px;'>{duration} seconds</p>", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:12px;line-height:1.5;'>"
                "Hold the affected hand outstretched and steady within the camera's view. "
                "Recording begins immediately.</p>",
                unsafe_allow_html=True,
            )

            st.markdown("<br/>", unsafe_allow_html=True)
            run = st.button("Start Recording", type="primary", use_container_width=True)
            run_face = False  # face mode not triggered

    # Session state — hand tremor
    for key in ["last_features", "last_severity", "last_ftm",
                "last_explanation", "last_result", "last_metadata"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Session state — facial hypomimia
    for key in ["last_facial_features", "last_facial_result"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Face mode flow ─────────────────────────────────────────────────────────
    if is_face_mode:
        if run_face:
            st.markdown(
                "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;"
                "padding:14px 20px;margin-bottom:20px;'>"
                "<p style='color:#1e40af;font-size:14px;margin:0;'>"
                "😊 <strong>Smile task:</strong> Face the camera, then hold a comfortable "
                "natural smile for the entire recording. Try not to talk or look away.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            preview_slot   = st.empty()
            progress_slot  = st.empty()
            status_slot    = st.empty()

            landmarks = run_face_capture(
                source=face_source_id,
                duration=float(face_duration),
                preview_slot=preview_slot,
                progress_slot=progress_slot,
                status_slot=status_slot,
            )

            if landmarks is None:
                st.stop()

            if landmarks.shape[0] < 10:
                status_slot.error(
                    "No face was detected during recording. "
                    "Make sure your face is well-lit and centred in the frame, then try again."
                )
                st.stop()

            progress_slot.empty()
            preview_slot.empty()
            status_slot.success(
                f"Captured {landmarks.shape[0]} face-landmark frames "
                f"({landmarks.shape[0] / face_duration:.1f} fps effective)."
            )

            processing_slot = st.empty()
            render_processing_screen(
                processing_slot,
                "Analyzing facial expressivity",
                "SENTINEL is measuring smile amplitude, cheek elevation, and facial movement variability.",
            )
            facial_features = analyze_facial_expression(landmarks, sample_rate=30.0)

            render_processing_screen(
                processing_slot,
                "Nemotron is reviewing the assessment",
                "The clinical model is interpreting the facial expressivity profile.",
            )
            facial_result = classify_hypomimia_with_nemotron(facial_features)
            processing_slot.empty()

            st.session_state.last_facial_features = facial_features
            st.session_state.last_facial_result   = facial_result

        # Results panel — face
        if st.session_state.last_facial_features:
            ff     = st.session_state.last_facial_features
            fr     = st.session_state.last_facial_result
            risk   = ff.risk_level
            color  = HYPO_RISK_COLOR.get(risk, "#6b7280")
            bg     = HYPO_RISK_BG.get(risk, "#f9fafb")
            score_pct = int(ff.hypomimia_score * 100)

            st.markdown(
                f"<div style='background:white;border:1px solid #e2e8f0;"
                f"border-left:5px solid {color};border-radius:8px;"
                f"padding:28px 32px;margin-bottom:24px;'>"
                f"<p style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:0.1em;"
                f"text-transform:uppercase;margin:0 0 6px 0;'>Hypomimia Risk</p>"
                f"<p style='color:{color};font-size:36px;font-weight:700;margin:0;'>{risk.upper()}</p>"
                f"<p style='color:#94a3b8;font-size:13px;margin:4px 0 0 0;'>"
                f"Score {score_pct}/100 — 0 = no hypomimia, 100 = severe</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Facial Signal Measurements")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Smile Amplitude",   f"{ff.smile_amplitude:.3f}",
                        help="Normalized mouth width at peak smile (healthy: 0.18–0.26)")
            col2.metric("Smile Variability", f"{ff.smile_variability:.3f}",
                        help="Std dev of mouth width — low = mask-like (healthy: 0.02–0.06)")
            col3.metric("Cheek Elevation",   f"{ff.cheek_elevation:.3f}",
                        help="AU6 proxy: cheek rise during smile (healthy: 0.05–0.15)")
            col4.metric("Signal Confidence", f"{int(ff.confidence * 100)}%")

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### Clinical Interpretation")
            interp = fr.get("interpretation", ff.notes)
            rec    = fr.get("recommendation", "Consult a neurologist.")
            st.markdown(
                f"<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;'>"
                f"<p style='color:#374151;font-size:14px;line-height:1.75;margin:0 0 12px 0;'>{interp}</p>"
                f"<p style='color:#2563eb;font-size:13px;font-weight:500;margin:0;'>Next step: {rec}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### Clinical Report")
            st.markdown(
                "<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;margin-bottom:16px;'>"
                "<p style='color:#374151;font-size:13px;margin:0;'>"
                "Generate a structured PDF report of your facial expressivity assessment. "
                "This report can be shared with your physician."
                "</p></div>",
                unsafe_allow_html=True,
            )
            col_btn, col_empty = st.columns([1, 3])
            with col_btn:
                if st.button("Generate PDF Report", key="face_pdf_btn", use_container_width=True):
                    from report_generator_face import generate_face_report
                    with st.spinner("Generating facial expressivity report..."):
                        pdf_bytes = generate_face_report(ff, fr)
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name=f"sentinel_face_report_{ff.risk_level}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

        elif not run_face:
            st.markdown(
                "<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;"
                "padding:80px 40px;text-align:center;margin-top:40px;'>"
                "<p style='font-size:32px;margin:0 0 12px 0;'>😊</p>"
                "<p style='color:#1e3a5f;font-size:18px;font-weight:600;margin:0 0 8px 0;'>Facial Expressivity Screening</p>"
                "<p style='color:#94a3b8;font-size:14px;margin:0 0 16px 0;'>"
                "This mode detects hypomimia — reduced facial movement that can be an early "
                "Parkinson's indicator — using a short smile recording.</p>"
                "<p style='color:#94a3b8;font-size:13px;margin:0;'>"
                "Select a camera source and click Start Recording to begin.</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;border-top:1px solid #e2e8f0;padding:20px 0 8px 0;'>"
            "<p style='color:#64748b;font-size:11px;margin:0;'>"
            "For screening purposes only - Not a substitute for medical diagnosis</p></div>",
            unsafe_allow_html=True,
        )
        return  # don't fall through to hand mode rendering

    # ── Hand tremor flow (existing) ────────────────────────────────────────────
    if run:
        st.markdown("#### Live Capture")
        preview_container = st.container()
        with preview_container:
            preview_slot = st.empty()
            progress_slot = st.empty()
            sample_count_slot = st.empty()
        status_slot = st.empty()

        hand_data = run_capture(
            source=source_id,
            hand=hand_id,
            duration=float(duration),
            fps=30,
            preview_slot=preview_slot,
            progress_slot=progress_slot,
            status_slot=status_slot,
            sample_count_slot=sample_count_slot,
        )

        if hand_data is None:
            st.stop()

        meta = hand_data.get("metadata", {})
        right_n = meta.get("right_samples", 0)
        left_n = meta.get("left_samples", 0)
        if right_n + left_n == 0:
            status_slot.error(
                "No hand was detected during recording. Make sure your hand is visible "
                "to the camera and try again."
            )
            st.stop()

        progress_slot.empty()
        preview_slot.empty()
        status_slot.success(
            f"Captured {right_n} right-hand and {left_n} left-hand samples "
            f"at ~{hand_data.get('sample_rate', 0):.1f} Hz "
            f"(units: {meta.get('units', 'unknown')})."
        )

        processing_slot = st.empty()
        render_processing_screen(
            processing_slot,
            "Analyzing movement signal",
            "SENTINEL is extracting tremor amplitude, frequency, symmetry, and signal quality from the captured video.",
        )
        features = analyze_tremor(hand_data)

        render_processing_screen(
            processing_slot,
            "Nemotron is reviewing the assessment",
            "The clinical model is interpreting the movement profile and preparing the screening result.",
        )
        result   = classify_with_nemotron(features)
        severity = result.get("severity", "unknown")
        ftm      = result.get("ftm_score", "?")

        render_processing_screen(
            processing_slot,
            "Preparing patient summary",
            "Nemotron is turning the assessment into plain-language guidance for the results panel.",
        )
        explanation = get_explanation(features, severity, ftm)
        processing_slot.empty()

        st.session_state.last_features    = features
        st.session_state.last_severity    = severity
        st.session_state.last_ftm         = ftm
        st.session_state.last_explanation = explanation
        st.session_state.last_result      = result
        st.session_state.last_metadata    = meta

    # Results panel
    if st.session_state.last_features:
        features    = st.session_state.last_features
        severity    = st.session_state.last_severity
        ftm         = st.session_state.last_ftm
        explanation = st.session_state.last_explanation
        result      = st.session_state.last_result
        color       = SEVERITY_COLOR.get(severity, "#6b7280")
        bg          = SEVERITY_BG.get(severity, "#f9fafb")

        st.markdown(
            f"<div style='background:white;border:1px solid #e2e8f0;"
            f"border-left:5px solid {color};border-radius:8px;"
            f"padding:28px 32px;margin-bottom:24px;'>"
            f"<p style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:0.1em;"
            f"text-transform:uppercase;margin:0 0 6px 0;'>Assessment Result</p>"
            f"<p style='color:{color};font-size:36px;font-weight:700;margin:0;'>{severity.upper()}</p>"
            f"<p style='color:#94a3b8;font-size:13px;margin:4px 0 0 0;'>Fahn-Tolosa-Marin Grade {ftm} / 4</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Signal Measurements")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amplitude",  f"{features.amplitude_mm} mm")
        col2.metric("Frequency",  f"{features.dominant_frequency_hz} Hz")
        col3.metric("Symmetry",   f"{features.symmetry_score} / 1.0")
        col4.metric("Risk Level", features.risk_level.capitalize())

        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown("#### Clinical Interpretation")
        st.markdown(
            f"<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;'>"
            f"<p style='color:#374151;font-size:14px;line-height:1.75;margin:0;'>{explanation}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### Clinical Report")
        st.markdown(
            "<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;margin-bottom:16px;'>"
            "<p style='color:#374151;font-size:13px;margin:0;'>"
            "Generate a structured PDF report containing your full assessment, measurements, "
            "clinical findings, and recommendations. This report can be shared with your physician."
            "</p></div>",
            unsafe_allow_html=True,
        )

        col_btn, col_empty = st.columns([1, 3])
        with col_btn:
            if st.button("Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating clinical report..."):
                    pdf_bytes = generate_report(features, severity, ftm)
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"sentinel_report_{severity}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    elif not run:
        st.markdown(
            "<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:80px 40px;text-align:center;margin-top:40px;'>"
            "<p style='font-size:32px;margin:0 0 12px 0;'>🩺</p>"
            "<p style='color:#1e3a5f;font-size:18px;font-weight:600;margin:0 0 8px 0;'>Ready for Assessment</p>"
            "<p style='color:#94a3b8;font-size:14px;margin:0;'>"
            "Select a camera source from the sidebar and click Start Recording to begin.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;border-top:1px solid #e2e8f0;padding:20px 0 8px 0;'>"
        "<p style='color:#64748b;font-size:11px;margin:0;'>"
        "For screening purposes only - Not a substitute for medical diagnosis</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()