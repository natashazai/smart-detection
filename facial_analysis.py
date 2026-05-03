"""
SENTINEL — Facial Expressivity / Hypomimia Module
===================================================
Standalone screening mode based on:
  "AI-Enabled Parkinson's Disease Screening Using Smile Videos"
  Adnan et al., NEJM AI 2025  (doi:10.1056/AIoa2400950)

Input:  (N_frames, 478, 3) MediaPipe FaceMesh landmark array
Output: FacialFeatures dataclass + Nemotron plain-English summary

Key signals:
  AU12 proxy — lip-corner pull  → smile_amplitude + smile_variability
  AU6  proxy — cheek raise      → cheek_elevation
  AU4  proxy — brow lowering    → brow_depression
  Overall kinetic energy        → facial_movement

Hypomimia (mask-face) = reduced values on all four signals.
Higher hypomimia_score = more mask-like = higher Parkinson's risk flag.

Mirrors tremor_analysis.py structure.  Does NOT require hand tracking.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ── MediaPipe FaceMesh 478-landmark indices ────────────────────────────────────
# Using refine_landmarks=True layout (468 face + 10 iris).
# Non-iris landmarks are identical between classic FaceMesh and Tasks API.
MOUTH_LEFT    = 61    # lip corner, left
MOUTH_RIGHT   = 291   # lip corner, right
UPPER_LIP     = 13    # upper lip midpoint
LOWER_LIP     = 14    # lower lip midpoint
LEFT_CHEEK    = 117   # cheek apex, left
RIGHT_CHEEK   = 346   # cheek apex, right
LEFT_BROW_IN  = 107   # inner brow, left
RIGHT_BROW_IN = 336   # inner brow, right
NOSE_TIP      = 1     # nose tip — stable anchor for normalization
FACE_LEFT     = 234   # left temple — face-width reference
FACE_RIGHT    = 454   # right temple — face-width reference


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FacialFeatures:
    """
    Structured output for Nemotron and the UI.
    All fields are plain types — JSON-serializable.

    Reference values (normalized to face width, healthy adult smiling):
        smile_amplitude:   0.18 – 0.26
        smile_variability: 0.02 – 0.06
        cheek_elevation:   0.05 – 0.15
        facial_movement:   0.002 – 0.008
    Lower values on all four signals indicate hypomimia.
    """
    smile_amplitude:   float   # peak normalized mouth width (AU12 proxy)
    smile_variability: float   # std of mouth width — low = mask-like
    cheek_elevation:   float   # AU6 proxy: cheek rise relative to nose
    brow_depression:   float   # AU4 proxy: inner-brow distance from nose
    facial_movement:   float   # mean frame-to-frame displacement (kinetic energy)
    hypomimia_score:   float   # 0 = none, 1 = severe
    risk_level:        str     # "low" | "moderate" | "high" | "unknown"
    confidence:        float   # 0 – 1, signal quality
    notes:             str     # human-readable flags for Nemotron


# ── Signal extraction ──────────────────────────────────────────────────────────

def _face_width(landmarks: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean distance between left and right temple (xy only)."""
    return np.linalg.norm(
        landmarks[:, FACE_RIGHT, :2] - landmarks[:, FACE_LEFT, :2], axis=1
    ).clip(min=1e-6)


def compute_smile_amplitude(landmarks: np.ndarray) -> tuple[float, float]:
    """
    AU12 (lip-corner pull) proxy.
    Mouth width normalized to face width — removes head distance from camera.

    Returns:
        peak_amplitude  — 90th-percentile (captures the widest smile frame)
        variability     — std across all frames (low = rigid/mask-like face)

    landmarks: (N, 478, 3)
    """
    fw = _face_width(landmarks)
    mouth_w = np.linalg.norm(
        landmarks[:, MOUTH_RIGHT, :2] - landmarks[:, MOUTH_LEFT, :2], axis=1
    )
    normalized = mouth_w / fw
    return float(np.percentile(normalized, 90)), float(np.std(normalized))


def compute_cheek_elevation(landmarks: np.ndarray) -> float:
    """
    AU6 (cheek raiser) proxy.
    Measures the range of vertical cheek motion across the recording —
    how much the cheeks travel upward (y decreasing in image coords) from
    their lowest position during the smile task.

    Using range (max - min of y) rather than an absolute position avoids
    the sign ambiguity of comparing cheek y to nose y, and is robust to
    variation in head tilt and camera distance.

    Returns a normalized float ≥ 0; higher = more cheek involvement.
    """
    fw = _face_width(landmarks)
    left_y  = landmarks[:, LEFT_CHEEK,  1] / fw
    right_y = landmarks[:, RIGHT_CHEEK, 1] / fw
    # Range: how much each cheek travels vertically during the recording
    left_range  = float(np.max(left_y)  - np.min(left_y))
    right_range = float(np.max(right_y) - np.min(right_y))
    return (left_range + right_range) / 2.0


def compute_brow_depression(landmarks: np.ndarray) -> float:
    """
    AU4 (brow lowerer) proxy.
    Inner-brow y distance from nose tip, normalized.
    Larger = brows higher up = no depression.
    PD patients show reduced brow mobility; this is a secondary signal.
    """
    fw = _face_width(landmarks)
    left_dist  = (landmarks[:, LEFT_BROW_IN,  1] - landmarks[:, NOSE_TIP, 1]) / fw
    right_dist = (landmarks[:, RIGHT_BROW_IN, 1] - landmarks[:, NOSE_TIP, 1]) / fw
    return float(np.mean((left_dist + right_dist) / 2.0))


def compute_facial_movement(landmarks: np.ndarray) -> float:
    """
    Overall kinetic energy: mean frame-to-frame xy displacement across all landmarks.
    Low value = face barely moves = hypomimia signature.
    """
    if landmarks.shape[0] < 2:
        return 0.0
    deltas = np.diff(landmarks[:, :, :2], axis=0)          # (N-1, 478, 2)
    per_frame_mean = np.sqrt((deltas**2).sum(axis=2)).mean(axis=1)  # (N-1,)
    return float(np.mean(per_frame_mean))


# ── Hypomimia score ────────────────────────────────────────────────────────────

def compute_hypomimia_score(
    smile_amplitude:   float,
    smile_variability: float,
    cheek_elevation:   float,
    facial_movement:   float,
) -> float:
    """
    Heuristic score 0 – 1.  Higher = more hypomimic = higher PD risk flag.

    Weights reflect feature importance from the NEJM AI paper:
    AU12 (amplitude + variability) most discriminative, then AU6, then overall motion.

    Reference thresholds (normalized to face width, MediaPipe real-world output):
        smile_amplitude:   ~0.40  — lip corners ≈ 40% of face width apart at peak
        smile_variability: ~0.04  — std of that distance across frames
        cheek_elevation:   ~0.03  — range of vertical cheek travel
        facial_movement:   ~0.005 — mean per-frame displacement across all landmarks

    Below these values the component contributes increasing hypomimia signal.
    Thresholds were calibrated from MediaPipe FaceMesh output on healthy adults;
    they will be refined as real patient data is collected.
    """
    amp_score   = float(np.clip(1.0 - smile_amplitude   / 0.48,  0.0, 1.0))
    var_score   = float(np.clip(1.0 - smile_variability / 0.04,  0.0, 1.0))
    cheek_score = float(np.clip(1.0 - cheek_elevation   / 0.04,  0.0, 1.0))
    move_score  = float(np.clip(1.0 - facial_movement   / 0.004, 0.0, 1.0))

    score = (
        0.35 * amp_score
        + 0.25 * var_score
        + 0.25 * cheek_score
        + 0.15 * move_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _classify_risk(hypomimia_score: float, confidence: float) -> str:
    if confidence < 0.25:
        return "unknown"
    if hypomimia_score < 0.35:
        return "low"
    if hypomimia_score < 0.60:
        return "moderate"
    return "high"


# ── Main analysis entry point ──────────────────────────────────────────────────

def analyze_facial_expression(
    face_landmarks: np.ndarray,
    sample_rate: float = 30.0,
) -> FacialFeatures:
    """
    Main entry point — mirrors analyze_tremor() in tremor_analysis.py.

    Args:
        face_landmarks: (N_frames, 478, 3) float array of normalized MediaPipe
                        FaceMesh landmarks. Coordinate range [0, 1].
        sample_rate:    Frames per second of the capture.

    Returns:
        FacialFeatures dataclass with all computed metrics.
    """
    if face_landmarks.ndim != 3 or face_landmarks.shape[1] < 291 or face_landmarks.shape[0] < 10:
        return FacialFeatures(
            smile_amplitude=0.0, smile_variability=0.0,
            cheek_elevation=0.0, brow_depression=0.0,
            facial_movement=0.0, hypomimia_score=1.0,
            risk_level="unknown", confidence=0.0,
            notes="Insufficient frames or landmark count for analysis.",
        )

    amp, variability = compute_smile_amplitude(face_landmarks)
    cheek            = compute_cheek_elevation(face_landmarks)
    brow             = compute_brow_depression(face_landmarks)
    movement         = compute_facial_movement(face_landmarks)

    # Signal quality: penalizes short recordings and suspiciously motionless signals.
    frame_quality   = min(face_landmarks.shape[0] / (sample_rate * 8.0), 1.0)
    motion_plausible = min(movement / 0.0005, 1.0)
    confidence = float(np.clip(0.3 + 0.5 * frame_quality + 0.2 * motion_plausible, 0.0, 1.0))

    hypo      = compute_hypomimia_score(amp, variability, cheek, movement)
    risk      = _classify_risk(hypo, confidence)

    notes: list[str] = []
    if amp < 0.12:
        notes.append("Very limited smile aperture detected.")
    if variability < 0.015:
        notes.append("Minimal facial movement variability — possible mask-like presentation.")
    if cheek < 0.04:
        notes.append("Reduced cheek elevation during smile task.")
    if movement < 0.001:
        notes.append("Near-zero overall facial kinetic energy.")
    if not notes:
        notes.append("Facial expressivity within expected range for this recording.")

    return FacialFeatures(
        smile_amplitude   = round(amp,        4),
        smile_variability = round(variability, 4),
        cheek_elevation   = round(cheek,       4),
        brow_depression   = round(brow,        4),
        facial_movement   = round(movement,    6),
        hypomimia_score   = round(hypo,        3),
        risk_level        = risk,
        confidence        = round(confidence,  3),
        notes             = " ".join(notes),
    )


# ── Mock data for testing ──────────────────────────────────────────────────────

def generate_mock_face_data(
    duration_seconds: float = 10.0,
    sample_rate:      float = 30.0,
    hypomimia_level:  float = 0.0,   # 0 = healthy smile, 1 = severe mask-face
    noise:            float = 0.002,
) -> np.ndarray:
    """
    Generates (N, 478, 3) synthetic face landmark data for a smile task.

    hypomimia_level=0  →  wide, dynamic smile with cheek involvement.
    hypomimia_level=1  →  near-static mask-face, minimal lip movement.
    """
    N   = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(7)

    # Base neutral face (rough normalized positions)
    base = rng.uniform(0.15, 0.85, (478, 3)).astype(np.float32)
    base[:, 2] *= 0.05   # z is very small in normalized coords

    # Anchor key landmarks to plausible positions
    base[FACE_LEFT]    = [0.15, 0.50, 0.00]
    base[FACE_RIGHT]   = [0.85, 0.50, 0.00]
    base[NOSE_TIP]     = [0.50, 0.55, 0.05]
    base[MOUTH_LEFT]   = [0.38, 0.75, 0.02]
    base[MOUTH_RIGHT]  = [0.62, 0.75, 0.02]
    base[LEFT_CHEEK]   = [0.28, 0.62, 0.02]
    base[RIGHT_CHEEK]  = [0.72, 0.62, 0.02]
    base[LEFT_BROW_IN] = [0.43, 0.36, 0.01]
    base[RIGHT_BROW_IN]= [0.57, 0.36, 0.01]

    frames = np.tile(base[np.newaxis], (N, 1, 1))

    # Smile envelope: ramp up, hold, ramp down
    t = np.linspace(0, np.pi, N)
    envelope = np.clip(np.sin(t), 0.0, 1.0)

    # Healthy smile magnitude; hypomimia shrinks it toward zero
    mag = 0.06 * (1.0 - hypomimia_level)

    for i, env in enumerate(envelope):
        # AU12 — lip corners pull outward and slightly upward
        frames[i, MOUTH_LEFT,  0] -= mag * env * 0.9
        frames[i, MOUTH_RIGHT, 0] += mag * env * 0.9
        frames[i, MOUTH_LEFT,  1] -= mag * env * 0.4
        frames[i, MOUTH_RIGHT, 1] -= mag * env * 0.4
        # AU6 — cheeks rise
        frames[i, LEFT_CHEEK,  1] -= mag * env * 0.35
        frames[i, RIGHT_CHEEK, 1] -= mag * env * 0.35

    frames += rng.normal(0, noise, frames.shape).astype(np.float32)
    return frames


# ── Nemotron integration ───────────────────────────────────────────────────────

def classify_hypomimia_with_nemotron(
    features:  FacialFeatures,
    api_key:   str | None = None,
    base_url:  str        = "https://integrate.api.nvidia.com/v1",
    model:     str        = "nvidia/nemotron-3-super-120b-a12b",
) -> dict:
    """
    Local classification is primary (hypomimia_score + risk_level are already set).
    Nemotron adds three short plain-English fields only — same architecture as
    classify_with_nemotron() in tremor_analysis.py.

    Returns dict with all FacialFeatures fields plus:
        interpretation, recommendation, clinical_note, latency_s
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    base: dict = {
        "hypomimia_score": features.hypomimia_score,
        "risk_level":      features.risk_level,
        "confidence":      int(features.confidence * 100),
        "smile_amplitude": features.smile_amplitude,
        "cheek_elevation": features.cheek_elevation,
    }

    t0 = time.time()
    try:
        if OpenAI is None:
            raise RuntimeError("openai package not installed")

        resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not resolved_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

        client = OpenAI(base_url=base_url, api_key=resolved_key)

        prompt = (
            f"Facial expressivity screening data:\n"
            f"  hypomimia_score={features.hypomimia_score:.2f} (0=none, 1=severe)\n"
            f"  smile_amplitude={features.smile_amplitude:.3f} (healthy: 0.18-0.26)\n"
            f"  smile_variability={features.smile_variability:.3f} (healthy: 0.02-0.06)\n"
            f"  cheek_elevation={features.cheek_elevation:.3f} (healthy: 0.05-0.15)\n"
            f"  facial_movement={features.facial_movement:.5f} (healthy: 0.002-0.008)\n"
            f"  risk_level={features.risk_level}\n"
            f"  Notes: {features.notes}\n\n"
            "Clinical context: Hypomimia is reduced facial expressivity, a motor symptom "
            "of Parkinson's disease. Low smile amplitude and variability indicate a "
            "mask-like face. This is a screening tool only — never diagnose.\n\n"
            "Output ONLY valid JSON, no thinking, no preamble:\n"
            '{"interpretation":"2-3 plain-English sentences explaining the facial screening result to the patient",'
            '"recommendation":"one specific next step for the patient",'
            '"clinical_note":"one sentence clinical summary for a neurologist"}'
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a neurological screening assistant. "
                        "Output ONLY valid compact JSON. No thinking. No markdown. "
                        "Start your response with { immediately."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.0,
        )

        latency = round(time.time() - t0, 2)
        raw = (response.choices[0].message.content or "").strip()

        start = raw.rfind("{")
        end   = raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])
            base.update(
                {
                    "interpretation": parsed.get("interpretation", ""),
                    "recommendation": parsed.get("recommendation", "Consult a neurologist."),
                    "clinical_note":  parsed.get("clinical_note", ""),
                    "latency_s":      latency,
                }
            )
        else:
            raise ValueError(f"No JSON in response: {raw[:120]}")

    except Exception as exc:
        latency = round(time.time() - t0, 2)
        base.update(
            {
                "interpretation": (
                    f"Your facial expressivity score is {features.hypomimia_score:.2f} out of 1.0, "
                    f"suggesting {features.risk_level} risk of hypomimia. "
                    f"{features.notes}"
                ),
                "recommendation": "Please consult a neurologist for formal evaluation.",
                "clinical_note":  f"API explanation unavailable: {exc}",
                "latency_s":      latency,
            }
        )

    return base


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    from dataclasses import asdict as _asdict

    print("=" * 60)
    print("  SENTINEL — Facial Expressivity Analysis (Mock Mode)")
    print("=" * 60)

    scenarios = [
        ("Healthy baseline",         0.0),
        ("Mild hypomimia",           0.4),
        ("Moderate hypomimia (PD)",  0.75),
        ("Severe hypomimia (PD)",    1.0),
    ]

    for label, level in scenarios:
        print(f"\n── {label} (hypomimia_level={level}) ────────────────")
        face_data = generate_mock_face_data(
            duration_seconds=10.0,
            sample_rate=30.0,
            hypomimia_level=level,
        )
        features = analyze_facial_expression(face_data)
        print(_json.dumps(_asdict(features), indent=2))