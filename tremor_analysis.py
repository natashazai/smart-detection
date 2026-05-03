"""
SENTINEL — Tremor Analysis Module
===================================
Person 2: Signal Processing & Feature Extraction
Person 3 (Natasha): LLM Clinical Interpretation

Input:  XYZ landmark time series (from Person 1 / OAK-D)
Output: Structured feature dict + Nemotron severity classification

Mock data is used until Person 1 has the camera ready.
"""

import numpy as np
from numpy.fft import fft, fftfreq
from dataclasses import dataclass, asdict
from typing import Any
import json
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TremorFeatures:
    """
    Structured output handed to Nemotron (Person 3).
    All fields are plain types — easy to serialize to JSON.
    """
    dominant_frequency_hz: float       # Parkinson's range: 4–6 Hz
    amplitude_mm: float                # Real-world mm (from depth data)
    symmetry_score: float              # 0.0 = fully asymmetric, 1.0 = symmetric
    tremor_type: str                   # "resting" | "postural" | "intentional" | "none"
    right_hand_frequency: float
    left_hand_frequency: float
    right_hand_amplitude: float
    left_hand_amplitude: float
    confidence: float                  # 0.0 – 1.0, how clean the signal was
    risk_level: str                    # "low" | "moderate" | "high" — preliminary only
    notes: str                         # human-readable flag for Nemotron


# ─────────────────────────────────────────────
# CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def extract_fingertip_movement(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Focuses on index fingertip (landmark 8) as primary tremor signal.
    Returns (N, 3) array — XYZ displacement from mean position.
    """
    if hand_landmarks.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    fingertip = hand_landmarks[:, 8, :]           # index fingertip
    displacement = fingertip - fingertip.mean(axis=0)
    return displacement


def compute_dominant_frequency(signal_xyz: np.ndarray, sample_rate: float) -> tuple[float, float]:
    """
    Runs FFT on each XYZ axis separately and averages the power spectra.

    Using np.linalg.norm(XYZ) before FFT doubles frequencies because
    ||sin(2πft)||² ∝ (1 − cos(4πft)), so 5 Hz would appear as 10 Hz.
    Per-axis FFT with averaged spectra avoids this.

    Returns (dominant_frequency_hz, confidence).

    Parkinson's tremor: 4–6 Hz resting
    Essential tremor:   6–12 Hz
    Physiological:      8–12 Hz (everyone has this at very low amplitude)
    """
    N = len(signal_xyz)
    if N < 4 or sample_rate <= 0 or signal_xyz.ndim < 2:
        return 0.0, 0.0

    freqs = fftfreq(N, d=1.0 / sample_rate)
    # Include sub-tremor range so we can detect and reject slow movement.
    positive_mask = (freqs > 0.3) & (freqs < min(20.0, sample_rate / 2.0))
    freqs_pos = freqs[positive_mask]
    if len(freqs_pos) == 0:
        return 0.0, 0.0

    # Average power spectra across XYZ axes
    combined_spectrum = np.zeros(int(positive_mask.sum()))
    valid_axes = 0
    for axis in range(signal_xyz.shape[1]):
        col = signal_xyz[:, axis]
        if not np.all(np.isfinite(col)) or np.allclose(col, col[0]):
            continue
        windowed = col * np.hanning(N)
        spectrum = np.abs(fft(windowed)) ** 2   # power spectrum per axis
        combined_spectrum += spectrum[positive_mask]
        valid_axes += 1

    if valid_axes == 0 or combined_spectrum.sum() <= 0:
        return 0.0, 0.0

    dominant_idx = int(np.argmax(combined_spectrum))
    dominant_freq = float(freqs_pos[dominant_idx])

    # Frequencies below 3 Hz are voluntary movement, not tremor.
    # Reject after finding true dominant so harmonics of slow movement
    # don't slip through (e.g. 2×2.81 = 5.62 Hz looks like Parkinson's).
    # if dominant_freq < 3.0:
    #     return 0.0, 0.0

    confidence = combined_spectrum[dominant_idx] / combined_spectrum.sum()
    return dominant_freq, float(np.clip(confidence * 5, 0, 1))



def compute_amplitude_mm(signal_xyz: np.ndarray) -> float:
    """
    Robust peak-to-peak amplitude of movement.
    Uses 5th–95th percentile range so a single noise spike or
    outlier frame doesn't inflate the reading.
    OAK-D depth gives real millimeter values; image_px mode is
    calibrated to mm before this function is called.
    """
    if signal_xyz.size == 0:
        return 0.0
    magnitude = np.linalg.norm(signal_xyz, axis=1)
    if not np.all(np.isfinite(magnitude)):
        return 0.0
    p5, p95 = np.percentile(magnitude, [5, 95])
    return float(p95 - p5)


def compute_symmetry_score(
    right_freq: float,
    left_freq: float,
    right_amp: float,
    left_amp: float
) -> float:
    """
    Compares both hands.
    Parkinson's characteristically starts asymmetric (one side first).
    Score: 1.0 = perfectly symmetric, 0.0 = completely one-sided.
    """
    freq_diff = abs(right_freq - left_freq) / (max(right_freq, left_freq) + 1e-6)
    amp_diff = abs(right_amp - left_amp) / (max(right_amp, left_amp) + 1e-6)
    asymmetry = (freq_diff + amp_diff) / 2
    return float(np.clip(1.0 - asymmetry, 0.0, 1.0))


def classify_tremor_type(frequency: float, amplitude: float) -> str:
    """
    Rough classification based on clinical literature.

    NOTE: This is a screening heuristic, NOT a diagnosis.

    Frequency floor: < 3 Hz is voluntary movement, not tremor.
    """
    if amplitude < 1.0 or frequency < 3.0:
        return "none"
    if 4.0 <= frequency <= 6.0:
        return "resting"       # Parkinson's profile
    if 6.0 < frequency <= 12.0:
        return "postural"      # Essential tremor profile
    return "intentional"


def assess_risk_level(
    frequency: float,
    amplitude: float,
    symmetry: float,
    tremor_type: str
) -> tuple[str, str]:
    """
    Combines signals into a preliminary risk level.
    Returns (risk_level, notes_for_nemotron).

    All scoring gates on frequency being in the clinical tremor range (≥3 Hz).
    Sub-threshold frequency means the signal is voluntary movement, so no
    risk points are awarded regardless of amplitude or asymmetry.
    """
    notes = []
    score = 0

    # Nothing to score if frequency is below clinical tremor floor.
    if frequency < 3.0 or tremor_type == "none":
        return "low", "No significant tremor indicators detected."

    # Frequency in Parkinson's range
    if 4.0 <= frequency <= 6.0:
        score += 2
        notes.append(f"Frequency {frequency:.1f}Hz is within Parkinson's resting tremor range (4-6Hz).")

    # Significant amplitude — only meaningful if frequency qualifies
    if amplitude > 3.0:
        score += 1
        notes.append(f"Amplitude {amplitude:.1f}mm exceeds typical physiological threshold.")

    # Asymmetric onset — only meaningful if frequency qualifies
    if symmetry < 0.5:
        score += 2
        notes.append(f"Asymmetry detected (score {symmetry:.2f}) - consistent with early unilateral onset.")

    if score >= 4:
        return "high", " ".join(notes)
    elif score >= 2:
        return "moderate", " ".join(notes)
    else:
        return "low", "No significant tremor indicators detected."


def _as_landmark_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 21, 3), dtype=np.float64)
    landmarks = np.asarray(value, dtype=np.float64)
    if landmarks.size == 0:
        return np.empty((0, 21, 3), dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError(
            f"Expected hand landmarks with shape (N, 21, 3), got {landmarks.shape}."
        )
    return landmarks


def _estimate_mm_per_unit(hand_landmarks: np.ndarray) -> float:
    """
    When no depth camera is available, landmark coords are MediaPipe-normalized
    (roughly 0-1 range).  We use the known anatomy of a human hand to recover
    a scale factor: wrist (idx 0) → index MCP (idx 5) is ~65 mm on an adult.

    Returns mm per coordinate unit, or 1.0 if the estimate is unreliable.
    """
    if hand_landmarks.size == 0 or hand_landmarks.shape[0] < 10:
        return 1.0
    # Use the median across all frames so one bad frame doesn't skew the scale.
    wrist = hand_landmarks[:, 0, :]      # (N, 3)
    index_mcp = hand_landmarks[:, 5, :]  # (N, 3)
    dists = np.linalg.norm(index_mcp - wrist, axis=1)
    median_dist = float(np.median(dists[dists > 1e-6]))
    if median_dist < 1e-6:
        return 1.0
    WRIST_TO_INDEX_MCP_MM = 65.0
    return WRIST_TO_INDEX_MCP_MM / median_dist


def classify_severity_local(
    frequency: float,
    amplitude_mm: float,
    tremor_type: str,
    symmetry: float,
    confidence: float,
) -> tuple[str, int]:
    """
    Deterministic FTM-based severity classification.  This replaces the LLM
    call for the actual grade — the LLM is kept only for plain-English
    *explanation*.

    Returns (severity_label, ftm_grade 0-4).

    Clinical reasoning:
    - Amplitude is the primary FTM driver.
    - But amplitude alone misleads if frequency is outside the tremor range.
      Physiological jitter at 11+ Hz with low amplitude is NOT a tremor.
    - We reduce effective amplitude when the signal looks like noise
      (high frequency, high symmetry, low confidence) to avoid false positives.
    """
    if confidence < 0.05 or amplitude_mm < 0.1 or frequency < 3.0:
        return "none", 0

    # Parkinson's profile: 4–6 Hz, asymmetric.
    # Essential tremor:    6–12 Hz, symmetric.
    # High freq (>12 Hz) + high symmetry → likely physiological jitter.
    in_parkinsons_range = 4.0 <= frequency <= 6.5
    in_essential_range  = 6.0 < frequency <= 15.0   # broader — includes action tremor
    is_high_freq_noise  = frequency > 15.0 and symmetry > 0.75 and confidence < 0.2

    if is_high_freq_noise:
        return "none", 0

    # Scale FTM thresholds: tighten for non-tremor-frequency signals
    # so a fast, symmetric jitter doesn't land in "moderate".
    if in_parkinsons_range or in_essential_range:
        thresholds = [(0.3, "none", 0), (2.0, "mild", 1),
                      (5.0, "moderate", 2), (12.0, "marked", 3)]
    else:
        # Intentional / unclassified — require larger amplitude to flag
        thresholds = [(1.0, "none", 0), (6.0, "mild", 1),
                      (12.0, "moderate", 2), (20.0, "marked", 3)]

    for threshold, label, grade in reversed(thresholds):
        if amplitude_mm >= threshold:
            return label, grade

    return "none", 0


    if value is None:
        return np.empty((0, 21, 3), dtype=np.float64)
    landmarks = np.asarray(value, dtype=np.float64)
    if landmarks.size == 0:
        return np.empty((0, 21, 3), dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError(
            f"Expected hand landmarks with shape (N, 21, 3), got {landmarks.shape}."
        )
    return landmarks


def _timestamps_for_hand(hand_data: dict, hand: str, count: int, fallback_rate: float) -> np.ndarray:
    hand_key = f"{hand}_timestamps"
    if hand_key in hand_data:
        timestamps = np.asarray(hand_data[hand_key], dtype=np.float64)
    elif "timestamps" in hand_data:
        timestamps = np.asarray(hand_data["timestamps"], dtype=np.float64)
    else:
        timestamps = np.arange(count, dtype=np.float64) / max(float(fallback_rate), 1e-6)

    if timestamps.size < count:
        generated = np.arange(count, dtype=np.float64) / max(float(fallback_rate), 1e-6)
        generated[:timestamps.size] = timestamps
        timestamps = generated
    return timestamps[:count]


def _estimate_sample_rate(timestamps: np.ndarray, fallback_rate: float) -> float:
    if timestamps.size < 2:
        return float(fallback_rate)
    diffs = np.diff(timestamps)
    valid = diffs[diffs > 1e-6]
    if valid.size == 0:
        return float(fallback_rate)
    return float(1.0 / np.median(valid))


def _analyze_hand_signal(
    hand_landmarks: np.ndarray,
    timestamps: np.ndarray,
    fallback_sample_rate: float,
) -> tuple[float, float, float]:
    if hand_landmarks.size == 0:
        return 0.0, 0.0, 0.0
    signal_xyz = extract_fingertip_movement(hand_landmarks)
    sample_rate = _estimate_sample_rate(timestamps, fallback_sample_rate)
    frequency, confidence = compute_dominant_frequency(signal_xyz, sample_rate)
    amplitude = compute_amplitude_mm(signal_xyz)
    return frequency, confidence, amplitude


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_tremor(hand_data: dict) -> TremorFeatures:
    """
    Full pipeline: raw landmark data -> structured TremorFeatures.

    hand_data format (from Person 1):
    {
        "right": np.array (N, 21, 3),
        "left":  np.array (N, 21, 3),
        "timestamps": np.array (N,),
        "sample_rate": int
    }
    """
    sample_rate = float(hand_data.get("sample_rate", 30))
    right_hand = _as_landmark_array(hand_data.get("right"))
    left_hand = _as_landmark_array(hand_data.get("left"))
    right_timestamps = _timestamps_for_hand(hand_data, "right", len(right_hand), sample_rate)
    left_timestamps = _timestamps_for_hand(hand_data, "left", len(left_hand), sample_rate)

    right_freq, right_conf, right_amp = _analyze_hand_signal(
        right_hand,
        right_timestamps,
        sample_rate,
    )
    left_freq, left_conf, left_amp = _analyze_hand_signal(
        left_hand,
        left_timestamps,
        sample_rate,
    )

    has_right = len(right_hand) > 0
    has_left = len(left_hand) > 0
    if not has_right and not has_left:
        return TremorFeatures(
            dominant_frequency_hz=0.0,
            amplitude_mm=0.0,
            symmetry_score=0.0,
            tremor_type="none",
            right_hand_frequency=0.0,
            left_hand_frequency=0.0,
            right_hand_amplitude=0.0,
            left_hand_amplitude=0.0,
            confidence=0.0,
            risk_level="low",
            notes="No hand landmarks were captured.",
        )

    metadata = hand_data.get("metadata", {})
    units = metadata.get("units", "mm") if isinstance(metadata, dict) else "mm"

    # ── Unit calibration ──────────────────────────────────────────────────────
    # When there's no depth camera the pipeline stores normalized MediaPipe
    # coords (0-1 range).  Convert to mm using hand anatomy as a ruler so
    # the FTM amplitude thresholds mean something real.
    right_scale = left_scale = 1.0
    if units not in ("mm", "mock"):
        if has_right:
            right_scale = _estimate_mm_per_unit(right_hand)
        if has_left:
            left_scale  = _estimate_mm_per_unit(left_hand)

    right_amp = right_amp * right_scale
    left_amp  = left_amp  * left_scale

    if has_right and (right_amp >= left_amp or not has_left):
        dominant_freq = right_freq
        dominant_amp  = right_amp
        confidence    = right_conf
    else:
        dominant_freq = left_freq
        dominant_amp  = left_amp
        confidence    = left_conf

    # Higher-level features
    symmetry     = compute_symmetry_score(right_freq, left_freq, right_amp, left_amp) if has_right and has_left else 0.0
    tremor_type  = classify_tremor_type(dominant_freq, dominant_amp)
    risk, notes  = assess_risk_level(dominant_freq, dominant_amp, symmetry, tremor_type)
    if not has_right or not has_left:
        captured = "right" if has_right else "left"
        notes = f"{notes} Only the {captured} hand was captured; symmetry could not be assessed."
    if units not in (None, "mm", "mock"):
        notes = (
            f"{notes} No depth camera — amplitudes estimated from hand-size calibration."
        )

    return TremorFeatures(
        dominant_frequency_hz   = round(dominant_freq, 2),
        amplitude_mm            = round(dominant_amp, 2),
        symmetry_score          = round(symmetry, 2),
        tremor_type             = tremor_type,
        right_hand_frequency    = round(right_freq, 2),
        left_hand_frequency     = round(left_freq, 2),
        right_hand_amplitude    = round(right_amp, 2),
        left_hand_amplitude     = round(left_amp, 2),
        confidence              = round(confidence, 2),
        risk_level              = risk,
        notes                   = notes
    )


# ─────────────────────────────────────────────
# NEMOTRON HANDOFF
# ─────────────────────────────────────────────

def features_to_nemotron_prompt(features: TremorFeatures) -> str:
    """
    Formats extracted features into a structured prompt for Person 3 / Nemotron.
    """
    return f"""
You are a neurological screening assistant. A patient has completed a 30-second hand tremor assessment.
The following features were extracted from their hand movement:

- Dominant Tremor Frequency: {features.dominant_frequency_hz} Hz
- Movement Amplitude: {features.amplitude_mm} mm
- Tremor Type: {features.tremor_type}
- Hand Symmetry Score: {features.symmetry_score} (1.0 = symmetric, 0.0 = one-sided)
- Right Hand: {features.right_hand_frequency} Hz, {features.right_hand_amplitude} mm
- Left Hand:  {features.left_hand_frequency} Hz, {features.left_hand_amplitude} mm
- Signal Confidence: {features.confidence}
- Preliminary Risk: {features.risk_level}
- Analysis Notes: {features.notes}

Clinical reference:
- Parkinson's resting tremor: 4-6 Hz, asymmetric onset, amplitude > 2mm
- Essential tremor: 6-12 Hz, typically symmetric
- Physiological tremor: < 1mm amplitude, not clinically significant

Based on this data, provide:
1. A plain-English interpretation of the tremor pattern (2-3 sentences)
2. A clear risk assessment (low / moderate / high)
3. A specific recommendation for next steps

IMPORTANT: You are a screening tool only. Always recommend consulting a neurologist for confirmation.
Do not diagnose. Do not alarm unnecessarily.
""".strip()


import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

client = (
    OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    if OpenAI is not None
    else None
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"


# ─────────────────────────────────────────────
# Nemotron handoff — explanation only
#
# Severity classification is now done locally by classify_severity_local()
# so a Nemotron API failure can never produce a wrong severity grade.
# Nemotron's job is writing the plain-English patient explanation.
# ─────────────────────────────────────────────
def classify_with_nemotron(features: "TremorFeatures | float") -> dict:
    """
    Accepts either a full TremorFeatures dataclass or a bare amplitude_mm
    float (kept for backwards compatibility).

    Returns dict with keys:
        severity, ftm_score, risk_level, interpretation, recommendation,
        latency_s, confidence, asymmetry_flag, clinical_note
    """
    # ── 1. Local deterministic classification (never fails) ─────────────────
    if isinstance(features, (int, float)):
        # Legacy call with bare amplitude — reconstruct minimal features
        amplitude_mm = float(features)
        frequency    = 0.0
        tremor_type  = "unknown"
        symmetry     = 1.0
        confidence   = 0.5
        risk_level   = "unknown"
    else:
        amplitude_mm = features.amplitude_mm
        frequency    = features.dominant_frequency_hz
        tremor_type  = features.tremor_type
        symmetry     = features.symmetry_score
        confidence   = features.confidence
        risk_level   = features.risk_level

    severity, ftm_score = classify_severity_local(
        frequency, amplitude_mm, tremor_type, symmetry, confidence
    )

    base_result = {
        "severity":      severity,
        "ftm_score":     ftm_score,
        "risk_level":    risk_level,
        "asymmetry_flag": symmetry < 0.6,
        "confidence":    int(confidence * 100),
    }

    # ── 2. Nemotron for plain-English explanation (optional, non-blocking) ───
    t0 = time.time()
    try:
        if client is None:
            raise RuntimeError("OpenAI SDK not installed")

        # Compact prompt — Nemotron is a chain-of-thought model so we tell it
        # explicitly to skip reasoning and output JSON immediately.
        prompt = (
            f"Tremor data: frequency={frequency:.1f}Hz, amplitude={amplitude_mm:.1f}mm, "
            f"type={tremor_type}, symmetry={symmetry:.2f}, severity={severity} (FTM {ftm_score}), "
            f"risk={risk_level}.\n\n"
            "Clinical refs: Parkinson's = 4-6Hz resting, asymmetric. "
            "Essential = 6-12Hz, symmetric. Physiological < 1mm.\n\n"
            "Output ONLY valid JSON, no thinking, no preamble:\n"
            '{"interpretation":"2-3 plain-English sentences for patient",'
            '"recommendation":"one specific next step",'
            '"clinical_note":"one sentence clinical summary"}'
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content":
                    "You are a neurological screening assistant. "
                    "Output ONLY valid compact JSON. No thinking. No markdown. "
                    "Start your response with { immediately."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.0,
        )

        latency = round(time.time() - t0, 2)
        raw = (response.choices[0].message.content or "").strip()

        # Nemotron reasoning models often prefix with chain-of-thought.
        # Find the LAST complete JSON object in the output.
        start = raw.rfind('{')
        end   = raw.rfind('}')
        if start != -1 and end > start:
            explanation = json.loads(raw[start:end + 1])
            base_result.update({
                "interpretation":  explanation.get("interpretation", ""),
                "recommendation":  explanation.get("recommendation", "Consult a neurologist."),
                "clinical_note":   explanation.get("clinical_note", ""),
                "latency_s":       latency,
            })
        else:
            raise ValueError(f"No JSON found in: {raw[:200]}")

    except Exception as e:
        latency = round(time.time() - t0, 2)
        # API failed — severity is already set correctly, just fill explanation defaults.
        base_result.update({
            "interpretation":  (
                f"Your tremor reading shows {severity} activity at {frequency:.1f} Hz "
                f"with {amplitude_mm:.1f} mm amplitude."
            ),
            "recommendation":  "Please consult a neurologist for formal evaluation.",
            "clinical_note":   f"API explanation unavailable: {e}",
            "latency_s":       latency,
        })

    return base_result
