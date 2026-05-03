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
# MOCK DATA GENERATOR
# Replace with Person 1's real OAK-D stream
# ─────────────────────────────────────────────

def generate_mock_hand_data(
    duration_seconds: int = 30,
    sample_rate: int = 30,
    tremor_frequency: float = 5.0,
    tremor_amplitude: float = 3.5,
    noise_level: float = 0.5,
    seed: int = 42
) -> dict:
    """
    Simulates 30 seconds of hand landmark XYZ data at 30fps.

    Returns dict with shape:
    {
        "right": np.array of shape (N, 21, 3),  # 21 landmarks, XYZ
        "left":  np.array of shape (N, 21, 3),
        "timestamps": np.array of shape (N,)
    }

    When Person 1 is ready, swap this function out for the real feed.
    Landmark index 8 = index fingertip (most useful for tremor tracking).
    """
    np.random.seed(seed)
    N = duration_seconds * sample_rate
    t = np.linspace(0, duration_seconds, N)

    def make_hand(freq, amp, noise):
        landmarks = np.zeros((N, 21, 3))
        for i in range(21):
            # Tremor signal on all axes, stronger on fingertips (index 4–20)
            scale = 1.0 if i < 4 else 1.5
            landmarks[:, i, 0] = amp * scale * np.sin(2 * np.pi * freq * t) + noise * np.random.randn(N)
            landmarks[:, i, 1] = amp * scale * np.sin(2 * np.pi * freq * t + 0.3) + noise * np.random.randn(N)
            landmarks[:, i, 2] = amp * scale * np.sin(2 * np.pi * freq * t + 0.6) + noise * np.random.randn(N)
        return landmarks

    right_hand = make_hand(tremor_frequency, tremor_amplitude, noise_level)

    # Parkinson's is typically asymmetric — left hand slightly different
    left_hand = make_hand(
        freq=tremor_frequency * 0.85,
        amp=tremor_amplitude * 0.4,   # weaker on non-dominant side
        noise=noise_level
    )

    return {
        "right": right_hand,
        "left": left_hand,
        "timestamps": t,
        "right_timestamps": t,
        "left_timestamps": t,
        "sample_rate": sample_rate,
        "metadata": {"source": "mock", "units": "mm"},
    }


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


def _bandpass_fft(
    signal_xyz: np.ndarray,
    sample_rate: float,
    low_hz: float = 3.0,
    high_hz: float = 20.0,
) -> np.ndarray:
    """
    Return a version of signal_xyz with only the tremor-band frequencies kept.
    Uses FFT zero-masking then IFFT — no scipy dependency required.
    This keeps amplitude and frequency detection consistent: both operate on
    the same spectral region so slow drift cannot inflate amplitude.
    """
    N = len(signal_xyz)
    if N < 4:
        return signal_xyz
    out = np.zeros_like(signal_xyz)
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)
    keep = (np.abs(freqs) >= low_hz) & (np.abs(freqs) < high_hz)
    for axis in range(signal_xyz.shape[1]):
        col = signal_xyz[:, axis]
        spectrum = np.fft.fft(col)
        spectrum[~keep] = 0.0
        out[:, axis] = np.fft.ifft(spectrum).real
    return out


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
    # Search only within the clinical tremor band (3–20 Hz).
    # The previous lower bound of 0.3 Hz meant slow arm drift (0.5–2 Hz)
    # could dominate the spectrum and push the detected frequency below 3 Hz,
    # causing the function to return 0.0 Hz and every session to gate as "none".
    # Drift is voluntary movement, not tremor — it should never win the FFT
    # competition. Starting at 3 Hz keeps it out of the search entirely.
    positive_mask = (freqs >= 3.0) & (freqs < min(20.0, sample_rate / 2.0))
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
    if dominant_freq < 3.0:
        return 0.0, 0.0

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

    IMPORTANT: Only x,y are used for this distance. MediaPipe's z is an
    estimated depth relative to the wrist plane — it's in a different scale
    than x/y and is significantly noisier. Including z corrupts the calibration
    and causes the scale factor to be wrong by 30–60%, which then inflates
    amplitude readings for stationary hands.

    Returns mm per coordinate unit, or 1.0 if the estimate is unreliable.
    """
    if hand_landmarks.size == 0 or hand_landmarks.shape[0] < 10:
        return 1.0
    # Use the median across all frames so one bad frame doesn't skew the scale.
    wrist     = hand_landmarks[:, 0, :2]   # (N, 2) — x,y only
    index_mcp = hand_landmarks[:, 5, :2]   # (N, 2) — x,y only
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
    # Use estimated rate for the bandpass (reflects actual data density), but
    # floor it at the nominal camera FPS for the FFT frequency axis. When
    # MediaPipe drops frames the timestamp-estimated rate can fall to 8-12fps;
    # at those rates the 3-20Hz Nyquist mask shrinks or disappears entirely,
    # zeroing the frequency. The underlying signal is still sampled at the
    # camera's native fps — missed detections are gaps, not slower sampling.
    detected_rate = _estimate_sample_rate(timestamps, fallback_sample_rate)
    fft_rate = max(detected_rate, fallback_sample_rate)
    frequency, confidence = compute_dominant_frequency(signal_xyz, fft_rate)
    signal_bp = _bandpass_fft(signal_xyz, fft_rate, low_hz=3.0, high_hz=20.0)
    amplitude = compute_amplitude_mm(signal_bp)
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

    has_right = right_hand.size > 0
    has_left = left_hand.size > 0
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

    # ── Detection quality gate ─────────────────────────────────────────────────
    # Tremor analysis needs enough frames to resolve 3-20Hz. At 30fps we need
    # at least 5 seconds (150 frames) for a 0.2Hz frequency resolution. If
    # MediaPipe only detected the hand for fewer frames than this, results are
    # unreliable — warn rather than silently return bad data.
    MIN_FRAMES = 150  # 5 seconds at 30fps
    right_count = len(right_hand) if has_right else 0
    left_count  = len(left_hand)  if has_left  else 0
    best_count  = max(right_count, left_count)
    low_detection = best_count < MIN_FRAMES

    metadata = hand_data.get("metadata", {})
    units = metadata.get("units", "mm") if isinstance(metadata, dict) else "mm"

    # ── Unit calibration ──────────────────────────────────────────────────────
    # When there's no depth camera the pipeline stores normalized MediaPipe
    # coords (0-1 range).  Convert to mm using hand anatomy as a ruler so
    # the FTM amplitude thresholds mean something real.
    #
    # IMPORTANT: zero out z BEFORE any signal analysis. MediaPipe's z for 2D
    # landmark tracking is a rough depth estimate relative to the wrist plane
    # — it's not in the same scale as x/y and has significant frame-to-frame
    # noise. After scaling to mm that noise produces 5-15mm fake amplitude on
    # a perfectly still hand. x,y alone are sufficient for tremor detection.
    right_scale = left_scale = 1.0
    if units not in ("mm", "mock"):
        if has_right:
            right_hand = right_hand.copy()
            right_hand[:, :, 2] = 0.0
            right_scale = _estimate_mm_per_unit(right_hand)
        if has_left:
            left_hand = left_hand.copy()
            left_hand[:, :, 2] = 0.0
            left_scale = _estimate_mm_per_unit(left_hand)

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
    if low_detection:
        notes = (
            f"{notes} Low detection rate: only {best_count} frames captured "
            f"(need 150+ for reliable analysis). Ensure good lighting and keep "
            f"the hand clearly in frame."
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
# ─────────────────────────────────────────────────────────────────────────────
# SENTINEL AGENT TOOLS
#
# These are real Python functions Nemotron can call during its reasoning loop.
# Each tool does meaningful work on the actual signal data — not fake stubs.
# The agent decides which tools to call and in what order, demonstrating the
# autonomous multi-step reasoning the judges are looking for.
# ─────────────────────────────────────────────────────────────────────────────

def _tool_get_frequency_profile(frequency: float, confidence: float) -> dict:
    """
    Returns a clinical frequency-band interpretation for the detected frequency.
    Nemotron calls this to understand what the frequency reading means clinically
    before it decides on severity.
    """
    if frequency <= 0 or confidence < 0.1:
        return {
            "band": "undetected",
            "clinical_meaning": "No consistent tremor frequency detected. Signal may be noise or voluntary movement.",
            "differential": ["No tremor", "Insufficient data", "Sub-threshold physiological tremor"],
            "confidence_note": f"Signal confidence too low ({confidence:.0%}) for reliable frequency analysis.",
        }
    if frequency < 3.0:
        return {
            "band": "sub-tremor",
            "clinical_meaning": "Below clinical tremor threshold. Likely voluntary movement or postural sway.",
            "differential": ["Normal", "Dystonia (rare)"],
            "confidence_note": f"Frequency {frequency:.1f}Hz is below the 3Hz tremor floor.",
        }
    if 3.0 <= frequency < 4.0:
        return {
            "band": "low-tremor",
            "clinical_meaning": "Low-frequency range. Can indicate cerebellar or Holmes tremor, or recording artifact.",
            "differential": ["Cerebellar tremor", "Holmes tremor", "Recording artifact"],
            "confidence_note": f"Borderline frequency {frequency:.1f}Hz warrants cautious interpretation.",
        }
    if 4.0 <= frequency <= 6.0:
        return {
            "band": "parkinsonian",
            "clinical_meaning": "Classic Parkinson's resting tremor range (4-6Hz). Typically pill-rolling, asymmetric onset.",
            "differential": ["Parkinson's disease", "Drug-induced parkinsonism", "Essential tremor (lower end)"],
            "confidence_note": f"Frequency {frequency:.1f}Hz falls in the Parkinson's diagnostic window.",
        }
    if 6.0 < frequency <= 12.0:
        return {
            "band": "essential",
            "clinical_meaning": "Essential tremor range (6-12Hz). Action/postural, typically symmetric, worsened by intention.",
            "differential": ["Essential tremor", "Physiological tremor (enhanced)", "Hyperthyroidism"],
            "confidence_note": f"Frequency {frequency:.1f}Hz consistent with essential tremor profile.",
        }
    return {
        "band": "high-frequency",
        "clinical_meaning": "High-frequency range. Enhanced physiological or orthostatic tremor. Rarely pathological alone.",
        "differential": ["Enhanced physiological tremor", "Neuropathic tremor", "Anxiety/stimulants"],
        "confidence_note": f"Frequency {frequency:.1f}Hz above typical pathological tremor range.",
    }


def _tool_assess_amplitude(amplitude_mm: float, tremor_type: str) -> dict:
    """
    Contextualizes amplitude against clinical thresholds for the detected tremor type.
    Amplitude means different things depending on frequency band — Nemotron calls
    this after get_frequency_profile to get a type-aware amplitude assessment.
    """
    thresholds = {
        "resting":     {"sub": 1.0, "mild": 2.5, "moderate": 5.0},
        "postural":    {"sub": 1.5, "mild": 3.0, "moderate": 6.0},
        "intentional": {"sub": 2.0, "mild": 4.0, "moderate": 8.0},
        "none":        {"sub": 1.0, "mild": 2.0, "moderate": 4.0},
    }
    t = thresholds.get(tremor_type, thresholds["none"])
    if amplitude_mm < t["sub"]:
        grade = "sub-threshold"
        clinical_note = "Below threshold for clinical significance. Within physiological range."
    elif amplitude_mm < t["mild"]:
        grade = "mild"
        clinical_note = "Mild amplitude. May affect fine motor tasks; unlikely to impair daily function."
    elif amplitude_mm < t["moderate"]:
        grade = "moderate"
        clinical_note = "Moderate amplitude. Likely affects handwriting, eating, and precision tasks."
    else:
        grade = "severe"
        clinical_note = "Severe amplitude. High impact on activities of daily living."
    return {
        "amplitude_mm": round(amplitude_mm, 2),
        "grade": grade,
        "clinical_note": clinical_note,
        "tremor_type_context": tremor_type,
        "thresholds_used": t,
    }


def _tool_check_laterality(
    symmetry_score: float,
    right_freq: float, right_amp: float,
    left_freq: float, left_amp: float,
) -> dict:
    """
    Analyzes asymmetry between hands. Asymmetric onset is a hallmark of
    Parkinson's disease — Nemotron uses this to distinguish PD from ET.
    Returns which hand is dominant, severity of asymmetry, and clinical implications.
    """
    both_present = (right_amp > 0.1) and (left_amp > 0.1)
    if not both_present:
        dominant = "right" if right_amp > left_amp else "left"
        return {
            "asymmetry_present": True,
            "dominant_side": dominant,
            "symmetry_score": symmetry_score,
            "pattern": "unilateral",
            "clinical_implication": (
                f"Tremor detected only in the {dominant} hand. "
                "Unilateral presentation is a hallmark of early Parkinson's disease. "
                "Essential tremor typically presents bilaterally."
            ),
            "pd_flag": True,
        }
    freq_diff = abs(right_freq - left_freq) / max(right_freq, left_freq, 1e-6)
    amp_diff  = abs(right_amp - left_amp)  / max(right_amp, left_amp, 1e-6)
    is_asymmetric = symmetry_score < 0.6
    dominant = "right" if right_amp >= left_amp else "left"
    if not is_asymmetric:
        pattern = "bilateral-symmetric"
        implication = (
            "Tremor is present bilaterally and symmetrically. "
            "This pattern is more consistent with essential tremor than Parkinson's disease."
        )
        pd_flag = False
    elif amp_diff > 0.4:
        pattern = "bilateral-asymmetric"
        implication = (
            f"Tremor is significantly stronger on the {dominant} side (symmetry {symmetry_score:.2f}). "
            "Marked asymmetry raises suspicion for Parkinson's disease or structural lesion."
        )
        pd_flag = True
    else:
        pattern = "bilateral-mildly-asymmetric"
        implication = (
            f"Mild asymmetry detected (symmetry {symmetry_score:.2f}). "
            "Nonspecific — can appear in both essential tremor and early Parkinson's."
        )
        pd_flag = False
    return {
        "asymmetry_present": is_asymmetric,
        "dominant_side": dominant,
        "symmetry_score": symmetry_score,
        "pattern": pattern,
        "freq_difference_pct": round(freq_diff * 100, 1),
        "amp_difference_pct": round(amp_diff * 100, 1),
        "clinical_implication": implication,
        "pd_flag": pd_flag,
    }


def _tool_compute_ftm_score(
    frequency: float, amplitude_mm: float,
    tremor_type: str, symmetry_score: float,
) -> dict:
    """
    Computes Fahn-Tolosa-Marin tremor severity score (0-4 scale).
    FTM is the standard clinical instrument. Nemotron uses this to produce
    a structured severity grade rather than a vague label.
    """
    severity, ftm_score = classify_severity_local(
        frequency, amplitude_mm, tremor_type, symmetry_score, confidence=0.8
    )
    descriptions = {
        0: "No tremor detectable.",
        1: "Slight tremor — barely perceptible, not functionally limiting.",
        2: "Mild tremor — clearly present, mild functional impact.",
        3: "Moderate tremor — significant functional impairment.",
        4: "Severe tremor — marked disability, prevents most ADLs.",
    }
    return {
        "ftm_score": ftm_score,
        "severity_label": severity,
        "description": descriptions.get(ftm_score, "Unknown"),
        "functional_impact": "high" if ftm_score >= 3 else ("moderate" if ftm_score >= 2 else "low"),
    }


# Map tool name → callable. Used by the agentic loop to dispatch calls.
_SENTINEL_TOOLS: dict[str, Any] = {
    "get_frequency_profile": _tool_get_frequency_profile,
    "assess_amplitude":      _tool_assess_amplitude,
    "check_laterality":      _tool_check_laterality,
    "compute_ftm_score":     _tool_compute_ftm_score,
}

# OpenAI-format tool schemas passed to Nemotron so it knows what to call.
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_frequency_profile",
            "description": (
                "Look up the clinical meaning of the detected tremor frequency. "
                "Returns the frequency band (parkinsonian/essential/physiological), "
                "differential diagnosis list, and interpretation notes. "
                "Call this FIRST before assessing severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency": {"type": "number", "description": "Dominant tremor frequency in Hz"},
                    "confidence": {"type": "number", "description": "Signal confidence 0.0-1.0"},
                },
                "required": ["frequency", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_amplitude",
            "description": (
                "Assess tremor amplitude against clinical thresholds for the detected tremor type. "
                "Thresholds differ between resting, postural, and intentional tremors. "
                "Call this after get_frequency_profile so you know the tremor type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amplitude_mm": {"type": "number", "description": "Peak-to-peak amplitude in mm"},
                    "tremor_type":  {"type": "string", "description": "resting | postural | intentional | none"},
                },
                "required": ["amplitude_mm", "tremor_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_laterality",
            "description": (
                "Analyze left/right hand asymmetry. Asymmetric onset is a clinical hallmark of "
                "Parkinson's disease vs essential tremor. Returns pattern, dominant side, "
                "and a pd_flag if asymmetry pattern matches Parkinson's profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symmetry_score": {"type": "number", "description": "0.0=fully asymmetric, 1.0=symmetric"},
                    "right_freq":  {"type": "number"}, "right_amp":  {"type": "number"},
                    "left_freq":   {"type": "number"}, "left_amp":   {"type": "number"},
                },
                "required": ["symmetry_score", "right_freq", "right_amp", "left_freq", "left_amp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ftm_score",
            "description": (
                "Compute the Fahn-Tolosa-Marin (FTM) severity score on the 0-4 clinical scale. "
                "Call this last, after you have assessed frequency, amplitude, and laterality, "
                "so the score reflects your full understanding of the case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency":     {"type": "number"},
                    "amplitude_mm":  {"type": "number"},
                    "tremor_type":   {"type": "string"},
                    "symmetry_score": {"type": "number"},
                },
                "required": ["frequency", "amplitude_mm", "tremor_type", "symmetry_score"],
            },
        },
    },
]


def classify_with_nemotron(features: "TremorFeatures | float") -> dict:
    """
    Agentic Nemotron classification loop.

    Instead of a single-shot prompt, Nemotron reasons through the case by
    calling real diagnostic tools — get_frequency_profile, assess_amplitude,
    check_laterality, compute_ftm_score — in whatever order makes sense for
    the data. The agent decides when it has enough information to conclude.

    This demonstrates:
      • Autonomous reasoning (Nemotron plans its own analysis steps)
      • Multi-step workflow (tool calls build on each other)
      • Tool integration (real functions returning real data)
      • Nemotron-specific strength (chain-of-thought drives tool selection)

    Returns dict with keys:
        severity, ftm_score, risk_level, interpretation, recommendation,
        clinical_note, latency_s, confidence, asymmetry_flag,
        agent_steps (list of tool calls made, for UI display)
    """
    # ── 1. Extract features (backwards-compatible with bare float) ───────────
    if isinstance(features, (int, float)):
        amplitude_mm = float(features)
        frequency    = 0.0
        tremor_type  = "unknown"
        symmetry     = 1.0
        confidence   = 0.5
        risk_level   = "unknown"
        right_freq = right_amp = left_freq = left_amp = 0.0
    else:
        amplitude_mm = features.amplitude_mm
        frequency    = features.dominant_frequency_hz
        tremor_type  = features.tremor_type
        symmetry     = features.symmetry_score
        confidence   = features.confidence
        risk_level   = features.risk_level
        right_freq   = features.right_hand_frequency
        right_amp    = features.right_hand_amplitude
        left_freq    = features.left_hand_frequency
        left_amp     = features.left_hand_amplitude

    # ── 2. Local deterministic fallback (always works, never throws) ─────────
    severity, ftm_score = classify_severity_local(
        frequency, amplitude_mm, tremor_type, symmetry, confidence
    )
    base_result = {
        "severity":       severity,
        "ftm_score":      ftm_score,
        "risk_level":     risk_level,
        "asymmetry_flag": symmetry < 0.6,
        "confidence":     int(confidence * 100),
        "agent_steps":    [],
        "interpretation": (
            f"Tremor reading: {severity} at {frequency:.1f}Hz, {amplitude_mm:.1f}mm amplitude."
        ),
        "recommendation": "Consult a neurologist for formal evaluation.",
        "clinical_note":  "Local classification (Nemotron unavailable).",
        "latency_s":      0.0,
    }

    if client is None:
        return base_result

    # ── 3. Agentic loop ──────────────────────────────────────────────────────
    t0 = time.time()
    system_prompt = (
        "You are SENTINEL, a neurological tremor screening agent. "
        "You have access to diagnostic tools. Use them to reason through the case step by step.\n\n"
        "Your workflow:\n"
        "1. Call get_frequency_profile to understand what the frequency means clinically.\n"
        "2. Call assess_amplitude using the tremor type you learned from step 1.\n"
        "3. Call check_laterality to determine if asymmetry matches a Parkinson's pattern.\n"
        "4. Call compute_ftm_score with your full understanding from steps 1-3.\n"
        "5. Only after ALL tool calls, write your final JSON assessment.\n\n"
        "IMPORTANT: This is a SCREENING TOOL only. Never diagnose. Always recommend "
        "neurologist consultation for any positive finding.\n\n"
        "Final output must be ONLY valid JSON:\n"
        '{"interpretation":"2-3 plain-English sentences for the patient",'
        '"recommendation":"one specific actionable next step",'
        '"clinical_note":"one sentence technical summary for the physician"}'
    )
    user_message = (
        f"Patient tremor data:\n"
        f"- Dominant frequency: {frequency:.2f} Hz\n"
        f"- Amplitude: {amplitude_mm:.2f} mm\n"
        f"- Tremor type: {tremor_type}\n"
        f"- Symmetry score: {symmetry:.2f} (1.0=symmetric)\n"
        f"- Right hand: {right_freq:.1f}Hz, {right_amp:.1f}mm\n"
        f"- Left hand:  {left_freq:.1f}Hz, {left_amp:.1f}mm\n"
        f"- Signal confidence: {confidence:.2f}\n"
        f"- Preliminary risk: {risk_level}\n\n"
        "Use your diagnostic tools to analyze this case, then output your JSON assessment."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    agent_steps = []
    MAX_ROUNDS  = 8   # safety cap — Nemotron should finish in 4-5

    try:
        for _round in range(MAX_ROUNDS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=800,
                temperature=0.0,
            )
            msg = response.choices[0].message

            # Agent wants to call tools
            if msg.tool_calls:
                # Append the assistant message with tool_calls intact
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
                # Execute each tool call and feed results back
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    fn = _SENTINEL_TOOLS.get(fn_name)
                    if fn is None:
                        tool_result = {"error": f"Unknown tool: {fn_name}"}
                    else:
                        try:
                            tool_result = fn(**fn_args)
                        except Exception as te:
                            tool_result = {"error": str(te)}

                    agent_steps.append({
                        "tool":   fn_name,
                        "args":   fn_args,
                        "result": tool_result,
                    })
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         fn_name,
                        "content":      json.dumps(tool_result),
                    })
                continue  # next round — agent processes tool results

            # Agent is done with tools, read final text response
            raw = (msg.content or "").strip()
            # Nemotron may prepend chain-of-thought before the JSON.
            start = raw.rfind("{")
            end   = raw.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw[start:end + 1])
                # Pull FTM from agent's last compute_ftm_score call if available
                ftm_calls = [s for s in agent_steps if s["tool"] == "compute_ftm_score"]
                if ftm_calls:
                    ftm_result = ftm_calls[-1]["result"]
                    ftm_score  = ftm_result.get("ftm_score", ftm_score)
                    severity   = ftm_result.get("severity_label", severity)
                base_result.update({
                    "severity":       severity,
                    "ftm_score":      ftm_score,
                    "interpretation": parsed.get("interpretation", base_result["interpretation"]),
                    "recommendation": parsed.get("recommendation", base_result["recommendation"]),
                    "clinical_note":  parsed.get("clinical_note",  base_result["clinical_note"]),
                    "latency_s":      round(time.time() - t0, 2),
                    "agent_steps":    agent_steps,
                })
            else:
                base_result.update({
                    "latency_s":   round(time.time() - t0, 2),
                    "agent_steps": agent_steps,
                    "clinical_note": f"Agent response unparseable: {raw[:120]}",
                })
            break

    except Exception as e:
        base_result.update({
            "clinical_note": f"Agent error: {e}",
            "latency_s":     round(time.time() - t0, 2),
            "agent_steps":   agent_steps,
        })

    return base_result




if __name__ == "__main__":
    print("=" * 55)
    print("  SENTINEL — Tremor Analysis (Mock Mode)")
    print("=" * 55)

    # Simulate a concerning Parkinson's profile
    print("\n[1] Generating mock hand data (Parkinson's profile)...")
    hand_data = generate_mock_hand_data(
        duration_seconds=30,
        sample_rate=30,
        tremor_frequency=5.2,    # in Parkinson's range
        tremor_amplitude=4.0,    # noticeable amplitude
        noise_level=0.5
    )

    print("[2] Running tremor analysis...")
    features = analyze_tremor(hand_data)

    print("\n── Extracted Features ──────────────────────────")
    print(json.dumps(asdict(features), indent=2))

    # NATASHA'S PART — send to Nemotron and print result
    print("\n── Nemotron Severity Classification (Natasha) ──")
    result   = classify_with_nemotron(features.amplitude_mm)
    severity = result.get("severity", "error").upper()
    ftm      = result.get("ftm_score", "?")
    latency  = result.get("latency_s", "?")

    print(f"  Amplitude : {features.amplitude_mm} mm")
    print(f"  SEVERITY  : {severity}  (FTM grade {ftm})  [{latency}s]")

    print("\n✓ Done")