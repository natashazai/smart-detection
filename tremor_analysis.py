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
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, asdict
from typing import Optional
import json
import time
from openai import OpenAI


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
        "sample_rate": sample_rate
    }


# ─────────────────────────────────────────────
# CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def extract_fingertip_movement(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Focuses on index fingertip (landmark 8) as primary tremor signal.
    Returns (N, 3) array — XYZ displacement from mean position.
    """
    fingertip = hand_landmarks[:, 8, :]           # index fingertip
    displacement = fingertip - fingertip.mean(axis=0)
    return displacement


def compute_dominant_frequency(signal_xyz: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """
    Runs FFT on combined XYZ signal.
    Returns (dominant_frequency_hz, confidence).

    Parkinson's tremor: 4–6 Hz resting
    Essential tremor:   6–12 Hz
    Physiological:      8–12 Hz (everyone has this at very low amplitude)
    """
    N = len(signal_xyz)

    # Combine XYZ into single magnitude signal
    magnitude = np.linalg.norm(signal_xyz, axis=1)

    # Apply Hanning window to reduce spectral leakage
    windowed = magnitude * np.hanning(N)

    # FFT
    freqs = fftfreq(N, d=1.0 / sample_rate)
    spectrum = np.abs(fft(windowed))

    # Only look at positive frequencies, cap at 20 Hz (above that = noise)
    positive_mask = (freqs > 0.5) & (freqs < 20.0)
    freqs_pos = freqs[positive_mask]
    spectrum_pos = spectrum[positive_mask]

    if len(spectrum_pos) == 0:
        return 0.0, 0.0

    dominant_idx = np.argmax(spectrum_pos)
    dominant_freq = freqs_pos[dominant_idx]

    # Confidence = how much energy is concentrated at the dominant frequency
    confidence = spectrum_pos[dominant_idx] / spectrum_pos.sum()

    return float(dominant_freq), float(np.clip(confidence * 5, 0, 1))


def compute_amplitude_mm(signal_xyz: np.ndarray) -> float:
    """
    Peak-to-peak amplitude of movement.
    OAK-D depth gives us real millimeter values.
    With mock data, units are simulated mm.
    """
    magnitude = np.linalg.norm(signal_xyz, axis=1)
    peak_to_peak = magnitude.max() - magnitude.min()
    return float(peak_to_peak)


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
    """
    if amplitude < 1.0:
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

    Nemotron will do the real reasoning — this is just a structured hint.
    """
    notes = []
    score = 0

    # Frequency in Parkinson's range
    if 4.0 <= frequency <= 6.0:
        score += 2
        notes.append(f"Frequency {frequency:.1f}Hz is within Parkinson's resting tremor range (4-6Hz).")

    # Significant amplitude
    if amplitude > 3.0:
        score += 1
        notes.append(f"Amplitude {amplitude:.1f}mm exceeds typical physiological threshold.")

    # Asymmetric onset
    if symmetry < 0.6:
        score += 2
        notes.append(f"Asymmetry detected (score {symmetry:.2f}) - consistent with early unilateral onset.")

    if score >= 4:
        return "high", " ".join(notes)
    elif score >= 2:
        return "moderate", " ".join(notes)
    else:
        return "low", "No significant tremor indicators detected."


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
    sample_rate = hand_data["sample_rate"]

    # Extract fingertip displacement for each hand
    right_signal = extract_fingertip_movement(hand_data["right"])
    left_signal  = extract_fingertip_movement(hand_data["left"])

    # Frequency analysis
    right_freq, right_conf = compute_dominant_frequency(right_signal, sample_rate)
    left_freq,  left_conf  = compute_dominant_frequency(left_signal,  sample_rate)

    # Amplitude
    right_amp = compute_amplitude_mm(right_signal)
    left_amp  = compute_amplitude_mm(left_signal)

    # Use the more prominent hand as the primary signal
    if right_amp >= left_amp:
        dominant_freq = right_freq
        dominant_amp  = right_amp
        confidence    = right_conf
    else:
        dominant_freq = left_freq
        dominant_amp  = left_amp
        confidence    = left_conf

    # Higher-level features
    symmetry     = compute_symmetry_score(right_freq, left_freq, right_amp, left_amp)
    tremor_type  = classify_tremor_type(dominant_freq, dominant_amp)
    risk, notes  = assess_risk_level(dominant_freq, dominant_amp, symmetry, tremor_type)

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


from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "nvidia/nemotron-3-super-120b-a12b"


# ─────────────────────────────────────────────
# Nemotron severity classifier
#
# Sends amplitude to Nemotron 120B and gets back
# FTM severity classification in caps for the UI.
# ─────────────────────────────────────────────
def classify_with_nemotron(amplitude_mm: float) -> dict:
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a clinical AI. Classify tremor severity by amplitude using the FTM scale.

FTM Severity Scale:
  none     = < 0.1 mm
  mild     = 0.1-5 mm    (FTM grade 1)
  moderate = 5-10 mm     (FTM grade 2)
  marked   = 10-20 mm    (FTM grade 3)
  severe   = > 20 mm     (FTM grade 4)

Respond with ONLY this JSON, no thinking, no explanation:
{"severity": "none|mild|moderate|marked|severe", "ftm_score": 0}"""
                },
                {
                    "role": "user",
                    "content": f"Amplitude: {amplitude_mm} mm. Output JSON only, start with {{"
                },
            ],
            max_tokens=2000,
            temperature=0.0,
        )

        latency = round(time.time() - t0, 2)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response")

        raw   = content.strip()
        start = raw.find('{')
        end   = raw.rfind('}')
        if start == -1 or end == -1:
            raise ValueError(f"No JSON found: {raw}")

        result = json.loads(raw[start:end+1])
        result["latency_s"] = latency
        return result

    except Exception as e:
        return {"severity": "error", "ftm_score": -1,
                "latency_s": round(time.time()-t0, 2), "error": str(e)}



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