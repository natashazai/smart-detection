# Early Parkison's Screening 

Our project is a Python-based tremor screening prototype that uses **computer vision hand tracking** (MediaPipe) to capture 21-point hand landmarks from a live camera feed, extracts **tremor frequency/amplitude/symmetry** features, and generates a **clinical-style summary and PDF report**.

> Important: This project is a **screening / research prototype**. It is **not** a medical device and must not be used to diagnose or treat disease.

## Features

- **Live capture** from a webcam (and optional OAK-D / OAK-D Lite support)
- **Hand landmark tracking** (21 landmarks) via MediaPipe
- Tremor signal processing:
  - Dominant frequency (FFT-based)
  - Tremor amplitude (robust percentile peak-to-peak)
  - Laterality & symmetry scoring
  - Tremor type heuristic (resting / postural / intentional / none)
- **Nemotron-powered clinical text** (NVIDIA hosted model) for:
  - Plain-English patient explanation (UI)
  - Structured clinical report text (PDF)
- **Streamlit UI** to run an end-to-end screening session and download/open a report

## Repository layout

- `UI.py` — Streamlit dashboard (main app)
- `pipeline.py` — camera capture + MediaPipe landmark pipeline (webcam + optional DepthAI/OAK)
- `tremor_analysis.py` — signal processing + feature extraction + optional Nemotron classification/explanation
- `report_generator.py` — generates a polished PDF report (ReportLab) and calls Nemotron for structured report text
- `hand_landmarker.task` — MediaPipe Tasks model file (used when `mp.solutions` is unavailable)
- `hand_xyz.csv` — sample/recorded landmark export (CSV)

## Requirements

- Python 3.10+ recommended
- A working webcam (default)

Python packages (typical):

- `streamlit`
- `opencv-python`
- `mediapipe`
- `numpy`
- `python-dotenv`
- `openai` (used here as a client for NVIDIA's OpenAI-compatible endpoint)
- `reportlab`

Optional (for OAK-D / OAK-D Lite capture):

- `depthai`

## Setup

1. **Clone the repo**

```bash
git clone https://github.com/natashazai/smart-detection.git
cd smart-detection
```

2. **Create a virtual environment**

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. **Install dependencies**

If you don't have a `requirements.txt`, install the key packages manually:

```bash
pip install streamlit opencv-python mediapipe numpy python-dotenv openai reportlab
```

(Optional for OAK):

```bash
pip install depthai
```

4. **Set environment variables**

Create a `.env` file in the project root:

```bash
NVIDIA_API_KEY=your_key_here
```

The Streamlit UI uses `NVIDIA_API_KEY` to call the Nemotron model endpoint.

## Run the app (Streamlit UI)

```bash
streamlit run UI.py
```

Then:

1. Choose camera source and hand selection in the sidebar.
2. Click **Start Recording**.
3. After capture, SENTINEL computes tremor features and shows:
   - severity label and FTM grade
   - amplitude/frequency/symmetry metrics
   - a plain-English explanation
4. A PDF report is generated and can be opened/downloaded.

## Run capture + analysis from the CLI

You can capture landmarks and optionally print analysis features directly:

```bash
python pipeline.py --source webcam --duration 30 --hand both --output hand_xyz.csv --analyze
```

Sources:

- `--source webcam` (default laptop camera)
- `--source oak` (OAK-D Lite RGB + depth)
- `--source oak-rgb` (OAK RGB-only)

If your MediaPipe install requires Tasks mode, pass the model path:

```bash
python pipeline.py --source webcam --model hand_landmarker.task
```

## Notes on measurement units

- With **OAK depth**, the pipeline projects landmark pixels into approximate **millimeters**.
- With **webcam-only**, landmarks are in normalized image coordinates; `tremor_analysis.py` estimates a mm scale using a wrist-to-index-MCP anatomical heuristic.

## Safety / disclaimer

This repository is for educational and screening research purposes only:

- Not a medical diagnosis
- Not validated clinically
- Results can be affected by lighting, camera FPS, occlusion, and tracking quality
