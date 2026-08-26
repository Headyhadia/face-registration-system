# Student Face Registration & Liveness Detection System

A Python prototype that uses a webcam to register a student's face, verifying
the face belongs to a live person (not a printed photo or a phone/screen
image) before accepting the registration.

## Features

- Live webcam feed with real-time face detection
- Multi-frame liveness/anti-spoofing confirmation before capturing (a single
  frame is never enough to trigger a save — several consecutive "real" reads
  are required, which makes the system meaningfully harder to fool than a
  one-shot check)
- Identity locking — once a face is being tracked, the system keeps tracking
  that same face by position, so a bystander leaning into frame can't hijack
  an in-progress registration
- Grace period on momentary face loss (a brief glance away or half-second
  step out of frame doesn't instantly cancel the session — a short countdown
  is shown before the attempt is actually cancelled)
- Time-limited sessions — a spoof attempt or unresolved liveness check
  auto-cancels after a fixed time window instead of running indefinitely
- Nothing is written to disk unless a registration fully succeeds — no
  partial folders or partial image sets from cancelled/failed attempts
- Saved images are cropped face regions (with a small margin), not full
  scene frames — keeps other people out of the dataset and matches what
  downstream recognition pipelines typically expect as input
- Background-threaded model loading with a proper loading screen, so the
  camera window never freezes on startup

## Requirements

- Python 3.10–3.12
- A webcam
- ~2GB free disk space (TensorFlow + PyTorch + model weights)
- Internet connection for first-time setup (dependency + model weight downloads)

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd face-registration-system
```

### 2. Create and activate a virtual environment

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the anti-spoofing model weights (manual step)

DeepFace normally auto-downloads these on first run, but the automatic
download from GitHub can occasionally fail (network timeout, rate limiting).
If you hit a `ValueError` mentioning a failed download when first running
the app, download the weight files manually instead:

```bash
mkdir -p ~/.deepface/weights
cd ~/.deepface/weights
wget https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
wget https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth
```

(Use `curl -LO <url>` in place of `wget` if it isn't available on your system.)

### 5. Run

```bash
python src/main.py
```

Enter a Student ID when prompted, then look at the camera. Registration
starts automatically once liveness is confirmed, and captured images are
saved to `dataset/Student_<ID>/`. Press `q` at any time to cancel.

## Design notes / known trade-offs

- **Detector backend: OpenCV, not MediaPipe.** MediaPipe was the original
  choice for a better accuracy/speed trade-off, but on Python 3.12 the
  installed MediaPipe version conflicts with the protobuf version
  TensorFlow (a DeepFace dependency) requires, causing an unresolvable
  import error. Given the project timeline, the OpenCV (Haar Cascade)
  detector backend was used instead — it's less accurate on difficult
  angles/lighting than MediaPipe would be, but stable and
  dependency-conflict-free.
- **Anti-spoofing classification can occasionally flicker.** Face
  *detection* on printed photos and phone/screen images is reliable — the
  detector consistently finds the face. What can occasionally flicker
  instead is the anti-spoofing classifier itself, briefly reading a spoof
  source as "real" for a frame or two before correcting. This is
  infrequent and is fully absorbed by the existing safeguards: the
  multi-frame consecutive-live requirement means a couple of stray "real"
  reads are not enough to trigger a capture, and the session time limit
  ensures a spoof attempt is eventually cancelled rather than left
  hanging. No additional handling was needed beyond what's already in
  place.
- **CPU-only.** No GPU is required or used; the anti-spoofing model
  (Fasnet, PyTorch-based) and DeepFace's TensorFlow-based components both
  run fine on CPU for this use case, at a small latency cost.

## Manual test results

The following scenarios were manually tested against this implementation:

| # | Test | Result |
|---|------|--------|
| 1.1 | Camera opens, correct size, no UI freeze | Pass |
| 1.2 | Face detection on real face | Pass |
| 1.3 | No face in frame | Pass |
| 1.4 | Angled/partial/poor lighting face | Pass |
| 2.1 | Valid Student ID | Pass |
| 2.2 | Empty Student ID | Pass |
| 2.3 | Student ID with invalid characters (e.g. spaces) | Pass |
| 2.4 | Duplicate Student ID (overwrite prompt) | Pass |
| 3.1 | Real person registers successfully | Pass |
| 3.2 | Printed photo held to camera (including a printed photo containing multiple people) | Pass |
| 3.3 | Phone/screen photo held to camera | Pass |
| 3.4 | Quit during a spoof attempt | Pass |
| 3.5 | Quit during a genuine (non-spoof) attempt | Pass |
| 4.1 | Bystander leans in closer during registration | Pass |
| 4.2 | Registered person steps away mid-registration | Pass |
| 5.1 | Saved images have no UI overlay burned in | Pass |
| 5.2 | Saved images are reasonably framed face crops | Pass |
| 5.3 | Folder structure matches spec (`dataset/Student_<ID>/image_N.jpg`) | Pass |
| 5.4 | Image count within required range | Pass |
| 6.1 | Webcam unavailable | Pass |
| 6.2 | Multiple faces present with no prior lock | Pass |
| 7.1 | Exact "Face registration successful." message | Pass |
| 7.2 | Exact spoof warning message | Pass |

## Project structure

```
face-registration-system/
├── dataset/                  # created at runtime, one folder per registered student
│   └── Student_<ID>/
│       ├── image_1.jpg
│       ├── image_2.jpg
│       └── ...
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```