# Traffic & Vehicle Monitoring System

A traffic and vehicle monitoring system built with Python, OpenCV, YOLOv8, and CARLA — with a retro cyberpunk pygame UI. It detects vehicles, pedestrians, and accidents across three monitoring modes.

## Features

**Option 1 — Live Camera Feed**
Stream video from a second laptop's camera (`cam.exe`) over the network into the main app for real-time monitoring.

**Option 2 — Video Analysis**
Upload a traffic video for vehicle/pedestrian detection, accident detection, and severity estimation, with playback controls (pause, seek/scrub timeline) and on-screen analytics. Accident detection combines:
- A motion tracker that flags an abrupt, sustained stop on collision-style overlap (the primary, most reliable signal)
- A trained scene classifier that can independently confirm an accident across sustained frames (catches footage where the crash already happened before the clip started)
- Recording support to save the analysis session

**Option 3 — CARLA Simulation**
A 2×2 CCTV camera grid inside a live CARLA simulation with ~120 background vehicles on autopilot. An "Accident" toggle spawns a deliberate two-vehicle collision; detection runs via a layered combination of a motion tracker, a dedicated trained accident detector, and CARLA's own physics collision sensors as ground truth. Detection is gated behind the toggle — accidents are only ever reported while it's switched on.

All three options support in-app screen recording, and the Options screen prompts to download CARLA if it isn't installed/configured yet.

## Requirements

- Python 3.12 (a virtual environment is recommended — see `venv/`)
- For Option 3: [CARLA Simulator](https://carla.org/) installed separately (~10+ GB; not bundled)
- For Option 1: a second machine with a webcam, running `cam.exe`/`camera_streamer.py`, on the same network

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> **Note on `carla`:** the pinned version must match your installed CARLA server version. If `pip install carla==0.9.16` fails for your platform, install the wheel shipped with your CARLA download instead (see `requirements.txt` for the path).

## Running

```bash
venv\Scripts\python.exe main.py
```

On first launch, click the gear icon (top-right) to configure:
- **CARLA installation path** (needed for Option 3 — a "DOWNLOAD CARLA" button appears on the Options screen if this isn't set)
- **Camera IP / port** (needed for Option 1, matching whatever `cam.exe` reports on the second laptop)

### Pre-built executables (no Python required)

Standalone builds live in `assets/export/`:
- `Car/car.exe` — the full application (Options 1–3). Includes a starter `config.json` and `README.txt`.
- `Cam/cam.exe` — the camera streamer for the second laptop. Fully self-contained, zero extra files needed.

## Training the models

**Option 2 — accident/severity classifiers**
```bash
venv\Scripts\python.exe scripts/option_2/train.py
```
Trains on `assets/training/traffic_dataset/training/` (Accident / NonAccident / SeverityScore folders), balancing classes automatically. Outputs to `assets/training/traffic_dataset/model/`.

**Option 3 — CARLA dedicated accident detector**
```bash
venv\Scripts\python.exe scripts/option_3/train2.py
```
Requires a running (or launchable) CARLA server. Spawns randomized crash scenes across all 4 camera junctions, projects exact 3D→2D bounding boxes for ground-truth labels, and trains a YOLOv8 detector. Outputs to `assets/training/simulation_dataset/simulation_accident_detector.pt`.

`scripts/option_3/train.py` trains the general vehicle detector (`simulation_detector.pt`) the same way.

## Building the executables

```bash
venv\Scripts\python.exe build_car.py                    # -> assets/export/Car/car.exe
venv\Scripts\python.exe scripts/option_1/build_cam.py    # -> assets/export/Cam/cam.exe
```

Both are PyInstaller `--onefile` builds with all dependencies, icons, and trained models bundled in. `build_car.py` automatically skips any model file that hasn't been trained yet (the app falls back to detection-only heuristics).

## Project structure

```
main.py                  Main application (pygame UI, all 3 options)
config.py                Settings load/save (config.json)
resource_path.py         Asset path resolution (dev mode + frozen .exe mode)
video_analyzer.py        YOLO detection + accident/severity classification (Option 2)
ui_widgets.py            Shared UI components (buttons, cyber-styled frames)
preview.py               GIF preview player for the Options screen

scripts/
  option_1/               Camera streamer (cam.exe) + its build script
  option_2/               Training (train.py) + standalone video_monitor.py
  option_3/               CARLA simulation client + training scripts (train.py, train2.py)

assets/
  icon/                   App icons (.ico/.png), main-menu hover-car art
  gif/                    Options-screen preview GIFs
  training/               Datasets + trained model weights
  export/                 Built executables (Car/, Cam/)

build_car.py              Builds car.exe
requirements.txt
```

## Notes

- `.gitignore` excludes `venv/`, `__pycache__/`, build artifacts, and `*.exe` — but **not** the training datasets or trained model weights (`assets/training/**/*.pt`, `*.pt` in the repo root). Those are large; consider excluding them yourself if pushing to a remote with size limits, and re-run the relevant `train.py` to regenerate them instead.
- The accident classifier in Option 2 requires both positive (accident) and negative (clean) examples drawn from a similar visual source to avoid overfitting on incidental video style rather than genuine crash content — keep this balance in mind if you add more training data.
