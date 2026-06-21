"""
Builds the main application (main.py) into a standalone car.exe so it can
run on machines without Python installed.

Usage:
    venv/Scripts/python.exe build_car.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
ICON_PATH = ROOT / "assets" / "icon" / "car.ico"
DIST_PATH = ROOT / "assets" / "export" / "Car"


def collect_data_args() -> list[str]:
    """
    Builds --add-data entries for runtime assets. Deliberately excludes raw
    training images (thousands of files, multiple GB) — only the trained
    model weights and UI assets are needed once the app is running.
    """
    args = []

    def add(src: Path, dest: str):
        if not src.exists():
            return
        sep = ";" if os.name == "nt" else ":"
        args.append(f"--add-data={src}{sep}{dest}")

    # Icons + main-menu art
    add(ROOT / "assets" / "icon", "assets/icon")

    # Preview GIFs shown on the Options screen
    if (ROOT / "assets" / "gif").exists():
        add(ROOT / "assets" / "gif", "assets/gif")

    # Trained traffic-monitoring models (Option 2) — only if training has
    # produced them; the app falls back to heuristics if these are missing
    model_dir = ROOT / "assets" / "training" / "traffic_dataset" / "model"
    for fname in ("accident_detector.pt", "severity_detector.pt"):
        fpath = model_dir / fname
        if fpath.exists():
            add(fpath, "assets/training/traffic_dataset/model")
        else:
            print(f"[WARN] {fname} not found — skipping (train.py hasn't produced it yet).")

    # Trained CARLA simulation detectors (Option 3) — vehicle detector and
    # the dedicated accident detector trained via train2.py
    sim_model_dir = ROOT / "assets" / "training" / "simulation_dataset"
    for fname in ("simulation_detector.pt", "simulation_accident_detector.pt"):
        fpath = sim_model_dir / fname
        if fpath.exists():
            add(fpath, "assets/training/simulation_dataset")
        else:
            print(f"[WARN] {fname} not found — skipping.")

    # App config (created fresh on first run if absent, but ship defaults)
    if (ROOT / "config.json").exists():
        add(ROOT / "config.json", ".")

    return args


def main():
    print("=========================================================")
    print("             COMPILING MAIN APPLICATION TO EXE           ")
    print("=========================================================")

    os.chdir(ROOT)
    print(f"[INFO] Working directory set to: {ROOT}")

    try:
        import PyInstaller  # noqa: F401
        print("[INFO] PyInstaller is already installed.")
    except ImportError:
        print("[INFO] PyInstaller not detected. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    if not ICON_PATH.exists():
        print(f"[CRITICAL] Icon not found at {ICON_PATH}. Run the icon conversion step first.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name=car",
        f"--icon={ICON_PATH}",
        f"--distpath={DIST_PATH}",
        "--workpath=build",
        "--specpath=.",
        # ultralytics ships its own default-config YAMLs / data files that
        # plain --hidden-import won't catch
        "--collect-all=ultralytics",
        "--hidden-import=scripts.option_3.simulation_client",
        *collect_data_args(),
        "main.py",
    ]

    print(f"\n[EXEC] Running command:\n{' '.join(cmd)}\n")

    try:
        subprocess.check_call(cmd)

        # Ship a starter config.json alongside the exe — config.py reads/
        # writes it relative to wherever the exe is launched from, so having
        # a sensible default there avoids a blank first-run state
        cfg_src = ROOT / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, DIST_PATH / "config.json")

        print("\n=========================================================")
        print(" SUCCESS: Compiling finished successfully!")
        print(f" Executable 'car.exe' is available in '{DIST_PATH}'.")
        print("=========================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL] Compilation failed with error code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
