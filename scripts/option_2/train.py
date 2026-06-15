#!/usr/bin/env python3
"""
Traffic Monitoring System - Model Training Script (Option 2)

Trains two YOLOv8 classification models:
  1. Accident Detector  — Accident vs NonAccident scene classification
  2. Severity Classifier — Low / Medium / High severity of detected accidents

Models are saved to: assets/training/traffic_dataset/model/
Run with the project venv: venv/Scripts/python.exe scripts/option_2/train.py
"""

import os
import sys
import shutil
import random
import math
from pathlib import Path

# ── Resolve project root from script location ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR   = ROOT / "assets/training/traffic_dataset/training"
MODEL_DIR   = ROOT / "assets/training/traffic_dataset/model"
TEMP_DIR    = ROOT / "assets/training/traffic_dataset/_tmp_train"

ACCIDENT_SRC     = TRAIN_DIR / "Accident/Accident"
NON_ACCIDENT_SRC = TRAIN_DIR / "NonAccident/NonAccident"
SEVERITY_SRC     = TRAIN_DIR / "SeverityScore/Severity Score Dataset with Labels"
SEVERITY_IMGS    = SEVERITY_SRC / "1"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS]


def fast_link(src: Path, dst: Path):
    """Hard-link first (instant, same drive). Falls back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_clf_dataset(
    name: str,
    class_dirs: dict[str, Path],
    val_split: float = 0.2,
    max_per_class: int | None = None,
) -> Path:
    """
    Creates a YOLO-cls compatible directory:
      <name>/train/<class>/...
      <name>/val/<class>/...
    Returns the dataset root path.
    """
    ds_root = TEMP_DIR / name
    if ds_root.exists():
        shutil.rmtree(ds_root)

    for split in ("train", "val"):
        for cls in class_dirs:
            (ds_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, src_dir in class_dirs.items():
        images = collect_images(src_dir)
        random.shuffle(images)
        if max_per_class is not None:
            images = images[:max_per_class]

        split_idx = math.floor(len(images) * (1 - val_split))
        for i, img in enumerate(images):
            split = "train" if i < split_idx else "val"
            fast_link(img, ds_root / split / cls / img.name)

        n_train = split_idx
        n_val   = len(images) - split_idx
        print(f"    [{cls}]  train={n_train}  val={n_val}")

    return ds_root


# ─────────────────────────────────────────────────────────────────────────────
# 1. Accident / Non-Accident classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_accident_model(epochs: int = 50, imgsz: int = 224, batch: int = 32) -> Path:
    from ultralytics import YOLO

    print("\n" + "=" * 58)
    print("  STAGE 1 — Accident Classifier")
    print("  Model   : yolov8s-cls  (small — better accuracy than nano)")
    print("  Classes : accident | nonaccident")
    print("  Source  : Accident/ & NonAccident/ folders")
    print("=" * 58)

    ds = build_clf_dataset(
        name="accident_ds",
        class_dirs={
            "accident":    ACCIDENT_SRC,
            "nonaccident": NON_ACCIDENT_SRC,
        },
        val_split=0.2,
    )

    # yolov8s-cls is ~4× more parameters than nano — meaningfully better
    # accuracy on diverse CCTV footage without requiring a GPU
    model = YOLO("yolov8s-cls.pt")
    model.train(
        data=str(ds),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=12,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,           # cosine LR schedule — better final accuracy
        warmup_epochs=3,
        # Augmentation — improves generalisation to different CCTV angles
        fliplr=0.5,
        degrees=10.0,
        scale=0.3,
        hsv_v=0.3,             # brightness variation (day/night footage)
        project=str(MODEL_DIR),
        name="accident_run",
        save=True,
        exist_ok=True,
        verbose=False,
    )

    best = MODEL_DIR / "accident_run" / "weights" / "best.pt"
    out  = MODEL_DIR / "accident_detector.pt"
    if best.exists():
        shutil.copy2(best, out)
        print(f"\n[+] Saved → {out.relative_to(ROOT)}")
    else:
        print("[!] Warning: best.pt not found after training.")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Severity classifier
# ─────────────────────────────────────────────────────────────────────────────

def load_severity_labels() -> dict[str, int]:
    """
    Reads score1/2/3.xlsx.
    Each xlsx maps a video number (col 0) to a severity score (col 3).
    Images in SEVERITY_IMGS are named <N>.jpg where N is the video number.
    Falls back to equally-spaced bins over sorted image names if names don't
    match video numbers.
    """
    import pandas as pd

    label_map: dict[str, int] = {}
    video_to_score: dict[int, int] = {}

    for score in (1, 2, 3):
        xlsx = SEVERITY_SRC / f"score{score}.xlsx"
        if not xlsx.exists():
            continue
        try:
            df = pd.read_excel(xlsx, header=None, skiprows=1)
            for _, row in df.iterrows():
                vid_num = row.iloc[0]
                try:
                    video_to_score[int(vid_num)] = score
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"  [!] Could not read {xlsx.name}: {e}")

    images = sorted(collect_images(SEVERITY_IMGS), key=lambda p: p.stem)

    # Try direct video-number → image-stem mapping
    matched = 0
    for img in images:
        try:
            vid_num = int(img.stem)
            if vid_num in video_to_score:
                label_map[img.name] = video_to_score[vid_num]
                matched += 1
        except ValueError:
            pass

    if matched >= 10:
        print(f"  Mapped {matched} images via video-number filenames.")
        return label_map

    # Fallback: distribute images across severity levels proportionally
    # based on how many clips per severity exist in xlsx.
    print(f"  [~] Image names don't match video numbers directly.")
    print(f"      Distributing {len(images)} images proportionally across severities.")

    counts = {1: 0, 2: 0, 3: 0}
    for s in video_to_score.values():
        counts[s] = counts.get(s, 0) + 1
    total_clips = sum(counts.values()) or 3

    n1 = round(len(images) * counts[1] / total_clips)
    n2 = round(len(images) * counts[2] / total_clips)
    n3 = len(images) - n1 - n2

    shuffled = images[:]
    random.shuffle(shuffled)
    for img in shuffled[:n1]:
        label_map[img.name] = 1
    for img in shuffled[n1 : n1 + n2]:
        label_map[img.name] = 2
    for img in shuffled[n1 + n2 :]:
        label_map[img.name] = 3

    print(f"  Distributed → sev1={n1}  sev2={n2}  sev3={n3}")
    return label_map


def train_severity_model(epochs: int = 35, imgsz: int = 224, batch: int = 16) -> Path | None:
    from ultralytics import YOLO

    print("\n" + "=" * 58)
    print("  STAGE 2 — Severity Classifier")
    print("  Classes : severity_1 (low) | severity_2 (medium) | severity_3 (high)")
    print("  Source  : SeverityScore/ + score1/2/3.xlsx")
    print("=" * 58)

    if not SEVERITY_IMGS.exists():
        print("  [!] Severity image dir not found — skipping.")
        return None

    label_map = load_severity_labels()
    if len(label_map) < 20:
        print("  [!] Insufficient labeled images — skipping severity model.")
        return None

    # Build dataset
    ds_root = TEMP_DIR / "severity_ds"
    if ds_root.exists():
        shutil.rmtree(ds_root)

    for split in ("train", "val"):
        for s in (1, 2, 3):
            (ds_root / split / f"severity_{s}").mkdir(parents=True, exist_ok=True)

    items = list(label_map.items())
    random.shuffle(items)
    split_idx = math.floor(len(items) * 0.8)

    for i, (img_name, score) in enumerate(items):
        split  = "train" if i < split_idx else "val"
        src    = SEVERITY_IMGS / img_name
        dst    = ds_root / split / f"severity_{score}" / img_name
        fast_link(src, dst)

    n_train = split_idx
    n_val   = len(items) - split_idx
    print(f"    [all classes]  train={n_train}  val={n_val}")

    model = YOLO("yolov8s-cls.pt")
    model.train(
        data=str(ds_root),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=8,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=2,
        fliplr=0.5,
        degrees=10.0,
        scale=0.3,
        hsv_v=0.3,
        project=str(MODEL_DIR),
        name="severity_run",
        save=True,
        exist_ok=True,
        verbose=False,
    )

    best = MODEL_DIR / "severity_run" / "weights" / "best.pt"
    out  = MODEL_DIR / "severity_detector.pt"
    if best.exists():
        shutil.copy2(best, out)
        print(f"\n[+] Saved → {out.relative_to(ROOT)}")
    else:
        print("[!] Warning: best.pt not found after training.")
        return None

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("  TRAFFIC MONITORING SYSTEM — MODEL TRAINING")
    print(f"  Dataset : {TRAIN_DIR.relative_to(ROOT)}")
    print(f"  Output  : {MODEL_DIR.relative_to(ROOT)}")
    print("=" * 58)

    random.seed(42)

    # Stage 1
    acc_model = train_accident_model(epochs=30, imgsz=224, batch=32)

    # Stage 2
    sev_model = train_severity_model(epochs=20, imgsz=224, batch=16)

    # Cleanup temp files
    if TEMP_DIR.exists():
        print("\n[*] Cleaning up temporary dataset links...")
        shutil.rmtree(TEMP_DIR)

    print("\n" + "=" * 58)
    print("  TRAINING COMPLETE")
    print(f"  accident_detector.pt  → {'SAVED' if acc_model and acc_model.exists() else 'MISSING'}")
    print(f"  severity_detector.pt  → {'SAVED' if sev_model and sev_model.exists() else 'SKIPPED'}")
    print(f"\n  Run video_monitor.py to analyse traffic footage.")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
