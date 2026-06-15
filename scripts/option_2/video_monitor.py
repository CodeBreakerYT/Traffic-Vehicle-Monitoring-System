#!/usr/bin/env python3
"""
Traffic Monitoring System - Option 2: Video Upload & Real-time Analysis

Opens a video file via file-dialog, then displays a composite window:
  LEFT  — video feed with YOLO bounding boxes (vehicles + people)
  RIGHT — live analytics panel (counts, accident status, severity, FPS)

Requires models trained by train.py:
  assets/training/traffic_dataset/model/accident_detector.pt
  assets/training/traffic_dataset/model/severity_detector.pt  (optional)

Run:
  venv/Scripts/python.exe scripts/option_2/video_monitor.py
Keys:
  Q / ESC — quit
  SPACE   — pause / resume
"""

import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets/training/traffic_dataset/model"

# COCO class IDs relevant to traffic monitoring
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
PERSON_CLASS    = 0

# Detection confidence threshold
CONF_THRESH = 0.35

# How often (frames) to run the heavier accident/severity classification
CLASSIFY_EVERY = 12

# ── Dark cyber-theme colours (BGR for OpenCV) ─────────────────────────────────
C_BG       = (25,  10,  35)
C_PANEL    = (12,   5,  20)
C_CYAN     = (220, 210,  10)
C_GREEN    = (60, 255,  80)
C_MAGENTA  = (200,  10, 200)
C_RED      = ( 20,  20, 255)
C_YELLOW   = (  0, 220, 255)
C_WHITE    = (240, 240, 255)
C_MUTED    = (100,  90, 130)
C_ACCENT   = ( 10, 180, 200)
C_GRID     = ( 30,  10,  45)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """Returns (detector, accident_clf, severity_clf). Classifiers may be None."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed.")
        print(f"    Run: {sys.executable} -m pip install ultralytics")
        sys.exit(1)

    print("[*] Loading YOLOv8n detection model (pretrained COCO)...")
    detector = YOLO("yolov8n.pt")

    acc_path = MODEL_DIR / "accident_detector.pt"
    if acc_path.exists():
        print(f"[*] Loading accident classifier: {acc_path.name}")
        accident_clf = YOLO(str(acc_path))
    else:
        print("[!] accident_detector.pt not found — run train.py first.")
        print("    Accident detection will be unavailable.")
        accident_clf = None

    sev_path = MODEL_DIR / "severity_detector.pt"
    if sev_path.exists():
        print(f"[*] Loading severity classifier: {sev_path.name}")
        severity_clf = YOLO(str(sev_path))
    else:
        print("[~] severity_detector.pt not found — severity will use heuristic fallback.")
        severity_clf = None

    return detector, accident_clf, severity_clf


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(detector, frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run vehicle + person detection on frame.
    Returns annotated frame copy and count dict.
    """
    annotated = frame.copy()
    counts: dict[str, int] = {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0}

    results = detector(
        frame,
        verbose=False,
        conf=CONF_THRESH,
        classes=[PERSON_CLASS] + list(VEHICLE_CLASSES.keys()),
    )

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == PERSON_CLASS:
                label  = "Person"
                color  = C_YELLOW
            elif cls_id in VEHICLE_CLASSES:
                label  = VEHICLE_CLASSES[cls_id]
                color  = C_CYAN
            else:
                continue

            counts[label] += 1

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Corner tick marks (cyber look)
            tick = 10
            for (sx, sy), (dx, dy) in [
                ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
            ]:
                cv2.line(annotated, (sx, sy), (sx + dx * tick, sy), color, 2)
                cv2.line(annotated, (sx, sy), (sx, sy + dy * tick), color, 2)

            # Label background + text
            tag = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 3), FONT, 0.45, C_BG, 1, cv2.LINE_AA)

    return annotated, counts


def heuristic_severity(counts: dict, accident: bool) -> int | None:
    """Fallback severity: 1-3 based on total vehicle/person density."""
    if not accident:
        return None
    total = sum(counts.values())
    if total >= 10:
        return 3
    elif total >= 5:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Analytics panel drawing
# ─────────────────────────────────────────────────────────────────────────────

def make_panel(stats: dict, panel_w: int, panel_h: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), C_PANEL, dtype=np.uint8)

    # Subtle grid lines
    for y in range(0, panel_h, 30):
        cv2.line(panel, (0, y), (panel_w, y), C_GRID, 1)

    def txt(msg, x, y, color=C_WHITE, scale=0.50, bold=False):
        cv2.putText(panel, msg, (x, y), FONT, scale, color, 2 if bold else 1, cv2.LINE_AA)

    def hdiv(y):
        cv2.line(panel, (12, y), (panel_w - 12, y), (55, 20, 70), 1)

    y = 28

    # ── Header ───────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (panel_w, 38), (40, 10, 55), -1)
    txt("ANALYTICS", panel_w // 2 - 50, 25, C_CYAN, 0.65, bold=True)

    y = 55

    # ── System info ───────────────────────────────────────────────────────────
    txt("SYSTEM", 14, y, C_MUTED, 0.38)
    y += 18
    txt(f"FPS    :  {stats['fps']:>5.1f}", 14, y, C_GREEN)
    y += 22
    txt(f"FRAME  :  {stats['frame']:>5d} / {stats['total']:d}", 14, y, C_GREEN)
    y += 22
    txt(f"TIME   :  {stats['elapsed']}", 14, y, C_GREEN)
    y += 10
    hdiv(y); y += 16

    # ── Vehicles ─────────────────────────────────────────────────────────────
    txt("VEHICLES", 14, y, C_MUTED, 0.38)
    y += 18
    counts = stats.get("counts", {})
    total_vehicles = 0
    for label in ("Car", "Truck", "Bus", "Motorcycle"):
        n = counts.get(label, 0)
        total_vehicles += n
        col = C_CYAN if n > 0 else C_MUTED
        txt(f"{label:<11}: {n}", 14, y, col)
        y += 22

    txt(f"{'TOTAL':<11}: {total_vehicles}", 14, y, C_YELLOW, bold=True)
    y += 12
    hdiv(y); y += 16

    # ── Pedestrians ───────────────────────────────────────────────────────────
    txt("PEDESTRIANS", 14, y, C_MUTED, 0.38)
    y += 18
    people = counts.get("Person", 0)
    txt(f"{'COUNT':<11}: {people}", 14, y, C_YELLOW if people else C_MUTED, bold=bool(people))
    y += 12
    hdiv(y); y += 16

    # ── Accident status ───────────────────────────────────────────────────────
    txt("ACCIDENT", 14, y, C_MUTED, 0.38)
    y += 18
    accident = stats.get("accident")
    if accident is None:
        txt("STATUS  : SCANNING", 14, y, C_MUTED)
        y += 22
    elif accident:
        txt("DETECTED: YES", 14, y, C_RED, bold=True)
        y += 22
        conf = stats.get("accident_conf", 0.0)
        txt(f"CONF    : {conf:.0%}", 14, y, C_RED)
        y += 22
    else:
        txt("DETECTED: NO", 14, y, C_GREEN)
        y += 22
        conf = stats.get("accident_conf", 0.0)
        txt(f"CONF    : {conf:.0%}", 14, y, C_MUTED)
        y += 22

    hdiv(y); y += 16

    # ── Severity ─────────────────────────────────────────────────────────────
    txt("SEVERITY", 14, y, C_MUTED, 0.38)
    y += 18
    severity = stats.get("severity")
    sev_label = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    sev_color = {1: C_GREEN, 2: C_YELLOW, 3: C_RED}

    if severity is None or not accident:
        txt("LEVEL   : N/A", 14, y, C_MUTED)
    else:
        lbl = sev_label.get(severity, str(severity))
        col = sev_color.get(severity, C_WHITE)
        txt(f"LEVEL   : {lbl}", 14, y, col, bold=True)
        if stats.get("severity_heuristic"):
            y += 20
            txt("(estimated)", 14, y, C_MUTED, 0.35)
    y += 22
    hdiv(y); y += 16

    # ── Total detections ─────────────────────────────────────────────────────
    txt("DETECTIONS", 14, y, C_MUTED, 0.38)
    y += 18
    total_det = sum(counts.values())
    txt(f"TOTAL   : {total_det}", 14, y, C_WHITE, bold=True)
    y += 12
    hdiv(y); y += 16

    # ── Status bar ────────────────────────────────────────────────────────────
    state_str = "PAUSED" if stats.get("paused") else "PLAYING"
    state_col = C_YELLOW if stats.get("paused") else C_GREEN
    txt(f"STATE   : {state_str}", 14, y, state_col)
    y += 22

    # ── Blinking REC dot ─────────────────────────────────────────────────────
    if not stats.get("paused") and int(time.time() * 2) % 2 == 0:
        cv2.circle(panel, (panel_w - 18, 20), 5, C_RED, -1)

    # ── Footer ────────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, panel_h - 38), (panel_w, panel_h), (25, 8, 35), -1)
    txt("SPACE:PAUSE  Q/ESC:QUIT", 8, panel_h - 22, C_MUTED, 0.35)
    txt("TRAFFIC MONITOR v2.0", 8, panel_h - 8, C_MUTED, 0.30)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Video HUD overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_video_hud(frame: np.ndarray, stats: dict):
    """Draw top-left info bar and accident alert on the video frame."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    overlay_txt = (
        f"FPS {stats['fps']:.1f}  |  "
        f"Frame {stats['frame']}/{stats['total']}  |  "
        f"Time {stats['elapsed']}"
    )
    cv2.putText(frame, overlay_txt, (10, 22), FONT, 0.55, C_GREEN, 1, cv2.LINE_AA)

    # Bottom accident alert
    if stats.get("accident"):
        bar_h = 38
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        sev = stats.get("severity")
        sev_str = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}.get(sev, "UNKNOWN") if sev else ""
        alert = f"!! ACCIDENT DETECTED  |  SEVERITY: {sev_str}  !!"
        (tw, _), _ = cv2.getTextSize(alert, FONT, 0.65, 2)
        x = (w - tw) // 2
        # Flash red/yellow
        col = C_RED if int(time.time() * 3) % 2 == 0 else C_YELLOW
        cv2.putText(frame, alert, (x, h - 10), FONT, 0.65, col, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar at bottom of window
# ─────────────────────────────────────────────────────────────────────────────

def draw_progress_bar(composite: np.ndarray, frame_num: int, total: int, bar_h: int = 5):
    h, w = composite.shape[:2]
    progress = frame_num / max(total, 1)
    filled_w = int(w * progress)
    cv2.rectangle(composite, (0, h - bar_h), (w, h), (40, 20, 55), -1)
    cv2.rectangle(composite, (0, h - bar_h), (filled_w, h), C_ACCENT, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor(video_path: str, detector, accident_clf, severity_clf):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video:\n{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale video to fit 720p height
    display_h = 720
    display_w = int(vid_w * display_h / vid_h)
    panel_w   = 280

    win_title = "TRAFFIC VEHICLE MONITORING SYSTEM — OPTION 2"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, display_w + panel_w, display_h + 5)

    stats = {
        "fps":            0.0,
        "frame":          0,
        "total":          total_frames,
        "elapsed":        "00:00",
        "counts":         {},
        "accident":       None,
        "accident_conf":  0.0,
        "severity":       None,
        "severity_heuristic": False,
        "paused":         False,
    }

    frame_num   = 0
    fps_counter = 0
    fps_timer   = time.time()
    start_time  = time.time() - 0.0001   # avoid division by zero

    print(f"\n[*] Video: {Path(video_path).name}")
    print(f"    {vid_w}x{vid_h}  |  {src_fps:.1f} fps  |  {total_frames} frames")
    print("[*] SPACE = pause/resume   Q/ESC = quit\n")

    while True:
        # ── Handle pause ──────────────────────────────────────────────────────
        if not stats["paused"]:
            ret, frame = cap.read()
            if not ret:
                print("[*] End of video.")
                break
            frame_num += 1
            frame = cv2.resize(frame, (display_w, display_h))

            # ── Detection (every frame) ───────────────────────────────────────
            annotated, counts = run_detection(detector, frame)
            stats["counts"] = counts

            # ── Scene classification (every N frames) ─────────────────────────
            if frame_num % CLASSIFY_EVERY == 0:
                if accident_clf is not None:
                    acc_res = accident_clf(frame, verbose=False)
                    probs   = acc_res[0].probs
                    # YOLOv8-cls orders classes alphabetically:
                    # 0 = accident, 1 = nonaccident
                    top1       = int(probs.top1)
                    conf       = float(probs.top1conf)
                    is_acc     = (top1 == 0)
                    stats["accident"]      = is_acc
                    stats["accident_conf"] = conf

                    if is_acc:
                        if severity_clf is not None:
                            sev_res  = severity_clf(frame, verbose=False)
                            sev_top1 = int(sev_res[0].probs.top1)
                            # 0=severity_1, 1=severity_2, 2=severity_3
                            stats["severity"]           = sev_top1 + 1
                            stats["severity_heuristic"] = False
                        else:
                            stats["severity"]           = heuristic_severity(counts, is_acc)
                            stats["severity_heuristic"] = True
                    else:
                        stats["severity"] = None
                else:
                    # No classifier — heuristic only
                    stats["accident"] = None
                    stats["severity"] = None

            # ── Update stats ──────────────────────────────────────────────────
            stats["frame"]   = frame_num
            stats["elapsed"] = f"{int(time.time() - start_time) // 60:02d}:{int(time.time() - start_time) % 60:02d}"

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 0.5:
                stats["fps"] = fps_counter / (now - fps_timer)
                fps_counter  = 0
                fps_timer    = now

            # ── HUD on video ──────────────────────────────────────────────────
            draw_video_hud(annotated, stats)

        else:
            # Paused — reuse last annotated frame (if it exists)
            if "last_annotated" not in stats:
                key = cv2.waitKey(30) & 0xFF
                if key in (32,):  # SPACE
                    stats["paused"] = False
                elif key in (ord("q"), ord("Q"), 27):
                    break
                continue

            annotated = stats["last_annotated"]

        stats["last_annotated"] = annotated.copy()

        # ── Build composite (video | panel) ──────────────────────────────────
        panel     = make_panel(stats, panel_w, display_h)
        composite = np.hstack([annotated, panel])
        draw_progress_bar(composite, frame_num, total_frames)

        cv2.imshow(win_title, composite)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == 32:  # SPACE
            stats["paused"] = not stats["paused"]
            if not stats["paused"]:
                # Reset FPS timer so we don't get a huge spike
                fps_timer   = time.time()
                fps_counter = 0

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Session ended.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # File picker (hidden root window)
    tk_root = tk.Tk()
    tk_root.withdraw()
    tk_root.lift()
    tk_root.attributes("-topmost", True)

    video_path = filedialog.askopenfilename(
        title="Select Traffic Video for Analysis",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.MOV *.MP4 *.AVI"),
            ("All Files",   "*.*"),
        ],
    )
    tk_root.destroy()

    if not video_path:
        print("[!] No file selected — exiting.")
        return

    detector, accident_clf, severity_clf = load_models()
    run_monitor(video_path, detector, accident_clf, severity_clf)


if __name__ == "__main__":
    main()
