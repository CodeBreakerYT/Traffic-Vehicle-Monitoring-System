"""
YOLO-based video frame analyzer.
Imported by main.py; ultralytics is loaded lazily on first call to .load().

Detection:  yolov8s.pt  (pretrained COCO — vehicles + people)
Accident:   accident_detector.pt  (trained on Accident/NonAccident dataset)
Severity:   severity_detector.pt  (trained on SeverityScore dataset)

Accident confirmation is gated primarily on an ABRUPT, near-instant full
stop detected by the vehicle tracker (a real collision signature) — not on
proximity or gradual deceleration, which happen constantly in normal
traffic (braking at lights, queuing, slowing for turns). The scene
classifier can only boost confidence on top of that, never trigger alone,
since a scene-level classifier trained on a small/mixed dataset is prone to
overfitting on irrelevant visual cues and producing confident false
positives.
"""

import math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "assets/training/traffic_dataset/model"

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
PERSON_CLASS    = 0

CONF_THRESHOLD  = 0.25   # lower than default to catch distant / partial vehicles
CLASSIFY_EVERY  = 10     # run classifiers every N frames in the processing loop

# Per-class BGR colours — distinct so each type is instantly readable on screen
_CLASS_COLOR: dict[str, tuple] = {
    "Car":        (220, 195,  10),   # cyan
    "Motorcycle": ( 30, 220,  30),   # green
    "Bus":        (  0, 100, 255),   # orange
    "Truck":      (255,  60, 200),   # purple
    "Person":     (  0, 215, 255),   # yellow
}
_CLR_TXT_BG   = (20, 10, 30)
_CLR_ACCIDENT = (0, 0, 255)

# Vehicle-vehicle: boxes must genuinely overlap (touching), not just be near
_VEHICLE_COLLISION_IOU = 0.20
_MIN_BOX_AREA_PCT      = 0.005   # ignore tiny/far-away boxes

# How many frames a confirmed accident box stays on screen without being
# re-confirmed, before it's cleared (avoids single-frame flicker either way)
_ACCIDENT_PERSIST_FRAMES = 20

# Classifier-only confirmation (no motion evidence available): needs this
# confidence, sustained for this many consecutive classify cycles
_CLF_CONFIRM_CONF   = 0.75
_CLF_CONFIRM_STREAK = 2


def _iou(b1: tuple, b2: tuple) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class _VehicleTracker:
    """
    Centroid tracker with a strict collision test: a vehicle must have been
    moving at a meaningful clip, then come to a near-total stop within a
    single frame-to-frame step, and STAY stopped on the following step too
    (filters out single-frame tracking jitter). That sudden-stop signature
    is what an actual impact looks like — normal braking decelerates
    gradually over many frames, not instantly.

    - If the stopped vehicle's box overlaps another vehicle's box  -> "vehicle" collision.
    - Otherwise (isolated sudden stop, nothing else nearby)        -> "object" collision
      (proxy for hitting a pole/wall/curb — we can't detect the object
      itself since it's not a tracked class, only that the vehicle's own
      motion abruptly and totally arrested).
    """

    def __init__(self, max_match_dist=90.0, max_missed=6, history_len=6):
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.max_match_dist = max_match_dist
        self.max_missed = max_missed
        self.history_len = history_len

    def reset(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes: list[tuple]):
        """
        Matches tracks to this frame's detections using optimal (Hungarian)
        assignment over a cost that combines centroid distance with box-size
        similarity — not just "nearest centroid". A greedy nearest-neighbor
        matcher (the old approach) is exactly wrong at the moment that
        matters most: when two vehicles are converging/overlapping right
        before and during a collision, their centroids are close together
        and a greedy matcher can swap which track owns which box, causing
        the accident box to land on the wrong vehicle.
        """
        n_tracks = len(self.tracks)
        n_boxes  = len(boxes)

        if n_tracks == 0:
            for b in boxes:
                self._add_track(b)
            return
        if n_boxes == 0:
            for tr in self.tracks.values():
                tr["missed"] += 1
            self._prune()
            return

        track_ids = list(self.tracks.keys())
        centers   = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in boxes]
        areas     = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]

        INVALID = 1e6
        cost = np.full((n_tracks, n_boxes), INVALID)
        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            tb = tr["box"]
            tr_area = max((tb[2]-tb[0])*(tb[3]-tb[1]), 1)
            for j, c in enumerate(centers):
                d = math.hypot(c[0] - tr["center"][0], c[1] - tr["center"][1])
                if d > self.max_match_dist:
                    continue
                # Reject matches where box size differs too much to plausibly
                # be the same vehicle (prevents snapping onto a different,
                # nearby vehicle of a different size)
                size_ratio = max(tr_area, areas[j]) / max(min(tr_area, areas[j]), 1)
                if size_ratio > 2.5:
                    continue
                cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks, matched_boxes = set(), set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= INVALID:
                continue
            tid, tr = track_ids[i], self.tracks[track_ids[i]]
            tr["history"].append(tr["center"])
            if len(tr["history"]) > self.history_len:
                tr["history"].pop(0)
            tr["center"] = centers[j]
            tr["box"]    = boxes[j]
            tr["missed"] = 0
            matched_tracks.add(tid)
            matched_boxes.add(j)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
        self._prune()

        for j, b in enumerate(boxes):
            if j not in matched_boxes:
                self._add_track(b)

    def _add_track(self, box: tuple):
        center = ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)
        self.tracks[self.next_id] = {"center": center, "box": box, "history": [], "missed": 0}
        self.next_id += 1

    def _prune(self):
        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr["missed"] <= self.max_missed}

    def detect_collisions(self, frame_w: int, frame_h: int) -> list[dict]:
        """
        Returns a list of {"box", "kind" ("vehicle"|"object"), "conf"} for
        every track whose motion this frame matches a real-impact signature.
        """
        diag = math.hypot(frame_w, frame_h)

        # Vehicle-vehicle: must have been moving, then near-total stop,
        # sustained for 2 consecutive steps (not a single noisy blip)
        min_speed_veh = diag * 0.018
        stop_ratio_veh = 0.12

        # Isolated/object: demand an even faster prior speed and an even
        # harder stop, since we have no way to confirm what was struck —
        # this keeps it from firing on routine hard braking
        min_speed_obj = diag * 0.026
        stop_ratio_obj = 0.08

        frame_area = frame_w * frame_h
        results: list[dict] = []

        for tid, tr in self.tracks.items():
            hist = tr["history"]
            if len(hist) < 4:
                continue
            pts = hist + [tr["center"]]
            speeds = [math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts) - 1)]
            if len(speeds) < 3:
                continue
            speed_before, speed_1, speed_2 = speeds[-3], speeds[-2], speeds[-1]

            box = tr["box"]
            if (box[2]-box[0]) * (box[3]-box[1]) < frame_area * _MIN_BOX_AREA_PCT:
                continue  # too small/far away to trust the motion signal

            # Does this box currently overlap another vehicle's box?
            best_iou = 0.0
            for tid2, tr2 in self.tracks.items():
                if tid2 == tid:
                    continue
                best_iou = max(best_iou, _iou(box, tr2["box"]))

            if (speed_before >= min_speed_veh
                    and speed_1 <= speed_before * stop_ratio_veh
                    and speed_2 <= speed_before * stop_ratio_veh
                    and best_iou >= _VEHICLE_COLLISION_IOU):
                conf = min(0.65 + best_iou * 0.4, 0.93)
                results.append({"box": box, "kind": "vehicle", "conf": conf})

            elif (speed_before >= min_speed_obj
                    and speed_1 <= speed_before * stop_ratio_obj
                    and speed_2 <= speed_before * stop_ratio_obj
                    and best_iou < _VEHICLE_COLLISION_IOU):
                results.append({"box": box, "kind": "object", "conf": 0.58})

        return results


def draw_accident_box(frame, box: tuple, kind: str = "vehicle"):
    """Draws a thick red box + ACCIDENT label around the damaged vehicle, plus a pointer arrow above it."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), _CLR_ACCIDENT, 4)

    label = "ACCIDENT" if kind == "vehicle" else "ACCIDENT (OBJECT)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    lx = max(2, min(x1, w - tw - 8))
    ly = max(th + 6, y1 - 8)
    cv2.rectangle(frame, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 4), _CLR_ACCIDENT, -1)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cx = (x1 + x2) // 2
    tip_y  = max(28, y1 - 14)
    tail_y = max(10, tip_y - 30)
    cv2.arrowedLine(frame, (cx, tail_y), (cx, tip_y), _CLR_ACCIDENT, 3, tipLength=0.45)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class VideoAnalyzer:
    """Wraps pretrained detection + trained accident / severity classifiers."""

    def __init__(self):
        self._detector     = None
        self._accident_clf = None
        self._severity_clf = None
        self.ready         = False
        self.has_accident  = False
        self.has_severity  = False
        self._tracker = _VehicleTracker()

        # Persisted confirmed-accident state (so the box doesn't flicker
        # frame-to-frame between re-confirmations)
        self._persist_box   = None
        self._persist_kind  = "vehicle"
        self._persist_conf  = 0.0
        self._persist_ttl   = 0

        # Consecutive classify-cycles where the scene classifier has
        # independently said "accident" with high confidence — used to let
        # the classifier confirm a crash on its own (see process() docstring
        # for why this path exists)
        self._clf_streak = 0

    def reset(self):
        """Call when starting a new video so stale tracking state doesn't leak in."""
        self._tracker.reset()
        self._persist_box  = None
        self._persist_ttl  = 0
        self._clf_streak   = 0

    def load(self) -> tuple[bool, bool]:
        """
        Loads all models.  Safe to call multiple times.
        Returns (has_accident_model, has_severity_model).
        """
        from ultralytics import YOLO

        # yolov8s gives significantly better recall for distant / small objects
        # (buses at the back, motorcycles between cars) vs yolov8n
        print("[VideoAnalyzer] Loading yolov8s detector (pretrained COCO)...")
        self._detector = YOLO("yolov8s.pt")

        acc_p = MODEL_DIR / "accident_detector.pt"
        if acc_p.exists():
            print(f"[VideoAnalyzer] Loading accident classifier: {acc_p.name}")
            self._accident_clf = YOLO(str(acc_p))
            self.has_accident  = True
        else:
            print("[VideoAnalyzer] accident_detector.pt not found — using motion heuristic only.")

        sev_p = MODEL_DIR / "severity_detector.pt"
        if sev_p.exists():
            print(f"[VideoAnalyzer] Loading severity classifier: {sev_p.name}")
            self._severity_clf = YOLO(str(sev_p))
            self.has_severity  = True
        else:
            print("[VideoAnalyzer] severity_detector.pt not found — using density heuristic.")

        self.ready = True
        return self.has_accident, self.has_severity

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, frame, classify: bool = False) -> tuple:
        """
        frame   : BGR numpy array.
        classify: run accident + severity models this call (motion-based
                  collision detection always runs, every frame, since it's cheap).

        Accident confirmation has two paths:
          1. Motion-confirmed (primary, strongest): an abrupt sustained stop
             was detected this frame or recently. Always wins when present.
          2. Classifier-confirmed (fallback): used when there's no motion
             evidence to find — e.g. footage that starts after the crash
             already happened (common in clipped/compilation footage, or a
             live feed that joins mid-event), where the vehicle was never
             seen "moving then stopping". Requires the classifier to say
             "accident" with high confidence on 2 consecutive classify
             cycles, not just one — this is what keeps it from reverting to
             the original false-positive problem (a single confident-but-
             wrong frame no longer flips the verdict on its own).

        Returns (annotated_bgr, stats_dict).
        stats_dict keys:
            counts           : dict {label: int}
            accident         : bool | None   (None when classify=False)
            accident_conf    : float
            severity         : int 1-3 | None
            severity_heuristic : bool
        """
        if not self.ready:
            return frame, _empty_stats()

        h, w = frame.shape[:2]
        annotated = frame.copy()
        counts: dict[str, int] = {
            "Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0
        }
        vehicle_boxes: list[tuple] = []

        # ── Object detection ─────────────────────────────────────────────────
        results = self._detector(
            frame,
            verbose=False,
            conf=CONF_THRESHOLD,
            classes=[PERSON_CLASS] + list(VEHICLE_CLASSES.keys()),
        )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == PERSON_CLASS:
                    label = "Person"
                elif cls_id in VEHICLE_CLASSES:
                    label = VEHICLE_CLASSES[cls_id]
                    vehicle_boxes.append((x1, y1, x2, y2))
                else:
                    continue

                counts[label] += 1
                color = _CLASS_COLOR[label]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                tk = 11
                for (sx, sy), (dx, dy) in [
                    ((x1, y1), ( 1,  1)), ((x2, y1), (-1,  1)),
                    ((x1, y2), ( 1, -1)), ((x2, y2), (-1, -1)),
                ]:
                    cv2.line(annotated, (sx, sy), (sx + dx * tk, sy), color, 2)
                    cv2.line(annotated, (sx, sy), (sx, sy + dy * tk), color, 2)

                tag = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated, tag, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_TXT_BG, 1, cv2.LINE_AA,
                )

        # ── Motion-based collision detection (every frame, classifier-free) ───
        # This is the PRIMARY signal — an abrupt, sustained full stop is what
        # an actual impact looks like. Mere proximity or gradual braking
        # (constant in normal traffic) does not qualify.
        self._tracker.update(vehicle_boxes)
        collisions = self._tracker.detect_collisions(w, h)

        if collisions:
            best = max(collisions, key=lambda c: c["conf"])
            self._persist_box  = best["box"]
            self._persist_kind = best["kind"]
            self._persist_conf = best["conf"]
            self._persist_ttl  = _ACCIDENT_PERSIST_FRAMES
        elif self._persist_ttl > 0:
            self._persist_ttl -= 1
            if self._persist_ttl == 0:
                self._persist_box = None

        motion_confirmed = self._persist_box is not None

        # ── Accident / severity classification ─────────────────────────────────
        accident, acc_conf, severity, sev_heuristic = None, 0.0, None, False

        if classify:
            if motion_confirmed:
                accident = True
                acc_conf = self._persist_conf
                self._clf_streak = 0  # motion already won; streak not needed

                # The classifier can boost confidence on top of motion
                # evidence, but motion is what decided this is an accident
                if self._accident_clf is not None:
                    acc_res = self._accident_clf(frame, verbose=False)
                    if int(acc_res[0].probs.top1) == 0:  # 0=accident (alphabetical)
                        acc_conf = min(acc_conf + 0.05, 0.97)

            elif self._accident_clf is not None:
                # No motion evidence this frame — fall back to the scene
                # classifier alone, but only confirm after it agrees on
                # several consecutive cycles (filters out single-frame
                # false positives, which is what broke this before)
                acc_res  = self._accident_clf(frame, verbose=False)
                clf_says = int(acc_res[0].probs.top1) == 0
                clf_conf = float(acc_res[0].probs.top1conf)

                if clf_says and clf_conf >= _CLF_CONFIRM_CONF:
                    self._clf_streak += 1
                else:
                    self._clf_streak = 0

                if self._clf_streak >= _CLF_CONFIRM_STREAK:
                    accident = True
                    acc_conf = min(0.55 + 0.08 * self._clf_streak, 0.9)
                    # No motion-derived box available — point at the
                    # largest detected vehicle (most likely the damaged
                    # one in a wreck/aftermath scene) or fall back to
                    # upper-frame center for a scene-level call
                    if vehicle_boxes:
                        box = max(vehicle_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    else:
                        box = (w//2 - 40, h//4 - 30, w//2 + 40, h//4 + 30)
                    self._persist_box  = box
                    self._persist_kind = "vehicle"
                    self._persist_conf = acc_conf
                    self._persist_ttl  = _ACCIDENT_PERSIST_FRAMES
                else:
                    accident = False
                    acc_conf = clf_conf if not clf_says else 0.0
            else:
                accident = False
                acc_conf = 0.0

            if accident:
                if self._severity_clf is not None:
                    sev_res  = self._severity_clf(frame, verbose=False)
                    severity = int(sev_res[0].probs.top1) + 1   # 0-indexed → 1,2,3
                else:
                    total_veh = sum(v for k, v in counts.items() if k != "Person")
                    if self._persist_kind == "vehicle" and total_veh >= 6:
                        severity = 3
                    elif self._persist_kind == "vehicle":
                        severity = 2
                    else:
                        severity = 1
                    sev_heuristic = True

        # Draw the confirmed accident box on every frame while it's still
        # within its persistence window (not just the classify frame)
        if self._persist_box is not None:
            draw_accident_box(annotated, self._persist_box, self._persist_kind)

        return annotated, {
            "counts":             counts,
            "accident":           accident,
            "accident_conf":      acc_conf,
            "severity":           severity,
            "severity_heuristic": sev_heuristic,
            "accident_location":  self._persist_box,
        }


def _empty_stats() -> dict:
    return {
        "counts":             {},
        "accident":           None,
        "accident_conf":      0.0,
        "severity":           None,
        "severity_heuristic": False,
        "accident_location":  None,
    }
