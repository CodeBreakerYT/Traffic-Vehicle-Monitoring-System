"""
YOLO-based video frame analyzer.
Imported by main.py; ultralytics is loaded lazily on first call to .load().

Detection:  yolov8s.pt  (pretrained COCO — vehicles + people)
Accident:   accident_detector.pt  (trained on Accident/NonAccident dataset)
Severity:   severity_detector.pt  (trained on SeverityScore dataset)

When the trained classifiers are absent a geometry-based heuristic
(IoU overlap between vehicle bounding-boxes) is used as a fallback.
"""

from pathlib import Path
import cv2

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
_CLR_TXT_BG = (20, 10, 30)

# Overlap heuristic thresholds
_OVERLAP_IOU_MIN  = 0.10   # IoU above this flags a potential collision
_OVERLAP_IOU_HIGH = 0.30   # IoU above this is strong evidence of collision
_MIN_BOX_AREA_PCT = 0.005  # ignore vehicle boxes smaller than 0.5 % of frame


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _max_vehicle_iou(boxes: list[tuple], frame_w: int, frame_h: int) -> float:
    """
    Returns the maximum pairwise IoU among vehicle bounding-boxes.
    Tiny boxes (far-away vehicles) are filtered out to reduce false positives.
    """
    frame_area = frame_w * frame_h
    big = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) > frame_area * _MIN_BOX_AREA_PCT]
    if len(big) < 2:
        return 0.0
    max_iou = 0.0
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            max_iou = max(max_iou, _iou(big[i], big[j]))
    return max_iou


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
            print("[VideoAnalyzer] accident_detector.pt not found — using overlap heuristic.")

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
        classify: run accident + severity models (or heuristic) this call.

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

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Cyber corner tick marks
                tk = 11
                for (sx, sy), (dx, dy) in [
                    ((x1, y1), ( 1,  1)), ((x2, y1), (-1,  1)),
                    ((x1, y2), ( 1, -1)), ((x2, y2), (-1, -1)),
                ]:
                    cv2.line(annotated, (sx, sy), (sx + dx * tk, sy), color, 2)
                    cv2.line(annotated, (sx, sy), (sx, sy + dy * tk), color, 2)

                # Label tag
                tag = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(
                    annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    annotated, tag, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_TXT_BG, 1, cv2.LINE_AA,
                )

        # ── Accident / severity ───────────────────────────────────────────────
        accident, acc_conf, severity, sev_heuristic = None, 0.0, None, False

        if classify:
            # Geometry signal: max IoU between any two vehicle boxes
            overlap_iou  = _max_vehicle_iou(vehicle_boxes, w, h)
            overlap_flag = overlap_iou >= _OVERLAP_IOU_MIN

            if self._accident_clf is not None:
                # ── Trained classifier path ──────────────────────────────────
                acc_res  = self._accident_clf(frame, verbose=False)
                probs    = acc_res[0].probs
                top1     = int(probs.top1)
                clf_conf = float(probs.top1conf)
                # YOLOv8-cls alphabetical order: 0=accident, 1=nonaccident
                clf_accident = (top1 == 0)

                # Combine: classifier OR strong overlap
                if clf_accident:
                    accident = True
                    # Boost confidence if overlap corroborates
                    acc_conf = min(clf_conf + 0.10 * overlap_flag, 1.0)
                elif overlap_iou >= _OVERLAP_IOU_HIGH:
                    # Overlap strong enough to override a "no accident" prediction
                    accident = True
                    acc_conf = 0.65
                else:
                    accident = False
                    acc_conf = clf_conf

            else:
                # ── Heuristic-only path (no trained classifier) ──────────────
                accident     = overlap_flag
                acc_conf     = min(0.45 + overlap_iou * 2.0, 0.92) if overlap_flag else 0.0
                sev_heuristic = True

            if accident:
                if self._severity_clf is not None:
                    # ── Trained severity model ───────────────────────────────
                    sev_res  = self._severity_clf(frame, verbose=False)
                    severity = int(sev_res[0].probs.top1) + 1   # 0-indexed → 1,2,3
                else:
                    # ── Density + overlap heuristic ──────────────────────────
                    total_veh = sum(v for k, v in counts.items() if k != "Person")
                    if overlap_iou >= _OVERLAP_IOU_HIGH or total_veh >= 8:
                        severity = 3
                    elif overlap_iou >= _OVERLAP_IOU_MIN or total_veh >= 4:
                        severity = 2
                    else:
                        severity = 1
                    sev_heuristic = True

        return annotated, {
            "counts":             counts,
            "accident":           accident,
            "accident_conf":      acc_conf,
            "severity":           severity,
            "severity_heuristic": sev_heuristic,
        }


def _empty_stats() -> dict:
    return {
        "counts":             {},
        "accident":           None,
        "accident_conf":      0.0,
        "severity":           None,
        "severity_heuristic": False,
    }
