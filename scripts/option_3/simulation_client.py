import os
import sys
import time
import queue
import random
import threading
import subprocess
from pathlib import Path

try:
    import cv2
    import numpy as np
    import carla
    from ultralytics import YOLO
except ImportError as e:
    print(f"[!] Import error in simulation client: {e}")

try:
    from video_analyzer import _VehicleTracker, draw_accident_box
except ImportError:
    _VehicleTracker, draw_accident_box = None, None

# Class mapping depending on model
FALLBACK_CLASS_MAP = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 0: "Person"}
CUSTOM_CLASS_MAP = {0: "Car", 1: "Truck", 2: "Bus", 3: "Motorcycle", 4: "Accident"}

# Per-class colours matching the main video analyzer
CLASS_COLOR = {
    "Car":        (220, 195,  10),   # cyan
    "Motorcycle": ( 30, 220,  30),   # green
    "Bus":        (  0, 100, 255),   # orange
    "Truck":      (255,  60, 200),   # purple
    "Person":     (  0, 215, 255),   # yellow
    "Accident":   (  0,   0, 255),   # red (BGR)
}

# Vehicles closer than this (metres) are treated as collided even if the
# CARLA collision sensor callback doesn't fire (covers grazing contacts).
_COLLISION_DISTANCE_M = 4.5

# Closing speed for deliberately-spawned crash vehicles. Lower than the
# original 12 m/s — high-speed velocity overrides fighting the physics
# engine's own collision response is what was launching vehicles airborne.
_CRASH_SPEED = 8.0


def draw_accident_arrow(frame, location, label="ACCIDENT"):
    """Draws a downward-pointing arrow + tag above the given (x, y) point."""
    import cv2 as _cv2
    h, w = frame.shape[:2]
    cx, cy = location
    tip_y  = max(20, cy - 16)
    tail_y = max(8, tip_y - 32)
    cx     = max(15, min(cx, w - 15))

    color = (0, 0, 255)
    _cv2.arrowedLine(frame, (cx, tail_y), (cx, tip_y), color, 2, tipLength=0.45)

    (tw, th), _ = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    lx = max(2, min(cx - tw // 2, w - tw - 4))
    ly = max(th + 2, tail_y - 4)
    _cv2.rectangle(frame, (lx - 3, ly - th - 4), (lx + tw + 3, ly + 3), color, -1)
    _cv2.putText(
        frame, label, (lx, ly),
        _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _cv2.LINE_AA,
    )

class CARLASimulationClient:
    def __init__(self, carla_path, model_path, on_frame_callback, log_callback=None):
        self.carla_path = carla_path
        self.model_path = Path(model_path)
        self.on_frame = on_frame_callback
        self.log_callback = log_callback or (lambda msg: print(msg))
        
        self.client = None
        self.world = None
        self.cameras = []
        self.spawned_actors = []
        self.carla_proc = None
        
        self.running = False
        self.thread = None
        self.image_queues = [queue.Queue(maxsize=2) for _ in range(4)]
        
        self.is_fallback = True
        self.detector_normal = None
        self.detector_accident = None
        
        self.accident_requested = False
        self.accident_active = False
        self.accident_actors = []
        self.accident_pairs = []      # list of dicts tracking each spawned crash pair
        self.junction_collided = []   # ground-truth per-junction collision flags
        self.cctv_transforms = []

        # Vision-based accident detection — always active regardless of the
        # toggle. The toggle only controls whether deliberate crash-pair
        # vehicles get spawned; detection itself must work on ANY vehicle
        # (background traffic included) at any time.
        self.junction_trackers = []        # one _VehicleTracker per camera
        self.junction_persist  = []        # persisted {box,kind,conf,ttl} per camera
        self._junction_was_accident = []   # previous-frame state, for counting
        self.session_accident_count = 0
        
    def log(self, message):
        self.log_callback(message)

    def toggle_accident(self, enabled):
        self.accident_requested = enabled
        self.log(f"[CONTROL] Accident simulation request set to: {enabled}")

    def load_model(self):
        """Loads pretrained YOLOv8n detector for robust vehicle tracking, and custom accident model if exists."""
        self.log("Loading robust YOLOv8n detector for standard vehicle tracking...")
        try:
            self.detector_normal = YOLO("yolov8n.pt")
            self.is_fallback = True
            self.log("[SYSTEM] Robust standard vehicle detector loaded successfully.")
        except Exception as e:
            self.log(f"[ERROR] Error loading YOLOv8n model: {e}")
            
        self.detector_accident = None
        # Load dedicated accident model if it exists (from train2.py)
        try:
            from resource_path import resource_path
            accident_model_path = Path(resource_path("assets", "training", "simulation_dataset", "simulation_accident_detector.pt"))
        except ImportError:
            accident_model_path = Path("assets/training/simulation_dataset/simulation_accident_detector.pt")
        if accident_model_path.exists():
            self.log(f"Loading custom accident detector from {accident_model_path}...")
            try:
                self.detector_accident = YOLO(str(accident_model_path))
                self.log("[SYSTEM] Custom accident detector loaded successfully.")
            except Exception as e:
                self.log(f"[ERROR] Error loading custom accident detector: {e}")
                
    def check_carla_running(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', 2000))
            s.close()
            return True
        except Exception:
            return False

    def launch_carla(self):
        if self.check_carla_running():
            self.log("CARLA simulator is already running.")
            return True
            
        if not self.carla_path or not os.path.isdir(self.carla_path):
            self.log("[ERROR] CARLA path is invalid. Configure it in Settings.")
            return False
            
        exe_path = Path(self.carla_path) / "CarlaUE4.exe"
        if not exe_path.exists():
            self.log(f"[ERROR] CarlaUE4.exe not found at {exe_path}")
            return False
            
        self.log("Launching CARLA in windowed, low-spec mode...")
        try:
            self.carla_proc = subprocess.Popen([
                str(exe_path),
                "-windowed",
                "-ResX=800",
                "-ResY=600",
                "-quality-level=Low"
            ])
            self.log("Waiting for simulator engine to start up (22s)...")
            time.sleep(22.0)
            return True
        except Exception as e:
            self.log(f"[ERROR] Subprocess launch error: {e}")
            return False

    def get_cctv_transforms(self):
        transforms = []
        traffic_lights = list(self.world.get_actors().filter('traffic.traffic_light'))
        
        # Filter to get lights that are spaced apart
        selected_lights = []
        for tl in traffic_lights:
            if len(selected_lights) >= 4:
                break
            if all(tl.get_location().distance(s.get_location()) > 25.0 for s in selected_lights):
                selected_lights.append(tl)
                
        # Fallback to any traffic lights if not enough spaced ones
        if len(selected_lights) < 4:
            for tl in traffic_lights:
                if len(selected_lights) >= 4:
                    break
                if tl not in selected_lights:
                    selected_lights.append(tl)
                    
        # Generate camera transforms for selected traffic lights
        for tl in selected_lights:
            loc = tl.get_location()
            waypoints = tl.get_affected_lane_waypoints()
            if waypoints:
                wp = waypoints[0]
                wp_loc = wp.transform.location
                wp_rot = wp.transform.rotation
                forward = wp_rot.get_forward_vector()
                # Position camera 12 meters back and 8 meters high along the forward lane vector
                cam_loc = wp_loc - forward * 12.0 + carla.Location(z=8.0)
                # Face directly down the road lane (aligned with the waypoint yaw)
                cam_rot = carla.Rotation(pitch=-35, yaw=wp_rot.yaw, roll=0)
                transforms.append(carla.Transform(cam_loc, cam_rot))
            else:
                rot = tl.get_transform().rotation
                forward = rot.get_forward_vector()
                # Position camera 8 meters up and 2.5 meters back, looking down
                cam_loc = loc + carla.Location(z=8.0) - forward * 2.5
                cam_rot = carla.Rotation(pitch=-35, yaw=rot.yaw, roll=0)
                transforms.append(carla.Transform(cam_loc, cam_rot))
            
        # If still less than 4, fallback to random spawn points
        if len(transforms) < 4:
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            for i in range(4 - len(transforms)):
                sp = spawn_points[i]
                cam_loc = sp.location + carla.Location(z=8.0)
                cam_rot = carla.Rotation(pitch=-35, yaw=sp.rotation.yaw, roll=0)
                transforms.append(carla.Transform(cam_loc, cam_rot))
                
        return transforms[:4]

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
            
    def spawn_additional_traffic(self, count=15):
        """Spawns more dynamic vehicles on autopilot."""
        if not self.world or not self.client:
            self.log("Cannot spawn traffic: client not connected.")
            return
            
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")
        
        traffic_manager = self.client.get_trafficmanager(8000)
        
        spawned = 0
        for sp in spawn_points:
            if spawned >= count:
                break
            bp = random.choice(vehicle_bps)
            actor = self.world.try_spawn_actor(bp, sp)
            if actor:
                actor.set_autopilot(True, traffic_manager.get_port())
                self.spawned_actors.append(actor)
                spawned += 1
                
        self.log(f"Spawned {spawned} additional vehicles.")

    def _put_in_queue(self, idx, img):
        q = self.image_queues[idx]
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(img)

    def _run_loop(self):
        try:
            # Load model weights
            self.load_model()
            
            # Start CARLA
            if not self.launch_carla():
                self.log("[ERROR] CARLA launch aborted.")
                return
                
            self.log("Connecting to CARLA server on port 2000...")
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(45.0)
            
            self.world = self.client.get_world()
            self.log("World loaded successfully.")
            
            # Force asynchronous mode to ensure traffic manager ticks automatically
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                self.log("[SYSTEM] Reverting world settings to asynchronous mode...")
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            
            # Setup traffic manager
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(False)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            
            blueprint_library = self.world.get_blueprint_library()
            
            # Spawn initial traffic
            self.log("Spawning initial traffic vehicles...")
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            vehicle_bps = blueprint_library.filter("vehicle.*")
            initial_spawn = min(120, len(spawn_points) - 5)
            
            for i in range(initial_spawn):
                bp = random.choice(vehicle_bps)
                v = self.world.try_spawn_actor(bp, spawn_points[i])
                if v:
                    v.set_autopilot(True, traffic_manager.get_port())
                    self.spawned_actors.append(v)
                    
            self.log(f"Spawned {len(self.spawned_actors)} background vehicles.")
            
            # Spawn 4 fixed CCTV cameras at intersections
            self.log("Spawning 4 fixed CCTV cameras at junctions...")
            self.cctv_transforms = self.get_cctv_transforms()
            
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            # 400x225 resolution per feed makes a combined 800x450 grid image
            camera_bp.set_attribute('image_size_x', '400')
            camera_bp.set_attribute('image_size_y', '225')
            # Wider FOV (was 90) so the 12m-out crash spawn points stay
            # inside frame for their whole approach, not just near impact
            camera_bp.set_attribute('fov', '110')
            
            def make_callback(index):
                return lambda img: self._put_in_queue(index, img)
                
            for idx, trans in enumerate(self.cctv_transforms):
                cam = self.world.spawn_actor(camera_bp, trans)
                cam.listen(make_callback(idx))
                self.cameras.append(cam)
                self.log(f"  -> CAM {idx+1} positioned at traffic light junction.")

            # One vision tracker per camera — runs unconditionally every
            # frame, independent of the accident toggle
            n_cams = len(self.cctv_transforms)
            self.junction_trackers = [_VehicleTracker() for _ in range(n_cams)] if _VehicleTracker else []
            self.junction_persist  = [{"box": None, "kind": "vehicle", "conf": 0.0, "ttl": 0} for _ in range(n_cams)]
            self._junction_was_accident = [False] * n_cams

            self.log("[SYSTEM] CARLA 2x2 CCTV Grid active. Monitoring online.")
            
            # Choose class mapping dynamically inside loop
            class_map = FALLBACK_CLASS_MAP
            valid_classes = list(class_map.keys())
            
            while self.running:
                # Handle accident simulation toggle safely on client thread
                if self.accident_requested != self.accident_active:
                    if self.accident_requested:
                        self.log("[SYSTEM] Spawning accidents at all junctions...")
                        collision_bp = blueprint_library.find('sensor.other.collision')
                        self.junction_collided = [False] * len(self.cctv_transforms)

                        for idx, trans in enumerate(self.cctv_transforms):
                            forward = trans.rotation.get_forward_vector()
                            right = trans.rotation.get_right_vector()

                            # Project camera forward vector to horizontal plane and normalize
                            forward_h = carla.Vector3D(forward.x, forward.y, 0.0)
                            f_norm = np.sqrt(forward_h.x**2 + forward_h.y**2)
                            if f_norm > 0:
                                forward_h = carla.Vector3D(forward_h.x / f_norm, forward_h.y / f_norm, 0.0)

                            # Center of the junction on the road — kept close to
                            # the camera (10m, was 13m) so the collision lands
                            # well within frame instead of near the FOV edge
                            junction_center = trans.location + forward_h * 10.0
                            junction_center.z = trans.location.z - 8.0  # Camera height is 8 meters

                            v_bps_1 = blueprint_library.filter("vehicle.audi.etron")
                            v_bps_2 = blueprint_library.filter("vehicle.tesla.model3")
                            bp1 = v_bps_1[0] if v_bps_1 else blueprint_library.filter("vehicle.*")[0]
                            bp2 = v_bps_2[0] if v_bps_2 else blueprint_library.filter("vehicle.*")[0]

                            # Spawn Vehicle 1 on the cross street (left/right) —
                            # 12m out (was 20m) keeps it inside the camera's
                            # horizontal field of view at this depth
                            side_multiplier = 1.0 if idx % 2 == 0 else -1.0
                            t1 = carla.Transform(
                                junction_center + right * (12.0 * side_multiplier) + carla.Location(z=0.5),
                                carla.Rotation(yaw=trans.rotation.yaw + (90 if side_multiplier < 0 else -90))
                            )

                            # Spawn Vehicle 2 approaching from further down the
                            # same lane (12m, was 20m) — clearly visible the
                            # whole approach instead of starting near the edge
                            t2 = carla.Transform(
                                junction_center - forward_h * 12.0 + carla.Location(z=0.5),
                                carla.Rotation(yaw=trans.rotation.yaw)
                            )

                            veh1 = self.world.try_spawn_actor(bp1, t1)
                            veh2 = self.world.try_spawn_actor(bp2, t2)

                            if veh1 and veh2:
                                veh1.set_autopilot(False)
                                veh2.set_autopilot(False)

                                pair = {
                                    "idx": idx,
                                    "veh1": veh1, "veh2": veh2,
                                    "collided": False,
                                    "velocity_killed": False,
                                }

                                # Collision sensors give ground-truth proof that an
                                # actual physical crash happened (not just a 2D camera
                                # bounding-box overlap caused by perspective)
                                def make_collision_cb(p):
                                    def _cb(event):
                                        p["collided"] = True
                                    return _cb

                                sensor1 = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=veh1)
                                sensor2 = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=veh2)
                                sensor1.listen(make_collision_cb(pair))
                                sensor2.listen(make_collision_cb(pair))

                                self.accident_pairs.append(pair)
                                self.accident_actors.extend([veh1, veh2, sensor1, sensor2])
                            elif veh1:
                                veh1.destroy()
                            elif veh2:
                                veh2.destroy()

                        self.accident_active = True
                        self.log("[SYSTEM] Accidents spawned — vehicles driving toward collision course.")
                    else:
                        self.log("[SYSTEM] Clearing accident vehicles...")
                        for actor in self.accident_actors:
                            try:
                                if actor.is_alive:
                                    actor.destroy()
                            except Exception as e:
                                self.log(f"Error destroying actor: {e}")
                        self.accident_actors = []
                        self.accident_pairs = []
                        self.junction_collided = []
                        self.accident_active = False
                        self.log("[SYSTEM] Accident vehicles cleared.")

                # Keep driving accident vehicles toward each other every tick.
                # Pursuit steering: recompute direction toward the OTHER
                # vehicle's CURRENT live position every tick (not a fixed
                # pre-computed heading) — this guarantees the two vehicles
                # converge and make contact regardless of imperfect spawn
                # geometry at any given junction, which is what was causing
                # collisions to only reliably land at one of the four.
                if self.accident_active:
                    for pair in self.accident_pairs:
                        v1, v2 = pair["veh1"], pair["veh2"]

                        # Once collided (whether confirmed by the async sensor
                        # callback or our own distance check below), kill
                        # velocity exactly once. Continuing to force speed
                        # into an already-overlapping pair is what launches
                        # vehicles airborne (our override fighting the
                        # physics engine's own collision response).
                        if pair["collided"]:
                            if not pair["velocity_killed"]:
                                try:
                                    if v1.is_alive:
                                        v1.set_target_velocity(carla.Vector3D(0, 0, 0))
                                    if v2.is_alive:
                                        v2.set_target_velocity(carla.Vector3D(0, 0, 0))
                                except Exception:
                                    pass
                                pair["velocity_killed"] = True
                            continue

                        try:
                            if v1.is_alive and v2.is_alive:
                                loc1, loc2 = v1.get_location(), v2.get_location()
                                dist = loc1.distance(loc2)

                                dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
                                n = np.hypot(dx, dy) or 1.0
                                v1.set_target_velocity(carla.Vector3D(dx/n*_CRASH_SPEED, dy/n*_CRASH_SPEED, 0))
                                v2.set_target_velocity(carla.Vector3D(-dx/n*_CRASH_SPEED, -dy/n*_CRASH_SPEED, 0))

                                # Distance fallback: confirms collision even if the
                                # physics engine doesn't fire a contact callback
                                # (e.g. clipped corners / minor bounding overlaps)
                                if dist < _COLLISION_DISTANCE_M:
                                    pair["collided"] = True
                            else:
                                pair["collided"] = True
                        except Exception:
                            pair["collided"] = True

                        if pair["collided"]:
                            self.junction_collided[pair["idx"]] = True
                            self.log(f"[SYSTEM] Collision confirmed at junction {chr(65 + pair['idx'])}.")

                # Wait until all 4 camera queues have at least one frame to avoid starvation
                all_ready = True
                for q in self.image_queues:
                    if q.empty():
                        all_ready = False
                        break
                
                if not all_ready:
                    if self.carla_proc and self.carla_proc.poll() is not None:
                        self.log("[ERROR] CARLA simulator terminated unexpectedly.")
                        break
                    time.sleep(0.01)
                    continue
                
                # Retrieve frames from all 4 camera queues
                imgs = []
                for q in self.image_queues:
                    try:
                        imgs.append(q.get_nowait())
                    except queue.Empty:
                        pass
                        
                if len(imgs) < 4:
                    continue
                
                quadrants = []
                junction_stats = []
                
                # Pre-convert frames to cv2 BGR images
                quadrants_raw = []
                for img in imgs:
                    array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    quadrants_raw.append(array[:, :, :3].copy())
                
                # 1. Run standard vehicle detection (always using self.detector_normal = yolov8n.pt)
                # We query COCO classes: 2 (Car), 3 (Motorcycle), 5 (Bus), 7 (Truck), 0 (Person)
                # We use conf=0.45 to filter out the false positive garbage cans
                valid_classes_normal = [0, 2, 3, 5, 7]
                results_normal = self.detector_normal(
                    quadrants_raw,
                    verbose=False,
                    conf=0.45,
                    classes=valid_classes_normal
                )

                # 2. Dedicated accident detector (trained via train2.py) — runs
                # ALWAYS, independent of the toggle, same as the vision
                # tracker. Its training set had no negative (non-accident)
                # examples, so it's treated as a corroborating signal for
                # precise box localisation, not trusted to decide on its own.
                results_accident = None
                if self.detector_accident is not None:
                    results_accident = self.detector_accident(
                        quadrants_raw, verbose=False, conf=0.5
                    )

                for idx, r in enumerate(results_normal):
                    bgr_quad = quadrants_raw[idx]
                    h_q, w_q = bgr_quad.shape[:2]

                    counts = {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0, "accident": False}
                    detected_boxes = []

                    # Parse standard vehicle detections
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        label = FALLBACK_CLASS_MAP.get(cls_id, "Car")
                        counts[label] += 1
                        color = CLASS_COLOR.get(label, (0, 255, 0))

                        # Draw thin bounding box
                        cv2.rectangle(bgr_quad, (x1, y1), (x2, y2), color, 1)

                        # Draw thin cyber corner tick marks
                        tk = 6
                        for (sx, sy), (dx, dy) in [
                            ((x1, y1), ( 1,  1)), ((x2, y1), (-1,  1)),
                            ((x1, y2), ( 1, -1)), ((x2, y2), (-1, -1)),
                        ]:
                            cv2.line(bgr_quad, (sx, sy), (sx + dx * tk, sy), color, 1)
                            cv2.line(bgr_quad, (sx, sy), (sx, sy + dy * tk), color, 1)

                        # Draw thin label tag (small font size)
                        tag = f"{label} {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(bgr_quad, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                        cv2.putText(bgr_quad, tag, (x1 + 1, y1 - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 10, 30), 1, cv2.LINE_AA)

                        # Save vehicle box for IoU check
                        if label in ["Car", "Truck", "Bus", "Motorcycle"]:
                            detected_boxes.append((x1, y1, x2, y2))

                    # ── Accident detection — gated behind the toggle. With
                    # 120+ background vehicles on autopilot, CARLA's traffic
                    # manager occasionally produces real incidental contact
                    # between ordinary traffic; flagging those as "accidents"
                    # whenever the vision tracker happened to be watching is
                    # not what's wanted. Accidents should only ever be
                    # reported while the toggle is on. Three signals feed
                    # the decision, in priority order, once toggle-gated:
                    #
                    # 1. Vision-based abrupt-stop tracker (same logic as
                    #    Option 2) — a vehicle that was moving and came to a
                    #    near-total, sustained stop while overlapping another
                    #    vehicle (or isolated, hitting an unseen object).
                    # 2. Dedicated accident detector (train2.py) — precise
                    #    pixel-projected boxes, but its training set had no
                    #    negative examples, so it's gated behind a basic
                    #    sanity check (2+ real vehicles actually nearby).
                    # 3. Ground-truth physics collision sensor — only
                    #    meaningful for the toggle-spawned pairs, used as a
                    #    last-resort fallback localisation.
                    p = self.junction_persist[idx] if idx < len(self.junction_persist) else None
                    vision_hits = []
                    if p is not None and self.junction_trackers:
                        self.junction_trackers[idx].update(detected_boxes)
                        vision_hits = self.junction_trackers[idx].detect_collisions(w_q, h_q)

                    dedicated_box, dedicated_conf = None, 0.0
                    if results_accident is not None:
                        r_acc = results_accident[idx]
                        for box in r_acc.boxes:
                            conf = float(box.conf[0])
                            if conf > dedicated_conf:
                                dedicated_conf = conf
                                dedicated_box = tuple(map(int, box.xyxy[0]))

                    ground_truth = (
                        self.accident_active
                        and idx < len(self.junction_collided)
                        and self.junction_collided[idx]
                    )

                    if not self.accident_active:
                        # Toggle is off — accidents only happen when toggled
                        # on. Clear any persisted box immediately rather than
                        # letting it decay, so nothing lingers after toggling off.
                        if p is not None:
                            p["box"], p["ttl"] = None, 0
                        accident_detected = False
                    elif p is not None:
                        if vision_hits:
                            best = max(vision_hits, key=lambda c: c["conf"])
                            box, kind, conf = best["box"], best["kind"], best["conf"]
                            if dedicated_box is not None:
                                box = dedicated_box  # more precisely localised
                                conf = min(conf + 0.1, 0.97)
                            p["box"], p["kind"], p["conf"], p["ttl"] = box, kind, conf, 20
                        elif dedicated_box is not None and len(detected_boxes) >= 2 and p["ttl"] <= 0:
                            p["box"], p["kind"], p["conf"], p["ttl"] = dedicated_box, "vehicle", dedicated_conf * 0.85, 20
                        elif ground_truth and p["ttl"] <= 0:
                            # Sensor fired but neither vision signal localised
                            # it yet — best-guess box is the largest vehicle in view
                            if detected_boxes:
                                box = max(detected_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                            else:
                                box = (w_q//2 - 40, h_q//2 - 30, w_q//2 + 40, h_q//2 + 30)
                            p["box"], p["kind"], p["conf"], p["ttl"] = box, "vehicle", 0.8, 20
                        elif p["ttl"] > 0:
                            p["ttl"] -= 1

                        accident_detected = p["ttl"] > 0
                    else:
                        accident_detected = ground_truth

                    counts["accident"] = accident_detected

                    # Count each NEW accident (false->true transition) once
                    if idx < len(self._junction_was_accident):
                        if accident_detected and not self._junction_was_accident[idx]:
                            self.session_accident_count += 1
                            self.log(f"[ALERT] Accident detected at junction {chr(65+idx)} "
                                     f"(session total: {self.session_accident_count})")
                        self._junction_was_accident[idx] = accident_detected

                    # Red box + ACCIDENT label on the actual vehicle, plus a
                    # flashing border so it's unmissable in the grid view
                    if accident_detected and p is not None and p["box"] is not None:
                        if draw_accident_box is not None:
                            draw_accident_box(bgr_quad, p["box"], p["kind"])
                        else:
                            draw_accident_arrow(bgr_quad, ((p["box"][0]+p["box"][2])//2, (p["box"][1]+p["box"][3])//2))
                        if int(time.time() * 3.3) % 2 == 0:
                            cv2.rectangle(bgr_quad, (0, 0), (w_q - 1, h_q - 1), (0, 0, 255), 3)
                            cv2.putText(bgr_quad, "[!!] ACCIDENT DETECTED [!!]", (12, h_q - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                    # Overlay CCTV layout labels
                    cctv_name = f"CAM {idx+1} // JUNCTION {chr(65+idx)}"
                    cv2.putText(bgr_quad, cctv_name, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 243, 255), 1, cv2.LINE_AA)
                    
                    # Quadrant frame borders
                    cv2.rectangle(bgr_quad, (0, 0), (w_q-1, h_q-1), (30, 20, 40), 1)
                    
                    quadrants.append(bgr_quad)
                    junction_stats.append(counts)
                
                # Stitch 4 feeds into 2x2 grid image (Combined width=800, height=450)
                row1 = np.hstack([quadrants[0], quadrants[1]])
                row2 = np.hstack([quadrants[2], quadrants[3]])
                grid_img = np.vstack([row1, row2])
                
                # Draw vertical and horizontal grid dividing lines
                cv2.line(grid_img, (400, 0), (400, 450), (0, 243, 255), 1)
                cv2.line(grid_img, (0, 225), (800, 225), (0, 243, 255), 1)
                
                # Compute global aggregate counts
                global_counts = {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0}
                for j_stat in junction_stats:
                    for k, v in j_stat.items():
                        if k in global_counts:
                            global_counts[k] += v
                        
                stats = {
                    "counts": global_counts,
                    "junctions": junction_stats,
                    "fallback": (self.detector_accident is None),
                    "accident_count": self.session_accident_count,
                }
                
                # Callback to main UI
                self.on_frame(grid_img, stats)
                
        except Exception as e:
            self.log(f"[CRITICAL] Client Loop Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.log("Initiating shutdown procedures...")
            
            # Stop cameras
            for cam in self.cameras:
                try:
                    cam.destroy()
                except Exception:
                    pass
            self.cameras = []
            
            # Despawn accident vehicles + their collision sensors
            if self.accident_actors:
                self.log(f"Despawning {len(self.accident_actors)} accident actors...")
                for actor in self.accident_actors:
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except Exception:
                        pass
                self.accident_actors = []
                self.accident_pairs = []
                self.junction_collided = []
                    
            # Despawn vehicles
            if self.world and self.spawned_actors:
                self.log(f"Despawning {len(self.spawned_actors)} actors...")
                try:
                    self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.spawned_actors])
                except Exception as e:
                    self.log(f"Despawn error: {e}")
            self.spawned_actors = []
            
            # Terminate simulator process if we launched it
            if self.carla_proc:
                self.log("Terminating CARLA simulator process...")
                try:
                    self.carla_proc.terminate()
                    self.carla_proc.wait(timeout=2.0)
                except Exception:
                    pass
                self.carla_proc = None
            
            # Force kill any running CARLA instance on Windows to guarantee it closes completely
            try:
                import platform
                if platform.system() == "Windows":
                    self.log("[SYSTEM] Force-killing CARLA simulator processes on exit...")
                    subprocess.run(["taskkill", "/f", "/im", "CarlaUE4.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["taskkill", "/f", "/im", "CarlaUE4-Win64-Shipping.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
                    
            self.log("[SYSTEM] CARLA client offline.")
