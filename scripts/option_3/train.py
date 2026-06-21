#!/usr/bin/env python3
"""
Traffic Monitoring System - CARLA Model Training Script (Option 3)

1. Connects to CARLA simulation (launches CarlaUE4.exe if configured & offline).
2. Spawns dynamic vehicles and a hero vehicle equipped with a camera.
3. Automatically computes 2D bounding boxes using 3D coordinate projection.
4. Generates a labeled YOLOv8 dataset in assets/training/simulation_dataset.
5. Trains a YOLOv8 detection model and saves it as simulation_detector.pt.

Run with: venv/Scripts/python.exe scripts/option_3/train.py
"""

import os
import sys
import json
import time
import random
import shutil
import subprocess
from pathlib import Path
import queue

try:
    import numpy as np
    import cv2
    import carla
    from ultralytics import YOLO
except ImportError as e:
    print(f"[!] Import error: {e}")
    print("[!] Please ensure the virtual environment is active and all packages are installed.")
    sys.exit(1)

# ── Resolve project roots ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "assets/training/simulation_dataset"
CONFIG_FILE = ROOT / "config.json"

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 90

# ── Bounding Box Projection Helper ─────────────────────────────────────────────
class SimulationBBoxProjector:
    @staticmethod
    def get_matrix(transform):
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        
        matrix = np.identity(4)
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """Projects 3D bounding box to 2D camera coordinates."""
        # 3D bounding box corners relative to vehicle center
        extent = vehicle.bounding_box.extent
        cords = np.zeros((8, 4))
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

        # Transform points: vehicle -> world -> sensor
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = SimulationBBoxProjector.get_matrix(bb_transform)
        vehicle_world_matrix = SimulationBBoxProjector.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))

        sensor_world_matrix = SimulationBBoxProjector.get_matrix(camera.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        cords_x_y_z = np.dot(world_sensor_matrix, world_cords)[:3, :]

        # Check if points are behind sensor
        if all(cords_x_y_z[0, :] <= 0):
            return None

        # Convert coordinates to camera coordinate system
        cords_y_minus_z_x = np.vstack([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        
        # Project using camera calibration matrix
        calib = np.asarray(camera.calibration)
        cords_y_minus_z_x = np.asarray(cords_y_minus_z_x)
        bbox = np.transpose(np.dot(calib, cords_y_minus_z_x))
        bbox = np.asarray(bbox)
        
        x_proj = bbox[:, 0] / bbox[:, 2]
        y_proj = bbox[:, 1] / bbox[:, 2]
        depth = bbox[:, 2]
        
        camera_bbox = np.stack([x_proj, y_proj, depth], axis=1)
        return camera_bbox

# ── Load config helper ────────────────────────────────────────────────────────
def load_carla_path():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                return cfg.get("carla_path", "")
        except Exception:
            pass
    return ""

def launch_carla_simulator(carla_dir):
    """Launches CarlaUE4.exe in windowed low-performance mode."""
    if not carla_dir:
        return None
    exe_path = Path(carla_dir) / "CarlaUE4.exe"
    if not exe_path.exists():
        print(f"[!] CarlaUE4.exe not found at: {exe_path}")
        return None
    
    print("[*] Spawning CARLA simulator process (windowed, low-spec mode)...")
    try:
        proc = subprocess.Popen([
            str(exe_path),
            "-windowed",
            "-ResX=800",
            "-ResY=600",
            "-quality-level=Low"
        ])
        return proc
    except Exception as e:
        print(f"[!] Error launching CARLA: {e}")
        return None

# ── CCTV Camera Positioning helper ─────────────────────────────────────────────
def get_cctv_transforms(world):
    transforms = []
    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    
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
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for i in range(4 - len(transforms)):
            sp = spawn_points[i]
            cam_loc = sp.location + carla.Location(z=8.0)
            cam_rot = carla.Rotation(pitch=-35, yaw=sp.rotation.yaw, roll=0)
            transforms.append(carla.Transform(cam_loc, cam_rot))
            
    return transforms[:4]

# ── Dataset capture & label script ─────────────────────────────────────────────
def collect_dataset(client, num_frames=150):
    print("\n" + "=" * 58)
    print("  STAGE 1 — CARLA Dataset Collection (4 Fixed CCTV Cameras)")
    print(f"  Target  : {num_frames} frames")
    print("=" * 58)
    
    world = client.get_world()
    
    # Enable synchronous mode for exact frame-sensor alignment
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20 FPS
    world.apply_settings(settings)
    
    # Get blueprints
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter("vehicle.*")
    
    # Setup directories
    images_train = DATASET_DIR / "images/train"
    images_val = DATASET_DIR / "images/val"
    labels_train = DATASET_DIR / "labels/train"
    labels_val = DATASET_DIR / "labels/val"
    
    # Re-create clean dataset folders
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)
        
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    # Spawn Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    
    # Spawn dynamic traffic
    print("[*] Spawning traffic vehicles...")
    spawned_vehicles = []
    num_to_spawn = min(50, len(spawn_points) - 5)
    for i in range(num_to_spawn):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        vehicle = world.try_spawn_actor(bp, spawn_points[i])
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            spawned_vehicles.append(vehicle)
            
    print(f"[+] Successfully spawned {len(spawned_vehicles)} vehicles.")
    
    # Position and spawn 4 fixed CCTV cameras at intersections
    print("[*] Position and spawn 4 fixed CCTV cameras at intersections...")
    cctv_transforms = get_cctv_transforms(world)
    
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
    camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
    camera_bp.set_attribute('fov', str(VIEW_FOV))
    
    cameras = []
    image_queues = []
    
    def make_queue_listener(q):
        return lambda img: q.put(img)
        
    for idx, trans in enumerate(cctv_transforms):
        cam = world.spawn_actor(camera_bp, trans)
        
        # Set calibration matrix
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        cam.calibration = calibration
        
        q = queue.Queue()
        cam.listen(make_queue_listener(q))
        
        cameras.append(cam)
        image_queues.append(q)
        print(f"  -> Spawned CCTV Camera {idx+1} at location: {trans.location}")
    
    # Map vehicle type_id to class indices
    # 0: Car, 1: Truck, 2: Bus, 3: Motorcycle, 4: Accident
    def get_class_id(v):
        tid = v.type_id.lower()
        if 'motorcycle' in tid or 'bike' in tid:
            return 3
        elif 'truck' in tid or 'firetruck' in tid or 'ambulance' in tid:
            return 1
        elif 'bus' in tid:
            return 2
        return 0

    # Helper function to spawn crash at a junction transform
    def spawn_accident_vehicles(world, cam_transform, blueprint_library):
        forward = cam_transform.rotation.get_forward_vector()
        junction_center = cam_transform.location + forward * 13.0
        
        # Spawn vehicle 1 sideways
        v_bps = blueprint_library.filter("vehicle.audi.etron")
        bp1 = v_bps[0] if v_bps else blueprint_library.filter("vehicle.*")[0]
        t1 = carla.Transform(junction_center + carla.Location(z=0.5), 
                             carla.Rotation(yaw=cam_transform.rotation.yaw + 90))
        
        # Spawn vehicle 2 facing vehicle 1 (collision path)
        v_bps = blueprint_library.filter("vehicle.tesla.model3")
        bp2 = v_bps[0] if v_bps else blueprint_library.filter("vehicle.*")[0]
        t2 = carla.Transform(junction_center + forward * -2.5 + carla.Location(z=0.5), 
                             carla.Rotation(yaw=cam_transform.rotation.yaw))
        
        veh1 = world.try_spawn_actor(bp1, t1)
        veh2 = world.try_spawn_actor(bp2, t2)
        
        actors = []
        if veh1:
            actors.append(veh1)
        if veh2:
            actors.append(veh2)
        return actors

    print("[*] Commencing frame and bounding box projection capture...")
    
    captured_count = 0
    step_interval = 12 # Capture every N ticks to ensure dataset variety
    tick_count = 0
    
    accident_actors = []
    accident_triggered = False
    
    try:
        while captured_count < num_frames:
            world.tick()
            
            # Retrieve images from all cameras
            imgs = []
            for q in image_queues:
                imgs.append(q.get())
                
            tick_count += 1
            if tick_count % step_interval != 0:
                continue
                
            # Simulate accidents in the town to capture crash dataset after 40 frames
            if captured_count >= 40 and not accident_triggered:
                print("\n[*] Simulating accidents at junctions for training...")
                actors1 = spawn_accident_vehicles(world, cctv_transforms[0], blueprint_library)
                actors2 = spawn_accident_vehicles(world, cctv_transforms[1], blueprint_library)
                accident_actors.extend(actors1 + actors2)
                spawned_vehicles.extend(actors1 + actors2)
                accident_triggered = True
                
            # Process each camera frame
            for cam_idx, img in enumerate(imgs):
                if captured_count >= num_frames:
                    break
                    
                # Process image raw bytes
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                bgr_frame = array[:, :, :3]
                
                # Find boxes for all vehicles
                bboxes_lines = []
                all_vehicles = world.get_actors().filter('vehicle.*')
                cam = cameras[cam_idx]
                
                for v in all_vehicles:
                    bbox_cam = SimulationBBoxProjector.get_bounding_box(v, cam)
                    if bbox_cam is not None:
                        # Filter points behind camera or depth <= 0
                        x_coords = bbox_cam[:, 0]
                        y_coords = bbox_cam[:, 1]
                        z_coords = bbox_cam[:, 2]
                        
                        if np.any(z_coords <= 0):
                            continue
                            
                        xmin = np.min(x_coords)
                        xmax = np.max(x_coords)
                        ymin = np.min(y_coords)
                        ymax = np.max(y_coords)
                        
                        # Compute vehicle distance
                        v_loc = v.get_location()
                        cam_loc = cam.get_location()
                        dist = v_loc.distance(cam_loc)
                        
                        # Filter: ignore vehicles out of screen bounds or too far
                        if xmax < 0 or xmin > VIEW_WIDTH or ymax < 0 or ymin > VIEW_HEIGHT:
                            continue
                        if dist > 85.0: # Keep within 85 meters
                            continue
                            
                        # Clip boundaries
                        xmin_clip = max(0.0, min(xmin, VIEW_WIDTH))
                        xmax_clip = max(0.0, min(xmax, VIEW_WIDTH))
                        ymin_clip = max(0.0, min(ymin, VIEW_HEIGHT))
                        ymax_clip = max(0.0, min(ymax, VIEW_HEIGHT))
                        
                        w_box = xmax_clip - xmin_clip
                        h_box = ymax_clip - ymin_clip
                        
                        # Ignore tiny boxes
                        if w_box < 8 or h_box < 8:
                            continue
                            
                        # Convert to YOLO format (normalized center x, center y, w, h)
                        x_center = (xmin_clip + xmax_clip) / 2.0 / VIEW_WIDTH
                        y_center = (ymin_clip + ymax_clip) / 2.0 / VIEW_HEIGHT
                        norm_w = w_box / VIEW_WIDTH
                        norm_h = h_box / VIEW_HEIGHT
                        
                        class_id = get_class_id(v)
                        bboxes_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                # Check for crash boxes to label as class 4 (Accident)
                if accident_triggered:
                    cam_accident_actors = [a for a in accident_actors if a.is_alive]
                    cam_boxes = []
                    for a in cam_accident_actors:
                        bbox_cam = SimulationBBoxProjector.get_bounding_box(a, cam)
                        if bbox_cam is not None:
                            x_coords = bbox_cam[:, 0]
                            y_coords = bbox_cam[:, 1]
                            z_coords = bbox_cam[:, 2]
                            if np.any(z_coords <= 0):
                                continue
                            xmin = np.min(x_coords)
                            xmax = np.max(x_coords)
                            ymin = np.min(y_coords)
                            ymax = np.max(y_coords)
                            if not (xmax < 0 or xmin > VIEW_WIDTH or ymax < 0 or ymin > VIEW_HEIGHT):
                                xmin_clip = max(0.0, min(xmin, VIEW_WIDTH))
                                xmax_clip = max(0.0, min(xmax, VIEW_WIDTH))
                                ymin_clip = max(0.0, min(ymin, VIEW_HEIGHT))
                                ymax_clip = max(0.0, min(ymax, VIEW_HEIGHT))
                                cam_boxes.append((xmin_clip, xmax_clip, ymin_clip, ymax_clip))
                                
                    if len(cam_boxes) >= 2:
                        # Combine accident boxes into one merged Accident bounding box
                        xmin_acc = min(b[0] for b in cam_boxes)
                        xmax_acc = max(b[1] for b in cam_boxes)
                        ymin_acc = min(b[2] for b in cam_boxes)
                        ymax_acc = max(b[3] for b in cam_boxes)
                        
                        w_box = xmax_acc - xmin_acc
                        h_box = ymax_acc - ymin_acc
                        
                        if w_box >= 12 and h_box >= 12:
                            x_center = (xmin_acc + xmax_acc) / 2.0 / VIEW_WIDTH
                            y_center = (ymin_acc + ymax_acc) / 2.0 / VIEW_HEIGHT
                            norm_w = w_box / VIEW_WIDTH
                            norm_h = h_box / VIEW_HEIGHT
                            bboxes_lines.append(f"4 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                # Only save frame if we detected at least one vehicle
                if len(bboxes_lines) > 0:
                    # 80/20 train/val split
                    is_val = (captured_count % 5 == 0)
                    img_dest = images_val if is_val else images_train
                    lbl_dest = labels_val if is_val else labels_train
                    
                    img_file = img_dest / f"frame_{captured_count:04d}.jpg"
                    lbl_file = lbl_dest / f"frame_{captured_count:04d}.txt"
                    
                    # Save frame
                    cv2.imwrite(str(img_file), bgr_frame)
                    
                    # Save labels
                    with open(lbl_file, 'w') as f:
                        f.write("\n".join(bboxes_lines))
                        
                    captured_count += 1
                    
                    # Draw a simple console progress bar
                    percent = int(100 * captured_count / num_frames)
                    bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
                    sys.stdout.write(f"\rCapture progress: [{bar}] {percent}% ({captured_count}/{num_frames})")
                    sys.stdout.flush()
                
    finally:
        print("\n[*] Restoring CARLA settings and cleaning up actors...")
        world.apply_settings(original_settings)
        for cam in cameras:
            try:
                cam.destroy()
            except Exception:
                pass
        
        # Batch destroy spawned actors
        client.apply_batch([carla.command.DestroyActor(x.id) for x in spawned_vehicles])
        print("[+] CARLA cleanup complete.")

# ── Model Training Script ─────────────────────────────────────────────────────
def train_simulation_model(epochs=3, batch_size=16):
    print("\n" + "=" * 58)
    print("  STAGE 2 — YOLOv8 Model Training")
    print(f"  Model   : yolov8n.pt")
    print(f"  Epochs  : {epochs}")
    print("=" * 58)
    
    # Create dataset.yaml
    yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: Car
  1: Truck
  2: Bus
  3: Motorcycle
  4: Accident
"""
    yaml_file = DATASET_DIR / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
        
    print(f"[*] Written dataset configuration to {yaml_file}")
    
    # Load pretrained YOLOv8n detector model
    model = YOLO("yolov8n.pt")
    
    # Train
    model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=320,
        batch=batch_size,
        workers=2,
        project=str(DATASET_DIR / "runs"),
        name="simulation_run",
        exist_ok=True
    )
    
    # Copy best weights
    best_weights = DATASET_DIR / "runs/simulation_run/weights/best.pt"
    output_model = DATASET_DIR / "simulation_detector.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, output_model)
        print(f"\n[+] SUCCESS: Custom model trained and saved to: {output_model}")
        
        # Clean up runs directory
        runs_dir = DATASET_DIR / "runs"
        if runs_dir.exists():
            try:
                shutil.rmtree(runs_dir)
                print("[*] Cleaned up temporary runs directories.")
            except Exception:
                pass
    else:
        print("[!] Warning: best.pt weights not found. Check training logs.")

# ── Main Entry ────────────────────────────────────────────────────────────────
def main():
    print("=========================================================")
    print("      CARLA SIMULATION VEHICLE TRACKING - TRAINING       ")
    print("=========================================================")
    
    carla_path = load_carla_path()
    carla_proc = None
    
    # Check if CARLA server is already running
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    connected = False
    try:
        s.connect(('127.0.0.1', 2000))
        connected = True
        s.close()
        print("[*] CARLA server already running at 127.0.0.1:2000")
    except Exception:
        pass
        
    if not connected:
        if not carla_path:
            print("[!] CARLA server is not running, and carla_path is not configured.")
            print("[!] Please run CARLA manually or configure its path in Settings.")
            sys.exit(1)
        carla_proc = launch_carla_simulator(carla_path)
        if not carla_proc:
            sys.exit(1)
        print("[*] Waiting for CARLA server to start up (22s)...")
        time.sleep(22.0)
        
    try:
        # Establish CARLA client connection
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(45.0)
        
        # Verify connection by loading the world
        _ = client.get_world()
        print("[+] CARLA Connection established successfully.")
        
        # Step 1: Collect dataset
        collect_dataset(client, num_frames=160)
        
        # Step 2: Train Model
        train_simulation_model(epochs=8, batch_size=16)
        
    except Exception as e:
        print(f"\n[CRITICAL] Error occurred during process: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if carla_proc:
            print("[*] Terminating launched CARLA simulator process...")
            carla_proc.terminate()
            carla_proc.wait()
            print("[+] CARLA simulator process closed.")
            
    print("\n[INFO] End of Script.")

if __name__ == "__main__":
    main()
