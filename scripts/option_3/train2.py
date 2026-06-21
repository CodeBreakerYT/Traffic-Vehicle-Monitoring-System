#!/usr/bin/env python3
"""
Traffic Monitoring System - Dedicated Accident Model Training Script (Option 3 - Train 2)

1. Connects to CARLA simulation.
2. Spawns vehicles and immediately triggers crash scenarios at all junctions.
3. Automatically computes 2D bounding boxes and labels overlapping vehicle pairs as Accident (Class 4).
4. Generates a labeled YOLOv8 dataset in assets/training/simulation_accident_dataset.
5. Trains a YOLOv8 detection model and saves it as simulation_accident_detector.pt.

Run with: venv/Scripts/python.exe scripts/option_3/train2.py
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
    print("[!] Please ensure the virtual environment is active.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "assets/training/simulation_accident_dataset"
CONFIG_FILE = ROOT / "config.json"
MODEL_OUTPUT_DIR = ROOT / "assets/training/simulation_dataset"

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 90

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

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = SimulationBBoxProjector.get_matrix(bb_transform)
        vehicle_world_matrix = SimulationBBoxProjector.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))

        sensor_world_matrix = SimulationBBoxProjector.get_matrix(camera.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        cords_x_y_z = np.dot(world_sensor_matrix, world_cords)[:3, :]

        if all(cords_x_y_z[0, :] <= 0):
            return None

        cords_y_minus_z_x = np.vstack([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        
        calib = np.asarray(camera.calibration)
        cords_y_minus_z_x = np.asarray(cords_y_minus_z_x)
        bbox = np.transpose(np.dot(calib, cords_y_minus_z_x))
        bbox = np.asarray(bbox)
        
        x_proj = bbox[:, 0] / bbox[:, 2]
        y_proj = bbox[:, 1] / bbox[:, 2]
        depth = bbox[:, 2]
        
        camera_bbox = np.stack([x_proj, y_proj, depth], axis=1)
        return camera_bbox

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
    if not carla_dir:
        return None
    exe_path = Path(carla_dir) / "CarlaUE4.exe"
    if not exe_path.exists():
        print(f"[!] CarlaUE4.exe not found at: {exe_path}")
        return None
    
    print("[*] Spawning CARLA simulator process...")
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

def get_cctv_transforms(world):
    transforms = []
    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    selected_lights = []
    for tl in traffic_lights:
        if len(selected_lights) >= 4:
            break
        if all(tl.get_location().distance(s.get_location()) > 25.0 for s in selected_lights):
            selected_lights.append(tl)
            
    if len(selected_lights) < 4:
        for tl in traffic_lights:
            if len(selected_lights) >= 4:
                break
            if tl not in selected_lights:
                selected_lights.append(tl)
                
    for tl in selected_lights:
        loc = tl.get_location()
        waypoints = tl.get_affected_lane_waypoints()
        if waypoints:
            wp = waypoints[0]
            wp_loc = wp.transform.location
            wp_rot = wp.transform.rotation
            forward = wp_rot.get_forward_vector()
            cam_loc = wp_loc - forward * 12.0 + carla.Location(z=8.0)
            cam_rot = carla.Rotation(pitch=-35, yaw=wp_rot.yaw, roll=0)
            transforms.append(carla.Transform(cam_loc, cam_rot))
        else:
            rot = tl.get_transform().rotation
            forward = rot.get_forward_vector()
            cam_loc = loc + carla.Location(z=8.0) - forward * 2.5
            cam_rot = carla.Rotation(pitch=-35, yaw=rot.yaw, roll=0)
            transforms.append(carla.Transform(cam_loc, cam_rot))
            
    if len(transforms) < 4:
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for i in range(4 - len(transforms)):
            sp = spawn_points[i]
            cam_loc = sp.location + carla.Location(z=8.0)
            cam_rot = carla.Rotation(pitch=-35, yaw=sp.rotation.yaw, roll=0)
            transforms.append(carla.Transform(cam_loc, cam_rot))
            
    return transforms[:4]

def collect_dataset(client, num_frames=120):
    print("\n" + "=" * 58)
    print("  STAGE 1 — CARLA Accident Dataset Collection")
    print(f"  Target  : {num_frames} frames")
    print("=" * 58)
    
    world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    images_train = DATASET_DIR / "images/train"
    images_val = DATASET_DIR / "images/val"
    labels_train = DATASET_DIR / "labels/train"
    labels_val = DATASET_DIR / "labels/val"
    
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)
        
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
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        cam.calibration = calibration
        
        q = queue.Queue()
        cam.listen(make_queue_listener(q))
        cameras.append(cam)
        image_queues.append(q)
        print(f"  -> Spawned CCTV Camera {idx+1}")

    # Helper function to spawn crash — randomised vehicle pair, separation,
    # and angle each call so the dataset isn't just 4 fixed, identical poses
    # repeated hundreds of times (which would teach the model one specific
    # arrangement rather than "vehicles overlapping at a junction" generally)
    vehicle_bps_all = blueprint_library.filter("vehicle.*")

    def spawn_accident_vehicles(world, cam_transform, blueprint_library):
        forward = cam_transform.rotation.get_forward_vector()
        right = cam_transform.rotation.get_right_vector()
        junction_center = cam_transform.location + forward * 13.0

        bp1 = random.choice(vehicle_bps_all)
        bp2 = random.choice(vehicle_bps_all)

        separation   = random.uniform(2.0, 4.5)
        lateral_jit  = random.uniform(-1.5, 1.5)
        angle_jit    = random.uniform(-25, 25)

        t1 = carla.Transform(
            junction_center + right * lateral_jit + carla.Location(z=0.5),
            carla.Rotation(yaw=cam_transform.rotation.yaw + 90 + angle_jit)
        )
        t2 = carla.Transform(
            junction_center + forward * -separation + carla.Location(z=0.5),
            carla.Rotation(yaw=cam_transform.rotation.yaw + angle_jit * 0.5)
        )

        veh1 = world.try_spawn_actor(bp1, t1)
        veh2 = world.try_spawn_actor(bp2, t2)
        actors = []
        if veh1:
            actors.append(veh1)
        if veh2:
            actors.append(veh2)
        return actors

    # Spawn accidents at all junctions immediately
    print("[*] Spawning crash scenes at all junctions...")
    accident_actors = []
    for trans in cctv_transforms:
        actors = spawn_accident_vehicles(world, trans, blueprint_library)
        accident_actors.extend(actors)

    captured_count = 0
    step_interval = 8
    tick_count = 0
    RESPAWN_EVERY = 40   # captured frames between re-randomising the crash poses
    last_respawn_at = 0

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

            # Periodically re-randomise crash poses for visual diversity
            if captured_count - last_respawn_at >= RESPAWN_EVERY and captured_count > 0:
                for actor in accident_actors:
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except Exception:
                        pass
                accident_actors = []
                for trans in cctv_transforms:
                    accident_actors.extend(spawn_accident_vehicles(world, trans, blueprint_library))
                last_respawn_at = captured_count

            # Process each camera frame
            for cam_idx, img in enumerate(imgs):
                if captured_count >= num_frames:
                    break
                    
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                bgr_frame = array[:, :, :3]
                
                bboxes_lines = []
                cam = cameras[cam_idx]
                
                # Check for crash boxes to label as class 4 (Accident)
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
                        # Write as class 4: Accident
                        bboxes_lines.append(f"4 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                # Only save frame if we detected an accident
                if len(bboxes_lines) > 0:
                    is_val = (captured_count % 5 == 0)
                    img_dest = images_val if is_val else images_train
                    lbl_dest = labels_val if is_val else labels_train
                    
                    img_file = img_dest / f"accident_{captured_count:04d}.jpg"
                    lbl_file = lbl_dest / f"accident_{captured_count:04d}.txt"
                    
                    cv2.imwrite(str(img_file), bgr_frame)
                    with open(lbl_file, 'w') as f:
                        f.write("\n".join(bboxes_lines))
                        
                    captured_count += 1
                    
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
        
        for x in accident_actors:
            try:
                x.destroy()
            except Exception:
                pass
        print("[+] CARLA cleanup complete.")

def train_accident_model(epochs=3, batch_size=16):
    print("\n" + "=" * 58)
    print("  STAGE 2 — YOLOv8 Accident Model Training")
    print(f"  Model   : yolov8n.pt")
    print(f"  Epochs  : {epochs}")
    print("=" * 58)
    
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
        
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        workers=0,
        project=str(DATASET_DIR / "runs"),
        name="accident_run",
        exist_ok=True
    )
    
    best_weights = DATASET_DIR / "runs/accident_run/weights/best.pt"
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_model = MODEL_OUTPUT_DIR / "simulation_accident_detector.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, output_model)
        print(f"\n[+] SUCCESS: Accident model saved to: {output_model}")
        
        runs_dir = DATASET_DIR / "runs"
        if runs_dir.exists():
            try:
                shutil.rmtree(runs_dir)
            except Exception:
                pass
    else:
        print("[!] Warning: best.pt weights not found. Check training logs.")

def main():
    print("=========================================================")
    print("      CARLA DEDICATED ACCIDENT TRAINING (TRAIN2)         ")
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
            sys.exit(1)
        carla_proc = launch_carla_simulator(carla_path)
        if not carla_proc:
            sys.exit(1)
        print("[*] Waiting for CARLA server to start up (22s)...")
        time.sleep(22.0)
        
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(45.0)
        _ = client.get_world()
        print("[+] CARLA Connection established successfully.")
        
        # Step 1: Collect dataset
        collect_dataset(client, num_frames=320)

        # Step 2: Train Model
        train_accident_model(epochs=30, batch_size=16)
        
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
