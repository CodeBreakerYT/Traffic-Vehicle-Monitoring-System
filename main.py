import os
import sys
import math
import pygame
import tkinter as tk
from tkinter import filedialog
import socket
import struct
import threading
import time
import cv2
import numpy as np
import video_analyzer as va_module
import webbrowser

# Import modular components
import config
from resource_path import resource_path
from ui_widgets import (
    COLOR_BG, COLOR_GRID, COLOR_CYAN, COLOR_MAGENTA, COLOR_GREEN,
    COLOR_YELLOW, COLOR_WHITE, COLOR_MUTED, COLOR_CARD, Button, GearButton,
    draw_cyber_rect, draw_pixel_frame
)
from preview import SimulationPreview

CARLA_DOWNLOAD_URL = "https://carla.org/2025/09/16/release-0.9.16/"

# Initialize Tkinter root and hide it (for file dialogs)
try:
    tk_root = tk.Tk()
    tk_root.withdraw()
except Exception:
    tk_root = None

class Application:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TRAFFIC VEHICLE MONITORING SYSTEM")

        # Window/taskbar icon (separate from the .exe file icon, which
        # PyInstaller embeds at build time via build_car.py --icon)
        try:
            icon_surf = pygame.image.load(resource_path("assets", "icon", "car_ico.png"))
            pygame.display.set_icon(icon_surf)
        except Exception as e:
            print(f"[WARN] Could not load window icon: {e}")

        self.screen_width = 1280
        self.screen_height = 720
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True

        # Load configurations
        self.config_data = config.load_config()

        # Load fonts
        self.setup_fonts()

        # State management
        self.state = "MAIN_MENU"

        # Background grid animation variables
        self.grid_offset_y = 0
        self.grid_speed = 0.5

        # Initialize screen UI elements
        self.init_ui()

        # Video Analysis (Option 2) shared state
        self.video_analyzer  = va_module.VideoAnalyzer()
        self.va_latest_bgr   = None          # numpy BGR array from processing thread
        self.va_stats        = {}            # latest detection/classification stats
        self.va_lock         = threading.Lock()
        self.va_paused       = False
        self.va_seek_to      = None          # frame number to seek to (set by main thread)
        self.va_pos          = 0             # current frame position
        self.va_total        = 0             # total frames in video
        self.va_source_fps   = 30.0
        self.va_disp_fps     = 0.0
        self.va_ended        = False
        self.va_shutdown     = None
        self.va_thread       = None
        self.va_scrubbing    = False

        # CARLA Simulation (Option 3) variables
        self.sim_client = None
        self.sim_latest_bgr = None
        self.sim_stats = {}
        self.sim_status = "DISCONNECTED"
        self.sim_logs = []
        self.sim_lock = threading.Lock()
        self.sim_accident_active = False

        # Screen recording — shared across all 3 monitoring options
        self.recording        = False
        self.record_writer    = None
        self.record_path      = None
        self.record_start_time = 0.0
        self.record_last_write = 0.0

    def setup_fonts(self):
        """Sets up scalable system fonts with retro/cyber fallbacks."""
        try:
            self.font_title = pygame.font.SysFont("impact", 42)
            self.font_header = pygame.font.SysFont("consolas", 28, bold=True)
            self.font_body = pygame.font.SysFont("consolas", 16, bold=True)
            self.font_sm = pygame.font.SysFont("consolas", 10, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 48)
            self.font_header = pygame.font.Font(None, 32)
            self.font_body = pygame.font.Font(None, 20)
            self.font_sm = pygame.font.Font(None, 14)

    def init_ui(self):
        # Settings Gear Icon (top right)
        self.gear_button = GearButton((self.screen_width - 64, 20), self.go_to_settings)

        # 1. MAIN MENU WIDGETS
        self.main_buttons = [
            Button((self.screen_width // 2 - 150, 320, 300, 50), "START ACCESS", self.go_to_options),
            Button((self.screen_width // 2 - 150, 390, 300, 50), "SYSTEM SHUTDOWN", self.quit_app, COLOR_MAGENTA, COLOR_CYAN)
        ]

        # 2. SETTINGS WIDGETS
        self.settings_buttons = [
            Button((100, 480, 240, 50), "SELECT CARLA PATH", self.browse_carla_path),
            Button((380, 480, 240, 50), "SET CAMERA IP", self.prompt_camera_ip),
            Button((660, 480, 240, 50), "SET CAMERA PORT", self.prompt_camera_port),
            Button((self.screen_width // 2 - 120, 600, 240, 50), "SAVE & RETURN", self.save_settings, COLOR_GREEN, COLOR_CYAN)
        ]
        self.toggle_grid_btn = Button((650, 320, 160, 40), "TOGGLE GRID", self.toggle_grid, COLOR_MUTED)
        self.toggle_scan_btn = Button((650, 380, 160, 40), "TOGGLE CRT", self.toggle_scanlines, COLOR_MUTED)
        self.settings_buttons.extend([self.toggle_grid_btn, self.toggle_scan_btn])

        # 3. OPTIONS WIDGETS
        self.opt_back_btn = Button((20, 20, 160, 36), "< BACK", self.go_to_main_menu, COLOR_MUTED, COLOR_CYAN)
        
        # Space coordinates for buttons below the preview frame
        self.option_buttons = [
            # Row 1 (Options 1 and 2)
            Button((240, 410, 380, 46), "CONNECT LIVE CAMERA (CAM.EXE)", self.action_live_hookup),
            Button((660, 410, 380, 46), "PROCESS VIDEO / GIF FILE", self.action_video_analysis),
            # Row 2 (Option 3, centered)
            Button((450, 476, 380, 46), "LAUNCH CARLA SIMULATION", self.action_carla_simulation, COLOR_GREEN, COLOR_MAGENTA)
        ]
        # Shown below the CARLA button only when CARLA isn't installed/configured
        self.download_carla_btn = Button((520, 552, 240, 38), "DOWNLOAD CARLA", self.action_download_carla, COLOR_MAGENTA, COLOR_YELLOW)

        # 4. LIVE MONITORING WIDGETS
        self.cam_feed_rect = pygame.Rect(40, 100, 720, 480)
        self.cam_telemetry_rect = pygame.Rect(780, 100, 460, 480)
        self.cam_back_btn = Button((40, 600, 140, 46), "< BACK", self.stop_live_stream, COLOR_MUTED, COLOR_CYAN)
        self.cam_record_btn = Button((190, 600, 180, 46), "● RECORD", None, COLOR_MUTED, COLOR_MAGENTA)
        self.cam_record_btn.action = lambda: self.toggle_recording("live_stream", self.cam_record_btn)
        self.cam_discover_btn = Button((935, 600, 170, 46), "AUTO-DETECT", self.trigger_auto_detect, COLOR_YELLOW, COLOR_CYAN)
        self.cam_reconnect_btn = Button((1120, 600, 120, 46), "RECONNECT", self.reconnect_live_stream, COLOR_GREEN, COLOR_CYAN)

        # Preview module placed above options (Y=86 to Y=386)
        self.preview_module = SimulationPreview((374, 86, 533, 300))

        # Decorative hovering car art — main menu only
        self.hover_car_img = None
        try:
            raw = pygame.image.load(resource_path("assets", "menu", "hover_car_pixel.png")).convert_alpha()
            target_w = 230
            target_h = int(target_w * raw.get_height() / raw.get_width())
            self.hover_car_img = pygame.transform.scale(raw, (target_w, target_h))
            self.hover_car_rect = self.hover_car_img.get_rect(center=(self.screen_width // 2, 95))
        except Exception as e:
            print(f"[WARN] Could not load hover car art: {e}")

        # 5. VIDEO ANALYSIS WIDGETS
        # Layout: video feed (800x450) left | analytics panel (430px) right
        # Timeline scrubber below video | buttons below timeline
        self.va_feed_rect     = pygame.Rect(20,  52, 800, 450)
        self.va_panel_rect    = pygame.Rect(830, 52, 430, 510)
        self.va_timeline_rect = pygame.Rect(20, 510, 800,  22)
        self.va_back_btn  = Button((20,  556, 130, 40), "< BACK",    self.stop_video_analysis,  COLOR_MUTED,    COLOR_CYAN)
        self.va_pause_btn = Button((160, 556, 150, 40), "|| PAUSE",  self.toggle_video_pause,   COLOR_YELLOW,   COLOR_CYAN)
        self.va_record_btn = Button((320, 556, 160, 40), "● RECORD", None, COLOR_MUTED, COLOR_MAGENTA)
        self.va_record_btn.action = lambda: self.toggle_recording("video_analysis", self.va_record_btn)

        # 6. CARLA SIMULATION WIDGETS
        self.sim_feed_rect  = pygame.Rect(20,  52, 800, 450)
        self.sim_panel_rect = pygame.Rect(830, 52, 430, 510)
        self.sim_back_btn   = Button((20,  556, 150, 40), "< BACK",    self.stop_carla_simulation, COLOR_MUTED,    COLOR_CYAN)
        self.sim_accident_btn = Button((185, 556, 220, 40), "ACCIDENT: OFF", self.toggle_sim_accident, COLOR_MUTED, COLOR_MAGENTA)
        self.sim_record_btn = Button((415, 556, 170, 40), "● RECORD", None, COLOR_MUTED, COLOR_MAGENTA)
        self.sim_record_btn.action = lambda: self.toggle_recording("simulation", self.sim_record_btn)

    def go_to_main_menu(self):
        self.state = "MAIN_MENU"

    def go_to_options(self):
        self.state = "OPTIONS"

    def go_to_settings(self):
        self.temp_carla_path = self.config_data.get("carla_path", "")
        self.temp_camera_ip = self.config_data.get("camera_ip", "127.0.0.1")
        self.temp_camera_port = self.config_data.get("camera_port", 5000)
        self.state = "SETTINGS"

    def quit_app(self):
        # Gracefully stop stream if running before exit
        if hasattr(self, 'stream_shutdown'):
            self.stream_shutdown.set()
        if hasattr(self, 'sim_client') and self.sim_client:
            self.sim_client.stop()
        self.stop_recording()
        self.running = False

    # ── Screen recording (shared by all 3 monitoring options) ──────────────────

    def start_recording(self, tag):
        if self.recording:
            return
        import datetime
        out_dir = "recordings"
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = os.path.join(out_dir, f"{tag}_{stamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.record_writer = cv2.VideoWriter(
            self.record_path, fourcc, 20.0, (self.screen_width, self.screen_height)
        )
        if not self.record_writer.isOpened():
            print(f"[RECORD] Failed to open writer for {self.record_path}")
            self.record_writer = None
            return

        self.recording        = True
        self.record_start_time = time.time()
        self.record_last_write = 0.0
        print(f"[RECORD] Started → {self.record_path}")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.record_writer:
            self.record_writer.release()
            self.record_writer = None
        print(f"[RECORD] Saved → {self.record_path}")

    def toggle_recording(self, tag, button):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording(tag)
        button.text = "■ STOP REC" if self.recording else "● RECORD"
        button.color = COLOR_MAGENTA if self.recording else COLOR_MUTED
        button.accent_color = COLOR_CYAN if self.recording else COLOR_MAGENTA

    def _capture_record_frame(self):
        if not self.recording or not self.record_writer:
            return
        now = time.time()
        if now - self.record_last_write < (1.0 / 20.0):
            return
        self.record_last_write = now
        frame_arr = pygame.surfarray.array3d(self.screen)   # (W, H, 3) RGB
        frame_arr = frame_arr.transpose([1, 0, 2])            # (H, W, 3)
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        self.record_writer.write(frame_bgr)

    def _draw_recording_indicator(self):
        elapsed = time.time() - self.record_start_time
        mins, secs = int(elapsed) // 60, int(elapsed) % 60
        if int(time.time() * 2) % 2 == 0:
            pygame.draw.circle(self.screen, COLOR_MAGENTA, (self.screen_width - 112, 16), 7)
        rec_text = self.font_body.render(f"REC {mins:02d}:{secs:02d}", True, COLOR_MAGENTA)
        self.screen.blit(rec_text, (self.screen_width - 96, 6))

    def browse_carla_path(self):
        if tk_root:
            selected_dir = filedialog.askdirectory(
                initialdir=self.temp_carla_path or "C:\\",
                title="Select CARLA Installation Folder"
            )
            if selected_dir:
                self.temp_carla_path = os.path.normpath(selected_dir)
                print(f"Path selected: {self.temp_carla_path}")

    def prompt_camera_ip(self):
        if tk_root:
            from tkinter import simpledialog
            ip = simpledialog.askstring(
                "Camera Settings", 
                "Enter Camera IP (2nd laptop IP address):",
                initialvalue=self.temp_camera_ip
            )
            if ip:
                self.temp_camera_ip = ip.strip()

    def prompt_camera_port(self):
        if tk_root:
            from tkinter import simpledialog
            port = simpledialog.askinteger(
                "Camera Settings", 
                "Enter Camera Port:",
                initialvalue=self.temp_camera_port
            )
            if port is not None:
                self.temp_camera_port = port

    def save_settings(self):
        self.config_data["carla_path"] = self.temp_carla_path
        self.config_data["camera_ip"] = self.temp_camera_ip
        self.config_data["camera_port"] = self.temp_camera_port
        config.save_config(self.config_data)
        self.go_to_main_menu()

    def toggle_grid(self):
        self.config_data["grid_enabled"] = not self.config_data.get("grid_enabled", True)
        config.save_config(self.config_data)

    def toggle_scanlines(self):
        self.config_data["scanlines_enabled"] = not self.config_data.get("scanlines_enabled", True)
        config.save_config(self.config_data)

    def log_stream(self, message):
        timestamp = pygame.time.get_ticks() / 1000.0
        log_entry = f"[{timestamp:.1f}s] {message}"
        if not hasattr(self, 'stream_logs'):
            self.stream_logs = []
        self.stream_logs.append(log_entry)
        if len(self.stream_logs) > 11:
            self.stream_logs.pop(0)
        print(message)

    def stream_receiver_loop(self, ip, port, shutdown_event):
        self.stream_status = "CONNECTING"
        self.log_stream(f"Opening port to {ip}:{port}...")
        
        while not shutdown_event.is_set():
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(2.0)
            
            try:
                client_socket.connect((ip, port))
                self.stream_status = "CONNECTED"
                self.log_stream("Channel established. Receiving frame bytes...")
                client_socket.settimeout(1.5)
                
                frame_count = 0
                fps_start_time = time.time()
                
                while not shutdown_event.is_set():
                    # Read size header (4 bytes)
                    header = b""
                    while len(header) < 4:
                        if shutdown_event.is_set():
                            break
                        chunk = client_socket.recv(4 - len(header))
                        if not chunk:
                            raise socket.error("Connection closed by remote server.")
                        header += chunk
                    
                    if shutdown_event.is_set():
                        break
                        
                    size = struct.unpack("!I", header)[0]
                    self.stream_frame_size = size / 1024.0
                    
                    # Read image payload
                    data = b""
                    while len(data) < size:
                        if shutdown_event.is_set():
                            break
                        chunk = client_socket.recv(size - len(data))
                        if not chunk:
                            raise socket.error("Connection truncated during data read.")
                        data += chunk
                        
                    if shutdown_event.is_set():
                        break
                        
                    # Decode frame
                    nparr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_rgb = cv2.resize(frame_rgb, (self.cam_feed_rect.width, self.cam_feed_rect.height))
                        frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                        
                        with self.stream_lock:
                            self.stream_frame = frame_surf
                            
                        frame_count += 1
                        elapsed = time.time() - fps_start_time
                        if elapsed >= 1.0:
                            self.stream_fps = int(frame_count / elapsed)
                            frame_count = 0
                            fps_start_time = time.time()
                            
            except (socket.error, ConnectionResetError, BrokenPipeError) as e:
                self.stream_status = "RECONNECTING"
                self.log_stream(f"Connection lost: {e}")
                with self.stream_lock:
                    self.stream_frame = None
                client_socket.close()
                for _ in range(20):
                    if shutdown_event.is_set():
                        break
                    time.sleep(0.1)
            except Exception as e:
                self.stream_status = "RECONNECTING"
                self.log_stream(f"Network error: {e}")
                with self.stream_lock:
                    self.stream_frame = None
                client_socket.close()
                for _ in range(20):
                    if shutdown_event.is_set():
                        break
                    time.sleep(0.1)
            finally:
                try:
                    client_socket.close()
                except Exception:
                    pass

    def start_live_stream(self):
        self.state = "LIVE_STREAM"
        self.stream_status = "CONNECTING"
        self.stream_frame = None
        self.stream_fps = 0
        self.stream_frame_size = 0.0
        self.stream_logs = []
        
        self.stream_shutdown = threading.Event()
        self.stream_lock = threading.Lock()
        
        ip = self.config_data.get("camera_ip", "127.0.0.1")
        port = self.config_data.get("camera_port", 5000)
        
        self.log_stream(f"Initializing monitor connection loop...")
        
        self.stream_thread = threading.Thread(
            target=self.stream_receiver_loop,
            args=(ip, port, self.stream_shutdown),
            daemon=True
        )
        self.stream_thread.start()

    def stop_live_stream(self):
        self.log_stream("Closing sensor connection thread...")
        self.stream_shutdown.set()
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=0.5)
        self.stream_frame = None
        if self.recording:
            self.toggle_recording("live_stream", self.cam_record_btn)
        self.state = "OPTIONS"

    def reconnect_live_stream(self):
        self.log_stream("Resetting interface socket binding...")
        self.stream_shutdown.set()
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=0.5)
        self.start_live_stream()

    def trigger_auto_detect(self):
        if hasattr(self, 'discovery_active') and self.discovery_active:
            self.log_stream("Auto-detection already in progress...")
            return
        self.discovery_active = True
        threading.Thread(target=self.bg_discovery_thread, daemon=True).start()

    def bg_discovery_thread(self):
        try:
            self.log_stream("Initializing auto-detection...")
            found_ip = self.discover_camera()
            if found_ip:
                self.log_stream(f"Updating connection host to {found_ip}...")
                self.reconnect_live_stream()
            else:
                self.log_stream("Auto-detection finished. Device not found.")
        except Exception as e:
            self.log_stream(f"Discovery error: {e}")
        finally:
            self.discovery_active = False

    def discover_camera(self):
        """Scans the local subnet for a running camera streamer on the configured port."""
        self.log_stream("Scanning local network interfaces...")
        local_ips = []
        try:
            hostname = socket.gethostname()
            # Retrieve all local IPs including virtual adapters
            for info in socket.getaddrinfo(hostname, None):
                ip = info[4][0]
                if "." in ip and not ip.startswith("127.") and ip not in local_ips:
                    local_ips.append(ip)
        except Exception:
            pass
            
        # Fallback UDP socket trick
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
            s.close()
            if primary_ip not in local_ips and not primary_ip.startswith("127."):
                local_ips.insert(0, primary_ip)
        except Exception:
            pass

        if not local_ips:
            # Fallback to local subnet check assuming common subnets
            local_ips = ["192.168.1.100", "192.168.0.100", "10.0.0.100"]
            self.log_stream("No active interface. Trying common subnets...")

        port = self.config_data.get("camera_port", 5000)
        found_ip = None
        
        import concurrent.futures
        
        # Scan subnets of detected local IPs
        for local_ip in local_ips:
            parts = local_ip.split('.')
            if len(parts) != 4:
                continue
            subnet_prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
            self.log_stream(f"Scanning subnet {subnet_prefix}0/24 on port {port}...")
            
            def check_ip(ip_suffix):
                target_ip = f"{subnet_prefix}{ip_suffix}"
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.2) # Short timeout for quick scanning
                    s.connect((target_ip, port))
                    s.close()
                    return target_ip
                except Exception:
                    return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
                futures = {executor.submit(check_ip, i): i for i in range(1, 255)}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        found_ip = res
                        break
            if found_ip:
                break

        if found_ip:
            self.log_stream(f"SUCCESS: Camera found at {found_ip}!")
            self.config_data["camera_ip"] = found_ip
            self.temp_camera_ip = found_ip
            config.save_config(self.config_data)
            return found_ip
        else:
            self.log_stream("Camera streamer not found in local subnets.")
            return None

    def action_live_hookup(self):
        self.start_live_stream()

    def action_video_analysis(self):
        if tk_root:
            selected_file = filedialog.askopenfilename(
                title="Select Traffic Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.MOV *.MP4"), ("All Files", "*.*")]
            )
            if selected_file:
                self.start_video_analysis(selected_file)

    # ── Video Analysis (Option 2) ──────────────────────────────────────────────

    def start_video_analysis(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[!] Cannot open video: {path}")
            return

        self.va_cap        = cap
        self.va_total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.va_source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.va_pos        = 0
        self.va_paused     = False
        self.va_seek_to    = None
        self.va_ended      = False
        self.va_disp_fps   = 0.0
        self.va_scrubbing  = False

        with self.va_lock:
            self.va_latest_bgr = None
            self.va_stats = {
                "counts": {}, "accident": None,
                "accident_conf": 0.0, "severity": None, "severity_heuristic": False,
            }

        if not self.video_analyzer.ready:
            print("[*] Loading YOLO models — first launch may take ~30 s...")
            self.video_analyzer.load()
            if self.video_analyzer.has_accident:
                print("[+] Accident detector loaded.")
            else:
                print("[~] No trained accident_detector.pt — run train.py first.")

        # Clear vehicle-tracking state so the previous video's motion
        # history doesn't bleed into this one
        self.video_analyzer.reset()

        self.va_pause_btn.text = "|| PAUSE"
        self.va_shutdown = threading.Event()
        self.va_thread   = threading.Thread(target=self._va_proc_loop, daemon=True)
        self.va_thread.start()
        self.state = "VIDEO_ANALYSIS"

    def stop_video_analysis(self):
        if self.va_shutdown:
            self.va_shutdown.set()
        if self.va_thread and self.va_thread.is_alive():
            self.va_thread.join(timeout=1.5)
        if hasattr(self, "va_cap") and self.va_cap:
            self.va_cap.release()
        with self.va_lock:
            self.va_latest_bgr = None
        if self.recording:
            self.toggle_recording("video_analysis", self.va_record_btn)
        self.state = "OPTIONS"

    def toggle_video_pause(self):
        self.va_paused = not self.va_paused
        self.va_pause_btn.text = "▶ RESUME" if self.va_paused else "|| PAUSE"

    def _va_seek_from_mouse(self, mouse_x):
        tr  = self.va_timeline_rect
        rel = max(0, min(mouse_x - tr.x, tr.width))
        self.va_seek_to = int(rel / tr.width * self.va_total)

    def _va_proc_loop(self):
        clf_counter = 0
        fps_timer   = time.time()
        fps_count   = 0

        while not self.va_shutdown.is_set():

            # Handle seek request (works even while paused or after end)
            seek = self.va_seek_to
            if seek is not None:
                self.va_cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                self.va_seek_to = None
                self.va_ended   = False
                clf_counter     = 0
                # Jumping to an arbitrary point invalidates the tracker's
                # motion history — without this, stale pre-seek positions get
                # compared against the new frame, producing bogus "abrupt
                # stop" detections tied to a different point in the video,
                # and can misattribute the accident box to the wrong vehicle
                self.video_analyzer.reset()
                if self.va_paused:
                    # Render just this one seeked frame so the display updates
                    ret, frame = self.va_cap.read()
                    if ret:
                        annotated, s = self.video_analyzer.process(frame, classify=True)
                        with self.va_lock:
                            self.va_latest_bgr = annotated
                            self.va_stats.update(s)
                            self.va_pos = int(self.va_cap.get(cv2.CAP_PROP_POS_FRAMES))
                continue

            if self.va_paused or self.va_ended:
                time.sleep(0.03)
                continue

            t0  = time.time()
            ret, frame = self.va_cap.read()
            if not ret:
                self.va_ended = True
                continue

            clf_counter += 1
            # Classify more often while an accident is already active
            interval = va_module.CLASSIFY_EVERY // 2 if self.va_stats.get("accident") else va_module.CLASSIFY_EVERY
            do_classify = (clf_counter % interval == 0)

            annotated, s = self.video_analyzer.process(frame, classify=do_classify)

            with self.va_lock:
                self.va_latest_bgr = annotated
                # Always update per-frame counts
                self.va_stats["counts"] = s["counts"]
                # Only update accident/severity when classification actually ran
                # (prevents None from overwriting a valid detection on non-classify frames)
                if do_classify:
                    self.va_stats["accident"]           = s["accident"]
                    self.va_stats["accident_conf"]      = s["accident_conf"]
                    self.va_stats["severity"]           = s["severity"]
                    self.va_stats["severity_heuristic"] = s["severity_heuristic"]
                self.va_pos = int(self.va_cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Measure processing FPS
            fps_count += 1
            now = time.time()
            if now - fps_timer >= 0.5:
                self.va_disp_fps = fps_count / (now - fps_timer)
                fps_count  = 0
                fps_timer  = now

            # Throttle to source video FPS
            spent   = time.time() - t0
            slack   = (1.0 / self.va_source_fps) - spent
            if slack > 0:
                time.sleep(slack)

    def is_carla_installed(self):
        carla_dir = self.config_data.get("carla_path", "")
        if not carla_dir or not os.path.isdir(carla_dir):
            return False
        return os.path.exists(os.path.join(carla_dir, "CarlaUE4.exe"))

    def action_carla_simulation(self):
        print("[ACTION] Launching CARLA Simulation process...")
        carla_dir = self.config_data.get("carla_path", "")
        if not carla_dir or not os.path.isdir(carla_dir):
            print("[WARNING] CARLA directory invalid or not set.")
            self.go_to_settings()
        else:
            print(f"[ACTION] Found valid CARLA directory: {carla_dir}")
            self.start_carla_simulation()

    def action_download_carla(self):
        print(f"[ACTION] Opening CARLA download page: {CARLA_DOWNLOAD_URL}")
        webbrowser.open(CARLA_DOWNLOAD_URL)

    def log_sim(self, message):
        timestamp = pygame.time.get_ticks() / 1000.0
        log_entry = f"[{timestamp:.1f}s] {message}"
        if not hasattr(self, 'sim_logs'):
            self.sim_logs = []
        self.sim_logs.append(log_entry)
        if len(self.sim_logs) > 6:
            self.sim_logs.pop(0)
        print(message)

    def on_sim_frame(self, bgr, stats):
        with self.sim_lock:
            self.sim_latest_bgr = bgr.copy()
            self.sim_stats = stats
            self.sim_status = "CONNECTED"

    def start_carla_simulation(self):
        self.state = "CARLA_SIMULATION"
        self.sim_latest_bgr = None
        self.sim_stats = {}
        self.sim_status = "CONNECTING"
        self.sim_logs = []
        self.sim_accident_active = False
        self.sim_accident_btn.text = "ACCIDENT: OFF"
        self.sim_accident_btn.color = COLOR_MUTED
        self.sim_accident_btn.accent_color = COLOR_MAGENTA
        
        self.log_sim("Initializing CARLA client...")
        
        try:
            from scripts.option_3.simulation_client import CARLASimulationClient
            
            carla_dir = self.config_data.get("carla_path", "")
            model_path = resource_path("assets", "training", "simulation_dataset", "simulation_detector.pt")
            
            self.sim_client = CARLASimulationClient(
                carla_path=carla_dir,
                model_path=model_path,
                on_frame_callback=self.on_sim_frame,
                log_callback=self.log_sim
            )
            self.sim_client.start()
        except Exception as e:
            self.log_sim(f"Error starting simulation: {e}")
            self.sim_status = "ERROR"

    def stop_carla_simulation(self):
        self.log_sim("Shutting down CARLA client thread...")
        self.sim_accident_active = False
        if self.sim_client:
            self.sim_client.stop()
            self.sim_client = None
        if self.recording:
            self.toggle_recording("simulation", self.sim_record_btn)
        self.state = "OPTIONS"

    def trigger_sim_spawn(self):
        if self.sim_client:
            self.sim_client.spawn_additional_traffic(15)

    def toggle_sim_accident(self):
        if self.sim_client:
            self.sim_accident_active = not self.sim_accident_active
            self.sim_client.toggle_accident(self.sim_accident_active)
            self.sim_accident_btn.text = "ACCIDENT: ON" if self.sim_accident_active else "ACCIDENT: OFF"
            self.sim_accident_btn.color = COLOR_MAGENTA if self.sim_accident_active else COLOR_MUTED
            self.sim_accident_btn.accent_color = COLOR_CYAN if self.sim_accident_active else COLOR_MAGENTA

    def update(self):
        mouse_pos = pygame.mouse.get_pos()
        self.grid_offset_y = (self.grid_offset_y + self.grid_speed) % 40

        if self.state in ["MAIN_MENU", "OPTIONS"]:
            self.gear_button.update(mouse_pos)

        if self.state == "MAIN_MENU":
            for btn in self.main_buttons:
                btn.update(mouse_pos)
        elif self.state == "SETTINGS":
            for btn in self.settings_buttons:
                btn.update(mouse_pos)
        elif self.state == "OPTIONS":
            self.opt_back_btn.update(mouse_pos)
            for btn in self.option_buttons:
                btn.update(mouse_pos)
            if not self.is_carla_installed():
                self.download_carla_btn.update(mouse_pos)

            # Determine which GIF to display in the preview module
            # Default is 1.gif
            target_gif = resource_path("assets", "gif", "1.gif")

            # Hover over Button 1 (Connect Live Camera) -> 1.gif
            if self.option_buttons[0].hovered:
                target_gif = resource_path("assets", "gif", "1.gif")
            # Hover over Button 2 (Process Video) -> 2.gif
            elif self.option_buttons[1].hovered:
                target_gif = resource_path("assets", "gif", "2.gif")
            # Hover over Button 3 (CARLA Simulation) -> 3.gif
            elif self.option_buttons[2].hovered:
                target_gif = resource_path("assets", "gif", "3.gif")
                
            self.preview_module.set_active_gif(target_gif)
            self.preview_module.update()
        elif self.state == "LIVE_STREAM":
            self.cam_back_btn.update(mouse_pos)
            self.cam_record_btn.update(mouse_pos)
            self.cam_discover_btn.update(mouse_pos)
            self.cam_reconnect_btn.update(mouse_pos)
        elif self.state == "VIDEO_ANALYSIS":
            self.va_back_btn.update(mouse_pos)
            self.va_pause_btn.update(mouse_pos)
            self.va_record_btn.update(mouse_pos)
        elif self.state == "CARLA_SIMULATION":
            self.sim_back_btn.update(mouse_pos)
            self.sim_accident_btn.update(mouse_pos)
            self.sim_record_btn.update(mouse_pos)

    def draw_background(self):
        self.screen.fill(COLOR_BG)
        if self.config_data.get("grid_enabled", True):
            for y in range(0, self.screen_height, 40):
                animated_y = (y + self.grid_offset_y) % self.screen_height
                pygame.draw.line(self.screen, COLOR_GRID, (0, animated_y), (self.screen_width, animated_y), 1)
            for x in range(0, self.screen_width, 40):
                pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, self.screen_height), 1)

    def draw_scanlines(self):
        if self.config_data.get("scanlines_enabled", True):
            for y in range(0, self.screen_height, 4):
                line_surf = pygame.Surface((self.screen_width, 1), pygame.SRCALPHA)
                line_surf.fill((0, 0, 0, 45))
                self.screen.blit(line_surf, (0, y))

    def draw(self):
        self.draw_background()

        if self.state == "MAIN_MENU":
            self.draw_main_menu()
        elif self.state == "SETTINGS":
            self.draw_settings()
        elif self.state == "OPTIONS":
            self.draw_options()
        elif self.state == "LIVE_STREAM":
            self.draw_live_stream()
        elif self.state == "VIDEO_ANALYSIS":
            self.draw_video_analysis()
        elif self.state == "CARLA_SIMULATION":
            self.draw_carla_simulation()

        if self.state in ["MAIN_MENU", "OPTIONS"]:
            self.gear_button.draw(self.screen)

        if self.recording and self.state in ["LIVE_STREAM", "VIDEO_ANALYSIS", "CARLA_SIMULATION"]:
            self._draw_recording_indicator()

        self.draw_scanlines()
        self._capture_record_frame()
        pygame.display.flip()

    def draw_carla_simulation(self):
        with self.sim_lock:
            bgr = self.sim_latest_bgr
            stats = dict(self.sim_stats)
        status = self.sim_status
        
        # ── Title ────────────────────────────────────────────────────────────
        title_surf = self.font_header.render("MONITORING PROTOCOL: CARLA SIMULATION", True, COLOR_CYAN)
        self.screen.blit(title_surf, (20, 14))

        # ── Video feed area ───────────────────────────────────────────────────
        pygame.draw.rect(self.screen, COLOR_CARD, self.sim_feed_rect)

        if bgr is not None:
            fw, fh = self.sim_feed_rect.width, self.sim_feed_rect.height
            resized = cv2.resize(bgr, (fw, fh))
            rgb_arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(rgb_arr.swapaxes(0, 1))
            self.screen.blit(frame_surf, self.sim_feed_rect.topleft)
        else:
            # Loading placeholder
            status_desc = "ESTABLISHING TCP HANDSHAKE..."
            if status == "CONNECTING":
                status_desc = "CONNECTING TO CARLA SERVER..."
            elif status == "ERROR":
                status_desc = "CONNECTION ERROR - CHECK SETTINGS"
            loading_surf = self.font_header.render(status_desc, True, COLOR_MAGENTA)
            self.screen.blit(loading_surf, loading_surf.get_rect(center=self.sim_feed_rect.center))

        draw_pixel_frame(self.screen, self.sim_feed_rect, COLOR_CYAN, COLOR_GREEN)

        # Custom/Fallback model info tag top-left of feed
        is_fallback = stats.get("fallback", True)
        model_name = "COCO PRETRAINED (FALLBACK)" if is_fallback else "CUSTOM SIMULATION MODEL"
        model_color = COLOR_YELLOW if is_fallback else COLOR_GREEN
        model_tag = self.font_sm.render(f"MODEL: {model_name}", True, model_color)
        self.screen.blit(model_tag, (self.sim_feed_rect.x + 10, self.sim_feed_rect.y + 8))

        # Flashing Live Indicator
        if pygame.time.get_ticks() % 1000 < 500:
            rec_color = COLOR_MAGENTA if status == "CONNECTED" else COLOR_MUTED
            pygame.draw.circle(self.screen, rec_color, (self.sim_feed_rect.right - 90, self.sim_feed_rect.top + 20), 6)
            rec_text = self.font_body.render("SIM ACTIVE", True, rec_color)
            self.screen.blit(rec_text, (self.sim_feed_rect.right - 80, self.sim_feed_rect.top + 10))

        # Flashing Critical Alert HUD overlay if any accident detected
        any_accident = any(j.get("accident", False) for j in stats.get("junctions", []))
        if any_accident and (pygame.time.get_ticks() % 600 < 300):
            alert_badge = self.font_body.render("CRITICAL ALERT", True, COLOR_MAGENTA)
            self.screen.blit(alert_badge, (self.sim_feed_rect.right - 240, self.sim_feed_rect.top + 10))

        # ── Analytics & Terminal panel ─────────────────────────────────────────
        self._draw_sim_panel(stats, status)

        # ── Buttons ───────────────────────────────────────────────────────────
        self.sim_back_btn.draw(self.screen, self.font_body)
        self.sim_accident_btn.draw(self.screen, self.font_body)
        self.sim_record_btn.draw(self.screen, self.font_body)

    def _draw_sim_panel(self, stats, status):
        px, py = self.sim_panel_rect.x, self.sim_panel_rect.y
        pw     = self.sim_panel_rect.width
        ph     = self.sim_panel_rect.height

        draw_cyber_rect(self.screen, COLOR_MUTED, self.sim_panel_rect, cut_size=14, thickness=2)

        # Header
        hdr = self.font_header.render("TELEMETRY & CONTROL", True, COLOR_CYAN)
        self.screen.blit(hdr, (px + 20, py + 16))
        
        # Check if any junction has an accident for global panel alert
        junctions = stats.get("junctions", [])
        any_accident = any(j.get("accident", False) for j in junctions)
        if any_accident and (pygame.time.get_ticks() % 600 < 300):
            alert_surf = self.font_body.render("[!!] ALARM", True, COLOR_MAGENTA)
            self.screen.blit(alert_surf, (px + pw - 120, py + 22))
            
        pygame.draw.line(self.screen, COLOR_MUTED, (px + 10, py + 46), (px + pw - 10, py + 46), 1)

        lx   = px + 20
        cy   = py + 58
        rh   = 24       # row height
        vcol = lx + 160 # value column x

        def row(label, value, color=COLOR_WHITE):
            nonlocal cy
            self.screen.blit(self.font_body.render(label, True, COLOR_MUTED), (lx, cy))
            self.screen.blit(self.font_body.render(str(value), True, color),  (vcol, cy))
            cy += rh

        # Status
        led_color = COLOR_GREEN if status == "CONNECTED" else COLOR_MAGENTA
        pygame.draw.circle(self.screen, led_color, (lx + 8, cy + 10), 6)
        self.screen.blit(self.font_body.render("CARLA ENGINE", True, COLOR_MUTED), (lx + 24, cy))
        self.screen.blit(self.font_body.render(status, True, led_color), (vcol, cy))
        cy += rh

        # Junction Breakdown Section
        cy += 4
        self.screen.blit(self.font_sm.render("JUNCTION TRAFFIC BREAKDOWN", True, COLOR_MUTED), (lx, cy))
        cy += 14
        
        junctions = stats.get("junctions", [])
        for i, name in enumerate(["ALPHA (CAM 1)", "BETA  (CAM 2)", "GAMMA (CAM 3)", "DELTA (CAM 4)"]):
            total_j = 0
            has_accident = False
            if i < len(junctions):
                total_j = sum(v for k, v in junctions[i].items() if k not in ["Person", "accident"])
                has_accident = junctions[i].get("accident", False)
                
            self.screen.blit(self.font_body.render(f"JNC {name}", True, COLOR_MUTED), (lx, cy))
            
            val_color = COLOR_CYAN if total_j > 0 else COLOR_MUTED
            if has_accident:
                val_color = COLOR_MAGENTA
            self.screen.blit(self.font_body.render(str(total_j), True, val_color), (vcol, cy))
            
            if has_accident:
                tick = pygame.time.get_ticks()
                if tick % 600 < 300:
                    alert_lbl = self.font_sm.render("[!!] ACCIDENT [!!]", True, COLOR_MAGENTA)
                    self.screen.blit(alert_lbl, (vcol + 45, cy + 4))
            cy += rh

        # Global Statistics Section
        cy += 4
        self.screen.blit(self.font_sm.render("GLOBAL TRACKED STATS", True, COLOR_MUTED), (lx, cy))
        cy += 14

        counts = stats.get("counts", {})
        for lbl in ("Car", "Truck", "Bus", "Motorcycle"):
            n = counts.get(lbl, 0)
            row(lbl.upper(), n, COLOR_CYAN if n > 0 else COLOR_MUTED)

        accident_count = stats.get("accident_count", 0)
        row("ACCIDENT COUNT", accident_count, COLOR_MAGENTA if accident_count > 0 else COLOR_MUTED)
        
        # Logs/Terminal Panel inside dashboard
        pygame.draw.line(self.screen, (55, 22, 75), (lx, cy), (px + pw - 10, cy), 1)
        cy += 8
        self.screen.blit(self.font_sm.render("SIMULATION LOGS", True, COLOR_MUTED), (lx, cy))
        cy += 14
        
        log_panel_rect = pygame.Rect(lx, cy, pw - 40, ph - (cy - py) - 20)
        draw_cyber_rect(self.screen, COLOR_GRID, log_panel_rect, cut_size=8, thickness=1, fill=True)
        pygame.draw.rect(self.screen, COLOR_MUTED, log_panel_rect, 1)
        
        log_y = log_panel_rect.top + 8
        for log in self.sim_logs:
            log_surf = self.font_sm.render(log, True, COLOR_GREEN)
            self.screen.blit(log_surf, (log_panel_rect.left + 10, log_y))
            log_y += 15

    def draw_main_menu(self):
        if self.hover_car_img:
            bob = int(8 * math.sin(pygame.time.get_ticks() / 450.0))
            pos = self.hover_car_rect.move(0, bob)
            self.screen.blit(self.hover_car_img, pos)

        title = "TRAFFIC VEHICLE MONITORING SYSTEM"
        title_shadow = self.font_title.render(title, True, COLOR_MAGENTA)
        title_text = self.font_title.render(title, True, COLOR_WHITE)
        self.screen.blit(title_shadow, (self.screen_width // 2 - title_text.get_width() // 2 + 4, 154))
        self.screen.blit(title_text, (self.screen_width // 2 - title_text.get_width() // 2, 150))

        sub = "ACCESS CONSOLE V1.0 // CONNECTIVITY PREPARED"
        sub_text = self.font_body.render(sub, True, COLOR_GREEN)
        self.screen.blit(sub_text, (self.screen_width // 2 - sub_text.get_width() // 2, 220))

        for btn in self.main_buttons:
            btn.draw(self.screen, self.font_body)

    def draw_settings(self):
        title_surf = self.font_header.render("SYSTEM SETTINGS CONFIGURATION", True, COLOR_CYAN)
        self.screen.blit(title_surf, (100, 60))

        card_rect = pygame.Rect(100, 140, 1080, 300)
        draw_cyber_rect(self.screen, COLOR_MUTED, card_rect, cut_size=16, thickness=2)
        
        # Left Side: Carla path info
        path_title = self.font_body.render("CARLA SIMULATION PATH:", True, COLOR_WHITE)
        self.screen.blit(path_title, (130, 180))
        
        path_str = self.temp_carla_path or "[NOT CONFIGURED // DEFAULT WILL BE USED]"
        if len(path_str) > 50:
            path_str = "..." + path_str[-47:]
        path_color = COLOR_YELLOW if self.temp_carla_path else COLOR_MAGENTA
        path_text = self.font_body.render(path_str, True, path_color)
        self.screen.blit(path_text, (130, 210))

        help_text1 = self.font_sm.render("* Required path where CarlaUE4.exe and PythonAPI reside.", True, COLOR_MUTED)
        help_text2 = self.font_sm.render("* If not configured, simulation launcher will look in typical directories.", True, COLOR_MUTED)
        self.screen.blit(help_text1, (130, 245))
        self.screen.blit(help_text2, (130, 265))

        # Right Side: Camera stream IP/Port info
        cam_title = self.font_body.render("CAMERA STREAM ADDRESS (2ND LAPTOP):", True, COLOR_WHITE)
        self.screen.blit(cam_title, (650, 180))
        
        cam_str = f"{self.temp_camera_ip}:{self.temp_camera_port}"
        cam_text = self.font_body.render(cam_str, True, COLOR_YELLOW)
        self.screen.blit(cam_text, (650, 210))
        
        cam_help1 = self.font_sm.render("* Configure to the IP address displayed by cam.exe on 2nd laptop.", True, COLOR_MUTED)
        cam_help2 = self.font_sm.render("* Ensure both laptops are on the same Wi-Fi network.", True, COLOR_MUTED)
        self.screen.blit(cam_help1, (650, 245))
        self.screen.blit(cam_help2, (650, 265))

        grid_status = "ENABLED" if self.config_data.get("grid_enabled", True) else "DISABLED"
        scan_status = "ENABLED" if self.config_data.get("scanlines_enabled", True) else "DISABLED"
        
        grid_label = self.font_body.render(f"Grid Background: {grid_status}", True, COLOR_CYAN)
        scan_label = self.font_body.render(f"CRT Scanlines Overlay: {scan_status}", True, COLOR_CYAN)
        self.screen.blit(grid_label, (130, 310))
        self.screen.blit(scan_label, (130, 360))

        for btn in self.settings_buttons:
            btn.draw(self.screen, self.font_body)

    def draw_options(self):
        title_surf = self.font_header.render("SELECT MONITORING PROTOCOL", True, COLOR_CYAN)
        self.screen.blit(title_surf, (240, 24))

        self.opt_back_btn.draw(self.screen, self.font_body)

        for btn in self.option_buttons:
            btn.draw(self.screen, self.font_body)

        self.preview_module.draw(self.screen, self.font_sm)

        if not self.is_carla_installed():
            warn_surf = self.font_body.render("⚠ CARLA NOT DOWNLOADED", True, COLOR_MAGENTA)
            self.screen.blit(warn_surf, warn_surf.get_rect(center=(self.screen_width // 2, 534)))
            self.download_carla_btn.draw(self.screen, self.font_sm)

    def draw_live_stream(self):
        # 1. Main Titles
        title_surf = self.font_header.render("MONITORING PROTOCOL: LIVE SENSOR FEED", True, COLOR_CYAN)
        self.screen.blit(title_surf, (40, 24))
        
        sub = "SECURE ENCRYPTED NETWORK CHANNEL // DECRYPT ACTIVE"
        sub_text = self.font_body.render(sub, True, COLOR_GREEN)
        self.screen.blit(sub_text, (40, 60))
        
        # 2. Draw camera feed frame background
        pygame.draw.rect(self.screen, COLOR_CARD, self.cam_feed_rect)
        
        # Get frame safely
        feed_surface = None
        with self.stream_lock:
            if self.stream_frame:
                feed_surface = self.stream_frame.copy()
                
        if feed_surface:
            self.screen.blit(feed_surface, self.cam_feed_rect.topleft)
            
            # Interactive HUD details on feed (crosshairs)
            cx, cy = self.cam_feed_rect.center
            pygame.draw.circle(self.screen, COLOR_GREEN, (cx, cy), 15, 1)
            pygame.draw.line(self.screen, COLOR_GREEN, (cx - 25, cy), (cx - 5, cy), 1)
            pygame.draw.line(self.screen, COLOR_GREEN, (cx + 5, cy), (cx + 25, cy), 1)
            pygame.draw.line(self.screen, COLOR_GREEN, (cx, cy - 25), (cx, cy - 5), 1)
            pygame.draw.line(self.screen, COLOR_GREEN, (cx, cy + 5), (cx, cy + 25), 1)
        else:
            # Draw empty static overlay with loading text
            pygame.draw.rect(self.screen, (15, 15, 25), self.cam_feed_rect)
            
            status_desc = "ESTABLISHING TCP HANDSHAKE..."
            if self.stream_status == "RECONNECTING":
                status_desc = "LINK LOSS DETECTED - REESTABLISHING..."
                
            warn_surf = self.font_header.render(status_desc, True, COLOR_MAGENTA)
            w_rect = warn_surf.get_rect(center=self.cam_feed_rect.center)
            self.screen.blit(warn_surf, w_rect)
            
            # Draw technical grid lines
            for x in range(self.cam_feed_rect.left, self.cam_feed_rect.right, 40):
                pygame.draw.line(self.screen, (25, 15, 25), (x, self.cam_feed_rect.top), (x, self.cam_feed_rect.bottom), 1)
            for y in range(self.cam_feed_rect.top, self.cam_feed_rect.bottom, 40):
                pygame.draw.line(self.screen, (25, 15, 25), (self.cam_feed_rect.left, y), (self.cam_feed_rect.right, y), 1)

        # Draw pixelated cyber borders on camera feed frame
        draw_pixel_frame(self.screen, self.cam_feed_rect, COLOR_CYAN, COLOR_GREEN)
        
        # 3. Draw Telemetry Dashboard
        draw_cyber_rect(self.screen, COLOR_MUTED, self.cam_telemetry_rect, cut_size=16, thickness=2)
        
        # Dashboard title
        tel_title = self.font_header.render("SYSTEM TELEMETRY", True, COLOR_WHITE)
        self.screen.blit(tel_title, (self.cam_telemetry_rect.left + 20, self.cam_telemetry_rect.top + 20))
        
        # Status LED
        led_color = COLOR_GREEN if self.stream_status == "CONNECTED" else COLOR_MAGENTA
        pygame.draw.circle(self.screen, led_color, (self.cam_telemetry_rect.left + 40, self.cam_telemetry_rect.top + 80), 8)
        
        status_label = self.font_body.render(f"CHANNEL STATUS: {self.stream_status}", True, led_color)
        self.screen.blit(status_label, (self.cam_telemetry_rect.left + 60, self.cam_telemetry_rect.top + 70))
        
        # Target address details
        ip = self.config_data.get("camera_ip", "127.0.0.1")
        port = self.config_data.get("camera_port", 5000)
        addr_text = self.font_body.render(f"SOURCE HOST   : {ip}:{port}", True, COLOR_WHITE)
        self.screen.blit(addr_text, (self.cam_telemetry_rect.left + 40, self.cam_telemetry_rect.top + 110))
        
        # Connection Metrics
        fps_val = self.stream_fps if self.stream_status == "CONNECTED" else 0
        size_val = self.stream_frame_size if self.stream_status == "CONNECTED" else 0.0
        bw_val = size_val * fps_val
        
        fps_text = self.font_body.render(f"SENSOR FPS    : {fps_val} FPS", True, COLOR_CYAN)
        size_text = self.font_body.render(f"FRAME SIZE    : {size_val:.1f} KB", True, COLOR_CYAN)
        bw_text = self.font_body.render(f"BANDWIDTH     : {bw_val:.1f} KB/s", True, COLOR_CYAN)
        
        self.screen.blit(fps_text, (self.cam_telemetry_rect.left + 40, self.cam_telemetry_rect.top + 140))
        self.screen.blit(size_text, (self.cam_telemetry_rect.left + 40, self.cam_telemetry_rect.top + 170))
        self.screen.blit(bw_text, (self.cam_telemetry_rect.left + 40, self.cam_telemetry_rect.top + 200))
        
        # 4. Draw Terminal/Log panel inside telemetry dashboard
        log_panel_rect = pygame.Rect(self.cam_telemetry_rect.left + 20, self.cam_telemetry_rect.top + 250, 420, 200)
        draw_cyber_rect(self.screen, COLOR_GRID, log_panel_rect, cut_size=8, thickness=1, fill=True)
        pygame.draw.rect(self.screen, COLOR_MUTED, log_panel_rect, 1)
        
        # Rolling logs text
        log_y = log_panel_rect.top + 12
        for log in self.stream_logs:
            log_surf = self.font_sm.render(log, True, COLOR_GREEN)
            self.screen.blit(log_surf, (log_panel_rect.left + 15, log_y))
            log_y += 15
            
        # Flashing Live Indicator
        if pygame.time.get_ticks() % 1000 < 500:
            rec_color = COLOR_MAGENTA if self.stream_status == "CONNECTED" else COLOR_MUTED
            pygame.draw.circle(self.screen, rec_color, (self.cam_feed_rect.right - 90, self.cam_feed_rect.top + 20), 6)
            rec_text = self.font_body.render("LIVE REC", True, rec_color)
            self.screen.blit(rec_text, (self.cam_feed_rect.right - 80, self.cam_feed_rect.top + 10))

        # 5. Draw Buttons
        self.cam_back_btn.draw(self.screen, self.font_body)
        self.cam_record_btn.draw(self.screen, self.font_body)
        self.cam_discover_btn.draw(self.screen, self.font_body)
        self.cam_reconnect_btn.draw(self.screen, self.font_body)

    def draw_video_analysis(self):
        # Grab latest data from the processing thread
        with self.va_lock:
            bgr   = self.va_latest_bgr
            stats = dict(self.va_stats)
        pos      = self.va_pos
        total    = self.va_total
        src_fps  = self.va_source_fps
        proc_fps = self.va_disp_fps
        paused   = self.va_paused

        # ── Title ────────────────────────────────────────────────────────────
        title_surf = self.font_header.render("MONITORING PROTOCOL: VIDEO ANALYSIS", True, COLOR_CYAN)
        self.screen.blit(title_surf, (20, 14))

        # ── Video feed area ───────────────────────────────────────────────────
        pygame.draw.rect(self.screen, COLOR_CARD, self.va_feed_rect)

        if bgr is not None:
            fw, fh = self.va_feed_rect.width, self.va_feed_rect.height
            resized = cv2.resize(bgr, (fw, fh))
            rgb_arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(rgb_arr.swapaxes(0, 1))
            self.screen.blit(frame_surf, self.va_feed_rect.topleft)

            # Red flashing overlay on accident
            if stats.get("accident"):
                tick = pygame.time.get_ticks()
                if tick % 800 < 400:
                    flash = pygame.Surface((fw, fh), pygame.SRCALPHA)
                    flash.fill((255, 0, 0, 28))
                    self.screen.blit(flash, self.va_feed_rect.topleft)
        else:
            # Loading placeholder
            loading_surf = self.font_header.render(
                "LOADING MODELS..." if not self.video_analyzer.ready else "INITIALISING...",
                True, COLOR_MAGENTA
            )
            self.screen.blit(loading_surf, loading_surf.get_rect(center=self.va_feed_rect.center))

        draw_pixel_frame(self.screen, self.va_feed_rect, COLOR_CYAN, COLOR_GREEN)

        # FPS tag top-left of feed
        fps_tag = self.font_sm.render(
            f"PROC {proc_fps:.1f} FPS  |  SRC {src_fps:.0f} FPS", True, COLOR_GREEN
        )
        self.screen.blit(fps_tag, (self.va_feed_rect.x + 10, self.va_feed_rect.y + 8))

        # PAUSED badge centre of feed
        if paused:
            p_surf = self.font_header.render("|| PAUSED", True, COLOR_YELLOW)
            pr = p_surf.get_rect(center=(self.va_feed_rect.centerx, self.va_feed_rect.y + 36))
            pygame.draw.rect(self.screen, COLOR_BG, pr.inflate(20, 8))
            self.screen.blit(p_surf, pr)

        # END OF VIDEO badge
        if self.va_ended:
            end_surf = self.font_header.render("END OF VIDEO", True, COLOR_MAGENTA)
            er = end_surf.get_rect(center=self.va_feed_rect.center)
            pygame.draw.rect(self.screen, COLOR_CARD, er.inflate(24, 12))
            self.screen.blit(end_surf, er)

        # ── Analytics panel ───────────────────────────────────────────────────
        self._draw_va_analytics(stats)

        # ── Timeline scrubber ─────────────────────────────────────────────────
        self._draw_va_timeline(pos, total)

        # ── Timestamp + frame counter ─────────────────────────────────────────
        elapsed_s = pos   / max(src_fps, 1)
        total_s   = total / max(src_fps, 1)
        ts_str = (
            f"{int(elapsed_s)//60:02d}:{int(elapsed_s)%60:02d}"
            f"  /  {int(total_s)//60:02d}:{int(total_s)%60:02d}"
            f"     FRAME {pos} / {total}"
        )
        ts_surf = self.font_body.render(ts_str, True, COLOR_WHITE)
        self.screen.blit(ts_surf, (500, 566))

        # ── Buttons ───────────────────────────────────────────────────────────
        self.va_back_btn.draw(self.screen,  self.font_body)
        self.va_pause_btn.draw(self.screen, self.font_body)
        self.va_record_btn.draw(self.screen, self.font_body)

    def _draw_va_analytics(self, stats):
        px, py = self.va_panel_rect.x, self.va_panel_rect.y
        pw     = self.va_panel_rect.width

        draw_cyber_rect(self.screen, COLOR_MUTED, self.va_panel_rect, cut_size=14, thickness=2)

        # Header
        hdr = self.font_header.render("ANALYTICS", True, COLOR_CYAN)
        self.screen.blit(hdr, (px + 20, py + 16))
        pygame.draw.line(
            self.screen, COLOR_MUTED,
            (px + 10, py + 46), (px + pw - 10, py + 46), 1
        )

        lx   = px + 20
        cy   = py + 58
        rh   = 26       # row height
        vcol = lx + 148 # value column x

        def row(label, value, color=COLOR_WHITE):
            nonlocal cy
            self.screen.blit(self.font_body.render(label, True, COLOR_MUTED), (lx, cy))
            self.screen.blit(self.font_body.render(str(value), True, color),  (vcol, cy))
            cy += rh

        def section(title):
            nonlocal cy
            self.screen.blit(self.font_sm.render(title, True, COLOR_MUTED), (lx, cy))
            cy += 18

        def divider():
            nonlocal cy
            pygame.draw.line(
                self.screen, (55, 22, 75),
                (lx, cy), (px + pw - 10, cy), 1
            )
            cy += 10

        counts   = stats.get("counts",  {})
        accident = stats.get("accident")
        sev      = stats.get("severity")

        # Vehicles
        section("VEHICLES")
        total_v = 0
        for lbl in ("Car", "Truck", "Bus", "Motorcycle"):
            n = counts.get(lbl, 0)
            total_v += n
            row(lbl.upper(), n, COLOR_CYAN if n > 0 else COLOR_MUTED)
        row("TOTAL", total_v, COLOR_YELLOW)
        divider()

        # Pedestrians
        section("PEDESTRIANS")
        ppl = counts.get("Person", 0)
        row("COUNT", ppl, COLOR_YELLOW if ppl else COLOR_MUTED)
        divider()

        # Accident
        section("ACCIDENT")
        if accident is None:
            row("STATUS", "SCANNING...", COLOR_MUTED)
        elif accident:
            row("DETECTED", "YES", COLOR_MAGENTA)
            row("CONFIDENCE", f"{stats.get('accident_conf', 0):.0%}", COLOR_MAGENTA)
        else:
            row("DETECTED", "NO", COLOR_GREEN)
            row("CONFIDENCE", f"{stats.get('accident_conf', 0):.0%}", COLOR_MUTED)
        divider()

        # Severity
        section("SEVERITY")
        sev_labels = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        sev_colors = {1: COLOR_GREEN, 2: COLOR_YELLOW, 3: COLOR_MAGENTA}
        if sev and accident:
            row("LEVEL", sev_labels.get(sev, str(sev)), sev_colors.get(sev, COLOR_WHITE))
            if stats.get("severity_heuristic"):
                hint = self.font_sm.render("(density estimate)", True, COLOR_MUTED)
                self.screen.blit(hint, (lx, cy))
                cy += 16
        else:
            row("LEVEL", "N/A", COLOR_MUTED)
        divider()

        # Total objects
        row("ALL OBJECTS", sum(counts.values()), COLOR_WHITE)

    def _draw_va_timeline(self, pos, total):
        tr = self.va_timeline_rect
        mx, my = pygame.mouse.get_pos()
        hovering = tr.collidepoint(mx, my) or self.va_scrubbing

        # Track background
        pygame.draw.rect(self.screen, (28, 12, 42), tr)
        border_col = COLOR_CYAN if hovering else COLOR_MUTED
        pygame.draw.rect(self.screen, border_col, tr, 1)

        if total > 0:
            fill_w = int(tr.width * pos / total)
            if fill_w > 0:
                pygame.draw.rect(
                    self.screen, COLOR_CYAN,
                    (tr.x, tr.y, fill_w, tr.height)
                )

            # Hover ghost position
            if hovering:
                ghost_x = max(tr.x, min(mx, tr.right))
                ghost_w = ghost_x - tr.x
                ghost   = pygame.Surface((ghost_w, tr.height), pygame.SRCALPHA)
                ghost.fill((0, 243, 255, 40))
                self.screen.blit(ghost, tr.topleft)

            # Playhead knob
            knob_x = tr.x + max(0, min(fill_w, tr.width))
            pygame.draw.circle(self.screen, COLOR_WHITE, (knob_x, tr.centery), 7)
            pygame.draw.circle(self.screen, COLOR_CYAN,  (knob_x, tr.centery), 5)

            # Hover time-stamp tooltip
            if hovering:
                hov_frac = (mx - tr.x) / tr.width
                hov_s    = hov_frac * (total / max(self.va_source_fps, 1))
                tip_str  = f"{int(hov_s)//60:02d}:{int(hov_s)%60:02d}"
                tip      = self.font_sm.render(tip_str, True, COLOR_CYAN)
                tip_x    = max(tr.x, min(mx - tip.get_width() // 2, tr.right - tip.get_width()))
                self.screen.blit(tip, (tip_x, tr.y - 16))

        # "SEEK" label left of bar
        seek_lbl = self.font_sm.render("SEEK", True, COLOR_MUTED)
        self.screen.blit(seek_lbl, (tr.x - 0, tr.y - 16))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if self.state in ["MAIN_MENU", "OPTIONS"]:
                if self.gear_button.handle_event(event):
                    continue

            if self.state == "MAIN_MENU":
                for btn in self.main_buttons:
                    if btn.handle_event(event):
                        break
            elif self.state == "SETTINGS":
                for btn in self.settings_buttons:
                    if btn.handle_event(event):
                        break
            elif self.state == "OPTIONS":
                if self.opt_back_btn.handle_event(event):
                    continue
                if not self.is_carla_installed() and self.download_carla_btn.handle_event(event):
                    continue
                for btn in self.option_buttons:
                    if btn.handle_event(event):
                        break
            elif self.state == "LIVE_STREAM":
                if self.cam_back_btn.handle_event(event):
                    continue
                if self.cam_record_btn.handle_event(event):
                    continue
                if self.cam_discover_btn.handle_event(event):
                    continue
                if self.cam_reconnect_btn.handle_event(event):
                    continue
            elif self.state == "VIDEO_ANALYSIS":
                # Timeline scrubbing — check before buttons so clicks on the
                # timeline bar don't also trigger button actions.
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.va_timeline_rect.collidepoint(event.pos):
                        self.va_scrubbing = True
                        self._va_seek_from_mouse(event.pos[0])
                        continue
                elif event.type == pygame.MOUSEMOTION:
                    if self.va_scrubbing:
                        self._va_seek_from_mouse(event.pos[0])
                        continue
                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.va_scrubbing:
                        self.va_scrubbing = False
                        continue

                # Keyboard shortcuts
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_video_pause()
                    elif event.key == pygame.K_ESCAPE:
                        self.stop_video_analysis()

                # Buttons
                if self.va_back_btn.handle_event(event):
                    continue
                if self.va_pause_btn.handle_event(event):
                    continue
                if self.va_record_btn.handle_event(event):
                    continue
            elif self.state == "CARLA_SIMULATION":
                if self.sim_back_btn.handle_event(event):
                    continue
                if self.sim_accident_btn.handle_event(event):
                    continue
                if self.sim_record_btn.handle_event(event):
                    continue

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = Application()
    app.run()
