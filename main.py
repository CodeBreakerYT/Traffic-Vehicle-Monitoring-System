import os
import sys
import pygame
import tkinter as tk
from tkinter import filedialog
import socket
import struct
import threading
import time
import cv2
import numpy as np

# Import modular components
import config
from ui_widgets import (
    COLOR_BG, COLOR_GRID, COLOR_CYAN, COLOR_MAGENTA, COLOR_GREEN, 
    COLOR_YELLOW, COLOR_WHITE, COLOR_MUTED, COLOR_CARD, Button, GearButton, 
    draw_cyber_rect, draw_pixel_frame
)
from preview import SimulationPreview

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

        # 4. LIVE MONITORING WIDGETS
        self.cam_feed_rect = pygame.Rect(40, 100, 720, 480)
        self.cam_telemetry_rect = pygame.Rect(780, 100, 460, 480)
        self.cam_back_btn = Button((780, 600, 210, 46), "< BACK", self.stop_live_stream, COLOR_MUTED, COLOR_CYAN)
        self.cam_reconnect_btn = Button((1030, 600, 210, 46), "RECONNECT", self.reconnect_live_stream, COLOR_GREEN, COLOR_CYAN)

        # Preview module placed above options (Y=86 to Y=386)
        self.preview_module = SimulationPreview((374, 86, 533, 300))

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
        self.running = False

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
        self.state = "OPTIONS"

    def reconnect_live_stream(self):
        self.log_stream("Resetting interface socket binding...")
        self.stream_shutdown.set()
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=0.5)
        self.start_live_stream()

    def action_live_hookup(self):
        self.start_live_stream()

    def action_video_analysis(self):
        print("[ACTION] Triggering raw video analysis...")
        if tk_root:
            selected_file = filedialog.askopenfilename(
                title="Select Raw Video File",
                filetypes=[("Video Files", "*.mp4 *.gif *.avi *.mkv")]
            )
            if selected_file:
                print(f"[ACTION] Loaded video file: {selected_file}")

    def action_carla_simulation(self):
        print("[ACTION] Launching CARLA Simulation process...")
        carla_dir = self.config_data.get("carla_path", "")
        if not carla_dir or not os.path.isdir(carla_dir):
            print("[WARNING] CARLA directory invalid or not set.")
            self.go_to_settings()
        else:
            print(f"[ACTION] Found valid CARLA directory: {carla_dir}")

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

            # Determine which GIF to display in the preview module
            # Default is 1.gif
            target_gif = "assets/gif/1.gif"
            
            # Hover over Button 1 (Connect Live Camera) -> 1.gif
            if self.option_buttons[0].hovered:
                target_gif = "assets/gif/1.gif"
            # Hover over Button 2 (Process Video) -> 2.gif
            elif self.option_buttons[1].hovered:
                target_gif = "assets/gif/2.gif"
            # Hover over Button 3 (CARLA Simulation) -> 3.gif
            elif self.option_buttons[2].hovered:
                target_gif = "assets/gif/3.gif"
                
            self.preview_module.set_active_gif(target_gif)
            self.preview_module.update()
        elif self.state == "LIVE_STREAM":
            self.cam_back_btn.update(mouse_pos)
            self.cam_reconnect_btn.update(mouse_pos)

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

        if self.state in ["MAIN_MENU", "OPTIONS"]:
            self.gear_button.draw(self.screen)

        self.draw_scanlines()
        pygame.display.flip()

    def draw_main_menu(self):
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
        self.cam_reconnect_btn.draw(self.screen, self.font_body)

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
                for btn in self.option_buttons:
                    if btn.handle_event(event):
                        break
            elif self.state == "LIVE_STREAM":
                if self.cam_back_btn.handle_event(event):
                    continue
                if self.cam_reconnect_btn.handle_event(event):
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
