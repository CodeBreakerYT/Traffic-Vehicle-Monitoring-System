import os
import sys
import pygame
import tkinter as tk
from tkinter import filedialog

# Import modular components
import config
from ui_widgets import (
    COLOR_BG, COLOR_GRID, COLOR_CYAN, COLOR_MAGENTA, COLOR_GREEN, 
    COLOR_YELLOW, COLOR_WHITE, COLOR_MUTED, Button, GearButton, draw_cyber_rect
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

        # Preview module placed above options (Y=86 to Y=386)
        self.preview_module = SimulationPreview((374, 86, 533, 300))

    def go_to_main_menu(self):
        self.state = "MAIN_MENU"

    def go_to_options(self):
        self.state = "OPTIONS"

    def go_to_settings(self):
        self.temp_carla_path = self.config_data.get("carla_path", "")
        self.state = "SETTINGS"

    def quit_app(self):
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

    def save_settings(self):
        self.config_data["carla_path"] = self.temp_carla_path
        config.save_config(self.config_data)
        self.go_to_main_menu()

    def toggle_grid(self):
        self.config_data["grid_enabled"] = not self.config_data.get("grid_enabled", True)
        config.save_config(self.config_data)

    def toggle_scanlines(self):
        self.config_data["scanlines_enabled"] = not self.config_data.get("scanlines_enabled", True)
        config.save_config(self.config_data)

    def action_live_hookup(self):
        print("[ACTION] Hooking up to local camera stream...")

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
        
        path_title = self.font_body.render("CARLA SIMULATION PATH:", True, COLOR_WHITE)
        self.screen.blit(path_title, (130, 180))
        
        path_str = self.temp_carla_path or "[NOT CONFIGURED // DEFAULT WILL BE USED]"
        if len(path_str) > 75:
            path_str = "..." + path_str[-72:]
        path_color = COLOR_YELLOW if self.temp_carla_path else COLOR_MAGENTA
        path_text = self.font_body.render(path_str, True, path_color)
        self.screen.blit(path_text, (130, 210))

        help_text1 = self.font_sm.render("* Required path where CarlaUE4.exe and PythonAPI reside.", True, COLOR_MUTED)
        help_text2 = self.font_sm.render("* If not configured, simulation launcher will look in typical directories.", True, COLOR_MUTED)
        self.screen.blit(help_text1, (130, 245))
        self.screen.blit(help_text2, (130, 265))

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
