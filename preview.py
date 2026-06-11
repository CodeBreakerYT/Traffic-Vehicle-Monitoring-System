import os
import cv2
import pygame
from ui_widgets import (
    COLOR_CARD, COLOR_CYAN, COLOR_GREEN, COLOR_WHITE,
    draw_pixel_frame
)

class GifPlayer:
    def __init__(self, filepath, target_size=None):
        self.filepath = filepath
        self.target_size = target_size
        self.frames = []
        self.frame_idx = 0
        self.last_update = 0
        self.fps = 10  # Fallback frame rate
        self.load_gif()

    def load_gif(self):
        if not os.path.exists(self.filepath):
            return
        
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            return
        
        # Determine actual FPS of the GIF
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.fps = fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV decodes as BGR; convert to RGB for Pygame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit preview area if target_size is set
            if self.target_size:
                frame_rgb = cv2.resize(frame_rgb, self.target_size)
                
            # Convert frame numpy array to Pygame surface
            # Transpose axes (0, 1) because numpy shape is (height, width) but Pygame surface is (width, height)
            frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.frames.append(frame_surf)
            
        cap.release()

    def update(self, current_time):
        if not self.frames:
            return None
        
        # Calculate milliseconds per frame
        frame_delay = 1000.0 / self.fps
        if current_time - self.last_update >= frame_delay:
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            self.last_update = current_time
            
        return self.frames[self.frame_idx]

class SimulationPreview:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.active_path = None
        self.gif_player = None
        self.scan_y = self.rect.top
        self.scan_dir = 1

    def set_active_gif(self, filepath):
        """Changes the playing GIF if the path is different."""
        if self.active_path != filepath:
            self.active_path = filepath
            if os.path.exists(filepath):
                # Target dimensions match preview rect width and height
                self.gif_player = GifPlayer(filepath, (self.rect.width, self.rect.height))
            else:
                self.gif_player = None

    def update(self):
        # Update scanline sweep overlay
        self.scan_y += 1.5 * self.scan_dir
        if self.scan_y >= self.rect.bottom or self.scan_y <= self.rect.top:
            self.scan_dir *= -1

    def draw(self, surface, font_sm):
        # 1. Fill base preview area card
        pygame.draw.rect(surface, COLOR_CARD, self.rect)
        
        # 2. Draw GIF Frame if available, otherwise keep frame blank
        gif_frame = None
        if self.gif_player:
            gif_frame = self.gif_player.update(pygame.time.get_ticks())
            
        if gif_frame:
            surface.blit(gif_frame, self.rect.topleft)
            # HUD Label showing active GIF
            gif_name = os.path.basename(self.active_path).upper()
            title_text = f" [ PREVIEW PROTOCOL: {gif_name} ] "
            label_color = COLOR_GREEN
        else:
            # Draw placeholder text when blank
            title_text = " [ NO ACTIVE PREVIEW FEED ] "
            label_color = COLOR_WHITE
            placeholder = font_sm.render("AWAITING STREAM PROTOCOL...", True, (80, 90, 110))
            p_rect = placeholder.get_rect(center=self.rect.center)
            surface.blit(placeholder, p_rect)

        # 3. Draw procedural pixelated border
        draw_pixel_frame(surface, self.rect, COLOR_CYAN, COLOR_GREEN)

        # 4. Draw HUD Label on the top-left edge
        label_surf = font_sm.render(title_text, True, label_color)
        surface.blit(label_surf, (self.rect.left + 20, self.rect.top - 7))
        
        # 5. Draw animated scanline sweep overlay
        pygame.draw.line(surface, (0, 243, 255, 120), (self.rect.left, int(self.scan_y)), (self.rect.right, int(self.scan_y)), 1)
