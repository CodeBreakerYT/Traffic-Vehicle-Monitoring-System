import math
import pygame

# ==============================================================================
# COLOR SYSTEM (NEON CYBERPUNK)
# ==============================================================================
COLOR_BG = (13, 13, 18)          # Deep Charcoal/Black
COLOR_GRID = (25, 25, 38)        # Dark Indigo-Grey
COLOR_CYAN = (0, 243, 255)       # Neon Cyan
COLOR_MAGENTA = (255, 0, 127)    # Neon Magenta
COLOR_GREEN = (57, 255, 20)      # Neon Green
COLOR_YELLOW = (255, 220, 0)     # Neon Yellow
COLOR_WHITE = (230, 245, 255)    # Light Cyan-White
COLOR_MUTED = (100, 110, 140)    # Slate Muted Blue
COLOR_CARD = (22, 22, 32)        # Slightly lighter BG card

# ==============================================================================
# DRAWING UTILITIES
# ==============================================================================
def draw_cyber_rect(surface, color, rect, cut_size=12, thickness=2, fill=False, alpha=255):
    """Draws a polygon with cut corners (cyberpunk tech style)."""
    x, y, w, h = rect
    points = [
        (x + cut_size, y),
        (x + w - cut_size, y),
        (x + w, y + cut_size),
        (x + w, y + h - cut_size),
        (x + w - cut_size, y + h),
        (x + cut_size, y + h),
        (x, y + h - cut_size),
        (x, y + cut_size)
    ]
    if fill:
        if alpha < 255:
            # Create transparent surface
            temp_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            offset_points = [(p[0] - x, p[1] - y) for p in points]
            pygame.draw.polygon(temp_surf, color + (alpha,), offset_points)
            surface.blit(temp_surf, (x, y))
        else:
            pygame.draw.polygon(surface, color, points)
    else:
        pygame.draw.polygon(surface, color, points, thickness)

def draw_corner_brackets(surface, color, rect, length=15, thickness=2):
    """Draws HUD-style corner bracket lines around a rect."""
    x, y, w, h = rect
    # Top-Left
    pygame.draw.line(surface, color, (x, y), (x + length, y), thickness)
    pygame.draw.line(surface, color, (x, y), (x, y + length), thickness)
    # Top-Right
    pygame.draw.line(surface, color, (x + w, y), (x + w - length, y), thickness)
    pygame.draw.line(surface, color, (x + w, y), (x + w, y + length), thickness)
    # Bottom-Left
    pygame.draw.line(surface, color, (x, y + h), (x + length, y + h), thickness)
    pygame.draw.line(surface, color, (x, y + h), (x, y + h - length), thickness)
    # Bottom-Right
    pygame.draw.line(surface, color, (x + w, y + h), (x + w - length, y + h), thickness)
    pygame.draw.line(surface, color, (x + w, y + h), (x + w, y + h - length), thickness)

def draw_pixel_frame(surface, rect, outer_color=COLOR_CYAN, inner_color=COLOR_GREEN):
    """Draws a retro pixel-like double frame border with corner block details."""
    x, y, w, h = rect
    # 1. Outer thick rectangle
    pygame.draw.rect(surface, outer_color, rect, 3)
    
    # 2. Inner thin border offset by 6px
    inner_rect = pygame.Rect(x + 6, y + 6, w - 12, h - 12)
    pygame.draw.rect(surface, inner_color, inner_rect, 1)
    
    # 3. Square retro corner pixel blocks (8x8 pixels)
    block_size = 8
    # Top-Left
    pygame.draw.rect(surface, outer_color, (x, y, block_size, block_size))
    # Top-Right
    pygame.draw.rect(surface, outer_color, (x + w - block_size, y, block_size, block_size))
    # Bottom-Left
    pygame.draw.rect(surface, outer_color, (x, y + h - block_size, block_size, block_size))
    # Bottom-Right
    pygame.draw.rect(surface, outer_color, (x + w - block_size, y + h - block_size, block_size, block_size))

# ==============================================================================
# UI COMPONENT CLASSES
# ==============================================================================
class Button:
    def __init__(self, rect, text, action=None, color=COLOR_CYAN, accent_color=COLOR_MAGENTA):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.action = action
        self.color = color
        self.accent_color = accent_color
        self.hovered = False
        self.pulse = 0
        self.pulse_dir = 1

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        if self.hovered:
            self.pulse += 5 * self.pulse_dir
            if self.pulse >= 100 or self.pulse <= 0:
                self.pulse_dir *= -1
        else:
            self.pulse = 0

    def draw(self, surface, font):
        # Determine active colors
        draw_color = self.accent_color if self.hovered else self.color
        
        # Glow effect if hovered
        if self.hovered:
            glow_rect = (self.rect.x - 4, self.rect.y - 4, self.rect.width + 8, self.rect.height + 8)
            draw_cyber_rect(surface, draw_color, glow_rect, cut_size=14, thickness=1)
            
            # Subtle filled background glow
            bg_alpha = 25 + int(self.pulse * 0.2)
            draw_cyber_rect(surface, draw_color, self.rect, cut_size=12, fill=True, alpha=bg_alpha)

        # Draw main border
        draw_cyber_rect(surface, draw_color, self.rect, cut_size=12, thickness=2)

        # Draw text
        text_surf = font.render(self.text, True, COLOR_WHITE if not self.hovered else draw_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

        # Interactive corner markers when hovered
        if self.hovered:
            draw_corner_brackets(surface, draw_color, self.rect, length=10, thickness=2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.action()
                return True
        return False

class GearButton:
    def __init__(self, pos, action=None):
        self.rect = pygame.Rect(pos[0], pos[1], 44, 44)
        self.action = action
        self.hovered = False
        self.angle = 0.0

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        if self.hovered:
            self.angle += 3.0  # Rotate when hovered
        else:
            self.angle += 0.2

    def draw(self, surface):
        cx, cy = self.rect.center
        color = COLOR_MAGENTA if self.hovered else COLOR_CYAN
        radius = 16

        # Draw outer glowing ring if hovered
        if self.hovered:
            pygame.draw.circle(surface, COLOR_MAGENTA, (cx, cy), radius + 6, 1)

        # Draw main core
        pygame.draw.circle(surface, color, (cx, cy), radius, 2)
        pygame.draw.circle(surface, color, (cx, cy), int(radius * 0.4), 2)

        # Draw teeth
        for i in range(8):
            t_angle = math.radians(i * 45 + self.angle)
            # Inner teeth base
            x1 = cx + math.cos(t_angle) * (radius - 2)
            y1 = cy + math.sin(t_angle) * (radius - 2)
            # Outer teeth tip
            x2 = cx + math.cos(t_angle) * (radius + 5)
            y2 = cy + math.sin(t_angle) * (radius + 5)
            
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 4)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.action()
                return True
        return False
