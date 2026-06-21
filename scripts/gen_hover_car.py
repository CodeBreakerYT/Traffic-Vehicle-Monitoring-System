"""
Procedurally generates pixel-art of a futuristic hovering car for the main
menu screen. Drawn at low resolution with hand-placed primitives, then
upscaled with NEAREST resampling + palette quantization to keep the crisp,
blocky retro-pixel look (matches the existing car_ico.png aesthetic).

Run: venv/Scripts/python.exe scripts/gen_hover_car.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "menu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base canvas — small, so every brush stroke reads as a "pixel block"
W, H = 96, 56
SCALE = 8  # final image = 768x448

# Cyberpunk palette
C_BODY_DARK   = (28, 34, 56, 255)
C_BODY_MID    = (52, 60, 92, 255)
C_BODY_LIGHT  = (150, 165, 200, 255)
C_BODY_WHITE  = (225, 235, 250, 255)
C_CANOPY      = (10, 14, 24, 255)
C_CANOPY_GLOW = (0, 220, 255, 220)
C_CYAN        = (0, 243, 255, 255)
C_MAGENTA     = (255, 0, 160, 255)
C_YELLOW      = (255, 220, 0, 255)


def draw_glow_ellipse(draw, cx, cy, rx, ry, color, steps=5):
    """Soft radial glow by stacking shrinking translucent ellipses."""
    r, g, b, a = color
    for i in range(steps, 0, -1):
        frac = i / steps
        alpha = int(a * (1 - frac) * 0.5)
        draw.ellipse(
            [cx - rx * frac, cy - ry * frac, cx + rx * frac, cy + ry * frac],
            fill=(r, g, b, alpha),
        )


def main():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = W * 0.52, H * 0.46  # car center

    # ── Motion trail streaks (behind the car, suggesting forward travel) ───
    for i, (dx, dy, length, alpha) in enumerate([
        (-30, -4, 14, 90), (-32, 0, 18, 130), (-30, 4, 14, 90), (-26, 8, 10, 60)
    ]):
        x0 = cx + dx
        y0 = cy + dy
        draw.line([(x0 - length, y0), (x0, y0)], fill=(0, 243, 255, alpha), width=2)

    # ── Hover glow beneath the car (the "repulsor" effect) ─────────────────
    draw_glow_ellipse(draw, cx - 6, cy + 15, 22, 5, C_CYAN)
    draw_glow_ellipse(draw, cx + 16, cy + 15, 14, 4, C_MAGENTA)

    # ── Tail fin ─────────────────────────────────────────────────────────────
    draw.polygon(
        [(cx - 34, cy - 2), (cx - 26, cy - 14), (cx - 22, cy - 2)],
        fill=C_BODY_MID,
    )

    # ── Main body silhouette (sleek teardrop, tapered nose to the right) ───
    body_pts = [
        (cx - 30, cy + 2),    # rear-bottom
        (cx - 30, cy - 4),    # rear-top
        (cx - 18, cy - 10),   # rear roofline rise
        (cx + 2,  cy - 12),   # roof peak
        (cx + 22, cy - 8),    # canopy taper begins
        (cx + 34, cy - 2),    # nose taper
        (cx + 38, cy + 3),    # nose tip
        (cx + 30, cy + 7),    # underside front
        (cx - 4,  cy + 9),    # underside mid
        (cx - 26, cy + 7),    # underside rear
    ]
    draw.polygon(body_pts, fill=C_BODY_DARK)

    # Lighter top highlight band (gives it a glossy, rounded look)
    draw.polygon(
        [(cx - 18, cy - 9), (cx + 2, cy - 11), (cx + 20, cy - 7), (cx + 10, cy - 4), (cx - 12, cy - 5)],
        fill=C_BODY_LIGHT,
    )
    draw.line([(cx - 10, cy - 6), (cx + 16, cy - 8)], fill=C_BODY_WHITE, width=1)

    # ── Cockpit canopy (dark glass + cyan glow streak) ──────────────────────
    draw.polygon(
        [(cx - 14, cy - 8), (cx + 2, cy - 12), (cx + 16, cy - 8), (cx + 6, cy - 3), (cx - 10, cy - 3)],
        fill=C_CANOPY,
    )
    draw.line([(cx - 10, cy - 7), (cx + 10, cy - 9)], fill=C_CANOPY_GLOW, width=1)

    # ── Underside thruster strip (glowing edge along the belly) ────────────
    draw.line([(cx - 24, cy + 7), (cx + 28, cy + 6)], fill=C_CYAN, width=1)
    draw.line([(cx - 6, cy + 9), (cx + 18, cy + 7)], fill=C_MAGENTA, width=1)

    # ── Nose light ───────────────────────────────────────────────────────────
    draw.ellipse([cx + 33, cy - 1, cx + 37, cy + 2], fill=C_YELLOW)

    # ── Small sensor antenna on the roof ────────────────────────────────────
    draw.line([(cx - 2, cy - 12), (cx - 2, cy - 16)], fill=C_BODY_LIGHT, width=1)
    draw.ellipse([cx - 3, cy - 18, cx - 1, cy - 16], fill=C_CYAN)

    # ── Upscale with crisp NEAREST resampling to keep the blocky pixel look ─
    big = img.resize((W * SCALE, H * SCALE), Image.NEAREST)

    out_path = OUT_DIR / "hover_car_pixel.png"
    big.save(out_path)
    print(f"Saved {out_path} ({big.size[0]}x{big.size[1]})")


if __name__ == "__main__":
    main()
