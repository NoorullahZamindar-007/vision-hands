"""
Hand Portal Effect Switcher
----------------------------
A portal window is stretched between the INDEX FINGERTIPS of both hands
(each fingertip is one diagonal corner; the box is the axis-aligned
bounding rectangle between the two points). The region inside gets
replaced with a processed version of itself (thermal, edge-mesh,
posterize, chromatic, etc). Pinching (thumb tip + index tip touching) on
EITHER hand cycles to the next effect exactly once per pinch — hold it as
long as you want, it only fires again after you release and pinch again.

Rendering rules that are permanent:
  - The effect content is NEVER perspective-warped. The box's bounding
    region is cropped, processed completely straight/undistorted, pasted
    back, then masked to the portal shape. Any future tilt is handled by
    masking/clipping only.
  - Pinch-to-switch is architecturally separate from the box-building:
    it reads raw thumb/index landmarks directly and uses an armed/disarmed
    edge-trigger (no time-based cooldown).

Visuals: feathered portal edge, pulsing rim-glow, corner bracket accents,
energy particles, cinematic vignette, and a designed HUD with an animated
effect label, FPS display, and controls hint.

Controls:
  q            - quit
  s            - screenshot (incrementing filename)
  1..8         - jump directly to a specific effect
  +/-          - adjust pinch sensitivity (threshold printed to console)
  f            - toggle fullscreen
  d            - toggle downscaled hand detection (faster, but slightly
                 less accurate; off by default)
  v            - toggle vignette
  Pinch        - cycle effect (thumb + index touch, either hand, once per pinch)
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import random
import subprocess
import time
import atexit

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from handportal.cli import parse_args
from handportal.effects import EFFECT_NAMES, EFFECTS
from handportal.gestures import PinchTrigger, pinch_distance as gesture_pinch_distance
from handportal.model import ensure_model

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = ImageDraw = ImageFont = None

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

PINCH_DIST_THRESHOLD = 0.055   # pinch entry threshold; release uses a wider threshold

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Landmark indices we care about
THUMB_TIP = 4
INDEX_TIP = 8

# ---- Theme (BGR) -----------------------------------------------------
ACCENT         = (255, 170, 64)    # electric cyan
ACCENT_VIOLET  = (230, 70, 190)    # violet, for the slow color pulse
TEXT_HI        = (240, 244, 248)   # near-white
TEXT_LO        = (150, 160, 175)   # muted slate
PANEL_FILL     = (24, 26, 38)      # HUD panel background
PANEL_EDGE     = (58, 78, 100)     # HUD panel hairline

# ---- Transitions -----------------------------------------------------
CROSSFADE_S    = 0.22    # effect crossfade duration (seconds)
PORTAL_FADE_S  = 0.30    # portal fade in/out time constant
LABEL_ANIM_S   = 0.30    # effect-name label slide/fade duration

# ---- HUD ---------------------------------------------------------------
HINT_SHOW_S    = 5.0     # controls hint stays visible for this long...
HINT_FADE_S    = 1.5     # ...then fades out over this many seconds

# ---- Portal glow -----------------------------------------------------
GLOW_PAD       = 26      # how far the glow extends beyond the box edge
GLOW_SIGMA     = 9.0     # bloom softness

# ---- Particles -------------------------------------------------------
MAX_PARTICLES  = 150

# ---- Window / performance --------------------------------------------
FULLSCREEN       = True   # open fullscreen (toggle with 'f' at runtime)
DETECT_DOWNSCALE = False  # toggle with 'd' at runtime (off = max accuracy)
VIGNETTE         = True   # toggle with 'v' at runtime


def _detect_screen_resolution():
    """Best-effort screen resolution via multiple fallbacks (Linux/macOS/Win).
    Returns (width, height) or None."""
    # 1. tkinter — works on most desktops and is in the stdlib
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        sw, sh = _root.winfo_screenwidth(), _root.winfo_screenheight()
        _root.destroy()
        if sw > 0 and sh > 0:
            return (sw, sh)
    except Exception:
        pass
    # 2. xrandr (Linux/X11)
    try:
        out = subprocess.check_output(["xrandr"], stderr=subprocess.DEVNULL, timeout=3).decode()
        for line in out.splitlines():
            if "*" in line:  # the line with the asterisk is the current mode
                res = line.split()[0]  # e.g. "1920x1080"
                parts = res.split("x")
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    # 3. xdpyinfo (Linux/X11)
    try:
        out = subprocess.check_output(["xdpyinfo"], stderr=subprocess.DEVNULL, timeout=3).decode()
        for line in out.splitlines():
            if "dimensions:" in line:
                # e.g. "  dimensions:    1920x1080 pixels ..."
                token = line.split()[1]  # "1920x1080"
                parts = token.split("x")
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    # 4. Hard fallback — common 16:9 resolution
    return (1920, 1080)


def ensure_model_downloaded():
    """Compatibility wrapper around the verified, atomic downloader."""
    return ensure_model(MODEL_PATH)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def landmark_px(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


class QuadSmoother:
    """
    Exponential moving average smoother for the 4 quad points, to remove
    frame-to-frame jitter from raw MediaPipe landmarks. Lower `alpha` =
    smoother but more lag; higher `alpha` = snappier but more jitter.
    Handles hand-count changes gracefully by resetting state.
    """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.smoothed = None

    def update(self, quad_pts):
        if self.smoothed is None or self.smoothed.shape != quad_pts.shape:
            self.smoothed = quad_pts.copy()
        else:
            self.smoothed = self.alpha * quad_pts + (1 - self.alpha) * self.smoothed
        return self.smoothed

    def reset(self):
        self.smoothed = None


def build_quad_from_index_fingers(hand_a, hand_b, w, h):
    """
    Rectangle box driven ONLY by index fingertip of each hand (no thumb).
    Matches the reference: one index fingertip is the top corner, the
    other is the opposite (diagonal) bottom corner — like dragging a
    selection box between two diagonal points. The box is simply the
    axis-aligned bounding rectangle of those two points.

    Whichever hand happens to be higher (smaller y) forms the top corner
    on its own side (left or right, whichever x it's on); the other hand
    forms the opposite corner automatically since it's just a bounding box.

    Returns points ordered TL, TR, BR, BL for consistent use downstream.
    """
    tip_a = landmark_px(hand_a[INDEX_TIP], w, h)
    tip_b = landmark_px(hand_b[INDEX_TIP], w, h)

    x_min, x_max = sorted([tip_a[0], tip_b[0]])
    y_min, y_max = sorted([tip_a[1], tip_b[1]])

    # small floor so the box doesn't collapse to a sliver if hands are
    # briefly at the same height
    if (y_max - y_min) < 20:
        mid = (y_max + y_min) / 2
        y_min, y_max = mid - 10, mid + 10
    if (x_max - x_min) < 20:
        mid = (x_max + x_min) / 2
        x_min, x_max = mid - 10, mid + 10

    tl = np.array([x_min, y_min], dtype=np.float32)
    tr = np.array([x_max, y_min], dtype=np.float32)
    br = np.array([x_max, y_max], dtype=np.float32)
    bl = np.array([x_min, y_max], dtype=np.float32)

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----------------------------------------------------------------------
# Small color helpers
# ----------------------------------------------------------------------
def _dim(color, f):
    """Scale a BGR color toward black by factor f (0..1)."""
    return tuple(max(0, min(255, int(round(c * f)))) for c in color)


def _lerp_color(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _pil_rgb(color, a=255):
    """BGR -> PIL RGBA fill tuple."""
    return (int(color[2]), int(color[1]), int(color[0]), int(a))


# ----------------------------------------------------------------------
# Portal rendering
# ----------------------------------------------------------------------
FEATHER_PX = 8   # mask edge softened by a few pixels (blend margin)
FEATHER_SIGMA = 2.0  # Gaussian sigma of the feather


def render_portal_into_quad(frame, effect_fn, quad_pts,
                            prev_effect_fn=None, blend=1.0, alpha=1.0):
    """
    Composite `effect_fn` output inside the quad region.

    Deliberately does NOT use cv2.warpPerspective on the processed image
    (that was the cause of the old squished/distorted-face look). The box's
    bounding region is cropped, the effect is applied to that crop COMPLETELY
    UNWARPED/UNDISTORTED, the result is pasted back at the same position and
    masked to the quad's shape. This rule is permanent — never warp the
    processed pixels, even if the box is ever tilted.

    New vs the original version:
      - The crop is inflated by FEATHER_PX so the feathered mask band falls
        INSIDE the crop (a blur clamped at the crop boundary would kill the
        soft edge for axis-aligned boxes). The mask is Gaussian-softened a
        few pixels, so the portal blends into the frame instead of a
        razor-hard cutout.
      - `prev_effect_fn` + `blend` crossfade between the old and new effect
        during a switch (blend = weight of `effect_fn`, 0..1).
      - `alpha` (0..1) gates the whole composite, used for the portal
        fade-in / fade-out.
      - The composite is pure uint8 OpenCV math (no slow fancy indexing).

    Edits `frame` in place.
    """
    h, w = frame.shape[:2]
    x_min = max(int(math.floor(np.min(quad_pts[:, 0]))) - FEATHER_PX, 0)
    x_max = min(int(math.ceil(np.max(quad_pts[:, 0]))) + FEATHER_PX, w)
    y_min = max(int(math.floor(np.min(quad_pts[:, 1]))) - FEATHER_PX, 0)
    y_max = min(int(math.ceil(np.max(quad_pts[:, 1]))) + FEATHER_PX, h)

    if x_max - x_min < 5 or y_max - y_min < 5:
        return

    # crop is a view — every effect returns a fresh array, never writes input
    crop = frame[y_min:y_max, x_min:x_max]
    processed = effect_fn(crop)
    if prev_effect_fn is not None:
        prev = prev_effect_fn(crop)
        processed = cv2.addWeighted(processed, float(blend), prev, 1.0 - float(blend), 0)

    # feathered mask: fill the quad shape, then soften the edge a few px
    rh, rw = y_max - y_min, x_max - x_min
    mask = np.zeros((rh, rw), dtype=np.uint8)
    local = quad_pts.astype(np.float32) - np.array([x_min, y_min], dtype=np.float32)
    cv2.fillConvexPoly(mask, local.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=FEATHER_SIGMA)

    if alpha < 1.0:
        # scale the whole mask by the portal alpha (uint8 fast path)
        mask = cv2.multiply(mask, np.array([int(round(255 * alpha))], dtype=np.uint8),
                            scale=1.0 / 255.0)

    # masked blend in uint8 arithmetic (alpha = mask/255)
    mask3 = cv2.merge([mask, mask, mask])
    region = frame[y_min:y_max, x_min:x_max]
    out = cv2.add(cv2.multiply(processed, mask3, scale=1.0 / 255.0),
                  cv2.multiply(region, 255 - mask3, scale=1.0 / 255.0))
    frame[y_min:y_max, x_min:x_max] = out


def draw_portal_glow(frame, quad_pts, color, intensity):
    """
    Soft additive rim-glow around the portal edge: a wide Gaussian halo
    plus a thin bright DoG "core" band hugging the outline. No hard
    outline is drawn — this reads as an energy window, not a border.
    `intensity` 0..1 scales the bloom.
    """
    h, w = frame.shape[:2]
    x_min = max(int(math.floor(np.min(quad_pts[:, 0]))) - GLOW_PAD, 0)
    x_max = min(int(math.ceil(np.max(quad_pts[:, 0]))) + GLOW_PAD, w)
    y_min = max(int(math.floor(np.min(quad_pts[:, 1]))) - GLOW_PAD, 0)
    y_max = min(int(math.ceil(np.max(quad_pts[:, 1]))) + GLOW_PAD, h)

    if x_max - x_min < 5 or y_max - y_min < 5:
        return

    rw, rh = x_max - x_min, y_max - y_min
    local = quad_pts.astype(np.float32) - np.array([x_min, y_min], dtype=np.float32)
    fill = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(fill, local.astype(np.int32), 255)

    # Wide bloom computed at 1/4 scale (same physical radius, ~1/16 the cost),
    # then upsampled. rim = blur(fill) - fill = soft glow OUTSIDE the edge;
    # core = a DoG of that = a thin bright band hugging the outline.
    ds = 0.25
    sw, sh = max(1, int(round(rw * ds))), max(1, int(round(rh * ds)))
    small = cv2.resize(fill, (sw, sh), interpolation=cv2.INTER_AREA)
    halo_s = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, GLOW_SIGMA * ds))
    rim_s = cv2.subtract(halo_s, small)
    core_s = cv2.subtract(rim_s, cv2.GaussianBlur(rim_s, (0, 0), sigmaX=1.3))
    rim = cv2.resize(rim_s, (rw, rh), interpolation=cv2.INTER_LINEAR)
    core = cv2.resize(core_s, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # tint in uint8 arithmetic (avoids float64 promotion) and additively blend
    col = np.array(color, dtype=np.uint8)
    rim3 = cv2.multiply(cv2.merge([rim, rim, rim]), col, scale=1.0 / 255.0)
    core3 = cv2.multiply(cv2.merge([core, core, core]), col, scale=1.0 / 255.0)
    glow = cv2.addWeighted(rim3, 0.55, core3, 1.0, 0)
    glow = cv2.multiply(glow, np.array([max(1, int(round(255 * intensity)))] * 3, np.uint8),
                        scale=1.0 / 255.0)

    frame[y_min:y_max, x_min:x_max] = cv2.add(frame[y_min:y_max, x_min:x_max], glow)


def draw_corner_accents(frame, quad_pts, color, t, alpha=1.0, length=16):
    """
    Minimal AR-target style corner brackets at each of the 4 corners,
    oriented toward the box center, with a gentle per-corner shimmer.
    `alpha` fades them with the portal.
    """
    cx = float(np.mean(quad_pts[:, 0]))
    cy = float(np.mean(quad_pts[:, 1]))
    for i, pt in enumerate(quad_pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        dx = 1 if x < cx else -1
        dy = 1 if y < cy else -1
        shimmer = 0.78 + 0.22 * math.sin(t * 3.1 + i * 1.7)
        L = int(length * shimmer * alpha)
        if L < 3:
            continue
        # soft halo dot behind each bracket
        cv2.circle(frame, (x, y), 5, _dim(color, 0.30 * alpha), -1, cv2.LINE_AA)
        # crisp brackets
        cv2.line(frame, (x, y), (x + dx * L, y), _dim(color, 0.92 * alpha), 2, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * L), _dim(color, 0.92 * alpha), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Particles (pure decoration — never touches box/mask logic)
# ----------------------------------------------------------------------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x, y, vx, vy, life, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size


def spawn_particles(particles, quad_pts, count):
    if len(particles) >= MAX_PARTICLES:
        return
    corners = quad_pts
    for _ in range(count):
        c = corners[random.randrange(4)]
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(16, 70)
        particles.append(Particle(
            float(c[0]), float(c[1]),
            math.cos(ang) * spd, math.sin(ang) * spd,
            random.uniform(0.45, 1.0),
            random.uniform(0.25, 0.5)))


def update_particles(particles, dt):
    drag = math.exp(-1.8 * dt)
    alive = []
    for p in particles:
        p.life -= dt
        if p.life <= 0:
            continue
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.vx *= drag
        p.vy *= drag
        alive.append(p)
    return alive


def draw_particles(frame, particles, color):
    for p in particles:
        a = max(0.0, p.life / p.max_life)
        r = max(1, int(round(p.size * (2.2 + 3.0 * a))))
        cv2.circle(frame, (int(p.x), int(p.y)), r, _dim(color, 0.65 * a), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Vignette
# ----------------------------------------------------------------------
def make_vignette_layer(w, h, strength=0.30, inner=0.45):
    """Pre-computed radial vignette as a uint8 BGR layer (0 = black)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    d = np.sqrt(nx * nx + ny * ny)
    t = np.clip((d - inner) / (1.6 - inner), 0.0, 1.0)
    v = 1.0 - strength * t * t
    layer = np.stack([v] * 3, axis=-1)
    return (layer * 255.0).astype(np.uint8)


# ----------------------------------------------------------------------
# Pinch detection (completely independent of the box-building system)
# ----------------------------------------------------------------------
def pinch_distance(hand_landmarks_list):
    """Returns normalized thumb-index distance for one hand."""
    return gesture_pinch_distance(hand_landmarks_list, THUMB_TIP, INDEX_TIP)


# ----------------------------------------------------------------------
# HUD — designed panels, PIL typography with an OpenCV fallback
# ----------------------------------------------------------------------
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _find_font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


FONT_PATH = _find_font_path()
_FONT_CACHE = {}


def _font(size, bold=False):
    """
    Returns a font of `size`. Prefers a real TTF (bold when available);
    falls back to Pillow's scalable default font (ImageFont.load_default
    supports a size on Pillow >= 10.1), so the designed HUD still renders
    correctly on systems with no font files installed.
    """
    if not PIL_AVAILABLE:
        return None
    key = (size, bold)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            try:
                f = ImageFont.load_default(size)
            except TypeError:  # very old Pillow without size support
                f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return f


def _text_width(text, font):
    if not PIL_AVAILABLE or font is None:
        return int(8 * len(text))
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def blend_panel(frame, panel, x, y, alpha=1.0):
    """Alpha-composite a small RGBA numpy panel onto the BGR frame.
    `alpha` (0..1) scales the panel's opacity (used to fade the hint out)."""
    if panel is None or alpha <= 0.0:
        return
    ph, pw = panel.shape[:2]
    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x) + pw, frame.shape[1])
    y1 = min(int(y) + ph, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    a = panel[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x), 3:4].astype(np.float32) / 255.0
    a *= float(alpha)
    rgb = panel[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x), :3].astype(np.float32)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (rgb * a + region * (1.0 - a)).astype(np.uint8)


def build_effect_panel(effect_idx, name, from_name, anim_p, accent):
    """
    Minimal HUD element: accent effect number + the effect name, with the
    slide/fade switch animation. That's all that's shown on screen — the
    pinch calibration data is deliberately not displayed.
    Returns an RGBA numpy array, or None if PIL is unavailable.
    """
    if not PIL_AVAILABLE:
        return None
    font_num = _font(13, bold=True)
    font_name = _font(28, bold=True)
    name_w = _text_width(name, font_name)
    from_w = _text_width(from_name, font_name) if from_name else 0
    W = max(300, 70 + max(name_w, from_w))
    H = 62
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=13,
                        fill=_pil_rgb(PANEL_FILL, 178), outline=_pil_rgb(PANEL_EDGE, 110), width=1)
    d.text((18, H // 2), f"{effect_idx + 1:02d}", font=font_num,
           fill=_pil_rgb(accent, 240), anchor="lm")
    y_name = H // 2
    if from_name is not None:
        fade = int(210 * (1.0 - anim_p))
        if fade > 2:
            d.text((52 - int(12 * anim_p), y_name), from_name,
                   font=font_name, fill=_pil_rgb(TEXT_HI, fade), anchor="lm")
        d.text((52 + int(12 * (1.0 - anim_p)), y_name), name,
               font=font_name, fill=_pil_rgb(TEXT_HI, int(255 * anim_p)), anchor="lm")
    else:
        d.text((52, y_name), name, font=font_name, fill=_pil_rgb(TEXT_HI, 255), anchor="lm")
    return np.array(img)


def build_fps_pill(fps):
    """Tiny unobtrusive FPS readout, top-right corner."""
    if not PIL_AVAILABLE:
        return None
    txt = f"{fps:3.0f} FPS"
    font = _font(12, bold=True)
    W = _text_width(txt, font) + 28
    H = 24
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(PANEL_FILL, 135), outline=_pil_rgb(PANEL_EDGE, 75), width=1)
    d.text((14, H // 2), txt, font=font, fill=_pil_rgb(TEXT_LO, 235), anchor="lm")
    return np.array(img)


def build_hint_panel():
    """Short controls hint; fades out a few seconds after launch."""
    if not PIL_AVAILABLE:
        return None
    txt = "pinch: switch  |  1-8: jump  |  f: fullscreen  |  q: quit"
    font = _font(12)
    W = _text_width(txt, font) + 36
    H = 28
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=14,
                        fill=_pil_rgb(PANEL_FILL, 150), outline=_pil_rgb(PANEL_EDGE, 90), width=1)
    d.text((18, H // 2), txt, font=font, fill=_pil_rgb(TEXT_LO, 240), anchor="lm")
    return np.array(img)


def _hud_fallback(frame, effect_name, fps):
    """
    Minimal single-accent OpenCV HUD (only used if PIL is unavailable):
    effect name + tiny FPS, nothing else.
    """
    w = frame.shape[1]
    cv2.putText(frame, effect_name, (16, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 118, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_LO, 1, cv2.LINE_AA)


def _fill_window(frame, win_w, win_h):
    """
    Resize `frame` to exactly fill the window (win_w x win_h) while keeping
    the frame's aspect ratio: uniform scale, then a symmetric center crop of
    whatever overflows. The image is NEVER stretched and no letterbox bars
    appear. On a matching-aspect display the crop is exactly zero, so a 16:9
    camera on a 16:9 monitor fills the screen 1:1.
    """
    fh, fw = frame.shape[:2]
    if win_w <= 0 or win_h <= 0 or fw <= 0 or fh <= 0:
        return frame
    if fw == win_w and fh == win_h:
        return frame
    scale = max(win_w / fw, win_h / fh)          # cover: fills both dimensions
    nw = int(math.ceil(fw * scale))
    nh = int(math.ceil(fh * scale))
    if nw != fw or nh != fh:
        scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        scaled = frame
    x0 = (nw - win_w) // 2
    y0 = (nh - win_h) // 2
    return scaled[y0:y0 + win_h, x0:x0 + win_w]


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main(argv=None):
    global CAM_INDEX, FRAME_W, FRAME_H, FULLSCREEN, DETECT_DOWNSCALE, VIGNETTE, MODEL_PATH
    args = parse_args(argv)
    CAM_INDEX, FRAME_W, FRAME_H = args.camera, args.width, args.height
    FULLSCREEN = not args.windowed
    DETECT_DOWNSCALE = args.downscale
    VIGNETTE = not args.no_vignette
    if args.model:
        MODEL_PATH = os.path.abspath(os.path.expanduser(args.model))
    try:
        ensure_model_downloaded()
    except RuntimeError as exc:
        print(f"Startup error: {exc}")
        return 1

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"Could not open webcam {CAM_INDEX}. Try --camera with another index.")
        return 1
    atexit.register(cap.release)

    # Ask for a 16:9 feed so the fullscreen fill is a 1:1 pixel map on 16:9
    # monitors (zero crop, zero stretch). Some backends ignore the request,
    # so read back what we actually got and fall back to other common 16:9
    # modes if the primary request didn't stick.
    for want_w, want_h in ((FRAME_W, FRAME_H), (1920, 1080), (960, 540), (640, 360)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if got_w > 0 and got_h > 0 and abs(got_w / got_h - want_w / want_h) < 0.02:
            break
    print(f"Camera feed: {got_w}x{got_h}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    try:
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        cap.release()
        print(f"Could not initialize MediaPipe: {exc}")
        return 1
    atexit.register(landmarker.close)
    atexit.register(cv2.destroyAllWindows)

    # ---- app state ---------------------------------------------------
    effect_idx = 0
    shot_count = 0
    pinch_trigger = PinchTrigger(args.pinch_threshold, min(0.22, args.pinch_threshold + 0.015))
    quad_smoother = QuadSmoother(alpha=0.35)

    detect_downscale = DETECT_DOWNSCALE
    vignette_enabled = VIGNETTE

    # transition state
    crossfade_prev_idx = None   # effect being faded out of
    crossfade_start = 0.0
    label_from = None           # effect name sliding out
    label_anim_start = 0.0
    portal_alpha = 0.0          # 0 = portal hidden, 1 = fully visible
    last_quad = None            # last known quad, kept for fade-out
    particles = []

    # timing / fps
    t0 = time.perf_counter()
    last_time = t0
    fps_accum = 0.0
    fps_frames = 0
    fps = 30.0

    hint_panel = build_hint_panel() if PIL_AVAILABLE else None
    fps_pill = build_fps_pill(fps) if PIL_AVAILABLE else None
    vignette_layer = make_vignette_layer(FRAME_W, FRAME_H)  # precomputed once

    cv2.namedWindow("Hand Portal", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow("Hand Portal", FRAME_W, FRAME_H)

    screen_res = _detect_screen_resolution()   # fallback for getWindowImageRect
    if screen_res:
        print(f"Detected screen: {screen_res[0]}x{screen_res[1]}")

    fullscreen = FULLSCREEN
    window_target = None     # (w, h) the video is scaled to, to fill the window
    window_check_t = 0.0     # when the window rect was last queried
    aspect_noted = False     # one-time notice if camera/screen aspects differ
    if fullscreen:
        try:
            cv2.waitKey(1)  # let the window initialize before going fullscreen
            cv2.setWindowProperty("Hand Portal", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            # Seed window_target immediately so the very first frame is filled
            if screen_res:
                window_target = screen_res
        except Exception:
            fullscreen = False

    print("Controls: 'q' quit | 's' screenshot | '1-8' jump to effect | '+/-' pinch sensitivity "
          "| 'f' fullscreen | 'd' downscaled detection | 'v' vignette "
          "| pinch (thumb+index, either hand) to switch effect")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped returning frames; closing HandPortal.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.perf_counter()
        dt = min(now - last_time, 0.1)
        last_time = now
        t = now - t0

        def do_switch(new_idx):
            nonlocal effect_idx, crossfade_prev_idx, crossfade_start, label_from, label_anim_start
            if new_idx == effect_idx:
                return
            crossfade_prev_idx = effect_idx
            crossfade_start = now
            label_from = EFFECT_NAMES[effect_idx]
            label_anim_start = now
            effect_idx = new_idx

        # ---- hand detection (optionally downscaled; normalized landmark
        #      coords are scale-independent, so the full-res math is unchanged)
        scale = 0.5 if detect_downscale else 1.0
        if scale < 1.0:
            rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale)),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode wants a monotonically increasing timestamp in ms
        timestamp_ms = int((now - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ---- pinch + box (independent systems, as required) -----------
        pinch_distances = []
        quad_pts = None

        hands_list = result.hand_landmarks  # list of list-of-21-landmarks

        if len(hands_list) == 2:
            hand_a, hand_b = hands_list
            raw_quad = build_quad_from_index_fingers(hand_a, hand_b, w, h)
            quad_pts = quad_smoother.update(raw_quad)

            for hand in (hand_a, hand_b):
                d = pinch_distance(hand)
                pinch_distances.append(d)

        elif len(hands_list) == 1:
            # Still check single-hand pinch even if portal isn't formed
            quad_smoother.reset()
            for hand in hands_list:
                d = pinch_distance(hand)
                pinch_distances.append(d)

        else:
            quad_smoother.reset()

        # Edge-triggered switch: fire exactly once when pinch is first detected,
        # then require release (distance back above threshold) before it can
        # fire again. (No time-based cooldown — armed/disarmed state machine.)
        if pinch_trigger.update(pinch_distances):
            do_switch((effect_idx + 1) % len(EFFECTS))

        # ---- crossfade + label animation progress ---------------------
        if crossfade_prev_idx is not None:
            cross_p = min(1.0, (now - crossfade_start) / CROSSFADE_S)
            if cross_p >= 1.0:
                crossfade_prev_idx = None
        else:
            cross_p = 1.0

        if label_from is not None:
            label_p = min(1.0, (now - label_anim_start) / LABEL_ANIM_S)
            if label_p >= 1.0:
                label_from = None
        else:
            label_p = 1.0

        # ---- portal fade in / out + scale-in --------------------------
        target_alpha = 1.0 if quad_pts is not None else 0.0
        k = 1.0 - math.exp(-dt / PORTAL_FADE_S)
        portal_alpha += (target_alpha - portal_alpha) * k
        if quad_pts is not None:
            last_quad = quad_pts
        elif portal_alpha < 0.01:
            last_quad = None

        pulse = 0.5 + 0.5 * math.sin(t * 1.25)
        glow_color = _lerp_color(ACCENT, ACCENT_VIOLET, pulse)

        if last_quad is not None and portal_alpha > 0.005:
            quad = last_quad
            # gentle scale-in while the portal materializes (mask-only;
            # the effect pixels are never warped)
            grow = 0.92 + 0.08 * min(1.0, portal_alpha)
            if grow != 1.0:
                c = quad.mean(axis=0)
                quad = c + (quad - c) * grow

            prev_fn = EFFECTS[crossfade_prev_idx] if crossfade_prev_idx is not None else None
            render_portal_into_quad(frame, EFFECTS[effect_idx], quad,
                                    prev_effect_fn=prev_fn, blend=cross_p,
                                    alpha=portal_alpha)

            draw_portal_glow(frame, quad, glow_color,
                             (0.5 + 0.28 * math.sin(t * 2.1)) * portal_alpha)
            draw_corner_accents(frame, quad, glow_color, t, alpha=portal_alpha)

            if portal_alpha > 0.4:
                spawn_particles(particles, quad, 2 if random.random() < 0.7 else 1)

        particles = update_particles(particles, dt)
        draw_particles(frame, particles, glow_color)

        # ---- vignette --------------------------------------------------
        if vignette_enabled:
            if vignette_layer is None or vignette_layer.shape[:2] != (h, w):
                vignette_layer = make_vignette_layer(w, h)
            frame = cv2.multiply(frame, vignette_layer, scale=1.0 / 255.0)

        # ---- FPS -------------------------------------------------------
        fps_accum += dt
        fps_frames += 1
        if fps_accum >= 0.25:
            fps = fps_frames / fps_accum
            fps_frames = 0
            fps_accum = 0.0
            if PIL_AVAILABLE:
                fps_pill = build_fps_pill(fps)

        # ---- fill the window edge-to-edge (aspect-preserving, no bars) -----
        if now - window_check_t > 0.5:
            window_check_t = now
            try:
                rect = cv2.getWindowImageRect("Hand Portal")
            except Exception:
                rect = None
            if rect is not None and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                window_target = (rect[2], rect[3])
            elif fullscreen and screen_res:
                # getWindowImageRect failed — use detected screen resolution
                window_target = screen_res
            if window_target is not None and not aspect_noted:
                f_aspect = w / max(h, 1)
                w_aspect = window_target[0] / max(window_target[1], 1)
                if abs(f_aspect - w_aspect) > 0.03:
                    print("Note: camera aspect != screen aspect; the image is filled "
                          "with a symmetric edge-crop (no stretching, no bars). "
                          "A 16:9 camera/mode gives a perfect fill.")
                    aspect_noted = True
        if window_target is not None:
            frame = _fill_window(frame, window_target[0], window_target[1])

        # ---- HUD drawn AFTER the window fill, at display coordinates -------
        # (so the label/FPS/hint are never cropped by the center-crop, and stay
        # crisp instead of being upscaled with the video)
        dw, dh = frame.shape[1], frame.shape[0]
        if PIL_AVAILABLE:
            effect_panel = build_effect_panel(effect_idx, EFFECT_NAMES[effect_idx],
                                              label_from, label_p, glow_color)
            blend_panel(frame, effect_panel, 16, 16)
            if fps_pill is not None:
                blend_panel(frame, fps_pill, dw - fps_pill.shape[1] - 16, 16)
            # controls hint fades away a few seconds after launch
            hint_alpha = 1.0 - max(0.0, (t - HINT_SHOW_S) / HINT_FADE_S)
            if hint_panel is not None and hint_alpha > 0.02:
                blend_panel(frame, hint_panel, (dw - hint_panel.shape[1]) // 2,
                            dh - hint_panel.shape[0] - 14, alpha=hint_alpha)
        else:
            _hud_fallback(frame, EFFECT_NAMES[effect_idx], fps)

        cv2.imshow("Hand Portal", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_count += 1
            fname = f"portal_shot_{time.strftime('%Y%m%d-%H%M%S')}-{shot_count}.png"
            if cv2.imwrite(fname, frame):
                print(f"Saved {fname}")
            else:
                print(f"Could not save screenshot to {os.path.abspath(fname)}")
        elif key in (ord('+'), ord('=')):
            pinch_trigger.adjust(0.005)
            print(f"Pinch thresholds: {pinch_trigger.enter_threshold:.3f}/{pinch_trigger.release_threshold:.3f}")
        elif key == ord('-'):
            pinch_trigger.adjust(-0.005)
            print(f"Pinch thresholds: {pinch_trigger.enter_threshold:.3f}/{pinch_trigger.release_threshold:.3f}")
        elif ord('1') <= key <= ord('9'):
            n = key - ord('1')
            if n < len(EFFECTS):
                do_switch(n)
                print(f"Effect: {EFFECT_NAMES[effect_idx]}")
        elif key == ord('d'):
            detect_downscale = not detect_downscale
            print(f"Downscaled detection: {'ON' if detect_downscale else 'OFF'}")
        elif key == ord('v'):
            vignette_enabled = not vignette_enabled
            print(f"Vignette: {'ON' if vignette_enabled else 'OFF'}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            try:
                if fullscreen:
                    cv2.setWindowProperty("Hand Portal", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                    if screen_res:
                        window_target = screen_res
                else:
                    cv2.setWindowProperty("Hand Portal", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Hand Portal", FRAME_W, FRAME_H)
                    window_target = None
            except Exception:
                pass
            window_check_t = 0.0   # re-measure the window size immediately
            print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
