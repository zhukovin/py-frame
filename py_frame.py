import threading
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import List, Tuple, Literal

import pygame
from PIL import Image, ImageOps
import random
from threading import Lock, Thread
from typing import Optional
from web_server import run_web

import faulthandler
import signal
import resource
import os


faulthandler.register(signal.SIGUSR1)

Orientation = Literal["P", "L"]  # P = Portrait, L = Landscape

seconds_to_display = 15

MAX_HISTORY_SCREENS = 5  # or 20, tune as you like


def log_mem(tag=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"[MEM {tag}] pid={os.getpid()} rss={usage.ru_maxrss} B")


@dataclass
class Slide:
    path: str
    surface: pygame.Surface
    orientation: Orientation  # "P" or "L"
    load_bytes: int = 0       # raw file size read from disk/NFS
    load_seconds: float = 0.0  # time spent reading those bytes (excludes decode)


class SlideshowController:
    def __init__(self):
        self.lock = Lock()
        # Current screen
        self.current_slides: list[Slide] = []
        self.current_pattern_type: Optional[int] = None
        self.current_marks: set[int] = set()

        # History
        self.history: list[tuple[list[Slide], int]] = []
        self.history_index: int = -1

        # Pending command from web: {"type": "...", "steps": int}
        self.pending_command: Optional[dict] = None

        # Exclusions
        self.excluded_paths: set[str] = set()
        self.exclusions_file: str = "exclusions.txt"

        # paused state
        self.paused: bool = False
        # black-screen mode (screen fully black while slideshow paused)
        self.black_screen: bool = False

        # Diagnostics: whether the most recent photo load attempt succeeded,
        # and the recent read speed (bytes/sec) over the last few successful loads.
        self.drive_ok: bool = True
        self.download_bytes_per_sec: Optional[float] = None

        # Diagnostics: rolling history of the last 20 load attempts (success
        # or failure) for the load-time / size histogram. Each entry is
        # {"success": bool, "bytes": int, "seconds": float}.
        self.load_history: deque = deque(maxlen=20)


def make_old_paper_surface(size):
    w, h = size
    surf = pygame.Surface(size)
    base_color = (235, 222, 193)
    surf.fill(base_color)

    # Add subtle random noise spots
    for _ in range(8000):
        x = random.randrange(w)
        y = random.randrange(h)
        noise = random.randint(-10, 10)
        r = min(max(base_color[0] + noise, 0), 255)
        g = min(max(base_color[1] + noise, 0), 255)
        b = min(max(base_color[2] + noise, 0), 255)
        surf.set_at((x, y), (r, g, b))

    return surf


def load_slide(path: str) -> Slide:
    """
    Load a slide from disk, rotate it according to EXIF orientation,
    classify it as Portrait (P) or Landscape (L), and convert to a
    pygame.Surface.
    """
    import io
    import time

    # 1) Read the raw bytes ourselves (timed) so callers can measure
    #    disk/NFS read speed separately from JPEG decode (CPU) time.
    t0 = time.monotonic()
    with open(path, "rb") as f:
        raw = f.read()
    load_seconds = time.monotonic() - t0
    load_bytes = len(raw)

    # 2) Load with Pillow from the in-memory bytes
    img = Image.open(io.BytesIO(raw))

    # 3) Apply EXIF orientation (rotates/flip as needed)
    img = ImageOps.exif_transpose(img)

    # 4) Classify after rotation
    width, height = img.size
    orientation: Orientation = "P" if height > width else "L"

    # 5) Ensure mode is suitable for pygame
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    mode = img.mode
    size = img.size
    data = img.tobytes()

    # 6) Convert to pygame surface
    if mode == "RGBA":
        surface = pygame.image.fromstring(data, size, mode).convert_alpha()
    else:  # "RGB"
        surface = pygame.image.fromstring(data, size, mode).convert()

    return Slide(
        path=path,
        surface=surface,
        orientation=orientation,
        load_bytes=load_bytes,
        load_seconds=load_seconds,
    )


# ============================================================
# Pattern extraction (up to 5 entries)
# ============================================================

def extract_pattern_from_deque(dq: deque[Slide]) -> Tuple[List[Slide], int]:
    """
    Examine up to the first 5 elements of dq.

    Find one of the patterns based on orientation:
        Type 1: PPP      (3 Portrait)
        Type 2: PPLLL    (2 Portrait, 3 Landscape)
        Type 3: PLLL     (1 Portrait, 3 Landscape)

    Remove the chosen Slide objects from dq and return them as a list
    (in the order they were encountered). Any scanned but unused items
    are pushed back to the *left* of dq in their original order.
    """
    window = list(islice(dq, 0, 5))
    if not window or window[0].orientation != "P":
        raise ValueError("Deque must start with a portrait (P) slide for pattern logic.")

    count_p = sum(1 for s in window if s.orientation == "P")
    count_l = sum(1 for s in window if s.orientation == "L")

    # Determine pattern type & needed counts
    if count_p >= 3:
        needP, needL = 3, 0
        pattern_type = 1
    elif count_p >= 2 and count_l >= 3:
        needP, needL = 2, 3
        pattern_type = 2
    elif count_p >= 1 and count_l >= 3:
        needP, needL = 1, 3
        pattern_type = 3
    else:
        raise ValueError("No valid PPP / PPLLL / PLLL pattern in first 5 entries.")

    # Select slides that satisfy the pattern, in order
    extracted: List[Slide] = []
    unused: List[Slide] = []
    p_left, l_left = needP, needL

    for slide in window:
        if slide.orientation == "P" and p_left > 0:
            extracted.append(slide)
            p_left -= 1
        elif slide.orientation == "L" and l_left > 0:
            extracted.append(slide)
            l_left -= 1
        else:
            unused.append(slide)

        if p_left == 0 and l_left == 0:
            break

    if p_left != 0 or l_left != 0:
        raise RuntimeError("Pattern selection logic failed.")

    # Remove all scanned items from dq
    for _ in range(len(window)):
        dq.popleft()

    # Put unused scanned items back on the *left* in correct order
    for slide in reversed(unused):
        dq.appendleft(slide)

    return extracted, pattern_type


# ============================================================
# Image fetcher thread
# ============================================================

def image_fetcher_thread(
        file_paths: list[str],
        dq: deque[Slide],
        lock: threading.Lock,
        not_full: threading.Condition,
        producer_done: threading.Event,
        controller: SlideshowController,
        max_size: int = 5,
):
    import time

    print("Image fetcher thread native_id:", threading.get_native_id())

    # Rolling window of (bytes, seconds) for the last few successful reads,
    # used to report a live download-speed estimate (not an all-time average).
    recent_load_stats: deque[tuple[int, float]] = deque(maxlen=5)

    try:
        if not file_paths:
            producer_done.set()
            return

        idx = 0
        n = len(file_paths)

        while True:
            path = file_paths[idx]
            idx = (idx + 1) % n
            # path = random.choice(file_paths)

            with controller.lock:
                if path in controller.excluded_paths:
                    excluded = True
                else:
                    excluded = False
            if excluded:
                time.sleep(0.3)
                continue

            with not_full:
                while len(dq) >= max_size:
                    not_full.wait()

            try:
                slide = load_slide(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                recent_load_stats.clear()
                with controller.lock:
                    controller.drive_ok = False
                    controller.download_bytes_per_sec = None
                    controller.load_history.append({"success": False, "bytes": 0, "seconds": 0.0})
                time.sleep(0.5)
                continue

            recent_load_stats.append((slide.load_bytes, slide.load_seconds))
            total_bytes = sum(b for b, _ in recent_load_stats)
            total_seconds = sum(s for _, s in recent_load_stats)
            bytes_per_sec = total_bytes / total_seconds if total_seconds > 0 else None

            with controller.lock:
                controller.drive_ok = True
                controller.download_bytes_per_sec = bytes_per_sec
                controller.load_history.append({
                    "success": True,
                    "bytes": slide.load_bytes,
                    "seconds": slide.load_seconds,
                })

            with not_full:
                dq.append(slide)
                not_full.notify_all()

    finally:
        producer_done.set()


def smoothscale_safe(img: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
    """
    Wrap pygame.transform.smoothscale so that the input surface is guaranteed
    to be 24- or 32-bit, as required by smoothscale.
    """
    # Ensure 24/32-bit surface
    if img.get_bitsize() not in (24, 32):
        # Convert to 32-bit with alpha (safe for anything)
        tmp = pygame.Surface(img.get_size(), flags=pygame.SRCALPHA, depth=32)
        tmp.blit(img, (0, 0))
        img = tmp

    return pygame.transform.smoothscale(img, size)


# ============================================================
# Rendering helpers (layouts)
# ============================================================

def blit_scaled(surface: pygame.Surface, img: pygame.Surface, target_rect: pygame.Rect):
    """Scale `img` to fit inside `target_rect` preserving aspect ratio, then blit centered."""
    iw, ih = img.get_width(), img.get_height()
    tw, th = target_rect.width, target_rect.height

    if iw == 0 or ih == 0:
        return

    scale = min(tw / iw, th / ih)
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))

    scaled = smoothscale_safe(img, (new_w, new_h))

    x = target_rect.x + (tw - new_w) // 2
    y = target_rect.y + (th - new_h) // 2
    surface.blit(scaled, (x, y))


def draw_slot_overlay(screen: pygame.Surface,
                      rect: pygame.Rect,
                      slot_index: int,
                      marked: bool,
                      font: pygame.font.Font):
    OLD_PAPER = (235, 222, 193)  # warm beige
    RED = (255, 64, 64)
    # border color
    color = RED if marked else OLD_PAPER
    pygame.draw.rect(screen, color, rect, 3)

    # label box top-left
    label = str(slot_index + 1)
    text_surf = font.render(label, True, (0, 0, 0))
    padding = 4
    box_w = text_surf.get_width() + 2 * padding
    box_h = text_surf.get_height() + 2 * padding
    box_rect = pygame.Rect(rect.x + 8, rect.y + 8, box_w, box_h)
    pygame.draw.rect(screen, color, box_rect)
    screen.blit(text_surf, (box_rect.x + padding, box_rect.y + padding))


def format_speed(bytes_per_sec: Optional[float]) -> str:
    """
    Format a bytes/sec figure with an auto-scaled unit (Bps, KBps, MBps,
    GBps), rounded to 2 decimal places using round-half-up (not Python's
    default round-half-to-even, and not raw float rounding, since both can
    give surprising results right at a .xx5 boundary).
    """
    if bytes_per_sec is None:
        return "-- Bps"

    from decimal import Decimal, ROUND_HALF_UP

    units = ["Bps", "KBps", "MBps", "GBps"]
    value = bytes_per_sec

    for i, unit in enumerate(units):
        rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if rounded < 1000 or i == len(units) - 1:
            return f"{rounded} {unit}"
        value /= 1000

    return f"{value:.2f} {units[-1]}"  # unreachable, satisfies static analysis


def draw_status_overlay(screen: pygame.Surface,
                        font: pygame.font.Font,
                        paused: bool,
                        drive_ok: bool,
                        download_bytes_per_sec: Optional[float]) -> pygame.Rect:
    """
    Draw a small diagnostics box in the bottom-right corner: Playing/Paused,
    drive mount health, and recent read speed. Drawn last (on top of
    everything, including black-screen mode) so stalls can be diagnosed
    without waiting for the next slide change.

    Returns the box's rect so callers (e.g. the load-history histogram) can
    lay out other diagnostics relative to it.
    """
    LIGHT_GRAY = (211, 211, 211)
    BLACK = (0, 0, 0)

    lines = [
        "Paused" if paused else "Playing",
        "Drive: OK" if drive_ok else "Drive: DISCONNECTED",
        format_speed(download_bytes_per_sec),
    ]

    padding = 8
    line_spacing = 2
    text_surfaces = [font.render(line, True, BLACK) for line in lines]

    box_w = max(s.get_width() for s in text_surfaces) + 2 * padding
    box_h = sum(s.get_height() for s in text_surfaces) + 2 * padding + line_spacing * (len(lines) - 1)

    screen_w, screen_h = screen.get_size()
    margin = 10
    box_rect = pygame.Rect(
        screen_w - box_w - margin,
        screen_h - box_h - margin,
        box_w,
        box_h,
    )

    pygame.draw.rect(screen, LIGHT_GRAY, box_rect)

    y = box_rect.y + padding
    for surf in text_surfaces:
        screen.blit(surf, (box_rect.x + padding, y))
        y += surf.get_height() + line_spacing

    return box_rect


def draw_load_history_overlay(screen: pygame.Surface,
                              status_box_rect: pygame.Rect,
                              load_history: list):
    """
    Draw a dark-gray histogram strip to the left of the status box, spanning
    the rest of the available width. Each of the last 20 load attempts gets
    a slot: a green bar (load time), blue bar (size), and magenta bar (that
    file's own load speed) side by side, each scaled relative to the max of
    its own metric across the current window, or a single full-height red
    bar if that attempt failed. Newest is on the right, oldest on the left;
    slots are right-aligned so the strip fills in from the right before the
    window has 20 entries.
    """
    DARK_GRAY = (60, 60, 60)
    GREEN = (60, 200, 60)
    BLUE = (70, 140, 220)
    MAGENTA = (220, 60, 220)
    RED = (220, 60, 60)

    margin = 10
    gap = 10
    rect = pygame.Rect(
        margin,
        status_box_rect.y,
        status_box_rect.x - gap - margin,
        status_box_rect.height,
    )

    if rect.width <= 0:
        return

    pygame.draw.rect(screen, DARK_GRAY, rect)

    num_slots = 20
    slot_w = rect.width / num_slots
    top_padding = 6
    max_bar_height = rect.height - top_padding

    successful = [h for h in load_history if h["success"]]
    max_seconds = max((h["seconds"] for h in successful), default=0)
    max_bytes = max((h["bytes"] for h in successful), default=0)
    max_speed = max(
        (h["bytes"] / h["seconds"] for h in successful if h["seconds"] > 0),
        default=0,
    )

    empty_slots = num_slots - len(load_history)

    for i, entry in enumerate(load_history):
        slot_index = empty_slots + i
        slot_x = rect.x + slot_index * slot_w

        if not entry["success"]:
            bar_rect = pygame.Rect(
                int(slot_x + slot_w * 0.05),
                rect.y + top_padding,
                int(slot_w * 0.9),
                int(max_bar_height),
            )
            pygame.draw.rect(screen, RED, bar_rect)
            continue

        time_height = int(max_bar_height * (entry["seconds"] / max_seconds)) if max_seconds > 0 else 0
        green_rect = pygame.Rect(
            int(slot_x + slot_w * 0.05),
            rect.bottom - time_height,
            int(slot_w * 0.28),
            time_height,
        )
        pygame.draw.rect(screen, GREEN, green_rect)

        size_height = int(max_bar_height * (entry["bytes"] / max_bytes)) if max_bytes > 0 else 0
        blue_rect = pygame.Rect(
            int(slot_x + slot_w * 0.36),
            rect.bottom - size_height,
            int(slot_w * 0.28),
            size_height,
        )
        pygame.draw.rect(screen, BLUE, blue_rect)

        entry_speed = entry["bytes"] / entry["seconds"] if entry["seconds"] > 0 else 0
        speed_height = int(max_bar_height * (entry_speed / max_speed)) if max_speed > 0 else 0
        magenta_rect = pygame.Rect(
            int(slot_x + slot_w * 0.67),
            rect.bottom - speed_height,
            int(slot_w * 0.28),
            speed_height,
        )
        pygame.draw.rect(screen, MAGENTA, magenta_rect)


def build_blurred_background(screen_size, slide_rects):
    """
    screen_size: (W, H)
    slide_rects: list of (surface, rect) for each image in its final position

    Returns a pygame.Surface with a blurred, image-derived background that
    covers the whole screen and fills gaps smoothly.
    """
    W, H = screen_size
    temp = pygame.Surface((W, H)).convert()

    # Base fill: use average-ish color from first slide if available,
    # so even areas with no rectangles have a nice tint.
    if slide_rects:
        first_surf = slide_rects[0][0]
        # scale to 1x1 to get a crude average color
        tiny = smoothscale_safe(first_surf, (1, 1))
        avg_color = tiny.get_at((0, 0))[:3]
    else:
        avg_color = (0, 0, 0)

    temp.fill(avg_color)

    # IMPORTANT CHANGE:
    # For the background we stretch images to completely fill their rects
    # (no aspect ratio preservation), so there are no local "holes".
    for slide_surface, rect in slide_rects:
        # Stretch to the full rect size
        if rect.width > 0 and rect.height > 0:
            stretched = smoothscale_safe(
                slide_surface,
                (rect.width, rect.height)
            )
            temp.blit(stretched, rect.topleft)

    # Now blur by downscaling and upscaling
    factor = 8  # tweak for blur strength & performance
    small_w = max(1, W // factor)
    small_h = max(1, H // factor)

    small = smoothscale_safe(temp, (small_w, small_h))
    blurred = smoothscale_safe(small, (W, H))

    return blurred


def render_single_landscape(screen: pygame.Surface,
                            slide: Slide,
                            background: Optional[pygame.Surface],
                            font: pygame.font.Font,
                            marks: set[int]):
    if background is not None:
        screen.blit(background, (0, 0))
    else:
        screen.fill((0, 0, 0))

    rect = screen.get_rect()
    blit_scaled(screen, slide.surface, rect)

    # slot index 0 always
    draw_slot_overlay(screen, rect, 0, (0 in marks), font)


def compute_pattern_rects(screen: pygame.Surface, slides: List[Slide], pattern_type: int) -> List[
    tuple[pygame.Surface, pygame.Rect]]:
    """
    Render slides according to pattern type:
      1: PPP   -> 3 portrait images side-by-side, full screen (3 columns)
      2: PPLLL -> Column 1: 3 L stacked; Column 2 & 3: 2 P full-height
      3: PLLL  -> Column 3: P full-height; Columns 1-2: top L spans both,
                              bottom two L's share the bottom half (one in col1, one in col2)
    """
    W, H = screen.get_width(), screen.get_height()
    col_w = W // 3
    rects: List[tuple[pygame.Surface, pygame.Rect]] = []

    if pattern_type == 1:
        # PPP: 3 images → columns 0,1,2 full-height
        for idx, slide in enumerate(slides[:3]):
            rect = pygame.Rect(idx * col_w, 0, col_w, H)
            rects.append((slide.surface, rect))

    elif pattern_type == 2:
        # PPLLL: 3 L stacked in first column, 2 P full-height in columns 2 and 3
        Ls = [s for s in slides if s.orientation == "L"]
        Ps = [s for s in slides if s.orientation == "P"]

        # Column 1: 3 L stacked
        if len(Ls) >= 3:
            h3 = H // 3
            for i in range(3):
                rect = pygame.Rect(
                    0,
                    i * h3,
                    col_w,
                    h3 if i < 2 else H - 2 * h3
                )
                rects.append((Ls[i].surface, rect))

        # Columns 2 & 3: P full-height
        if len(Ps) >= 1:
            rects.append((Ps[0].surface, pygame.Rect(col_w, 0, col_w, H)))
        if len(Ps) >= 2:
            rects.append((Ps[1].surface, pygame.Rect(2 * col_w, 0, col_w, H)))

    elif pattern_type == 3:
        # PLLL: 1 P, 3 L
        Ls = [s for s in slides if s.orientation == "L"]
        Ps = [s for s in slides if s.orientation == "P"]
        if not Ps or not Ls:
            return rects

        # Column 3: P full-height
        rects.append((Ps[0].surface, pygame.Rect(2 * col_w, 0, col_w, H)))

        # Top L spans columns 1+2, top half
        rects.append((Ls[0].surface, pygame.Rect(0, 0, 2 * col_w, H // 2)))

        # Bottom two Ls share bottom half in columns 1 & 2
        if len(Ls) >= 2:
            rects.append((Ls[1].surface, pygame.Rect(0, H // 2, col_w, H // 2)))
        if len(Ls) >= 3:
            rects.append((Ls[2].surface, pygame.Rect(col_w, H // 2, col_w, H // 2)))

    return rects


def render_pattern(screen: pygame.Surface,
                   slides: list[Slide],
                   pattern_type: int,
                   background: Optional[pygame.Surface],
                   font: pygame.font.Font,
                   marks: set[int]):
    if background is not None:
        screen.blit(background, (0, 0))
    else:
        screen.fill((0, 0, 0))

    rects = compute_pattern_rects(screen, slides, pattern_type)
    # print("marks", marks)
    for idx, (surf, rect) in enumerate(rects):
        blit_scaled(screen, surf, rect)
        draw_slot_overlay(screen, rect, idx, (idx in marks), font)


def load_exclusions(controller: SlideshowController):
    """
    Populate controller.excluded_paths from controller.exclusions_file, if it exists.
    Called once at startup so exclusions persist across restarts.
    """
    if not os.path.exists(controller.exclusions_file):
        return

    with open(controller.exclusions_file, "r") as f:
        for line in f:
            path = line.strip()
            if path:
                controller.excluded_paths.add(path)


def reclassify_pattern_type(slides: List[Slide], original_ptype: int) -> Optional[int]:
    """
    Decide how a history entry should be classified after some of its slides
    were removed due to exclusion.

    Patterns 0 (solo slide) and 1 (PPP) degrade gracefully as slides are
    removed, so they keep their original type. Patterns 2 (PPLLL) and 3
    (PLLL) depend on specific P/L proportions to avoid blank gaps in
    compute_pattern_rects, so they are re-derived from what's left, using the
    same thresholds as extract_pattern_from_deque. Returns None if nothing
    valid remains and the entry should be dropped.
    """
    if not slides:
        return None

    if original_ptype in (0, 1):
        return original_ptype

    count_p = sum(1 for s in slides if s.orientation == "P")
    count_l = sum(1 for s in slides if s.orientation == "L")

    if count_p >= 3:
        return 1
    elif count_p >= 2 and count_l >= 3:
        return 2
    elif count_p >= 1 and count_l >= 3:
        return 3
    elif len(slides) == 1:
        return 0
    else:
        return None


def finalize_exclusions(controller: SlideshowController):
    """
    Take currently marked slots from the *current* screen,
    add their paths to excluded_paths + exclusions file,
    and also clean up history so excluded images never appear again
    (even when navigating back with Prev).
    """
    # If there is no current screen, just clear marks and return
    if not controller.current_slides:
        with controller.lock:
            controller.current_marks.clear()
        return

    # Grab and clear marks under lock
    with controller.lock:
        marked_indices = list(controller.current_marks)
        controller.current_marks.clear()

        if not marked_indices:
            # nothing was marked, no exclusions to process
            return

        # 1) Add marked slide paths to excluded_paths
        new_paths: list[str] = []
        for i in marked_indices:
            if 0 <= i < len(controller.current_slides):
                path = controller.current_slides[i].path
                if path not in controller.excluded_paths:
                    controller.excluded_paths.add(path)
                    new_paths.append(path)

        # 2) Clean up history: remove any slides whose path is now excluded,
        #    and reclassify (or drop) entries that no longer fit their
        #    pattern's required P/L composition.
        if controller.history:
            new_history: list[tuple[list[Slide], int]] = []
            for slides, ptype in controller.history:
                filtered = [s for s in slides if s.path not in controller.excluded_paths]
                new_ptype = reclassify_pattern_type(filtered, ptype)
                if new_ptype is not None:
                    new_history.append((filtered, new_ptype))

            controller.history = new_history

            # Fix history_index so it stays in range, or becomes -1 if history is empty
            if controller.history:
                controller.history_index = min(
                    controller.history_index,
                    len(controller.history) - 1,
                    )
            else:
                controller.history_index = -1

    # 3) Append newly excluded paths to the exclusions file (outside the lock)
    if new_paths:
        with open(controller.exclusions_file, "a") as f:
            for p in new_paths:
                f.write(p + "\n")


def downscale_slide_to_screen(slide: Slide, max_w: int, max_h: int):
    """
    Ensure slide.surface fits within (max_w, max_h).
    If it's larger, downscale it (preserving aspect ratio) IN-PLACE.
    This reduces memory usage for history and current slides.
    """
    surf = slide.surface
    w, h = surf.get_width(), surf.get_height()

    # Already small enough
    if w <= max_w and h <= max_h:
        return

    # Compute scale factor to fit within screen
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    slide.surface = smoothscale_safe(surf, (new_w, new_h))


def downscale_slides_to_screen(slides: list[Slide], max_w: int, max_h: int):
    for s in slides:
        downscale_slide_to_screen(s, max_w, max_h)


def render_loop(
        dq: deque[Slide],
        lock: threading.Lock,
        not_full: threading.Condition,
        producer_done: threading.Event,
        controller: SlideshowController,
        seconds_to_display: int = 15,
):
    import time
    import os
    import platform

    # Must set env var BEFORE pygame.init()
    if platform.system() == "Darwin":  # macOS
        # Put window at top-left of primary display
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

    pygame.init()

    if platform.system() == "Darwin":
        # On macOS: use a borderless window the size of the main desktop,
        # so other monitors remain usable.
        display_sizes = pygame.display.get_desktop_sizes()
        main_w, main_h = display_sizes[0]  # assume first is main screen
        screen = pygame.display.set_mode((main_w, main_h), pygame.NOFRAME)
    else:
        # On Pi (and other platforms) keep true fullscreen
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont(None, 40)
    status_font = pygame.font.SysFont(None, 46)

    current_slides: list[Slide] = []
    current_pattern_type: Optional[int] = None
    current_background: Optional[pygame.Surface] = None
    current_end_time: float = 0.0

    last_marks: set[int] = set()
    first_run = True
    prev_is_night: Optional[bool] = None  # for schedule transitions
    last_status_render_time: float = 0.0
    STATUS_REFRESH_SECONDS = 1.0

    running = True
    while running:
        now = time.time()

        # --- Time-based schedule: 22:00–07:00 => screen off ---
        lt = time.localtime(now)
        hour = lt.tm_hour
        is_night = (hour >= 22 or hour < 7)

        if prev_is_night is None:
            prev_is_night = is_night

        if is_night != prev_is_night:
            # We just crossed the boundary (day -> night or night -> day)
            if is_night:
                # Entering night: auto screen_off + pause
                with controller.lock:
                    controller.black_screen = True
                    controller.paused = True
            else:
                # Leaving night: auto screen_on + resume
                with controller.lock:
                    controller.black_screen = False
                    controller.paused = False
            prev_is_night = is_night

        # --- Handle keyboard / ESC ---
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                continue
            elif event.type == pygame.QUIT:
                running = False
                continue

        # --- Take pending web command + snapshot marks + black_screen ---
        with controller.lock:
            cmd = controller.pending_command
            controller.pending_command = None
            paused = controller.paused
            black_screen = controller.black_screen
            current_marks_snapshot = set(controller.current_marks)
            drive_ok = controller.drive_ok
            download_bytes_per_sec = controller.download_bytes_per_sec
            load_history_snapshot = list(controller.load_history)

        # Detect mark changes (compare snapshots)
        marks_changed = (current_marks_snapshot != last_marks)
        last_marks = current_marks_snapshot

        force_next = False
        force_prev = False
        steps = 1

        if cmd:
            ctype = cmd.get("type")
            if ctype == "next":
                force_next = True
                steps = max(1, int(cmd.get("steps", 1)))
            elif ctype == "prev":
                force_prev = True
                steps = max(1, int(cmd.get("steps", 1)))
            elif ctype == "pause":
                with controller.lock:
                    controller.paused = True
                paused = True
            elif ctype == "play":
                with controller.lock:
                    controller.paused = False
                paused = False
            elif ctype == "screen_off":
                with controller.lock:
                    controller.black_screen = True
                    controller.paused = True
                black_screen = True
                paused = True
            elif ctype == "screen_on":
                with controller.lock:
                    controller.black_screen = False
                    controller.paused = False
                black_screen = False
                paused = False

        # --- Decide if slideshow should advance ---
        need_advance = False
        backward = False

        if force_prev:
            need_advance = True
            backward = True
        elif force_next:
            need_advance = True
            backward = False
        elif (not current_slides) or (now >= current_end_time and not paused and not black_screen):
            # auto-advance only if not paused and not in black-screen mode
            need_advance = True
            backward = False

        need_to_render = False

        if first_run:
            need_to_render = True
            first_run = False

        if marks_changed and not black_screen:
            need_to_render = True

        # If we just turned screen_on/off, we should redraw
        if cmd and cmd.get("type") in ("screen_off", "screen_on"):
            need_to_render = True

        # Refresh the status overlay periodically even if nothing else
        # changed, so drive/speed issues show up promptly instead of
        # waiting for the next slide change.
        if now - last_status_render_time >= STATUS_REFRESH_SECONDS:
            need_to_render = True

        # --- Slide switching logic ---
        if need_advance:
            print("needs redraw")
            finalize_exclusions(controller)

            if backward:
                # history backwards
                with controller.lock:
                    if controller.history:
                        controller.history_index = max(
                            0,
                            controller.history_index - steps
                            if controller.history_index >= 0
                            else len(controller.history) - 1 - steps
                        )
                        idx = controller.history_index
                        slides, ptype = controller.history[idx]
                    else:
                        slides, ptype = [], None

                current_slides = slides
                current_pattern_type = ptype

                if current_slides and current_pattern_type is not None:
                    # (optional) downscale to screen size if you've added that helper
                    screen_w, screen_h = screen.get_size()
                    downscale_slides_to_screen(current_slides, screen_w, screen_h)

                    if current_pattern_type == 0:
                        rects = [(current_slides[0].surface, screen.get_rect())]
                    else:
                        rects = compute_pattern_rects(
                            screen, current_slides, current_pattern_type
                        )

                    current_background = build_blurred_background(
                        screen.get_size(), rects
                    )
                    current_end_time = now + seconds_to_display

                    with controller.lock:
                        controller.current_slides = current_slides
                        controller.current_pattern_type = current_pattern_type

                need_to_render = True

            else:  # --- Forward direction ---
                with controller.lock:
                    hist_len = len(controller.history)
                    idx = controller.history_index

                if hist_len > 0 and 0 <= idx < hist_len - 1:
                    new_index = min(hist_len - 1, idx + steps)
                    with controller.lock:
                        controller.history_index = new_index
                        slides, ptype = controller.history[new_index]

                    current_slides = slides
                    current_pattern_type = ptype

                else:
                    # need a new pattern from deque
                    with not_full:
                        if len(dq) == 0 and producer_done.is_set():
                            running = False
                            continue
                        elif len(dq) >= 5:
                            first = dq[0]
                            if first.orientation == "L":
                                slide = dq.popleft()
                                not_full.notify_all()
                                current_slides = [slide]
                                current_pattern_type = 0
                            else:
                                slides, ptype = extract_pattern_from_deque(dq)
                                not_full.notify_all()
                                current_slides = slides
                                current_pattern_type = ptype
                        else:
                            # not enough images -> keep current screen
                            continue

                    if current_slides and current_pattern_type is not None:
                        with controller.lock:
                            # Enforce max history size
                            if len(controller.history) >= MAX_HISTORY_SCREENS:
                                # Drop the oldest entry
                                controller.history.pop(0)
                                # Adjust index because we removed index 0
                                controller.history_index = max(0, controller.history_index - 1)
                            controller.history.append((current_slides, current_pattern_type))
                            controller.history_index = len(controller.history) - 1

                need_to_render = True

                # --- Build blurred background for new screen ---
                if current_slides and current_pattern_type is not None:
                    screen_w, screen_h = screen.get_size()
                    downscale_slides_to_screen(current_slides, screen_w, screen_h)

                    if current_pattern_type == 0:
                        rects = [(current_slides[0].surface, screen.get_rect())]
                    else:
                        rects = compute_pattern_rects(
                            screen, current_slides, current_pattern_type
                        )

                    current_background = build_blurred_background(
                        screen.get_size(), rects
                    )
                    current_end_time = now + seconds_to_display

                    with controller.lock:
                        controller.current_slides = current_slides
                        controller.current_pattern_type = current_pattern_type

        # --- Render only when needed ---
        if need_to_render:
            print("Rendering")

            if black_screen:
                # Show pure black, regardless of slides
                screen.fill((0, 0, 0))
            else:
                if current_slides and current_pattern_type is not None:
                    with controller.lock:
                        marks_copy = set(controller.current_marks)

                    if current_pattern_type == 0:
                        render_single_landscape(
                            screen,
                            current_slides[0],
                            current_background,
                            font,
                            marks_copy,
                        )
                    else:
                        render_pattern(
                            screen,
                            current_slides,
                            current_pattern_type,
                            current_background,
                            font,
                            marks_copy,
                        )
                else:
                    # no slides, just clear
                    screen.fill((0, 0, 0))

            # Diagnostics overlay: drawn last, on top of everything
            # (including black-screen mode) so stalls are visible.
            status_box_rect = draw_status_overlay(screen, status_font, paused, drive_ok, download_bytes_per_sec)
            draw_load_history_overlay(screen, status_box_rect, load_history_snapshot)

            pygame.display.flip()
            last_status_render_time = now

        if need_advance:
            log_mem("after_advance")

        time.sleep(0.2)   # keeps CPU load low without hammering the Pi

    pygame.quit()


# ============================================================
# Main entry
# ============================================================


def read_file_list(list_path: str) -> List[str]:
    paths: List[str] = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Only keep JPG/JPEG
            if line.lower().endswith((".jpg", ".jpeg")):
                paths.append(line)

    if not paths:
        return paths

    # 👉 Rotate list by a random offset so we start at a random line
    offset = random.randrange(len(paths))
    paths = paths[offset:] + paths[:offset]

    return paths


def main():
    import sys

    print("Main thread native_id:", threading.get_native_id())

    if len(sys.argv) < 2:
        print("Usage: python slideshow.py file_list.txt")
        sys.exit(1)

    controller = SlideshowController()
    load_exclusions(controller)

    list_path = sys.argv[1]
    file_paths = read_file_list(list_path)

    if not file_paths:
        print(f"No .jpg/.jpeg entries found in {list_path}; nothing to display.")
        sys.exit(1)

    shared_deque: deque[Slide] = deque()
    lock = threading.Lock()
    not_full = threading.Condition(lock)
    producer_done = threading.Event()

    # start web server in background
    web_thread = Thread(target=run_web, args=(controller,), daemon=True)
    web_thread.start()

    # Start fetcher thread
    fetcher = threading.Thread(
        target=image_fetcher_thread,
        args=(file_paths, shared_deque, lock, not_full, producer_done, controller, 5),
        daemon=True,
    )
    fetcher.start()

    # Start render loop in main thread
    render_loop(shared_deque, lock, not_full, producer_done, controller, seconds_to_display)


if __name__ == "__main__":
    main()
