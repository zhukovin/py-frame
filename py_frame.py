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
import logging


faulthandler.register(signal.SIGUSR1)

logger = logging.getLogger("py_frame")

ERROR_LOG_FILE = "app_errors.log"

Orientation = Literal["P", "L"]  # P = Portrait, L = Landscape

seconds_to_display = 15

MAX_HISTORY_SCREENS = 5  # or 20, tune as you like


def log_mem(tag=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"[MEM {tag}] pid={os.getpid()} rss={usage.ru_maxrss} B")


_logging_configured = False


def setup_logging():
    """
    Configure a rotating file handler on the root logger so every
    exception -- caught-and-handled or truly uncaught, from any thread
    (main, image fetcher, Flask's web server) -- ends up in ERROR_LOG_FILE
    for later analysis, in addition to the usual terminal/journal output.

    Attached to the root logger (not just "py_frame") specifically so
    Flask's own app.logger, which propagates to root by default, is
    captured too without needing any changes in web_server.py. Safe to
    call more than once (e.g. across tests) -- only configures once.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    import sys
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(ERROR_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    uncaught_logger = logging.getLogger("py_frame.uncaught")

    def log_uncaught_main_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        uncaught_logger.critical(
            "Uncaught exception in main thread",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_uncaught_main_exception

    def log_uncaught_thread_exception(args):
        uncaught_logger.critical(
            f"Uncaught exception in thread {args.thread.name!r}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        threading.__excepthook__(args)

    threading.excepthook = log_uncaught_thread_exception


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

        # Every load attempt (success or failure) is appended as a CSV row
        # here for later offline analysis of size/time/speed correlation.
        self.measurements_file: str = "load_measurements.csv"

        # Photo display order at startup: shuffle (fully randomized order)
        # vs. random-start (original file order, but starting from a random
        # point and wrapping around). Persisted to disk so the web UI
        # toggle survives restarts; takes effect the next time the app
        # starts, since the fetcher thread's order is fixed once built.
        self.shuffle_enabled: bool = True
        self.settings_file: str = "settings.json"


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


class ImageDecodeError(Exception):
    """
    Raised when a file was read successfully but its image content couldn't
    be decoded (corrupt file, unsupported format, etc). Deliberately
    distinct from I/O errors raised while reading the raw bytes, so callers
    can tell "one bad file" apart from "the drive/NFS mount is unreachable"
    instead of conflating both into the same failure signal.

    Carries the read measurements (the raw read succeeded even though the
    decode didn't), so callers logging load performance can still record a
    valid size/time/speed data point for the read itself.
    """
    def __init__(self, message: str, load_bytes: int = 0, load_seconds: float = 0.0):
        super().__init__(message)
        self.load_bytes = load_bytes
        self.load_seconds = load_seconds


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
    #    Exceptions here (missing file, permission error, stale/disconnected
    #    NFS mount) are real I/O problems and propagate as-is.
    t0 = time.monotonic()
    with open(path, "rb") as f:
        raw = f.read()
    load_seconds = time.monotonic() - t0
    load_bytes = len(raw)

    # 2) Decode with Pillow. Failures here mean the file itself is
    #    corrupt/unsupported, not that the drive is unreachable, so they're
    #    wrapped in ImageDecodeError rather than left as a bare Exception.
    try:
        img = Image.open(io.BytesIO(raw))

        # Apply EXIF orientation (rotates/flip as needed)
        img = ImageOps.exif_transpose(img)

        # Classify after rotation
        width, height = img.size
        orientation: Orientation = "P" if height > width else "L"

        # Ensure mode is suitable for pygame
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        mode = img.mode
        size = img.size
        data = img.tobytes()

        # Convert to pygame surface
        if mode == "RGBA":
            surface = pygame.image.fromstring(data, size, mode).convert_alpha()
        else:  # "RGB"
            surface = pygame.image.fromstring(data, size, mode).convert()
    except Exception as e:
        raise ImageDecodeError(
            f"failed to decode {path}: {e}",
            load_bytes=load_bytes,
            load_seconds=load_seconds,
        ) from e

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

def classify_pattern_type(count_p: int, count_l: int) -> Optional[Tuple[int, int, int]]:
    """
    Given counts of Portrait/Landscape slides, decide which pattern type (if
    any) they satisfy and how many of each orientation that pattern needs:
        Type 1: PPP      (needs 3 Portrait, 0 Landscape)
        Type 2: PPLLL    (needs 2 Portrait, 3 Landscape)
        Type 3: PLLL     (needs 1 Portrait, 3 Landscape)
    Returns None if no valid pattern fits. Shared by extract_pattern_from_deque
    (fresh extraction from the incoming deque) and reclassify_pattern_type
    (re-deriving a history entry's type after exclusion) so the thresholds
    can't silently drift between the two.
    """
    if count_p >= 3:
        return 1, 3, 0
    elif count_p >= 2 and count_l >= 3:
        return 2, 2, 3
    elif count_p >= 1 and count_l >= 3:
        return 3, 1, 3
    else:
        return None


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
    classification = classify_pattern_type(count_p, count_l)
    if classification is None:
        raise ValueError("No valid PPP / PPLLL / PLLL pattern in first 5 entries.")
    pattern_type, needP, needL = classification

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

def _record_load_attempt(controller: SlideshowController,
                         success: bool,
                         bytes_per_sec: Optional[float] = None,
                         update_drive_status: bool = True):
    """
    Record the outcome of one photo load attempt for the status overlay.
    Pass update_drive_status=False for a failure that reflects a bad file
    rather than a drive/connectivity problem (e.g. a corrupt image), so
    drive_ok and the speed estimate are left untouched.
    """
    if not update_drive_status:
        return
    with controller.lock:
        controller.drive_ok = success
        controller.download_bytes_per_sec = bytes_per_sec


def log_load_measurement(log_file: str, path: str, outcome: str, load_bytes: int = 0,
                         load_seconds: float = 0.0, error_type: str = ""):
    """
    Append one load-attempt measurement to a CSV file for later offline
    analysis of size/time/speed correlation (e.g. "why is loading time or
    transfer speed inconsistent"). One row per attempt, regardless of
    outcome:
        outcome="ok"             - full success; bytes/seconds/speed are valid
        outcome="decode_error"   - the raw read succeeded (bytes/seconds are
                                    still valid) but the image itself was
                                    corrupt/unsupported
        outcome="file_not_found" - this one path doesn't exist (deleted/
                                    renamed on the NAS, a stale list entry,
                                    etc) -- says nothing about whether the
                                    drive/mount itself is reachable
        outcome="io_error"       - some other read failure (disconnected
                                    mount, permission error, timeout, etc);
                                    no measurement was possible, so
                                    bytes/seconds are blank
    error_type is the underlying exception's class name (e.g.
    "FileNotFoundError", "UnidentifiedImageError", "ConnectionResetError"),
    for telling apart *why* a failure happened, not just that it did.
    """
    import csv
    from datetime import datetime

    has_measurement = outcome in ("ok", "decode_error")
    bytes_per_sec = load_bytes / load_seconds if has_measurement and load_seconds > 0 else ""

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "path", "outcome", "bytes", "seconds", "bytes_per_sec", "error_type"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            path,
            outcome,
            load_bytes if has_measurement else "",
            f"{load_seconds:.4f}" if has_measurement else "",
            f"{bytes_per_sec:.2f}" if bytes_per_sec != "" else "",
            error_type,
        ])


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
            except ImageDecodeError as e:
                # A bad/corrupt file, not a drive problem: don't flag the
                # drive down or discard the legitimate rolling speed
                # average, but the read itself succeeded so still log its
                # size/time as a valid measurement.
                logger.warning(f"Skipping unreadable image {path}", exc_info=True)
                _record_load_attempt(controller, success=False, update_drive_status=False)
                log_load_measurement(
                    controller.measurements_file, path, "decode_error",
                    load_bytes=e.load_bytes, load_seconds=e.load_seconds,
                    error_type=type(e.__cause__).__name__ if e.__cause__ else type(e).__name__,
                )
                time.sleep(0.5)
                continue
            except FileNotFoundError as e:
                # This one path doesn't exist -- e.g. deleted/renamed on the
                # NAS, or a stale list entry. Says nothing about whether the
                # drive/mount itself is reachable, so don't flag it down or
                # discard the rolling speed average (unlike a real io_error).
                logger.warning(f"Photo not found: {path}", exc_info=True)
                _record_load_attempt(controller, success=False, update_drive_status=False)
                log_load_measurement(
                    controller.measurements_file, path, "file_not_found",
                    error_type=type(e).__name__,
                )
                time.sleep(0.5)
                continue
            except Exception as e:
                logger.error(f"Failed to load {path}", exc_info=True)
                recent_load_stats.clear()
                _record_load_attempt(controller, success=False)
                log_load_measurement(
                    controller.measurements_file, path, "io_error",
                    error_type=type(e).__name__,
                )
                time.sleep(0.5)
                continue

            recent_load_stats.append((slide.load_bytes, slide.load_seconds))
            total_bytes = sum(b for b, _ in recent_load_stats)
            total_seconds = sum(s for _, s in recent_load_stats)
            bytes_per_sec = total_bytes / total_seconds if total_seconds > 0 else None

            _record_load_attempt(controller, success=True, bytes_per_sec=bytes_per_sec)
            log_load_measurement(
                controller.measurements_file, path, "ok",
                load_bytes=slide.load_bytes, load_seconds=slide.load_seconds,
            )

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


STATUS_PADDING = 8
STATUS_LINE_SPACING = 4
STATUS_MARGIN = 10
STATUS_OUTLINE_PX = 2
# Sized against the worst case of each line (not whichever happens to be
# showing right now), so the box's position/size stays stable across
# Paused/Playing or Drive: OK/DISCONNECTED transitions.
STATUS_FIXED_VOCABULARY = ["Paused", "Playing", "Drive: OK", "Drive: DISCONNECTED"]


def compute_status_box_rect(screen: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
    """
    Compute the status overlay's bottom-right rect. Fixed size/position for
    a given screen+font (independent of the current paused/drive_ok/speed
    values), so callers can snapshot this exact region once and reuse it to
    erase old text before redrawing -- there's no background box to do that
    erasing for them anymore.
    """
    stable_widths = [font.size(text)[0] for text in STATUS_FIXED_VOCABULARY]
    box_w = max(stable_widths) + 2 * STATUS_PADDING + 2 * STATUS_OUTLINE_PX
    line_height = font.get_height()
    box_h = (
        line_height * 3
        + STATUS_LINE_SPACING * 2
        + 2 * STATUS_PADDING
        + 2 * STATUS_OUTLINE_PX
    )
    screen_w, screen_h = screen.get_size()
    return pygame.Rect(
        screen_w - box_w - STATUS_MARGIN,
        screen_h - box_h - STATUS_MARGIN,
        box_w,
        box_h,
    )


def draw_status_overlay(screen: pygame.Surface,
                        font: pygame.font.Font,
                        paused: bool,
                        drive_ok: bool,
                        download_bytes_per_sec: Optional[float],
                        box_rect: Optional[pygame.Rect] = None) -> pygame.Rect:
    """
    Draw the status text -- Playing/Paused, drive mount health, and recent
    read speed -- black with a white outline, no filled background, in the
    bottom-right corner. Drawn last (on top of everything, including
    black-screen mode) so stalls can be diagnosed without waiting for the
    next slide change.

    Pass the same box_rect (from compute_status_box_rect) every call so the
    text is always positioned identically; this is what the caller
    snapshots/restores to erase old text before redrawing (see render_loop).
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    if box_rect is None:
        box_rect = compute_status_box_rect(screen, font)

    lines = [
        "Paused" if paused else "Playing",
        "Drive: OK" if drive_ok else "Drive: DISCONNECTED",
        format_speed(download_bytes_per_sec),
    ]
    text_surfaces = [font.render(line, True, BLACK) for line in lines]
    outline_surfaces = [font.render(line, True, WHITE) for line in lines]

    right_edge = box_rect.right - STATUS_PADDING - STATUS_OUTLINE_PX
    y = box_rect.y + STATUS_PADDING + STATUS_OUTLINE_PX
    for text_surf, outline_surf in zip(text_surfaces, outline_surfaces):
        x = right_edge - text_surf.get_width()
        for dx in range(-STATUS_OUTLINE_PX, STATUS_OUTLINE_PX + 1):
            for dy in range(-STATUS_OUTLINE_PX, STATUS_OUTLINE_PX + 1):
                if dx == 0 and dy == 0:
                    continue
                screen.blit(outline_surf, (x + dx, y + dy))
        screen.blit(text_surf, (x, y))
        y += text_surf.get_height() + STATUS_LINE_SPACING

    return box_rect


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


def load_settings(controller: SlideshowController):
    """
    Load persisted app settings (currently just shuffle_enabled) from
    controller.settings_file, if it exists. Called once at startup, before
    read_file_list, so the web UI's shuffle/random-start toggle survives
    restarts. A missing or corrupt settings file falls back to the
    defaults already set on the controller rather than crashing startup.
    """
    if not os.path.exists(controller.settings_file):
        return

    import json

    try:
        with open(controller.settings_file, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.warning(f"Could not read {controller.settings_file}, using defaults", exc_info=True)
        return

    if "shuffle_enabled" in data:
        controller.shuffle_enabled = bool(data["shuffle_enabled"])


CONFIG_FILE = "py-frame.conf"
DEFAULT_NIGHT_START = (22, 0)
DEFAULT_NIGHT_END = (7, 0)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour_str, _, minute_str = value.strip().partition(":")
    hour = int(hour_str)
    minute = int(minute_str) if minute_str else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"time out of range: {value!r}")
    return hour, minute


def load_schedule_config(path: str = CONFIG_FILE) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Read the auto-scheduling night window (start, end) from an INI-style
    config file, e.g.:

        [schedule]
        start = 22:00
        end = 07:00

    Falls back to the DEFAULT_NIGHT_START/END window when the file is
    missing or malformed, so a bad or absent config never blocks startup.
    """
    if not os.path.exists(path):
        return DEFAULT_NIGHT_START, DEFAULT_NIGHT_END

    import configparser

    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        start = _parse_hhmm(parser.get("schedule", "start", fallback="22:00"))
        end = _parse_hhmm(parser.get("schedule", "end", fallback="07:00"))
    except (configparser.Error, ValueError):
        logger.warning(f"Could not parse {path}, using default schedule 22:00-07:00", exc_info=True)
        return DEFAULT_NIGHT_START, DEFAULT_NIGHT_END

    return start, end


def is_within_night_window(hour: int, minute: int, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """
    True if hour:minute falls within the [start, end) night window.
    Handles windows that wrap past midnight (e.g. 22:00 -> 07:00) as well
    as same-day windows (e.g. 13:00 -> 15:00). A zero-length window
    (start == end) never counts as night.
    """
    now_minutes = hour * 60 + minute
    start_minutes = start[0] * 60 + start[1]
    end_minutes = end[0] * 60 + end[1]

    if start_minutes == end_minutes:
        return False
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    return now_minutes >= start_minutes or now_minutes < end_minutes


def reclassify_pattern_type(slides: List[Slide], original_ptype: int) -> Optional[int]:
    """
    Decide how a history entry should be classified after some of its slides
    were removed due to exclusion.

    Patterns 0 (solo slide) and 1 (PPP) degrade gracefully as slides are
    removed, so they keep their original type. Patterns 2 (PPLLL) and 3
    (PLLL) depend on specific P/L proportions to avoid blank gaps in
    compute_pattern_rects, so they are re-derived from what's left using
    classify_pattern_type (the same thresholds extract_pattern_from_deque
    uses). Returns None if nothing valid remains and the entry should be
    dropped.
    """
    if not slides:
        return None

    if original_ptype in (0, 1):
        return original_ptype

    count_p = sum(1 for s in slides if s.orientation == "P")
    count_l = sum(1 for s in slides if s.orientation == "L")

    classification = classify_pattern_type(count_p, count_l)
    if classification is not None:
        return classification[0]
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
            old_index = controller.history_index
            new_history: list[tuple[list[Slide], int]] = []
            # Track where the entry the user was actually viewing ends up,
            # so history_index can follow it instead of drifting onto a
            # different entry when something *earlier* in the list gets
            # dropped (a plain index clamp only guards the tail, not this).
            new_index_for_old_index: Optional[int] = None

            for i, (slides, ptype) in enumerate(controller.history):
                filtered = [s for s in slides if s.path not in controller.excluded_paths]
                new_ptype = reclassify_pattern_type(filtered, ptype)
                if new_ptype is not None:
                    if i == old_index:
                        new_index_for_old_index = len(new_history)
                    new_history.append((filtered, new_ptype))

            controller.history = new_history

            if not controller.history:
                controller.history_index = -1
            elif new_index_for_old_index is not None:
                # The viewed entry survived (possibly at a new position).
                controller.history_index = new_index_for_old_index
            else:
                # The viewed entry itself was dropped; clamp into range.
                controller.history_index = min(old_index, len(controller.history) - 1)

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
        night_start: tuple[int, int] = DEFAULT_NIGHT_START,
        night_end: tuple[int, int] = DEFAULT_NIGHT_END,
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
    status_box_rect = compute_status_box_rect(screen, status_font)
    # Snapshot of whatever's behind the status text, taken right after each
    # full scene redraw (before the text is drawn). There's no background
    # box anymore, so when only the text changes (e.g. the Kbps figure),
    # this is what erases the old glyphs before the new ones are drawn --
    # otherwise their white outlines would ghost through each other.
    clean_status_corner: Optional[pygame.Surface] = None

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

        # --- Time-based schedule (configured via py-frame.conf) => screen off ---
        lt = time.localtime(now)
        is_night = is_within_night_window(lt.tm_hour, lt.tm_min, night_start, night_end)

        if prev_is_night is None:
            prev_is_night = is_night

        night_transition = is_night != prev_is_night
        if night_transition:
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

        # Automatic day/night boundary crossing also needs an immediate
        # redraw -- otherwise the last slide stays on screen (only the
        # status corner keeps refreshing) until something else changes it.
        if night_transition:
            need_to_render = True

        # Refresh just the diagnostics corner periodically even if nothing
        # else changed, so drive/speed issues show up promptly instead of
        # waiting for the next slide change. This is a much cheaper partial
        # update than a full scene redraw (see need_to_render handling
        # below), since only three lines of text change.
        need_status_refresh = (now - last_status_render_time) >= STATUS_REFRESH_SECONDS

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
                    need_more_images = False
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
                            # not enough images buffered yet
                            need_more_images = True

                    if need_more_images:
                        # Back off briefly instead of busy-spinning the CPU
                        # while waiting for the fetcher thread to refill the
                        # buffer (lock already released above).
                        time.sleep(0.2)
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

            # Snapshot whatever's behind the status text (before drawing
            # it), so a later text-only refresh can erase the old glyphs
            # cleanly -- there's no background box to do that anymore.
            safe_status_rect = status_box_rect.clip(screen.get_rect())
            if safe_status_rect.width > 0 and safe_status_rect.height > 0:
                clean_status_corner = screen.subsurface(safe_status_rect).copy()
            else:
                clean_status_corner = None

            # Diagnostics overlay: drawn last, on top of everything
            # (including black-screen mode) so stalls are visible.
            draw_status_overlay(screen, status_font, paused, drive_ok, download_bytes_per_sec, box_rect=status_box_rect)

            pygame.display.flip()
            last_status_render_time = now

        elif need_status_refresh:
            # Nothing else changed: erase the old status text using the
            # clean snapshot, redraw it, and push only that region --
            # instead of re-rendering and flipping the whole scene purely
            # to refresh three lines of text.
            if clean_status_corner is not None:
                screen.blit(clean_status_corner, safe_status_rect.topleft)
            draw_status_overlay(screen, status_font, paused, drive_ok, download_bytes_per_sec, box_rect=status_box_rect)
            pygame.display.update(status_box_rect)
            last_status_render_time = now

        if need_advance:
            log_mem("after_advance")

        time.sleep(0.2)   # keeps CPU load low without hammering the Pi

    pygame.quit()


# ============================================================
# Main entry
# ============================================================


def read_file_list(list_path: str, shuffle: bool = True) -> List[str]:
    """
    Read the photo list file and decide the display order:
      shuffle=True  - fully randomize the order (kept entirely in memory;
                       nothing downstream re-reads the source file, so
                       there's no need to persist the shuffled order).
      shuffle=False - keep the file's original relative order, but start
                       from a random point and wrap around.
    """
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

    if shuffle:
        random.shuffle(paths)
    else:
        offset = random.randrange(len(paths))
        paths = paths[offset:] + paths[:offset]

    return paths


def main():
    import sys

    setup_logging()

    print("Main thread native_id:", threading.get_native_id())

    if len(sys.argv) < 2:
        print("Usage: python slideshow.py file_list.txt")
        sys.exit(1)

    controller = SlideshowController()
    load_exclusions(controller)
    load_settings(controller)
    night_start, night_end = load_schedule_config()

    list_path = sys.argv[1]
    file_paths = read_file_list(list_path, shuffle=controller.shuffle_enabled)

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
    render_loop(shared_deque, lock, not_full, producer_done, controller, seconds_to_display,
                night_start=night_start, night_end=night_end)


if __name__ == "__main__":
    main()
