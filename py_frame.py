import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice, product
from typing import List, Tuple, Literal, Optional

import pygame
from PIL import Image, ExifTags, ImageOps
import random
from threading import Lock, Thread
from typing import Optional
from web_server import run_web

Orientation = Literal["P", "L"]  # P = Portrait, L = Landscape

seconds_to_display = 3


# ============================================================
# Slide object
# ============================================================

@dataclass
class Slide:
    path: str
    surface: pygame.Surface
    orientation: Orientation  # "P" or "L"


# ============================================================
# EXIF / orientation helpers
# ============================================================

# EXIF orientation tag id
EXIF_ORIENTATION_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        EXIF_ORIENTATION_TAG = k
        break


class SlideshowController:
    def __init__(self):
        self.lock = Lock()
        # Current screen
        self.current_slides: list[Slide] = []
        self.current_pattern_type: Optional[int] = None  # 0=single L, 1/2/3 = PPP / PPLLL / PLLL
        self.current_marks: set[int] = set()  # indices 0..4 marked for exclusion

        # History of screens (for back/forward)
        # each entry: (slides, pattern_type)
        self.history: list[tuple[list[Slide], int]] = []
        self.history_index: int = -1  # -1 means “no history yet”

        # Remote command: {"type": "next"|"prev", "steps": int}
        self.pending_command: Optional[dict] = None

        # Exclusions
        self.excluded_paths: set[str] = set()
        self.exclusions_file: str = "exclusions.txt"


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
    # 1) Load with Pillow
    img = Image.open(path)

    # 2) Apply EXIF orientation (rotates/flip as needed)
    img = ImageOps.exif_transpose(img)

    # 3) Classify after rotation
    width, height = img.size
    orientation: Orientation = "P" if height > width else "L"

    # 4) Ensure mode is suitable for pygame
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    mode = img.mode
    size = img.size
    data = img.tobytes()

    # 5) Convert to pygame surface
    if mode == "RGBA":
        surface = pygame.image.fromstring(data, size, mode).convert_alpha()
    else:  # "RGB"
        surface = pygame.image.fromstring(data, size, mode).convert()

    return Slide(path=path, surface=surface, orientation=orientation)


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
    try:
        if not file_paths:
            producer_done.set()
            return

        idx = 0
        n = len(file_paths)

        while True:
            path = file_paths[idx]
            idx = (idx + 1) % n

            with controller.lock:
                if path in controller.excluded_paths:
                    continue

            with not_full:
                while len(dq) >= max_size:
                    not_full.wait()

            try:
                slide = load_slide(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

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
    # border color
    color = (255, 64, 64) if marked else (240, 240, 240)
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


OLD_PAPER = (235, 222, 193)  # warm beige


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

    for idx, (surf, rect) in enumerate(rects):
        blit_scaled(screen, surf, rect)
        draw_slot_overlay(screen, rect, idx, (idx in marks), font)


def finalize_exclusions(controller: SlideshowController):
    """Write current_marks from current_slides to exclusions file and set."""
    if not controller.current_slides:
        controller.current_marks.clear()
        return

    with controller.lock:
        marked_indices = list(controller.current_marks)
        controller.current_marks.clear()

    if not marked_indices:
        return

    new_paths = []
    for i in marked_indices:
        if 0 <= i < len(controller.current_slides):
            path = controller.current_slides[i].path
            if path not in controller.excluded_paths:
                controller.excluded_paths.add(path)
                new_paths.append(path)

    if new_paths:
        with open(controller.exclusions_file, "a") as f:
            for p in new_paths:
                f.write(p + "\n")


def render_loop(
        dq: deque[Slide],
        lock: threading.Lock,
        not_full: threading.Condition,
        producer_done: threading.Event,
        controller: SlideshowController,
        seconds_to_display: int = 15,
):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 40)

    current_slides: list[Slide] = []
    current_pattern_type: Optional[int] = None
    current_background: Optional[pygame.Surface] = None
    current_end_time: float = 0.0

    running = True
    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.QUIT:
                running = False

        # Take any pending command from web
        with controller.lock:
            cmd = controller.pending_command
            controller.pending_command = None

        force_next = False
        force_prev = False
        steps = 1
        if cmd:
            if cmd["type"] == "next":
                force_next = True
            elif cmd["type"] == "prev":
                force_prev = True
            steps = max(1, int(cmd.get("steps", 1)))

        # Decide if we need new screen
        need_advance = False
        backward = False

        if force_prev:
            need_advance = True
            backward = True
        elif force_next:
            need_advance = True
            backward = False
        elif (not current_slides) or (now >= current_end_time):
            need_advance = True
            backward = False

        if need_advance:
            # finalize exclusions for current screen
            finalize_exclusions(controller)

            if backward:
                # move back in history
                with controller.lock:
                    if controller.history:
                        controller.history_index = max(0, controller.history_index - steps
                        if controller.history_index >= 0
                        else len(controller.history) - 1 - steps)
                        idx = controller.history_index
                        slides, ptype = controller.history[idx]
                    else:
                        slides, ptype = [], None
                current_slides = slides
                current_pattern_type = ptype
                if current_slides and current_pattern_type is not None:
                    if current_pattern_type == 0:
                        rects = [(current_slides[0].surface, screen.get_rect())]
                    else:
                        rects = compute_pattern_rects(screen, current_slides, current_pattern_type)
                    current_background = build_blurred_background(screen.get_size(), rects)
                    current_end_time = now + seconds_to_display
                continue  # rendering at bottom

            # Forward: history-forward or new from deque
            with controller.lock:
                hist_len = len(controller.history)
                idx = controller.history_index

            used_history = False
            if hist_len > 0 and 0 <= idx < hist_len - 1:
                # forward in history
                new_index = min(hist_len - 1, idx + steps)
                with controller.lock:
                    controller.history_index = new_index
                    slides, ptype = controller.history[new_index]
                current_slides = slides
                current_pattern_type = ptype
                used_history = True
            else:
                # need a brand new screen from deque
                with not_full:
                    if len(dq) == 0 and producer_done.is_set():
                        running = False
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
                        pass

                if current_slides and current_pattern_type is not None:
                    with controller.lock:
                        controller.history.append((current_slides, current_pattern_type))
                        controller.history_index = len(controller.history) - 1

            # Build background for current screen
            if current_slides and current_pattern_type is not None:
                if current_pattern_type == 0:
                    rects = [(current_slides[0].surface, screen.get_rect())]
                else:
                    rects = compute_pattern_rects(screen, current_slides, current_pattern_type)
                current_background = build_blurred_background(screen.get_size(), rects)
                current_end_time = now + seconds_to_display

            # Update controller current_slides/pattern for web/state
            with controller.lock:
                controller.current_slides = current_slides
                controller.current_pattern_type = current_pattern_type

        # Render current screen
        if current_slides and current_pattern_type is not None:
            with controller.lock:
                marks_copy = set(controller.current_marks)

            if current_pattern_type == 0:
                render_single_landscape(screen, current_slides[0],
                                        current_background, font, marks_copy)
            else:
                render_pattern(screen, current_slides,
                               current_pattern_type, current_background,
                               font, marks_copy)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ============================================================
# Tests for extract_pattern_from_deque logic
# ============================================================

class DummySlide(Slide):
    """Slide subclass for testing, ignoring actual pygame surfaces."""

    def __init__(self, orientation: Orientation):
        self.path = ""
        self.surface = None  # type: ignore
        self.orientation = orientation


def test_extract_pattern_all_len5():
    """
    For all 5-length orientation sequences starting with P,
    check that pattern extraction:
      - picks PPP if possible
      - else PPLLL if possible
      - else PLLL if possible
       - extracts correct counts and returns remaining correctly.
    """
    for bits in product("PL", repeat=5):
        seq = "".join(bits)
        if seq[0] != "P":
            continue

        # Determine expected pattern type and needed counts
        cP = seq.count("P")
        cL = seq.count("L")

        if cP >= 3:
            exp_type = 1
            needP, needL = 3, 0
        elif cP >= 2 and cL >= 3:
            exp_type = 2
            needP, needL = 2, 3
        elif cP >= 1 and cL >= 3:
            exp_type = 3
            needP, needL = 1, 3
        else:
            raise AssertionError(f"Unexpected no-pattern case for {seq}")

        dq = deque(DummySlide(o) for o in seq)
        extracted, out_type = extract_pattern_from_deque(dq)

        assert out_type == exp_type, f"{seq}: expected type {exp_type}, got {out_type}"
        assert sum(1 for s in extracted if s.orientation == "P") == needP
        assert sum(1 for s in extracted if s.orientation == "L") == needL

        # simulate expected remaining
        window = list(seq[:5])
        p_left, l_left = needP, needL
        unused = []
        for ch in window:
            if ch == "P" and p_left > 0:
                p_left -= 1
            elif ch == "L" and l_left > 0:
                l_left -= 1
            else:
                unused.append(ch)
            if p_left == 0 and l_left == 0:
                break
        expected_remaining = unused + list(seq[5:])
        actual_remaining = [s.orientation for s in dq]

        assert expected_remaining == actual_remaining, \
            f"{seq}: expected remaining {expected_remaining}, got {actual_remaining}"


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

    if len(sys.argv) < 2:
        print("Usage: python slideshow.py file_list.txt")
        sys.exit(1)

    controller = SlideshowController()

    list_path = sys.argv[1]
    file_paths = read_file_list(list_path)

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

    # Optional: run pattern tests (for logic sanity)
    # test_extract_pattern_all_len5()

    # Start render loop in main thread
    render_loop(shared_deque, lock, not_full, producer_done, controller, seconds_to_display)


if __name__ == "__main__":
    main()
