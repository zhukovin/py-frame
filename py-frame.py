import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice, product
from typing import List, Tuple, Literal, Optional

import pygame
from PIL import Image, ExifTags, ImageOps
import random

Orientation = Literal["P", "L"]  # P = Portrait, L = Landscape

seconds_to_display = 10


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
        file_paths: List[str],
        dq: deque[Slide],
        lock: threading.Lock,
        not_full: threading.Condition,
        producer_done: threading.Event,  # kept for compatibility, but no longer used
        max_size: int = 5,
):
    """
    Producer: endlessly cycles over file_paths, filling dq up to max_size.
    Starts from whatever order file_paths already has (which we rotated randomly).
    """
    try:
        if not file_paths:
            # Nothing to do; just signal done (optional)
            producer_done.set()
            return

        idx = 0
        n = len(file_paths)

        while True:  # 👉 infinite loop = infinite slideshow
            path = file_paths[idx]

            # Move to next index (wrap around)
            idx = (idx + 1) % n

            # Wait for space in deque
            with not_full:
                while len(dq) >= max_size:
                    not_full.wait()

            try:
                slide = load_slide(path)
            except Exception as e:
                # If an image fails to load, skip it
                print(f"Failed to load {path}: {e}")
                continue

            with not_full:
                dq.append(slide)
                not_full.notify_all()

    finally:
        # In practice we never reach here, but keep for symmetry
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
    new_w = int(iw * scale)
    new_h = int(ih * scale)

    scaled = smoothscale_safe(img, (new_w, new_h))

    x = target_rect.x + (tw - new_w) // 2
    y = target_rect.y + (th - new_h) // 2
    surface.blit(scaled, (x, y))


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


def render_single_landscape(screen: pygame.Surface, slide: Slide, background: pygame.Surface):
    if background is not None:
        screen.blit(background, (0, 0))
    else:
        screen.fill(OLD_PAPER)
    rect = screen.get_rect()
    blit_scaled(screen, slide.surface, rect)


def render_pattern(screen: pygame.Surface, slides: List[Slide], pattern_type: int, background: pygame.Surface):
    """
    Render slides according to pattern type:
      1: PPP   -> 3 portrait images side-by-side, full screen (3 columns)
      2: PPLLL -> Column 1: 3 L stacked; Column 2 & 3: 2 P full-height
      3: PLLL  -> Column 3: P full-height; Columns 1-2: top L spans both,
                              bottom two L's share the bottom half (one in col1, one in col2)
    """
    if background is not None:
        screen.blit(background, (0, 0))
    else:
        screen.fill(OLD_PAPER)

    W, H = screen.get_width(), screen.get_height()
    col_w = W // 3

    if pattern_type == 1:
        # PPP: 3 images → columns 0,1,2 full-height
        for idx, slide in enumerate(slides[:3]):
            rect = pygame.Rect(idx * col_w, 0, col_w, H)
            blit_scaled(screen, slide.surface, rect)

    elif pattern_type == 2:
        # PPLLL: we should have 2 P and 3 L in slides
        Ls = [s for s in slides if s.orientation == "L"]
        Ps = [s for s in slides if s.orientation == "P"]

        # Column 1: 3 L stacked
        if len(Ls) >= 3:
            h3 = H // 3
            for i in range(3):
                rect = pygame.Rect(0, i * h3, col_w, h3 if i < 2 else H - 2 * h3)
                blit_scaled(screen, Ls[i].surface, rect)

        # Columns 2 & 3: 2 P full-height
        if len(Ps) >= 1:
            rect = pygame.Rect(col_w, 0, col_w, H)
            blit_scaled(screen, Ps[0].surface, rect)
        if len(Ps) >= 2:
            rect = pygame.Rect(2 * col_w, 0, col_w, H)
            blit_scaled(screen, Ps[1].surface, rect)

    elif pattern_type == 3:
        # PLLL: 1 P, 3 L
        Ls = [s for s in slides if s.orientation == "L"]
        Ps = [s for s in slides if s.orientation == "P"]
        if not Ps or len(Ls) < 1:
            return

        # Column 3: P full-height
        rect_p = pygame.Rect(2 * col_w, 0, col_w, H)
        blit_scaled(screen, Ps[0].surface, rect_p)

        # Top L spans columns 1+2 (width = 2 * col_w), top half of screen
        top_L = Ls[0]
        rect_top = pygame.Rect(0, 0, 2 * col_w, H // 2)
        blit_scaled(screen, top_L.surface, rect_top)

        # Remaining 2 Ls (if present) in bottom half, columns 1 and 2
        if len(Ls) >= 2:
            rect_bottom_left = pygame.Rect(0, H // 2, col_w, H // 2)
            blit_scaled(screen, Ls[1].surface, rect_bottom_left)
        if len(Ls) >= 3:
            rect_bottom_right = pygame.Rect(col_w, H // 2, col_w, H // 2)
            blit_scaled(screen, Ls[2].surface, rect_bottom_right)


def compute_pattern_rects(screen: pygame.Surface, slides: List[Slide], pattern_type: int) -> List[
    tuple[pygame.Surface, pygame.Rect]]:
    """
    Compute rects for each slide according to pattern_type (1,2,3),
    matching your render_pattern layout.
    Returns list of (surface, rect).
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


# ============================================================
# Render loop (consumer)
# ============================================================

def render_loop(
        dq: deque[Slide],
        lock: threading.Lock,
        not_full: threading.Condition,
        producer_done: threading.Event,
):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    current_slides: List[Slide] = []
    current_pattern_type: Optional[int] = None  # 0 for single landscape
    current_end_time: float = 0.0
    current_background: Optional[pygame.Surface] = None  # ✅ persist across frames

    running = True
    while running:
        now = time.time()

        # Handle events (ESC/close)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.QUIT:
                running = False

        # Decide if we need to load a new image / pattern
        if (not current_slides) or (now >= current_end_time):
            # Get new stuff if available
            with not_full:
                # If deque empty and producer done -> we're done
                if len(dq) == 0 and producer_done.is_set():
                    running = False

                elif len(dq) >= 5:
                    # We have enough to safely run pattern logic
                    first = dq[0]
                    if first.orientation == "L":
                        # Single landscape fullscreen
                        slide = dq.popleft()
                        not_full.notify_all()
                        current_slides = [slide]
                        current_pattern_type = 0  # single landscape
                        current_end_time = now + seconds_to_display

                        # Build background for this single image
                        screen_size = screen.get_size()
                        rect = screen.get_rect()
                        current_background = build_blurred_background(
                            screen_size,
                            [(slide.surface, rect)]
                        )
                    else:
                        # Portrait: use pattern extraction
                        slides, ptype = extract_pattern_from_deque(dq)
                        not_full.notify_all()
                        current_slides = slides
                        current_pattern_type = ptype
                        current_end_time = now + seconds_to_display

                        # ✅ Build background for this pattern
                        rects = compute_pattern_rects(screen, current_slides, current_pattern_type)
                        current_background = build_blurred_background(
                            screen.get_size(),
                            rects
                        )
                else:
                    # Less than 5 images – keep showing previous slides if any
                    # (do nothing, current_slides and current_background remain)
                    pass

        # Render current slides if any
        if current_slides:
            if current_pattern_type == 0:
                render_single_landscape(screen, current_slides[0], current_background)
            else:
                render_pattern(screen, current_slides, current_pattern_type, current_background)

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

    list_path = sys.argv[1]
    file_paths = read_file_list(list_path)

    shared_deque: deque[Slide] = deque()
    lock = threading.Lock()
    not_full = threading.Condition(lock)
    producer_done = threading.Event()

    # Start fetcher thread
    fetcher = threading.Thread(
        target=image_fetcher_thread,
        args=(file_paths, shared_deque, lock, not_full, producer_done, 5),
        daemon=True,
    )
    fetcher.start()

    # Optional: run pattern tests (for logic sanity)
    test_extract_pattern_all_len5()

    # Start render loop in main thread
    render_loop(shared_deque, lock, not_full, producer_done)


if __name__ == "__main__":
    main()
