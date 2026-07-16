"""
Comprehensive test suite for py_frame.py
This file extracts and runs existing tests from py_frame.py
"""
import pytest
from collections import deque
from itertools import product
import pygame
import os
import sys
import logging
import tempfile
import threading
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

import py_frame

# Import the test function and dependencies from py_frame
from py_frame import (
    Slide,
    SlideshowController,
    classify_pattern_type,
    extract_pattern_from_deque,
    make_old_paper_surface,
    load_slide,
    ImageDecodeError,
    smoothscale_safe,
    blit_scaled,
    draw_slot_overlay,
    draw_status_overlay,
    format_speed,
    compute_pattern_rects,
    load_exclusions,
    load_settings,
    finalize_exclusions,
    reclassify_pattern_type,
    downscale_slide_to_screen,
    downscale_slides_to_screen,
    read_file_list,
    image_fetcher_thread,
    log_load_measurement,
    setup_logging,
    main,
    Orientation
)


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


class TestSlideshowController:
    """Test suite for SlideshowController class"""
    
    def test_initialization(self):
        """Test controller initializes with correct default values"""
        controller = SlideshowController()
        
        assert controller.current_slides == []
        assert controller.current_pattern_type is None
        assert controller.current_marks == set()
        assert controller.history == []
        assert controller.history_index == -1
        assert controller.pending_command is None
        assert controller.excluded_paths == set()
        assert controller.exclusions_file == "exclusions.txt"
        assert controller.paused is False
        assert controller.black_screen is False
        assert controller.drive_ok is True
        assert controller.download_bytes_per_sec is None
        assert controller.measurements_file == "load_measurements.csv"
        assert controller.shuffle_enabled is True
        assert controller.settings_file == "settings.json"

    def test_marks_management(self):
        """Test marking and unmarking slides"""
        controller = SlideshowController()
        
        # Add marks
        controller.current_marks.add(0)
        controller.current_marks.add(2)
        assert 0 in controller.current_marks
        assert 2 in controller.current_marks
        assert 1 not in controller.current_marks
        
        # Remove marks
        controller.current_marks.remove(0)
        assert 0 not in controller.current_marks
        assert 2 in controller.current_marks
    
    def test_pause_state(self):
        """Test pause and play states"""
        controller = SlideshowController()
        
        assert controller.paused is False
        controller.paused = True
        assert controller.paused is True
        controller.paused = False
        assert controller.paused is False
    
    def test_black_screen_mode(self):
        """Test black screen mode"""
        controller = SlideshowController()
        
        assert controller.black_screen is False
        controller.black_screen = True
        assert controller.black_screen is True


class TestMakeOldPaperSurface:
    """Test suite for make_old_paper_surface function"""
    
    def setup_method(self):
        """Initialize pygame for each test"""
        pygame.init()
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_creates_surface_with_correct_size(self):
        """Test that surface is created with requested dimensions"""
        width, height = 100, 200
        surface = make_old_paper_surface((width, height))
        
        assert surface.get_width() == width
        assert surface.get_height() == height
    
    def test_surface_has_old_paper_base_color(self):
        """Test that surface has beige/old paper color as base"""
        surface = make_old_paper_surface((10, 10))
        
        # Get color at center (should be close to base color with some noise)
        center_color = surface.get_at((5, 5))
        base_color = (235, 222, 193)
        
        # Colors should be close to base (within noise range)
        for i in range(3):
            assert abs(center_color[i] - base_color[i]) <= 20


class TestLoadSlide:
    """Test suite for load_slide function"""
    
    def setup_method(self):
        """Initialize pygame and create test images"""
        pygame.init()
        # Set a video mode for load_slide to work
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.display.set_mode((1, 1))
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test files and pygame"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        pygame.quit()
    
    def test_load_portrait_image(self):
        """Test loading a portrait orientation image"""
        # Create a portrait test image (height > width)
        img_path = os.path.join(self.temp_dir, "portrait.jpg")
        img = Image.new("RGB", (100, 200), color="red")
        img.save(img_path)
        
        slide = load_slide(img_path)
        
        assert slide.path == img_path
        assert slide.orientation == "P"
        assert slide.surface is not None
        assert slide.surface.get_width() == 100
        assert slide.surface.get_height() == 200
    
    def test_load_landscape_image(self):
        """Test loading a landscape orientation image"""
        # Create a landscape test image (width > height)
        img_path = os.path.join(self.temp_dir, "landscape.jpg")
        img = Image.new("RGB", (200, 100), color="blue")
        img.save(img_path)
        
        slide = load_slide(img_path)
        
        assert slide.path == img_path
        assert slide.orientation == "L"
        assert slide.surface is not None
        assert slide.surface.get_width() == 200
        assert slide.surface.get_height() == 100
    
    def test_load_square_image(self):
        """Test loading a square image (should be classified as landscape)"""
        img_path = os.path.join(self.temp_dir, "square.jpg")
        img = Image.new("RGB", (100, 100), color="green")
        img.save(img_path)

        slide = load_slide(img_path)

        assert slide.orientation == "L"  # Equal dimensions = landscape

    def test_load_slide_reports_read_size_and_timing(self):
        """Test that load_slide reports the raw bytes read and time taken,
        used for the download-speed diagnostics overlay"""
        img_path = os.path.join(self.temp_dir, "timed.jpg")
        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(img_path)

        slide = load_slide(img_path)

        assert slide.load_bytes == os.path.getsize(img_path)
        assert slide.load_seconds >= 0

    def test_corrupt_file_raises_image_decode_error(self):
        """A file that reads fine but isn't a valid image should raise
        ImageDecodeError, not a bare exception, so callers can tell a bad
        file apart from a real drive/NFS problem"""
        bad_path = os.path.join(self.temp_dir, "corrupt.jpg")
        content = b"this is not a real jpeg file"
        with open(bad_path, "wb") as f:
            f.write(content)

        with pytest.raises(ImageDecodeError) as exc_info:
            load_slide(bad_path)

        # The raw read succeeded before decoding failed, so the exception
        # should still carry that valid measurement for logging purposes.
        assert exc_info.value.load_bytes == len(content)
        assert exc_info.value.load_seconds >= 0

    def test_missing_file_raises_plain_oserror_not_image_decode_error(self):
        """A missing file fails at the read stage (before decoding is even
        attempted), so it must NOT be wrapped as ImageDecodeError -- callers
        rely on that distinction to tell "bad file" apart from "can't reach
        the drive at all\""""
        missing_path = os.path.join(self.temp_dir, "does_not_exist.jpg")

        with pytest.raises(OSError) as exc_info:
            load_slide(missing_path)
        assert not isinstance(exc_info.value, ImageDecodeError)


class TestLogLoadMeasurement:
    """Test suite for log_load_measurement, which appends one CSV row per
    photo load attempt for later offline analysis of size/time/speed
    correlation"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "measurements.csv")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _read_rows(self):
        import csv
        with open(self.log_file, newline="") as f:
            return list(csv.DictReader(f))

    def test_writes_header_once_then_appends(self):
        log_load_measurement(self.log_file, "a.jpg", "ok", load_bytes=1000, load_seconds=0.5)
        log_load_measurement(self.log_file, "b.jpg", "ok", load_bytes=2000, load_seconds=1.0)

        with open(self.log_file) as f:
            lines = f.readlines()

        assert lines[0].strip() == "timestamp,path,outcome,bytes,seconds,bytes_per_sec,error_type"
        assert len(lines) == 3  # header + 2 rows

    def test_ok_outcome_records_full_measurement(self):
        log_load_measurement(self.log_file, "a.jpg", "ok", load_bytes=1000, load_seconds=0.5)

        row = self._read_rows()[0]
        assert row["path"] == "a.jpg"
        assert row["outcome"] == "ok"
        assert int(row["bytes"]) == 1000
        assert float(row["seconds"]) == 0.5
        assert float(row["bytes_per_sec"]) == 2000.0
        assert row["error_type"] == ""

    def test_decode_error_still_records_the_valid_read_measurement(self):
        log_load_measurement(
            self.log_file, "bad.jpg", "decode_error", load_bytes=500, load_seconds=0.25,
            error_type="UnidentifiedImageError",
        )

        row = self._read_rows()[0]
        assert row["outcome"] == "decode_error"
        assert int(row["bytes"]) == 500
        assert float(row["seconds"]) == 0.25
        assert float(row["bytes_per_sec"]) == 2000.0
        assert row["error_type"] == "UnidentifiedImageError"

    def test_io_error_leaves_measurement_fields_blank(self):
        log_load_measurement(self.log_file, "missing.jpg", "io_error", error_type="ConnectionResetError")

        row = self._read_rows()[0]
        assert row["outcome"] == "io_error"
        assert row["bytes"] == ""
        assert row["seconds"] == ""
        assert row["bytes_per_sec"] == ""
        assert row["error_type"] == "ConnectionResetError"

    def test_file_not_found_outcome_is_distinct_from_io_error(self):
        log_load_measurement(self.log_file, "gone.jpg", "file_not_found", error_type="FileNotFoundError")

        row = self._read_rows()[0]
        assert row["outcome"] == "file_not_found"
        assert row["error_type"] == "FileNotFoundError"
        assert row["bytes"] == ""
        assert row["seconds"] == ""


class TestSmoothscaleSafe:
    """Test suite for smoothscale_safe function"""
    
    def setup_method(self):
        """Initialize pygame"""
        pygame.init()
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_scale_24bit_surface(self):
        """Test scaling a 24-bit surface"""
        original = pygame.Surface((100, 100), depth=24)
        scaled = smoothscale_safe(original, (50, 50))
        
        assert scaled.get_width() == 50
        assert scaled.get_height() == 50
    
    def test_scale_32bit_surface(self):
        """Test scaling a 32-bit surface"""
        original = pygame.Surface((100, 100), flags=pygame.SRCALPHA, depth=32)
        scaled = smoothscale_safe(original, (200, 200))
        
        assert scaled.get_width() == 200
        assert scaled.get_height() == 200
    
    def test_scale_8bit_surface(self):
        """Test scaling an 8-bit surface (should be converted first)"""
        original = pygame.Surface((100, 100), depth=8)
        scaled = smoothscale_safe(original, (50, 50))
        
        assert scaled.get_width() == 50
        assert scaled.get_height() == 50


class TestBlitScaled:
    """Test suite for blit_scaled function"""
    
    def setup_method(self):
        """Initialize pygame"""
        pygame.init()
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_blit_scaled_fits_within_target(self):
        """Test that image is scaled to fit within target rect"""
        surface = pygame.Surface((800, 600))
        img = pygame.Surface((400, 300))
        target_rect = pygame.Rect(0, 0, 200, 200)
        
        # Should not raise any errors
        blit_scaled(surface, img, target_rect)
    
    def test_blit_scaled_preserves_aspect_ratio(self):
        """Test that aspect ratio is preserved when scaling"""
        surface = pygame.Surface((800, 600))
        # Wide image
        img = pygame.Surface((400, 100))
        target_rect = pygame.Rect(0, 0, 200, 200)
        
        # Should scale to fit width (200) and maintain aspect ratio
        blit_scaled(surface, img, target_rect)


class TestFormatSpeed:
    """Test suite for format_speed function"""

    def test_none_shows_placeholder(self):
        assert format_speed(None) == "-- Bps"

    def test_stays_in_bps_below_1000(self):
        assert format_speed(999) == "999.00 Bps"

    def test_switches_to_kbps_at_1000(self):
        assert format_speed(1000) == "1.00 KBps"

    def test_switches_to_mbps_just_over_1000_kbps(self):
        # 1001 KBps -> 1.00 MBps
        assert format_speed(1001 * 1000) == "1.00 MBps"

    def test_rounds_half_up_to_two_decimals(self):
        # 12.345 KBps -> 12.35 KBps (round-half-up, not banker's rounding
        # and not naive float rounding, which could give 12.34 instead)
        assert format_speed(12345) == "12.35 KBps"

    def test_rounding_can_push_into_the_next_unit(self):
        # 999.996 KBps rounds to 1000.00 KBps, which should instead
        # display as 1.00 MBps rather than showing "1000.00 KBps"
        assert format_speed(999996) == "1.00 MBps"

    def test_mbps_scale(self):
        assert format_speed(2_500_000) == "2.50 MBps"

    def test_gbps_scale(self):
        assert format_speed(1_500_000_000) == "1.50 GBps"


class TestDrawStatusOverlay:
    """Test suite for draw_status_overlay function"""

    def setup_method(self):
        """Initialize pygame"""
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)

    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()

    def test_draws_light_gray_box_in_bottom_right_corner(self):
        """Test that a light-gray box appears near the bottom-right corner,
        and the rest of the screen is left untouched"""
        screen = pygame.Surface((400, 300))
        screen.fill((0, 0, 0))

        draw_status_overlay(screen, self.font, paused=False, drive_ok=True, download_bytes_per_sec=150.0)

        # draw_status_overlay uses a 10px margin from the screen edge, so
        # probe just inside that margin rather than the literal corner pixel.
        corner_pixel = screen.get_at((388, 288))[:3]
        assert corner_pixel == (211, 211, 211)

        far_pixel = screen.get_at((5, 5))[:3]
        assert far_pixel == (0, 0, 0)

    def test_handles_missing_download_speed(self):
        """Test that a None download_bytes_per_sec (e.g. before the first successful
        load) doesn't crash and still draws the box"""
        screen = pygame.Surface((400, 300))
        screen.fill((0, 0, 0))

        draw_status_overlay(screen, self.font, paused=True, drive_ok=False, download_bytes_per_sec=None)

        corner_pixel = screen.get_at((388, 288))[:3]
        assert corner_pixel == (211, 211, 211)

    def test_box_stays_within_screen_bounds_on_small_screen(self):
        """Test that a very small screen doesn't cause the overlay to error out"""
        screen = pygame.Surface((50, 50))
        screen.fill((0, 0, 0))

        # Should not raise, even though the box may not fully fit
        draw_status_overlay(screen, self.font, paused=False, drive_ok=True, download_bytes_per_sec=42.0)

    def test_returns_its_box_rect(self):
        """Test that the box's rect is returned, so callers can lay out
        other diagnostics (e.g. the load-history histogram) relative to it"""
        screen = pygame.Surface((400, 300))

        box_rect = draw_status_overlay(screen, self.font, paused=False, drive_ok=True, download_bytes_per_sec=42.0)

        assert isinstance(box_rect, pygame.Rect)
        assert box_rect.right == 400 - 10  # 10px margin from the right edge
        assert box_rect.bottom == 300 - 10  # 10px margin from the bottom edge

    def test_box_width_is_stable_across_drive_status_transitions(self):
        """Test that the box (and therefore the histogram to its left) does
        not change width when the drive status text changes -- otherwise the
        histogram would shrink exactly when a disconnect makes "Drive:
        DISCONNECTED" wider than "Drive: OK", i.e. when it's needed most"""
        screen = pygame.Surface((800, 300))

        ok_rect = draw_status_overlay(screen, self.font, paused=False, drive_ok=True, download_bytes_per_sec=1000.0)
        disconnected_rect = draw_status_overlay(screen, self.font, paused=False, drive_ok=False, download_bytes_per_sec=None)

        assert ok_rect.width == disconnected_rect.width
        assert ok_rect.x == disconnected_rect.x


class TestComputePatternRects:
    """Test suite for compute_pattern_rects function"""
    
    def setup_method(self):
        """Initialize pygame and create test slides"""
        pygame.init()
        self.screen = pygame.Surface((900, 600))
        
        # Create dummy slides with surfaces
        self.portrait_slide = Slide(
            path="p.jpg",
            surface=pygame.Surface((100, 200)),
            orientation="P"
        )
        self.landscape_slide = Slide(
            path="l.jpg",
            surface=pygame.Surface((200, 100)),
            orientation="L"
        )
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_pattern_type_1_ppp(self):
        """Test PPP pattern (3 portraits side by side)"""
        slides = [self.portrait_slide, self.portrait_slide, self.portrait_slide]
        rects = compute_pattern_rects(self.screen, slides, pattern_type=1)
        
        assert len(rects) == 3
        # Each should be 1/3 of screen width
        for i, (surf, rect) in enumerate(rects):
            assert rect.width == 300  # 900 / 3
            assert rect.height == 600
            assert rect.x == i * 300
    
    def test_pattern_type_2_pplll(self):
        """Test PPLLL pattern (3 L stacked, 2 P full-height)"""
        slides = [
            self.portrait_slide, self.portrait_slide,
            self.landscape_slide, self.landscape_slide, self.landscape_slide
        ]
        rects = compute_pattern_rects(self.screen, slides, pattern_type=2)
        
        # Should have 5 rects: 3 landscapes + 2 portraits
        assert len(rects) == 5
    
    def test_pattern_type_3_plll(self):
        """Test PLLL pattern (1 P + 3 L)"""
        slides = [
            self.portrait_slide,
            self.landscape_slide, self.landscape_slide, self.landscape_slide
        ]
        rects = compute_pattern_rects(self.screen, slides, pattern_type=3)
        
        # Should have 4 rects
        assert len(rects) == 4


class TestLoadExclusions:
    """Test suite for load_exclusions function"""

    def setup_method(self):
        """Create test controller pointing at a temp exclusions file"""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = SlideshowController()
        self.controller.exclusions_file = os.path.join(self.temp_dir, "exclusions.txt")

    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_missing_file_is_noop(self):
        """Test that a missing exclusions file leaves excluded_paths empty"""
        load_exclusions(self.controller)

        assert self.controller.excluded_paths == set()

    def test_load_populates_excluded_paths(self):
        """Test that existing exclusions file paths are loaded"""
        with open(self.controller.exclusions_file, "w") as f:
            f.write("test1.jpg\n")
            f.write("test2.jpg\n")

        load_exclusions(self.controller)

        assert self.controller.excluded_paths == {"test1.jpg", "test2.jpg"}

    def test_load_skips_blank_lines(self):
        """Test that blank lines in the exclusions file are ignored"""
        with open(self.controller.exclusions_file, "w") as f:
            f.write("test1.jpg\n")
            f.write("\n")
            f.write("   \n")
            f.write("test2.jpg\n")

        load_exclusions(self.controller)

        assert self.controller.excluded_paths == {"test1.jpg", "test2.jpg"}

    def test_load_then_finalize_appends(self):
        """Test that paths loaded at startup persist and new marks append to the file"""
        with open(self.controller.exclusions_file, "w") as f:
            f.write("old.jpg\n")

        load_exclusions(self.controller)
        assert "old.jpg" in self.controller.excluded_paths

        self.controller.current_slides = [
            Slide(path="new.jpg", surface=None, orientation="L")
        ]
        self.controller.current_marks = {0}
        finalize_exclusions(self.controller)

        assert self.controller.excluded_paths == {"old.jpg", "new.jpg"}
        with open(self.controller.exclusions_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert lines == ["old.jpg", "new.jpg"]


class TestLoadSettings:
    """Test suite for load_settings function"""

    def setup_method(self):
        """Create test controller pointing at a temp settings file"""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = SlideshowController()
        self.controller.settings_file = os.path.join(self.temp_dir, "settings.json")

    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_file_keeps_default(self):
        """Test that a missing settings file leaves the default (shuffle
        enabled) untouched"""
        load_settings(self.controller)

        assert self.controller.shuffle_enabled is True

    def test_loads_persisted_shuffle_disabled(self):
        """Test that a persisted shuffle_enabled: false is honored"""
        with open(self.controller.settings_file, "w") as f:
            f.write('{"shuffle_enabled": false}')

        load_settings(self.controller)

        assert self.controller.shuffle_enabled is False

    def test_loads_persisted_shuffle_enabled(self):
        """Test that a persisted shuffle_enabled: true is honored"""
        self.controller.shuffle_enabled = False  # start from the opposite
        with open(self.controller.settings_file, "w") as f:
            f.write('{"shuffle_enabled": true}')

        load_settings(self.controller)

        assert self.controller.shuffle_enabled is True

    def test_corrupt_file_falls_back_to_default_instead_of_crashing(self):
        """Test that invalid JSON doesn't crash startup, just keeps defaults"""
        with open(self.controller.settings_file, "w") as f:
            f.write("not valid json{{{")

        load_settings(self.controller)  # should not raise

        assert self.controller.shuffle_enabled is True


class TestClassifyPatternType:
    """Test suite for classify_pattern_type, the shared P/L threshold
    classifier used by both extract_pattern_from_deque and
    reclassify_pattern_type (kept in one place so the two can't drift)"""

    def test_ppp(self):
        assert classify_pattern_type(count_p=3, count_l=0) == (1, 3, 0)

    def test_pplll(self):
        assert classify_pattern_type(count_p=2, count_l=3) == (2, 2, 3)

    def test_plll(self):
        assert classify_pattern_type(count_p=1, count_l=3) == (3, 1, 3)

    def test_more_than_needed_still_matches_highest_priority_pattern(self):
        # 5 P's and 0 L's still satisfies PPP (only needs 3 P's)
        assert classify_pattern_type(count_p=5, count_l=0) == (1, 3, 0)

    def test_no_match_returns_none(self):
        assert classify_pattern_type(count_p=0, count_l=5) is None
        assert classify_pattern_type(count_p=1, count_l=2) is None
        assert classify_pattern_type(count_p=0, count_l=0) is None


class TestReclassifyPatternType:
    """Test suite for reclassify_pattern_type function"""

    def _slides(self, orientations):
        return [Slide(path=f"{o}{i}.jpg", surface=None, orientation=o) for i, o in enumerate(orientations)]

    def test_empty_slides_drops_entry(self):
        assert reclassify_pattern_type([], 2) is None

    def test_type0_keeps_type_when_nonempty(self):
        slides = self._slides("L")
        assert reclassify_pattern_type(slides, 0) == 0

    def test_type1_keeps_type_as_p_count_shrinks(self):
        slides = self._slides("PP")
        assert reclassify_pattern_type(slides, 1) == 1

    def test_type2_downgrades_to_type3_when_one_p_removed(self):
        # started as PPLLL (2P, 3L); one P got excluded -> 1P, 3L fits PLLL
        slides = self._slides("PLLL")
        assert reclassify_pattern_type(slides, 2) == 3

    def test_type2_dropped_when_l_count_falls_below_3(self):
        # PPLLL lost one L -> 2P, 2L doesn't fit any pattern
        slides = self._slides("PPLL")
        assert reclassify_pattern_type(slides, 2) is None

    def test_type2_dropped_when_all_p_excluded(self):
        # PPLLL lost both P's -> 3 L's alone don't fit any pattern
        slides = self._slides("LLL")
        assert reclassify_pattern_type(slides, 2) is None

    def test_type3_dropped_when_sole_portrait_excluded(self):
        # PLLL lost its only P -> 3 L's alone don't fit any pattern
        slides = self._slides("LLL")
        assert reclassify_pattern_type(slides, 3) is None

    def test_type3_downgrades_to_single_slide_when_only_one_remains(self):
        slides = self._slides("L")
        assert reclassify_pattern_type(slides, 3) == 0


class TestFinalizeExclusions:
    """Test suite for finalize_exclusions function"""
    
    def setup_method(self):
        """Initialize pygame and create test controller"""
        pygame.init()
        self.temp_dir = tempfile.mkdtemp()
        self.controller = SlideshowController()
        self.controller.exclusions_file = os.path.join(self.temp_dir, "exclusions.txt")
    
    def teardown_method(self):
        """Clean up test files and pygame"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        pygame.quit()
    
    def test_finalize_no_marks(self):
        """Test that finalize with no marks does nothing"""
        self.controller.current_slides = [
            Slide(path="test1.jpg", surface=pygame.Surface((100, 100)), orientation="L")
        ]
        
        finalize_exclusions(self.controller)
        
        assert len(self.controller.excluded_paths) == 0
        assert not os.path.exists(self.controller.exclusions_file)
    
    def test_finalize_with_marks(self):
        """Test that marked slides are added to exclusions"""
        self.controller.current_slides = [
            Slide(path="test1.jpg", surface=pygame.Surface((100, 100)), orientation="L"),
            Slide(path="test2.jpg", surface=pygame.Surface((100, 100)), orientation="L")
        ]
        self.controller.current_marks = {0, 1}
        
        finalize_exclusions(self.controller)
        
        assert "test1.jpg" in self.controller.excluded_paths
        assert "test2.jpg" in self.controller.excluded_paths
        assert len(self.controller.current_marks) == 0
        assert os.path.exists(self.controller.exclusions_file)
    
    def test_finalize_cleans_history(self):
        """Test that excluded paths are removed from history"""
        slide1 = Slide(path="test1.jpg", surface=pygame.Surface((100, 100)), orientation="L")
        slide2 = Slide(path="test2.jpg", surface=pygame.Surface((100, 100)), orientation="L")
        
        self.controller.current_slides = [slide1]
        self.controller.current_marks = {0}
        self.controller.history = [([slide1, slide2], 0)]
        self.controller.history_index = 0
        
        finalize_exclusions(self.controller)
        
        # slide1 should be removed from history
        assert len(self.controller.history) == 1
        assert len(self.controller.history[0][0]) == 1
        assert self.controller.history[0][0][0].path == "test2.jpg"

    def test_finalize_reclassifies_broken_pattern_in_history(self):
        """Marking a P slide on a PPLLL screen should downgrade a matching
        history entry to PLLL instead of leaving a broken pattern_type=2 entry"""
        p1 = Slide(path="p1.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        p2 = Slide(path="p2.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        l1 = Slide(path="l1.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l2 = Slide(path="l2.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l3 = Slide(path="l3.jpg", surface=pygame.Surface((20, 10)), orientation="L")

        self.controller.current_slides = [p1, p2, l1, l2, l3]
        self.controller.current_marks = {0}  # mark p1
        self.controller.history = [([p1, p2, l1, l2, l3], 2)]
        self.controller.history_index = 0

        finalize_exclusions(self.controller)

        assert len(self.controller.history) == 1
        remaining_slides, ptype = self.controller.history[0]
        assert ptype == 3
        assert [s.path for s in remaining_slides] == ["p2.jpg", "l1.jpg", "l2.jpg", "l3.jpg"]

    def test_finalize_drops_history_entry_that_no_longer_fits_any_pattern(self):
        """Excluding both portraits from a PPLLL screen leaves 3 L's, which
        doesn't fit any known pattern, so the whole entry should be dropped"""
        p1 = Slide(path="p1.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        p2 = Slide(path="p2.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        l1 = Slide(path="l1.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l2 = Slide(path="l2.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l3 = Slide(path="l3.jpg", surface=pygame.Surface((20, 10)), orientation="L")

        self.controller.current_slides = [p1, p2, l1, l2, l3]
        self.controller.current_marks = {0, 1}  # mark both P's
        self.controller.history = [([p1, p2, l1, l2, l3], 2)]
        self.controller.history_index = 0

        finalize_exclusions(self.controller)

        assert self.controller.history == []
        assert self.controller.history_index == -1

    def test_finalize_history_index_follows_viewed_entry_when_earlier_entry_dropped(self):
        """If an entry BEFORE the currently-viewed one gets dropped, history_index
        should follow the viewed entry to its new position, not drift onto
        whatever entry now occupies the old numeric index"""
        shared = Slide(path="shared.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        l1 = Slide(path="l1.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l2 = Slide(path="l2.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l3 = Slide(path="l3.jpg", surface=pygame.Surface((20, 10)), orientation="L")

        # Entry A: PLLL whose sole portrait is the same photo about to be
        # excluded from the currently-viewed screen -- A should get dropped.
        entry_a_slides = [
            Slide(path="shared.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
            l1, l2, l3,
        ]

        # Entry B: the currently-viewed screen, a PPP with "shared.jpg" as
        # one of three portraits -- degrades gracefully (stays type 1) and
        # survives the exclusion.
        p2 = Slide(path="p2.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        p3 = Slide(path="p3.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        entry_b_slides = [shared, p2, p3]

        # Entry C: unrelated, untouched.
        entry_c_slides = [
            Slide(path="c1.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
            Slide(path="c2.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
            Slide(path="c3.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
        ]

        self.controller.current_slides = entry_b_slides
        self.controller.current_marks = {0}  # mark "shared.jpg"
        self.controller.history = [
            (entry_a_slides, 3),
            (entry_b_slides, 1),
            (entry_c_slides, 1),
        ]
        self.controller.history_index = 1  # viewing B

        finalize_exclusions(self.controller)

        # A should have been dropped (its only P was excluded, leaving 3 L's,
        # which fits no pattern)
        assert len(self.controller.history) == 2
        remaining_paths = [[s.path for s in slides] for slides, _ in self.controller.history]
        assert remaining_paths[0] == ["p2.jpg", "p3.jpg"]  # B, shared.jpg removed
        assert remaining_paths[1] == ["c1.jpg", "c2.jpg", "c3.jpg"]  # C, untouched

        # history_index must still point at B (now at position 0), not
        # drift onto C just because C used to be at index 2.
        assert self.controller.history_index == 0

    def test_finalize_history_index_clamps_when_viewed_entry_itself_is_dropped(self):
        """If the currently-viewed entry itself no longer fits any pattern
        after exclusion, history_index should clamp into the remaining
        range rather than reference a slot that no longer exists"""
        p1 = Slide(path="p1.jpg", surface=pygame.Surface((10, 20)), orientation="P")
        l1 = Slide(path="l1.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l2 = Slide(path="l2.jpg", surface=pygame.Surface((20, 10)), orientation="L")
        l3 = Slide(path="l3.jpg", surface=pygame.Surface((20, 10)), orientation="L")

        entry_a_slides = [
            Slide(path="a1.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
            Slide(path="a2.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
            Slide(path="a3.jpg", surface=pygame.Surface((10, 20)), orientation="P"),
        ]
        entry_b_slides = [p1, l1, l2, l3]  # PLLL, the currently-viewed screen

        self.controller.current_slides = entry_b_slides
        self.controller.current_marks = {0}  # mark the sole P on this screen
        self.controller.history = [
            (entry_a_slides, 1),
            (entry_b_slides, 3),
        ]
        self.controller.history_index = 1  # viewing B

        finalize_exclusions(self.controller)

        # B is dropped (0 P's, 3 L's fits no pattern); only A remains
        assert len(self.controller.history) == 1
        assert self.controller.history_index == 0


class TestDownscaleSlideToScreen:
    """Test suite for downscale_slide_to_screen function"""
    
    def setup_method(self):
        """Initialize pygame"""
        pygame.init()
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_downscale_large_slide(self):
        """Test that large slide is downscaled"""
        slide = Slide(
            path="large.jpg",
            surface=pygame.Surface((2000, 1500)),
            orientation="L"
        )
        
        downscale_slide_to_screen(slide, 1920, 1080)
        
        # Should be scaled down
        assert slide.surface.get_width() <= 1920
        assert slide.surface.get_height() <= 1080
    
    def test_no_downscale_small_slide(self):
        """Test that small slide is not upscaled"""
        slide = Slide(
            path="small.jpg",
            surface=pygame.Surface((100, 100)),
            orientation="L"
        )
        original_w = slide.surface.get_width()
        original_h = slide.surface.get_height()
        
        downscale_slide_to_screen(slide, 1920, 1080)
        
        # Should remain the same
        assert slide.surface.get_width() == original_w
        assert slide.surface.get_height() == original_h


class TestReadFileList:
    """Test suite for read_file_list function"""
    
    def setup_method(self):
        """Create test file list"""
        self.temp_dir = tempfile.mkdtemp()
        self.list_path = os.path.join(self.temp_dir, "test.list")
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_read_jpg_files(self):
        """Test reading JPG files from list"""
        with open(self.list_path, "w") as f:
            f.write("image1.jpg\n")
            f.write("image2.JPG\n")
            f.write("image3.jpeg\n")
            f.write("image4.JPEG\n")
        
        paths = read_file_list(self.list_path)
        
        assert len(paths) == 4
        assert "image1.jpg" in paths
        assert "image2.JPG" in paths
        assert "image3.jpeg" in paths
        assert "image4.JPEG" in paths
    
    def test_skip_comments_and_empty_lines(self):
        """Test that comments and empty lines are skipped"""
        with open(self.list_path, "w") as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("image1.jpg\n")
            f.write("  \n")
            f.write("image2.jpg\n")
        
        paths = read_file_list(self.list_path)
        
        assert len(paths) == 2
    
    def test_skip_non_jpg_files(self):
        """Test that non-JPG files are filtered out"""
        with open(self.list_path, "w") as f:
            f.write("image1.jpg\n")
            f.write("image2.png\n")
            f.write("image3.gif\n")
            f.write("video.mp4\n")
        
        paths = read_file_list(self.list_path)
        
        assert len(paths) == 1
        assert paths[0] == "image1.jpg"
    
    def test_empty_file_list(self):
        """Test reading an empty file list"""
        with open(self.list_path, "w") as f:
            f.write("")

        paths = read_file_list(self.list_path)

        assert len(paths) == 0

    def test_shuffles_the_full_list_not_just_a_rotation(self):
        """Test that the display order is a genuine shuffle (random.shuffle),
        not the old behavior of rotating by a random offset"""
        expected = [f"image{i}.jpg" for i in range(10)]
        with open(self.list_path, "w") as f:
            for name in expected:
                f.write(name + "\n")

        with patch("random.shuffle") as mock_shuffle:
            read_file_list(self.list_path)

        mock_shuffle.assert_called_once()
        shuffled_arg = mock_shuffle.call_args[0][0]
        assert sorted(shuffled_arg) == sorted(expected)

    def test_shuffle_preserves_every_entry_exactly_once(self):
        """Test that shuffling doesn't drop or duplicate any path"""
        expected = [f"image{i}.jpg" for i in range(50)]
        with open(self.list_path, "w") as f:
            for name in expected:
                f.write(name + "\n")

        paths = read_file_list(self.list_path)

        assert sorted(paths) == sorted(expected)

    def test_shuffle_false_rotates_by_random_offset_preserving_relative_order(self):
        """Test that shuffle=False restores the original rotation behavior:
        same relative order as the file, just starting from a random point"""
        expected = [f"image{i}.jpg" for i in range(10)]
        with open(self.list_path, "w") as f:
            for name in expected:
                f.write(name + "\n")

        with patch("random.randrange", return_value=3):
            paths = read_file_list(self.list_path, shuffle=False)

        assert paths == expected[3:] + expected[:3]

    def test_shuffle_false_does_not_call_random_shuffle(self):
        """Test that shuffle=False takes the rotation path, not the shuffle one"""
        with open(self.list_path, "w") as f:
            f.write("image1.jpg\nimage2.jpg\n")

        with patch("random.shuffle") as mock_shuffle:
            read_file_list(self.list_path, shuffle=False)

        mock_shuffle.assert_not_called()


class TestSetupLogging:
    """Test suite for setup_logging, which routes every exception (caught
    and logged, or truly uncaught) to a rotating log file for later
    analysis, in addition to the normal terminal/journal output"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_errors.log")

        # setup_logging mutates process-wide state (root logger handlers,
        # sys.excepthook, threading.excepthook, and its own "already ran"
        # guard) -- snapshot it all so each test starts clean and other
        # tests/files aren't affected by what runs here.
        self.root_logger = logging.getLogger()
        self.prev_handlers = list(self.root_logger.handlers)
        self.prev_level = self.root_logger.level
        self.prev_sys_excepthook = sys.excepthook
        self.prev_thread_excepthook = threading.excepthook
        self.prev_configured = py_frame._logging_configured
        py_frame._logging_configured = False

    def teardown_method(self):
        for h in list(self.root_logger.handlers):
            if h not in self.prev_handlers:
                self.root_logger.removeHandler(h)
        self.root_logger.setLevel(self.prev_level)
        sys.excepthook = self.prev_sys_excepthook
        threading.excepthook = self.prev_thread_excepthook
        py_frame._logging_configured = self.prev_configured

        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_logged_exception_appears_in_the_file(self):
        """Test that logger.error(..., exc_info=True) anywhere in the app
        ends up in the log file with a full traceback"""
        with patch("py_frame.ERROR_LOG_FILE", self.log_file):
            setup_logging()

        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("py_frame").error("something failed", exc_info=True)

        with open(self.log_file) as f:
            content = f.read()

        assert "something failed" in content
        assert "ValueError: boom" in content

    def test_is_idempotent(self):
        """Test that calling setup_logging twice doesn't add duplicate
        handlers (e.g. across multiple test runs in the same process)"""
        with patch("py_frame.ERROR_LOG_FILE", self.log_file):
            setup_logging()
            handlers_after_first = len(self.root_logger.handlers)
            setup_logging()
            handlers_after_second = len(self.root_logger.handlers)

        assert handlers_after_first == handlers_after_second

    def test_uncaught_thread_exception_is_logged(self):
        """Test that an uncaught exception in a background thread is
        logged via threading.excepthook, not just silently printed"""
        with patch("py_frame.ERROR_LOG_FILE", self.log_file):
            setup_logging()

        def boom():
            raise RuntimeError("thread boom")

        t = threading.Thread(target=boom)
        t.start()
        t.join()

        with open(self.log_file) as f:
            content = f.read()

        assert "Uncaught exception in thread" in content
        assert "RuntimeError: thread boom" in content


class TestMainEmptyFileList:
    """Test suite for main()'s handling of an empty/invalid photo list"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.list_path = os.path.join(self.temp_dir, "empty.list")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_main_exits_when_no_valid_photos(self):
        """main() should print an error and exit(1) instead of silently
        starting a slideshow with nothing to show"""
        with open(self.list_path, "w") as f:
            f.write("not_a_photo.txt\n")

        # setup_logging() has real side effects (creates a log file in the
        # cwd, mutates sys.excepthook/threading.excepthook) that are
        # irrelevant to this test and would otherwise leak into the repo.
        with patch("py_frame.setup_logging"):
            with patch("sys.argv", ["py_frame.py", self.list_path]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1


class _StopFetcher(Exception):
    """Sentinel used to break image_fetcher_thread's infinite loop in tests."""
    pass


class TestImageFetcherThreadThrottling:
    """Test suite for image_fetcher_thread's busy-loop throttling on skip/failure paths"""

    def setup_method(self):
        self.controller = SlideshowController()
        # Needed for the success-path test, which performs a real load_slide()
        # call ending in Surface.convert(), which requires a display mode.
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.display.set_mode((1, 1))

        self.measurements_dir = tempfile.mkdtemp()
        self.controller.measurements_file = os.path.join(self.measurements_dir, "measurements.csv")

    def teardown_method(self):
        pygame.quit()
        import shutil
        shutil.rmtree(self.measurements_dir, ignore_errors=True)

    def _read_measurement_rows(self):
        import csv
        with open(self.controller.measurements_file, newline="") as f:
            return list(csv.DictReader(f))

    def _run_with_bounded_sleep(self, file_paths, max_calls=3):
        """Run image_fetcher_thread with time.sleep mocked to raise after
        max_calls, so the otherwise-infinite loop stops deterministically."""
        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= max_calls:
                raise _StopFetcher()

        dq = deque()
        lock = threading.Lock()
        not_full = threading.Condition(lock)
        producer_done = threading.Event()

        def quiet_excepthook(args):
            if args.exc_type is not _StopFetcher:
                threading.__excepthook__(args)

        prev_excepthook = threading.excepthook
        threading.excepthook = quiet_excepthook
        try:
            with patch("time.sleep", side_effect=fake_sleep):
                t = threading.Thread(
                    target=image_fetcher_thread,
                    args=(file_paths, dq, lock, not_full, producer_done, self.controller, 5),
                    daemon=True,
                )
                t.start()
                t.join(timeout=2)
        finally:
            threading.excepthook = prev_excepthook

        assert not t.is_alive(), "fetcher thread did not stop after the bounded sleep raised"
        return sleep_calls, dq

    def test_excluded_path_throttles_instead_of_busy_looping(self):
        """All paths excluded -> should sleep between skips, not spin"""
        self.controller.excluded_paths.add("excluded.jpg")

        sleep_calls, dq = self._run_with_bounded_sleep(["excluded.jpg"])

        assert sleep_calls, "expected time.sleep to be called while skipping excluded paths"
        assert all(c == 0.3 for c in sleep_calls)
        assert len(dq) == 0

    def test_load_failure_throttles_instead_of_busy_looping(self):
        """Unloadable path (e.g. missing file) -> should sleep between retries, not spin"""
        sleep_calls, dq = self._run_with_bounded_sleep(
            ["/nonexistent/path/does_not_exist.jpg"]
        )

        assert sleep_calls, "expected time.sleep to be called after a failed load"
        assert all(c == 0.5 for c in sleep_calls)
        assert len(dq) == 0

    def test_generic_load_failure_marks_drive_not_ok(self):
        """A genuine I/O failure (not a missing file) should flag the drive
        as unreadable and clear the speed estimate, so the on-screen
        diagnostics reflect the stall"""
        self.controller.drive_ok = True
        self.controller.download_bytes_per_sec = 123.0

        with patch("py_frame.load_slide", side_effect=ConnectionResetError("simulated reset")):
            self._run_with_bounded_sleep(["irrelevant-path.jpg"])

        assert self.controller.drive_ok is False
        assert self.controller.download_bytes_per_sec is None

        rows = self._read_measurement_rows()
        assert rows[-1]["outcome"] == "io_error"
        assert rows[-1]["error_type"] == "ConnectionResetError"
        assert rows[-1]["bytes"] == ""
        assert rows[-1]["seconds"] == ""

    def test_file_not_found_does_not_mark_drive_disconnected(self):
        """A single missing file (deleted/renamed on the NAS, a stale list
        entry, etc) should NOT be reported as a drive disconnect -- it says
        nothing about whether the mount itself is reachable"""
        self.controller.drive_ok = True
        self.controller.download_bytes_per_sec = 123.0

        self._run_with_bounded_sleep(["/nonexistent/path/does_not_exist.jpg"])

        # Recorded distinctly from a generic io_error...
        rows = self._read_measurement_rows()
        assert rows[-1]["outcome"] == "file_not_found"
        assert rows[-1]["error_type"] == "FileNotFoundError"
        assert rows[-1]["bytes"] == ""
        assert rows[-1]["seconds"] == ""

        # ...and the drive/speed diagnostics are left untouched.
        assert self.controller.drive_ok is True
        assert self.controller.download_bytes_per_sec == 123.0

    def test_corrupt_file_does_not_mark_drive_disconnected(self):
        """A bad/corrupt file (readable, but not a valid image) should NOT
        be reported as a drive disconnect -- only genuine I/O failures
        (missing file, unreadable mount, etc) should flip drive_ok"""
        temp_dir = tempfile.mkdtemp()
        try:
            bad_path = os.path.join(temp_dir, "corrupt.jpg")
            with open(bad_path, "wb") as f:
                f.write(b"not a real jpeg")

            self.controller.drive_ok = True
            self.controller.download_bytes_per_sec = 123.0

            self._run_with_bounded_sleep([bad_path])

            # The read succeeded, so it's still logged as a valid
            # size/time measurement (just flagged as a decode error)...
            rows = self._read_measurement_rows()
            assert rows[-1]["outcome"] == "decode_error"
            assert int(rows[-1]["bytes"]) == len(b"not a real jpeg")
            assert float(rows[-1]["seconds"]) >= 0

            # ...but the drive itself is not reported as disconnected, and
            # the existing speed estimate survives (unlike a real I/O failure).
            assert self.controller.drive_ok is True
            assert self.controller.download_bytes_per_sec == 123.0
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_successful_load_marks_drive_ok_and_reports_speed(self):
        """A successful load should flag the drive as OK and compute a
        Kbps estimate from the read time/size"""
        temp_dir = tempfile.mkdtemp()
        try:
            img_path = os.path.join(temp_dir, "photo.jpg")
            Image.new("RGB", (50, 50), color="blue").save(img_path)

            self.controller.drive_ok = False
            self.controller.download_bytes_per_sec = None

            dq = deque()
            lock = threading.Lock()
            not_full = threading.Condition(lock)
            producer_done = threading.Event()

            def fake_notify_all():
                raise _StopFetcher()

            def quiet_excepthook(args):
                if args.exc_type is not _StopFetcher:
                    threading.__excepthook__(args)

            prev_excepthook = threading.excepthook
            threading.excepthook = quiet_excepthook
            prev_notify_all = not_full.notify_all
            not_full.notify_all = fake_notify_all
            try:
                t = threading.Thread(
                    target=image_fetcher_thread,
                    args=([img_path], dq, lock, not_full, producer_done, self.controller, 5),
                    daemon=True,
                )
                t.start()
                t.join(timeout=2)
            finally:
                threading.excepthook = prev_excepthook
                not_full.notify_all = prev_notify_all

            assert not t.is_alive(), "fetcher thread did not stop after the bounded notify_all raised"
            assert self.controller.drive_ok is True
            assert self.controller.download_bytes_per_sec is not None
            assert self.controller.download_bytes_per_sec >= 0

            rows = self._read_measurement_rows()
            assert rows[-1]["outcome"] == "ok"
            assert int(rows[-1]["bytes"]) == os.path.getsize(img_path)
            assert float(rows[-1]["seconds"]) >= 0
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
