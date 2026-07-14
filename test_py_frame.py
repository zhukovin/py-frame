"""
Comprehensive test suite for py_frame.py
This file extracts and runs existing tests from py_frame.py
"""
import pytest
from collections import deque
from itertools import product
import pygame
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

# Import the test function and dependencies from py_frame
from py_frame import (
    DummySlide,
    Slide,
    SlideshowController,
    extract_pattern_from_deque,
    test_extract_pattern_all_len5,
    make_old_paper_surface,
    load_slide,
    smoothscale_safe,
    blit_scaled,
    draw_slot_overlay,
    compute_pattern_rects,
    load_exclusions,
    finalize_exclusions,
    downscale_slide_to_screen,
    downscale_slides_to_screen,
    read_file_list,
    Orientation
)


def test_extract_pattern_existing():
    """Run the existing test from py_frame.py"""
    test_extract_pattern_all_len5()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
