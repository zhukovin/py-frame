"""
Comprehensive test suite for web_server.py
Tests for Flask web interface and API endpoints
"""
import pytest
import pygame
from unittest.mock import Mock, MagicMock
from flask import Flask
from web_server import create_app
from py_frame import SlideshowController, Slide


class TestWebServer:
    """Test suite for web server endpoints"""
    
    def setup_method(self):
        """Setup test client and controller"""
        pygame.init()
        pygame.display.set_mode((1, 1))
        
        self.controller = SlideshowController()
        self.app = create_app(self.controller)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def teardown_method(self):
        """Clean up pygame"""
        pygame.quit()
    
    def test_api_state_empty(self):
        """Test /api/state endpoint with no slides"""
        response = self.client.get('/api/state')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'slides' in data
        assert 'paused' in data
        assert 'black' in data
        assert len(data['slides']) == 0
        assert data['paused'] is False
        assert data['black'] is False
    
    def test_api_state_with_slides(self):
        """Test /api/state endpoint with slides"""
        # Add some slides to controller
        slide1 = Slide(path="test1.jpg", surface=pygame.Surface((100, 100)), orientation="L")
        slide2 = Slide(path="test2.jpg", surface=pygame.Surface((100, 100)), orientation="P")
        
        self.controller.current_slides = [slide1, slide2]
        self.controller.current_pattern_type = 1
        self.controller.current_marks.add(0)
        
        response = self.client.get('/api/state')
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['slides']) == 2
        assert data['slides'][0]['path'] == "test1.jpg"
        assert data['slides'][0]['marked'] is True
        assert data['slides'][1]['path'] == "test2.jpg"
        assert data['slides'][1]['marked'] is False
        assert data['slides'][0]['pattern_type'] == 1
    
    def test_api_state_paused(self):
        """Test /api/state reflects paused state"""
        self.controller.paused = True
        self.controller.black_screen = True
        
        response = self.client.get('/api/state')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['paused'] is True
        assert data['black'] is True
    
    def _add_current_slides(self, count, orientation="L"):
        self.controller.current_slides = [
            Slide(path=f"test{i}.jpg", surface=pygame.Surface((10, 10)), orientation=orientation)
            for i in range(count)
        ]

    def test_api_mark_valid_slot(self):
        """Test /api/mark endpoint with valid slot"""
        self._add_current_slides(3)
        response = self.client.post('/api/mark', json={'slot': 2})

        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert 2 in self.controller.current_marks

    def test_api_mark_toggle(self):
        """Test /api/mark toggles marking"""
        self._add_current_slides(2)

        # Mark slot 1
        self.client.post('/api/mark', json={'slot': 1})
        assert 1 in self.controller.current_marks

        # Unmark slot 1
        self.client.post('/api/mark', json={'slot': 1})
        assert 1 not in self.controller.current_marks

    def test_api_mark_invalid_slot_negative(self):
        """Test /api/mark with invalid negative slot"""
        self._add_current_slides(1)
        response = self.client.post('/api/mark', json={'slot': -1})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False
        assert 'error' in data

    def test_api_mark_invalid_slot_too_large(self):
        """Test /api/mark with a slot beyond the number of current slides"""
        self._add_current_slides(3)
        response = self.client.post('/api/mark', json={'slot': 3})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False

    def test_api_mark_no_current_slides(self):
        """Test /api/mark rejects any slot when there are no current slides
        (guards against marking a slot with no photo behind it)"""
        response = self.client.post('/api/mark', json={'slot': 0})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False

    def test_api_mark_non_numeric_slot(self):
        """Test /api/mark with a non-numeric slot returns 400 instead of crashing"""
        self._add_current_slides(3)
        response = self.client.post('/api/mark', json={'slot': 'abc'})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False

    def test_api_mark_missing_slot(self):
        """Test /api/mark with missing slot parameter"""
        self._add_current_slides(3)
        response = self.client.post('/api/mark', json={})

        assert response.status_code == 400
    
    def test_api_command_next(self):
        """Test /api/command endpoint with next command"""
        response = self.client.post('/api/command', json={'cmd': 'next', 'steps': 1})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command == {'type': 'next', 'steps': 1}
    
    def test_api_command_prev(self):
        """Test /api/command endpoint with prev command"""
        response = self.client.post('/api/command', json={'cmd': 'prev', 'steps': 3})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command == {'type': 'prev', 'steps': 3}
    
    def test_api_command_pause(self):
        """Test /api/command endpoint with pause command"""
        response = self.client.post('/api/command', json={'cmd': 'pause'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command == {'type': 'pause'}
    
    def test_api_command_play(self):
        """Test /api/command endpoint with play command"""
        response = self.client.post('/api/command', json={'cmd': 'play'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command == {'type': 'play'}
    
    def test_api_command_screen_off(self):
        """Test /api/command endpoint with screen_off command"""
        response = self.client.post('/api/command', json={'cmd': 'screen_off'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command['type'] == 'screen_off'
    
    def test_api_command_screen_on(self):
        """Test /api/command endpoint with screen_on command"""
        response = self.client.post('/api/command', json={'cmd': 'screen_on'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ok'] is True
        assert self.controller.pending_command['type'] == 'screen_on'
    
    def test_api_command_default_steps(self):
        """Test /api/command with default steps value"""
        response = self.client.post('/api/command', json={'cmd': 'next'})
        
        assert response.status_code == 200
        assert self.controller.pending_command['steps'] == 1
    
    def test_api_command_invalid(self):
        """Test /api/command with invalid command"""
        response = self.client.post('/api/command', json={'cmd': 'invalid_cmd'})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False
        assert 'error' in data

    def test_api_command_non_numeric_steps(self):
        """Test /api/command with a non-numeric steps value returns 400 instead of crashing"""
        response = self.client.post('/api/command', json={'cmd': 'next', 'steps': 'abc'})

        assert response.status_code == 400
        data = response.get_json()
        assert data['ok'] is False

    def test_api_command_missing_cmd(self):
        """Test /api/command with missing cmd parameter"""
        response = self.client.post('/api/command', json={})
        
        assert response.status_code == 400
    
    def test_index_page(self):
        """Test that / endpoint returns HTML page"""
        response = self.client.get('/')
        
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data
        assert b'Frame Control' in response.data
        assert b'/api/state' in response.data
        assert b'Pause' in response.data
        assert b'Play' in response.data
    
    def test_index_page_has_controls(self):
        """Test that index page has all control buttons"""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        
        # Check for all control buttons
        assert 'Pause' in html
        assert 'Play' in html
        assert 'Prev' in html
        assert 'Next' in html
        assert 'Screen Off' in html
        assert 'Screen On' in html
    
    def test_index_page_has_javascript(self):
        """Test that index page includes JavaScript for interaction"""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        
        # Check for key JavaScript functions
        assert 'refreshState' in html
        assert 'toggleMark' in html
        assert 'sendCommand' in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
