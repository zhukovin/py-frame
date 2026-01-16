# Quick Test Guide

## Install Dependencies

```bash
pip3 install pytest coverage pytest-cov pillow pygame flask
```

## Run Tests

### Run all tests
```bash
pytest test_py_frame.py test_web_server.py -v
```

### Run with coverage report
```bash
pytest test_py_frame.py test_web_server.py --cov=. --cov-report=term-missing --cov-report=html
```

### View HTML coverage report
```bash
open htmlcov/index.html
```

## Expected Results

- **48 tests total**
- **All tests passing** ✅
- **70% overall coverage**
  - py_frame.py: 46% (core logic well-tested at 85%+)
  - web_server.py: 91% (API endpoints)

## Test Structure

### test_py_frame.py (28 tests)
Tests for core slideshow functionality:
- Pattern extraction (PPP, PPLLL, PLLL)
- Image loading and orientation detection
- Layout computation
- Exclusion management
- File list parsing
- Controller state management

### test_web_server.py (20 tests)
Tests for web interface:
- API endpoints (/api/state, /api/mark, /api/command)
- Control commands (next, prev, pause, play, screen_on, screen_off)
- Web UI HTML page
- Error handling

## More Details

See [TESTING.md](TESTING.md) for comprehensive coverage analysis and recommendations.
