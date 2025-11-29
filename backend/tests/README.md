# Backend Tests

## Setup

Install test dependencies:

```bash
# If using uv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Running Tests

### Integration Tests (requires GEMINI_API_KEY)

These tests make actual calls to the Gemini API:

```bash
# Set your API key
export GEMINI_API_KEY=your_key_here

# Run all integration tests
pytest tests/test_extraction_integration.py -v

# Run a specific test
pytest tests/test_extraction_integration.py::test_extract_paper_sections_integration -v

# Run with output capture disabled to see print statements
pytest tests/test_extraction_integration.py -v -s

# Skip integration tests (if you want to run only unit tests later)
pytest -m "not integration"
```

## Test Fixtures

- `tests/fixtures/test_paper.pdf`: A test PDF paper used for integration tests

## Test Structure

- `test_extraction_integration.py`: Integration tests that call the actual Gemini API
  - `test_extract_paper_sections_integration`: Tests section extraction
  - `test_extract_paper_images_integration`: Tests image extraction
  - `test_extract_paper_full_integration`: Tests both extractions together

