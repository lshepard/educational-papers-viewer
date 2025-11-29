"""
Pytest configuration and fixtures for backend tests.
"""
import os
import pytest
from pathlib import Path

# Get the backend directory (parent of tests)
BACKEND_DIR = Path(__file__).parent.parent
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_pdf_path():
    """Fixture that returns the path to the test PDF."""
    pdf_path = TEST_FIXTURES_DIR / "test_paper.pdf"
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    return str(pdf_path)


@pytest.fixture
def gemini_api_key():
    """Fixture that returns the Gemini API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in environment")
    return api_key


