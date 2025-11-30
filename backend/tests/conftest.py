"""
Pytest configuration and shared fixtures for backend tests.
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    mock = Mock()
    mock.storage.from_.return_value.upload.return_value = None
    mock.storage.from_.return_value.get_public_url.return_value = "https://example.com/file.mp3"
    mock.storage.from_.return_value.remove.return_value = None
    return mock


@pytest.fixture
def mock_genai_client():
    """Mock Gemini AI client."""
    mock = Mock()
    return mock


@pytest.fixture
def sample_paper():
    """Sample paper data for testing."""
    return {
        "id": "test-paper-123",
        "title": "Test Research Paper on Machine Learning",
        "authors": "John Doe, Jane Smith",
        "year": 2024,
        "source_url": "https://example.com/paper.pdf"
    }


@pytest.fixture
def sample_episode():
    """Sample podcast episode data for testing."""
    return {
        "id": "episode-123",
        "paper_id": "paper-123",
        "title": "Podcast Episode Title",
        "description": "Episode description",
        "audio_url": "https://example.com/audio.mp3",
        "duration_seconds": 600,
        "published_at": "2024-01-15T10:00:00Z",
        "generation_status": "completed"
    }
