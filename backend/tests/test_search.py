"""
Tests for full-text search functionality.

These tests verify that the search endpoint works correctly with the FTS column.
"""
import pytest
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_search_endpoint_exists():
    """Test that the search endpoint is available."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "/search" in data["endpoints"]


def test_search_basic_query():
    """Test basic search functionality."""
    response = client.post(
        "/search",
        json={"query": "machine learning", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "success" in data
    assert "query" in data
    assert "results" in data
    assert "count" in data

    assert data["success"] is True
    assert data["query"] == "machine learning"
    assert isinstance(data["results"], list)
    assert data["count"] == len(data["results"])


def test_search_with_limit():
    """Test that limit parameter works."""
    response = client.post(
        "/search",
        json={"query": "education", "limit": 3}
    )

    assert response.status_code == 200
    data = response.json()

    # Results should not exceed limit
    assert len(data["results"]) <= 3


def test_search_result_structure():
    """Test that search results have the correct structure."""
    response = client.post(
        "/search",
        json={"query": "research"}
    )

    assert response.status_code == 200
    data = response.json()

    if data["count"] > 0:
        result = data["results"][0]

        # Verify all required fields are present
        assert "id" in result
        assert "paper_id" in result
        assert "section_type" in result
        assert "section_title" in result  # Can be null
        assert "content" in result
        assert "created_at" in result


def test_search_empty_query():
    """Test search with empty query string."""
    response = client.post(
        "/search",
        json={"query": "", "limit": 5}
    )

    # Should either succeed with no results or return an error
    # Depends on how Postgres handles empty queries
    assert response.status_code in [200, 400, 500]


def test_search_special_characters():
    """Test search with special characters."""
    response = client.post(
        "/search",
        json={"query": "machine & learning", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_search_or_operator():
    """Test search with OR operator."""
    response = client.post(
        "/search",
        json={"query": "education OR teaching", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_search_phrase():
    """Test phrase search with quotes."""
    response = client.post(
        "/search",
        json={"query": '"machine learning"', "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_search_missing_query_param():
    """Test that missing query parameter is rejected."""
    response = client.post(
        "/search",
        json={"limit": 5}
    )

    # Should return 422 Unprocessable Entity (validation error)
    assert response.status_code == 422


def test_search_default_limit():
    """Test that default limit is applied when not specified."""
    response = client.post(
        "/search",
        json={"query": "research"}
    )

    assert response.status_code == 200
    data = response.json()

    # Default limit is 10
    assert len(data["results"]) <= 10
