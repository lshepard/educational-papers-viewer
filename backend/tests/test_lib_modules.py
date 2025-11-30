"""
Comprehensive tests for lib modules - happy path testing.
"""

import pytest
from lib.storage import upload_audio_to_storage, get_public_url, delete_from_storage
from lib.pdf_analyzer import create_paper_slug, PaperSections
from lib.podcast_generator import convert_audio_to_mp3
from lib.rss_feed import format_duration, format_rfc2822_date


class TestStorage:
    """Tests for lib/storage.py"""

    def test_get_public_url(self, mock_supabase):
        """Test getting public URL for a storage object."""
        url = get_public_url(mock_supabase, "episodes", "test-paper/episode.mp3")
        assert url == "https://example.com/file.mp3"
        mock_supabase.storage.from_.assert_called_with("episodes")

    def test_upload_audio_to_storage(self, mock_supabase):
        """Test uploading audio to storage."""
        audio_data = b"fake audio data"
        path = upload_audio_to_storage(
            mock_supabase,
            audio_data,
            "paper-123",
            "episode-456"
        )
        assert path == "paper-123/episode-456.mp3"
        mock_supabase.storage.from_.assert_called()

    def test_delete_from_storage(self, mock_supabase):
        """Test deleting file from storage."""
        result = delete_from_storage(mock_supabase, "episodes", "test-path.mp3")
        assert result == True
        mock_supabase.storage.from_.assert_called()


class TestPDFAnalyzer:
    """Tests for lib/pdf_analyzer.py"""

    def test_create_paper_slug(self):
        """Test creating URL-friendly slug from title."""
        title = "Machine Learning: A New Approach to Problem Solving!"
        slug = create_paper_slug(title)
        assert slug == "machine-learning-a-new-approach-to-problem-solving"
        assert len(slug) <= 80
        assert " " not in slug
        assert ":" not in slug

    def test_create_paper_slug_long_title(self):
        """Test slug creation with very long title."""
        title = "A" * 100
        slug = create_paper_slug(title)
        assert len(slug) <= 80

    def test_paper_sections_model(self):
        """Test PaperSections Pydantic model."""
        sections = PaperSections(
            abstract="This is the abstract",
            introduction="This is the introduction",
            methods=None,
            results="These are the results",
            discussion=None,
            conclusion="This is the conclusion",
            other=None
        )
        assert sections.abstract == "This is the abstract"
        assert sections.methods is None
        assert sections.results == "These are the results"


class TestPodcastGenerator:
    """Tests for lib/podcast_generator.py"""

    def test_convert_audio_to_mp3(self):
        """Test converting audio to MP3 format."""
        # Create a minimal WAV file (44 bytes header + some data)
        wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00]\x00\x00\x00]\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        wav_data = wav_header + b'\x00' * 100  # Add some audio data

        mp3_data = convert_audio_to_mp3(wav_data, source_format='wav')
        assert isinstance(mp3_data, bytes)
        assert len(mp3_data) > 0
        # MP3 files start with ID3 tag or MPEG sync bytes
        assert mp3_data[:3] == b'ID3' or mp3_data[0:2] == b'\xff\xfb'


class TestRSSFeed:
    """Tests for lib/rss_feed.py"""

    def test_format_duration_minutes(self):
        """Test formatting duration for episodes under 1 hour."""
        duration = format_duration(600)  # 10 minutes
        assert duration == "10:00"

        duration = format_duration(125)  # 2:05
        assert duration == "02:05"

    def test_format_duration_hours(self):
        """Test formatting duration for episodes over 1 hour."""
        duration = format_duration(3665)  # 1:01:05
        assert duration == "01:01:05"

        duration = format_duration(7200)  # 2:00:00
        assert duration == "02:00:00"

    def test_format_rfc2822_date(self):
        """Test converting ISO date to RFC 2822 format."""
        iso_date = "2024-01-15T10:30:00Z"
        rfc_date = format_rfc2822_date(iso_date)
        assert "2024" in rfc_date
        assert "Jan" in rfc_date
        assert "15" in rfc_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
