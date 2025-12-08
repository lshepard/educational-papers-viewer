"""
Comprehensive tests for podcast generation pipeline.
Mocks LLM calls to test parameter passing and logic without long-running operations.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
import io


class TestPodcastGenerationPipeline:
    """Tests for the complete podcast generation pipeline."""

    @pytest.fixture
    def mock_audio_data(self):
        """Generate fake MP3 audio data."""
        # Minimal valid MP3 header + some data
        mp3_header = b'\xff\xfb\x90\x00'
        fake_audio = mp3_header + b'\x00' * 1000
        return fake_audio

    @pytest.fixture
    def mock_script(self):
        """Generate fake podcast script."""
        return """
        <Person1>
        Welcome to today's podcast! We're discussing an exciting new paper about machine learning.
        </Person1>
        <Person2>
        That's right! This paper introduces a novel approach to neural network optimization.
        </Person2>
        <Person1>
        Let's dive into the details...
        </Person1>
        """

    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing."""
        return [
            {
                "id": "paper-1",
                "title": "Machine Learning Advances",
                "authors": "John Doe, Jane Smith",
                "year": 2024,
                "abstract": "This paper presents advances in machine learning."
            },
            {
                "id": "paper-2",
                "title": "Deep Learning Techniques",
                "authors": "Alice Johnson",
                "year": 2023,
                "abstract": "Novel deep learning techniques are explored."
            }
        ]

    @pytest.fixture
    def mock_supabase_full(self, mock_supabase):
        """Enhanced mock Supabase with table operations."""
        # Mock table operations
        mock_supabase.table.return_value = Mock()
        mock_supabase.table.return_value.select.return_value = Mock()
        mock_supabase.table.return_value.select.return_value.in_.return_value = Mock()
        mock_supabase.table.return_value.select.return_value.in_.return_value.execute.return_value = Mock()

        mock_supabase.table.return_value.select.return_value.eq.return_value = Mock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock()

        mock_supabase.table.return_value.insert.return_value = Mock()
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        mock_supabase.table.return_value.update.return_value = Mock()
        mock_supabase.table.return_value.update.return_value.eq.return_value = Mock()
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        return mock_supabase

    @pytest.mark.asyncio
    async def test_single_paper_podcast_generation(
        self,
        mock_supabase_full,
        mock_genai_client,
        sample_papers,
        mock_script,
        mock_audio_data
    ):
        """Test podcast generation for a single paper."""
        from routers.podcasts import generate_podcast, PodcastGenerationRequest

        # Setup mock data
        single_paper = [sample_papers[0]]
        mock_supabase_full.table.return_value.select.return_value.in_.return_value.execute.return_value.data = single_paper
        mock_supabase_full.table.return_value.insert.return_value.execute.return_value.data = [{"id": "episode-123"}]

        # Create request
        request = PodcastGenerationRequest(
            paper_ids=["paper-1"],
            title="Test Episode",
            description="Test Description"
        )

        # Mock the podcast agent and audio generation
        with patch('routers.podcasts.PodcastAgent') as mock_agent_class, \
             patch('routers.podcasts.generate_audio_from_script') as mock_audio_gen, \
             patch('routers.podcasts.convert_to_mp3') as mock_convert, \
             patch('lib.storage.upload_audio_to_storage') as mock_upload, \
             patch('lib.storage.get_public_url') as mock_get_url, \
             patch('routers.podcasts.GeminiFileManager') as mock_file_manager:

            # Setup mocks
            mock_agent = AsyncMock()
            mock_agent.generate_single_paper_script.return_value = mock_script
            mock_agent_class.return_value = mock_agent

            mock_audio_gen.return_value = mock_audio_data
            mock_convert.return_value = mock_audio_data
            mock_upload.return_value = "paper-1/episode-123.mp3"
            mock_get_url.return_value = "https://example.com/audio.mp3"

            # Mock Gemini file manager context
            mock_file = Mock()
            mock_file.uri = "gemini://files/test-file"

            # Create proper async context manager mock
            mock_context_manager = MagicMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_file)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            mock_file_manager.return_value.upload_pdf_from_url = Mock(return_value=mock_context_manager)

            # Mock get_storage_url
            with patch('routers.podcasts._get_storage_url') as mock_get_storage_url:
                mock_get_storage_url.return_value = "https://example.com/paper.pdf"

                # Execute
                result = await generate_podcast(
                    request=request,
                    supabase=mock_supabase_full,
                    genai_client=mock_genai_client,
                    gemini_manager=mock_file_manager(),
                    perplexity_api_key="test-key"
                )

            # Verify result structure
            assert result.success is True
            assert hasattr(result, "episode_id")
            assert result.episode_id == "episode-123"

            # Verify critical function calls with correct parameters
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args
            assert call_args.kwargs["supabase"] == mock_supabase_full
            assert call_args.kwargs["audio_data"] == mock_audio_data
            assert call_args.kwargs["paper_id"] == "paper-1"
            assert call_args.kwargs["episode_id"] == "episode-123"
            assert call_args.kwargs["bucket"] == "episodes"

    @pytest.mark.asyncio
    async def test_multi_paper_podcast_generation(
        self,
        mock_supabase_full,
        mock_genai_client,
        sample_papers,
        mock_script,
        mock_audio_data
    ):
        """Test podcast generation for multiple papers."""
        from routers.podcasts import generate_podcast, PodcastGenerationRequest

        # Setup mock data
        mock_supabase_full.table.return_value.select.return_value.in_.return_value.execute.return_value.data = sample_papers
        mock_supabase_full.table.return_value.insert.return_value.execute.return_value.data = [{"id": "episode-456"}]

        # Create request
        request = PodcastGenerationRequest(
            paper_ids=["paper-1", "paper-2"],
            title="Multi-Paper Episode",
            description="Discussion of multiple papers"
        )

        # Mock the podcast agent and audio generation
        with patch('routers.podcasts.PodcastAgent') as mock_agent_class, \
             patch('routers.podcasts.generate_audio_from_script') as mock_audio_gen, \
             patch('routers.podcasts.convert_to_mp3') as mock_convert, \
             patch('lib.storage.upload_audio_to_storage') as mock_upload, \
             patch('lib.storage.get_public_url') as mock_get_url:

            # Setup mocks
            mock_agent = AsyncMock()
            mock_agent.generate_multi_paper_script.return_value = mock_script
            mock_agent_class.return_value = mock_agent

            mock_audio_gen.return_value = mock_audio_data
            mock_convert.return_value = mock_audio_data
            mock_upload.return_value = "multi-paper/episode-456.mp3"
            mock_get_url.return_value = "https://example.com/audio.mp3"

            # Execute
            result = await generate_podcast(
                request=request,
                supabase=mock_supabase_full,
                genai_client=mock_genai_client,
                gemini_manager=Mock(),
                perplexity_api_key="test-key"
            )

            # Verify result
            assert result.success is True
            assert result.episode_id == "episode-456"

            # Verify multi-paper handling
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args
            assert call_args.kwargs["paper_id"] == "multi-paper"
            assert call_args.kwargs["episode_id"] == "episode-456"

            # Verify episode_papers junction table inserts
            insert_calls = [
                call for call in mock_supabase_full.table.return_value.insert.call_args_list
                if len(call[0]) > 0
            ]
            # Should have inserts for episode and 2 papers
            assert len(insert_calls) >= 2

    @pytest.mark.asyncio
    async def test_episode_papers_junction_table_insert(
        self,
        mock_supabase_full,
        mock_genai_client,
        sample_papers,
        mock_script,
        mock_audio_data
    ):
        """Test that episode_papers junction table receives correct data."""
        from routers.podcasts import generate_podcast, PodcastGenerationRequest

        # Setup mock data
        mock_supabase_full.table.return_value.select.return_value.in_.return_value.execute.return_value.data = sample_papers
        mock_supabase_full.table.return_value.insert.return_value.execute.return_value.data = [{"id": "episode-789"}]

        request = PodcastGenerationRequest(
            paper_ids=["paper-1", "paper-2"]
        )

        # Track all insert calls
        insert_calls = []

        def track_insert(data):
            insert_calls.append(data)
            mock_execute = Mock()
            mock_execute.execute.return_value.data = [{"id": "episode-789"}]
            return mock_execute

        mock_supabase_full.table.return_value.insert = Mock(side_effect=track_insert)

        with patch('routers.podcasts.PodcastAgent') as mock_agent_class, \
             patch('routers.podcasts.generate_audio_from_script') as mock_audio_gen, \
             patch('routers.podcasts.convert_to_mp3') as mock_convert, \
             patch('lib.storage.upload_audio_to_storage') as mock_upload, \
             patch('lib.storage.get_public_url') as mock_get_url, \
             patch('routers.podcasts.generate_preliminary_metadata') as mock_metadata:

            # Setup mocks
            mock_agent = AsyncMock()
            mock_agent.generate_multi_paper_script.return_value = mock_script
            mock_agent_class.return_value = mock_agent

            mock_metadata.return_value = {
                "title": "Generated Title",
                "description": "Generated Description"
            }

            mock_audio_gen.return_value = mock_audio_data
            mock_convert.return_value = mock_audio_data
            mock_upload.return_value = "multi-paper/episode-789.mp3"
            mock_get_url.return_value = "https://example.com/audio.mp3"

            await generate_podcast(
                request=request,
                supabase=mock_supabase_full,
                genai_client=mock_genai_client,
                gemini_manager=Mock(),
                perplexity_api_key="test-key"
            )

            # Find episode_papers inserts (they have paper_title field)
            episode_paper_inserts = [
                call for call in insert_calls
                if isinstance(call, dict) and "paper_title" in call
            ]

            # Should have 2 episode_papers inserts
            assert len(episode_paper_inserts) == 2

            # Verify required fields are present
            for insert in episode_paper_inserts:
                assert "episode_id" in insert
                assert "paper_id" in insert
                assert "paper_title" in insert
                assert "paper_authors" in insert
                assert "paper_year" in insert
                # Verify no invalid fields
                assert "paper_abstract" not in insert
                assert "paper_pdf_url" not in insert

    @pytest.mark.asyncio
    async def test_regenerate_audio_endpoint(
        self,
        mock_supabase_full,
        mock_genai_client,
        mock_script,
        mock_audio_data
    ):
        """Test audio regeneration from existing script."""
        from routers.podcasts import regenerate_audio

        # Setup mock episode with existing script and episode_papers
        # We need to mock multiple queries
        mock_episode_response = Mock()
        mock_episode_response.data = [
            {
                "id": "episode-999",
                "script": mock_script,
                "title": "Existing Episode"
            }
        ]

        mock_papers_response = Mock()
        mock_papers_response.data = [
            {"paper_id": "paper-1"}
        ]

        # Set up table mock to return appropriate data for different queries
        call_count = [0]
        def select_side_effect(*args):
            call_count[0] += 1
            mock_select = Mock()
            mock_eq = Mock()
            # First call is for podcast_episodes, second for episode_papers
            if call_count[0] == 1:
                mock_eq.execute.return_value = mock_episode_response
            else:
                mock_eq.execute.return_value = mock_papers_response
            mock_select.eq.return_value = mock_eq
            return mock_select

        mock_supabase_full.table.return_value.select.side_effect = select_side_effect

        with patch('routers.podcasts.generate_audio_from_script') as mock_audio_gen, \
             patch('routers.podcasts.convert_to_mp3') as mock_convert, \
             patch('lib.storage.upload_audio_to_storage') as mock_upload, \
             patch('lib.storage.get_public_url') as mock_get_url:

            mock_audio_gen.return_value = mock_audio_data
            mock_convert.return_value = mock_audio_data
            mock_upload.return_value = "paper-1/episode-999.mp3"
            mock_get_url.return_value = "https://example.com/audio.mp3"

            result = await regenerate_audio(
                episode_id="episode-999",
                supabase=mock_supabase_full,
                genai_client=mock_genai_client
            )

            # Verify result
            assert result["success"] is True
            assert result["episode_id"] == "episode-999"
            assert "audio_url" in result

            # Verify upload was called with correct parameters
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args
            assert call_args.kwargs["supabase"] == mock_supabase_full
            assert call_args.kwargs["audio_data"] == mock_audio_data
            assert call_args.kwargs["paper_id"] == "paper-1"
            assert call_args.kwargs["episode_id"] == "episode-999"

    @pytest.mark.asyncio
    async def test_parameter_passing_consistency(
        self,
        mock_supabase_full,
        mock_genai_client,
        sample_papers,
        mock_script,
        mock_audio_data
    ):
        """Test that all parameters are passed consistently through the pipeline."""
        from routers.podcasts import generate_podcast, PodcastGenerationRequest

        mock_supabase_full.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [sample_papers[0]]
        mock_supabase_full.table.return_value.insert.return_value.execute.return_value.data = [{"id": "test-episode"}]

        request = PodcastGenerationRequest(
            paper_ids=["paper-1"],
            title="Test",
            description="Test Desc"
        )

        with patch('routers.podcasts.PodcastAgent') as mock_agent_class, \
             patch('routers.podcasts.generate_audio_from_script') as mock_audio_gen, \
             patch('routers.podcasts.convert_to_mp3') as mock_convert, \
             patch('lib.storage.upload_audio_to_storage') as mock_upload, \
             patch('lib.storage.get_public_url') as mock_get_url, \
             patch('routers.podcasts.GeminiFileManager') as mock_file_manager, \
             patch('routers.podcasts._get_storage_url') as mock_get_storage_url:

            mock_agent = AsyncMock()
            mock_agent.generate_single_paper_script.return_value = mock_script
            mock_agent_class.return_value = mock_agent

            mock_audio_gen.return_value = mock_audio_data
            mock_convert.return_value = mock_audio_data
            mock_upload.return_value = "test-path.mp3"
            mock_get_url.return_value = "https://example.com/test.mp3"
            mock_get_storage_url.return_value = "https://example.com/paper.pdf"

            mock_file = Mock()
            mock_file.uri = "gemini://test"

            # Create proper async context manager mock
            mock_context_manager = MagicMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_file)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)

            mock_file_manager.return_value.upload_pdf_from_url = Mock(return_value=mock_context_manager)

            await generate_podcast(
                request=request,
                supabase=mock_supabase_full,
                genai_client=mock_genai_client,
                gemini_manager=mock_file_manager(),
                perplexity_api_key="test-perplexity-key"
            )

            # Verify all functions received their required parameters
            assert mock_audio_gen.called
            assert mock_convert.called
            assert mock_upload.called

            # Check that upload got all required params
            upload_kwargs = mock_upload.call_args.kwargs
            required_params = ["supabase", "audio_data", "paper_id", "episode_id", "bucket"]
            for param in required_params:
                assert param in upload_kwargs, f"Missing required parameter: {param}"
