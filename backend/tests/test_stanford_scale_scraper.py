"""
Tests for Stanford SCALE repository scraper.

Uses mock HTML data to test parsing without network calls.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from lib.ingestion.stanford_scale import StanfordScaleSource, BASE_URL


# Sample HTML from a SCALE repository listing page
MOCK_LISTING_HTML = """
<!DOCTYPE html>
<html>
<head><title>Research Study Repository</title></head>
<body>
<div class="view-content">
  <div class="views-row">
    <h5><a href="/ai/repository/generative-ai-education-foundational-insights">Generative AI in Education: From Foundational Insights to the Socratic Playground</a></h5>
    <p>Xiangen Hu, Sheng Xu, Richard Tong, Art Graesser. (01/2025). <em>arXiv</em>.</p>
    <p><strong>What is the application?</strong> Tutoring</p>
    <p><strong>Who is the user?</strong> Students</p>
    <p><strong>Study design</strong>: Literature review</p>
  </div>

  <div class="views-row">
    <h5><a href="/ai/repository/impact-chatgpt-student-learning-outcomes">The Impact of ChatGPT on Student Learning Outcomes</a></h5>
    <p>Jane Smith, John Doe. (12/2024). <em>Nature Education</em>.</p>
    <p><strong>What is the application?</strong> Writing assistance</p>
    <p><strong>Who is the user?</strong> K-12 students</p>
    <p><strong>Which age?</strong> High school</p>
    <p><strong>Study design</strong>: Randomized controlled trial</p>
  </div>

  <div class="views-row">
    <h5><a href="/ai/repository/ai-math-tutoring-effectiveness">AI Math Tutoring Effectiveness in Middle Schools</a></h5>
    <p>Alice Johnson. (06/2024). <em>Journal of Educational Technology</em>.</p>
  </div>
</div>
</body>
</html>
"""

# Sample HTML from a SCALE paper detail page
MOCK_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<head><title>Generative AI in Education</title></head>
<body>
<article>
  <h1>Generative AI in Education: From Foundational Insights to the Socratic Playground</h1>
  <div class="field--name-field-pub-link">
    <div class="field__item">
      <a href="http://arxiv.org/pdf/2501.06682v1">View Paper</a>
    </div>
  </div>
</article>
</body>
</html>
"""

# Detail page with DOI link instead of arXiv
MOCK_DETAIL_HTML_DOI = """
<!DOCTYPE html>
<html>
<body>
<article>
  <h1>The Impact of ChatGPT</h1>
  <div class="field--name-field-pub-link">
    <div class="field__item">
      <a href="https://doi.org/10.1234/example.2024">View Paper</a>
    </div>
  </div>
</article>
</body>
</html>
"""

# Detail page with no paper link
MOCK_DETAIL_HTML_NO_LINK = """
<!DOCTYPE html>
<html>
<body>
<article>
  <h1>Some Paper</h1>
  <p>No link available</p>
</article>
</body>
</html>
"""


class TestStanfordScaleScraper:
    """Tests for StanfordScaleSource."""

    @pytest.fixture
    def source(self):
        """Create a source instance."""
        return StanfordScaleSource()

    @pytest.mark.asyncio
    async def test_fetch_page_parses_papers(self, source):
        """Test that fetch_page correctly parses paper entries from HTML."""
        mock_response = MagicMock()
        mock_response.text = MOCK_LISTING_HTML
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            papers = await source.fetch_page(0)

            # Verify correct URL was called
            mock_get.assert_called_once_with(f"{BASE_URL}/ai/repository?page=0")

            # Should find 3 papers
            assert len(papers) == 3

            # Check first paper
            paper1 = papers[0]
            assert paper1.title == "Generative AI in Education: From Foundational Insights to the Socratic Playground"
            assert paper1.source_url == f"{BASE_URL}/ai/repository/generative-ai-education-foundational-insights"

            # Check second paper
            paper2 = papers[1]
            assert paper2.title == "The Impact of ChatGPT on Student Learning Outcomes"
            assert paper2.source_url == f"{BASE_URL}/ai/repository/impact-chatgpt-student-learning-outcomes"

            # Check third paper
            paper3 = papers[2]
            assert paper3.title == "AI Math Tutoring Effectiveness in Middle Schools"

    @pytest.mark.asyncio
    async def test_fetch_page_with_different_page_numbers(self, source):
        """Test that page numbers are correctly passed to URL."""
        mock_response = MagicMock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            await source.fetch_page(5)
            mock_get.assert_called_once_with(f"{BASE_URL}/ai/repository?page=5")

            mock_get.reset_mock()
            await source.fetch_page(0)
            mock_get.assert_called_once_with(f"{BASE_URL}/ai/repository?page=0")

    @pytest.mark.asyncio
    async def test_fetch_page_returns_empty_on_no_papers(self, source):
        """Test that empty page returns empty list."""
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>No results</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            papers = await source.fetch_page(100)
            assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_page_handles_http_error(self, source):
        """Test that HTTP errors return empty list."""
        import httpx

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            papers = await source.fetch_page(0)
            assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_paper_url_extracts_arxiv_link(self, source):
        """Test extracting paper URL from detail page with arXiv link."""
        mock_response = MagicMock()
        mock_response.text = MOCK_DETAIL_HTML
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            url = await source.fetch_paper_url("https://scale.stanford.edu/ai/repository/some-paper")

            assert url == "http://arxiv.org/pdf/2501.06682v1"

    @pytest.mark.asyncio
    async def test_fetch_paper_url_extracts_doi_link(self, source):
        """Test extracting paper URL from detail page with DOI link."""
        mock_response = MagicMock()
        mock_response.text = MOCK_DETAIL_HTML_DOI
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            url = await source.fetch_paper_url("https://scale.stanford.edu/ai/repository/some-paper")

            assert url == "https://doi.org/10.1234/example.2024"

    @pytest.mark.asyncio
    async def test_fetch_paper_url_returns_none_when_no_link(self, source):
        """Test that missing paper link returns None."""
        mock_response = MagicMock()
        mock_response.text = MOCK_DETAIL_HTML_NO_LINK
        mock_response.raise_for_status = MagicMock()

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            url = await source.fetch_paper_url("https://scale.stanford.edu/ai/repository/some-paper")

            assert url is None

    @pytest.mark.asyncio
    async def test_fetch_paper_url_handles_http_error(self, source):
        """Test that HTTP errors return None."""
        import httpx

        with patch.object(source.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            url = await source.fetch_paper_url("https://scale.stanford.edu/ai/repository/some-paper")
            assert url is None

    def test_normalize_pdf_url_arxiv_abs_to_pdf(self, source):
        """Test arXiv abstract URL is converted to PDF URL."""
        abs_url = "https://arxiv.org/abs/2501.06682"
        pdf_url = source.normalize_pdf_url(abs_url)
        assert pdf_url == "https://arxiv.org/pdf/2501.06682.pdf"

    def test_normalize_pdf_url_arxiv_with_version(self, source):
        """Test arXiv URL with version is converted correctly."""
        abs_url = "https://arxiv.org/abs/2501.06682v2"
        pdf_url = source.normalize_pdf_url(abs_url)
        assert pdf_url == "https://arxiv.org/pdf/2501.06682v2.pdf"

    def test_normalize_pdf_url_non_arxiv_unchanged(self, source):
        """Test non-arXiv URLs are returned unchanged."""
        doi_url = "https://doi.org/10.1234/example"
        assert source.normalize_pdf_url(doi_url) == doi_url

        pdf_url = "https://example.com/paper.pdf"
        assert source.normalize_pdf_url(pdf_url) == pdf_url

    def test_normalize_pdf_url_already_pdf(self, source):
        """Test arXiv PDF URLs are returned unchanged."""
        pdf_url = "https://arxiv.org/pdf/2501.06682.pdf"
        assert source.normalize_pdf_url(pdf_url) == pdf_url

    def test_normalize_pdf_url_none_input(self, source):
        """Test None input returns None."""
        assert source.normalize_pdf_url(None) is None

    def test_normalize_pdf_url_empty_string(self, source):
        """Test empty string returns empty string."""
        assert source.normalize_pdf_url("") == ""

    def test_get_source_type(self, source):
        """Test source type identifier."""
        assert source.get_source_type() == "stanford_scale"

    @pytest.mark.asyncio
    async def test_close_closes_client(self, source):
        """Test that close() properly closes the HTTP client."""
        with patch.object(source.client, 'aclose', new_callable=AsyncMock) as mock_close:
            await source.close()
            mock_close.assert_called_once()
