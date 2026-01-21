"""
Stanford SCALE Repository Paper Source

Scrapes papers from https://scale.stanford.edu/ai/repository
"""

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .base import PaperMetadata, PaperSource

logger = logging.getLogger(__name__)

BASE_URL = "https://scale.stanford.edu"
REPOSITORY_URL = f"{BASE_URL}/ai/repository"


class StanfordScaleSource(PaperSource):
    """Paper source for Stanford SCALE AI in Education repository."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def get_source_type(self) -> str:
        return "stanford_scale"

    async def fetch_page(self, page: int) -> list[PaperMetadata]:
        """
        Fetch a page of papers from the repository.

        Args:
            page: Page number (0-indexed)

        Returns:
            List of PaperMetadata for papers on that page.
        """
        url = f"{REPOSITORY_URL}?page={page}"
        logger.info(f"Fetching SCALE repository page {page}: {url}")

        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            return []

        soup = BeautifulSoup(response.text, "lxml")
        papers = []

        # Find all paper entries - they're in h5 tags with links
        for h5 in soup.find_all("h5"):
            link = h5.find("a")
            if not link or not link.get("href", "").startswith("/ai/repository/"):
                continue

            title = link.get_text(strip=True)
            detail_path = link["href"]

            # Skip category/filter links (they don't have the right structure)
            if not title or len(title) < 10:
                continue

            source_url = f"{BASE_URL}{detail_path}"

            # Parse the metadata from the text following the h5
            # Format: "Authors. (MM/YYYY). _Source_."
            metadata = self._parse_entry_metadata(h5)

            papers.append(
                PaperMetadata(
                    source_url=source_url,
                    title=title,
                    authors=metadata.get("authors"),
                    year=metadata.get("year"),
                    month=metadata.get("month"),
                    venue=metadata.get("venue"),
                    paper_url=None,  # Will be fetched from detail page
                    source_metadata=metadata.get("tags", {}),
                )
            )

        logger.info(f"Found {len(papers)} papers on page {page}")
        return papers

    def _parse_entry_metadata(self, h5_element) -> dict:
        """Parse metadata from the entry following the h5 title."""
        metadata = {"tags": {}}

        # Get the parent container and extract text
        parent = h5_element.parent
        if not parent:
            return metadata

        # Get all text content after the h5
        text_parts = []
        for sibling in h5_element.next_siblings:
            if hasattr(sibling, "get_text"):
                text_parts.append(sibling.get_text(strip=True))
            elif isinstance(sibling, str):
                text_parts.append(sibling.strip())

        full_text = " ".join(text_parts)

        # Extract authors and date: "Author1, Author2. (MM/YYYY). _Venue_."
        # Pattern: text before (MM/YYYY)
        date_match = re.search(r"\((\d{1,2})/(\d{4})\)", full_text)
        if date_match:
            month = date_match.group(1)
            year = date_match.group(2)
            metadata["year"] = int(year)
            metadata["month"] = int(month)

            # Authors are before the date
            authors_text = full_text[: date_match.start()].strip()
            if authors_text.endswith("."):
                authors_text = authors_text[:-1]
            if authors_text:
                metadata["authors"] = authors_text

        # Extract venue (between underscores or after date)
        venue_match = re.search(r"_([^_]+)_", full_text)
        if venue_match:
            metadata["venue"] = venue_match.group(1).strip()

        # Extract tags from bold labels
        # Pattern: **Label** value1, value2
        tag_patterns = [
            (r"\*\*What is the application\?\*\*\s*([^*]+?)(?=\*\*|$)", "application"),
            (r"\*\*Who is the user\?\*\*\s*([^*]+?)(?=\*\*|$)", "users"),
            (r"\*\*Which age\?\*\*\s*([^*]+?)(?=\*\*|$)", "ages"),
            (r"\*\*Why use AI\?\*\*\s*([^*]+?)(?=\*\*|$)", "why"),
            (r"\*\*Study design\*\*:?\s*([^*]+?)(?=\*\*|$)", "study_design"),
        ]

        for pattern, key in tag_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r"\s+", " ", value)
                if value:
                    metadata["tags"][key] = value

        return metadata

    async def fetch_paper_url(self, source_url: str) -> Optional[str]:
        """
        Fetch the actual paper URL from a detail page.

        Args:
            source_url: URL of the paper detail page on SCALE

        Returns:
            The actual paper URL (e.g., arXiv link), or None if not found.
        """
        logger.debug(f"Fetching paper URL from: {source_url}")

        try:
            response = await self.client.get(source_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch detail page {source_url}: {e}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Look for the paper link using the CSS selector from n8n workflow
        # .field--name-field-pub-link .field__item a
        link_field = soup.select_one(".field--name-field-pub-link .field__item a")
        if link_field and link_field.get("href"):
            paper_url = link_field["href"]
            logger.debug(f"Found paper URL: {paper_url}")
            return paper_url

        # Fallback: look for any link that looks like a paper URL
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(
                domain in href
                for domain in ["arxiv.org", "doi.org", "acm.org", "ieee.org", ".pdf"]
            ):
                logger.debug(f"Found paper URL via fallback: {href}")
                return href

        logger.warning(f"No paper URL found on {source_url}")
        return None

    @staticmethod
    def normalize_pdf_url(url: str) -> str:
        """
        Normalize a paper URL to a direct PDF URL if possible.

        Args:
            url: Original paper URL

        Returns:
            Normalized URL (e.g., arXiv abs -> pdf)
        """
        if not url:
            return url

        # Convert arXiv abstract pages to PDF
        if "arxiv.org/abs/" in url:
            return url.replace("/abs/", "/pdf/") + ".pdf"

        return url
