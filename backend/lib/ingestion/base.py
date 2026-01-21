"""
Base classes for paper ingestion sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperMetadata:
    """Metadata for a paper discovered from a source."""

    source_url: str  # URL on the source site (used for deduplication)
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    paper_url: Optional[str] = None  # Actual paper/PDF URL
    source_metadata: dict = field(default_factory=dict)  # Source-specific tags


class PaperSource(ABC):
    """Abstract base class for paper sources."""

    @abstractmethod
    async def fetch_page(self, page: int) -> list[PaperMetadata]:
        """
        Fetch a page of papers from the source.

        Args:
            page: Page number (0-indexed)

        Returns:
            List of PaperMetadata for papers on that page.
            Empty list if no more pages.
        """
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """
        Get the source type identifier.

        Returns:
            String identifier (e.g., "stanford_scale", "arxiv")
        """
        pass
