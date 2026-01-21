"""
Paper Ingestion Runner

Orchestrates the paper ingestion process: fetch, deduplicate, import, extract.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from supabase import Client

from .base import PaperMetadata, PaperSource
from .stanford_scale import StanfordScaleSource

logger = logging.getLogger(__name__)


@dataclass
class IngestionProgress:
    """Progress update during ingestion."""

    status: str  # "running", "completed", "failed"
    pages_scanned: int = 0
    papers_found: int = 0
    papers_imported: int = 0
    papers_skipped: int = 0
    current_paper: Optional[str] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Final result of an ingestion run."""

    status: str  # "completed", "failed"
    pages_scanned: int = 0
    papers_found: int = 0
    papers_imported: int = 0
    papers_skipped: int = 0
    errors: list[str] = field(default_factory=list)


async def check_existing_papers_by_source_url(supabase: Client, source_urls: list[str]) -> set[str]:
    """
    Check which source URLs already exist in the database.

    Args:
        supabase: Supabase client
        source_urls: List of source URLs to check

    Returns:
        Set of source URLs that already exist
    """
    if not source_urls:
        return set()

    response = (
        supabase.table("papers")
        .select("source_url")
        .in_("source_url", source_urls)
        .execute()
    )

    return {row["source_url"] for row in response.data}


def get_arxiv_url_variants(url: str) -> list[str]:
    """
    Get all URL variants for an arXiv paper (abs, pdf, with/without .pdf).

    Args:
        url: Any arXiv URL

    Returns:
        List of possible URL variants to check
    """
    import re

    variants = [url]

    # Extract arXiv ID from various formats
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', url)
    if match:
        arxiv_id = match.group(1)
        variants = [
            f"https://arxiv.org/abs/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            f"http://arxiv.org/abs/{arxiv_id}",
            f"http://arxiv.org/pdf/{arxiv_id}",
            f"http://arxiv.org/pdf/{arxiv_id}.pdf",
        ]

    return variants


async def check_paper_url_exists(supabase: Client, paper_url: str) -> bool:
    """
    Check if a paper with this paper_url already exists.
    Handles arXiv URL variants (abs vs pdf).

    Args:
        supabase: Supabase client
        paper_url: The actual paper URL to check

    Returns:
        True if exists, False otherwise
    """
    if not paper_url:
        return False

    # Get all possible URL variants (especially for arXiv)
    url_variants = get_arxiv_url_variants(paper_url)

    response = (
        supabase.table("papers")
        .select("id")
        .in_("paper_url", url_variants)
        .limit(1)
        .execute()
    )

    return len(response.data) > 0


async def import_paper(
    paper: PaperMetadata,
    paper_url: str,
    supabase: Client,
    source_type: str,
) -> Optional[str]:
    """
    Import a single paper into the database.

    Args:
        paper: Paper metadata from the source
        paper_url: Actual paper/PDF URL
        supabase: Supabase client
        source_type: Source type identifier

    Returns:
        Paper ID if successful, None otherwise
    """
    from lib.paper_import import import_paper_from_url

    try:
        result = await import_paper_from_url(
            url=paper_url,
            supabase=supabase,
            auto_extract=False,  # We'll handle extraction separately
        )

        paper_id = result["paper_id"]

        # Update with additional metadata from the source
        update_data = {
            "source_url": paper.source_url,
            "source_type": source_type,
            "source_metadata": paper.source_metadata,
        }

        # Override with source metadata if we have better data
        if paper.title and len(paper.title) > len(result.get("title", "") or ""):
            update_data["title"] = paper.title
        if paper.authors:
            update_data["authors"] = paper.authors
        if paper.year:
            update_data["year"] = paper.year
        if paper.month:
            update_data["month"] = paper.month
        if paper.venue:
            update_data["venue"] = paper.venue

        # Set published_at date (first of month if we have month, else Jan 1)
        if paper.year:
            month = paper.month if paper.month else 1
            update_data["published_at"] = f"{paper.year}-{month:02d}-01"

        supabase.table("papers").update(update_data).eq("id", paper_id).execute()

        return paper_id

    except Exception as e:
        logger.error(f"Failed to import paper {paper.source_url}: {e}")
        raise


async def run_ingestion(
    source: PaperSource,
    supabase: Client,
    extraction_service=None,
    on_progress: Optional[Callable[[IngestionProgress], None]] = None,
    max_pages: Optional[int] = None,
) -> IngestionResult:
    """
    Run the paper ingestion process.

    Uses early-exit pagination: stops when all papers on a page already exist.

    Args:
        source: Paper source to ingest from
        supabase: Supabase client
        extraction_service: Optional extraction service for auto-extraction
        on_progress: Optional callback for progress updates
        max_pages: Optional limit on pages to scan (for testing)

    Returns:
        IngestionResult with final statistics
    """
    progress = IngestionProgress(status="running")

    def update_progress():
        if on_progress:
            on_progress(progress)

    try:
        page = 0
        while max_pages is None or page < max_pages:
            # Fetch page of papers
            papers = await source.fetch_page(page)

            if not papers:
                logger.info(f"No papers found on page {page}, stopping")
                break

            progress.pages_scanned = page + 1
            progress.papers_found += len(papers)
            update_progress()

            # Check which papers already exist by source_url
            source_urls = [p.source_url for p in papers]
            existing = await check_existing_papers_by_source_url(supabase, source_urls)
            new_papers = [p for p in papers if p.source_url not in existing]

            if not new_papers:
                # All papers on this page exist - early exit
                logger.info(
                    f"Page {page}: All {len(papers)} papers already exist, stopping"
                )
                progress.papers_skipped += len(papers)
                update_progress()
                break

            # Process new papers
            skipped_on_page = len(papers) - len(new_papers)
            progress.papers_skipped += skipped_on_page

            for paper in new_papers:
                progress.current_paper = paper.title
                update_progress()

                try:
                    # Fetch the actual paper URL from detail page
                    if hasattr(source, "fetch_paper_url"):
                        paper_url = await source.fetch_paper_url(paper.source_url)
                        if hasattr(source, "normalize_pdf_url"):
                            paper_url = source.normalize_pdf_url(paper_url)
                    else:
                        paper_url = paper.paper_url

                    if not paper_url:
                        logger.warning(
                            f"No paper URL found for {paper.title}, skipping"
                        )
                        progress.papers_skipped += 1
                        progress.errors.append(f"No paper URL: {paper.title[:50]}...")
                        continue

                    # Check if paper_url already exists (different source_url but same paper)
                    if await check_paper_url_exists(supabase, paper_url):
                        logger.info(
                            f"Paper URL already exists: {paper.title[:50]}..., skipping"
                        )
                        progress.papers_skipped += 1
                        continue

                    # Import the paper
                    paper_id = await import_paper(
                        paper=paper,
                        paper_url=paper_url,
                        supabase=supabase,
                        source_type=source.get_source_type(),
                    )

                    if paper_id:
                        progress.papers_imported += 1
                        logger.info(f"Imported: {paper.title[:50]}...")

                        # Auto-extract content if service provided
                        if extraction_service:
                            try:
                                await extraction_service.extract_from_storage(paper_id)
                                logger.info(f"Extracted content for {paper_id}")
                            except Exception as e:
                                logger.error(f"Extraction failed for {paper_id}: {e}")
                                progress.errors.append(
                                    f"Extraction failed: {paper.title[:30]}..."
                                )
                    else:
                        progress.papers_skipped += 1

                except Exception as e:
                    logger.error(f"Failed to process {paper.title}: {e}")
                    progress.errors.append(f"Import failed: {paper.title[:30]}... - {e}")
                    progress.papers_skipped += 1

                update_progress()

            page += 1

        progress.status = "completed"
        progress.current_paper = None
        update_progress()

        return IngestionResult(
            status="completed",
            pages_scanned=progress.pages_scanned,
            papers_found=progress.papers_found,
            papers_imported=progress.papers_imported,
            papers_skipped=progress.papers_skipped,
            errors=progress.errors,
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        progress.status = "failed"
        progress.errors.append(f"Fatal error: {e}")
        update_progress()

        return IngestionResult(
            status="failed",
            pages_scanned=progress.pages_scanned,
            papers_found=progress.papers_found,
            papers_imported=progress.papers_imported,
            papers_skipped=progress.papers_skipped,
            errors=progress.errors,
        )

    finally:
        # Clean up source resources
        if hasattr(source, "close"):
            await source.close()


async def run_stanford_scale_ingestion(
    supabase: Client,
    extraction_service=None,
    on_progress: Optional[Callable[[IngestionProgress], None]] = None,
    max_pages: Optional[int] = None,
) -> IngestionResult:
    """
    Convenience function to run Stanford SCALE ingestion.

    Args:
        supabase: Supabase client
        extraction_service: Optional extraction service
        on_progress: Optional progress callback
        max_pages: Optional page limit

    Returns:
        IngestionResult
    """
    source = StanfordScaleSource()
    return await run_ingestion(
        source=source,
        supabase=supabase,
        extraction_service=extraction_service,
        on_progress=on_progress,
        max_pages=max_pages,
    )
