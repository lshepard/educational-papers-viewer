"""
Admin router - handles paper import and administrative functions.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ==================== Request/Response Models ====================

class ImportPaperRequest(BaseModel):
    url: str  # Can be arXiv URL, PDF URL, or paper landing page


class ImportPaperResponse(BaseModel):
    success: bool
    paper_id: str
    message: str


class IngestionRequest(BaseModel):
    source: str = "stanford_scale"
    max_pages: Optional[int] = None  # Limit pages for testing
    auto_extract: bool = True  # Whether to auto-extract content after import


class IngestionResponse(BaseModel):
    success: bool
    status: str
    pages_scanned: int
    papers_found: int
    papers_imported: int
    papers_skipped: int
    errors: List[str]


# ==================== Dependencies ====================

def get_supabase():
    """Dependency to get Supabase client."""
    from main import supabase
    return supabase


def get_genai_client():
    """Dependency to get Gemini client."""
    from main import app
    return app.state.genai_client


def get_scrapegraphai_api_key():
    """Dependency to get ScrapeGraphAI API key."""
    import os
    return os.getenv("SCRAPEGRAPHAI_API_KEY")


# ==================== Endpoints ====================

@router.post("/import", response_model=ImportPaperResponse)
async def import_paper(
    request: ImportPaperRequest,
    supabase = Depends(get_supabase),
    scrapegraphai_api_key = Depends(get_scrapegraphai_api_key)
):
    """
    Import a paper from arXiv, PDF URL, or paper landing page.

    Supports:
    - arXiv URLs: https://arxiv.org/abs/1234.56789
    - Direct PDF URLs: https://example.com/paper.pdf
    - Paper landing pages (uses ScrapeGraphAI to find PDF)

    The import process:
    1. Detects URL type (arXiv, PDF, or landing page)
    2. Downloads or locates PDF
    3. Extracts metadata
    4. Uploads to Supabase storage
    5. Creates paper record in database
    """
    try:
        from lib.paper_import import import_paper_from_url

        result = await import_paper_from_url(
            url=request.url,
            supabase=supabase,
            scrapegraph_api_key=scrapegraphai_api_key
        )

        return ImportPaperResponse(
            success=True,
            paper_id=result["paper_id"],
            message=result.get("message", "Paper imported successfully")
        )

    except Exception as e:
        logger.error(f"Paper import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/populate-research")
async def populate_research_metadata(
    limit: Optional[int] = None,
    force_refresh: bool = False,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client)
):
    """
    Populate research metadata for existing papers using Gemini.

    This generates summaries, key findings, and research context
    for papers that don't have this metadata yet.

    Query parameters:
    - limit: Maximum number of papers to process
    - force_refresh: Re-generate metadata for papers that already have it
    """
    try:
        from lib.research import populate_research_for_existing_papers

        results = await populate_research_for_existing_papers(
            genai_client=genai_client,
            supabase_client=supabase,
            limit=limit,
            force_refresh=force_refresh
        )

        return {
            "success": True,
            "processed": results.get("processed", 0),
            "updated": results.get("updated", 0),
            "skipped": results.get("skipped", 0),
            "message": "Research metadata population complete"
        }

    except Exception as e:
        logger.error(f"Research metadata population failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_extraction_service(request: Request):
    """Dependency to get extraction service from app state."""
    return request.app.state.extraction_service


@router.post("/run-ingestion", response_model=IngestionResponse)
async def run_ingestion(
    request: IngestionRequest,
    supabase=Depends(get_supabase),
    extraction_service=Depends(get_extraction_service),
):
    """
    Run paper ingestion from a configured source.

    Currently supported sources:
    - stanford_scale: Stanford SCALE AI in Education repository

    The ingestion process:
    1. Scrapes the source for papers (paginated)
    2. Checks which papers already exist (by source_url)
    3. Stops early when all papers on a page already exist
    4. Downloads PDFs and imports new papers
    5. Optionally extracts content (sections/images)

    Query parameters:
    - source: Source to ingest from (default: stanford_scale)
    - max_pages: Limit pages to scan (for testing)
    - auto_extract: Whether to extract content after import (default: true)
    """
    try:
        from lib.ingestion import run_ingestion as do_run_ingestion
        from lib.ingestion.stanford_scale import StanfordScaleSource

        # Select source
        if request.source == "stanford_scale":
            source = StanfordScaleSource()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown source: {request.source}"
            )

        logger.info(f"Starting ingestion from {request.source}")

        # Run ingestion
        result = await do_run_ingestion(
            source=source,
            supabase=supabase,
            extraction_service=extraction_service if request.auto_extract else None,
            max_pages=request.max_pages,
        )

        logger.info(
            f"Ingestion complete: {result.papers_imported} imported, "
            f"{result.papers_skipped} skipped, {len(result.errors)} errors"
        )

        return IngestionResponse(
            success=result.status == "completed",
            status=result.status,
            pages_scanned=result.pages_scanned,
            papers_found=result.papers_found,
            papers_imported=result.papers_imported,
            papers_skipped=result.papers_skipped,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
