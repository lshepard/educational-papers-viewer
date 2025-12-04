"""
Papers router - handles paper extraction, batch processing, and import.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


# ==================== Request/Response Models ====================

class ExtractionRequest(BaseModel):
    paper_id: str


class ExtractionResponse(BaseModel):
    success: bool
    paper_id: str
    sections_count: int
    images_count: int
    message: str


class ProcessingStats(BaseModel):
    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    total_sections: Optional[int] = None


class BatchExtractionResponse(BaseModel):
    success: bool
    processed: int
    succeeded: int
    failed: int
    results: list


class ImportPaperRequest(BaseModel):
    url: str  # Can be arXiv URL, PDF URL, or paper landing page


class ImportPaperResponse(BaseModel):
    success: bool
    paper_id: str
    message: str


# ==================== Dependencies ====================

def get_extraction_service():
    """Dependency to get extraction service - will be injected from main app state."""
    from main import app
    return app.state.extraction_service


def get_supabase():
    """Dependency to get Supabase client - will be injected from main app state."""
    from main import app, supabase
    return supabase


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
            scrapegraphai_api_key=scrapegraphai_api_key
        )

        return ImportPaperResponse(
            success=True,
            paper_id=result["paper_id"],
            message=result.get("message", "Paper imported successfully")
        )

    except Exception as e:
        logger.error(f"Paper import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract", response_model=ExtractionResponse)
async def extract_paper_content(
    request: ExtractionRequest,
    extraction_service = Depends(get_extraction_service)
):
    """
    Extract sections and images from a research paper using Gemini AI.

    This endpoint:
    1. Fetches the paper from Supabase storage
    2. Uploads to Gemini Files API
    3. Extracts sections and images in parallel
    4. Stores results in database
    """
    try:
        logger.info(f"Starting extraction for paper: {request.paper_id}")

        result = await extraction_service.extract_from_storage(request.paper_id)

        return ExtractionResponse(
            success=True,
            paper_id=request.paper_id,
            sections_count=result.sections_count,
            images_count=result.images_count,
            message=f"Successfully extracted {result.sections_count} sections and {result.images_count} images"
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-extract", response_model=BatchExtractionResponse)
async def batch_extract_papers(
    limit: Optional[int] = None,
    status: str = "pending",
    extraction_service = Depends(get_extraction_service),
    supabase = Depends(get_supabase)
):
    """
    Batch extract content from multiple papers.

    Processes papers sequentially to avoid API rate limits.

    Query parameters:
    - limit: Maximum number of papers to process (default: all)
    - status: Filter by processing status (default: "pending")
    """
    try:
        logger.info(f"Starting batch extraction (status={status}, limit={limit})")

        # Fetch papers to process
        query = supabase.table("papers").select("*").eq("processing_status", status)

        if limit:
            query = query.limit(limit)

        response = query.execute()
        papers = response.data

        if not papers:
            return BatchExtractionResponse(
                success=True,
                processed=0,
                succeeded=0,
                failed=0,
                results=[],
            )

        logger.info(f"Found {len(papers)} papers to process")

        # Process papers sequentially
        results = []
        succeeded = 0
        failed = 0

        for paper in papers:
            try:
                result = await extraction_service.extract_from_storage(paper["id"])
                results.append({
                    "success": True,
                    "paper_id": paper["id"],
                    "sections_count": result.sections_count,
                    "images_count": result.images_count
                })
                succeeded += 1
                logger.info(f"✓ Processed: {paper.get('title', paper['id'])}")

            except Exception as e:
                results.append({
                    "success": False,
                    "paper_id": paper["id"],
                    "error": str(e)
                })
                failed += 1
                logger.error(f"✗ Failed: {paper.get('title', paper['id'])}: {e}")

            # Small delay between papers to avoid rate limits
            import asyncio
            await asyncio.sleep(1)

        logger.info(f"Batch complete: {succeeded} succeeded, {failed} failed")

        return BatchExtractionResponse(
            success=True,
            processed=len(papers),
            succeeded=succeeded,
            failed=failed,
            results=results
        )

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_papers(q: str, supabase = Depends(get_supabase)):
    """
    Search papers in the database by keywords.

    Searches across title, authors, application, venue, and why fields.
    Returns matching papers with all their metadata.
    """
    try:
        search_term = f"%{q}%"

        response = supabase.table("papers").select("*").or_(
            f"title.ilike.{search_term},"
            f"authors.ilike.{search_term},"
            f"application.ilike.{search_term},"
            f"venue.ilike.{search_term},"
            f"why.ilike.{search_term}"
        ).limit(50).execute()

        return {
            "success": True,
            "papers": response.data,
            "count": len(response.data)
        }

    except Exception as e:
        logger.error(f"Failed to search papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ProcessingStats)
async def get_processing_stats(supabase = Depends(get_supabase)):
    """Get statistics about paper processing status."""
    try:
        # Get counts by status
        all_papers = supabase.table("papers").select("processing_status").execute()

        stats = {
            "total": len(all_papers.data),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }

        for paper in all_papers.data:
            status = paper.get("processing_status", "pending")
            if status in stats:
                stats[status] += 1

        # Get total sections count
        sections_response = supabase.table("paper_sections").select("id", count="exact").execute()
        stats["total_sections"] = sections_response.count

        return ProcessingStats(**stats)

    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
