"""
Admin router - handles paper import and administrative functions.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
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
