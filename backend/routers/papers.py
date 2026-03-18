"""
Papers router - handles paper extraction, batch processing, and import.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
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
    cookies: Optional[dict] = None  # Optional cookies for Cloudflare-protected sites


class SetCookiesRequest(BaseModel):
    domain: str  # e.g., "papers.ssrn.com"
    cookies: dict  # e.g., {"cf_clearance": "..."}


class SetCookiesResponse(BaseModel):
    success: bool
    message: str


class ImportPaperResponse(BaseModel):
    success: bool
    paper_id: str
    message: str


class PaperNoteRequest(BaseModel):
    rating: Optional[str] = None  # 'ignore', 'ok', 'highlight', or None
    notes: Optional[str] = None


class PaperNoteResponse(BaseModel):
    id: Optional[str] = None
    paper_id: str
    rating: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


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


@router.post("/upload", response_model=ImportPaperResponse)
async def upload_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    supabase = Depends(get_supabase)
):
    """
    Upload a PDF file directly.

    Use this when automatic download fails (e.g., Cloudflare-protected sites).
    Download the PDF in your browser, then upload it here.
    """
    try:
        import tempfile
        from pathlib import Path
        from lib.paper_import import extract_pdf_metadata
        from lib.pdf_analyzer import create_paper_slug

        # Verify it's a PDF
        content = await file.read()
        if not content.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="File is not a valid PDF")

        # Save to temp file for metadata extraction
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Extract metadata from PDF
            metadata = await extract_pdf_metadata(tmp_path)

            # Use provided title or extract from PDF
            paper_title = title or metadata.get("title") or file.filename or "Uploaded Paper"

            # Generate slug for storage
            slug = create_paper_slug(paper_title)
            storage_path = f"{slug}/paper.pdf"

            # Upload to Supabase storage
            supabase.storage.from_("papers").upload(
                path=storage_path,
                file=content,
                file_options={
                    "content-type": "application/pdf",
                    "upsert": "true"
                }
            )

            logger.info(f"Uploaded PDF to storage: {storage_path}")

            # Create database record
            paper_data = {
                "title": paper_title,
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "source_url": source_url,
                "file_kind": "pdf",
                "storage_bucket": "papers",
                "storage_path": storage_path,
                "processing_status": "pending"
            }

            response = supabase.table("papers").insert(paper_data).execute()
            paper = response.data[0]

            logger.info(f"Created paper record: {paper['id']}")

            return ImportPaperResponse(
                success=True,
                paper_id=paper["id"],
                message=f"Paper '{paper_title}' uploaded successfully"
            )

        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cookies", response_model=SetCookiesResponse)
async def set_cookies(request: SetCookiesRequest):
    """
    Save cookies for a domain to help bypass Cloudflare protection.

    Use this to save cookies from your browser session:
    1. Open the paper page in your browser
    2. Open DevTools > Application > Cookies
    3. Copy cf_clearance and other cookies
    4. Send them here

    Example:
    {
        "domain": "papers.ssrn.com",
        "cookies": {
            "cf_clearance": "your_cookie_value",
            "__cf_bm": "another_value"
        }
    }
    """
    try:
        from lib.paper_import import save_cookies
        save_cookies(request.domain, request.cookies)
        logger.info(f"Saved cookies for {request.domain}: {list(request.cookies.keys())}")
        return SetCookiesResponse(
            success=True,
            message=f"Saved {len(request.cookies)} cookies for {request.domain}"
        )
    except Exception as e:
        logger.error(f"Failed to save cookies: {e}", exc_info=True)
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


@router.get("/{paper_id}/notes", response_model=PaperNoteResponse)
async def get_paper_notes(paper_id: str, supabase = Depends(get_supabase)):
    """
    Get notes and rating for a specific paper.

    Returns the note record if it exists, or a default response if not.
    """
    try:
        response = supabase.table("paper_notes").select("*").eq("paper_id", paper_id).execute()

        if response.data:
            note = response.data[0]
            return PaperNoteResponse(
                id=note["id"],
                paper_id=note["paper_id"],
                rating=note.get("rating"),
                notes=note.get("notes"),
                created_at=note.get("created_at"),
                updated_at=note.get("updated_at"),
            )
        else:
            # Return empty response for paper with no notes
            return PaperNoteResponse(paper_id=paper_id)

    except Exception as e:
        logger.error(f"Failed to get paper notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{paper_id}/notes", response_model=PaperNoteResponse)
async def update_paper_notes(
    paper_id: str,
    request: PaperNoteRequest,
    supabase = Depends(get_supabase)
):
    """
    Create or update notes and rating for a paper (upsert).

    If a note record exists, it will be updated. Otherwise, a new one is created.
    """
    try:
        # Validate rating if provided
        if request.rating and request.rating not in ('ignore', 'ok', 'highlight'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rating: {request.rating}. Must be 'ignore', 'ok', or 'highlight'"
            )

        # Check if note exists
        existing = supabase.table("paper_notes").select("id").eq("paper_id", paper_id).execute()

        from datetime import datetime
        now = datetime.utcnow().isoformat()

        if existing.data:
            # Update existing note
            update_data = {"updated_at": now}
            if request.rating is not None:
                update_data["rating"] = request.rating if request.rating else None
            if request.notes is not None:
                update_data["notes"] = request.notes if request.notes else None

            response = (
                supabase.table("paper_notes")
                .update(update_data)
                .eq("paper_id", paper_id)
                .execute()
            )
        else:
            # Create new note
            insert_data = {
                "paper_id": paper_id,
                "rating": request.rating if request.rating else None,
                "notes": request.notes if request.notes else None,
                "created_at": now,
                "updated_at": now,
            }
            response = supabase.table("paper_notes").insert(insert_data).execute()

        if response.data:
            note = response.data[0]
            return PaperNoteResponse(
                id=note["id"],
                paper_id=note["paper_id"],
                rating=note.get("rating"),
                notes=note.get("notes"),
                created_at=note.get("created_at"),
                updated_at=note.get("updated_at"),
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save paper notes")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update paper notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
