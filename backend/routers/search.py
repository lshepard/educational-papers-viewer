"""
Search router - handles paper search and external API integrations.
"""

import logging
import asyncio
import httpx
from typing import List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Main search router for internal full-text search
router = APIRouter(prefix="/search", tags=["search"])

# Semantic Scholar router (no prefix to match /semantic-scholar/*)
ss_router = APIRouter(prefix="/semantic-scholar", tags=["semantic-scholar"])


# ==================== Request/Response Models ====================

class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class SearchTestRequest(BaseModel):
    query: str
    limit: int = 5


class CitationsRequest(BaseModel):
    paper_ids: List[str]


# ==================== Dependencies ====================

def get_supabase():
    """Dependency to get Supabase client."""
    from main import supabase
    return supabase


# ==================== Endpoints ====================

@router.post("/papers")
async def search_papers(request: SearchRequest, supabase = Depends(get_supabase)):
    """
    Full-text search across paper sections using PostgreSQL FTS.

    Supports:
    - Plain text: "machine learning"
    - OR operator: "neural OR network"
    - Quotes for phrases: '"deep learning"'
    - Negation: "AI -healthcare"
    """
    try:
        # Search paper sections
        response = supabase.table("paper_sections")\
            .select("id, paper_id, section_type, section_title, content, created_at")\
            .text_search("fts", request.query, config="english")\
            .limit(request.limit)\
            .execute()

        sections = response.data

        # Get unique paper IDs
        paper_ids = list(set(section["paper_id"] for section in sections))

        if not paper_ids:
            return {"success": True, "results": []}

        # Fetch paper metadata
        papers_response = supabase.table("papers")\
            .select("*")\
            .in_("id", paper_ids)\
            .execute()

        # Create lookup map
        papers_map = {p["id"]: p for p in papers_response.data}

        # Combine sections with paper metadata
        results = []
        for section in sections:
            paper = papers_map.get(section["paper_id"])
            if paper:
                results.append({
                    "section_id": section["id"],
                    "paper_id": section["paper_id"],
                    "paper_title": paper.get("title"),
                    "paper_authors": paper.get("authors"),
                    "paper_year": paper.get("year"),
                    "section_type": section["section_type"],
                    "section_title": section.get("section_title"),
                    "content_preview": section["content"][:500] if section.get("content") else ""
                })

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_search(request: SearchTestRequest, supabase = Depends(get_supabase)):
    """
    Test full-text search and return enriched results.
    Used for verifying search functionality.
    """
    try:
        response = supabase.table("paper_sections")\
            .select("id, paper_id, section_type, section_title, content")\
            .limit(request.limit)\
            .text_search("fts", request.query)\
            .execute()

        sections = response.data

        # Get paper metadata
        paper_ids = list(set(s["paper_id"] for s in sections))

        if not paper_ids:
            return {"success": True, "results": []}

        papers_response = supabase.table("papers")\
            .select("id, title, authors")\
            .in_("id", paper_ids)\
            .execute()

        papers_map = {p["id"]: p for p in papers_response.data}

        # Enrich results
        results = []
        for section in sections:
            paper = papers_map.get(section["paper_id"])
            results.append({
                "section_id": section["id"],
                "paper_id": section["paper_id"],
                "paper_title": paper.get("title") if paper else None,
                "paper_authors": paper.get("authors") if paper else None,
                "section_type": section["section_type"],
                "section_title": section.get("section_title"),
                "content_preview": section["content"][:300] if section.get("content") else ""
            })

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Search test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Semantic Scholar Endpoints ====================

@ss_router.get("/search")
async def search_semantic_scholar(q: str = Query(...), limit: int = 10):
    """
    Search Semantic Scholar for papers.

    Proxy to Semantic Scholar API with retry logic for rate limiting.
    Returns papers with metadata including title, authors, citations, etc.
    """
    try:
        max_retries = 3

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": q,
                            "limit": limit,
                            "fields": "paperId,title,authors,year,abstract,venue,citationCount,externalIds,url"
                        },
                        timeout=30.0
                    )

                    if response.status_code == 429:  # Rate limited
                        if attempt < max_retries - 1:
                            delay = 2 ** attempt
                            logger.warning(f"Semantic Scholar rate limit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise HTTPException(status_code=429, detail="Semantic Scholar API rate limit exceeded")

                    response.raise_for_status()
                    data = response.json()

                    return {
                        "success": True,
                        "papers": data.get("data", []),  # Match old API
                        "total": data.get("total", 0)
                    }

                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 429:
                        raise

        raise HTTPException(status_code=500, detail="Failed after all retries")

    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ss_router.post("/citations")
async def fetch_citations(request: CitationsRequest):
    """
    Fetch citation data for multiple papers from Semantic Scholar.

    Accepts a list of paper IDs and returns citation metadata for each.
    Tries multiple ID formats (with/without ARXIV: prefix).
    """
    try:
        results = []
        seen_paper_ids = set()

        async with httpx.AsyncClient() as client:
            for idx, paper_id in enumerate(request.paper_ids):
                # Add delay between requests
                if idx > 0:
                    await asyncio.sleep(0.5)

                # Skip duplicates
                if paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(paper_id)

                try:
                    # Try different ID formats
                    for id_prefix in ["ARXIV:", ""]:
                        try:
                            search_id = f"{id_prefix}{paper_id}"
                            response = await client.get(
                                f"https://api.semanticscholar.org/graph/v1/paper/{search_id}",
                                params={"fields": "paperId,title,authors,year,citationCount,venue"},
                                timeout=10.0
                            )

                            if response.status_code == 200:
                                data = response.json()
                                results.append({
                                    "paper_id": paper_id,
                                    "found": True,
                                    "data": data
                                })
                                break

                        except Exception:
                            continue
                    else:
                        # Not found with any format
                        results.append({
                            "paper_id": paper_id,
                            "found": False,
                            "data": None
                        })

                except Exception as e:
                    logger.warning(f"Failed to fetch citations for {paper_id}: {e}")
                    results.append({
                        "paper_id": paper_id,
                        "found": False,
                        "error": str(e)
                    })

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Citations fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
