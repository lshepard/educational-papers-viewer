"""
Batch Processing Module

Process multiple papers in the background for content extraction and indexing.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from supabase import Client
from google import genai

logger = logging.getLogger(__name__)


async def extract_single_paper(
    paper_id: str,
    supabase: Client,
    genai_client: genai.Client
) -> Dict[str, Any]:
    """
    Extract content from a single paper.

    Args:
        paper_id: Paper ID to process
        supabase: Supabase client
        genai_client: Google GenAI client for file uploads

    Returns:
        Result dictionary with success status
    """
    try:
        # Import here to avoid circular dependency
        from lib.pdf_analyzer import extract_paper_sections, extract_images_from_pdf, create_paper_slug
        import tempfile
        import os
        from pathlib import Path

        logger.info(f"Processing paper: {paper_id}")

        # Update status
        supabase.table("papers").update({
            "processing_status": "processing"
        }).eq("id", paper_id).execute()

        # Fetch paper
        paper_response = supabase.table("papers").select("*").eq("id", paper_id).single().execute()
        paper = paper_response.data

        if not paper:
            raise ValueError("Paper not found")

        # Get PDF URL
        if paper.get("storage_bucket") and paper.get("storage_path"):
            # Clean storage path
            storage_path = paper["storage_path"]
            bucket_prefix = f"{paper['storage_bucket']}/"
            if storage_path.startswith(bucket_prefix):
                storage_path = storage_path[len(bucket_prefix):]

            pdf_url = supabase.storage.from_(paper["storage_bucket"]).get_public_url(storage_path)
        elif paper.get("paper_url"):
            pdf_url = paper["paper_url"]
        else:
            raise ValueError("No PDF URL available")

        # Download PDF
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, timeout=60.0)
            response.raise_for_status()
            pdf_content = response.content

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(pdf_content)
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            # Upload to Gemini
            upload_response = genai_client.files.upload(file=temp_file_path)
            gemini_file = upload_response

            # Extract sections and images in parallel
            sections_task = asyncio.to_thread(extract_paper_sections, gemini_file)
            images_task = asyncio.to_thread(extract_images_from_pdf, temp_file_path)

            sections, images = await asyncio.gather(sections_task, images_task)

            logger.info(f"Extracted {len(sections)} sections, {len(images)} images")

            # Create paper slug for organizing images
            paper_slug = create_paper_slug(paper.get("title", paper_id))

            # Upload images
            from lib.storage import upload_image_to_storage

            images_stored = []
            for img in images:
                filename = f"page-{img['page_num']}-image-{img['image_index']}.{img['ext']}"

                storage_path = upload_image_to_storage(
                    image_bytes=img['image_bytes'],
                    paper_slug=paper_slug,
                    filename=filename,
                    bucket=paper.get("storage_bucket", "papers")
                )

                if storage_path:
                    image_type = 'vector' if img.get('source') == 'rendered' else 'embedded'
                    images_stored.append({
                        "paper_id": paper_id,
                        "page_number": img['page_num'],
                        "image_type": image_type,
                        "storage_path": storage_path,
                        "width": img['width'],
                        "height": img['height']
                    })

            # Store in database
            if sections:
                # Delete existing sections
                supabase.table("paper_sections").delete().eq("paper_id", paper_id).execute()

                sections_to_insert = [
                    {
                        "paper_id": paper_id,
                        "section_type": section["section_type"],
                        "section_title": section.get("section_title"),
                        "content": section["content"]
                    }
                    for section in sections
                ]
                supabase.table("paper_sections").insert(sections_to_insert).execute()

            if images_stored:
                # Delete existing images
                supabase.table("paper_images").delete().eq("paper_id", paper_id).execute()
                supabase.table("paper_images").insert(images_stored).execute()

            # Update status
            supabase.table("papers").update({
                "processing_status": "completed",
                "processed_at": "now()"
            }).eq("id", paper_id).execute()

            # Cleanup Gemini file
            genai_client.files.delete(name=gemini_file.name)

            return {
                "success": True,
                "paper_id": paper_id,
                "sections_count": len(sections),
                "images_count": len(images_stored)
            }

        finally:
            # Cleanup temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Failed to process paper {paper_id}: {e}", exc_info=True)

        # Update status to failed
        supabase.table("papers").update({
            "processing_status": "failed",
            "processing_error": str(e)
        }).eq("id", paper_id).execute()

        return {
            "success": False,
            "paper_id": paper_id,
            "error": str(e)
        }


async def batch_extract_papers(
    supabase: Client,
    genai_client: genai.Client,
    limit: Optional[int] = None,
    status_filter: str = "pending"
) -> Dict[str, Any]:
    """
    Batch extract content from multiple papers.

    Args:
        supabase: Supabase client
        genai_client: Google GenAI client for file uploads
        limit: Maximum number of papers to process (None = all)
        status_filter: Only process papers with this status

    Returns:
        Summary of batch processing results
    """
    try:
        logger.info(f"Starting batch extraction (status={status_filter}, limit={limit})")

        # Fetch papers to process
        query = supabase.table("papers").select("id, title").eq("processing_status", status_filter)

        if limit:
            query = query.limit(limit)

        response = query.execute()
        papers = response.data

        if not papers:
            return {
                "success": True,
                "message": "No papers to process",
                "total": 0,
                "processed": 0,
                "succeeded": 0,
                "failed": 0
            }

        logger.info(f"Found {len(papers)} papers to process")

        # Process papers sequentially (to avoid overwhelming API limits)
        results = []
        succeeded = 0
        failed = 0

        for paper in papers:
            result = await extract_single_paper(paper["id"], supabase, genai_client)
            results.append(result)

            if result["success"]:
                succeeded += 1
                logger.info(f"✓ Processed: {paper['title']}")
            else:
                failed += 1
                logger.error(f"✗ Failed: {paper['title']} - {result.get('error')}")

            # Small delay between papers to be respectful to API
            await asyncio.sleep(1)

        return {
            "success": True,
            "message": f"Processed {len(papers)} papers",
            "total": len(papers),
            "processed": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "total": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0
        }


async def get_processing_stats(supabase: Client) -> Dict[str, Any]:
    """
    Get statistics about paper processing status.

    Returns:
        Dictionary with counts by status
    """
    try:
        # Get counts by status
        response = supabase.table("papers").select("processing_status").execute()
        papers = response.data

        stats = {
            "total": len(papers),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }

        for paper in papers:
            status = paper.get("processing_status", "pending")
            if status in stats:
                stats[status] += 1

        # Get section counts
        sections_response = supabase.table("paper_sections").select("id", count="exact").execute()
        stats["total_sections"] = sections_response.count if hasattr(sections_response, 'count') else 0

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }
