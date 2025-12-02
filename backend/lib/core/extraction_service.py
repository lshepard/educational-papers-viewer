"""
Unified paper extraction service.

Consolidates all paper content extraction logic (sections, images) into a single service.
"""

import logging
import asyncio
import tempfile
from typing import Dict, Any, List
from supabase import Client

from lib.pdf_analyzer import extract_paper_sections, extract_images_from_pdf, create_paper_slug
from lib.storage import upload_image_to_storage
from .gemini_client import GeminiFileManager

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Result of paper extraction."""

    def __init__(self, sections: List[Dict], images: List[Dict], sections_count: int, images_count: int):
        self.sections = sections
        self.images = images
        self.sections_count = sections_count
        self.images_count = images_count


class PaperExtractionService:
    """
    Unified service for extracting content from research papers.

    Handles:
    - Downloading PDFs from storage or external URLs
    - Uploading to Gemini for AI processing
    - Extracting sections and images in parallel
    - Storing results in database
    - Status tracking and error handling
    """

    def __init__(self, supabase: Client, gemini_manager: GeminiFileManager):
        self.db = supabase
        self.gemini = gemini_manager

    async def extract_from_storage(self, paper_id: str) -> ExtractionResult:
        """
        Extract content from a paper already in Supabase storage.

        Args:
            paper_id: Paper ID in database

        Returns:
            ExtractionResult with sections and images

        Raises:
            Exception if extraction fails
        """
        # Fetch paper details
        paper_response = self.db.table("papers").select("*").eq("id", paper_id).execute()

        if not paper_response.data:
            raise ValueError(f"Paper not found: {paper_id}")

        paper = paper_response.data[0]

        # Get storage URL
        pdf_url = self._get_storage_url(paper)
        logger.info(f"Using storage URL: {pdf_url}")

        return await self._extract_from_url(pdf_url, paper_id, paper)

    async def extract_from_url(self, url: str, paper_id: str) -> ExtractionResult:
        """
        Extract content from a paper at an external URL.

        Args:
            url: URL to download PDF from
            paper_id: Paper ID in database

        Returns:
            ExtractionResult with sections and images
        """
        # Fetch paper details
        paper_response = self.db.table("papers").select("*").eq("id", paper_id).execute()
        paper = paper_response.data[0] if paper_response.data else None

        return await self._extract_from_url(url, paper_id, paper)

    async def _extract_from_url(
        self,
        url: str,
        paper_id: str,
        paper: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Core extraction logic - single implementation used by all entry points.

        Args:
            url: PDF URL
            paper_id: Paper ID
            paper: Paper metadata dict

        Returns:
            ExtractionResult
        """
        await self._update_status(paper_id, "processing")

        try:
            # Download and upload to Gemini with automatic cleanup
            async with self.gemini.upload_pdf_from_url(url) as gemini_file:

                # Extract sections and images in parallel
                logger.info("Starting parallel extraction: sections and images...")
                sections_task = asyncio.to_thread(extract_paper_sections, gemini_file)

                # For images, we need to download the PDF again to a temp file
                images_task = self._extract_images_from_url(url)

                sections, images = await asyncio.gather(sections_task, images_task)

                logger.info(f"Extracted {len(sections)} sections, {len(images)} images")

                # Store results in database
                await self._store_results(paper_id, paper, sections, images)

                # Update status to completed
                await self._update_status(paper_id, "completed")

                return ExtractionResult(
                    sections=sections,
                    images=images,
                    sections_count=len(sections),
                    images_count=len(images)
                )

        except Exception as e:
            logger.error(f"Extraction failed for {paper_id}: {e}", exc_info=True)
            await self._update_status(paper_id, "failed", error=str(e))
            raise

    async def _extract_images_from_url(self, url: str) -> List[Dict[str, Any]]:
        """Extract images from PDF at URL."""
        import httpx

        # Download PDF to temp file
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60.0)
            response.raise_for_status()
            pdf_content = response.content

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name

        try:
            # Extract images
            images = await asyncio.to_thread(extract_images_from_pdf, temp_file_path)
            return images
        finally:
            # Cleanup
            import os
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

    async def _store_results(
        self,
        paper_id: str,
        paper: Dict[str, Any],
        sections: List[Dict[str, Any]],
        images: List[Dict[str, Any]]
    ):
        """Store extraction results in database."""

        # Store sections
        if sections:
            # Delete existing sections
            self.db.table("paper_sections").delete().eq("paper_id", paper_id).execute()

            sections_to_insert = [
                {
                    "paper_id": paper_id,
                    "section_type": section["section_type"],
                    "section_title": section.get("section_title"),
                    "content": section["content"]
                }
                for section in sections
            ]
            self.db.table("paper_sections").insert(sections_to_insert).execute()

        # Store images
        if images and paper:
            paper_slug = create_paper_slug(paper.get("title", paper_id))
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

            if images_stored:
                # Delete existing images
                self.db.table("paper_images").delete().eq("paper_id", paper_id).execute()
                self.db.table("paper_images").insert(images_stored).execute()

    async def _update_status(self, paper_id: str, status: str, error: str = None):
        """Update paper processing status."""
        update_data = {"processing_status": status}

        if status == "completed":
            update_data["processed_at"] = "now()"
            update_data["processing_error"] = None
        elif status == "failed" and error:
            update_data["processing_error"] = error

        self.db.table("papers").update(update_data).eq("id", paper_id).execute()

    def _get_storage_url(self, paper: Dict[str, Any]) -> str:
        """Get public storage URL for paper PDF."""
        storage_bucket = paper.get("storage_bucket", "papers")
        storage_path = paper.get("storage_path")

        if not storage_path:
            raise ValueError("Paper has no storage_path")

        # Get Supabase storage URL
        supabase_url = self.db.supabase_url
        pdf_url = f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{storage_path}"

        return pdf_url
