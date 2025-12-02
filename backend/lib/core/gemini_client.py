"""
Centralized Gemini AI file management.

Handles file uploads, downloads, and cleanup with automatic resource management.
"""

import logging
import tempfile
import httpx
from typing import Optional
from contextlib import asynccontextmanager

try:
    from google import genai
except ImportError:
    import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiFileManager:
    """Manages file uploads to Gemini AI with automatic cleanup."""

    def __init__(self, client: genai.Client):
        self.client = client

    @asynccontextmanager
    async def upload_pdf(self, file_path: str):
        """
        Upload PDF file to Gemini with automatic cleanup.

        Args:
            file_path: Path to local PDF file

        Yields:
            Gemini file upload response

        Example:
            async with gemini_manager.upload_pdf(path) as gemini_file:
                sections = await extract_sections(gemini_file)
                # Automatic cleanup on exit
        """
        gemini_file = None
        try:
            logger.info(f"Uploading PDF to Gemini: {file_path}")
            upload_response = self.client.files.upload(file=file_path)
            gemini_file = upload_response
            logger.info(f"File uploaded: {upload_response.uri} (name: {upload_response.name})")
            yield gemini_file
        finally:
            if gemini_file:
                try:
                    self.client.files.delete(name=gemini_file.name)
                    logger.debug(f"Cleaned up Gemini file: {gemini_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete Gemini file: {e}")

    @asynccontextmanager
    async def upload_pdf_from_url(self, url: str):
        """
        Download PDF from URL and upload to Gemini with automatic cleanup.

        Args:
            url: URL to download PDF from

        Yields:
            Gemini file upload response
        """
        temp_file_path = None
        try:
            # Download PDF
            logger.info(f"Downloading PDF from: {url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()
                pdf_content = response.content

            logger.info(f"PDF downloaded, size: {len(pdf_content)} bytes")

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name

            # Upload to Gemini
            async with self.upload_pdf(temp_file_path) as gemini_file:
                yield gemini_file

        finally:
            # Cleanup temp file
            if temp_file_path:
                try:
                    import os
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
