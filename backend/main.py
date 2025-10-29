#!/usr/bin/env python3
"""
Papers Viewer Backend - PDF Extraction Service
FastAPI backend for extracting content from research papers using Gemini AI
"""

import os
import json
import logging
import tempfile
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
genai.configure(api_key=GEMINI_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting Papers Viewer Backend...")
    logger.info("Gemini API configured")
    logger.info("Supabase client initialized")
    yield
    logger.info("Shutting down Papers Viewer Backend...")


# Create FastAPI app
app = FastAPI(
    title="Papers Viewer Backend",
    description="Backend API for extracting content from research papers",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractionRequest(BaseModel):
    """Request model for paper extraction."""
    paper_id: str


class ExtractionResponse(BaseModel):
    """Response model for paper extraction."""
    success: bool
    paper_id: str
    sections_count: int
    images_count: int


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "success": True,
        "message": "Papers Viewer Backend API",
        "version": "0.1.0",
        "endpoints": {
            "/extract": "POST - Extract content from paper",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Supabase connection
        supabase.table("papers").select("id").limit(1).execute()

        return {
            "success": True,
            "status": "healthy",
            "services": {
                "supabase": "connected",
                "gemini": "configured"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_paper_content(request: ExtractionRequest):
    """
    Extract sections and images from a research paper using Gemini AI.

    This endpoint:
    1. Fetches the paper from Supabase
    2. Downloads the PDF
    3. Uploads to Gemini Files API
    4. Runs two extraction passes (sections + images)
    5. Stores results in Supabase
    """
    paper_id = request.paper_id
    temp_file_path = None
    gemini_file = None

    try:
        logger.info(f"Starting extraction for paper: {paper_id}")

        # Update status to processing
        supabase.table("papers").update({
            "processing_status": "processing"
        }).eq("id", paper_id).execute()

        # Fetch paper details
        paper_response = supabase.table("papers").select("*").eq("id", paper_id).single().execute()
        paper = paper_response.data

        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        # Get PDF URL
        if paper.get("storage_bucket") and paper.get("storage_path"):
            # Clean storage path (remove duplicate bucket prefix)
            storage_path = paper["storage_path"]
            bucket_prefix = f"{paper['storage_bucket']}/"
            if storage_path.startswith(bucket_prefix):
                storage_path = storage_path[len(bucket_prefix):]

            # Get public URL
            pdf_url = supabase.storage.from_(paper["storage_bucket"]).get_public_url(storage_path)
            logger.info(f"Using storage URL: {pdf_url}")
        elif paper.get("paper_url"):
            pdf_url = paper["paper_url"]
            logger.info(f"Using external URL: {pdf_url}")
        else:
            raise HTTPException(status_code=400, detail="No PDF URL available")

        # Download PDF
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, timeout=60.0)
            response.raise_for_status()
            pdf_content = response.content

        logger.info(f"PDF downloaded, size: {len(pdf_content)} bytes")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name

        # Upload to Gemini
        logger.info("Uploading PDF to Gemini...")
        gemini_file = genai.upload_file(temp_file_path)
        logger.info(f"File uploaded to Gemini: {gemini_file.name}")

        # Initialize model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # ========== EXTRACTION 1: Text Sections ==========
        logger.info("Extracting text sections...")
        sections_prompt = """
Extract all text sections from this research paper PDF.

Return ONLY valid JSON with this structure (no markdown, no explanations):
{"sections":[{"section_type":"abstract","section_title":"Abstract","content":"full text here"},...]}

Extract these sections if present:
- Abstract (section_type: "abstract")
- Introduction (section_type: "introduction")
- Background/Related Work (section_type: "background")
- Methods/Methodology (section_type: "methods") - BE VERY DETAILED HERE
- Results (section_type: "results")
- Discussion (section_type: "discussion")
- Conclusion (section_type: "conclusion")
- Any other sections (section_type: "other")

Include the COMPLETE text content for each section. For Methods, extract every detail about the methodology.
"""

        sections_response = model.generate_content([gemini_file, sections_prompt])
        sections_text = sections_response.text

        # Clean and parse JSON
        sections_text = sections_text.replace("```json", "").replace("```", "").strip()
        sections_data = json.loads(sections_text)
        sections = sections_data.get("sections", [])

        logger.info(f"Extracted {len(sections)} sections")

        # ========== EXTRACTION 2: Images and Figures ==========
        logger.info("Extracting images and figures...")
        images_prompt = """
Identify and describe ALL images, figures, charts, diagrams, screenshots, and tables in this research paper PDF.

Return ONLY valid JSON with this structure (no markdown, no explanations):
{"images":[{"image_type":"figure","caption":"Figure 1: ...","description":"detailed description","page_number":1},...]}

For each visual element:
- image_type: choose from screenshot, chart, figure, diagram, table, other
- caption: extract the exact caption if present
- description: describe what the image shows in detail
- page_number: the page it appears on (if identifiable)

Be thorough - identify EVERY visual element in the paper.
"""

        images_response = model.generate_content([gemini_file, images_prompt])
        images_text = images_response.text

        # Clean and parse JSON
        images_text = images_text.replace("```json", "").replace("```", "").strip()
        images_data = json.loads(images_text)
        images = images_data.get("images", [])

        logger.info(f"Extracted {len(images)} images")

        # ========== Store in Supabase ==========
        # Store sections
        if sections:
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
            logger.info(f"Stored {len(sections)} sections in database")

        # Store images
        if images:
            images_to_insert = [
                {
                    "paper_id": paper_id,
                    "image_type": image["image_type"],
                    "caption": image.get("caption"),
                    "description": image.get("description"),
                    "page_number": image.get("page_number")
                }
                for image in images
            ]
            supabase.table("paper_images").insert(images_to_insert).execute()
            logger.info(f"Stored {len(images)} images in database")

        # Update paper status
        from datetime import datetime
        supabase.table("papers").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", paper_id).execute()

        logger.info(f"Extraction completed successfully for paper {paper_id}")

        return ExtractionResponse(
            success=True,
            paper_id=paper_id,
            sections_count=len(sections),
            images_count=len(images)
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini: {e}")
        supabase.table("papers").update({
            "processing_status": "failed",
            "processing_error": f"JSON parse error: {str(e)}"
        }).eq("id", paper_id).execute()
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        supabase.table("papers").update({
            "processing_status": "failed",
            "processing_error": str(e)
        }).eq("id", paper_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

        if gemini_file:
            try:
                gemini_file.delete()
                logger.debug(f"Deleted Gemini file: {gemini_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete Gemini file: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
