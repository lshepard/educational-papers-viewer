#!/usr/bin/env python3
"""
Papers Viewer Backend - PDF Extraction Service
FastAPI backend for extracting content from research papers using Gemini AI
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List
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


class PaperSections(BaseModel):
    """Structured output model for extracted paper sections."""
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    methods: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None
    other: Optional[str] = None



def clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unsupported fields from JSON schema for Gemini API.
    
    Gemini Schema proto doesn't support: title, anyOf, oneOf, allOf, definitions, $defs
    
    Args:
        schema: The JSON schema dictionary
        
    Returns:
        A cleaned schema dictionary compatible with Gemini Schema proto
    """
    if not isinstance(schema, dict):
        return schema
    
    # Fields not supported by Gemini Schema proto
    unsupported_fields = {"title", "anyOf", "oneOf", "allOf", "definitions", "$defs", "default"}
    
    # Create a copy to avoid modifying the original, excluding unsupported fields
    cleaned = {k: v for k, v in schema.items() if k not in unsupported_fields}
    
    # Recursively clean nested schemas
    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {
            k: clean_schema_for_gemini(v) for k, v in cleaned["properties"].items()
        }
    
    if "items" in cleaned:
        cleaned["items"] = clean_schema_for_gemini(cleaned["items"])
    
    return cleaned


def save_error_response(response_text: str, error_type: str, error: Exception) -> str:
    """
    Save error response to a file and log it.
    
    Args:
        response_text: The raw response text that failed to parse
        error_type: Type of error (e.g., 'sections', 'images')
        error: The exception that occurred
        
    Returns:
        Path to the saved error file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_dir = "error_responses"
    os.makedirs(error_dir, exist_ok=True)
    
    error_file = os.path.join(error_dir, f"{error_type}_error_{timestamp}.txt")
    
    try:
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Error: {str(error)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write("RAW RESPONSE:\n")
            f.write("=" * 80 + "\n")
            f.write(response_text)
        
        logger.error(f"Saved error response to: {error_file}")
        logger.error(f"Error response content (first 1000 chars): {response_text[:1000]}")
    except Exception as save_error:
        logger.error(f"Failed to save error response to file: {save_error}")
        logger.error(f"Error response content (first 1000 chars): {response_text[:1000]}")
    
    return error_file


def extract_paper_sections(gemini_file) -> list[Dict[str, Any]]:
    """
    Extract sections from a research paper PDF using Gemini structured output.
    
    Args:
        gemini_file: The Gemini file object representing the uploaded PDF
        
    Returns:
        A list of section dictionaries with section_type, section_title, and content
    """
    logger.info("Extracting paper sections with structured output...")
    
    sections_prompt = """
Extract all text sections from this research paper PDF.

Extract these sections if present:
- Abstract
- Introduction  
- Methods/Methodology - BE VERY DETAILED HERE, extract every detail about the methodology
- Results
- Discussion
- Conclusion
- Any other sections (captured in the 'other' field)

Include the COMPLETE text content for each section. For Methods, extract every detail about the methodology.
"""
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": PaperSections
        },
    )
    
    response = model.generate_content([gemini_file, sections_prompt])
    
    # Parse the structured response
    try:
        sections_data = json.loads(response.text)
    except Exception as e:
        error_file = save_error_response(response.text, "sections", e)
        raise ValueError(f"Failed to parse JSON from Gemini sections response: {e}. Full response saved to: {error_file}")
    
    paper_sections = PaperSections(**sections_data)
    
    # Convert to the format expected by the rest of the code
    sections = []
    section_mapping = {
        "abstract": ("abstract", "Abstract"),
        "introduction": ("introduction", "Introduction"),
        "methods": ("methods", "Methods"),
        "results": ("results", "Results"),
        "discussion": ("discussion", "Discussion"),
        "conclusion": ("conclusion", "Conclusion"),
        "other": ("other", "Other"),
    }
    
    for field_name, (section_type, section_title) in section_mapping.items():
        content = getattr(paper_sections, field_name)
        if content:
            sections.append({
                "section_type": section_type,
                "section_title": section_title,
                "content": content
            })
    
    logger.info(f"Extracted {len(sections)} sections from structured output")
    return sections


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
    Extract sections from a research paper using Gemini AI.

    This endpoint:
    1. Fetches the paper from Supabase
    2. Downloads the PDF
    3. Uploads to Gemini Files API
    4. Runs two extraction passes (sections)
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

        # ========== EXTRACTION 1: Text Sections ==========
        logger.info("Extracting text sections...")
        sections = extract_paper_sections(gemini_file)

        logger.info(f"Extracted {len(sections)} sections")

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

        # Update paper status
        supabase.table("papers").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", paper_id).execute()

        logger.info(f"Extraction completed successfully for paper {paper_id}")

        return ExtractionResponse(
            success=True,
            paper_id=paper_id,
            sections_count=len(sections)
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
