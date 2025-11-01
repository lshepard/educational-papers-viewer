#!/usr/bin/env python3
"""
Papers Viewer Backend - PDF Extraction Service
FastAPI backend for extracting content from research papers using Gemini AI
"""

import os
import json
import logging
import tempfile
import io
import re
import asyncio
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
import fitz  # PyMuPDF
from PIL import Image

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


class PaperSections(BaseModel):
    """Structured output model for extracted paper sections."""
    abstract: Optional[str]
    introduction: Optional[str]
    methods: Optional[str]
    results: Optional[str]
    discussion: Optional[str]
    conclusion: Optional[str]
    other: Optional[str]


class SearchRequest(BaseModel):
    """Request model for full-text search."""
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    paper_id: str
    section_type: str
    section_title: Optional[str]
    content: str
    created_at: str


class SearchResponse(BaseModel):
    """Response model for full-text search."""
    success: bool
    query: str
    results: List[SearchResult]
    count: int





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


def create_paper_slug(title: str) -> str:
    """
    Create a URL-friendly slug from paper title.

    Args:
        title: Paper title

    Returns:
        Slug like "problem-solving-teaching-problem-solving-aligning-llms"
    """
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    # Limit length
    slug = slug[:80].strip('-')
    return slug


def extract_figure_regions_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract figure/table regions from PDF by detecting captions and rendering those areas.
    This captures vector graphics (charts, graphs) that aren't embedded as images.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with rendered figure images
    """
    logger.info(f"Extracting figure regions from PDF: {pdf_path}")
    figures = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Find figure/table captions
            text_dict = page.get_text("dict")
            captions = []

            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]])
                    # Look for figure/table captions
                    if any(keyword in text for keyword in ["Figure ", "Table ", "Fig. "]):
                        captions.append({
                            'text': text,
                            'bbox': block['bbox']
                        })

            # For each caption, render a region around it
            for cap_idx, caption in enumerate(captions):
                try:
                    # Expand the bounding box to capture the figure
                    # Typical figures are 200-400px tall
                    bbox = caption['bbox']
                    x0, y0, x1, y1 = bbox

                    # Expand region (adjust as needed)
                    figure_bbox = fitz.Rect(
                        max(0, x0 - 20),
                        max(0, y0 - 250),  # Look above caption
                        min(page.rect.width, x1 + 20),
                        min(page.rect.height, y1 + 20)
                    )

                    # Render this region at high DPI
                    mat = fitz.Matrix(2, 2)  # 2x zoom = ~144 DPI
                    pix = page.get_pixmap(matrix=mat, clip=figure_bbox)

                    # Convert to PNG bytes
                    image_bytes = pix.tobytes("png")

                    # Get dimensions
                    width, height = pix.width, pix.height

                    # Skip if too small
                    if width < 100 or height < 100:
                        continue

                    figures.append({
                        'page_num': page_num + 1,
                        'image_index': cap_idx + 1,
                        'image_bytes': image_bytes,
                        'ext': 'png',
                        'width': width,
                        'height': height,
                        'caption': caption['text'][:100]  # First 100 chars
                    })

                    logger.info(f"Rendered figure: page {page_num + 1}, caption: {caption['text'][:50]}..., size {width}x{height}")

                except Exception as e:
                    logger.warning(f"Failed to render figure region on page {page_num}: {e}")
                    continue

        doc.close()
        logger.info(f"Extracted {len(figures)} figure regions from PDF")
        return figures

    except Exception as e:
        logger.error(f"Failed to extract figure regions from PDF: {e}")
        return []


def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all images from a PDF using PyMuPDF.
    This includes both:
    1. Embedded raster images (PNG/JPG)
    2. Rendered figure/table regions (vector graphics)

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with: {
            'page_num': int,
            'image_index': int,
            'image_bytes': bytes,
            'ext': str (png/jpg),
            'width': int,
            'height': int,
            'caption': str (optional, for rendered figures)
        }
    """
    logger.info(f"Extracting images from PDF: {pdf_path}")
    images = []

    try:
        doc = fitz.open(pdf_path)

        # Method 1: Extract embedded images
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]  # Image XREF number

                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Get image dimensions
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size

                    # Skip very small images (likely logos, icons)
                    if width < 100 or height < 100:
                        logger.debug(f"Skipping small embedded image: {width}x{height}")
                        continue

                    images.append({
                        'page_num': page_num + 1,  # 1-indexed
                        'image_index': img_index + 1,
                        'image_bytes': image_bytes,
                        'ext': image_ext,
                        'width': width,
                        'height': height,
                        'source': 'embedded'
                    })

                    logger.info(f"Extracted embedded image: page {page_num + 1}, index {img_index + 1}, size {width}x{height}, format {image_ext}")

                except Exception as e:
                    logger.warning(f"Failed to extract embedded image {img_index} from page {page_num}: {e}")
                    continue

        doc.close()

        # Method 2: Render figure regions (for vector graphics)
        figure_images = extract_figure_regions_from_pdf(pdf_path)

        # Merge both methods
        # Re-index the figure images
        next_index = {}
        for img in images:
            page = img['page_num']
            next_index[page] = max(next_index.get(page, 0), img['image_index'])

        for fig in figure_images:
            page = fig['page_num']
            fig['image_index'] = next_index.get(page, 0) + 1
            next_index[page] = fig['image_index']
            fig['source'] = 'rendered'
            images.append(fig)

        logger.info(f"Extracted {len(images)} total images from PDF (embedded + rendered)")
        return images

    except Exception as e:
        logger.error(f"Failed to extract images from PDF: {e}")
        return []


def upload_image_to_storage(
    image_bytes: bytes,
    paper_slug: str,
    filename: str,
    bucket: str = "papers"
) -> Optional[str]:
    """
    Upload an image to Supabase storage.

    Args:
        image_bytes: Image data
        paper_slug: Paper slug for organizing images
        filename: Filename (e.g., "page-1-image-1.png")
        bucket: Supabase storage bucket name

    Returns:
        Storage path if successful, None otherwise
    """
    storage_path = f"images/{paper_slug}/{filename}"

    try:
        # Upload to Supabase storage
        supabase.storage.from_(bucket).upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": f"image/{filename.split('.')[-1]}"}
        )

        logger.info(f"Uploaded image to storage: {storage_path}")
        return storage_path

    except Exception as e:
        logger.error(f"Failed to upload image to storage: {e}")
        return None


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
            "/search": "POST - Full-text search across paper sections",
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


@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Full-text search across paper sections using PostgreSQL's tsvector.

    This endpoint searches the `fts` column which contains indexed text from
    both section_title and content fields.

    Query examples:
    - "machine learning" - searches for both words (AND)
    - "neural | network" - searches for either word (OR)
    - "deep & learning" - explicit AND
    - "transformer & !attention" - excludes documents with "attention"

    Args:
        request: SearchRequest with query string and optional limit

    Returns:
        SearchResponse with matching sections ordered by relevance
    """
    try:
        logger.info(f"Searching for: {request.query}")

        # Use Supabase's textSearch method on the fts column
        # The query is processed using websearch_to_tsquery which supports:
        # - Plain text (converted to AND queries)
        # - "quoted phrases"
        # - word1 OR word2
        # - -excluded
        response = supabase.table("paper_sections") \
            .select("id, paper_id, section_type, section_title, content, created_at") \
            .limit(request.limit) \
            .text_search("fts", request.query) \
            .execute()

        results = []
        for section in response.data:
            results.append(SearchResult(
                id=section["id"],
                paper_id=section["paper_id"],
                section_type=section["section_type"],
                section_title=section.get("section_title"),
                content=section["content"],
                created_at=section["created_at"]
            ))

        logger.info(f"Found {len(results)} results for query: {request.query}")

        return SearchResponse(
            success=True,
            query=request.query,
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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

        # ========== PARALLEL EXTRACTION ==========
        logger.info("Starting parallel extraction: text sections and images...")

        # Run both extractions in parallel using asyncio.gather
        sections_task = asyncio.to_thread(extract_paper_sections, gemini_file)
        images_task = asyncio.to_thread(extract_images_from_pdf, temp_file_path)

        sections, images = await asyncio.gather(sections_task, images_task)

        logger.info(f"Parallel extraction complete: {len(sections)} sections, {len(images)} images")

        # Create paper slug for organizing images
        paper_title = paper.get("title", paper_id)
        paper_slug = create_paper_slug(paper_title)

        # Upload images and store metadata
        images_stored = []
        if images:
            for img in images:
                filename = f"page-{img['page_num']}-image-{img['image_index']}.{img['ext']}"

                # Upload to Supabase storage
                storage_path = upload_image_to_storage(
                    image_bytes=img['image_bytes'],
                    paper_slug=paper_slug,
                    filename=filename,
                    bucket=paper.get("storage_bucket", "papers")
                )

                if storage_path:
                    # Determine image type based on source
                    # 'embedded' = raster image (PNG/JPG) embedded in PDF
                    # 'vector' = rendered figure/chart/table (originally vector graphics)
                    image_type = 'vector' if img.get('source') == 'rendered' else 'embedded'

                    images_stored.append({
                        "paper_id": paper_id,
                        "page_number": img['page_num'],
                        "image_type": image_type,
                        "storage_path": storage_path,
                        "width": img['width'],
                        "height": img['height']
                    })

            logger.info(f"Uploaded {len(images_stored)} images to storage")

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
        if images_stored:
            supabase.table("paper_images").insert(images_stored).execute()
            logger.info(f"Stored {len(images_stored)} image records in database")

        # Update paper status
        supabase.table("papers").update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", paper_id).execute()

        logger.info(f"Extraction completed successfully for paper {paper_id}")

        return ExtractionResponse(
            success=True,
            paper_id=paper_id,
            sections_count=len(sections),
            images_count=len(images_stored)
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
