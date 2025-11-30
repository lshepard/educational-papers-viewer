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
from google import genai
from google.genai import types
from supabase import create_client, Client
import fitz  # PyMuPDF
from PIL import Image
from pydub import AudioSegment
from research_agent import create_research_agent

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
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def convert_audio_to_mp3(audio_data: bytes, source_format: str = 'wav', sample_rate: int = 24000, channels: int = 1) -> bytes:
    """
    Convert audio data to MP3 format.

    Args:
        audio_data: Raw audio bytes
        source_format: Source audio format ('wav', 'raw', 'pcm')
        sample_rate: Sample rate in Hz (for raw PCM)
        channels: Number of audio channels (for raw PCM)

    Returns:
        MP3-encoded audio bytes
    """
    try:
        if source_format == 'pcm' or source_format == 'raw':
            # Raw PCM data - use from_raw()
            # L16 means 16-bit = 2 bytes sample width
            audio = AudioSegment.from_raw(
                io.BytesIO(audio_data),
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=sample_rate,
                channels=channels
            )
        else:
            # File with headers (WAV, etc.) - use from_file()
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)

        # Export as MP3
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format='mp3', bitrate='128k')
        mp3_buffer.seek(0)

        return mp3_buffer.read()
    except Exception as e:
        logger.error(f"Failed to convert audio to MP3: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting Papers Viewer Backend...")
    logger.info("Gemini API configured")
    logger.info("Supabase client initialized")

    # Check Gemini native audio capabilities
    try:
        logger.info("Checking Gemini native audio capabilities...")
        # Initialize genai client
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        app.state.genai_client = genai_client
        logger.info("✅ Gemini 2.5 Pro TTS configured successfully")
        app.state.tts_available = True
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ GEMINI TTS NOT CONFIGURED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Podcast generation will NOT be available.")
        logger.error("=" * 80)

        # Fail to start if TTS is not configured
        raise RuntimeError(
            "Gemini TTS not configured. Check GEMINI_API_KEY. "
            "See logs above for details."
        )

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
# Allow both localhost and production frontend
allowed_origins = [
    "http://localhost:3000",
    FRONTEND_URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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


class PodcastGenerationRequest(BaseModel):
    """Request model for podcast generation."""
    paper_id: str


class PodcastGenerationResponse(BaseModel):
    """Response model for podcast generation."""
    success: bool
    episode_id: str
    audio_url: str
    message: str





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
        # Try to upload with upsert (overwrite if exists)
        supabase.storage.from_(bucket).upload(
            path=storage_path,
            file=image_bytes,
            file_options={
                "content-type": f"image/{filename.split('.')[-1]}",
                "upsert": "true"
            }
        )

        logger.info(f"Uploaded image to storage: {storage_path}")
        return storage_path

    except Exception as e:
        # If it still fails, try to update instead
        try:
            supabase.storage.from_(bucket).update(
                path=storage_path,
                file=image_bytes,
                file_options={"content-type": f"image/{filename.split('.')[-1]}"}
            )
            logger.info(f"Updated existing image in storage: {storage_path}")
            return storage_path
        except Exception as update_error:
            logger.error(f"Failed to upload/update image to storage: {e}, {update_error}")
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
        model_name="gemini-3-pro-preview",
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
            "/podcast/generate": "POST - Generate podcast episode from paper (takes 2-5 min)",
            "/podcast/feed.xml": "GET - Get RSS podcast feed",
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
                "gemini": "configured",
                "gemini_tts": "configured" if getattr(app.state, 'tts_available', False) else "not configured"
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
        # Store sections (delete existing first to avoid duplicates)
        if sections:
            # Delete existing sections for this paper
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
            logger.info(f"Stored {len(sections)} sections in database")

        # Store images (delete existing first to avoid duplicates)
        if images_stored:
            # Delete existing images for this paper
            supabase.table("paper_images").delete().eq("paper_id", paper_id).execute()

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


async def _generate_podcast_from_paper(paper_id: str, episode_id: str = None) -> dict:
    """
    Internal function to generate a podcast from a paper.

    Args:
        paper_id: ID of the paper to generate podcast for
        episode_id: Optional existing episode ID (for regeneration). If None, creates new episode.

    Returns:
        dict with success, episode_id, audio_url, and message
    """
    temp_file_path = None
    gemini_file = None

    try:
        logger.info(f"Generating podcast for paper: {paper_id}")

        # Create or update episode record
        if episode_id is None:
            # Create new episode
            episode_data = {
                "paper_id": paper_id,
                "title": "Generating...",
                "description": "Podcast is being generated",
                "generation_status": "processing"
            }
            episode_response = supabase.table("podcast_episodes").insert(episode_data).execute()
            episode_id = episode_response.data[0]["id"]
        else:
            # Update existing episode
            supabase.table("podcast_episodes").update({
                "generation_status": "processing",
                "generation_error": None
            }).eq("id", episode_id).execute()

        # Fetch paper details
        paper_response = supabase.table("papers").select("*").eq("id", paper_id).single().execute()
        paper = paper_response.data

        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        # Get PDF URL
        if paper.get("storage_bucket") and paper.get("storage_path"):
            storage_path = paper["storage_path"]
            bucket_prefix = f"{paper['storage_bucket']}/"
            if storage_path.startswith(bucket_prefix):
                storage_path = storage_path[len(bucket_prefix):]
            pdf_url = supabase.storage.from_(paper["storage_bucket"]).get_public_url(storage_path)
        elif paper.get("paper_url"):
            pdf_url = paper["paper_url"]
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

        # Upload to Gemini using Files API
        logger.info("Uploading PDF to Gemini...")
        genai_client = app.state.genai_client

        # Upload file
        upload_response = genai_client.files.upload(file=temp_file_path)
        gemini_file_name = upload_response.name
        gemini_file_uri = upload_response.uri
        logger.info(f"File uploaded to Gemini: {gemini_file_uri} (name: {gemini_file_name})")

        # Research paper context using Semantic Scholar and web search
        logger.info("Researching paper context...")
        research_agent = create_research_agent(genai_client)
        research_context = research_agent.research_paper_context(
            title=paper.get('title', 'Untitled Paper'),
            authors=paper.get('authors'),
            extract_products=True
        )
        logger.info(f"Research complete: {research_context['research_summary']}")

        # Generate podcast script using Gemini
        logger.info("Generating podcast script with research context...")

        # Build enhanced script prompt with research context
        script_prompt = f"""You are creating a podcast script in the style of Google's NotebookLM Audio Overviews.

RESEARCH CONTEXT:
{research_context['research_summary']}

Paper Citation Count: {research_context['paper_metadata'].get('citation_count', 'unknown')}
Number of Influential References: {len(research_context['influential_references'])}

Create an engaging, conversational podcast discussion between two hosts about this research paper:
- Host 1 (Alex): Enthusiastic and asks great questions
- Host 2 (Sam): Knowledgeable and explains concepts clearly

The podcast should:
- Incorporate the research context naturally into the discussion (mention citation impact if significant)
- Reference the paper's place in the field and important prior work when relevant
- Be lighthearted and fun, but informative
- Discuss key findings, methodology, and real-world implications
- Use natural, conversational language - NO "um", "like", or filler words (the TTS will add natural pauses)
- Be about 3-5 minutes when spoken (roughly 450-750 words)
- Make complex topics accessible and engaging

Format the script like this:
Alex: [speaks naturally]
Sam: [responds naturally]
Alex: [continues conversation]
...

Focus on making the content digestible and interesting for casual listeners."""

        # Generate script with Gemini
        script_response = genai_client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[
                types.Part.from_uri(file_uri=gemini_file_uri, mime_type="application/pdf"),
                script_prompt
            ],
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )

        script_text = script_response.text

        logger.info(f"Generated script ({len(script_text)} characters)")
        logger.debug(f"Script preview: {script_text[:200]}...")

        # Generate audio using Gemini 2.5 Pro native TTS
        logger.info("Generating audio with Gemini 2.5 Pro native TTS...")

        # Get the genai client from app state
        genai_client = app.state.genai_client

        # Generate audio with multi-speaker configuration
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=script_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker='Alex',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Kore',  # Firm, conversational
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker='Sam',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Puck',  # Upbeat, engaging
                                    )
                                )
                            ),
                        ]
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )

        # Extract audio data from response
        audio_data = None
        audio_mime_type = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    audio_data = part.inline_data.data
                    audio_mime_type = getattr(part.inline_data, 'mime_type', 'audio/wav')
                    break

        if not audio_data:
            raise HTTPException(status_code=500, detail="No audio data in Gemini response")

        logger.info(f"Audio generated, size: {len(audio_data)} bytes, mime_type: {audio_mime_type}")

        # Convert to MP3 if not already MP3
        if audio_mime_type == 'audio/mpeg' or audio_mime_type == 'audio/mp3':
            logger.info("Audio already in MP3 format")
            mp3_data = audio_data
        else:
            logger.info(f"Converting audio from {audio_mime_type} to MP3...")

            # Parse mime type to determine format and parameters
            if 'L16' in audio_mime_type or 'pcm' in audio_mime_type.lower():
                # Raw PCM format
                source_format = 'pcm'
                # Extract sample rate from mime type (e.g., "rate=24000")
                sample_rate = 24000  # Default
                if 'rate=' in audio_mime_type:
                    import re
                    rate_match = re.search(r'rate=(\d+)', audio_mime_type)
                    if rate_match:
                        sample_rate = int(rate_match.group(1))

                mp3_data = convert_audio_to_mp3(audio_data, source_format='pcm', sample_rate=sample_rate, channels=1)
            elif audio_mime_type == 'audio/wav' or audio_mime_type == 'audio/x-wav':
                mp3_data = convert_audio_to_mp3(audio_data, source_format='wav')
            else:
                # Unknown format, try as WAV
                logger.warning(f"Unknown audio mime type: {audio_mime_type}, trying as WAV")
                mp3_data = convert_audio_to_mp3(audio_data, source_format='wav')

            logger.info(f"Converted to MP3, size: {len(mp3_data)} bytes")

        # Upload audio to Supabase storage as MP3
        filename = f"{episode_id}.mp3"
        storage_path_audio = f"{paper_id}/{filename}"

        logger.info(f"Uploading audio to storage: episodes/{storage_path_audio}")
        supabase.storage.from_("episodes").upload(
            path=storage_path_audio,
            file=mp3_data,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

        # Get public URL
        public_url = supabase.storage.from_("episodes").get_public_url(storage_path_audio)

        # Create episode metadata
        episode_title = f"Discussion: {paper.get('title', 'Untitled Paper')}"
        episode_description = f"An AI-generated podcast discussing the research paper"
        if paper.get('authors'):
            episode_description += f" by {paper['authors']}"
        if paper.get('year'):
            episode_description += f" ({paper['year']})"

        # Update episode record with script and audio
        supabase.table("podcast_episodes").update({
            "title": episode_title,
            "description": episode_description,
            "script": script_text,  # Save the generated script
            "storage_path": storage_path_audio,
            "audio_url": public_url,
            "generation_status": "completed",
            "generation_error": None
        }).eq("id", episode_id).execute()

        logger.info(f"Podcast generation completed: {episode_id}")

        return {
            "success": True,
            "episode_id": episode_id,
            "audio_url": public_url,
            "message": "Podcast generated successfully!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate podcast: {e}", exc_info=True)

        # Update episode status to failed if we have an episode_id
        if episode_id:
            supabase.table("podcast_episodes").update({
                "generation_status": "failed",
                "generation_error": str(e)
            }).eq("id", episode_id).execute()

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

        if 'gemini_file_name' in locals() and gemini_file_name:
            try:
                # Delete uploaded file from Gemini
                genai_client = app.state.genai_client
                genai_client.files.delete(name=gemini_file_name)
                logger.info(f"Deleted Gemini file: {gemini_file_name}")
            except Exception as e:
                logger.warning(f"Could not delete Gemini file: {e}")


@app.post("/podcast/generate", response_model=PodcastGenerationResponse)
async def generate_podcast(request: PodcastGenerationRequest):
    """
    Generate a podcast episode from a research paper using Gemini AI.

    This endpoint:
    1. Fetches the paper PDF from Supabase
    2. Uploads PDF to Gemini
    3. Generates a NotebookLM-style podcast script with two hosts
    4. Uses Google Cloud TTS to generate multi-speaker audio
    5. Stores the audio in Supabase storage
    6. Returns the episode with audio URL

    The generation happens synchronously and may take 2-5 minutes.
    """
    # Check if TTS is available (should always be true if server started successfully)
    if not getattr(app.state, 'tts_available', False):
        raise HTTPException(
            status_code=503,
            detail="Podcast generation is not available - Google Cloud TTS not configured"
        )

    result = await _generate_podcast_from_paper(paper_id=request.paper_id)
    return PodcastGenerationResponse(**result)




@app.post("/podcast/episodes/{episode_id}/regenerate")
async def regenerate_podcast_audio(episode_id: str):
    """
    Regenerate audio from an existing script.

    This allows you to regenerate the audio without re-analyzing the paper
    or regenerating the script. Useful after editing the script.
    """
    try:
        logger.info(f"Regenerating audio for episode: {episode_id}")

        # Fetch episode
        episode_response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).single().execute()
        episode = episode_response.data

        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")

        script_text = episode.get("script")
        if not script_text:
            raise HTTPException(status_code=400, detail="Episode has no script to regenerate from")

        # Update status
        supabase.table("podcast_episodes").update({
            "generation_status": "processing"
        }).eq("id", episode_id).execute()

        # Generate audio using Gemini 2.5 Pro native TTS
        logger.info("Generating audio with Gemini 2.5 Pro native TTS...")
        genai_client = app.state.genai_client

        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=script_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker='Alex',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Kore',
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker='Sam',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Puck',
                                    )
                                )
                            ),
                        ]
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )

        # Extract audio data
        audio_data = None
        audio_mime_type = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    audio_data = part.inline_data.data
                    audio_mime_type = getattr(part.inline_data, 'mime_type', 'audio/wav')
                    break

        if not audio_data:
            raise HTTPException(status_code=500, detail="No audio data in Gemini response")

        logger.info(f"Audio generated, size: {len(audio_data)} bytes, mime_type: {audio_mime_type}")

        # Convert to MP3 if not already MP3
        if audio_mime_type == 'audio/mpeg' or audio_mime_type == 'audio/mp3':
            logger.info("Audio already in MP3 format")
            mp3_data = audio_data
        else:
            logger.info(f"Converting audio from {audio_mime_type} to MP3...")

            # Parse mime type to determine format and parameters
            if 'L16' in audio_mime_type or 'pcm' in audio_mime_type.lower():
                # Raw PCM format
                source_format = 'pcm'
                # Extract sample rate from mime type (e.g., "rate=24000")
                sample_rate = 24000  # Default
                if 'rate=' in audio_mime_type:
                    import re
                    rate_match = re.search(r'rate=(\d+)', audio_mime_type)
                    if rate_match:
                        sample_rate = int(rate_match.group(1))

                mp3_data = convert_audio_to_mp3(audio_data, source_format='pcm', sample_rate=sample_rate, channels=1)
            elif audio_mime_type == 'audio/wav' or audio_mime_type == 'audio/x-wav':
                mp3_data = convert_audio_to_mp3(audio_data, source_format='wav')
            else:
                # Unknown format, try as WAV
                logger.warning(f"Unknown audio mime type: {audio_mime_type}, trying as WAV")
                mp3_data = convert_audio_to_mp3(audio_data, source_format='wav')

            logger.info(f"Converted to MP3, size: {len(mp3_data)} bytes")

        # Upload to storage as MP3
        paper_id = episode["paper_id"]
        filename = f"{episode_id}.mp3"
        storage_path = f"{paper_id}/{filename}"

        supabase.storage.from_("episodes").upload(
            path=storage_path,
            file=mp3_data,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

        public_url = supabase.storage.from_("episodes").get_public_url(storage_path)

        # Update episode
        supabase.table("podcast_episodes").update({
            "storage_path": storage_path,
            "audio_url": public_url,
            "generation_status": "completed"
        }).eq("id", episode_id).execute()

        return {
            "success": True,
            "episode_id": episode_id,
            "audio_url": public_url,
            "message": "Audio regenerated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate audio: {e}", exc_info=True)
        supabase.table("podcast_episodes").update({
            "generation_status": "failed",
            "generation_error": str(e)
        }).eq("id", episode_id).execute()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast/episodes/{episode_id}/regenerate-from-paper")
async def regenerate_podcast_from_paper(episode_id: str):
    """
    Regenerate the entire podcast (script + audio) from the original paper.

    This is useful for:
    - Retrying failed episodes
    - Getting a fresh generation with updated models
    - Starting over from scratch
    """
    # Fetch episode to get paper_id
    episode_response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).single().execute()
    episode = episode_response.data

    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    paper_id = episode["paper_id"]

    # Call shared generation function with existing episode_id
    result = await _generate_podcast_from_paper(paper_id=paper_id, episode_id=episode_id)
    result["message"] = "Podcast regenerated from paper successfully!"
    return result


@app.get("/podcast/episodes")
async def list_podcast_episodes():
    """List all podcast episodes."""
    try:
        response = supabase.table("podcast_episodes").select("*").order("created_at", desc=True).execute()
        return {
            "success": True,
            "episodes": response.data
        }
    except Exception as e:
        logger.error(f"Failed to list episodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcast/episodes/{episode_id}")
async def get_podcast_episode(episode_id: str):
    """Get a single podcast episode."""
    try:
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).single().execute()
        return {
            "success": True,
            "episode": response.data
        }
    except Exception as e:
        logger.error(f"Failed to get episode: {e}")
        raise HTTPException(status_code=404, detail="Episode not found")


@app.put("/podcast/episodes/{episode_id}")
async def update_podcast_episode(episode_id: str, updates: dict):
    """
    Update a podcast episode.

    Allows editing the script, title, description, etc.
    After editing the script, use /regenerate to create new audio.
    """
    try:
        # Validate that we're not updating restricted fields
        restricted_fields = ["id", "paper_id", "created_at", "audio_url", "storage_path"]
        for field in restricted_fields:
            if field in updates:
                del updates[field]

        response = supabase.table("podcast_episodes").update(updates).eq("id", episode_id).execute()
        return {
            "success": True,
            "episode": response.data[0] if response.data else None
        }
    except Exception as e:
        logger.error(f"Failed to update episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/podcast/episodes/{episode_id}")
async def delete_podcast_episode(episode_id: str):
    """Delete a podcast episode."""
    try:
        # Delete from storage first
        episode_response = supabase.table("podcast_episodes").select("storage_path").eq("id", episode_id).single().execute()
        if episode_response.data and episode_response.data.get("storage_path"):
            try:
                supabase.storage.from_("episodes").remove([episode_response.data["storage_path"]])
            except Exception as e:
                logger.warning(f"Failed to delete audio file: {e}")

        # Delete from database
        supabase.table("podcast_episodes").delete().eq("id", episode_id).execute()
        return {
            "success": True,
            "message": "Episode deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcast/feed.xml")
async def get_podcast_feed():
    """
    Generate an RSS feed for the podcast.

    This endpoint:
    1. Fetches the podcast feed configuration
    2. Fetches all completed episodes
    3. Generates an RSS 2.0 feed with iTunes podcast extensions
    4. Returns the XML feed

    The feed can be submitted to podcast directories like Apple Podcasts, Spotify, etc.
    """
    try:
        logger.info("Generating podcast RSS feed")

        # Fetch feed configuration
        feed_config_response = supabase.table("podcast_feed_config").select("*").limit(1).execute()
        if feed_config_response.data:
            feed_config = feed_config_response.data[0]
        else:
            # Use default config
            feed_config = {
                "title": "Research Papers Podcast",
                "description": "AI-generated podcasts discussing the latest research papers",
                "author": "Papers Viewer AI",
                "language": "en-us",
                "category": "Science",
                "explicit": False
            }

        # Fetch all completed episodes
        episodes_response = supabase.table("podcast_episodes").select("*").eq(
            "generation_status", "completed"
        ).order("published_at", desc=True).execute()
        episodes = episodes_response.data

        # Build RSS feed
        from xml.etree.ElementTree import Element, SubElement, tostring
        from datetime import datetime as dt

        # RSS root
        rss = Element("rss", version="2.0")
        rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
        rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")

        channel = SubElement(rss, "channel")

        # Channel metadata
        SubElement(channel, "title").text = feed_config.get("title", "Research Papers Podcast")
        SubElement(channel, "description").text = feed_config.get("description", "AI-generated podcasts about research papers")
        SubElement(channel, "language").text = feed_config.get("language", "en-us")

        # iTunes specific tags
        SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text = feed_config.get("author", "Papers Viewer AI")
        SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "yes" if feed_config.get("explicit", False) else "no"

        if feed_config.get("image_url"):
            itunes_image = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
            itunes_image.set("href", feed_config["image_url"])

        category = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}category")
        category.set("text", feed_config.get("category", "Science"))

        # Add episodes
        for episode in episodes:
            item = SubElement(channel, "item")

            SubElement(item, "title").text = episode["title"]
            SubElement(item, "description").text = episode.get("description", "")

            # Enclosure (audio file)
            if episode.get("audio_url"):
                enclosure = SubElement(item, "enclosure")
                enclosure.set("url", episode["audio_url"])
                enclosure.set("type", "audio/mpeg")
                # Note: We should ideally get the file size, but for now we'll omit it
                # enclosure.set("length", str(file_size))

            # Publication date (RFC 2822 format)
            if episode.get("published_at"):
                pub_date = dt.fromisoformat(episode["published_at"].replace('Z', '+00:00'))
                SubElement(item, "pubDate").text = pub_date.strftime("%a, %d %b %Y %H:%M:%S %z")

            # GUID
            SubElement(item, "guid", isPermaLink="false").text = episode["id"]

            # iTunes specific
            if episode.get("duration_seconds"):
                SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration").text = str(episode["duration_seconds"])

        # Convert to string
        from xml.dom import minidom
        xml_str = tostring(rss, encoding="unicode")
        # Pretty print
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Return as XML response
        from fastapi.responses import Response
        return Response(content=pretty_xml, media_type="application/xml")

    except Exception as e:
        logger.error(f"Failed to generate podcast feed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Use multiple workers for concurrent request handling (especially during long podcast generation)
    # Note: When using workers > 1, the app will be loaded in each worker process
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=4)
