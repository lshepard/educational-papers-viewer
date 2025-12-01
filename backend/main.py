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

# Import from lib modules
from lib.storage import upload_image_to_storage, upload_audio_to_storage, get_public_url, delete_from_storage
from lib.pdf_analyzer import (
    extract_paper_sections,
    extract_images_from_pdf,
    create_paper_slug,
    PaperSections,
    save_error_response
)
from lib.research import create_research_agent, populate_research_for_existing_papers
from lib.podcast_generator import convert_audio_to_mp3, generate_podcast_from_paper
from lib.rss_feed import format_duration, format_rfc2822_date, generate_podcast_rss_feed

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
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
SCRAPEGRAPHAI_API_KEY = os.getenv("SCRAPEGRAPHAI_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# CORS configuration
# Set ALLOW_ALL_ORIGINS=true to accept requests from any domain (useful for separate domain deployment)
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting Papers Viewer Backend...")
    logger.info("Gemini API configured")
    logger.info("Supabase client initialized")

    # Check Perplexity API configuration
    if PERPLEXITY_API_KEY:
        logger.info("✅ Perplexity API configured - research tool enabled for podcast generation")
    else:
        logger.warning("⚠️  Perplexity API not configured - podcast generation will work but without external research tool")

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
if ALLOW_ALL_ORIGINS:
    # Allow all origins - useful for separate domain deployment
    logger.info("CORS: Allowing all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Must be False when allowing all origins
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Allow specific origins only (more secure for production)
    allowed_origins = [
        "http://localhost:3000",
        FRONTEND_URL
    ]
    logger.info(f"CORS: Allowing specific origins: {allowed_origins}")
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
        genai_client = app.state.genai_client
        upload_response = genai_client.files.upload(file=temp_file_path)
        gemini_file = upload_response  # Keep the variable name for compatibility
        logger.info(f"File uploaded to Gemini: {upload_response.uri} (name: {upload_response.name})")

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
                genai_client = app.state.genai_client
                genai_client.files.delete(name=gemini_file.name)
                logger.debug(f"Deleted Gemini file: {gemini_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete Gemini file: {e}")


async def _generate_podcast_from_paper(paper_id: str, episode_id: str = None) -> dict:
    """
    Internal function to generate a podcast from a paper.

    This is a wrapper around the lib.podcast_generator.generate_podcast_from_paper function.

    Args:
        paper_id: ID of the paper to generate podcast for
        episode_id: Optional existing episode ID (for regeneration). If None, creates new episode.

    Returns:
        dict with success, episode_id, audio_url, and message
    """
    genai_client = app.state.genai_client
    return await generate_podcast_from_paper(
        paper_id,
        supabase,
        genai_client,
        episode_id,
        perplexity_api_key=PERPLEXITY_API_KEY
    )


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

    # Check if this is a multi-paper episode
    if episode.get("is_multi_paper"):
        raise HTTPException(
            status_code=400,
            detail="Cannot regenerate multi-paper episodes from paper. Use 'Regenerate Audio' to regenerate from the existing script."
        )

    paper_id = episode["paper_id"]

    if not paper_id:
        raise HTTPException(
            status_code=400,
            detail="Episode has no associated paper"
        )

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

        # Generate RSS feed using lib function
        website_url = feed_config.get("website_url", FRONTEND_URL)
        pretty_xml = generate_podcast_rss_feed(feed_config, episodes, website_url)

        # Return as XML response
        from fastapi.responses import Response
        return Response(content=pretty_xml, media_type="application/xml")

    except Exception as e:
        logger.error(f"Failed to generate podcast feed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/search")
async def search_papers(q: str):
    """
    Search papers in the database by theme/keywords.
    Searches across title, authors, application, venue, and why fields.
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
            "papers": response.data
        }

    except Exception as e:
        logger.error(f"Failed to search papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantic-scholar/search")
async def search_semantic_scholar(q: str, limit: int = 10):
    """
    Search Semantic Scholar for papers.
    Acts as a proxy to the Semantic Scholar API with retry logic for rate limiting.
    """
    try:
        import httpx
        import asyncio

        max_retries = 3
        base_delay = 1.0  # Start with 1 second delay

        # Search using Semantic Scholar API with retry logic
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": q,
                            "limit": limit,
                            "fields": "paperId,title,authors,year,venue,citationCount,abstract,url"
                        },
                        timeout=30.0
                    )

                    if response.status_code == 429:
                        # Rate limited - retry with exponential backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Semantic Scholar rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise HTTPException(
                                status_code=429,
                                detail="Semantic Scholar API rate limit exceeded. Please try again in a moment."
                            )

                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="Semantic Scholar API error")

                    data = response.json()

                    return {
                        "success": True,
                        "papers": data.get("data", [])
                    }

                except httpx.TimeoutException:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Semantic Scholar timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    raise HTTPException(status_code=504, detail="Semantic Scholar API timeout")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search Semantic Scholar: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CitationsRequest(BaseModel):
    paper_ids: List[str]


@app.post("/semantic-scholar/citations")
async def fetch_citations(request: CitationsRequest):
    """
    Fetch important citations from Semantic Scholar papers.
    Returns the most influential/highly cited references from the selected papers.
    """
    try:
        import httpx
        import asyncio

        all_citations = []
        seen_paper_ids = set()

        async with httpx.AsyncClient() as client:
            for idx, paper_id in enumerate(request.paper_ids):
                # Add small delay between requests to avoid rate limiting
                if idx > 0:
                    await asyncio.sleep(0.5)

                max_retries = 3
                base_delay = 1.0

                for attempt in range(max_retries):
                    try:
                        # Get paper details with references
                        response = await client.get(
                            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                            params={
                                "fields": "references,references.paperId,references.title,references.authors,references.year,references.citationCount,references.venue"
                            },
                            timeout=30.0
                        )

                        if response.status_code == 429:
                            # Rate limited - retry with exponential backoff
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                logger.warning(f"Rate limit hit for paper {paper_id}, retrying in {delay}s")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.warning(f"Rate limit exceeded for paper {paper_id}, skipping")
                                break

                        if response.status_code == 200:
                            data = response.json()
                            references = data.get("references", [])

                            # Filter for highly cited references (top 5 per paper)
                            sorted_refs = sorted(
                                references,
                                key=lambda r: r.get("citationCount", 0),
                                reverse=True
                            )[:5]

                            for ref in sorted_refs:
                                ref_paper_id = ref.get("paperId")
                                if ref_paper_id and ref_paper_id not in seen_paper_ids:
                                    seen_paper_ids.add(ref_paper_id)
                                    all_citations.append({
                                        "paperId": ref_paper_id,
                                        "title": ref.get("title", ""),
                                        "authors": ref.get("authors", []),
                                        "year": ref.get("year"),
                                        "venue": ref.get("venue"),
                                        "citationCount": ref.get("citationCount", 0),
                                        "abstract": None,
                                        "url": f"https://www.semanticscholar.org/paper/{ref_paper_id}"
                                    })
                        break  # Success - exit retry loop

                    except httpx.TimeoutException:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Timeout for paper {paper_id}, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        logger.warning(f"Timeout for paper {paper_id}, skipping")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to fetch citations for paper {paper_id}: {e}")
                        break

        # Sort by citation count and return top 10
        all_citations.sort(key=lambda p: p["citationCount"], reverse=True)

        return {
            "success": True,
            "citations": all_citations[:10]
        }

    except Exception as e:
        logger.error(f"Failed to fetch citations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CustomEpisodeRequest(BaseModel):
    theme: str
    papers: List[Dict[str, Any]]


class CustomEpisodeResponse(BaseModel):
    success: bool
    episode_id: str
    audio_url: str
    message: str


@app.post("/podcast/generate-custom", response_model=CustomEpisodeResponse)
async def generate_custom_episode(request: CustomEpisodeRequest):
    """
    Generate a custom podcast episode from multiple papers around a theme.
    Creates a 15-20 minute discussion about how the papers relate to the theme.
    """
    try:
        from lib.custom_podcast import generate_custom_themed_episode

        genai_client = app.state.genai_client

        result = await generate_custom_themed_episode(
            theme=request.theme,
            papers=request.papers,
            supabase=supabase,
            genai_client=genai_client,
            perplexity_api_key=PERPLEXITY_API_KEY
        )

        return CustomEpisodeResponse(
            success=True,
            episode_id=result["episode_id"],
            audio_url=result["audio_url"],
            message=result["message"]
        )

    except Exception as e:
        logger.error(f"Failed to generate custom episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PODCAST CREATOR ENDPOINTS
# Multi-step workflow for creating custom podcasts with research and clips
# ============================================================================

class CreateSessionRequest(BaseModel):
    theme: str
    description: Optional[str] = None
    resource_links: List[str] = []


class ResearchChatRequest(BaseModel):
    message: str


class YouTubeSearchRequest(BaseModel):
    query: str
    max_results: int = 10


class SaveClipRequest(BaseModel):
    video_id: str
    video_title: str
    channel_name: str
    start_time: float
    end_time: float
    clip_purpose: str
    quote_text: Optional[str] = None


@app.post("/podcast-creator/sessions")
async def create_podcast_creator_session(request: CreateSessionRequest):
    """Create a new podcast creator session."""
    try:
        session_data = {
            "theme": request.theme,
            "description": request.description,
            "resource_links": request.resource_links,
            "status": "research",
            "current_step": 1
        }

        response = supabase.table("podcast_creator_sessions").insert(session_data).execute()
        session = response.data[0]

        logger.info(f"Created podcast creator session: {session['id']}")

        return {
            "success": True,
            "session": session
        }

    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcast-creator/sessions/{session_id}")
async def get_podcast_creator_session(session_id: str):
    """Get podcast creator session with all related data."""
    try:
        # Get session
        session_response = supabase.table("podcast_creator_sessions").select("*").eq(
            "id", session_id
        ).single().execute()

        session = session_response.data

        # Get research messages
        messages_response = supabase.table("creator_research_messages").select("*").eq(
            "session_id", session_id
        ).order("message_order").execute()

        # Get clips
        clips_response = supabase.table("creator_youtube_clips").select("*").eq(
            "session_id", session_id
        ).order("play_order").execute()

        return {
            "success": True,
            "session": session,
            "messages": messages_response.data,
            "clips": clips_response.data
        }

    except Exception as e:
        logger.error(f"Failed to get session: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/podcast-creator/sessions/{session_id}/research")
async def research_chat(session_id: str, request: ResearchChatRequest):
    """Conduct research chat with Perplexity and bias analysis."""
    try:
        from lib.podcast_creator import research_chat_with_perplexity

        # Get session
        session_response = supabase.table("podcast_creator_sessions").select("*").eq(
            "id", session_id
        ).single().execute()

        session = session_response.data

        # Get conversation history
        messages_response = supabase.table("creator_research_messages").select(
            "role, content"
        ).eq("session_id", session_id).order("message_order").execute()

        conversation_history = messages_response.data

        # Call research function
        result = await research_chat_with_perplexity(
            session_id=session_id,
            user_message=request.message,
            conversation_history=conversation_history,
            theme=session["theme"],
            resource_links=session.get("resource_links", []),
            perplexity_api_key=PERPLEXITY_API_KEY,
            supabase=supabase
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:
        logger.error(f"Research chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast-creator/sessions/{session_id}/youtube/search")
async def search_youtube_videos(session_id: str, request: YouTubeSearchRequest):
    """Search YouTube for videos."""
    try:
        from lib.youtube_clips import search_youtube

        videos = await search_youtube(
            query=request.query,
            max_results=request.max_results
        )

        return {
            "success": True,
            "videos": videos
        }

    except Exception as e:
        logger.error(f"YouTube search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast-creator/sessions/{session_id}/clips")
async def save_clip(session_id: str, request: SaveClipRequest):
    """Save a YouTube clip selection and extract audio."""
    try:
        from lib.youtube_clips import save_clip_selection

        clip_id = await save_clip_selection(
            session_id=session_id,
            video_id=request.video_id,
            video_title=request.video_title,
            channel_name=request.channel_name,
            start_time=request.start_time,
            end_time=request.end_time,
            clip_purpose=request.clip_purpose,
            quote_text=request.quote_text,
            supabase=supabase
        )

        return {
            "success": True,
            "clip_id": clip_id,
            "message": "Clip saved and audio extraction started"
        }

    except Exception as e:
        logger.error(f"Failed to save clip: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/podcast-creator/sessions/{session_id}/show-notes/generate")
async def generate_show_notes_endpoint(session_id: str):
    """Generate show notes with quotes and clip markers."""
    try:
        from lib.podcast_creator import generate_show_notes

        # Get session
        session_response = supabase.table("podcast_creator_sessions").select("*").eq(
            "id", session_id
        ).single().execute()

        session = session_response.data

        # Get conversation history
        messages_response = supabase.table("creator_research_messages").select(
            "role, content"
        ).eq("session_id", session_id).order("message_order").execute()

        # Get clips
        clips_response = supabase.table("creator_youtube_clips").select("*").eq(
            "session_id", session_id
        ).order("play_order").execute()

        show_notes = await generate_show_notes(
            session_id=session_id,
            theme=session["theme"],
            conversation_history=messages_response.data,
            resource_links=session.get("resource_links", []),
            selected_clips=clips_response.data,
            genai_client=app.state.genai_client,
            supabase=supabase
        )

        return {
            "success": True,
            "show_notes": show_notes
        }

    except Exception as e:
        logger.error(f"Failed to generate show notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ProduceRequest(BaseModel):
    mode: str  # 'auto' or 'live'


@app.post("/podcast-creator/sessions/{session_id}/produce")
async def produce_podcast(session_id: str, request: ProduceRequest):
    """
    Produce the final podcast episode.

    Supports two modes:
    - auto: Fully automated generation with AI hosts and clips
    - live: Real-time conversation with user (WebSocket-based, not yet implemented)
    """
    try:
        if request.mode == "auto":
            from lib.podcast_production import produce_podcast_auto

            result = await produce_podcast_auto(
                session_id=session_id,
                supabase=supabase,
                genai_client=app.state.genai_client
            )

            return {
                "success": True,
                "episode": result
            }

        elif request.mode == "live":
            # TODO: Implement live recording mode with WebSocket
            raise HTTPException(
                status_code=501,
                detail="Live recording mode not yet implemented. Use 'auto' mode for now."
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be 'auto' or 'live'."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to produce podcast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ImportPaperRequest(BaseModel):
    url: str
    auto_extract: bool = False


@app.post("/papers/import")
async def import_paper(request: ImportPaperRequest):
    """
    Import a paper from URL (arXiv or direct PDF).

    Supports:
    - arXiv URLs: https://arxiv.org/abs/2510.12915
    - arXiv PDF URLs: https://arxiv.org/pdf/2510.12915.pdf
    - Direct PDF URLs

    Process:
    1. Downloads PDF
    2. Extracts metadata (from arXiv API if applicable)
    3. Uploads to Supabase storage
    4. Creates database record
    5. Optionally triggers content extraction
    """
    try:
        from lib.paper_import import import_paper_from_url

        result = await import_paper_from_url(
            url=request.url,
            supabase=supabase,
            auto_extract=request.auto_extract,
            scrapegraph_api_key=SCRAPEGRAPHAI_API_KEY
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:
        logger.error(f"Failed to import paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/batch-extract")
async def batch_extract_papers_endpoint(limit: Optional[int] = None, status: str = "pending"):
    """
    Batch extract content from multiple papers.

    This runs in the foreground and may take several minutes.
    Processes papers sequentially to avoid API rate limits.

    Query parameters:
    - limit: Maximum number of papers to process (default: all)
    - status: Filter by processing status (default: "pending")
    """
    try:
        from lib.batch_processing import batch_extract_papers

        genai_client = app.state.genai_client
        result = await batch_extract_papers(
            supabase=supabase,
            genai_client=genai_client,
            limit=limit,
            status_filter=status
        )

        return result

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/processing-stats")
async def get_processing_stats_endpoint():
    """
    Get statistics about paper processing status.

    Returns counts of papers by status (pending, processing, completed, failed).
    """
    try:
        from lib.batch_processing import get_processing_stats

        stats = await get_processing_stats(supabase)
        return stats

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class SearchTestRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/search/test")
async def test_search(request: SearchTestRequest):
    """
    Test full-text search and return detailed results.

    This endpoint helps verify that:
    1. Papers have been processed and have sections
    2. Full-text search index (fts) is working
    3. Search results are relevant

    Returns search results with paper titles and section info.
    """
    try:
        logger.info(f"Testing search for: {request.query}")

        # Perform search
        response = supabase.table("paper_sections") \
            .select("id, paper_id, section_type, section_title, content") \
            .limit(request.limit) \
            .text_search("fts", request.query) \
            .execute()

        results = response.data

        # Enrich with paper information
        enriched_results = []
        for section in results:
            # Get paper title
            paper_response = supabase.table("papers").select("title, authors").eq(
                "id", section["paper_id"]
            ).single().execute()

            paper = paper_response.data if paper_response.data else {}

            enriched_results.append({
                "section_id": section["id"],
                "paper_id": section["paper_id"],
                "paper_title": paper.get("title", "Unknown"),
                "paper_authors": paper.get("authors"),
                "section_type": section["section_type"],
                "section_title": section.get("section_title"),
                "content_preview": section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"]
            })

        return {
            "success": True,
            "query": request.query,
            "results_count": len(enriched_results),
            "results": enriched_results,
            "message": f"Found {len(enriched_results)} matching sections" if enriched_results else "No results found. Have papers been processed?"
        }

    except Exception as e:
        logger.error(f"Search test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/populate")
async def populate_research_metadata(limit: Optional[int] = None, force_refresh: bool = False):
    """
    Populate research metadata for existing papers in the database.

    This endpoint uses Semantic Scholar API to gather citation counts,
    influential references, and other metadata for papers.

    Args:
        limit: Optional limit on number of papers to process
        force_refresh: If True, refresh even if cache exists

    Returns:
        Summary of results including success/failure counts
    """
    try:
        genai_client = app.state.genai_client
        results = await populate_research_for_existing_papers(
            genai_client=genai_client,
            supabase_client=supabase,
            limit=limit,
            force_refresh=force_refresh
        )

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        logger.error(f"Failed to populate research metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Use multiple workers for concurrent request handling (especially during long podcast generation)
    # Note: When using workers > 1, the app will be loaded in each worker process
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=4)
