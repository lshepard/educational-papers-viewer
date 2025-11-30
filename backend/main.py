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

    This is a wrapper around the lib.podcast_generator.generate_podcast_from_paper function.

    Args:
        paper_id: ID of the paper to generate podcast for
        episode_id: Optional existing episode ID (for regeneration). If None, creates new episode.

    Returns:
        dict with success, episode_id, audio_url, and message
    """
    genai_client = app.state.genai_client
    return await generate_podcast_from_paper(paper_id, supabase, genai_client, episode_id)


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

        # Generate RSS feed using lib function
        website_url = feed_config.get("website_url", FRONTEND_URL)
        pretty_xml = generate_podcast_rss_feed(feed_config, episodes, website_url)

        # Return as XML response
        from fastapi.responses import Response
        return Response(content=pretty_xml, media_type="application/xml")

    except Exception as e:
        logger.error(f"Failed to generate podcast feed: {e}", exc_info=True)
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
