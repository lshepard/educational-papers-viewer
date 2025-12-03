"""
Podcasts router - unified podcast generation using agent architecture.

Clean implementation using PodcastAgent with tool calling.
"""

import logging
import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import Response
from pydantic import BaseModel

from lib.podcasts import PodcastAgent
from lib.podcasts.audio import generate_audio_from_script, convert_to_mp3
from lib.podcasts.script import generate_metadata
from lib.core import GeminiFileManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/podcast", tags=["podcasts"])


# ==================== Request/Response Models ====================

class PodcastGenerationRequest(BaseModel):
    """
    Unified request for podcast generation.

    Accepts both old format (paper_id: string) and new format (paper_ids: string[])
    for backward compatibility.
    """
    paper_id: Optional[str] = None  # Legacy: single paper
    paper_ids: Optional[List[str]] = None  # New: 1 or many papers
    title: Optional[str] = None
    description: Optional[str] = None
    theme: Optional[str] = None
    episode_number: Optional[int] = None
    season_number: Optional[int] = None

    def get_paper_ids(self) -> List[str]:
        """Get paper IDs as a list, regardless of input format."""
        if self.paper_ids:
            return self.paper_ids
        elif self.paper_id:
            return [self.paper_id]
        else:
            raise ValueError("Either paper_id or paper_ids must be provided")


class CustomEpisodeRequest(BaseModel):
    """Request for generating custom multi-paper episode with theme."""
    theme: str
    papers: List[dict]  # Papers with metadata (for /podcast/generate-custom backward compat)


class PodcastGenerationResponse(BaseModel):
    success: bool
    episode_id: str
    audio_url: str
    message: str


class EpisodeListResponse(BaseModel):
    success: bool
    episodes: list


class EpisodeDetailResponse(BaseModel):
    success: bool
    episode: dict


# ==================== Dependencies ====================

def get_supabase():
    """Get Supabase client."""
    from main import supabase
    return supabase


def get_genai_client():
    """Get Gemini client."""
    from main import app
    return app.state.genai_client


def get_gemini_manager():
    """Get GeminiFileManager."""
    from main import app
    return app.state.gemini_manager


def get_perplexity_api_key():
    """Get Perplexity API key."""
    import os
    return os.getenv("PERPLEXITY_API_KEY")


# ==================== Endpoints ====================

@router.post("/generate", response_model=PodcastGenerationResponse)
async def generate_podcast(
    request: PodcastGenerationRequest,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client),
    gemini_manager = Depends(get_gemini_manager),
    perplexity_api_key = Depends(get_perplexity_api_key)
):
    """
    Generate podcast episode from one or more papers using agent architecture.

    This endpoint:
    1. Fetches paper(s) from database
    2. Uploads PDF to Gemini (for single paper)
    3. Uses PodcastAgent to generate script with research tools
    4. Generates audio using Gemini TTS
    5. Stores episode in database

    Works for both single-paper and multi-paper episodes.
    """
    episode_id = None

    try:
        # Get paper IDs (handles both old and new format)
        try:
            paper_ids = request.get_paper_ids()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"Generating podcast for {len(paper_ids)} paper(s)")

        # Fetch papers
        papers_response = supabase.table("papers").select("*").in_("id", paper_ids).execute()

        if not papers_response.data:
            raise HTTPException(status_code=404, detail="Papers not found")

        papers = papers_response.data
        logger.info(f"Found {len(papers)} papers")

        # Create episode record with temporary title if not provided
        # (will be updated with AI-generated title after script generation)
        if not request.title:
            if len(papers) == 1:
                temp_title = f"Episode: {papers[0].get('title', 'Untitled Paper')[:80]}"
            else:
                temp_title = f"Multi-Paper Episode: {len(papers)} Papers"
        else:
            temp_title = request.title

        episode_data = {
            "title": temp_title,
            "generation_status": "processing",
            "is_multi_paper": len(papers) > 1,
            "episode_number": request.episode_number,
            "season_number": request.season_number
        }

        if request.description:
            episode_data["description"] = request.description

        episode_response = supabase.table("podcast_episodes").insert(episode_data).execute()
        episode_id = episode_response.data[0]["id"]

        # Create junction table entries
        for paper in papers:
            supabase.table("episode_papers").insert({
                "episode_id": episode_id,
                "paper_id": paper["id"]
            }).execute()

        # Initialize podcast agent
        agent = PodcastAgent(
            genai_client=genai_client,
            perplexity_api_key=perplexity_api_key
        )

        # Generate script
        if len(papers) == 1:
            # Single paper - upload PDF and use optimized flow
            paper = papers[0]
            pdf_url = _get_storage_url(supabase, paper)

            logger.info("Uploading PDF to Gemini...")
            async with gemini_manager.upload_pdf_from_url(pdf_url) as gemini_file:
                script = await agent.generate_single_paper_script(
                    paper=paper,
                    pdf_uri=gemini_file.uri
                )
        else:
            # Multi-paper
            script = await agent.generate_multi_paper_script(
                papers=papers,
                theme=request.theme
            )

        logger.info(f"Script generated: {len(script)} characters")

        # Generate audio
        logger.info("Generating audio...")
        audio_data = generate_audio_from_script(script, genai_client)

        # Convert to MP3
        logger.info("Converting to MP3...")
        mp3_data = convert_to_mp3(audio_data)

        # Upload to storage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.mp3"

        from lib.storage import upload_audio_to_storage, get_public_url

        storage_path = upload_audio_to_storage(
            audio_data=mp3_data,
            filename=filename,
            bucket="episodes"
        )

        audio_url = get_public_url("episodes", storage_path)

        # Generate metadata if not provided
        if not request.title or not request.description:
            metadata = generate_metadata(
                script=script,
                papers=papers,
                genai_client=genai_client,
                theme=request.theme
            )

            if not request.title:
                request.title = metadata["title"]
            if not request.description:
                request.description = metadata["description"]

        # Update episode
        supabase.table("podcast_episodes").update({
            "title": request.title,
            "description": request.description,
            "script": script,
            "storage_path": storage_path,
            "audio_url": audio_url,
            "generation_status": "completed",
            "generation_error": None
        }).eq("id", episode_id).execute()

        logger.info(f"Podcast generation complete: {episode_id}")

        return PodcastGenerationResponse(
            success=True,
            episode_id=episode_id,
            audio_url=audio_url,
            message=f"Podcast generated successfully!"
        )

    except Exception as e:
        logger.error(f"Podcast generation failed: {e}", exc_info=True)

        if episode_id:
            supabase.table("podcast_episodes").update({
                "generation_status": "failed",
                "generation_error": str(e)
            }).eq("id", episode_id).execute()

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-custom", response_model=PodcastGenerationResponse)
async def generate_custom_episode(
    request: CustomEpisodeRequest,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client),
    gemini_manager = Depends(get_gemini_manager),
    perplexity_api_key = Depends(get_perplexity_api_key)
):
    """
    Generate custom podcast episode from multiple papers with a theme.

    **Backward compatibility endpoint** - prefer using POST /generate with paper_ids and theme.

    This endpoint wraps the unified /generate endpoint with legacy format support.
    """
    try:
        # Extract paper IDs from papers array
        paper_ids = [p.get("id") or p.get("paperId") for p in request.papers if p.get("id") or p.get("paperId")]

        if not paper_ids:
            raise HTTPException(status_code=400, detail="No valid paper IDs found in papers array")

        # Convert to unified request format
        unified_request = PodcastGenerationRequest(
            paper_ids=paper_ids,
            theme=request.theme
        )

        # Call unified generation endpoint
        return await generate_podcast(
            request=unified_request,
            supabase=supabase,
            genai_client=genai_client,
            gemini_manager=gemini_manager,
            perplexity_api_key=perplexity_api_key
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom episode generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{episode_id}/regenerate-audio")
async def regenerate_audio(
    episode_id: str,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client)
):
    """Regenerate audio from existing script."""
    try:
        # Fetch episode
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        episode = response.data[0]
        script = episode.get("script")

        if not script:
            raise HTTPException(status_code=400, detail="No script available")

        logger.info(f"Regenerating audio for episode: {episode_id}")

        # Generate audio
        audio_data = generate_audio_from_script(script, genai_client)
        mp3_data = convert_to_mp3(audio_data)

        # Upload
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.mp3"

        from lib.storage import upload_audio_to_storage, get_public_url

        storage_path = upload_audio_to_storage(
            audio_data=mp3_data,
            filename=filename,
            bucket="episodes"
        )

        audio_url = get_public_url("episodes", storage_path)

        # Update episode
        supabase.table("podcast_episodes").update({
            "audio_url": audio_url,
            "storage_path": storage_path,
            "generation_status": "completed",
            "generation_error": None
        }).eq("id", episode_id).execute()

        return {
            "success": True,
            "episode_id": episode_id,
            "audio_url": audio_url,
            "message": "Audio regenerated"
        }

    except Exception as e:
        logger.error(f"Audio regeneration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes(supabase = Depends(get_supabase)):
    """List all podcast episodes."""
    try:
        response = supabase.table("podcast_episodes")\
            .select("*")\
            .order("created_at", desc=True)\
            .execute()

        return EpisodeListResponse(success=True, episodes=response.data)

    except Exception as e:
        logger.error(f"Failed to list episodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{episode_id}", response_model=EpisodeDetailResponse)
async def get_episode(episode_id: str, supabase = Depends(get_supabase)):
    """Get specific episode."""
    try:
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        return EpisodeDetailResponse(success=True, episode=response.data[0])

    except Exception as e:
        logger.error(f"Failed to get episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/episodes/{episode_id}")
async def update_episode(
    episode_id: str,
    updates: dict = Body(...),
    supabase = Depends(get_supabase)
):
    """Update episode metadata."""
    try:
        response = supabase.table("podcast_episodes")\
            .update(updates)\
            .eq("id", episode_id)\
            .execute()

        return {"success": True, "episode": response.data[0]}

    except Exception as e:
        logger.error(f"Failed to update episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str, supabase = Depends(get_supabase)):
    """Delete episode."""
    try:
        # Fetch episode
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        episode = response.data[0]

        # Delete from storage
        if episode.get("storage_path"):
            from lib.storage import delete_from_storage
            try:
                delete_from_storage(
                    bucket=episode.get("storage_bucket", "episodes"),
                    path=episode["storage_path"]
                )
            except Exception as e:
                logger.warning(f"Failed to delete audio: {e}")

        # Delete from database
        supabase.table("podcast_episodes").delete().eq("id", episode_id).execute()

        return {"success": True, "message": "Episode deleted"}

    except Exception as e:
        logger.error(f"Failed to delete episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feed.xml")
async def get_feed(supabase = Depends(get_supabase)):
    """Get RSS feed."""
    try:
        from lib.rss_feed import generate_podcast_rss_feed

        rss_xml = await generate_podcast_rss_feed(supabase)

        return Response(
            content=rss_xml,
            media_type="application/xml",
            headers={"Content-Type": "application/xml; charset=utf-8"}
        )

    except Exception as e:
        logger.error(f"Failed to generate RSS feed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Helper Functions ====================

def _get_storage_url(supabase, paper: dict) -> str:
    """Get storage URL for paper PDF."""
    storage_bucket = paper.get("storage_bucket", "papers")
    storage_path = paper.get("storage_path")

    if not storage_path:
        raise ValueError("Paper has no storage_path")

    supabase_url = supabase.supabase_url
    return f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{storage_path}"
