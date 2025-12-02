"""
Podcasts router - unified podcast generation for single or multiple papers.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/podcast", tags=["podcasts"])


# ==================== Request/Response Models ====================

class PodcastGenerationRequest(BaseModel):
    """
    Unified request for podcast generation.
    Works for both single-paper and multi-paper episodes.
    """
    paper_ids: List[str]  # Can be 1 or many papers
    title: Optional[str] = None
    description: Optional[str] = None
    theme: Optional[str] = None  # Optional theme/angle for multi-paper episodes
    episode_number: Optional[int] = None
    season_number: Optional[int] = None


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
    """Dependency to get Supabase client."""
    from main import supabase
    return supabase


def get_genai_client():
    """Dependency to get Gemini client."""
    from main import app
    return app.state.genai_client


def get_perplexity_api_key():
    """Dependency to get Perplexity API key."""
    import os
    return os.getenv("PERPLEXITY_API_KEY")


# ==================== Endpoints ====================

@router.post("/generate", response_model=PodcastGenerationResponse)
async def generate_podcast(
    request: PodcastGenerationRequest,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client),
    perplexity_api_key = Depends(get_perplexity_api_key)
):
    """
    Generate a podcast episode from one or more research papers.

    This unified endpoint handles:
    - Single paper episodes (pass 1 paper_id)
    - Multi-paper episodes (pass multiple paper_ids)
    - Custom themed episodes (add a theme)

    The generation process:
    1. Fetches paper(s) from database
    2. Extracts content and generates script using Gemini
    3. Generates audio using Gemini TTS
    4. Stores episode in database
    """
    try:
        if not request.paper_ids:
            raise HTTPException(status_code=400, detail="At least one paper_id is required")

        logger.info(f"Generating podcast for {len(request.paper_ids)} paper(s)")

        # Import the unified generation function
        from lib.podcast_generator import generate_podcast_from_papers

        # Call unified generation function
        result = await generate_podcast_from_papers(
            paper_ids=request.paper_ids,
            supabase=supabase,
            genai_client=genai_client,
            perplexity_api_key=perplexity_api_key,
            title=request.title,
            description=request.description,
            theme=request.theme,
            episode_number=request.episode_number,
            season_number=request.season_number
        )

        return PodcastGenerationResponse(
            success=True,
            episode_id=result["episode_id"],
            audio_url=result["audio_url"],
            message=result["message"]
        )

    except Exception as e:
        logger.error(f"Podcast generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{episode_id}/regenerate-audio")
async def regenerate_podcast_audio(
    episode_id: str,
    supabase = Depends(get_supabase),
    genai_client = Depends(get_genai_client)
):
    """Regenerate audio for an existing episode using its stored script."""
    try:
        # Fetch episode
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        episode = response.data[0]
        script = episode.get("script")

        if not script:
            raise HTTPException(status_code=400, detail="Episode has no script to regenerate from")

        logger.info(f"Regenerating audio for episode: {episode_id}")

        # Import audio generation
        from lib.podcast_generator import generate_audio_from_script, convert_audio_to_mp3, upload_audio_to_storage

        # Generate audio
        audio_data = generate_audio_from_script(script, genai_client)

        # Convert to MP3
        mp3_data = convert_audio_to_mp3(audio_data)

        # Upload to storage
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.mp3"

        from lib.storage import upload_audio_to_storage as upload_audio
        storage_path = upload_audio(
            audio_data=mp3_data,
            filename=filename,
            bucket="episodes"
        )

        # Get public URL
        from lib.storage import get_public_url
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
            "message": "Audio regenerated successfully"
        }

    except Exception as e:
        logger.error(f"Audio regeneration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes", response_model=EpisodeListResponse)
async def list_podcast_episodes(supabase = Depends(get_supabase)):
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
async def get_podcast_episode(episode_id: str, supabase = Depends(get_supabase)):
    """Get a specific podcast episode."""
    try:
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        return EpisodeDetailResponse(success=True, episode=response.data[0])

    except Exception as e:
        logger.error(f"Failed to get episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/episodes/{episode_id}")
async def update_podcast_episode(
    episode_id: str,
    updates: dict = Body(...),
    supabase = Depends(get_supabase)
):
    """Update podcast episode metadata."""
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
async def delete_podcast_episode(episode_id: str, supabase = Depends(get_supabase)):
    """Delete a podcast episode."""
    try:
        # Fetch episode to get storage path
        response = supabase.table("podcast_episodes").select("*").eq("id", episode_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Episode not found")

        episode = response.data[0]

        # Delete from storage if exists
        if episode.get("storage_path"):
            from lib.storage import delete_from_storage
            try:
                delete_from_storage(
                    bucket=episode.get("storage_bucket", "episodes"),
                    path=episode["storage_path"]
                )
            except Exception as e:
                logger.warning(f"Failed to delete audio file: {e}")

        # Delete from database
        supabase.table("podcast_episodes").delete().eq("id", episode_id).execute()

        return {"success": True, "message": "Episode deleted"}

    except Exception as e:
        logger.error(f"Failed to delete episode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feed.xml")
async def get_podcast_feed(supabase = Depends(get_supabase)):
    """Get RSS feed for podcast episodes."""
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
