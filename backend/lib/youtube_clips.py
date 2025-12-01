"""
YouTube Clip Management

Search YouTube, extract clips, and manage clip audio for podcast production.
"""

import logging
import tempfile
import os
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from supabase import Client

logger = logging.getLogger(__name__)


async def search_youtube(
    query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search YouTube for videos.

    Uses yt-dlp's search functionality.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of video dicts with id, title, channel, duration, url
    """
    try:
        # Use yt-dlp to search
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--flat-playlist",
            f"ytsearch{max_results}:{query}"
        ]

        logger.info(f"Searching YouTube for: {query}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.error(f"YouTube search failed: {result.stderr}")
            return []

        # Parse results (one JSON object per line)
        videos = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                video_data = json.loads(line)
                videos.append({
                    "video_id": video_data.get("id"),
                    "title": video_data.get("title"),
                    "channel": video_data.get("channel") or video_data.get("uploader"),
                    "duration_seconds": video_data.get("duration"),
                    "url": f"https://www.youtube.com/watch?v={video_data.get('id')}",
                    "thumbnail": video_data.get("thumbnail"),
                    "description": video_data.get("description", "")[:200]  # First 200 chars
                })
            except json.JSONDecodeError:
                continue

        logger.info(f"Found {len(videos)} YouTube videos")
        return videos

    except subprocess.TimeoutExpired:
        logger.error("YouTube search timed out")
        return []
    except Exception as e:
        logger.error(f"YouTube search error: {e}", exc_info=True)
        return []


async def extract_clip_audio(
    video_id: str,
    start_time: float,
    end_time: float,
    clip_id: str,
    supabase: Client
) -> Optional[str]:
    """
    Extract audio clip from YouTube video and upload to Supabase storage.

    Args:
        video_id: YouTube video ID
        start_time: Start time in seconds
        end_time: End time in seconds
        clip_id: Database clip ID for storage path
        supabase: Supabase client

    Returns:
        Public URL of uploaded audio clip, or None if failed
    """
    temp_dir = None
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="podcast_clip_")
        temp_path = Path(temp_dir)

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        duration = end_time - start_time

        # Download clip using yt-dlp
        output_file = temp_path / f"{clip_id}.mp3"

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "128k",
            "--postprocessor-args", f"ffmpeg:-ss {start_time} -t {duration}",
            "-o", str(output_file),
            video_url
        ]

        logger.info(f"Extracting clip from {video_id} ({start_time}s - {end_time}s)")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes max
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return None

        # Check if file exists
        if not output_file.exists():
            logger.error(f"Output file not created: {output_file}")
            return None

        # Read file
        with open(output_file, 'rb') as f:
            audio_data = f.read()

        logger.info(f"Extracted {len(audio_data)} bytes of audio")

        # Upload to Supabase storage
        storage_path = f"podcast-clips/{clip_id}.mp3"

        supabase.storage.from_("episodes").upload(
            path=storage_path,
            file=audio_data,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

        # Get public URL
        public_url = supabase.storage.from_("episodes").get_public_url(storage_path)

        logger.info(f"Clip uploaded: {public_url}")

        # Update clip record
        supabase.table("creator_youtube_clips").update({
            "audio_file_path": storage_path,
            "audio_url": public_url,
            "status": "ready"
        }).eq("id", clip_id).execute()

        return public_url

    except subprocess.TimeoutExpired:
        logger.error("Clip extraction timed out")
        supabase.table("creator_youtube_clips").update({
            "status": "failed",
            "error_message": "Extraction timed out"
        }).eq("id", clip_id).execute()
        return None

    except Exception as e:
        logger.error(f"Clip extraction failed: {e}", exc_info=True)
        supabase.table("creator_youtube_clips").update({
            "status": "failed",
            "error_message": str(e)
        }).eq("id", clip_id).execute()
        return None

    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


async def save_clip_selection(
    session_id: str,
    video_id: str,
    video_title: str,
    channel_name: str,
    start_time: float,
    end_time: float,
    clip_purpose: str,
    quote_text: Optional[str],
    supabase: Client
) -> str:
    """
    Save a user's clip selection to the database.

    Args:
        session_id: Podcast creator session ID
        video_id: YouTube video ID
        video_title: Video title
        channel_name: Channel name
        start_time: Start time in seconds
        end_time: End time in seconds
        clip_purpose: What point this illustrates
        quote_text: The quote/content from the clip
        supabase: Supabase client

    Returns:
        Clip ID (UUID)
    """
    try:
        duration = end_time - start_time

        clip_data = {
            "session_id": session_id,
            "video_id": video_id,
            "video_title": video_title,
            "channel_name": channel_name,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "duration_seconds": duration,
            "clip_purpose": clip_purpose,
            "quote_text": quote_text,
            "status": "pending"
        }

        response = supabase.table("creator_youtube_clips").insert(clip_data).execute()
        clip_id = response.data[0]["id"]

        logger.info(f"Saved clip selection: {clip_id}")

        # Trigger async extraction (don't wait)
        # In production, you'd use a task queue like Celery
        # For now, we'll extract synchronously
        await extract_clip_audio(video_id, start_time, end_time, clip_id, supabase)

        return clip_id

    except Exception as e:
        logger.error(f"Failed to save clip selection: {e}", exc_info=True)
        raise
