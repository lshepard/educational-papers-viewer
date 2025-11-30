"""
Supabase Storage Operations

This module handles all interactions with Supabase storage buckets,
including uploading images and audio files.
"""

import logging
from typing import Optional
from supabase import Client

logger = logging.getLogger(__name__)


def upload_image_to_storage(
    supabase: Client,
    image_bytes: bytes,
    paper_slug: str,
    filename: str,
    bucket: str = "papers"
) -> Optional[str]:
    """
    Upload an image to Supabase storage.

    Args:
        supabase: Supabase client instance
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


def upload_audio_to_storage(
    supabase: Client,
    audio_data: bytes,
    paper_id: str,
    episode_id: str,
    bucket: str = "episodes"
) -> Optional[str]:
    """
    Upload audio file (MP3) to Supabase storage.

    Args:
        supabase: Supabase client instance
        audio_data: MP3 audio data
        paper_id: Paper ID for organizing audio files
        episode_id: Episode ID for filename
        bucket: Supabase storage bucket name

    Returns:
        Storage path if successful, None otherwise
    """
    filename = f"{episode_id}.mp3"
    storage_path = f"{paper_id}/{filename}"

    try:
        supabase.storage.from_(bucket).upload(
            path=storage_path,
            file=audio_data,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

        logger.info(f"Uploaded audio to storage: {bucket}/{storage_path}")
        return storage_path

    except Exception as e:
        logger.error(f"Failed to upload audio to storage: {e}")
        return None


def get_public_url(
    supabase: Client,
    bucket: str,
    path: str
) -> str:
    """
    Get public URL for a storage object.

    Args:
        supabase: Supabase client instance
        bucket: Storage bucket name
        path: Object path within bucket

    Returns:
        Public URL for the object
    """
    return supabase.storage.from_(bucket).get_public_url(path)


def delete_from_storage(
    supabase: Client,
    bucket: str,
    path: str
) -> bool:
    """
    Delete a file from Supabase storage.

    Args:
        supabase: Supabase client instance
        bucket: Storage bucket name
        path: Object path to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        supabase.storage.from_(bucket).remove([path])
        logger.info(f"Deleted from storage: {bucket}/{path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete from storage: {e}")
        return False
