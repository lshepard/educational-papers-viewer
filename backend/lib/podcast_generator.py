"""
Podcast Generation and Audio Processing

This module handles podcast script generation, audio synthesis,
and metadata creation for research paper podcasts.
"""

import io
import json
import logging
import tempfile
from typing import Dict, Any, Optional
from pydub import AudioSegment
from google import genai
from google.genai import types
from supabase import Client

logger = logging.getLogger(__name__)


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


# Note: The main generation functions will be refactored from main.py
# generate_podcast_from_paper()
# generate_podcast_metadata()
# generate_podcast_script()
