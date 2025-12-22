"""
Audio generation and processing for podcasts.

Handles TTS generation, audio format conversion, and basic mixing.
"""

import io
import logging
from typing import List, Optional
from pydub import AudioSegment
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def generate_audio_from_script(
    script: str,
    genai_client: genai.Client,
    speaker_names: Optional[List[str]] = None
) -> bytes:
    """
    Generate audio from podcast script using Gemini TTS.

    Args:
        script: Podcast script text (formatted with [HOST] markers)
        genai_client: Gemini client
        speaker_names: Optional list of speaker names for multi-voice

    Returns:
        Audio data as bytes (PCM format)
    """
    logger.info("Generating audio with Gemini 2.5 Pro TTS...")

    try:
        # Configure speech generation
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"  # Default voice
                )
            )
        )

        # If multiple speakers requested, configure them
        if speaker_names and len(speaker_names) > 1:
            # TODO: Add multi-speaker support if needed
            # For now, use single voice
            pass

        # Generate audio
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=script,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config
            )
        )

        # Extract audio data
        audio_data = b""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_data += part.inline_data.data

        if not audio_data:
            raise ValueError("No audio data generated")

        logger.info(f"Audio generated: {len(audio_data)} bytes")
        return audio_data

    except Exception as e:
        logger.error(f"Audio generation failed: {e}", exc_info=True)
        raise


def convert_to_mp3(audio_data: bytes, bitrate: str = "192k") -> bytes:
    """
    Convert audio data to MP3 format.

    Args:
        audio_data: Raw audio data (PCM format from Gemini)
        bitrate: MP3 bitrate (default: 192k)

    Returns:
        MP3 audio data as bytes
    """
    logger.info("Converting audio to MP3...")

    try:
        # Load audio from bytes (assuming PCM format)
        # Gemini returns PCM, 24000 Hz, mono, 16-bit
        audio = AudioSegment.from_raw(
            io.BytesIO(audio_data),
            sample_width=2,  # 16-bit = 2 bytes
            frame_rate=24000,
            channels=1
        )

        # Export as MP3
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
        mp3_data = mp3_buffer.getvalue()

        logger.info(f"MP3 conversion complete: {len(mp3_data)} bytes")
        return mp3_data

    except Exception as e:
        logger.error(f"MP3 conversion failed: {e}", exc_info=True)
        raise


def mix_audio_clips(clips: List[bytes], crossfade_ms: int = 1000) -> bytes:
    """
    Mix multiple audio clips with optional crossfade.

    Args:
        clips: List of audio data (MP3 format)
        crossfade_ms: Crossfade duration in milliseconds

    Returns:
        Mixed audio as MP3 bytes
    """
    logger.info(f"Mixing {len(clips)} audio clips...")

    try:
        if not clips:
            raise ValueError("No clips to mix")

        if len(clips) == 1:
            return clips[0]

        # Load all clips
        audio_segments = []
        for clip_data in clips:
            segment = AudioSegment.from_mp3(io.BytesIO(clip_data))
            audio_segments.append(segment)

        # Combine with crossfade
        result = audio_segments[0]
        for segment in audio_segments[1:]:
            result = result.append(segment, crossfade=crossfade_ms)

        # Export mixed audio
        output_buffer = io.BytesIO()
        result.export(output_buffer, format="mp3", bitrate="192k")
        mixed_data = output_buffer.getvalue()

        logger.info(f"Audio mixing complete: {len(mixed_data)} bytes")
        return mixed_data

    except Exception as e:
        logger.error(f"Audio mixing failed: {e}", exc_info=True)
        raise


def get_audio_duration(audio_data: bytes) -> float:
    """
    Get duration of audio in seconds.

    Args:
        audio_data: Audio data (MP3 format)

    Returns:
        Duration in seconds
    """
    try:
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        return len(audio) / 1000.0  # Convert ms to seconds
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        return 0.0


def normalize_audio(audio_data: bytes, target_dBFS: float = -20.0) -> bytes:
    """
    Normalize audio to target loudness.

    Args:
        audio_data: Audio data (MP3 format)
        target_dBFS: Target loudness in dBFS

    Returns:
        Normalized audio as MP3 bytes
    """
    try:
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))

        # Calculate change needed
        change_in_dBFS = target_dBFS - audio.dBFS

        # Apply normalization
        normalized = audio.apply_gain(change_in_dBFS)

        # Export
        output_buffer = io.BytesIO()
        normalized.export(output_buffer, format="mp3", bitrate="192k")

        return output_buffer.getvalue()

    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        return audio_data  # Return original on error
