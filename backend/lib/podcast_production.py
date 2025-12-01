"""
Podcast Production Module

Handles final podcast production for the creator workflow.
Supports both auto-generation and live recording modes.
"""

import logging
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from supabase import Client
from google import genai
from google.genai import types
from pydub import AudioSegment
import httpx

logger = logging.getLogger(__name__)


async def generate_podcast_script(
    show_notes: Dict[str, Any],
    clips: List[Dict[str, Any]],
    theme: str,
    genai_client: genai.Client
) -> str:
    """
    Generate a podcast script from show notes and clips.

    Creates a NotebookLM-style dialogue between two hosts (Alex and Sam)
    with markers for where clips should be inserted.

    Args:
        show_notes: Structured show notes with segments, quotes, etc.
        clips: List of YouTube clips with purposes and quotes
        theme: Podcast theme
        genai_client: Gemini client

    Returns:
        Formatted script with speaker labels and clip markers
    """
    try:
        # Build clip reference map
        clip_map = {clip["id"]: clip for clip in clips}

        # Create prompt for script generation
        prompt = f"""Generate a podcast script for a NotebookLM-style discussion about: {theme}

The script should be a natural conversation between two hosts:
- Alex (Speaker 1): Enthusiastic and curious, asks questions
- Sam (Speaker 2): Knowledgeable and explanatory, provides insights

Show Notes:
{json.dumps(show_notes, indent=2)}

Available Clips:
{json.dumps([{{
    "id": c["id"],
    "purpose": c["clip_purpose"],
    "quote": c.get("quote_text", ""),
    "video": c["video_title"]
}} for c in clips], indent=2)}

Instructions:
1. Create a 10-15 minute conversation (roughly 2000-3000 words)
2. Use the show notes as the outline
3. Reference quotes naturally in the dialogue
4. When appropriate, add clip markers like: [CLIP: clip_id]
5. Make it engaging and conversational, not academic
6. Both hosts should have distinct personalities
7. Include brief bias warnings where relevant (from show notes)

Format each line as:
Alex: [dialogue text]
Sam: [dialogue text]
[CLIP: clip_id]

Generate the complete script now:"""

        logger.info("Generating podcast script with Gemini...")

        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                max_output_tokens=8192
            )
        )

        script = response.text
        logger.info(f"Generated script: {len(script)} characters")

        return script

    except Exception as e:
        logger.error(f"Failed to generate script: {e}", exc_info=True)
        raise


async def download_clip_audio(clip_url: str, output_path: Path) -> bool:
    """
    Download a clip's audio file from Supabase storage.

    Args:
        clip_url: Public URL of the clip audio
        output_path: Where to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(clip_url, timeout=30.0)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded clip to {output_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to download clip: {e}")
        return False


async def mix_podcast_audio(
    script: str,
    clips: List[Dict[str, Any]],
    genai_client: genai.Client
) -> bytes:
    """
    Mix podcast audio from script and clips.

    Process:
    1. Parse script to separate dialogue and clip markers
    2. Generate AI audio for dialogue sections
    3. Download clip audio files
    4. Mix everything together in sequence

    Args:
        script: Script with dialogue and clip markers
        clips: List of clip metadata including audio URLs
        genai_client: Gemini client for TTS

    Returns:
        Final mixed audio as MP3 bytes
    """
    temp_dir = None
    try:
        # Create temp directory for audio files
        temp_dir = tempfile.mkdtemp(prefix="podcast_production_")
        temp_path = Path(temp_dir)

        # Parse script into segments
        segments = []
        current_dialogue = []

        for line in script.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('[CLIP:'):
                # Clip marker
                if current_dialogue:
                    segments.append({
                        'type': 'dialogue',
                        'content': '\n'.join(current_dialogue)
                    })
                    current_dialogue = []

                # Extract clip ID
                clip_id = line.replace('[CLIP:', '').replace(']', '').strip()
                segments.append({
                    'type': 'clip',
                    'clip_id': clip_id
                })
            else:
                # Dialogue line
                current_dialogue.append(line)

        # Add remaining dialogue
        if current_dialogue:
            segments.append({
                'type': 'dialogue',
                'content': '\n'.join(current_dialogue)
            })

        logger.info(f"Parsed script into {len(segments)} segments")

        # Build clip lookup
        clip_map = {clip["id"]: clip for clip in clips}

        # Process segments and generate audio
        audio_segments = []

        for idx, segment in enumerate(segments):
            if segment['type'] == 'dialogue':
                # Generate AI audio for this dialogue section
                logger.info(f"Generating audio for dialogue segment {idx + 1}...")

                response = genai_client.models.generate_content(
                    model="gemini-2.5-pro-preview-tts",
                    contents=segment['content'],
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
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            audio_data = part.inline_data.data
                            break

                if not audio_data:
                    logger.warning(f"No audio generated for segment {idx + 1}, skipping")
                    continue

                # Save to temp file and load with pydub
                dialogue_file = temp_path / f"dialogue_{idx}.wav"
                with open(dialogue_file, 'wb') as f:
                    f.write(audio_data)

                audio_seg = AudioSegment.from_file(str(dialogue_file))
                audio_segments.append(audio_seg)

            elif segment['type'] == 'clip':
                # Download and add clip audio
                clip_id = segment['clip_id']
                clip = clip_map.get(clip_id)

                if not clip:
                    logger.warning(f"Clip {clip_id} not found, skipping")
                    continue

                if not clip.get('audio_url'):
                    logger.warning(f"Clip {clip_id} has no audio URL, skipping")
                    continue

                logger.info(f"Adding clip: {clip['video_title']}")

                clip_file = temp_path / f"clip_{clip_id}.mp3"
                success = await download_clip_audio(clip['audio_url'], clip_file)

                if success:
                    clip_audio = AudioSegment.from_file(str(clip_file))
                    audio_segments.append(clip_audio)

        # Concatenate all segments
        logger.info(f"Mixing {len(audio_segments)} audio segments...")
        final_audio = sum(audio_segments)

        # Export as MP3
        output_file = temp_path / "final_podcast.mp3"
        final_audio.export(
            str(output_file),
            format="mp3",
            bitrate="128k",
            parameters=["-q:a", "2"]
        )

        # Read final file
        with open(output_file, 'rb') as f:
            mp3_data = f.read()

        logger.info(f"Final podcast: {len(mp3_data)} bytes, {len(final_audio)/1000:.1f}s")

        return mp3_data

    except Exception as e:
        logger.error(f"Failed to mix podcast audio: {e}", exc_info=True)
        raise

    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


async def produce_podcast_auto(
    session_id: str,
    supabase: Client,
    genai_client: genai.Client
) -> Dict[str, Any]:
    """
    Produce a podcast using auto-generation mode.

    Process:
    1. Fetch session data (show notes, clips)
    2. Generate script
    3. Generate and mix audio
    4. Upload to storage
    5. Create episode record

    Args:
        session_id: Podcast creator session ID
        supabase: Supabase client
        genai_client: Gemini client

    Returns:
        Episode data with audio_url
    """
    try:
        logger.info(f"Starting auto-generation for session {session_id}")

        # Fetch session
        session_response = supabase.table("podcast_creator_sessions").select("*").eq(
            "id", session_id
        ).single().execute()
        session = session_response.data

        if not session.get("show_notes"):
            raise ValueError("Session has no show notes. Generate show notes first.")

        # Fetch clips (only ready ones)
        clips_response = supabase.table("creator_youtube_clips").select("*").eq(
            "session_id", session_id
        ).eq("status", "ready").execute()

        clips = clips_response.data
        logger.info(f"Found {len(clips)} ready clips")

        # Update session status
        supabase.table("podcast_creator_sessions").update({
            "status": "production",
            "production_mode": "auto"
        }).eq("id", session_id).execute()

        # Generate script
        script = await generate_podcast_script(
            show_notes=session["show_notes"],
            clips=clips,
            theme=session["theme"],
            genai_client=genai_client
        )

        # Mix audio
        audio_data = await mix_podcast_audio(
            script=script,
            clips=clips,
            genai_client=genai_client
        )

        # Calculate duration
        from pydub import AudioSegment
        import io
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        duration_seconds = len(audio_seg) / 1000

        # Upload to storage
        storage_path = f"creator-episodes/{session_id}.mp3"
        supabase.storage.from_("episodes").upload(
            path=storage_path,
            file=audio_data,
            file_options={
                "content-type": "audio/mpeg",
                "upsert": "true"
            }
        )

        audio_url = supabase.storage.from_("episodes").get_public_url(storage_path)

        # Create episode record
        show_notes = session["show_notes"]
        episode_data = {
            "title": show_notes.get("title", f"Custom Podcast: {session['theme']}"),
            "description": f"A custom podcast about {session['theme']}",
            "script": script,
            "audio_url": audio_url,
            "storage_path": storage_path,
            "duration_seconds": int(duration_seconds),
            "generation_status": "completed",
            "is_multi_paper": True,  # Custom podcasts are multi-topic
            "metadata": {
                "creator_session_id": session_id,
                "production_mode": "auto",
                "theme": session["theme"],
                "clips_count": len(clips)
            }
        }

        episode_response = supabase.table("podcast_episodes").insert(episode_data).execute()
        episode = episode_response.data[0]

        # Update session
        supabase.table("podcast_creator_sessions").update({
            "status": "completed",
            "episode_id": episode["id"],
            "audio_url": audio_url
        }).eq("id", session_id).execute()

        logger.info(f"Auto-generation complete: episode {episode['id']}")

        return {
            "episode_id": episode["id"],
            "audio_url": audio_url,
            "duration_seconds": int(duration_seconds),
            "title": episode["title"],
            "message": "Podcast generated successfully"
        }

    except Exception as e:
        # Update session status to failed
        supabase.table("podcast_creator_sessions").update({
            "status": "failed",
            "error_message": str(e)
        }).eq("id", session_id).execute()

        logger.error(f"Auto-generation failed: {e}", exc_info=True)
        raise
