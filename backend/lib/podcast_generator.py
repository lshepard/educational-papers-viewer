"""
Podcast Generation and Audio Processing

This module handles podcast script generation, audio synthesis,
and metadata creation for research paper podcasts.
"""

import io
import json
import logging
import tempfile
import os
import re
from typing import Dict, Any, Optional, List
from pydub import AudioSegment
from google import genai
from google.genai import types
from supabase import Client
from fastapi import HTTPException

from lib.research import create_research_agent

logger = logging.getLogger(__name__)


def generate_audio_from_script(
    script_text: str,
    genai_client: genai.Client,
    speaker_names: List[str] = None
) -> bytes:
    """
    Generate audio from script text using Gemini TTS and convert to MP3.

    Args:
        script_text: The podcast script text
        genai_client: Gemini AI client
        speaker_names: Optional list of speaker names (e.g., ['Alex', 'Sam'])

    Returns:
        MP3-encoded audio bytes
    """
    try:
        logger.info("Generating audio with Gemini 2.5 Pro TTS...")

        # Configure multi-speaker if speaker names provided
        if speaker_names and len(speaker_names) >= 2:
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker=speaker_names[0],
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Kore'
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker=speaker_names[1],
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Puck'
                                    )
                                )
                            ),
                        ]
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        else:
            # Single voice config
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Puck"
                        )
                    )
                )
            )

        # Generate audio
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=script_text,
            config=config
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
            raise ValueError("No audio data in Gemini response")

        logger.info(f"Audio generated, size: {len(audio_data)} bytes, mime_type: {audio_mime_type}")

        # Convert to MP3 if not already MP3
        if audio_mime_type == 'audio/mpeg' or audio_mime_type == 'audio/mp3':
            logger.info("Audio already in MP3 format")
            return audio_data
        else:
            logger.info(f"Converting audio from {audio_mime_type} to MP3...")

            # Parse mime type to determine format and parameters
            if 'L16' in audio_mime_type or 'pcm' in audio_mime_type.lower():
                # Raw PCM format
                source_format = 'pcm'
                # Extract sample rate from mime type (e.g., "rate=24000")
                sample_rate = 24000  # Default
                if 'rate=' in audio_mime_type:
                    rate_match = re.search(r'rate=(\d+)', audio_mime_type)
                    if rate_match:
                        sample_rate = int(rate_match.group(1))
                        logger.info(f"Detected sample rate: {sample_rate} Hz")

                mp3_data = convert_audio_to_mp3(
                    audio_data,
                    source_format='pcm',
                    sample_rate=sample_rate
                )
            else:
                # Try WAV format
                mp3_data = convert_audio_to_mp3(audio_data, source_format='wav')

            logger.info("Audio converted to MP3")
            return mp3_data

    except Exception as e:
        logger.error(f"Failed to generate audio: {e}", exc_info=True)
        raise


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


async def generate_podcast_from_paper(
    paper_id: str,
    supabase: Client,
    genai_client: genai.Client,
    episode_id: str = None
) -> dict:
    """
    Generate a podcast from a research paper.

    Args:
        paper_id: ID of the paper to generate podcast for
        supabase: Supabase client
        genai_client: Gemini AI client
        episode_id: Optional existing episode ID (for regeneration). If None, creates new episode.

    Returns:
        dict with success, episode_id, audio_url, and message
    """
    temp_file_path = None
    gemini_file_name = None

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

        # Upload file
        upload_response = genai_client.files.upload(file=temp_file_path)
        gemini_file_name = upload_response.name
        gemini_file_uri = upload_response.uri
        logger.info(f"File uploaded to Gemini: {gemini_file_uri} (name: {gemini_file_name})")

        # Research paper context using Semantic Scholar and web search
        logger.info("Researching paper context...")
        research_agent = create_research_agent(genai_client, supabase)
        research_context = research_agent.research_paper_context(
            paper_id=paper_id,
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

IMPORTANT: Start the episode with a brief introduction that sets the context:
- Mention the paper title and authors
- Note the authors' affiliations (if available in research context)
- Be HONEST about the paper's significance: Is it highly influential (many citations)?
  A well-recognized contribution? Or an emerging/newer work?
- Briefly mention if the authors are established researchers with significant prior work,
  or if they're newer to the field (based on the author background in research context)
- Set expectations appropriately - don't oversell a minor paper, but do highlight genuine importance

The podcast should:
- START with this context-setting introduction about the paper and authors
- Incorporate the research context naturally into the discussion
- Reference the paper's place in the field and important prior work when relevant
- Be lighthearted and fun, but informative and HONEST
- Discuss key findings, methodology, and real-world implications
- Use natural, conversational language - NO "um", "like", or filler words (the TTS will add natural pauses)
- Be about 3-5 minutes when spoken (roughly 450-750 words)
- Make complex topics accessible and engaging

Format the script like this:
Alex: [speaks naturally]
Sam: [responds naturally]
Alex: [continues conversation]
...

Focus on making the content digestible, honest, and interesting for casual listeners."""

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

        # Generate audio using shared function with multi-speaker support
        mp3_data = generate_audio_from_script(
            script_text=script_text,
            genai_client=genai_client,
            speaker_names=['Alex', 'Sam']
        )

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

        # Generate engaging episode metadata using Gemini
        logger.info("Generating podcast metadata (title and description)...")
        try:
            metadata_prompt = f"""Generate an engaging podcast episode title and description for this research paper.

Paper Title: {paper.get('title', 'Untitled Paper')}
Authors: {paper.get('authors', 'Unknown')}
Year: {paper.get('year', 'Unknown')}
Source URL: {paper.get('source_url', '')}

Research Context:
{research_context['research_summary']}

Generate a JSON response with:
1. "title": A clickbait-style headline (40-80 characters) that captures what makes this paper interesting and worth listening to. Focus on the impact, novelty, or surprising findings. Don't just restate the academic title.

2. "description": A compelling description with:
   - First 1-2 sentences: A catchy hook that makes people want to listen
   - Then: Links and details in this format:

     📄 Read the paper: [source_url]
     👥 Authors: [author names]
     📅 Published: [year]

     In this episode, we dive into [key topics covered]. Perfect for [target audience].

Make it engaging and podcast-friendly, not academic!"""

            metadata_response = genai_client.models.generate_content(
                model='gemini-exp-1206',
                contents=metadata_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            metadata = json.loads(metadata_response.text)
            episode_title = metadata.get("title", f"Discussion: {paper.get('title', 'Untitled Paper')}")
            episode_description = metadata.get("description", f"An AI-generated podcast discussing the research paper by {paper.get('authors', 'Unknown')}")
            logger.info(f"Generated podcast metadata - Title: {episode_title}")
        except Exception as e:
            # Fallback to simple metadata if generation fails
            logger.warning(f"Failed to generate podcast metadata with Gemini, using fallback: {e}")
            episode_title = f"Discussion: {paper.get('title', 'Untitled Paper')}"
            episode_description = f"An AI-generated podcast discussing the research paper"
            if paper.get('authors'):
                episode_description += f" by {paper['authors']}"
            if paper.get('year'):
                episode_description += f" ({paper['year']})"
            if paper.get('source_url'):
                episode_description += f"\n\n📄 Read the paper: {paper['source_url']}"

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

        if gemini_file_name:
            try:
                # Delete uploaded file from Gemini
                genai_client.files.delete(name=gemini_file_name)
                logger.info(f"Deleted Gemini file: {gemini_file_name}")
            except Exception as e:
                logger.warning(f"Could not delete Gemini file: {e}")
