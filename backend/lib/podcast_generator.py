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


def generate_script_with_research_tools(
    genai_client: genai.Client,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    pdf_uri: Optional[str] = None,
    perplexity_api_key: Optional[str] = None
) -> str:
    """
    Generate a podcast script using Gemini with research tools enabled.

    This function handles the research tool setup and function calling loop,
    allowing both single-paper and multi-paper podcast generation to use
    the same research capabilities.

    Args:
        genai_client: Gemini AI client
        prompt: The script generation prompt
        model: Gemini model to use
        pdf_uri: Optional PDF URI for single-paper podcasts
        perplexity_api_key: Optional Perplexity API key

    Returns:
        Generated script text
    """
    # Define research tools for Gemini function calling
    function_declarations = []

    # Add Semantic Scholar search tool (always available)
    function_declarations.append(
        types.FunctionDeclaration(
            name="search_papers",
            description="Search Semantic Scholar for research papers by title, author, topic, or methodology. Returns paper titles, authors, years, venues, citation counts, and abstracts.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search query for papers, authors, or research topics"
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of results to return (1-10, default 5)"
                    )
                },
                required=["query"]
            )
        )
    )

    # Add Perplexity search tool if API key is available
    if perplexity_api_key:
        function_declarations.append(
            types.FunctionDeclaration(
                name="search_related_work",
                description="Search for related research, similar studies, background information, or broader context about the research topic. Use this to find examples, comparisons, real-world applications, or general background that might not be in academic databases.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The search query about the research topic, methods, or concepts"
                        )
                    },
                    required=["query"]
                )
            )
        )

    research_tools = types.Tool(function_declarations=function_declarations)

    # Log which tools are enabled
    tools_description = "Semantic Scholar search"
    if perplexity_api_key:
        tools_description += " + Perplexity research"
    logger.info(f"Generating script with research tools: {tools_description}")

    # Build initial messages
    messages = []
    if pdf_uri:
        messages.append(types.Part.from_uri(file_uri=pdf_uri, mime_type="application/pdf"))
    messages.append(prompt)

    config = types.GenerateContentConfig(
        tools=[research_tools]
    )

    # Manual function calling loop
    max_iterations = 5
    for iteration in range(max_iterations):
        logger.info(f"Script generation iteration {iteration + 1}/{max_iterations}")

        script_response = genai_client.models.generate_content(
            model=model,
            contents=messages,
            config=config
        )

        # Check if there are function calls in the response
        function_calls = []
        if script_response.candidates and script_response.candidates[0].content.parts:
            for part in script_response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)

        # If no function calls, we're done
        if not function_calls:
            logger.info("No function calls - script generation complete")
            break

        # Execute function calls
        logger.info(f"Executing {len(function_calls)} function call(s)")
        messages.append(script_response.candidates[0].content)

        for fn_call in function_calls:
            logger.info(f"Calling: {fn_call.name} with query: {fn_call.args.get('query', '')[:100]}...")

            # Execute the function
            if fn_call.name == "search_papers":
                limit = fn_call.args.get("limit", 5)
                result = search_semantic_scholar_papers(
                    query=fn_call.args["query"],
                    limit=min(max(1, int(limit)), 10)
                )
            elif fn_call.name == "search_related_work":
                result = search_related_work_perplexity(
                    query=fn_call.args["query"],
                    perplexity_api_key=perplexity_api_key
                )
            else:
                result = "Function not found"

            # Add function response
            messages.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_call.name,
                        response={"result": result}
                    )
                )
            )

    return script_response.text


def search_related_work_perplexity(query: str, perplexity_api_key: str) -> str:
    """
    Search for related work and context using Perplexity AI.

    This function can be called by the podcast generation agent to gather
    additional context about the research topic, find similar studies,
    or get background information.

    Args:
        query: The search query about the research topic
        perplexity_api_key: Perplexity API key

    Returns:
        Search results and context from Perplexity
    """
    try:
        from perplexipy import PerplexityClient

        # Note: PerplexityClient uses 'key' parameter, not 'api_key'
        client = PerplexityClient(key=perplexity_api_key)

        # Use Perplexity to search for related work
        # The query() method returns the response directly as a string
        result = client.query(query)

        if result and isinstance(result, str) and len(result) > 0:
            logger.info(f"Perplexity search successful for query: {query[:100]}...")
            return result
        else:
            logger.warning("Perplexity returned empty response")
            return "No results found"

    except Exception as e:
        logger.error(f"Perplexity search failed: {e}")
        return f"Search failed: {str(e)}"


def search_semantic_scholar_papers(query: str, limit: int = 5) -> str:
    """
    Search for research papers on Semantic Scholar.

    This function can be called by the podcast generation agent to find
    specific papers, authors, or research topics for comparison and context.

    Args:
        query: Search query for papers, authors, or topics
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON string with paper results including titles, authors, years, abstracts
    """
    try:
        import httpx
        import asyncio

        async def do_search():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": limit,
                        "fields": "paperId,title,authors,year,venue,citationCount,abstract,url"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])

                    # Format results in a readable way
                    results = []
                    for paper in papers:
                        result = {
                            "title": paper.get("title", ""),
                            "authors": ", ".join([a["name"] for a in paper.get("authors", [])]),
                            "year": paper.get("year"),
                            "venue": paper.get("venue", ""),
                            "citations": paper.get("citationCount", 0),
                            "abstract": paper.get("abstract", "")[:300] + "..." if paper.get("abstract") else ""
                        }
                        results.append(result)

                    logger.info(f"Semantic Scholar search successful: {len(results)} papers found for query: {query[:100]}...")
                    return json.dumps(results, indent=2)
                else:
                    logger.warning(f"Semantic Scholar search returned status {response.status_code}")
                    return "No results found"

        # Run async search
        return asyncio.run(do_search())

    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return f"Search failed: {str(e)}"


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
    episode_id: str = None,
    perplexity_api_key: Optional[str] = None
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

**EPISODE STRUCTURE:**

1. **Opening Hook (30-60 seconds)**
   - Start with a SNAPPY, engaging opening that hooks the listener
   - Don't just state the paper title - create interest and curiosity
   - Example: "What if I told you that a simple change in how we teach math could double student retention?"

2. **Context & Importance (60-90 seconds)**
   - Mention the paper title and authors naturally in conversation
   - Note the authors' affiliations and background (if available)
   - Explain WHY this paper matters - what problem does it solve? What gap does it fill?
   - Be HONEST about significance: Is it highly influential? Novel? Incremental improvement?
   - Draw from the background/introduction and results/discussion sections to explain importance
   - Set appropriate expectations - don't oversell, but do highlight genuine contributions

3. **The Study Details (90-120 seconds)**
   - If the paper describes a specific experiment or system, tell the FULL STORY:
     * How many participants/subjects? What were the conditions?
     * What exactly did they do? Walk through the procedure step-by-step
     * What was being measured? How?
     * What were the controls/comparisons?
   - Include 2-3 DIRECT QUOTES from the paper that capture key insights or findings
     * Introduce quotes naturally: "The authors write that..." or "As they put it..."
     * Choose quotes that are clear and impactful, not overly technical
   - Explain the methodology in accessible terms

4. **Key Findings & Implications (60-90 seconds)**
   - What did they discover? What were the main results?
   - Include at least one more DIRECT QUOTE about the findings or implications
   - What does this mean for the field? For practice? For future research?
   - Connect back to why this matters (from the opening)

**IMPORTANT GUIDELINES:**
- Use natural, conversational language - NO "um", "like", or filler words (TTS adds pauses)
- Target 3-5 minutes when spoken (roughly 450-750 words)
- Make complex topics accessible and engaging
- Be lighthearted and fun, but informative and HONEST
- Pull quotes DIRECTLY from the paper text - don't paraphrase when quoting
- If there's a specific experiment/system, dedicate real time to explaining how it worked

**AVAILABLE TOOLS:**
You have access to research tools to enrich the discussion:

1. `search_related_work` - Search for broader context and real-world applications:
   - Similar studies or related work in this field
   - Background information on methods or concepts
   - Examples of how this research builds on or differs from prior work
   - Real-world applications or implementations

2. `search_papers` - Search Semantic Scholar for specific papers:
   - Find papers by specific authors or topics
   - Locate papers with similar methodologies
   - Get citation counts and publication details
   - Compare with related research

Use these tools as needed to provide richer context and connections.

Format the script like this:
Alex: [speaks naturally]
Sam: [responds naturally]
Alex: [continues conversation]
...

Focus on making the content digestible, honest, and interesting for casual listeners."""

        # Generate script using shared function with research tools
        script_text = generate_script_with_research_tools(
            genai_client=genai_client,
            prompt=script_prompt,
            model="gemini-3-pro-preview",
            pdf_uri=gemini_file_uri,
            perplexity_api_key=perplexity_api_key
        )

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
1. "title": A SNAPPY, engaging headline (40-80 characters) that makes people want to listen.
   - Focus on the IMPACT, the surprising finding, or the practical benefit
   - Use active, punchy language - think podcast episode, not academic paper
   - Examples of good vs bad:
     ❌ Bad: "A Study on Math Education Interventions"
     ✅ Good: "This Simple Math Trick Doubled Student Performance"
     ❌ Bad: "Analysis of Teacher Feedback Methods"
     ✅ Good: "Why Your Teacher's Praise Might Be Backfiring"
   - Don't just restate the academic title - translate it into human interest

2. "description": A compelling description with:
   - First 1-2 sentences: A catchy hook that creates curiosity and makes people want to listen
   - Highlight what's surprising, useful, or important about this research
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
