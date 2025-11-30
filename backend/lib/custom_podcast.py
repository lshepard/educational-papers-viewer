"""
Custom Themed Podcast Generation

This module generates podcast episodes from multiple papers around a central theme.
Creates longer-form (15-20 minute) discussions about how papers relate to a theme.
"""

import logging
import tempfile
import os
import httpx
from typing import Dict, Any, List
from datetime import datetime
from google import genai
from google.genai import types
from supabase import Client

from .storage import upload_audio_to_storage, get_public_url
from .podcast_generator import generate_audio_from_script

logger = logging.getLogger(__name__)


async def fetch_paper_content(paper: Dict[str, Any], supabase: Client) -> Dict[str, Any]:
    """
    Fetch content for a paper from database or Semantic Scholar.

    Args:
        paper: Paper dict with 'source', 'id', and 'paperId' fields
        supabase: Supabase client

    Returns:
        Dict with paper details and content
    """
    if paper["source"] == "database":
        # Fetch from database
        response = supabase.table("papers").select("*").eq("id", paper["id"]).single().execute()
        paper_data = response.data

        # Extract summary from markdown if available (first few paragraphs)
        markdown = paper_data.get("markdown", "")
        summary = ""
        if markdown:
            # Get first 500 characters of markdown as summary
            summary = markdown[:500] + "..." if len(markdown) > 500 else markdown

        return {
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", ""),
            "year": paper_data.get("year"),
            "abstract": summary,
            "venue": paper_data.get("venue", ""),
            "application": paper_data.get("application", ""),
            "why": paper_data.get("why", ""),
            "study_design": paper_data.get("study_design", "")
        }

    elif paper["source"] == "semantic-scholar":
        # Fetch from Semantic Scholar
        paper_id = paper.get("paperId")
        if not paper_id:
            return {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year"),
                "abstract": "",
                "venue": "",
                "key_findings": "",
                "methodology": ""
            }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                    params={
                        "fields": "title,authors,year,venue,abstract"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "title": data.get("title", ""),
                        "authors": ", ".join([a["name"] for a in data.get("authors", [])]),
                        "year": data.get("year"),
                        "abstract": data.get("abstract", ""),
                        "venue": data.get("venue", ""),
                        "key_findings": "",
                        "methodology": ""
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch Semantic Scholar paper {paper_id}: {e}")

        # Fallback to paper data
        return {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year"),
            "abstract": "",
            "venue": "",
            "key_findings": "",
            "methodology": ""
        }

    return {}


def format_paper_for_prompt(paper_content: Dict[str, Any]) -> str:
    """Format paper content for the script generation prompt."""
    parts = []

    parts.append(f"**{paper_content['title']}**")

    if paper_content.get("authors"):
        parts.append(f"Authors: {paper_content['authors']}")

    if paper_content.get("year"):
        parts.append(f"Year: {paper_content['year']}")

    if paper_content.get("venue"):
        parts.append(f"Venue: {paper_content['venue']}")

    if paper_content.get("abstract"):
        parts.append(f"\nAbstract: {paper_content['abstract']}")

    if paper_content.get("application"):
        parts.append(f"\nApplication: {paper_content['application']}")

    if paper_content.get("why"):
        parts.append(f"\nPurpose: {paper_content['why']}")

    if paper_content.get("study_design"):
        parts.append(f"\nStudy Design: {paper_content['study_design']}")

    return "\n".join(parts)


async def generate_themed_script(
    theme: str,
    papers_content: List[Dict[str, Any]],
    genai_client: genai.Client
) -> str:
    """
    Generate a podcast script discussing multiple papers around a theme.

    Args:
        theme: The central theme/topic for discussion
        papers_content: List of paper content dicts
        genai_client: Gemini API client

    Returns:
        Generated script text
    """
    papers_text = "\n\n---\n\n".join([
        format_paper_for_prompt(p) for p in papers_content
    ])

    prompt = f"""You are creating a podcast script for an engaging discussion about research papers.

**Theme:** {theme}

You have {len(papers_content)} research papers to discuss. Your goal is to create a natural, engaging conversation between two hosts (Alex and Jordan).

**Episode Structure:**

1. **Opening & Theme Exposition (3-5 minutes)**
   - Start with an engaging introduction to the theme
   - Explain why this theme is important and relevant today
   - Provide historical context or current developments in this area
   - Give an overview of how the selected papers collectively address this theme
   - Preview the key questions and insights the episode will explore

2. **Paper Discussions (10-15 minutes)**
   - Present papers in order of their importance/influence to the theme
   - For each paper: explain its contribution, methodology, and key findings
   - Connect each paper back to the central theme
   - Highlight how papers build on, complement, or contrast with each other
   - Use transitions to show relationships between papers

3. **Synthesis & Conclusion (2-3 minutes)**
   - Tie all papers together, showing the bigger picture
   - Discuss implications for the field and real-world applications
   - Highlight remaining questions and future directions
   - End with key takeaways

**Important Guidelines:**
- Target 15-20 minutes of audio (approximately 2500-3500 words)
- Make it conversational and engaging, not academic
- Use natural dialogue with questions, reactions, and insights
- Explain technical concepts in accessible ways
- Show enthusiasm and curiosity about the research
- Order papers by importance/influence, not chronologically
- Create smooth transitions that highlight connections between papers

**Papers to Discuss (order by importance):**

{papers_text}

Generate a natural, engaging podcast script with dialogue between Alex and Jordan. Use this format:

Alex: [dialogue]
Jordan: [dialogue]

Start the script now:"""

    try:
        logger.info("Generating themed podcast script...")
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )

        script = response.text.strip()
        logger.info(f"Generated script: {len(script)} characters")

        return script

    except Exception as e:
        logger.error(f"Failed to generate script: {e}", exc_info=True)
        raise


async def generate_custom_themed_episode(
    theme: str,
    papers: List[Dict[str, Any]],
    supabase: Client,
    genai_client: genai.Client
) -> Dict[str, Any]:
    """
    Generate a custom podcast episode from multiple papers around a theme.

    Args:
        theme: The central theme/topic
        papers: List of paper dicts with 'source', 'id', 'title', 'authors', etc.
        supabase: Supabase client
        genai_client: Gemini API client

    Returns:
        Dict with episode_id, audio_url, and message
    """
    try:
        logger.info(f"Generating custom themed episode: {theme}")
        logger.info(f"Papers: {len(papers)}")

        # Create episode record for multi-paper episode
        episode_data = {
            "paper_id": None,  # No single paper for custom episodes
            "title": f"{theme}",
            "description": f"A discussion of {len(papers)} papers exploring {theme}",
            "generation_status": "processing",
            "is_multi_paper": True
        }

        episode_response = supabase.table("podcast_episodes").insert(episode_data).execute()
        episode_id = episode_response.data[0]["id"]
        logger.info(f"Created episode record: {episode_id}")

        # Insert papers into junction table
        for idx, paper in enumerate(papers):
            paper_link = {
                "episode_id": episode_id,
                "paper_title": paper.get("title", ""),
                "paper_authors": paper.get("authors", ""),
                "paper_year": paper.get("year"),
                "display_order": idx
            }

            # Add either paper_id or semantic_scholar_id
            if paper["source"] == "database":
                paper_link["paper_id"] = paper["id"]
            else:  # semantic-scholar
                paper_link["semantic_scholar_id"] = paper.get("paperId", "")

            supabase.table("episode_papers").insert(paper_link).execute()

        logger.info(f"Linked {len(papers)} papers to episode")

        # Fetch content for all papers
        logger.info("Fetching paper content...")
        papers_content = []
        for paper in papers:
            try:
                content = await fetch_paper_content(paper, supabase)
                if content:
                    papers_content.append(content)
            except Exception as e:
                logger.warning(f"Failed to fetch content for paper {paper.get('id')}: {e}")

        if not papers_content:
            raise ValueError("Could not fetch content for any papers")

        logger.info(f"Fetched content for {len(papers_content)} papers")

        # Generate script
        logger.info("Generating podcast script...")
        script = await generate_themed_script(theme, papers_content, genai_client)

        # Save script to temp file for debugging
        script_fd, script_path = tempfile.mkstemp(suffix=".txt", prefix=f"podcast_script_{episode_id}_")
        try:
            with os.fdopen(script_fd, 'w') as f:
                f.write(script)
            logger.info(f"📝 Script saved to: {script_path}")
        except Exception as e:
            logger.warning(f"Failed to save script to temp file: {e}")

        # Update episode with script
        supabase.table("podcast_episodes").update({
            "script": script
        }).eq("id", episode_id).execute()

        logger.info("Script generated, creating audio...")

        # Generate audio using shared function
        try:
            mp3_data = generate_audio_from_script(
                script_text=script,
                genai_client=genai_client,
                speaker_names=['Alex', 'Jordan']
            )

            # Upload to storage
            logger.info("Uploading audio to storage...")
            storage_path = f"custom-episodes/{episode_id}.mp3"

            upload_audio_to_storage(
                supabase=supabase,
                audio_data=mp3_data,
                storage_path=storage_path,
                bucket_name="podcast-audio"
            )

            # Get public URL
            audio_url = get_public_url(
                supabase=supabase,
                storage_path=storage_path,
                bucket_name="podcast-audio"
            )

            logger.info(f"Audio uploaded: {audio_url}")

            # Update episode record
            supabase.table("podcast_episodes").update({
                "audio_url": audio_url,
                "storage_path": storage_path,
                "generation_status": "completed"
            }).eq("id", episode_id).execute()

            logger.info(f"Custom episode generated successfully: {episode_id}")

            return {
                "episode_id": episode_id,
                "audio_url": audio_url,
                "message": f"Custom episode created: {theme}"
            }

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}", exc_info=True)

            # Update episode with error
            supabase.table("podcast_episodes").update({
                "generation_status": "failed",
                "generation_error": str(e)
            }).eq("id", episode_id).execute()

            raise

    except Exception as e:
        logger.error(f"Failed to generate custom episode: {e}", exc_info=True)

        # Update episode with error
        try:
            supabase.table("podcast_episodes").update({
                "generation_status": "failed",
                "generation_error": str(e)
            }).eq("id", episode_id).execute()
        except:
            pass

        raise
