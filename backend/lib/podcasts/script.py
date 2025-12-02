"""
Script utilities for podcast generation.

Handles script formatting, metadata generation, and text processing.
"""

import json
import logging
from typing import Dict, Any, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def format_script_for_tts(script: str) -> str:
    """
    Format script for natural TTS output.

    Cleans up formatting markers and ensures natural speech flow.

    Args:
        script: Raw script text

    Returns:
        Formatted script for TTS
    """
    # Remove any formatting markers that might interfere with TTS
    formatted = script.strip()

    # Remove [HOST] markers (TTS doesn't need them)
    formatted = formatted.replace("[HOST]", "").replace("[HOST]:", "")

    # Ensure proper spacing
    formatted = " ".join(formatted.split())

    # Add pauses at section breaks (double newlines)
    formatted = formatted.replace("\n\n", " ... ")

    return formatted


def generate_metadata(
    script: str,
    papers: List[Dict[str, Any]],
    genai_client: genai.Client,
    theme: str = None
) -> Dict[str, str]:
    """
    Generate episode title and description from script and papers.

    Uses Gemini to create engaging metadata that captures the essence
    of the episode.

    Args:
        script: Generated podcast script
        papers: List of paper metadata dicts
        genai_client: Gemini client
        theme: Optional theme/angle

    Returns:
        Dict with 'title' and 'description'
    """
    logger.info("Generating podcast metadata...")

    try:
        # Build papers context
        papers_context = "\n\n".join([
            f"Paper {idx + 1}:\n"
            f"Title: {p.get('title', 'Untitled')}\n"
            f"Authors: {p.get('authors', 'Unknown')}\n"
            f"Year: {p.get('year', 'Unknown')}\n"
            f"URL: {p.get('source_url', 'N/A')}"
            for idx, p in enumerate(papers)
        ])

        theme_context = f"\n\nTheme: {theme}" if theme else ""

        # Create metadata generation prompt
        prompt = f"""Generate engaging podcast metadata based on this episode.

Papers discussed:
{papers_context}{theme_context}

Script excerpt:
{script[:1000]}...

Generate a JSON response with:

1. "title": A SNAPPY, engaging headline (40-80 characters) that makes people want to listen.
   - Focus on the IMPACT, the surprising finding, or the practical benefit
   - Use active, punchy language - think podcast episode, not academic paper
   - Examples of good titles:
     ✅ "This Simple Math Trick Doubled Student Performance"
     ✅ "Why Your Teacher's Praise Might Be Backfiring"
     ✅ "The Hidden Cost of Multitasking (Backed by Science)"
   - Don't just restate the academic title - translate it into human interest

2. "description": A compelling description with:
   - First 1-2 sentences: A catchy hook that creates curiosity
   - Highlight what's surprising, useful, or important
   - Include paper links and authors
   - Format like:

     📄 Read the papers:
     - [Paper 1 Title] by [Authors] ([Year]) - [URL]
     - [Paper 2 Title] by [Authors] ([Year]) - [URL]

     In this episode, we explore [key topics]. Perfect for [audience].

Make it engaging and podcast-friendly, not academic!"""

        response = genai_client.models.generate_content(
            model='gemini-exp-1206',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        metadata = json.loads(response.text)

        title = metadata.get("title", generate_fallback_title(papers))
        description = metadata.get("description", generate_fallback_description(papers))

        logger.info(f"Generated metadata - Title: {title}")

        return {
            "title": title,
            "description": description
        }

    except Exception as e:
        logger.warning(f"Metadata generation failed, using fallback: {e}")
        return {
            "title": generate_fallback_title(papers),
            "description": generate_fallback_description(papers)
        }


def generate_fallback_title(papers: List[Dict[str, Any]]) -> str:
    """Generate simple fallback title."""
    if len(papers) == 1:
        return f"Discussion: {papers[0].get('title', 'Research Paper')}"
    else:
        return f"Multi-Paper Discussion: {len(papers)} Papers"


def generate_fallback_description(papers: List[Dict[str, Any]]) -> str:
    """Generate simple fallback description."""
    description = "An AI-generated podcast episode discussing:\n\n"

    for idx, paper in enumerate(papers, 1):
        title = paper.get('title', 'Untitled Paper')
        authors = paper.get('authors', 'Unknown')
        year = paper.get('year', 'Unknown')
        url = paper.get('source_url', '')

        description += f"{idx}. {title}\n"
        description += f"   Authors: {authors} ({year})\n"
        if url:
            description += f"   📄 Read: {url}\n"
        description += "\n"

    return description


def extract_key_topics(script: str, max_topics: int = 5) -> List[str]:
    """
    Extract key topics/keywords from script.

    Simple extraction based on frequency and importance indicators.

    Args:
        script: Podcast script
        max_topics: Maximum topics to return

    Returns:
        List of key topics/phrases
    """
    # TODO: Implement if needed for tagging/categorization
    # Could use Gemini or simple NLP
    return []


def estimate_reading_time(script: str, words_per_minute: int = 150) -> int:
    """
    Estimate reading time in seconds.

    Args:
        script: Script text
        words_per_minute: Average speaking rate

    Returns:
        Estimated duration in seconds
    """
    word_count = len(script.split())
    minutes = word_count / words_per_minute
    return int(minutes * 60)


def validate_script(script: str) -> Dict[str, Any]:
    """
    Validate script quality and completeness.

    Args:
        script: Script to validate

    Returns:
        Dict with 'valid' (bool) and 'issues' (list of strings)
    """
    issues = []

    # Check minimum length
    word_count = len(script.split())
    if word_count < 500:
        issues.append(f"Script too short ({word_count} words, minimum 500)")

    # Check for common issues
    if not script.strip():
        issues.append("Script is empty")

    if "[TODO]" in script or "[FILL IN]" in script:
        issues.append("Script contains placeholder text")

    # Check for reasonable structure
    if "\n" not in script:
        issues.append("Script has no paragraph breaks")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "word_count": word_count,
        "estimated_duration_seconds": estimate_reading_time(script)
    }
