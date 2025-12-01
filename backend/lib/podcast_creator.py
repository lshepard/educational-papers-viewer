"""
Podcast Creator Module

Handles the multi-step workflow for creating custom podcasts:
1. Research chat with bias analysis
2. YouTube clip search and extraction
3. Show notes generation with quotes and clip markers
4. Auto-generation or live recording production
"""

import logging
import json
from typing import Dict, Any, List, Optional
from google import genai
from supabase import Client

logger = logging.getLogger(__name__)


def analyze_source_bias(source_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a source for potential bias.

    Checks for:
    - Publisher conflicts of interest (e.g., Google paper about Gemini)
    - Known bias patterns
    - Credibility indicators

    Args:
        source_info: Dict with keys like 'url', 'title', 'publisher', 'authors'

    Returns:
        Dict with 'bias_score' (0-10), 'bias_type', 'notes'
    """
    bias_analysis = {
        "bias_score": 5,  # Neutral by default
        "bias_type": "neutral",
        "notes": [],
        "credibility": "medium"
    }

    url = source_info.get("url", "").lower()
    title = source_info.get("title", "").lower()
    publisher = source_info.get("publisher", "").lower()

    # Check for corporate self-promotion
    if "google" in publisher and ("gemini" in title or "google" in title):
        bias_analysis["bias_score"] = 8
        bias_analysis["bias_type"] = "corporate_self_promotion"
        bias_analysis["notes"].append(
            "Publisher (Google) writing about their own product (Gemini) - expect promotional bias"
        )
        bias_analysis["credibility"] = "low_for_objectivity"

    elif "openai" in publisher and ("gpt" in title or "openai" in title):
        bias_analysis["bias_score"] = 8
        bias_analysis["bias_type"] = "corporate_self_promotion"
        bias_analysis["notes"].append(
            "Publisher (OpenAI) writing about their own product - expect promotional bias"
        )
        bias_analysis["credibility"] = "low_for_objectivity"

    elif "microsoft" in publisher and ("copilot" in title or "azure" in title):
        bias_analysis["bias_score"] = 7
        bias_analysis["bias_type"] = "corporate_self_promotion"
        bias_analysis["notes"].append(
            "Microsoft writing about their own AI products - likely promotional"
        )

    # Check for academic sources (generally more credible)
    elif any(domain in url for domain in ["arxiv.org", "acm.org", "ieee.org", ".edu"]):
        bias_analysis["bias_score"] = 3
        bias_analysis["bias_type"] = "academic"
        bias_analysis["credibility"] = "high"
        bias_analysis["notes"].append("Academic/research source - generally objective")

    # Check for news/media bias
    elif any(domain in url for domain in ["techcrunch", "verge", "wired"]):
        bias_analysis["bias_score"] = 5
        bias_analysis["bias_type"] = "tech_media"
        bias_analysis["credibility"] = "medium"
        bias_analysis["notes"].append("Tech media - may have editorial slant")

    return bias_analysis


async def research_chat_with_perplexity(
    session_id: str,
    user_message: str,
    conversation_history: List[Dict[str, str]],
    theme: str,
    resource_links: List[str],
    perplexity_api_key: str,
    supabase: Client
) -> Dict[str, Any]:
    """
    Conduct research chat using Perplexity, including bias analysis.

    Args:
        session_id: Podcast creator session ID
        user_message: User's research question/message
        conversation_history: Previous messages in format [{"role": "user|assistant", "content": "..."}]
        theme: The podcast theme
        resource_links: User-provided resource URLs
        perplexity_api_key: Perplexity API key
        supabase: Supabase client

    Returns:
        Dict with 'response', 'sources', 'bias_analysis'
    """
    try:
        from perplexipy import PerplexityClient

        # Build research context
        context_prompt = f"""You are a research assistant helping create a podcast about: {theme}

The user has provided these resources to consider:
{chr(10).join(f"- {link}" for link in resource_links) if resource_links else "No specific resources provided yet"}

When researching:
1. Find relevant sources and information
2. Note the publisher/author of each source
3. Be critical - identify potential bias (e.g., Google writing about Gemini)
4. Provide balanced perspectives

User's research question: {user_message}"""

        # Call Perplexity
        client = PerplexityClient(key=perplexity_api_key)
        response_text = client.query(context_prompt)

        # Extract sources from response (Perplexity typically includes citations)
        # For now, we'll parse sources from the response text
        # In production, you might want to use Perplexity's citation API if available
        sources = []

        # Simple source extraction - look for URLs in response
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response_text)

        for url in urls[:5]:  # Limit to 5 sources
            source_info = {
                "url": url,
                "title": "",  # Would need to fetch title
                "publisher": url.split('/')[2] if len(url.split('/')) > 2 else "unknown"
            }

            # Analyze for bias
            bias = analyze_source_bias(source_info)

            sources.append({
                **source_info,
                "bias_analysis": bias
            })

        # Get next message order
        existing_messages = supabase.table("creator_research_messages").select("message_order").eq(
            "session_id", session_id
        ).execute()

        next_order = max([m["message_order"] for m in existing_messages.data], default=-1) + 1

        # Save user message
        supabase.table("creator_research_messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": user_message,
            "message_order": next_order,
            "sources": None,
            "bias_analysis": None
        }).execute()

        # Save assistant response with sources
        supabase.table("creator_research_messages").insert({
            "session_id": session_id,
            "role": "assistant",
            "content": response_text,
            "message_order": next_order + 1,
            "sources": json.dumps(sources),
            "bias_analysis": json.dumps([s["bias_analysis"] for s in sources])
        }).execute()

        return {
            "response": response_text,
            "sources": sources,
            "message_order": next_order + 1
        }

    except Exception as e:
        logger.error(f"Research chat failed: {e}", exc_info=True)
        raise


async def generate_show_notes(
    session_id: str,
    theme: str,
    conversation_history: List[Dict[str, str]],
    resource_links: List[str],
    selected_clips: List[Dict[str, Any]],
    genai_client: genai.Client,
    supabase: Client
) -> Dict[str, Any]:
    """
    Generate structured show notes with quotes and clip markers.

    Args:
        session_id: Podcast creator session ID
        theme: Podcast theme
        conversation_history: Research chat history
        resource_links: Resource URLs
        selected_clips: YouTube clips selected
        genai_client: Gemini client
        supabase: Supabase client

    Returns:
        Dict with structured show notes
    """
    try:
        # Build context from research
        research_summary = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history
        ])

        clips_info = "\n".join([
            f"- {clip['video_title']} ({clip['duration_seconds']}s) - {clip['clip_purpose']}"
            for clip in selected_clips
        ]) if selected_clips else "No clips selected"

        prompt = f"""Generate detailed podcast show notes for a {theme} episode.

Research Context:
{research_summary}

Available Clips:
{clips_info}

Resource Links:
{chr(10).join(f"- {link}" for link in resource_links)}

Create a structured outline in JSON format with:
1. Opening hook (30-60 seconds)
2. Main segments (3-5 segments, each 2-4 minutes)
3. Conclusion and key takeaways

For each segment, include:
- Title and main point
- Direct quotes from sources (with attribution)
- Clip markers where clips should play
- Talking points for hosts
- Bias notes (e.g., "Note: This Google paper about Gemini - expect promotional tone")

Output JSON structure:
{{
  "title": "Episode title",
  "estimated_duration_minutes": 15,
  "segments": [
    {{
      "title": "Segment title",
      "duration_minutes": 3,
      "talking_points": ["point1", "point2"],
      "quotes": [
        {{
          "text": "Direct quote",
          "source": "Source name",
          "source_url": "URL",
          "bias_note": "Optional bias warning"
        }}
      ],
      "clips": [
        {{
          "clip_id": "clip UUID or marker",
          "play_at": "description of when to play",
          "purpose": "Why this clip"
        }}
      ]
    }}
  ]
}}"""

        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        show_notes = json.loads(response.text)

        # Save to database
        supabase.table("podcast_creator_sessions").update({
            "show_notes": show_notes,
            "status": "production",
            "current_step": 4
        }).eq("id", session_id).execute()

        logger.info(f"Generated show notes for session {session_id}")

        return show_notes

    except Exception as e:
        logger.error(f"Failed to generate show notes: {e}", exc_info=True)
        raise
