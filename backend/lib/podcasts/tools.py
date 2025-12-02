"""
Research and search tools for podcast agent.

All tools that the podcast generation agent can use via Gemini function calling.
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, List
from google.genai import types

logger = logging.getLogger(__name__)


# ==================== Tool Definitions ====================

def get_tool_definitions(perplexity_api_key: str = None) -> List[types.FunctionDeclaration]:
    """
    Get list of tool definitions for Gemini function calling.

    Args:
        perplexity_api_key: Optional Perplexity API key (enables related work search)

    Returns:
        List of FunctionDeclaration objects
    """
    tools = []

    # Semantic Scholar search (always available)
    tools.append(
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

    # Perplexity search (optional)
    if perplexity_api_key:
        tools.append(
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

    return tools


# ==================== Tool Implementations ====================

async def search_papers(query: str, limit: int = 5) -> str:
    """
    Search Semantic Scholar for research papers.

    Args:
        query: Search query
        limit: Maximum results (1-10)

    Returns:
        Formatted string with paper results
    """
    try:
        limit = min(max(1, int(limit)), 10)  # Clamp to 1-10

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,authors,year,abstract,venue,citationCount"
                },
                timeout=30.0
            )

            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit")
                return "Rate limit exceeded. Try again in a moment."

            response.raise_for_status()
            data = response.json()

            papers = data.get("data", [])
            if not papers:
                return f"No papers found for query: {query}"

            # Format results
            results = []
            for idx, paper in enumerate(papers, 1):
                title = paper.get("title", "Untitled")
                authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:3]])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                year = paper.get("year", "Unknown")
                venue = paper.get("venue", "Unknown")
                citations = paper.get("citationCount", 0)
                abstract = paper.get("abstract", "No abstract available")[:300]

                results.append(
                    f"{idx}. {title}\n"
                    f"   Authors: {authors}\n"
                    f"   Year: {year} | Venue: {venue} | Citations: {citations}\n"
                    f"   Abstract: {abstract}..."
                )

            return "\n\n".join(results)

    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return f"Search failed: {str(e)}"


async def search_related_work(query: str, perplexity_api_key: str) -> str:
    """
    Search for related work and context using Perplexity AI.

    Args:
        query: Search query about the research topic
        perplexity_api_key: Perplexity API key

    Returns:
        Search results and context
    """
    try:
        from perplexipy import PerplexityClient

        client = PerplexityClient(key=perplexity_api_key)
        result = client.query(query)

        if result and isinstance(result, str) and len(result) > 0:
            logger.info(f"Perplexity search successful for: {query[:100]}...")
            return result
        else:
            logger.warning("Perplexity returned empty response")
            return "No results found."

    except ImportError:
        logger.error("perplexipy not installed")
        return "Perplexity search not available (perplexipy not installed)"
    except Exception as e:
        logger.error(f"Perplexity search failed: {e}")
        return f"Search failed: {str(e)}"


async def search_youtube(query: str, max_results: int = 5) -> str:
    """
    Search YouTube for videos related to the research topic.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        Formatted string with video results
    """
    # TODO: Implement if needed
    # This would require YouTube Data API key
    return "YouTube search not yet implemented"


# ==================== Tool Executor ====================

async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    perplexity_api_key: str = None
) -> str:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        perplexity_api_key: Optional Perplexity API key

    Returns:
        Tool execution result as string
    """
    logger.info(f"Executing tool: {tool_name}")

    if tool_name == "search_papers":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        return await search_papers(query, limit)

    elif tool_name == "search_related_work":
        if not perplexity_api_key:
            return "Perplexity API key not configured"
        query = arguments.get("query", "")
        return await search_related_work(query, perplexity_api_key)

    elif tool_name == "search_youtube":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        return await search_youtube(query, max_results)

    else:
        logger.warning(f"Unknown tool: {tool_name}")
        return f"Unknown tool: {tool_name}"


# Export for convenience
PODCAST_TOOLS = get_tool_definitions
