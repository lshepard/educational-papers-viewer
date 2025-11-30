"""
Research Agent for Podcast Script Generation

This module provides research capabilities for gathering context about papers
before generating podcast scripts. It uses Semantic Scholar API, web search,
and Gemini as an orchestrating agent with function calling.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from semanticscholar import SemanticScholar
from google import genai
from google.genai import types
from supabase import Client

logger = logging.getLogger(__name__)


class PaperResearchAgent:
    """Agent that researches papers to gather context for podcast generation."""

    def __init__(self, genai_client: genai.Client, supabase_client: Optional[Client] = None):
        """
        Initialize the research agent.

        Args:
            genai_client: Configured Google GenAI client
            supabase_client: Optional Supabase client for caching research results
        """
        self.genai_client = genai_client
        self.supabase_client = supabase_client
        self._semantic_scholar = None  # Lazy initialization to avoid uvloop conflict
        self.cache_ttl_days = 7  # Cache research results for 7 days

    @property
    def semantic_scholar(self) -> SemanticScholar:
        """Lazy initialization of SemanticScholar to avoid uvloop conflicts."""
        if self._semantic_scholar is None:
            # Prevent nest_asyncio from patching the loop when using uvloop
            import sys
            import nest_asyncio

            # Temporarily disable nest_asyncio.apply() to prevent uvloop conflict
            original_apply = nest_asyncio.apply
            nest_asyncio.apply = lambda loop=None: None

            try:
                self._semantic_scholar = SemanticScholar()
            finally:
                # Restore original function
                nest_asyncio.apply = original_apply

        return self._semantic_scholar

    def search_paper_on_semantic_scholar(self, title: str, authors: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a paper on Semantic Scholar by title.

        Args:
            title: Paper title
            authors: Optional authors to help narrow search

        Returns:
            Dictionary with paper metadata including citation count
        """
        try:
            logger.info(f"Searching Semantic Scholar for: {title}")

            # Search for the paper
            query = title
            if authors:
                query = f"{title} {authors}"

            results = self.semantic_scholar.search_paper(
                query=query,
                limit=1,
                fields=['paperId', 'title', 'abstract', 'year', 'citationCount',
                        'influentialCitationCount', 'authors', 'authors.name', 'authors.authorId',
                        'authors.affiliations', 'authors.paperCount', 'authors.citationCount',
                        'authors.hIndex', 'venue', 'publicationTypes',
                        'isOpenAccess', 'fieldsOfStudy']
            )

            if not results or len(results.items) == 0:
                logger.warning(f"No results found on Semantic Scholar for: {title}")
                return {"error": "Paper not found"}

            paper = results.items[0]

            # Extract detailed author information
            authors_detailed = []
            if hasattr(paper, 'authors'):
                for a in paper.authors:
                    author_info = {
                        "name": a.name,
                        "author_id": a.authorId if hasattr(a, 'authorId') else None,
                        "affiliations": a.affiliations if hasattr(a, 'affiliations') else [],
                        "paper_count": a.paperCount if hasattr(a, 'paperCount') else 0,
                        "citation_count": a.citationCount if hasattr(a, 'citationCount') else 0,
                        "h_index": a.hIndex if hasattr(a, 'hIndex') else 0
                    }
                    authors_detailed.append(author_info)

            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "abstract": paper.abstract if hasattr(paper, 'abstract') else None,
                "year": paper.year,
                "citation_count": paper.citationCount if hasattr(paper, 'citationCount') else 0,
                "influential_citation_count": paper.influentialCitationCount if hasattr(paper, 'influentialCitationCount') else 0,
                "authors": [a["name"] for a in authors_detailed],  # Keep simple list for backward compat
                "authors_detailed": authors_detailed,  # Detailed author info
                "venue": paper.venue if hasattr(paper, 'venue') else None,
                "fields_of_study": paper.fieldsOfStudy if hasattr(paper, 'fieldsOfStudy') else [],
                "is_open_access": paper.isOpenAccess if hasattr(paper, 'isOpenAccess') else False
            }

        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return {"error": str(e)}

    def get_paper_references(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most influential papers cited by this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to return

        Returns:
            List of reference paper dictionaries
        """
        try:
            logger.info(f"Getting references for paper: {paper_id}")

            # Get references with citation counts to find influential ones
            references = self.semantic_scholar.get_paper_references(
                paper_id=paper_id,
                fields=['paperId', 'title', 'year', 'citationCount', 'influentialCitationCount', 'authors'],
                limit=100  # Get more to filter
            )

            if not references or not hasattr(references, 'items'):
                return []

            # Sort by citation count to get most influential
            sorted_refs = sorted(
                references.items,
                key=lambda x: x.citedPaper.citationCount if hasattr(x.citedPaper, 'citationCount') else 0,
                reverse=True
            )[:limit]

            result = []
            for ref in sorted_refs:
                paper = ref.citedPaper
                result.append({
                    "paper_id": paper.paperId,
                    "title": paper.title,
                    "year": paper.year if hasattr(paper, 'year') else None,
                    "citation_count": paper.citationCount if hasattr(paper, 'citationCount') else 0,
                    "authors": [a.name for a in paper.authors] if hasattr(paper, 'authors') else []
                })

            logger.info(f"Found {len(result)} influential references")
            return result

        except Exception as e:
            logger.error(f"Error getting paper references: {e}")
            return []

    def search_web_for_products(self, query: str) -> Dict[str, Any]:
        """
        Search the web for information about products or technologies mentioned in the paper.

        Args:
            query: Search query for products/technologies

        Returns:
            Dictionary with search results summary
        """
        try:
            logger.info(f"Searching web for: {query}")

            # Use Gemini to perform web search and summarize
            # Note: This requires Gemini's web search capability
            response = self.genai_client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=f"Search the web for information about: {query}. Provide a brief summary of what this product/technology is, who makes it, and its significance in the field.",
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )

            return {
                "query": query,
                "summary": response.text
            }

        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return {"query": query, "error": str(e)}

    def get_cached_research(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached research results for a paper.

        Args:
            paper_id: Paper ID (UUID)

        Returns:
            Cached research results if available and fresh, None otherwise
        """
        if not self.supabase_client:
            return None

        try:
            response = self.supabase_client.table("paper_research_metadata").select("*").eq("paper_id", paper_id).execute()

            if not response.data or len(response.data) == 0:
                logger.info(f"No cached research found for paper: {paper_id}")
                return None

            cache_entry = response.data[0]

            # Check if cache is still fresh
            last_updated = datetime.fromisoformat(cache_entry["last_updated"].replace('Z', '+00:00'))
            cache_age = datetime.now(last_updated.tzinfo) - last_updated

            if cache_age > timedelta(days=self.cache_ttl_days):
                logger.info(f"Cached research expired for paper: {paper_id} (age: {cache_age.days} days)")
                return None

            logger.info(f"Using cached research for paper: {paper_id}")

            # Reconstruct research results from cache
            return {
                "paper_metadata": cache_entry["metadata"],
                "influential_references": cache_entry["influential_references"],
                "product_searches": [],
                "research_summary": self._generate_research_summary({
                    "paper_metadata": cache_entry["metadata"],
                    "influential_references": cache_entry["influential_references"]
                })
            }

        except Exception as e:
            logger.error(f"Error reading research cache: {e}")
            return None

    def save_research_to_cache(self, paper_id: str, research_results: Dict[str, Any]) -> None:
        """
        Save research results to cache.

        Args:
            paper_id: Paper ID (UUID)
            research_results: Research results to cache
        """
        if not self.supabase_client:
            return

        try:
            metadata = research_results.get("paper_metadata", {})
            if "error" in metadata:
                logger.info("Skipping cache save - no valid metadata")
                return

            cache_data = {
                "paper_id": paper_id,
                "semantic_scholar_id": metadata.get("paper_id"),
                "citation_count": metadata.get("citation_count", 0),
                "influential_citation_count": metadata.get("influential_citation_count", 0),
                "metadata": metadata,
                "influential_references": research_results.get("influential_references", []),
                "last_updated": datetime.utcnow().isoformat()
            }

            # Upsert (insert or update)
            self.supabase_client.table("paper_research_metadata").upsert(cache_data).execute()
            logger.info(f"Saved research to cache for paper: {paper_id}")

        except Exception as e:
            logger.error(f"Error saving research to cache: {e}")

    def research_paper_context(self, paper_id: str, title: str, authors: Optional[str] = None,
                               extract_products: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a paper to gather context.

        This is the main entry point that orchestrates all research activities.

        Args:
            paper_id: Paper ID (UUID) for caching
            title: Paper title
            authors: Optional authors
            extract_products: Whether to search for products mentioned
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary with all research findings
        """
        logger.info(f"Starting comprehensive research for paper: {title}")

        # Check cache first
        if use_cache:
            cached_results = self.get_cached_research(paper_id)
            if cached_results:
                return cached_results

        research_results = {
            "paper_metadata": None,
            "influential_references": [],
            "product_searches": [],
            "research_summary": ""
        }

        # 1. Search Semantic Scholar for the paper
        paper_metadata = self.search_paper_on_semantic_scholar(title, authors)
        research_results["paper_metadata"] = paper_metadata

        if "error" in paper_metadata:
            logger.warning("Could not find paper on Semantic Scholar, continuing with limited context")
            return research_results

        # 2. Get influential references
        semantic_scholar_id = paper_metadata.get("paper_id")
        if semantic_scholar_id:
            references = self.get_paper_references(semantic_scholar_id, limit=5)
            research_results["influential_references"] = references

        # 3. Extract and search for products (if requested)
        # This would require parsing the paper content, which we'll do if extract_products is True
        # For now, we'll skip this step as it requires the paper text

        # 4. Generate a research summary
        research_summary = self._generate_research_summary(research_results)
        research_results["research_summary"] = research_summary

        # 5. Save to cache
        self.save_research_to_cache(paper_id, research_results)

        logger.info("Research complete")
        return research_results

    def _generate_research_summary(self, research_results: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of research findings.

        Args:
            research_results: Dictionary of research results

        Returns:
            Natural language summary
        """
        summary_parts = []

        metadata = research_results.get("paper_metadata", {})
        if metadata and "error" not in metadata:
            citation_count = metadata.get("citation_count", 0)
            year = metadata.get("year", "unknown")
            venue = metadata.get("venue", "unknown venue")

            summary_parts.append(
                f"This paper was published in {year} at {venue}. "
                f"It has been cited {citation_count} times, "
            )

            if citation_count > 1000:
                summary_parts.append("indicating it's a highly influential work in the field. ")
            elif citation_count > 100:
                summary_parts.append("showing it's a well-recognized contribution. ")
            else:
                summary_parts.append("representing an emerging contribution to the field. ")

            # Add author context
            authors_detailed = metadata.get("authors_detailed", [])
            if authors_detailed:
                summary_parts.append("\n\nAuthor Background: ")

                # Categorize authors by experience
                highly_established = []  # h-index > 30 or citations > 100k
                established = []  # h-index > 15 or citations > 50k
                emerging = []  # everyone else

                for author in authors_detailed:
                    h_index = author.get("h_index", 0)
                    total_citations = author.get("citation_count", 0)
                    name = author.get("name", "Unknown")
                    affiliations = author.get("affiliations", [])
                    affiliation_str = f" ({', '.join(affiliations)})" if affiliations else ""

                    if h_index > 30 or total_citations > 100000:
                        highly_established.append(f"{name}{affiliation_str} (h-index: {h_index}, {total_citations:,} total citations)")
                    elif h_index > 15 or total_citations > 50000:
                        established.append(f"{name}{affiliation_str} (h-index: {h_index})")
                    else:
                        paper_count = author.get("paper_count", 0)
                        emerging.append(f"{name}{affiliation_str} ({paper_count} papers)")

                if highly_established:
                    summary_parts.append(f"The team includes highly established researchers: {'; '.join(highly_established)}. ")
                if established:
                    summary_parts.append(f"Established researchers: {'; '.join(established)}. ")
                if emerging:
                    summary_parts.append(f"Emerging researchers: {'; '.join(emerging)}. ")

        references = research_results.get("influential_references", [])
        if references:
            top_ref = references[0]
            summary_parts.append(
                f"\n\nThe paper builds on important prior work including '{top_ref['title']}' "
                f"({top_ref.get('year', 'n/a')}), which has {top_ref.get('citation_count', 0)} citations. "
            )

        return "".join(summary_parts)


def create_research_agent(genai_client: genai.Client, supabase_client: Optional[Client] = None) -> PaperResearchAgent:
    """
    Factory function to create a research agent.

    Args:
        genai_client: Configured Google GenAI client
        supabase_client: Optional Supabase client for caching

    Returns:
        Initialized PaperResearchAgent
    """
    return PaperResearchAgent(genai_client, supabase_client)


async def populate_research_for_existing_papers(
    genai_client: genai.Client,
    supabase_client: Client,
    limit: Optional[int] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Populate research metadata for all existing papers in the database.

    Args:
        genai_client: Configured Google GenAI client
        supabase_client: Supabase client
        limit: Optional limit on number of papers to process
        force_refresh: If True, refresh even if cache exists

    Returns:
        Dictionary with summary of results
    """
    logger.info("Starting to populate research metadata for existing papers...")

    # Create research agent
    agent = create_research_agent(genai_client, supabase_client)

    # Fetch all papers
    query = supabase_client.table("papers").select("id, title, authors")
    if limit:
        query = query.limit(limit)

    response = query.execute()
    papers = response.data

    results = {
        "total": len(papers),
        "success": 0,
        "failed": 0,
        "cached": 0,
        "errors": []
    }

    for paper in papers:
        paper_id = paper["id"]
        title = paper.get("title", "Untitled")
        authors = paper.get("authors")

        try:
            logger.info(f"Processing paper: {title}")

            # Check if cached and skip if not forcing refresh
            if not force_refresh:
                cached = agent.get_cached_research(paper_id)
                if cached:
                    logger.info(f"Skipping {title} - already cached")
                    results["cached"] += 1
                    continue

            # Research the paper
            research_results = agent.research_paper_context(
                paper_id=paper_id,
                title=title,
                authors=authors,
                use_cache=False  # Force fresh research
            )

            if "error" not in research_results.get("paper_metadata", {}):
                results["success"] += 1
                logger.info(f"Successfully researched: {title}")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "paper_id": paper_id,
                    "title": title,
                    "error": "Paper not found on Semantic Scholar"
                })

        except Exception as e:
            logger.error(f"Error processing paper {title}: {e}")
            results["failed"] += 1
            results["errors"].append({
                "paper_id": paper_id,
                "title": title,
                "error": str(e)
            })

    logger.info(f"Research population complete: {results['success']} succeeded, {results['failed']} failed, {results['cached']} cached")
    return results
