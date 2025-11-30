"""
Research Agent for Podcast Script Generation

This module provides research capabilities for gathering context about papers
before generating podcast scripts. It uses Semantic Scholar API, web search,
and Gemini as an orchestrating agent with function calling.
"""

import logging
from typing import List, Dict, Any, Optional
from semanticscholar import SemanticScholar
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class PaperResearchAgent:
    """Agent that researches papers to gather context for podcast generation."""

    def __init__(self, genai_client: genai.Client):
        """
        Initialize the research agent.

        Args:
            genai_client: Configured Google GenAI client
        """
        self.genai_client = genai_client
        self.semantic_scholar = SemanticScholar()

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
                        'influentialCitationCount', 'authors', 'venue', 'publicationTypes',
                        'isOpenAccess', 'fieldsOfStudy']
            )

            if not results or len(results.items) == 0:
                logger.warning(f"No results found on Semantic Scholar for: {title}")
                return {"error": "Paper not found"}

            paper = results.items[0]

            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "abstract": paper.abstract if hasattr(paper, 'abstract') else None,
                "year": paper.year,
                "citation_count": paper.citationCount if hasattr(paper, 'citationCount') else 0,
                "influential_citation_count": paper.influentialCitationCount if hasattr(paper, 'influentialCitationCount') else 0,
                "authors": [a.name for a in paper.authors] if hasattr(paper, 'authors') else [],
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

    def research_paper_context(self, title: str, authors: Optional[str] = None,
                               extract_products: bool = True) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a paper to gather context.

        This is the main entry point that orchestrates all research activities.

        Args:
            title: Paper title
            authors: Optional authors
            extract_products: Whether to search for products mentioned

        Returns:
            Dictionary with all research findings
        """
        logger.info(f"Starting comprehensive research for paper: {title}")

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
        paper_id = paper_metadata.get("paper_id")
        if paper_id:
            references = self.get_paper_references(paper_id, limit=5)
            research_results["influential_references"] = references

        # 3. Extract and search for products (if requested)
        # This would require parsing the paper content, which we'll do if extract_products is True
        # For now, we'll skip this step as it requires the paper text

        # 4. Generate a research summary
        research_summary = self._generate_research_summary(research_results)
        research_results["research_summary"] = research_summary

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

        references = research_results.get("influential_references", [])
        if references:
            top_ref = references[0]
            summary_parts.append(
                f"The paper builds on important prior work including '{top_ref['title']}' "
                f"({top_ref.get('year', 'n/a')}), which has {top_ref.get('citation_count', 0)} citations. "
            )

        return "".join(summary_parts)


def create_research_agent(genai_client: genai.Client) -> PaperResearchAgent:
    """
    Factory function to create a research agent.

    Args:
        genai_client: Configured Google GenAI client

    Returns:
        Initialized PaperResearchAgent
    """
    return PaperResearchAgent(genai_client)
