"""
Podcast generation agent.

Core orchestrator that uses Gemini with function calling to generate
podcast scripts with research tool support.
"""

import logging
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from .tools import get_tool_definitions, execute_tool

logger = logging.getLogger(__name__)


class PodcastAgent:
    """
    Podcast generation agent using Gemini with tool calling.

    This agent:
    1. Takes a prompt and optional context (papers, theme, etc.)
    2. Uses Gemini with research tools enabled
    3. Lets the agent decide when to search for additional context
    4. Generates an engaging podcast script
    """

    def __init__(
        self,
        genai_client: genai.Client,
        perplexity_api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize podcast agent.

        Args:
            genai_client: Gemini client
            perplexity_api_key: Optional Perplexity API key
            model: Gemini model to use
        """
        self.client = genai_client
        self.perplexity_api_key = perplexity_api_key
        self.model = model

        # Get available tools
        self.tool_definitions = get_tool_definitions(perplexity_api_key)
        self.tools = types.Tool(function_declarations=self.tool_definitions)

        # Log available tools
        tool_names = [t.name for t in self.tool_definitions]
        logger.info(f"Podcast agent initialized with tools: {', '.join(tool_names)}")

    async def generate_script(
        self,
        prompt: str,
        pdf_uri: Optional[str] = None,
        max_iterations: int = 5
    ) -> str:
        """
        Generate podcast script using agent with tool calling.

        The agent will automatically:
        - Read the paper (if PDF URI provided)
        - Search for related papers when needed
        - Search for broader context when needed
        - Generate an engaging script

        Args:
            prompt: Base prompt for script generation
            pdf_uri: Optional Gemini Files API URI for paper PDF
            max_iterations: Maximum tool-calling iterations

        Returns:
            Generated script text
        """
        logger.info("Starting script generation with research tools")

        # Build initial messages
        messages = []
        if pdf_uri:
            messages.append(types.Part.from_uri(file_uri=pdf_uri, mime_type="application/pdf"))
        messages.append(prompt)

        config = types.GenerateContentConfig(tools=[self.tools])

        # Function calling loop
        for iteration in range(max_iterations):
            logger.info(f"Script generation iteration {iteration + 1}/{max_iterations}")

            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config
            )

            # Check for function calls
            function_calls = self._extract_function_calls(response)

            # If no function calls, we're done
            if not function_calls:
                logger.info("No function calls - script generation complete")
                return response.text

            # Execute function calls
            logger.info(f"Executing {len(function_calls)} function call(s)")
            messages.append(response.candidates[0].content)

            for fn_call in function_calls:
                query = fn_call.args.get('query', '')[:100]
                logger.info(f"Calling: {fn_call.name} with query: {query}...")

                # Execute tool
                result = await execute_tool(
                    tool_name=fn_call.name,
                    arguments=dict(fn_call.args),
                    perplexity_api_key=self.perplexity_api_key
                )

                # Add function response to conversation
                messages.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_call.name,
                            response={"result": result}
                        )
                    )
                )

        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached, returning current response")
        return response.text if hasattr(response, 'text') else "Script generation incomplete"

    def _extract_function_calls(self, response) -> List[Any]:
        """Extract function calls from Gemini response."""
        function_calls = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)

        return function_calls

    async def generate_single_paper_script(
        self,
        paper: Dict[str, Any],
        pdf_uri: str
    ) -> str:
        """
        Generate script for a single research paper.

        Args:
            paper: Paper metadata dict
            pdf_uri: Gemini Files API URI for the paper

        Returns:
            Generated script
        """
        prompt = f"""You are creating an engaging podcast episode about a research paper.

Paper: {paper.get('title', 'Untitled')}
Authors: {paper.get('authors', 'Unknown')}
Year: {paper.get('year', 'Unknown')}

Create a 5-7 minute podcast script that:
1. Opens with a compelling hook that grabs attention
2. Explains the research problem and why it matters
3. Describes the methodology in accessible terms
4. Highlights key findings and their significance
5. Discusses real-world applications and implications
6. Closes with a memorable takeaway

Use the search tools if you need:
- Related work or similar studies (search_papers)
- Broader context or real-world examples (search_related_work)

Format as natural speech with [HOST] markers. Be conversational and engaging!"""

        return await self.generate_script(prompt=prompt, pdf_uri=pdf_uri)

    async def generate_multi_paper_script(
        self,
        papers: List[Dict[str, Any]],
        theme: Optional[str] = None
    ) -> str:
        """
        Generate script for multiple papers.

        Args:
            papers: List of paper metadata dicts
            theme: Optional theme/angle for the episode

        Returns:
            Generated script
        """
        # Build papers context
        papers_context = "\n\n".join([
            f"Paper {idx + 1}:\n"
            f"Title: {p.get('title', 'Untitled')}\n"
            f"Authors: {p.get('authors', 'Unknown')}\n"
            f"Year: {p.get('year', 'Unknown')}\n"
            f"Abstract: {p.get('abstract', 'No abstract available')}"
            for idx, p in enumerate(papers)
        ])

        theme_instruction = f"\n\nTheme/Angle: {theme}" if theme else ""

        prompt = f"""You are creating a podcast episode discussing multiple research papers.

{papers_context}{theme_instruction}

Create an engaging 7-10 minute podcast script with:
1. Catchy introduction that frames the connection between papers
2. Discussion of each paper's key contributions
3. Synthesis showing how papers relate or build on each other
4. Practical implications and future directions
5. Memorable closing with call to action

Use the search tools to:
- Find additional related papers (search_papers)
- Get broader context or examples (search_related_work)

Format as natural speech with [HOST] markers. Be engaging and accessible!"""

        return await self.generate_script(prompt=prompt)
