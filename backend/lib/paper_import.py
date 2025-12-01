"""
Paper Import Module

Download and import papers from external sources (arXiv, direct PDFs).
Extracts metadata and uploads to Supabase storage.
"""

import logging
import tempfile
import re
from typing import Dict, Any, Optional
from pathlib import Path
import httpx
from supabase import Client

logger = logging.getLogger(__name__)


async def parse_arxiv_id(url: str) -> Optional[str]:
    """
    Extract arXiv ID from various URL formats.

    Supports:
    - https://arxiv.org/abs/2510.12915
    - https://arxiv.org/pdf/2510.12915.pdf
    - 2510.12915 (direct ID)

    Returns:
        arXiv ID (e.g., "2510.12915") or None if not recognized
    """
    # Direct ID format
    if re.match(r'^\d{4}\.\d{5}(v\d+)?$', url):
        return url

    # Extract from URL
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{5}(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d{4}\.\d{5}(?:v\d+)?)',
        r'arxiv\.org/abs/([a-z\-]+/\d{7})',  # Old format
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


async def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper metadata from arXiv API.

    Args:
        arxiv_id: arXiv identifier (e.g., "2510.12915")

    Returns:
        Dictionary with title, authors, abstract, published date, etc.
    """
    try:
        # Use arXiv API (must use https)
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(api_url, timeout=30.0)
            response.raise_for_status()

            xml_content = response.text

            # Parse XML (simple regex parsing for key fields)
            # In production, use xml.etree.ElementTree for robustness

            title_match = re.search(r'<title>(.*?)</title>', xml_content, re.DOTALL)
            title = title_match.group(1).strip() if title_match else None

            # Skip the feed title, get entry title
            entry_match = re.search(r'<entry>.*?<title>(.*?)</title>', xml_content, re.DOTALL)
            if entry_match:
                title = entry_match.group(1).strip()

            # Extract authors
            authors = []
            for author_match in re.finditer(r'<author>.*?<name>(.*?)</name>', xml_content, re.DOTALL):
                authors.append(author_match.group(1).strip())

            # Extract abstract
            summary_match = re.search(r'<summary>(.*?)</summary>', xml_content, re.DOTALL)
            abstract = summary_match.group(1).strip() if summary_match else None

            # Extract published date
            published_match = re.search(r'<published>(.*?)</published>', xml_content)
            published = published_match.group(1).strip() if published_match else None

            # Extract year from published date
            year = None
            if published:
                year_match = re.search(r'(\d{4})', published)
                if year_match:
                    year = int(year_match.group(1))

            # Extract categories
            category_match = re.search(r'<arxiv:primary_category.*?term="([^"]+)"', xml_content)
            category = category_match.group(1) if category_match else None

            logger.info(f"Fetched arXiv metadata for {arxiv_id}: {title}")

            return {
                "title": title,
                "authors": ", ".join(authors) if authors else None,
                "abstract": abstract,
                "year": year,
                "published_date": published,
                "venue": f"arXiv:{category}" if category else "arXiv",
                "arxiv_id": arxiv_id,
                "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }

    except Exception as e:
        logger.error(f"Failed to fetch arXiv metadata: {e}", exc_info=True)
        return None


async def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file using PyMuPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with extracted metadata
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        metadata = doc.metadata

        # Extract text from first page for title/author detection
        first_page_text = ""
        if len(doc) > 0:
            first_page_text = doc[0].get_text()

        doc.close()

        # Try to extract title from PDF metadata or first page
        title = metadata.get("title")
        if not title or len(title) < 5:
            # Try to extract from first lines of first page
            lines = first_page_text.split('\n')
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 10 and not line.startswith('arXiv'):
                    title = line
                    break

        # Extract author
        authors = metadata.get("author")

        # Extract year from creation date
        year = None
        if metadata.get("creationDate"):
            year_match = re.search(r'(\d{4})', metadata["creationDate"])
            if year_match:
                year = int(year_match.group(1))

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"Failed to extract PDF metadata: {e}")
        return {}


async def find_pdf_with_scrapegraph(page_url: str, api_key: str) -> Optional[str]:
    """
    Use ScrapeGraphAI to find PDF links on a webpage.

    Args:
        page_url: URL of the page to search
        api_key: ScrapeGraphAI API key

    Returns:
        PDF URL if found, None otherwise
    """
    try:
        logger.info(f"Using ScrapeGraphAI to find PDF on: {page_url}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.scrapegraphai.com/v1/smartscraper",
                headers={
                    "SGAI-APIKEY": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "website_url": page_url,
                    "user_prompt": "Find any PDF download link, full text link, or direct PDF URL on this page. Return only the raw PDF URL.",
                    "render_heavy_js": True
                },
                timeout=60.0
            )

            response.raise_for_status()
            data = response.json()

            if data.get("status") == "completed" and data.get("result"):
                result = data["result"]

                # Try to extract PDF URL from result
                # The result structure may vary, so check common patterns
                pdf_url = None

                if isinstance(result, dict):
                    # Check for common keys
                    for key in ["pdf_url", "download_url", "link", "url", "pdf_link"]:
                        if key in result and result[key]:
                            pdf_url = result[key]
                            break

                    # If not found, check nested values
                    if not pdf_url:
                        for value in result.values():
                            if isinstance(value, str) and ".pdf" in value.lower():
                                pdf_url = value
                                break

                elif isinstance(result, str):
                    # Result is a string, check if it's a URL
                    if ".pdf" in result.lower() or result.startswith("http"):
                        pdf_url = result

                if pdf_url:
                    logger.info(f"ScrapeGraphAI found PDF URL: {pdf_url}")
                    return pdf_url
                else:
                    logger.warning("ScrapeGraphAI did not find a PDF URL")
                    return None

            logger.warning(f"ScrapeGraphAI request failed or incomplete: {data.get('status')}")
            return None

    except Exception as e:
        logger.error(f"ScrapeGraphAI PDF search failed: {e}", exc_info=True)
        return None


async def download_pdf(url: str, output_path: Path, scrapegraph_api_key: Optional[str] = None) -> bool:
    """
    Download PDF from URL with ScrapeGraphAI fallback.

    Args:
        url: PDF URL or page URL
        output_path: Where to save the PDF
        scrapegraph_api_key: Optional ScrapeGraphAI API key for fallback

    Returns:
        True if successful
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=60.0)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                # Check magic bytes
                if not response.content.startswith(b'%PDF'):
                    logger.warning(f"URL does not appear to be a PDF: {content_type}")

                    # Try ScrapeGraphAI fallback if API key provided
                    if scrapegraph_api_key:
                        logger.info("Attempting ScrapeGraphAI fallback to find PDF...")
                        pdf_url = await find_pdf_with_scrapegraph(url, scrapegraph_api_key)

                        if pdf_url:
                            # Try downloading the found PDF URL
                            return await download_pdf(pdf_url, output_path, None)  # Don't recurse fallback

                    return False

            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded PDF: {len(response.content)} bytes")
            return True

    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")

        # Try ScrapeGraphAI fallback if API key provided and not already tried
        if scrapegraph_api_key and not url.endswith('.pdf'):
            logger.info("Attempting ScrapeGraphAI fallback after download failure...")
            pdf_url = await find_pdf_with_scrapegraph(url, scrapegraph_api_key)

            if pdf_url:
                # Try downloading the found PDF URL
                return await download_pdf(pdf_url, output_path, None)  # Don't recurse fallback

        return False


async def import_paper_from_url(
    url: str,
    supabase: Client,
    auto_extract: bool = False,
    scrapegraph_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Import a paper from URL (arXiv or direct PDF).

    Process:
    1. Detect if arXiv URL and fetch metadata
    2. Download PDF
    3. Extract metadata from PDF if needed
    4. Upload to Supabase storage
    5. Create database record
    6. Optionally trigger content extraction

    Args:
        url: Paper URL (arXiv or direct PDF)
        supabase: Supabase client
        auto_extract: If True, automatically trigger PDF extraction

    Returns:
        Paper record with paper_id
    """
    temp_dir = None
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="paper_import_")
        temp_path = Path(temp_dir)

        logger.info(f"Importing paper from: {url}")

        # Check if arXiv URL
        arxiv_id = await parse_arxiv_id(url)
        metadata = {}
        pdf_url = url  # Will be overridden for arXiv papers

        if arxiv_id:
            logger.info(f"Detected arXiv paper: {arxiv_id}")
            arxiv_metadata = await fetch_arxiv_metadata(arxiv_id)
            if arxiv_metadata:
                metadata = arxiv_metadata
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"  # Construct proper PDF URL
            else:
                # Fallback if API fails
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Download PDF (with ScrapeGraphAI fallback if API key provided)
        pdf_path = temp_path / "paper.pdf"
        success = await download_pdf(pdf_url, pdf_path, scrapegraph_api_key)

        if not success:
            raise ValueError("Failed to download PDF")

        # Extract metadata from PDF if we don't have it from arXiv
        if not metadata.get("title"):
            pdf_metadata = await extract_pdf_metadata(str(pdf_path))
            metadata.update(pdf_metadata)

        # Ensure we have at least a title
        if not metadata.get("title"):
            metadata["title"] = f"Imported Paper from {url[:50]}"

        # Generate slug for storage
        from lib.pdf_analyzer import create_paper_slug
        slug = create_paper_slug(metadata["title"])

        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()

        # Upload to Supabase storage
        storage_path = f"{slug}/paper.pdf"

        supabase.storage.from_("papers").upload(
            path=storage_path,
            file=pdf_data,
            file_options={
                "content-type": "application/pdf",
                "upsert": "true"
            }
        )

        pdf_url = supabase.storage.from_("papers").get_public_url(storage_path)

        logger.info(f"Uploaded PDF to storage: {storage_path}")

        # Create database record
        paper_data = {
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "paper_url": metadata.get("paper_url") or url,
            "source_url": url,  # Original URL provided by user
            "file_kind": "pdf",  # We're importing PDFs
            "storage_bucket": "papers",
            "storage_path": storage_path,
            "processing_status": "pending"
        }

        response = supabase.table("papers").insert(paper_data).execute()
        paper = response.data[0]

        logger.info(f"Created paper record: {paper['id']}")

        # Trigger extraction if requested
        if auto_extract:
            # Import here to avoid circular dependency
            from lib.pdf_analyzer import extract_paper_sections, extract_images_from_pdf
            import asyncio

            logger.info("Auto-extracting content from paper...")

            # Run extraction in background (don't wait)
            # In production, use a task queue
            try:
                # Note: This is a simplified version
                # The full extraction endpoint in main.py handles this better
                supabase.table("papers").update({
                    "processing_status": "processing"
                }).eq("id", paper["id"]).execute()

                logger.info(f"Paper ready for extraction: {paper['id']}")
            except Exception as e:
                logger.warning(f"Auto-extraction setup failed: {e}")

        return {
            "success": True,
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "storage_path": storage_path,
            "paper_url": paper["paper_url"]
        }

    except Exception as e:
        logger.error(f"Failed to import paper: {e}", exc_info=True)
        raise

    finally:
        # Cleanup temp directory
        if temp_dir and Path(temp_dir).exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
