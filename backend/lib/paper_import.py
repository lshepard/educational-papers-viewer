"""
Paper Import Module

Download and import papers from external sources (arXiv, direct PDFs).
Extracts metadata and uploads to Supabase storage.
"""

import logging
import tempfile
import re
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import httpx
from supabase import Client

logger = logging.getLogger(__name__)

# Cookie storage file for Cloudflare-protected sites
COOKIE_FILE = Path(__file__).parent.parent / ".cookies.json"


def load_cookies(domain: str) -> Dict[str, str]:
    """Load saved cookies for a domain."""
    if not COOKIE_FILE.exists():
        return {}
    try:
        with open(COOKIE_FILE, 'r') as f:
            all_cookies = json.load(f)
            return all_cookies.get(domain, {})
    except Exception:
        return {}


def save_cookies(domain: str, cookies: Dict[str, str]):
    """Save cookies for a domain."""
    try:
        all_cookies = {}
        if COOKIE_FILE.exists():
            with open(COOKIE_FILE, 'r') as f:
                all_cookies = json.load(f)
        all_cookies[domain] = cookies
        with open(COOKIE_FILE, 'w') as f:
            json.dump(all_cookies, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cookies: {e}")


async def download_pdf_with_scraperapi(url: str, output_path: Path) -> bool:
    """
    Download PDF using ScraperAPI to bypass Cloudflare.

    ScraperAPI handles anti-bot measures including Cloudflare.
    Requires SCRAPERAPI_KEY environment variable.
    """
    api_key = os.getenv("SCRAPERAPI_KEY")
    if not api_key:
        logger.warning("SCRAPERAPI_KEY not set, skipping ScraperAPI")
        return False

    try:
        # ScraperAPI URL format
        proxy_url = f"https://api.scraperapi.com?api_key={api_key}&url={url}&render=true"

        logger.info(f"Downloading via ScraperAPI: {url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(proxy_url)
            response.raise_for_status()

            # Check if we got a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower() or response.content.startswith(b'%PDF'):
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded PDF via ScraperAPI: {len(response.content)} bytes")
                return True
            else:
                # ScraperAPI returned the HTML page, try to find PDF link
                logger.info("ScraperAPI returned HTML, looking for PDF link...")

                # For SSRN, construct the delivery URL
                import re
                abstract_match = re.search(r'abstract_id=(\d+)', url)
                if abstract_match and "ssrn.com" in url:
                    abstract_id = abstract_match.group(1)
                    delivery_url = f"https://papers.ssrn.com/sol3/Delivery.cfm/{abstract_id}.pdf?abstractid={abstract_id}&mirid=1"

                    # Try to download the PDF directly via ScraperAPI
                    pdf_proxy_url = f"https://api.scraperapi.com?api_key={api_key}&url={delivery_url}"
                    pdf_response = await client.get(pdf_proxy_url)

                    if pdf_response.content.startswith(b'%PDF'):
                        with open(output_path, 'wb') as f:
                            f.write(pdf_response.content)
                        logger.info(f"Downloaded SSRN PDF via ScraperAPI: {len(pdf_response.content)} bytes")
                        return True

                logger.warning("ScraperAPI did not return a PDF")
                return False

    except Exception as e:
        logger.error(f"ScraperAPI download failed: {e}")
        return False


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


async def download_pdf_with_undetected_chrome(url: str, output_path: Path) -> bool:
    """
    Download PDF using undetected-chromedriver to bypass Cloudflare.

    This is specifically for sites with aggressive bot protection.
    """
    import asyncio

    def _download_sync():
        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import time

            logger.info(f"Using undetected Chrome for: {url}")

            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            # Set download directory
            download_dir = str(output_path.parent)
            prefs = {
                "download.default_directory": download_dir,
                "download.prompt_for_download": False,
                "plugins.always_open_pdf_externally": True,
            }
            options.add_experimental_option("prefs", prefs)

            driver = uc.Chrome(options=options, headless=False)

            try:
                driver.get(url)

                # Wait for Cloudflare to resolve (up to 30 seconds)
                for i in range(30):
                    if "Just a moment" not in driver.title:
                        break
                    logger.info(f"Waiting for Cloudflare... ({i+1}s)")
                    time.sleep(1)

                logger.info(f"Page title: {driver.title}")

                # Check if it's an SSRN abstract page
                if "ssrn.com" in url and "abstract_id=" in url:
                    # Extract abstract ID
                    import re
                    match = re.search(r'abstract_id=(\d+)', url)
                    if match:
                        abstract_id = match.group(1)
                        # Navigate to delivery URL
                        delivery_url = f"https://papers.ssrn.com/sol3/Delivery.cfm/{abstract_id}.pdf?abstractid={abstract_id}&mirid=1"
                        logger.info(f"Navigating to delivery URL: {delivery_url}")
                        driver.get(delivery_url)

                        # Wait for download to complete
                        time.sleep(5)

                        # Check if PDF was downloaded
                        pdf_file = output_path.parent / f"{abstract_id}.pdf"
                        if pdf_file.exists():
                            pdf_file.rename(output_path)
                            logger.info(f"Downloaded PDF: {output_path}")
                            return True

                        # Also check for paper.pdf
                        for f in output_path.parent.iterdir():
                            if f.suffix == ".pdf":
                                f.rename(output_path)
                                logger.info(f"Downloaded PDF (renamed): {output_path}")
                                return True

                return False

            finally:
                driver.quit()

        except Exception as e:
            logger.error(f"Undetected Chrome download failed: {e}")
            return False

    # Run sync code in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _download_sync)


async def download_pdf_with_browser(url: str, output_path: Path) -> bool:
    """
    Download PDF using Playwright browser automation.

    This handles sites that block bots, require JavaScript,
    or have Cloudflare protection.

    Args:
        url: PDF URL or page URL
        output_path: Where to save the PDF

    Returns:
        True if successful
    """
    try:
        from playwright.async_api import async_playwright

        logger.info(f"Attempting browser-based download for: {url}")

        async with async_playwright() as p:
            # Use non-headless mode to bypass Cloudflare detection
            browser = await p.chromium.launch(
                headless=False,  # Cloudflare detects headless browsers
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                accept_downloads=True,
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            # Check if it's an SSRN URL - need special handling
            is_ssrn = "ssrn.com" in url

            if is_ssrn and "abstract_id=" in url:
                # Navigate to the abstract page first
                logger.info("Detected SSRN abstract page, will click download button")
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)

                # Wait for Cloudflare challenge to resolve - check for actual content
                max_wait = 30  # seconds
                for i in range(max_wait):
                    title = await page.title()
                    if "Just a moment" not in title and "Cloudflare" not in title:
                        break
                    logger.info(f"Waiting for Cloudflare challenge... ({i+1}s)")
                    await page.wait_for_timeout(1000)

                # Log current URL and page title for debugging
                current_url = page.url
                title = await page.title()
                logger.info(f"Page loaded: {title} at {current_url}")

                # Try to find and click the download/PDF button
                download_selectors = [
                    'a[href*="Delivery.cfm"]',
                    'a:has-text("Download")',
                    'a:has-text("PDF")',
                    'button:has-text("Download")',
                    '.download-button',
                    '[data-abstract-id] a',
                ]

                clicked = False
                for selector in download_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            # Start waiting for download before clicking
                            async with page.expect_download(timeout=60000) as download_info:
                                await element.click()
                            download = await download_info.value
                            await download.save_as(output_path)
                            clicked = True
                            logger.info(f"Clicked download button with selector: {selector}")
                            break
                    except Exception as e:
                        logger.debug(f"Selector {selector} failed: {e}")
                        continue

                if not clicked:
                    # Try extracting the PDF URL from page and navigating directly
                    pdf_links = await page.query_selector_all('a[href*=".pdf"], a[href*="Delivery.cfm"]')
                    for link in pdf_links:
                        href = await link.get_attribute("href")
                        if href:
                            logger.info(f"Found PDF link: {href}")
                            if not href.startswith("http"):
                                href = f"https://papers.ssrn.com{href}"
                            async with page.expect_download(timeout=60000) as download_info:
                                await page.goto(href)
                            download = await download_info.value
                            await download.save_as(output_path)
                            clicked = True
                            break

                if not clicked:
                    # If Cloudflare is still showing, we can't proceed
                    title = await page.title()
                    if "Just a moment" in title:
                        logger.warning("Cloudflare challenge not resolved - may need manual intervention")
                        await browser.close()
                        return False

                    # Try to construct and navigate to PDF URL directly within the browser session
                    # Extract abstract ID and try delivery URL
                    import re
                    abstract_match = re.search(r'abstract_id=(\d+)', url)
                    if abstract_match:
                        abstract_id = abstract_match.group(1)
                        delivery_url = f"https://papers.ssrn.com/sol3/Delivery.cfm/{abstract_id}.pdf?abstractid={abstract_id}&mirid=1"
                        logger.info(f"Trying direct delivery URL: {delivery_url}")

                        try:
                            async with page.expect_download(timeout=60000) as download_info:
                                await page.goto(delivery_url)
                            download = await download_info.value
                            await download.save_as(output_path)
                            clicked = True
                        except Exception as e:
                            logger.warning(f"Direct delivery URL failed: {e}")

                if not clicked:
                    logger.warning("Could not find download button on SSRN page")
                    await browser.close()
                    return False
            else:
                # Direct PDF URL or other site - try direct navigation with download handling
                try:
                    async with page.expect_download(timeout=60000) as download_info:
                        await page.goto(url, wait_until="commit", timeout=60000)
                    download = await download_info.value
                    await download.save_as(output_path)
                except Exception:
                    # If no download triggered, page might have loaded - check for PDF content
                    # or look for download links
                    logger.info("No automatic download, looking for PDF links on page")
                    await page.wait_for_timeout(2000)

                    pdf_links = await page.query_selector_all('a[href*=".pdf"]')
                    for link in pdf_links:
                        href = await link.get_attribute("href")
                        if href:
                            try:
                                async with page.expect_download(timeout=30000) as download_info:
                                    await link.click()
                                download = await download_info.value
                                await download.save_as(output_path)
                                break
                            except Exception:
                                continue
                    else:
                        await browser.close()
                        return False

            await browser.close()

            # Verify it's a PDF
            if output_path.exists():
                with open(output_path, 'rb') as f:
                    if f.read(4) == b'%PDF':
                        logger.info(f"Browser downloaded PDF successfully: {output_path.stat().st_size} bytes")
                        return True
                    else:
                        logger.warning("Downloaded file is not a PDF")
                        return False
            return False

    except Exception as e:
        logger.error(f"Browser-based download failed: {e}")
        return False


async def download_pdf(url: str, output_path: Path, scrapegraph_api_key: Optional[str] = None) -> bool:
    """
    Download PDF from URL with browser fallback.

    Args:
        url: PDF URL or page URL
        output_path: Where to save the PDF
        scrapegraph_api_key: Optional ScrapeGraphAI API key for fallback

    Returns:
        True if successful
    """
    # Extract domain for cookie lookup
    from urllib.parse import urlparse
    domain = urlparse(url).netloc

    # First try simple httpx download with stored cookies
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        }

        # Load any stored cookies for this domain
        stored_cookies = load_cookies(domain)
        cookies = httpx.Cookies()
        for name, value in stored_cookies.items():
            cookies.set(name, value, domain=domain)

        if stored_cookies:
            logger.info(f"Using stored cookies for {domain}: {list(stored_cookies.keys())}")

        async with httpx.AsyncClient(follow_redirects=True, cookies=cookies) as client:
            response = await client.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower() or response.content.startswith(b'%PDF'):
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded PDF: {len(response.content)} bytes")
                return True
            else:
                logger.warning(f"URL does not appear to be a PDF: {content_type}")

    except Exception as e:
        logger.error(f"Simple download failed: {e}")

    # Try ScraperAPI for Cloudflare-protected sites (most reliable)
    logger.info("Attempting ScraperAPI fallback...")
    if await download_pdf_with_scraperapi(url, output_path):
        return True

    # Try browser-based download as fallback
    logger.info("Attempting browser-based download fallback...")
    if await download_pdf_with_browser(url, output_path):
        return True

    # Try ScrapeGraphAI to find PDF link as last resort
    if scrapegraph_api_key:
        logger.info("Attempting ScrapeGraphAI fallback to find PDF...")
        pdf_url = await find_pdf_with_scrapegraph(url, scrapegraph_api_key)

        if pdf_url and pdf_url != url:
            # Try downloading the found PDF URL (without recursing fallbacks)
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                }
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(pdf_url, headers=headers, timeout=60.0)
                    response.raise_for_status()

                    if response.content.startswith(b'%PDF'):
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Downloaded PDF via ScrapeGraphAI: {len(response.content)} bytes")
                        return True
            except Exception as e:
                logger.error(f"ScrapeGraphAI URL download failed: {e}")

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
