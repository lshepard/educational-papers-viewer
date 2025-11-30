"""
PDF Analysis and Content Extraction

This module handles PDF content extraction including:
- Structured section extraction using Gemini
- Image extraction (embedded and rendered figures)
- Text analysis and processing
"""

import os
import io
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

import fitz  # PyMuPDF
from PIL import Image
from google import genai

logger = logging.getLogger(__name__)


class PaperSections(BaseModel):
    """Structured output model for extracted paper sections."""
    abstract: Optional[str]
    introduction: Optional[str]
    methods: Optional[str]
    results: Optional[str]
    discussion: Optional[str]
    conclusion: Optional[str]
    other: Optional[str]


def create_paper_slug(title: str) -> str:
    """
    Create a URL-friendly slug from paper title.

    Args:
        title: Paper title

    Returns:
        Slug like "problem-solving-teaching-problem-solving-aligning-llms"
    """
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    # Limit length
    slug = slug[:80].strip('-')
    return slug


def save_error_response(response_text: str, error_type: str, error: Exception) -> str:
    """
    Save error response to a file and log it.

    Args:
        response_text: The raw response text that failed to parse
        error_type: Type of error (e.g., 'sections', 'images')
        error: The exception that occurred

    Returns:
        Path to the saved error file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_dir = "error_responses"
    os.makedirs(error_dir, exist_ok=True)

    error_file = os.path.join(error_dir, f"{error_type}_error_{timestamp}.txt")

    try:
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Error: {str(error)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write("RAW RESPONSE:\n")
            f.write("=" * 80 + "\n")
            f.write(response_text)

        logger.error(f"Saved error response to: {error_file}")
        logger.error(f"Error response content (first 1000 chars): {response_text[:1000]}")
    except Exception as save_error:
        logger.error(f"Failed to save error response to file: {save_error}")
        logger.error(f"Error response content (first 1000 chars): {response_text[:1000]}")

    return error_file


def extract_figure_regions_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract figure/table regions from PDF by detecting captions and rendering those areas.
    This captures vector graphics (charts, graphs) that aren't embedded as images.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with rendered figure images
    """
    logger.info(f"Extracting figure regions from PDF: {pdf_path}")
    figures = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Find figure/table captions
            text_dict = page.get_text("dict")
            captions = []

            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]])
                    # Look for figure/table captions
                    if any(keyword in text for keyword in ["Figure ", "Table ", "Fig. "]):
                        captions.append({
                            'text': text,
                            'bbox': block['bbox']
                        })

            # For each caption, render a region around it
            for cap_idx, caption in enumerate(captions):
                try:
                    # Expand the bounding box to capture the figure
                    # Typical figures are 200-400px tall
                    bbox = caption['bbox']
                    x0, y0, x1, y1 = bbox

                    # Expand region (adjust as needed)
                    figure_bbox = fitz.Rect(
                        max(0, x0 - 20),
                        max(0, y0 - 250),  # Look above caption
                        min(page.rect.width, x1 + 20),
                        min(page.rect.height, y1 + 20)
                    )

                    # Render this region at high DPI
                    mat = fitz.Matrix(2, 2)  # 2x zoom = ~144 DPI
                    pix = page.get_pixmap(matrix=mat, clip=figure_bbox)

                    # Convert to PNG bytes
                    image_bytes = pix.tobytes("png")

                    # Get dimensions
                    width, height = pix.width, pix.height

                    # Skip if too small
                    if width < 100 or height < 100:
                        continue

                    figures.append({
                        'page_num': page_num + 1,
                        'image_index': cap_idx + 1,
                        'image_bytes': image_bytes,
                        'ext': 'png',
                        'width': width,
                        'height': height,
                        'caption': caption['text'][:100]  # First 100 chars
                    })

                    logger.info(f"Rendered figure: page {page_num + 1}, caption: {caption['text'][:50]}..., size {width}x{height}")

                except Exception as e:
                    logger.warning(f"Failed to render figure region on page {page_num}: {e}")
                    continue

        doc.close()
        logger.info(f"Extracted {len(figures)} figure regions from PDF")
        return figures

    except Exception as e:
        logger.error(f"Failed to extract figure regions from PDF: {e}")
        return []


def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all images from a PDF using PyMuPDF.
    This includes both:
    1. Embedded raster images (PNG/JPG)
    2. Rendered figure/table regions (vector graphics)

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with: {
            'page_num': int,
            'image_index': int,
            'image_bytes': bytes,
            'ext': str (png/jpg),
            'width': int,
            'height': int,
            'caption': str (optional, for rendered figures)
        }
    """
    logger.info(f"Extracting images from PDF: {pdf_path}")
    images = []

    try:
        doc = fitz.open(pdf_path)

        # Method 1: Extract embedded images
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]  # Image XREF number

                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Get image dimensions
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size

                    # Skip very small images (likely logos, icons)
                    if width < 100 or height < 100:
                        logger.debug(f"Skipping small embedded image: {width}x{height}")
                        continue

                    images.append({
                        'page_num': page_num + 1,  # 1-indexed
                        'image_index': img_index + 1,
                        'image_bytes': image_bytes,
                        'ext': image_ext,
                        'width': width,
                        'height': height,
                        'source': 'embedded'
                    })

                    logger.info(f"Extracted embedded image: page {page_num + 1}, index {img_index + 1}, size {width}x{height}, format {image_ext}")

                except Exception as e:
                    logger.warning(f"Failed to extract embedded image {img_index} from page {page_num}: {e}")
                    continue

        doc.close()

        # Method 2: Render figure regions (for vector graphics)
        figure_images = extract_figure_regions_from_pdf(pdf_path)

        # Merge both methods
        # Re-index the figure images
        next_index = {}
        for img in images:
            page = img['page_num']
            next_index[page] = max(next_index.get(page, 0), img['image_index'])

        for fig in figure_images:
            page = fig['page_num']
            fig['image_index'] = next_index.get(page, 0) + 1
            next_index[page] = fig['image_index']
            fig['source'] = 'rendered'
            images.append(fig)

        logger.info(f"Extracted {len(images)} total images from PDF (embedded + rendered)")
        return images

    except Exception as e:
        logger.error(f"Failed to extract images from PDF: {e}")
        return []


def extract_paper_sections(gemini_file) -> list[Dict[str, Any]]:
    """
    Extract sections from a research paper PDF using Gemini structured output.

    Args:
        gemini_file: The Gemini file object representing the uploaded PDF

    Returns:
        A list of section dictionaries with section_type, section_title, and content
    """
    logger.info("Extracting paper sections with structured output...")

    sections_prompt = """
Extract all text sections from this research paper PDF.

Extract these sections if present:
- Abstract
- Introduction
- Methods/Methodology - BE VERY DETAILED HERE, extract every detail about the methodology
- Results
- Discussion
- Conclusion
- Any other sections (captured in the 'other' field)

Include the COMPLETE text content for each section. For Methods, extract every detail about the methodology.
"""
    model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": PaperSections
        },
    )

    response = model.generate_content([gemini_file, sections_prompt])

    # Parse the structured response
    try:
        sections_data = json.loads(response.text)
    except Exception as e:
        error_file = save_error_response(response.text, "sections", e)
        raise ValueError(f"Failed to parse JSON from Gemini sections response: {e}. Full response saved to: {error_file}")

    paper_sections = PaperSections(**sections_data)

    # Convert to the format expected by the rest of the code
    sections = []
    section_mapping = {
        "abstract": ("abstract", "Abstract"),
        "introduction": ("introduction", "Introduction"),
        "methods": ("methods", "Methods"),
        "results": ("results", "Results"),
        "discussion": ("discussion", "Discussion"),
        "conclusion": ("conclusion", "Conclusion"),
        "other": ("other", "Other"),
    }

    for field_name, (section_type, section_title) in section_mapping.items():
        content = getattr(paper_sections, field_name)
        if content:
            sections.append({
                "section_type": section_type,
                "section_title": section_title,
                "content": content
            })

    logger.info(f"Extracted {len(sections)} sections from structured output")
    return sections
