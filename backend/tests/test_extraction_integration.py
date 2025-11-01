"""
Integration tests for paper extraction functionality.

These tests make actual calls to the Gemini API and should be run
with a valid GEMINI_API_KEY environment variable set.
"""
import os
import pytest
import sys
from pathlib import Path

# Add backend directory to path so we can import main
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import google.generativeai as genai
from main import extract_paper_sections


@pytest.fixture(autouse=True)
def setup_gemini(gemini_api_key):
    """Configure Gemini API key before each test."""
    genai.configure(api_key=gemini_api_key)
    yield
    # Cleanup if needed


@pytest.mark.integration
def test_extract_paper_sections_integration(test_pdf_path, gemini_api_key):
    """
    Integration test for extracting paper sections from a PDF.
    
    This test:
    1. Uploads the test PDF to Gemini
    2. Extracts sections using the extract_paper_sections function
    3. Verifies the structure and content of the extracted sections
    4. Cleans up the uploaded file
    """
    gemini_file = None
    
    try:
        # Upload PDF to Gemini
        print(f"\nUploading test PDF: {test_pdf_path}")
        gemini_file = genai.upload_file(test_pdf_path)
        print(f"File uploaded to Gemini: {gemini_file.name}")
        
        # Extract sections
        print("Extracting sections...")
        sections = extract_paper_sections(gemini_file)
        
        # Verify results
        assert isinstance(sections, list), "Sections should be a list"
        assert len(sections) > 0, "Should extract at least one section"
        
        # Verify structure of each section
        required_keys = {"section_type", "section_title", "content"}
        for section in sections:
            assert isinstance(section, dict), "Each section should be a dictionary"
            assert required_keys.issubset(section.keys()), \
                f"Section missing required keys. Got: {section.keys()}"
            assert section["section_type"] in [
                "abstract", "introduction", "methods", "results", 
                "discussion", "conclusion", "other"
            ], f"Invalid section_type: {section['section_type']}"
            assert isinstance(section["content"], str), "Content should be a string"
            assert len(section["content"]) > 0, "Content should not be empty"
        
        # Check for expected sections based on the test paper
        section_types = [s["section_type"] for s in sections]
        assert "abstract" in section_types, "Should extract abstract section"
        
        print(f"\n✓ Successfully extracted {len(sections)} sections")
        print(f"  Section types: {', '.join(section_types)}")
        
        # Print a summary
        for section in sections:
            content_preview = section["content"][:100].replace("\n", " ")
            print(f"  - {section['section_type']}: {content_preview}...")
        
    finally:
        # Cleanup: delete uploaded file from Gemini
        if gemini_file:
            try:
                gemini_file.delete()
                print(f"\n✓ Cleaned up Gemini file: {gemini_file.name}")
            except Exception as e:
                print(f"\n⚠ Warning: Could not delete Gemini file: {e}")

