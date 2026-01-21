"""
Paper Ingestion Module

Automated ingestion of papers from various sources (Stanford SCALE, arXiv, etc.)
"""

from .base import PaperMetadata, PaperSource
from .runner import run_ingestion, IngestionResult

__all__ = ["PaperMetadata", "PaperSource", "run_ingestion", "IngestionResult"]
