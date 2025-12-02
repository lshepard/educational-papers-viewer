#!/usr/bin/env python3
"""
Papers Viewer Backend - Refactored Architecture

FastAPI backend with clean router-based organization.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client

# Import core services
from lib.core import GeminiFileManager, PaperExtractionService

# Import routers
from routers import papers_router, podcasts_router, search_router, semantic_scholar_router, admin_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# CORS configuration
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("=" * 80)
    logger.info("Starting Papers Viewer Backend (Refactored Architecture)")
    logger.info("=" * 80)

    # Initialize Gemini client
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        app.state.genai_client = genai_client
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
        raise

    # Initialize core services
    try:
        gemini_manager = GeminiFileManager(genai_client)
        extraction_service = PaperExtractionService(supabase, gemini_manager)

        app.state.gemini_manager = gemini_manager
        app.state.extraction_service = extraction_service

        logger.info("✅ Core services initialized")
        logger.info("   - GeminiFileManager: Centralized file upload")
        logger.info("   - PaperExtractionService: Unified extraction logic")
    except Exception as e:
        logger.error(f"❌ Failed to initialize core services: {e}")
        raise

    # Check optional services
    if PERPLEXITY_API_KEY:
        logger.info("✅ Perplexity API configured - research tools enabled")
    else:
        logger.info("⚠️  Perplexity API not configured - limited research capabilities")

    logger.info("=" * 80)
    logger.info("Backend ready to serve requests")
    logger.info("=" * 80)

    yield

    logger.info("Shutting down Papers Viewer Backend")


# Create FastAPI app
app = FastAPI(
    title="Papers Viewer API",
    description="Refactored backend with clean router-based architecture",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
if ALLOW_ALL_ORIGINS:
    logger.info("CORS: Allowing all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allowed_origins = [FRONTEND_URL, FRONTEND_URL]  # Duplicate for safety
    logger.info(f"CORS: Allowing specific origins: {allowed_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register routers
app.include_router(papers_router)
app.include_router(podcasts_router)
app.include_router(search_router)
app.include_router(semantic_scholar_router)
app.include_router(admin_router)


# ==================== Root Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Papers Viewer API (Refactored)",
        "version": "2.0.0",
        "architecture": "router-based",
        "routes": {
            "papers": "/papers/*",
            "podcasts": "/podcast/*",
            "search": "/search/*",
            "admin": "/admin/*"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "supabase": "connected",
            "gemini": "initialized",
            "extraction": "ready",
            "routers": ["papers", "podcasts", "search", "admin"]
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
