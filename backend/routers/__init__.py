"""
API routers for the Papers Viewer backend.
"""

from .papers import router as papers_router
from .podcasts import router as podcasts_router
from .search import router as search_router
from .admin import router as admin_router

__all__ = ['papers_router', 'podcasts_router', 'search_router', 'admin_router']
