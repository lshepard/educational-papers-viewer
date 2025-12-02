"""
Podcast generation system - Agent-centric architecture.

This module provides a clean, tool-based approach to podcast generation
using Gemini AI with function calling.
"""

from .agent import PodcastAgent
from .tools import PODCAST_TOOLS
from .audio import generate_audio_from_script, convert_to_mp3
from .script import format_script_for_tts, generate_metadata

__all__ = [
    'PodcastAgent',
    'PODCAST_TOOLS',
    'generate_audio_from_script',
    'convert_to_mp3',
    'format_script_for_tts',
    'generate_metadata',
]
