"""
RSS Feed Generation for Podcast

This module generates Apple Podcasts-compliant RSS feeds
with full iTunes metadata support.
"""

import logging
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

logger = logging.getLogger(__name__)


def format_duration(seconds: int) -> str:
    """
    Convert duration in seconds to HH:MM:SS or MM:SS format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_rfc2822_date(iso_date_str: str) -> str:
    """
    Convert ISO date string to RFC 2822 format for RSS.

    Args:
        iso_date_str: ISO format date string

    Returns:
        RFC 2822 formatted date string
    """
    dt = datetime.fromisoformat(iso_date_str.replace('Z', '+00:00'))
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")


# Note: RSS feed generation functions will be refactored from main.py
# generate_podcast_rss_feed()
# build_channel_metadata()
# build_episode_item()
