"""
RSS Feed Generation for Podcast

This module generates Apple Podcasts-compliant RSS feeds
with full iTunes metadata support.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
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


def build_channel_metadata(channel: Element, feed_config: Dict[str, Any], website_url: str, feed_url: str) -> None:
    """
    Build RSS channel metadata with iTunes podcast extensions.

    Args:
        channel: The XML channel element to populate
        feed_config: Podcast feed configuration dictionary
        website_url: URL to the podcast website
        feed_url: URL to the RSS feed itself
    """
    # Basic channel metadata
    SubElement(channel, "title").text = feed_config.get("title", "Research Papers Podcast")
    SubElement(channel, "description").text = feed_config.get("description", "AI-generated podcasts discussing the latest research papers")
    SubElement(channel, "link").text = website_url
    SubElement(channel, "language").text = feed_config.get("language", "en-us")

    # Copyright
    current_year = datetime.now().year
    copyright_text = feed_config.get("copyright", f"© {current_year} {feed_config.get('author', 'Papers Viewer AI')}")
    SubElement(channel, "copyright").text = copyright_text

    # Last build date
    SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

    # Self-referencing atom:link (required for podcast validation)
    atom_link = SubElement(channel, "{http://www.w3.org/2005/Atom}link")
    atom_link.set("href", feed_url)
    atom_link.set("rel", "self")
    atom_link.set("type", "application/rss+xml")

    # iTunes specific tags
    itunes_author = feed_config.get("author", "Papers Viewer AI")
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text = itunes_author

    # iTunes subtitle and summary
    itunes_subtitle = feed_config.get("subtitle", "AI-Powered Research Paper Discussions")
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}subtitle").text = itunes_subtitle

    itunes_summary = feed_config.get("summary", feed_config.get("description", "AI-generated podcasts discussing the latest research papers"))
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}summary").text = itunes_summary

    # Owner information (required by Apple Podcasts)
    owner = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}owner")
    SubElement(owner, "{http://www.itunes.com/dtds/podcast-1.0.dtd}name").text = feed_config.get("owner_name", itunes_author)
    SubElement(owner, "{http://www.itunes.com/dtds/podcast-1.0.dtd}email").text = feed_config.get("owner_email", "podcast@example.com")

    # Explicit content flag
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "yes" if feed_config.get("explicit", False) else "no"

    # Podcast type (episodic vs serial)
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}type").text = feed_config.get("type", "episodic")

    # Podcast artwork (required - must be square, 1400x1400 to 3000x3000 pixels)
    if feed_config.get("image_url"):
        # Standard RSS image
        image = SubElement(channel, "image")
        SubElement(image, "url").text = feed_config["image_url"]
        SubElement(image, "title").text = feed_config.get("title", "Research Papers Podcast")
        SubElement(image, "link").text = website_url

        # iTunes image
        itunes_image = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        itunes_image.set("href", feed_config["image_url"])

    # Category
    category = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}category")
    category.set("text", feed_config.get("category", "Science"))

    # Optional subcategory
    if feed_config.get("subcategory"):
        subcategory = SubElement(category, "{http://www.itunes.com/dtds/podcast-1.0.dtd}category")
        subcategory.set("text", feed_config["subcategory"])


def build_episode_item(
    channel: Element,
    episode: Dict[str, Any],
    episode_number: int,
    website_url: str,
    feed_config: Dict[str, Any]
) -> None:
    """
    Build a single RSS item element for a podcast episode.

    Args:
        channel: The XML channel element to add the item to
        episode: Episode data dictionary
        episode_number: Episode number for episodic shows
        website_url: URL to the podcast website
        feed_config: Podcast feed configuration
    """
    item = SubElement(channel, "item")

    # Basic episode metadata
    SubElement(item, "title").text = episode["title"]

    # Description
    episode_description = episode.get("description", "")
    SubElement(item, "description").text = episode_description

    # Link to episode page
    episode_link = f"{website_url}/papers/{episode['paper_id']}"
    SubElement(item, "link").text = episode_link

    # Enclosure (audio file) - required for podcast episodes
    if episode.get("audio_url"):
        enclosure = SubElement(item, "enclosure")
        enclosure.set("url", episode["audio_url"])
        enclosure.set("type", "audio/mpeg")
        enclosure.set("length", str(episode.get("file_size_bytes", 0)))

    # Publication date (RFC 2822 format) - required
    if episode.get("published_at"):
        pub_date = datetime.fromisoformat(episode["published_at"].replace('Z', '+00:00'))
        SubElement(item, "pubDate").text = pub_date.strftime("%a, %d %b %Y %H:%M:%S %z")

    # GUID - required, should be unique and permanent
    SubElement(item, "guid", isPermaLink="false").text = episode["id"]

    # iTunes episode-specific metadata
    SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}episode").text = str(episode.get("episode_number", episode_number))

    # Season number (optional)
    if episode.get("season_number"):
        SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}season").text = str(episode["season_number"])

    # Episode type
    episode_type = episode.get("episode_type", "full")
    SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}episodeType").text = episode_type

    # Duration
    if episode.get("duration_seconds"):
        duration_str = format_duration(episode["duration_seconds"])
        SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration").text = duration_str

    # iTunes subtitle (short description, max 255 chars)
    subtitle = episode_description.split('.')[0][:255] if episode_description else episode["title"]
    SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}subtitle").text = subtitle

    # iTunes summary
    SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}summary").text = episode_description

    # Explicit flag per episode
    episode_explicit = episode.get("explicit", feed_config.get("explicit", False))
    SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "yes" if episode_explicit else "no"

    # Episode image (optional)
    if episode.get("image_url"):
        episode_image = SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        episode_image.set("href", episode["image_url"])


def generate_podcast_rss_feed(
    feed_config: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    website_url: str
) -> str:
    """
    Generate a complete RSS 2.0 podcast feed with iTunes extensions.

    Args:
        feed_config: Podcast feed configuration dictionary
        episodes: List of completed episode dictionaries
        website_url: URL to the podcast website

    Returns:
        Pretty-printed XML string of the RSS feed
    """
    # RSS root with all necessary namespaces
    rss = Element("rss", version="2.0")
    rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")
    rss.set("xmlns:atom", "http://www.w3.org/2005/Atom")

    channel = SubElement(rss, "channel")

    # Build channel metadata
    feed_url = f"{website_url}/podcast/feed.xml"
    build_channel_metadata(channel, feed_config, website_url, feed_url)

    # Add episodes
    for idx, episode in enumerate(episodes, start=1):
        episode_number = episode.get("episode_number", len(episodes) - idx + 1)
        build_episode_item(channel, episode, episode_number, website_url, feed_config)

    # Convert to string and pretty print
    xml_str = tostring(rss, encoding="unicode")
    dom = minidom.parseString(xml_str)
    return dom.toprettyxml(indent="  ")
