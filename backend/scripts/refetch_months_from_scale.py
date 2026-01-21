#!/usr/bin/env python3
"""
One-time script to re-fetch month data from Stanford SCALE for existing papers
and populate the published_at field.

Run from backend directory:
    uv run python scripts/refetch_months_from_scale.py
"""

import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
from supabase import create_client

load_dotenv()


async def fetch_date_from_scale(client: httpx.AsyncClient, source_url: str) -> tuple[int | None, int | None]:
    """Fetch the month and year from a Stanford SCALE paper page."""
    try:
        response = await client.get(source_url)
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"    Error fetching {source_url}: {e}")
        return None, None

    soup = BeautifulSoup(response.text, "lxml")

    # Look for date pattern (MM/YYYY) in the page text
    text = soup.get_text()
    date_match = re.search(r"\((\d{1,2})/(\d{4})\)", text)
    if date_match:
        month = int(date_match.group(1))
        year = int(date_match.group(2))
        if 1 <= month <= 12:
            return month, year

    return None, None


async def main():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)

    # Fetch Stanford SCALE papers with null published_at
    print("Fetching Stanford SCALE papers with null published_at...")
    response = (
        supabase.table("papers")
        .select("id, source_url, title, month, year, published_at")
        .eq("source_type", "stanford_scale")
        .is_("published_at", "null")
        .execute()
    )

    papers = response.data
    print(f"Found {len(papers)} Stanford SCALE papers with null published_at\n")

    if not papers:
        print("No papers to update.")
        return

    updated = 0
    failed = 0

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for i, paper in enumerate(papers):
            title = (paper.get("title") or "Untitled")[:50]
            print(f"[{i+1}/{len(papers)}] {title}...")

            month, year = await fetch_date_from_scale(client, paper["source_url"])

            if year:
                # Use fetched month or default to 1 (January)
                m = month if month else 1
                published_at = f"{year}-{m:02d}-01"

                update_data = {"published_at": published_at}
                if month:
                    update_data["month"] = month
                if year:
                    update_data["year"] = year

                supabase.table("papers").update(update_data).eq("id", paper["id"]).execute()
                print(f"    Updated: published_at = {published_at}")
                updated += 1
            else:
                print(f"    No date found")
                failed += 1

            # Small delay to be polite to the server
            await asyncio.sleep(0.3)

    print(f"\nDone! Updated {updated} papers, {failed} had no date data")


if __name__ == "__main__":
    asyncio.run(main())
