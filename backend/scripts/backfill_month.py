#!/usr/bin/env python3
"""
One-time script to backfill month field from source_metadata.

Run from backend directory:
    python scripts/backfill_month.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def main():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)

    # Fetch all papers where month is null
    print("Fetching papers with null month...")
    response = supabase.table("papers").select("id, source_metadata, month").is_("month", "null").execute()

    papers = response.data
    print(f"Found {len(papers)} papers with null month")

    updated = 0
    skipped = 0

    for paper in papers:
        source_metadata = paper.get("source_metadata") or {}
        month_value = source_metadata.get("month")

        if month_value is not None:
            try:
                month_int = int(month_value)
                if 1 <= month_int <= 12:
                    supabase.table("papers").update({"month": month_int}).eq("id", paper["id"]).execute()
                    updated += 1
                    print(f"  Updated paper {paper['id']}: month = {month_int}")
                else:
                    print(f"  Skipped paper {paper['id']}: invalid month value {month_value}")
                    skipped += 1
            except (ValueError, TypeError):
                print(f"  Skipped paper {paper['id']}: could not parse month '{month_value}'")
                skipped += 1
        else:
            skipped += 1

    print(f"\nDone! Updated {updated} papers, skipped {skipped}")


if __name__ == "__main__":
    main()
