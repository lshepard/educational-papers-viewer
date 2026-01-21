-- Add published_at date column to papers table
ALTER TABLE papers ADD COLUMN IF NOT EXISTS published_at DATE;

-- Create index for sorting by publication date
CREATE INDEX IF NOT EXISTS idx_papers_published_at ON papers(published_at DESC NULLS LAST);

-- Backfill published_at from existing year/month data
-- Use first of month if month exists, otherwise Jan 1 if only year exists
UPDATE papers
SET published_at = CASE
    WHEN month IS NOT NULL AND year IS NOT NULL THEN
        make_date(year, month, 1)
    WHEN year IS NOT NULL THEN
        make_date(year, 1, 1)
    ELSE NULL
END
WHERE published_at IS NULL AND year IS NOT NULL;
