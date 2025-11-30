-- Enable Row Level Security on paper_research_metadata table
ALTER TABLE paper_research_metadata ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can read research metadata
-- This allows the frontend to display citation counts and other research info
CREATE POLICY "Research metadata is publicly readable"
    ON paper_research_metadata
    FOR SELECT
    USING (true);

-- Policy: Only service role can insert/update research metadata
-- This ensures only the backend can manage the cache
CREATE POLICY "Service role can insert research metadata"
    ON paper_research_metadata
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role can update research metadata"
    ON paper_research_metadata
    FOR UPDATE
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role can delete research metadata"
    ON paper_research_metadata
    FOR DELETE
    USING (auth.role() = 'service_role');
