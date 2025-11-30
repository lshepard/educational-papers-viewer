-- Create table to cache Semantic Scholar research metadata
CREATE TABLE paper_research_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,

    -- Semantic Scholar identifiers
    semantic_scholar_id TEXT,

    -- Citation metrics
    citation_count INTEGER DEFAULT 0,
    influential_citation_count INTEGER DEFAULT 0,

    -- Full metadata from Semantic Scholar (JSON)
    -- Includes: title, abstract, year, authors, venue, fieldsOfStudy, etc.
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Influential references (JSON array)
    -- Each reference includes: paperId, title, year, citationCount, authors
    influential_references JSONB DEFAULT '[]'::jsonb,

    -- Product/technology mentions (JSON array)
    -- For future use: products mentioned in the paper
    product_mentions JSONB DEFAULT '[]'::jsonb,

    -- Timestamps
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure one row per paper
    UNIQUE(paper_id)
);

-- Index for faster lookups
CREATE INDEX idx_paper_research_metadata_paper_id ON paper_research_metadata(paper_id);
CREATE INDEX idx_paper_research_metadata_semantic_scholar_id ON paper_research_metadata(semantic_scholar_id);

-- Index for citation count queries
CREATE INDEX idx_paper_research_metadata_citation_count ON paper_research_metadata(citation_count DESC);

-- Comment
COMMENT ON TABLE paper_research_metadata IS 'Caches research metadata from Semantic Scholar and other sources for each paper';
