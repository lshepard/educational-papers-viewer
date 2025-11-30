-- Create junction table for many-to-many relationship between episodes and papers
CREATE TABLE IF NOT EXISTS episode_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID NOT NULL REFERENCES podcast_episodes(id) ON DELETE CASCADE,
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    semantic_scholar_id VARCHAR(255),
    paper_title TEXT NOT NULL,
    paper_authors TEXT,
    paper_year INTEGER,
    display_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure we have either a paper_id or semantic_scholar_id
    CONSTRAINT check_paper_reference CHECK (
        paper_id IS NOT NULL OR semantic_scholar_id IS NOT NULL
    )
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_episode_papers_episode_id ON episode_papers(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_papers_paper_id ON episode_papers(paper_id);
CREATE INDEX IF NOT EXISTS idx_episode_papers_semantic_scholar_id ON episode_papers(semantic_scholar_id);

-- Add a flag to podcast_episodes to indicate if it's a custom multi-paper episode
ALTER TABLE podcast_episodes
ADD COLUMN IF NOT EXISTS is_multi_paper BOOLEAN DEFAULT FALSE;

-- Make paper_id nullable for multi-paper episodes
ALTER TABLE podcast_episodes
ALTER COLUMN paper_id DROP NOT NULL;

COMMENT ON TABLE episode_papers IS 'Junction table for many-to-many relationship between podcast episodes and papers';
COMMENT ON COLUMN episode_papers.paper_id IS 'Reference to paper in our database (nullable for external papers)';
COMMENT ON COLUMN episode_papers.semantic_scholar_id IS 'Semantic Scholar paper ID for external papers';
COMMENT ON COLUMN episode_papers.display_order IS 'Order in which papers should be displayed/discussed in the episode';
