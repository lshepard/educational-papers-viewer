-- ============================================================================
-- Podcast Episodes Table
-- ============================================================================
-- Store podcast episodes generated from research papers

CREATE TABLE IF NOT EXISTS public.podcast_episodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES public.papers(id) ON DELETE CASCADE,

  -- Episode metadata
  title TEXT NOT NULL,
  description TEXT,
  duration_seconds INTEGER, -- Duration in seconds

  -- Storage
  storage_bucket TEXT DEFAULT 'episodes',
  storage_path TEXT, -- Path to the audio file in Supabase storage
  audio_url TEXT, -- Public URL to the audio file

  -- AutoContent API tracking
  autocontent_request_id TEXT UNIQUE, -- Request ID from AutoContent API
  generation_status TEXT CHECK (generation_status IN (
    'pending',      -- Initial state
    'processing',   -- Generating with AutoContent API
    'downloading',  -- Downloading audio file
    'completed',    -- Successfully generated and stored
    'failed'        -- Generation failed
  )) DEFAULT 'pending',
  generation_error TEXT,

  -- Podcast feed metadata
  published_at TIMESTAMPTZ DEFAULT NOW(),
  episode_number INTEGER, -- Optional episode numbering
  season_number INTEGER,  -- Optional season grouping
  explicit BOOLEAN DEFAULT false,

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX podcast_episodes_paper_id_idx ON public.podcast_episodes(paper_id);
CREATE INDEX podcast_episodes_generation_status_idx ON public.podcast_episodes(generation_status);
CREATE INDEX podcast_episodes_published_at_idx ON public.podcast_episodes(published_at DESC);
CREATE INDEX podcast_episodes_autocontent_request_id_idx ON public.podcast_episodes(autocontent_request_id);

-- ============================================================================
-- Podcast Feed Configuration Table
-- ============================================================================
-- Store podcast feed metadata (title, description, author, etc.)
-- This allows for a single podcast feed with multiple episodes

CREATE TABLE IF NOT EXISTS public.podcast_feed_config (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Feed metadata
  title TEXT NOT NULL DEFAULT 'Research Papers Podcast',
  description TEXT DEFAULT 'AI-generated podcasts about research papers',
  author TEXT DEFAULT 'Papers Viewer',
  email TEXT,
  website_url TEXT,
  image_url TEXT, -- Cover art URL
  language TEXT DEFAULT 'en-us',
  category TEXT DEFAULT 'Science',
  explicit BOOLEAN DEFAULT false,

  -- iTunes/Apple Podcasts specific
  itunes_subtitle TEXT,
  itunes_keywords TEXT, -- Comma-separated keywords

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default podcast feed configuration
INSERT INTO public.podcast_feed_config (title, description, author)
VALUES (
  'Research Papers Podcast',
  'AI-generated podcasts discussing the latest research papers in science and technology',
  'Papers Viewer AI'
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- Row Level Security (RLS) Policies
-- ============================================================================

-- Enable RLS on podcast tables
ALTER TABLE public.podcast_episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.podcast_feed_config ENABLE ROW LEVEL SECURITY;

-- Allow public read access to completed episodes
CREATE POLICY "Allow public read access to completed podcast episodes"
ON public.podcast_episodes FOR SELECT TO public
USING (generation_status = 'completed');

-- Allow public read access to feed config
CREATE POLICY "Allow public read access to podcast feed config"
ON public.podcast_feed_config FOR SELECT TO public USING (true);

-- Allow authenticated users to manage episodes
CREATE POLICY "Allow authenticated users to insert podcast episodes"
ON public.podcast_episodes FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update podcast episodes"
ON public.podcast_episodes FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to delete podcast episodes"
ON public.podcast_episodes FOR DELETE TO authenticated USING (true);

-- Allow authenticated users to update feed config
CREATE POLICY "Allow authenticated users to update podcast feed config"
ON public.podcast_feed_config FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- ============================================================================
-- Updated At Trigger
-- ============================================================================

-- Create trigger function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER update_podcast_episodes_updated_at
  BEFORE UPDATE ON public.podcast_episodes
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_podcast_feed_config_updated_at
  BEFORE UPDATE ON public.podcast_feed_config
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
