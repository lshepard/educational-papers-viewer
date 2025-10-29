-- ============================================================================
-- pgvector + Full-Text Search for Hybrid Semantic Search
-- ============================================================================
-- Note: pgvector extension should already be enabled via Supabase dashboard
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Paper Sections Table
-- ============================================================================
-- Store extracted sections from papers (introduction, methods, results, etc.)
CREATE TABLE IF NOT EXISTS public.paper_sections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES public.papers(id) ON DELETE CASCADE,
  section_type TEXT NOT NULL CHECK (section_type IN (
    'introduction',
    'background',
    'methods',
    'results',
    'discussion',
    'conclusion',
    'abstract',
    'other'
  )),
  section_title TEXT,
  content TEXT NOT NULL,

  -- Full-text search column (generated automatically)
  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(section_title, '') || ' ' || content)) STORED,

  -- Vector embedding for semantic search (OpenAI ada-002: 1536 dimensions)
  embedding vector(1536),

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX paper_sections_paper_id_idx ON public.paper_sections(paper_id);
CREATE INDEX paper_sections_section_type_idx ON public.paper_sections(section_type);

-- Full-text search index (GIN)
CREATE INDEX paper_sections_fts_idx ON public.paper_sections USING gin(fts);

-- Vector search index (HNSW is better than IVFFlat for most cases)
CREATE INDEX paper_sections_embedding_idx ON public.paper_sections
  USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Paper Images Table
-- ============================================================================
-- Store extracted images from papers (screenshots, charts, figures)
CREATE TABLE IF NOT EXISTS public.paper_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES public.papers(id) ON DELETE CASCADE,
  image_type TEXT CHECK (image_type IN (
    'screenshot',
    'chart',
    'figure',
    'diagram',
    'table',
    'other'
  )),
  caption TEXT,
  description TEXT,

  -- Full-text search on caption and description
  fts tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(caption, '') || ' ' || coalesce(description, ''))
  ) STORED,

  storage_bucket TEXT,
  storage_path TEXT,
  page_number INTEGER,

  -- Embedding of the description
  embedding vector(1536),

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX paper_images_paper_id_idx ON public.paper_images(paper_id);
CREATE INDEX paper_images_image_type_idx ON public.paper_images(image_type);
CREATE INDEX paper_images_fts_idx ON public.paper_images USING gin(fts);
CREATE INDEX paper_images_embedding_idx ON public.paper_images
  USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Paper Embeddings Table
-- ============================================================================
-- Store document-level embeddings for overall paper
CREATE TABLE IF NOT EXISTS public.paper_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES public.papers(id) ON DELETE CASCADE,
  embedding_type TEXT NOT NULL CHECK (embedding_type IN (
    'title_abstract',
    'full_content',
    'methods',
    'key_findings'
  )),
  embedding vector(1536),
  metadata JSONB, -- Store additional info like token count, model version, etc.
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(paper_id, embedding_type)
);

-- Create indexes
CREATE INDEX paper_embeddings_paper_id_idx ON public.paper_embeddings(paper_id);
CREATE INDEX paper_embeddings_type_idx ON public.paper_embeddings(embedding_type);
CREATE INDEX paper_embeddings_embedding_idx ON public.paper_embeddings
  USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Row Level Security (RLS) Policies
-- ============================================================================
-- Enable RLS on new tables
ALTER TABLE public.paper_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_embeddings ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Allow public read access to paper sections"
ON public.paper_sections FOR SELECT TO public USING (true);

CREATE POLICY "Allow public read access to paper images"
ON public.paper_images FOR SELECT TO public USING (true);

CREATE POLICY "Allow public read access to paper embeddings"
ON public.paper_embeddings FOR SELECT TO public USING (true);

-- Allow authenticated users to write
CREATE POLICY "Allow authenticated users to insert paper sections"
ON public.paper_sections FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update paper sections"
ON public.paper_sections FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to delete paper sections"
ON public.paper_sections FOR DELETE TO authenticated USING (true);

CREATE POLICY "Allow authenticated users to insert paper images"
ON public.paper_images FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update paper images"
ON public.paper_images FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to delete paper images"
ON public.paper_images FOR DELETE TO authenticated USING (true);

CREATE POLICY "Allow authenticated users to insert paper embeddings"
ON public.paper_embeddings FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update paper embeddings"
ON public.paper_embeddings FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to delete paper embeddings"
ON public.paper_embeddings FOR DELETE TO authenticated USING (true);

-- ============================================================================
-- Hybrid Search Functions (Reciprocal Rank Fusion)
-- ============================================================================

-- Function to perform hybrid search on paper sections
CREATE OR REPLACE FUNCTION hybrid_search_sections(
  query_text TEXT,
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.5,
  match_count int DEFAULT 20,
  full_text_weight float DEFAULT 1.0,
  semantic_weight float DEFAULT 1.0
)
RETURNS TABLE (
  section_id UUID,
  paper_id UUID,
  section_type TEXT,
  section_title TEXT,
  content TEXT,
  similarity float,
  fts_rank float,
  hybrid_score float
)
LANGUAGE sql STABLE
AS $$
  WITH semantic_search AS (
    SELECT
      id,
      paper_id,
      section_type,
      section_title,
      content,
      1 - (embedding <=> query_embedding) as similarity,
      ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) as rank
    FROM public.paper_sections
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_embedding
    LIMIT match_count * 2
  ),
  full_text_search AS (
    SELECT
      id,
      paper_id,
      section_type,
      section_title,
      content,
      ts_rank(fts, websearch_to_tsquery('english', query_text)) as rank_score,
      ROW_NUMBER() OVER (ORDER BY ts_rank(fts, websearch_to_tsquery('english', query_text)) DESC) as rank
    FROM public.paper_sections
    WHERE fts @@ websearch_to_tsquery('english', query_text)
    ORDER BY rank_score DESC
    LIMIT match_count * 2
  )
  SELECT
    COALESCE(semantic_search.id, full_text_search.id) as section_id,
    COALESCE(semantic_search.paper_id, full_text_search.paper_id) as paper_id,
    COALESCE(semantic_search.section_type, full_text_search.section_type) as section_type,
    COALESCE(semantic_search.section_title, full_text_search.section_title) as section_title,
    COALESCE(semantic_search.content, full_text_search.content) as content,
    COALESCE(semantic_search.similarity, 0) as similarity,
    COALESCE(full_text_search.rank_score, 0) as fts_rank,
    -- Reciprocal Rank Fusion score
    (COALESCE(semantic_weight / (60 + semantic_search.rank), 0.0) +
     COALESCE(full_text_weight / (60 + full_text_search.rank), 0.0)) as hybrid_score
  FROM semantic_search
  FULL OUTER JOIN full_text_search ON semantic_search.id = full_text_search.id
  ORDER BY hybrid_score DESC
  LIMIT match_count;
$$;

-- Function for pure semantic search (for when you don't have a text query)
CREATE OR REPLACE FUNCTION semantic_search_sections(
  query_embedding vector(1536),
  section_type_filter text DEFAULT NULL,
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 20
)
RETURNS TABLE (
  section_id UUID,
  paper_id UUID,
  section_type TEXT,
  section_title TEXT,
  content TEXT,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    paper_id,
    section_type,
    section_title,
    content,
    1 - (embedding <=> query_embedding) as similarity
  FROM public.paper_sections
  WHERE (section_type_filter IS NULL OR section_type = section_type_filter)
    AND embedding IS NOT NULL
    AND 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Function for pure keyword search
CREATE OR REPLACE FUNCTION keyword_search_sections(
  query_text TEXT,
  section_type_filter text DEFAULT NULL,
  match_count int DEFAULT 20
)
RETURNS TABLE (
  section_id UUID,
  paper_id UUID,
  section_type TEXT,
  section_title TEXT,
  content TEXT,
  rank_score float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    paper_id,
    section_type,
    section_title,
    content,
    ts_rank(fts, websearch_to_tsquery('english', query_text)) as rank_score
  FROM public.paper_sections
  WHERE (section_type_filter IS NULL OR section_type = section_type_filter)
    AND fts @@ websearch_to_tsquery('english', query_text)
  ORDER BY rank_score DESC
  LIMIT match_count;
$$;

-- ============================================================================
-- Processing Status Tracking & Source Management
-- ============================================================================
-- Add columns to papers to track processing status and source
ALTER TABLE public.papers
ADD COLUMN IF NOT EXISTS source_type TEXT CHECK (source_type IN (
  'manual',           -- Manually added
  'n8n_workflow',     -- From existing n8n workflow
  'google_scholar',   -- Imported from Google Scholar
  'arxiv',            -- Imported from arXiv
  'semantic_scholar', -- Imported from Semantic Scholar
  'news',             -- News articles
  'agent_discovery'   -- Discovered by autonomous agent
)) DEFAULT 'manual',
ADD COLUMN IF NOT EXISTS source_metadata JSONB, -- Store search query, discovery reason, etc.
ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS processing_status TEXT CHECK (processing_status IN (
  'pending',
  'processing',
  'completed',
  'failed'
)) DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS processing_error TEXT;

-- Create indexes for filtering by processing status and source
CREATE INDEX IF NOT EXISTS papers_processing_status_idx
  ON public.papers(processing_status);
CREATE INDEX IF NOT EXISTS papers_source_type_idx
  ON public.papers(source_type);
CREATE INDEX IF NOT EXISTS papers_source_metadata_idx
  ON public.papers USING gin(source_metadata);
