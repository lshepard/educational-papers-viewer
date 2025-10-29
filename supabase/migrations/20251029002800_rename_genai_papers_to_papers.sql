-- ============================================================================
-- Rename genai_papers to papers for broader scope
-- ============================================================================
-- This migration renames the table to support multiple paper sources
-- (not just genai papers, but also arxiv, google scholar, news, etc.)

-- Rename the table
ALTER TABLE public.genai_papers RENAME TO papers;

-- Rename the RLS policies
ALTER POLICY "Allow public read access" ON public.papers RENAME TO "Allow public read access to papers";
ALTER POLICY "Allow authenticated users to insert" ON public.papers RENAME TO "Allow authenticated users to insert papers";
ALTER POLICY "Allow authenticated users to update" ON public.papers RENAME TO "Allow authenticated users to update papers";
ALTER POLICY "Allow authenticated users to delete" ON public.papers RENAME TO "Allow authenticated users to delete papers";

-- Note: Indexes are automatically renamed when the table is renamed
-- Foreign key constraints are also automatically updated
