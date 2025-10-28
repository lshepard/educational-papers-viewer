-- ============================================================================
-- Supabase Row Level Security (RLS) Setup for GenAI Papers Viewer
-- ============================================================================
-- This migration sets up RLS policies to allow:
-- - Public read access (no authentication needed)
-- - Authenticated admin users can create/update/delete papers
-- ============================================================================

-- Enable Row Level Security on genai_papers table
ALTER TABLE public.genai_papers ENABLE ROW LEVEL SECURITY;

-- Policy: Allow anyone to read papers (public access)
CREATE POLICY "Allow public read access"
ON public.genai_papers
FOR SELECT
TO public
USING (true);

-- Policy: Allow authenticated users to insert papers
CREATE POLICY "Allow authenticated users to insert"
ON public.genai_papers
FOR INSERT
TO authenticated
WITH CHECK (true);

-- Policy: Allow authenticated users to update papers
CREATE POLICY "Allow authenticated users to update"
ON public.genai_papers
FOR UPDATE
TO authenticated
USING (true)
WITH CHECK (true);

-- Policy: Allow authenticated users to delete papers
CREATE POLICY "Allow authenticated users to delete"
ON public.genai_papers
FOR DELETE
TO authenticated
USING (true);

-- ============================================================================
-- Storage Bucket Policies (for PDF uploads)
-- ============================================================================

-- Enable RLS on storage.objects for the papers bucket
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to papers bucket
CREATE POLICY "Allow public read access to papers bucket"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'papers');

-- Policy: Allow authenticated users to upload to papers bucket
CREATE POLICY "Allow authenticated users to upload to papers bucket"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'papers');

-- Policy: Allow authenticated users to update files in papers bucket
CREATE POLICY "Allow authenticated users to update papers bucket"
ON storage.objects
FOR UPDATE
TO authenticated
USING (bucket_id = 'papers')
WITH CHECK (bucket_id = 'papers');

-- Policy: Allow authenticated users to delete from papers bucket
CREATE POLICY "Allow authenticated users to delete from papers bucket"
ON storage.objects
FOR DELETE
TO authenticated
USING (bucket_id = 'papers');
