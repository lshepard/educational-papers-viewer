-- Remove the CHECK constraint on source_type to allow any string value
ALTER TABLE public.papers DROP CONSTRAINT IF EXISTS papers_source_type_check;
