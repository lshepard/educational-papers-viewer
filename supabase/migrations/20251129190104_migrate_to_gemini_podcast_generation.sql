-- ============================================================================
-- Migrate Podcast Episodes to Gemini-based Generation
-- ============================================================================
-- This migration removes AutoContent API dependencies and updates the podcast
-- episodes table to work with Gemini + Google Cloud TTS generation

-- Drop the autocontent_request_id index
DROP INDEX IF EXISTS public.podcast_episodes_autocontent_request_id_idx;

-- Drop the autocontent_request_id column
ALTER TABLE public.podcast_episodes
DROP COLUMN IF EXISTS autocontent_request_id;

-- Update the generation_status constraint to remove 'downloading' state
ALTER TABLE public.podcast_episodes
DROP CONSTRAINT IF EXISTS podcast_episodes_generation_status_check;

ALTER TABLE public.podcast_episodes
ADD CONSTRAINT podcast_episodes_generation_status_check
CHECK (generation_status IN (
  'pending',      -- Initial state
  'processing',   -- Generating with Gemini + Google TTS
  'completed',    -- Successfully generated and stored
  'failed'        -- Generation failed
));

-- Update any existing 'downloading' status episodes to 'processing'
UPDATE public.podcast_episodes
SET generation_status = 'processing'
WHERE generation_status = 'downloading';
