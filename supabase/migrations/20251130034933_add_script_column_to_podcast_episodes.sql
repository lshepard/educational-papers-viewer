-- ============================================================================
-- Add Script Column to Podcast Episodes
-- ============================================================================
-- This allows storing and editing the generated podcast script separately
-- from the audio, enabling script editing and audio regeneration

-- Add script column to store the podcast dialogue
ALTER TABLE public.podcast_episodes
ADD COLUMN IF NOT EXISTS script TEXT;

-- Add comment explaining the column
COMMENT ON COLUMN public.podcast_episodes.script IS 'The generated podcast script with speaker dialogue (Alex/Sam format). Can be edited and used to regenerate audio.';
