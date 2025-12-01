-- Podcast Creator Schema
-- Manages the multi-step podcast creation workflow

-- Main session table for podcast creator workflow
CREATE TABLE IF NOT EXISTS podcast_creator_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Step 1: Theme and resources
    theme TEXT NOT NULL,
    description TEXT,
    resource_links TEXT[], -- Array of URLs provided by user

    -- Current workflow status
    status TEXT DEFAULT 'research' CHECK (status IN ('research', 'clips', 'show_notes', 'production', 'completed', 'failed')),
    current_step INT DEFAULT 1,

    -- Generated content
    show_notes JSONB, -- Structured outline with quotes and clip markers
    script TEXT, -- Final generated script (if auto-generate mode)

    -- Production mode
    production_mode TEXT CHECK (production_mode IN ('auto', 'live', null)),

    -- Final episode reference
    episode_id UUID REFERENCES podcast_episodes(id),

    -- Metadata
    audio_url TEXT,
    duration_seconds INT,
    error_message TEXT
);

-- Research chat messages
CREATE TABLE IF NOT EXISTS creator_research_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES podcast_creator_sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Message content
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,

    -- Research context
    sources JSONB, -- Sources found/referenced in this message
    bias_analysis JSONB, -- Bias assessment for sources

    -- Ordering
    message_order INT NOT NULL,

    UNIQUE(session_id, message_order)
);

-- YouTube clips selected for the podcast
CREATE TABLE IF NOT EXISTS creator_youtube_clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES podcast_creator_sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- YouTube info
    video_id TEXT NOT NULL,
    video_title TEXT,
    channel_name TEXT,
    video_url TEXT,

    -- Clip timing
    start_time_seconds FLOAT NOT NULL DEFAULT 0,
    end_time_seconds FLOAT NOT NULL,
    duration_seconds FLOAT,

    -- Context
    clip_purpose TEXT, -- What point this clip illustrates
    quote_text TEXT, -- The actual quote/content from the clip

    -- Storage
    audio_file_path TEXT, -- Path in Supabase storage
    audio_url TEXT, -- Public URL for the extracted audio

    -- Show notes integration
    show_notes_marker_id TEXT, -- Reference to where in show notes this should appear
    play_order INT, -- Order in final production

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'downloading', 'ready', 'failed')),
    error_message TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_creator_sessions_status ON podcast_creator_sessions(status);
CREATE INDEX IF NOT EXISTS idx_creator_sessions_created_at ON podcast_creator_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_messages_session ON creator_research_messages(session_id, message_order);
CREATE INDEX IF NOT EXISTS idx_youtube_clips_session ON creator_youtube_clips(session_id, play_order);

-- Updated_at trigger for sessions
CREATE OR REPLACE FUNCTION update_creator_session_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER creator_session_updated_at
    BEFORE UPDATE ON podcast_creator_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_creator_session_updated_at();

-- Comments for documentation
COMMENT ON TABLE podcast_creator_sessions IS 'Main table for multi-step podcast creation workflow';
COMMENT ON TABLE creator_research_messages IS 'Research chat conversation history with bias analysis';
COMMENT ON TABLE creator_youtube_clips IS 'YouTube clips selected and extracted for podcast production';

COMMENT ON COLUMN podcast_creator_sessions.show_notes IS 'Structured JSON outline with quotes, timestamps, and clip markers';
COMMENT ON COLUMN creator_research_messages.bias_analysis IS 'Analysis of source credibility and potential bias (e.g., Google paper about Gemini)';
COMMENT ON COLUMN creator_youtube_clips.show_notes_marker_id IS 'Links clip to specific point in show notes outline';
