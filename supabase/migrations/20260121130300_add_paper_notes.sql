-- Add paper_notes table for user ratings and notes on papers
CREATE TABLE paper_notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  rating TEXT CHECK (rating IN ('ignore', 'ok', 'highlight')),
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(paper_id)  -- One note record per paper
);

-- Index for filtering by rating
CREATE INDEX idx_paper_notes_rating ON paper_notes(rating);
CREATE INDEX idx_paper_notes_paper_id ON paper_notes(paper_id);

-- RLS policies (public read, auth write)
ALTER TABLE paper_notes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read" ON paper_notes
  FOR SELECT USING (true);

CREATE POLICY "Allow authenticated insert" ON paper_notes
  FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated update" ON paper_notes
  FOR UPDATE USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated delete" ON paper_notes
  FOR DELETE USING (auth.role() = 'authenticated');
