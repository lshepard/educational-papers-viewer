import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL!
const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY!

export interface Paper {
  id: string
  source_url: string
  paper_url: string | null
  file_kind: 'pdf' | 'html' | 'markdown' | 'other'
  storage_bucket: string | null
  storage_path: string | null
  created_at: string
  markdown: string | null
  title: string | null
  authors: string | null
  month: number | null
  year: number | null
  venue: string | null
  application: string | null
  users: string | null
  ages: string | null
  why: string | null
  study_design: string | null
  source_type?: 'manual' | 'n8n_workflow' | 'google_scholar' | 'arxiv' | 'semantic_scholar' | 'news' | 'agent_discovery'
  source_metadata?: Record<string, any>
  processed_at?: string | null
  processing_status?: 'pending' | 'processing' | 'completed' | 'failed'
  processing_error?: string | null
}

// Legacy alias for backward compatibility
export type GenaiPaper = Paper

export const supabase = createClient(supabaseUrl, supabaseKey)