import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL!
const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY!

export interface GenaiPaper {
  id: string
  source_url: string
  paper_url: string
  file_kind: 'pdf' | 'html' | 'markdown' | 'other'
  storage_bucket: string | null
  storage_path: string | null
  created_at: string
  markdown: string | null
  title?: string
  authors?: string
  month?: string
  year?: number
  venue?: string
  application?: string
  users?: string
  ages?: string
  why?: string
  study_design?: string
}

export interface SheetsPaper {
  id: string
  title: string
  url: string
  authors: string
  month: number
  year: number
  venue: string
  application: string
  users: string
  ages: string
  why: string
  study_design: string
  page: string
  scraped_at: string
  request_id: string
  markdown: string
  error: string
  result: string
}

export interface CombinedPaper extends SheetsPaper {
  storage_bucket?: string | null
  storage_path?: string | null
  file_kind?: 'pdf' | 'html' | 'markdown' | 'other'
  source_url?: string
  paper_url?: string
  created_at?: string
}

export const supabase = createClient(supabaseUrl, supabaseKey)