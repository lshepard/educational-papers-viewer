import { supabase, GenaiPaper } from '../supabase'

export class PapersService {
  static async getAllPapers(): Promise<GenaiPaper[]> {
    try {
      const { data, error } = await supabase
        .from('genai_papers')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      return data || []
    } catch (error) {
      console.error('Failed to fetch papers from Supabase:', error)
      return []
    }
  }
}