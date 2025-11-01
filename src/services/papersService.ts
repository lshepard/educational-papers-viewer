import { supabase, Paper } from '../supabase'

export interface PaperSection {
  id: string
  paper_id: string
  section_type: string
  section_title: string | null
  content: string
  created_at: string
}

export interface SearchResult extends PaperSection {
  paper?: Paper
}

export class PapersService {
  static async getAllPapers(): Promise<Paper[]> {
    try {
      const { data, error } = await supabase
        .from('papers')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      return data || []
    } catch (error) {
      console.error('Failed to fetch papers from Supabase:', error)
      return []
    }
  }

  /**
   * Full-text search across paper sections using PostgreSQL FTS
   *
   * @param query - Search query string. Supports:
   *   - Plain text: "machine learning" (both words)
   *   - OR operator: "neural OR network"
   *   - Quotes for phrases: '"deep learning"'
   *   - Negation: "AI -healthcare"
   * @param limit - Maximum number of results to return
   * @returns Array of matching paper sections with paper metadata
   */
  static async searchPaperSections(query: string, limit: number = 20): Promise<SearchResult[]> {
    try {
      // Use textSearch with websearch type for user-friendly query syntax
      const { data, error } = await supabase
        .from('paper_sections')
        .select('id, paper_id, section_type, section_title, content, created_at')
        .textSearch('fts', query, {
          type: 'websearch',
          config: 'english'
        })
        .limit(limit)

      if (error) throw error

      // Fetch associated paper metadata for each result
      const paperIds = Array.from(new Set(data?.map(section => section.paper_id) || []))

      if (paperIds.length === 0) return []

      const { data: papers, error: papersError } = await supabase
        .from('papers')
        .select('*')
        .in('id', paperIds)

      if (papersError) throw papersError

      // Combine sections with their paper metadata
      const papersMap = new Map(papers?.map(p => [p.id, p]) || [])

      return (data || []).map(section => ({
        ...section,
        paper: papersMap.get(section.paper_id)
      }))
    } catch (error) {
      console.error('Failed to search paper sections:', error)
      throw error
    }
  }
}