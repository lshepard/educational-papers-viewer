import { supabase, Paper, PaperNote } from '../supabase'
import config from '../config'

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

  /**
   * Full-text search across paper content using PostgreSQL FTS
   * Returns unique paper IDs that match the search query
   *
   * @param query - Search query string
   * @param limit - Maximum number of paper IDs to return
   * @returns Array of paper IDs that have matching content
   */
  static async searchPaperIds(query: string, limit: number = 100): Promise<string[]> {
    try {
      const { data, error } = await supabase
        .from('paper_sections')
        .select('paper_id')
        .textSearch('fts', query, {
          type: 'websearch',
          config: 'english'
        })
        .limit(limit * 3) // Fetch more to account for deduplication

      if (error) throw error

      // Get unique paper IDs
      const uniqueIds = Array.from(new Set(data?.map(section => section.paper_id) || []))
      return uniqueIds.slice(0, limit)
    } catch (error) {
      console.error('Failed to search paper content:', error)
      return []
    }
  }

  /**
   * Get all papers with their notes (left join)
   *
   * @returns Array of papers with optional note data
   */
  static async getAllPapersWithNotes(): Promise<(Paper & { note?: PaperNote })[]> {
    try {
      // Fetch papers
      const { data: papers, error: papersError } = await supabase
        .from('papers')
        .select('*')
        .order('created_at', { ascending: false })

      if (papersError) throw papersError

      // Fetch all notes
      const { data: notes, error: notesError } = await supabase
        .from('paper_notes')
        .select('*')

      if (notesError) throw notesError

      // Create a map of paper_id -> note
      const notesMap = new Map<string, PaperNote>()
      for (const note of notes || []) {
        notesMap.set(note.paper_id, note)
      }

      // Merge papers with notes
      return (papers || []).map(paper => ({
        ...paper,
        note: notesMap.get(paper.id)
      }))
    } catch (error) {
      console.error('Failed to fetch papers with notes:', error)
      return []
    }
  }

  /**
   * Get notes for a specific paper
   *
   * @param paperId - Paper ID
   * @returns Paper note or null if not found
   */
  static async getPaperNotes(paperId: string): Promise<PaperNote | null> {
    try {
      const response = await fetch(`${config.backendUrl}/papers/${paperId}/notes`)
      if (!response.ok) throw new Error('Failed to fetch notes')
      const data = await response.json()
      return data.id ? data : null
    } catch (error) {
      console.error('Failed to fetch paper notes:', error)
      return null
    }
  }

  /**
   * Update notes and rating for a paper
   *
   * @param paperId - Paper ID
   * @param rating - Rating value or null
   * @param notes - Notes text or null
   * @returns Updated paper note
   */
  static async updatePaperNotes(
    paperId: string,
    rating: 'ignore' | 'ok' | 'highlight' | null,
    notes: string | null
  ): Promise<PaperNote> {
    try {
      const response = await fetch(`${config.backendUrl}/papers/${paperId}/notes`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rating, notes }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to update notes')
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to update paper notes:', error)
      throw error
    }
  }
}