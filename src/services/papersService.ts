import { supabase, GenaiPaper, SheetsPaper, CombinedPaper } from '../supabase'

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001'

export class PapersService {
  static async fetchSheetsData(): Promise<SheetsPaper[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/sheets`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include'
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // If the API returned an error with fallback data
      if (data.error && data.data) {
        console.warn('API returned error, using fallback data:', data.error)
        return data.data
      }
      
      // Filter out any rows that don't have a title (likely empty rows)
      const validData = data.filter((paper: SheetsPaper) => 
        paper.title && paper.title.trim() !== ''
      )
      
      return validData
    } catch (error) {
      console.error('Failed to fetch sheets data:', error)
      
      // Return fallback mock data if API is not available
      return [{
        id: "/ai/repository/ends-tests-possibilities-transformative-assessment-and-learning-generative-ai",
        title: "The Ends Of Tests: Possibilities For Transformative Assessment And Learning With Generative AI",
        url: "https://scale.stanford.edu/ai/repository/ends-tests-possibilities-transformative-assessment-and-learning-generative-ai",
        authors: "Bill Cope, Mary Kalantzis, Akash Kumar Saini",
        month: 9,
        year: 2025,
        venue: "Unesco",
        application: "Assessment and Testing",
        users: "Student; Educator",
        ages: "Elementary (PK5); Middle School (6-8); High School (9-12); Post-Secondary; Adult",
        why: "Assessment transformation",
        study_design: "Theoretical Framework",
        page: "",
        scraped_at: "",
        request_id: "",
        markdown: "",
        error: "",
        result: ""
      }]
    }
  }

  static async fetchSupabaseData(): Promise<GenaiPaper[]> {
    try {
      const { data, error } = await supabase
        .from('genai_papers')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      return data || []
    } catch (error) {
      console.error('Failed to fetch Supabase data:', error)
      return []
    }
  }

  static async getCombinedPapers(): Promise<CombinedPaper[]> {
    const [sheetsData, supabaseData] = await Promise.all([
      this.fetchSheetsData(),
      this.fetchSupabaseData()
    ])

    // Create a map of Supabase data by source URL for quick lookup
    const supabaseMap = new Map<string, GenaiPaper>()
    supabaseData.forEach(paper => {
      supabaseMap.set(paper.source_url, paper)
    })

    // Combine data, prioritizing sheets data for metadata
    const combinedPapers: CombinedPaper[] = sheetsData.map(sheetPaper => {
      const supabasePaper = supabaseMap.get(sheetPaper.url)
      
      return {
        ...sheetPaper,
        storage_bucket: supabasePaper?.storage_bucket || null,
        storage_path: supabasePaper?.storage_path || null,
        file_kind: supabasePaper?.file_kind || 'other',
        source_url: supabasePaper?.source_url || sheetPaper.url,
        paper_url: supabasePaper?.paper_url || sheetPaper.url,
        created_at: supabasePaper?.created_at || new Date().toISOString()
      }
    })

    // Add any Supabase papers that aren't in the sheets
    supabaseData.forEach(supabasePaper => {
      const existsInSheets = sheetsData.some(sheet => sheet.url === supabasePaper.source_url)
      if (!existsInSheets) {
        combinedPapers.push({
          id: supabasePaper.id,
          title: supabasePaper.title || 'Untitled',
          url: supabasePaper.paper_url,
          authors: supabasePaper.authors || '',
          month: parseInt(supabasePaper.month || '0'),
          year: supabasePaper.year || new Date().getFullYear(),
          venue: supabasePaper.venue || '',
          application: supabasePaper.application || '',
          users: supabasePaper.users || '',
          ages: supabasePaper.ages || '',
          why: supabasePaper.why || '',
          study_design: supabasePaper.study_design || '',
          page: '',
          scraped_at: '',
          request_id: '',
          markdown: supabasePaper.markdown || '',
          error: '',
          result: '',
          storage_bucket: supabasePaper.storage_bucket,
          storage_path: supabasePaper.storage_path,
          file_kind: supabasePaper.file_kind,
          source_url: supabasePaper.source_url,
          paper_url: supabasePaper.paper_url,
          created_at: supabasePaper.created_at
        })
      }
    })

    return combinedPapers
  }
}