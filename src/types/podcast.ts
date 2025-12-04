/**
 * Types for the enhanced podcast creator
 */

export interface UnifiedSource {
  id: string
  type: 'paper' | 'article' | 'video'
  source: 'database' | 'semantic_scholar' | 'perplexity' | 'youtube'
  title: string
  authors?: string
  year?: number
  abstract?: string
  url?: string
  citation_count?: number
  venue?: string
}

export interface SearchProviders {
  papers: boolean
  semantic_scholar: boolean
  perplexity: boolean
  youtube: boolean
}

export interface EpisodeConfig {
  theme: string
  title: string
  description: string
}

export interface PodcastCreationState {
  // Search state
  searchQuery: string
  searchProviders: SearchProviders
  searchResults: UnifiedSource[]
  isSearching: boolean

  // Selected sources
  selectedSources: UnifiedSource[]

  // Episode config
  config: EpisodeConfig

  // Generation state
  script: string | null
  isGeneratingScript: boolean
  audioUrl: string | null
  isGeneratingAudio: boolean
  episodeId: string | null
}
