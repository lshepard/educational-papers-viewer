import React, { createContext, useContext, useState, ReactNode } from 'react'
import config from '../config'
import { UnifiedSource, SearchProviders, EpisodeConfig, PodcastCreationState } from '../types/podcast'

interface PodcastCreationContextType extends PodcastCreationState {
  // Search actions
  setSearchQuery: (query: string) => void
  toggleProvider: (provider: keyof SearchProviders) => void
  search: () => Promise<void>
  clearResults: () => void

  // Source management
  addSource: (source: UnifiedSource) => void
  removeSource: (sourceId: string) => void
  clearSources: () => void

  // Episode config
  updateConfig: (updates: Partial<EpisodeConfig>) => void

  // Generation
  generatePodcast: () => Promise<void>
  reset: () => void
}

const PodcastCreationContext = createContext<PodcastCreationContextType | undefined>(undefined)

const initialState: PodcastCreationState = {
  searchQuery: '',
  searchProviders: {
    papers: true,
    semantic_scholar: true,
    perplexity: false,
    youtube: false,
  },
  searchResults: [],
  isSearching: false,
  selectedSources: [],
  config: {
    theme: '',
    title: '',
    description: '',
  },
  script: null,
  isGeneratingScript: false,
  audioUrl: null,
  isGeneratingAudio: false,
  episodeId: null,
}

export function PodcastCreationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PodcastCreationState>(initialState)

  const setSearchQuery = (query: string) => {
    setState(prev => ({ ...prev, searchQuery: query }))
  }

  const toggleProvider = (provider: keyof SearchProviders) => {
    setState(prev => ({
      ...prev,
      searchProviders: {
        ...prev.searchProviders,
        [provider]: !prev.searchProviders[provider],
      },
    }))
  }

  const search = async () => {
    if (!state.searchQuery.trim()) return

    setState(prev => ({ ...prev, isSearching: true, searchResults: [] }))

    try {
      // Get enabled providers
      const enabledProviders = Object.entries(state.searchProviders)
        .filter(([_, enabled]) => enabled)
        .map(([provider, _]) => provider)

      if (enabledProviders.length === 0) {
        throw new Error('Please select at least one search provider')
      }

      const response = await fetch(`${config.backendUrl}/search/unified`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: state.searchQuery,
          providers: enabledProviders,
          limit: 10,
        }),
      })

      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.detail || 'Search failed')
      }

      setState(prev => ({
        ...prev,
        searchResults: data.sources || [],
        isSearching: false,
      }))
    } catch (error) {
      console.error('Search failed:', error)
      alert(error instanceof Error ? error.message : 'Search failed')
      setState(prev => ({ ...prev, isSearching: false }))
    }
  }

  const clearResults = () => {
    setState(prev => ({ ...prev, searchResults: [] }))
  }

  const addSource = (source: UnifiedSource) => {
    setState(prev => {
      // Prevent duplicates
      if (prev.selectedSources.some(s => s.id === source.id)) {
        return prev
      }
      return {
        ...prev,
        selectedSources: [...prev.selectedSources, source],
      }
    })
  }

  const removeSource = (sourceId: string) => {
    setState(prev => ({
      ...prev,
      selectedSources: prev.selectedSources.filter(s => s.id !== sourceId),
    }))
  }

  const clearSources = () => {
    setState(prev => ({ ...prev, selectedSources: [] }))
  }

  const updateConfig = (updates: Partial<EpisodeConfig>) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, ...updates },
    }))
  }

  const generatePodcast = async () => {
    if (state.selectedSources.length === 0) {
      alert('Please select at least one source')
      return
    }

    if (!state.config.theme.trim()) {
      alert('Please enter a theme for the episode')
      return
    }

    if (!window.confirm(`Generate podcast with ${state.selectedSources.length} source(s)? This may take 5-10 minutes.`)) {
      return
    }

    setState(prev => ({
      ...prev,
      isGeneratingScript: true,
      isGeneratingAudio: true,
    }))

    try {
      // Extract paper IDs from sources
      const paperIds = state.selectedSources
        .filter(s => s.type === 'paper')
        .map(s => {
          // If from semantic scholar, strip "ss-" prefix
          if (s.source === 'semantic_scholar' && s.id.startsWith('ss-')) {
            return s.id.substring(3)
          }
          return s.id
        })

      if (paperIds.length === 0) {
        throw new Error('At least one paper source is required')
      }

      const response = await fetch(`${config.backendUrl}/podcast/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_ids: paperIds,
          theme: state.config.theme,
          title: state.config.title || undefined,
          description: state.config.description || undefined,
        }),
      })

      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.detail || data.message || 'Generation failed')
      }

      setState(prev => ({
        ...prev,
        episodeId: data.episode_id,
        audioUrl: data.audio_url,
        isGeneratingScript: false,
        isGeneratingAudio: false,
      }))

      alert('Podcast generated successfully!')

      if (data.audio_url) {
        window.open(data.audio_url, '_blank')
      }
    } catch (error) {
      console.error('Podcast generation failed:', error)
      alert(error instanceof Error ? error.message : 'Generation failed')
      setState(prev => ({
        ...prev,
        isGeneratingScript: false,
        isGeneratingAudio: false,
      }))
    }
  }

  const reset = () => {
    setState(initialState)
  }

  return (
    <PodcastCreationContext.Provider
      value={{
        ...state,
        setSearchQuery,
        toggleProvider,
        search,
        clearResults,
        addSource,
        removeSource,
        clearSources,
        updateConfig,
        generatePodcast,
        reset,
      }}
    >
      {children}
    </PodcastCreationContext.Provider>
  )
}

export function usePodcastCreation() {
  const context = useContext(PodcastCreationContext)
  if (!context) {
    throw new Error('usePodcastCreation must be used within PodcastCreationProvider')
  }
  return context
}
