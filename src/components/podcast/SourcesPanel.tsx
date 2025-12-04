import React from 'react'
import { usePodcastCreation } from '../../contexts/PodcastCreationContext'
import { UnifiedSource } from '../../types/podcast'
import './SourcesPanel.css'

const SourcesPanel: React.FC = () => {
  const {
    searchQuery,
    setSearchQuery,
    searchProviders,
    toggleProvider,
    search,
    searchResults,
    isSearching,
    selectedSources,
    addSource,
    removeSource,
  } = usePodcastCreation()

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    search()
  }

  return (
    <div className="sources-panel">
      {/* Search Section */}
      <div className="search-section">
        <h2>Search Sources</h2>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search papers, research..."
            className="search-input"
          />
          <button
            type="submit"
            disabled={isSearching || !searchQuery.trim()}
            className="search-btn"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </form>

        {/* Provider Checkboxes */}
        <div className="providers">
          <label className="provider-checkbox">
            <input
              type="checkbox"
              checked={searchProviders.papers}
              onChange={() => toggleProvider('papers')}
            />
            <span>Database Papers</span>
          </label>

          <label className="provider-checkbox">
            <input
              type="checkbox"
              checked={searchProviders.semantic_scholar}
              onChange={() => toggleProvider('semantic_scholar')}
            />
            <span>Semantic Scholar</span>
          </label>

          <label className="provider-checkbox disabled">
            <input
              type="checkbox"
              checked={searchProviders.perplexity}
              onChange={() => toggleProvider('perplexity')}
              disabled
            />
            <span>Perplexity (Coming Soon)</span>
          </label>

          <label className="provider-checkbox disabled">
            <input
              type="checkbox"
              checked={searchProviders.youtube}
              onChange={() => toggleProvider('youtube')}
              disabled
            />
            <span>YouTube (Coming Soon)</span>
          </label>
        </div>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="results-section">
          <h3>Results ({searchResults.length})</h3>
          <div className="results-list">
            {searchResults.map((source) => (
              <SourceCard
                key={source.id}
                source={source}
                onAdd={addSource}
                isSelected={selectedSources.some(s => s.id === source.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Selected Sources */}
      <div className="selected-section">
        <h3>Selected Sources ({selectedSources.length})</h3>
        {selectedSources.length === 0 ? (
          <p className="empty-state">No sources selected yet. Search above to add sources.</p>
        ) : (
          <div className="selected-list">
            {selectedSources.map((source) => (
              <div key={source.id} className="selected-source">
                <div className="source-info">
                  <div className="source-title">{source.title}</div>
                  <div className="source-meta">
                    {source.authors && <span>{source.authors}</span>}
                    {source.year && <span>({source.year})</span>}
                    <span className={`source-badge ${source.source}`}>
                      {source.source === 'database' ? 'DB' : 'SS'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => removeSource(source.id)}
                  className="remove-btn"
                  title="Remove source"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

interface SourceCardProps {
  source: UnifiedSource
  onAdd: (source: UnifiedSource) => void
  isSelected: boolean
}

const SourceCard: React.FC<SourceCardProps> = ({ source, onAdd, isSelected }) => {
  return (
    <div className="source-card">
      <div className="source-card-content">
        <div className="source-card-title">{source.title}</div>
        <div className="source-card-meta">
          {source.authors && <span>{source.authors}</span>}
          {source.year && <span>({source.year})</span>}
          {source.citation_count !== undefined && (
            <span>• {source.citation_count} citations</span>
          )}
        </div>
        {source.abstract && (
          <div className="source-card-abstract">
            {source.abstract.substring(0, 200)}...
          </div>
        )}
        <div className="source-card-footer">
          <span className={`source-badge ${source.source}`}>
            {source.source === 'database' ? 'Database' : 'Semantic Scholar'}
          </span>
          {source.venue && <span className="venue">{source.venue}</span>}
        </div>
      </div>
      <button
        onClick={() => onAdd(source)}
        disabled={isSelected}
        className="add-btn"
      >
        {isSelected ? '✓ Added' : '+ Add'}
      </button>
    </div>
  )
}

export default SourcesPanel
