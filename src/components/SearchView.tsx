import React, { useState } from 'react'
import { PapersService, SearchResult } from '../services/papersService'
import { GenaiPaper } from '../supabase'

interface SearchViewProps {
  onSelectPaper: (paper: GenaiPaper) => void
}

const SearchView: React.FC<SearchViewProps> = ({ onSelectPaper }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!searchQuery.trim()) {
      return
    }

    setLoading(true)
    setError(null)
    setHasSearched(true)

    try {
      const results = await PapersService.searchPaperSections(searchQuery, 20)
      setSearchResults(results)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  const highlightQuery = (text: string, query: string): string => {
    if (!query.trim()) return text

    // Simple highlighting - split by spaces and highlight each word
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2)
    let highlighted = text

    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi')
      highlighted = highlighted.replace(regex, '<mark>$1</mark>')
    })

    return highlighted
  }

  const truncateContent = (content: string, maxLength: number = 300): string => {
    if (content.length <= maxLength) return content
    return content.substring(0, maxLength) + '...'
  }

  return (
    <div className="search-view">
      <div className="search-header">
        <h2>Search Papers</h2>
        <p className="search-help">
          Search across all paper content. Try: <code>"machine learning"</code>, <code>neural OR network</code>, <code>transformer -attention</code>
        </p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search paper content..."
          className="search-input large"
          autoFocus
        />
        <button type="submit" disabled={loading || !searchQuery.trim()} className="search-btn">
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {loading && <div className="loading">Searching...</div>}

      {hasSearched && !loading && (
        <div className="search-results">
          <div className="results-header">
            <h3>{searchResults.length} result{searchResults.length !== 1 ? 's' : ''} found</h3>
          </div>

          {searchResults.length === 0 ? (
            <div className="no-results">
              No results found for <strong>"{searchQuery}"</strong>
            </div>
          ) : (
            <div className="results-list">
              {searchResults.map((result) => (
                <div key={result.id} className="search-result-item">
                  <div className="result-header">
                    <h4 className="result-paper-title">
                      {result.paper?.title || 'Untitled Paper'}
                    </h4>
                    <span className="result-section-type">{result.section_type}</span>
                  </div>

                  {result.section_title && (
                    <div className="result-section-title">
                      <strong>{result.section_title}</strong>
                    </div>
                  )}

                  <div
                    className="result-content"
                    dangerouslySetInnerHTML={{
                      __html: highlightQuery(truncateContent(result.content), searchQuery)
                    }}
                  />

                  <div className="result-meta">
                    {result.paper?.authors && (
                      <span className="result-authors">{result.paper.authors}</span>
                    )}
                    {result.paper?.year && (
                      <span className="result-year">({result.paper.year})</span>
                    )}
                  </div>

                  <div className="result-actions">
                    {result.paper && (
                      <button
                        onClick={() => onSelectPaper(result.paper!)}
                        className="view-paper-btn"
                        disabled={!result.paper.file_kind ||
                          (result.paper.file_kind !== 'pdf' && result.paper.file_kind !== 'markdown')}
                      >
                        View Full Paper
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SearchView
