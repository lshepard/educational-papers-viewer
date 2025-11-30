import React, { useState } from 'react'
import config from '../config'
import { GenaiPaper } from '../supabase'

interface SemanticScholarPaper {
  paperId: string
  title: string
  authors: Array<{ name: string }>
  year: number | null
  venue: string | null
  citationCount: number
  abstract: string | null
  url: string
}

interface SelectedPaper {
  id: string
  title: string
  authors: string
  year: number | null
  source: 'database' | 'semantic-scholar'
  paperId?: string // Semantic Scholar ID
}

const CustomEpisodeCreator: React.FC = () => {
  const [theme, setTheme] = useState('')
  const [selectedPapers, setSelectedPapers] = useState<SelectedPaper[]>([])

  // Database search state
  const [dbSearchTerm, setDbSearchTerm] = useState('')
  const [dbSearchResults, setDbSearchResults] = useState<GenaiPaper[]>([])
  const [dbSearching, setDbSearching] = useState(false)

  // Semantic Scholar search state
  const [ssSearchTerm, setSsSearchTerm] = useState('')
  const [ssSearchResults, setSsSearchResults] = useState<SemanticScholarPaper[]>([])
  const [ssSearching, setSsSearching] = useState(false)

  // Generation state
  const [generating, setGenerating] = useState(false)
  const [generationStatus, setGenerationStatus] = useState('')

  // Search database papers
  const handleDbSearch = async () => {
    if (!dbSearchTerm.trim()) return

    setDbSearching(true)
    try {
      const response = await fetch(
        `${config.backendUrl}/papers/search?q=${encodeURIComponent(dbSearchTerm)}`
      )
      const data = await response.json()
      setDbSearchResults(data.papers || [])
    } catch (error) {
      console.error('Failed to search database:', error)
    } finally {
      setDbSearching(false)
    }
  }

  // Search Semantic Scholar
  const handleSsSearch = async () => {
    if (!ssSearchTerm.trim()) return

    setSsSearching(true)
    try {
      const response = await fetch(
        `${config.backendUrl}/semantic-scholar/search?q=${encodeURIComponent(ssSearchTerm)}`
      )
      const data = await response.json()
      setSsSearchResults(data.papers || [])
    } catch (error) {
      console.error('Failed to search Semantic Scholar:', error)
    } finally {
      setSsSearching(false)
    }
  }

  // Add paper from database
  const handleAddDbPaper = (paper: GenaiPaper) => {
    if (selectedPapers.some(p => p.id === paper.id)) {
      return // Already added
    }

    setSelectedPapers([
      ...selectedPapers,
      {
        id: paper.id,
        title: paper.title || 'Untitled',
        authors: paper.authors || 'Unknown',
        year: paper.year,
        source: 'database'
      }
    ])
  }

  // Add paper from Semantic Scholar
  const handleAddSsPaper = (paper: SemanticScholarPaper) => {
    // Use paperId as unique identifier for SS papers
    const uniqueId = `ss-${paper.paperId}`

    if (selectedPapers.some(p => p.id === uniqueId)) {
      return // Already added
    }

    setSelectedPapers([
      ...selectedPapers,
      {
        id: uniqueId,
        title: paper.title,
        authors: paper.authors.map(a => a.name).join(', '),
        year: paper.year,
        source: 'semantic-scholar',
        paperId: paper.paperId
      }
    ])
  }

  // Remove paper from selection
  const handleRemovePaper = (id: string) => {
    setSelectedPapers(selectedPapers.filter(p => p.id !== id))
  }

  // Fetch citations for selected Semantic Scholar papers
  const handleFetchCitations = async () => {
    const ssPapers = selectedPapers.filter(p => p.source === 'semantic-scholar')
    if (ssPapers.length === 0) {
      alert('No Semantic Scholar papers selected')
      return
    }

    setGenerationStatus('Fetching citations from selected papers...')
    try {
      const paperIds = ssPapers.map(p => p.paperId).filter(Boolean)
      const response = await fetch(`${config.backendUrl}/semantic-scholar/citations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_ids: paperIds })
      })

      const data = await response.json()

      if (data.citations && data.citations.length > 0) {
        // Add cited papers to selected papers
        data.citations.forEach((paper: SemanticScholarPaper) => {
          handleAddSsPaper(paper)
        })
        setGenerationStatus(`Added ${data.citations.length} citations`)
      } else {
        setGenerationStatus('No additional citations found')
      }
    } catch (error) {
      console.error('Failed to fetch citations:', error)
      setGenerationStatus('Failed to fetch citations')
    }
  }

  // Generate custom episode
  const handleGenerateEpisode = async () => {
    if (!theme.trim()) {
      alert('Please enter a theme for the episode')
      return
    }

    if (selectedPapers.length === 0) {
      alert('Please select at least one paper')
      return
    }

    if (!window.confirm(`Generate a 15-20 minute podcast episode about "${theme}" with ${selectedPapers.length} papers? This will take 5-10 minutes.`)) {
      return
    }

    setGenerating(true)
    setGenerationStatus('Starting generation...')

    try {
      const response = await fetch(`${config.backendUrl}/podcast/generate-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          theme,
          papers: selectedPapers
        })
      })

      const data = await response.json()

      if (data.success) {
        setGenerationStatus('Episode generated successfully!')
        alert('Custom episode created! Opening audio...')

        if (data.audio_url) {
          window.open(data.audio_url, '_blank')
        }

        // Reset form
        setTheme('')
        setSelectedPapers([])
        setDbSearchResults([])
        setSsSearchResults([])
      } else {
        setGenerationStatus('Failed: ' + (data.message || 'Unknown error'))
        alert('Failed to generate episode: ' + (data.message || 'Unknown error'))
      }
    } catch (error) {
      console.error('Failed to generate episode:', error)
      setGenerationStatus('Failed to generate episode')
      alert('Failed to generate episode')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="custom-episode-creator">
      <h2>Create Custom Episode</h2>

      {/* Theme Input */}
      <div className="theme-section">
        <h3>Episode Theme</h3>
        <input
          type="text"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          placeholder="e.g., 'The evolution of transformer architectures'"
          className="theme-input"
        />
      </div>

      {/* Selected Papers */}
      <div className="selected-papers">
        <div className="section-header">
          <h3>Selected Papers ({selectedPapers.length})</h3>
          {selectedPapers.some(p => p.source === 'semantic-scholar') && (
            <button
              onClick={handleFetchCitations}
              disabled={generating}
              className="fetch-citations-btn"
            >
              Fetch Citations from Selected Papers
            </button>
          )}
        </div>
        {selectedPapers.length === 0 ? (
          <p className="empty-state">No papers selected yet. Search below to add papers.</p>
        ) : (
          <div className="selected-papers-list">
            {selectedPapers.map(paper => (
              <div key={paper.id} className="selected-paper-item">
                <div className="paper-info">
                  <div className="paper-title">{paper.title}</div>
                  <div className="paper-meta">
                    {paper.authors} {paper.year && `(${paper.year})`}
                    <span className={`source-badge ${paper.source}`}>
                      {paper.source === 'database' ? 'DB' : 'SS'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleRemovePaper(paper.id)}
                  className="remove-btn"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Database Search */}
      <div className="search-section">
        <h3>Search Database Papers</h3>
        <div className="search-box">
          <input
            type="text"
            value={dbSearchTerm}
            onChange={(e) => setDbSearchTerm(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleDbSearch()}
            placeholder="Search your papers..."
            className="search-input"
          />
          <button
            onClick={handleDbSearch}
            disabled={dbSearching}
            className="search-btn"
          >
            {dbSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
        {dbSearchResults.length > 0 && (
          <div className="search-results">
            {dbSearchResults.map(paper => (
              <div key={paper.id} className="search-result-item">
                <div className="paper-info">
                  <div className="paper-title">{paper.title || 'Untitled'}</div>
                  <div className="paper-meta">
                    {paper.authors || 'Unknown'} {paper.year && `(${paper.year})`}
                  </div>
                </div>
                <button
                  onClick={() => handleAddDbPaper(paper)}
                  disabled={selectedPapers.some(p => p.id === paper.id)}
                  className="add-btn"
                >
                  {selectedPapers.some(p => p.id === paper.id) ? '✓ Added' : '+ Add'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Semantic Scholar Search */}
      <div className="search-section">
        <h3>Search Semantic Scholar</h3>
        <div className="search-box">
          <input
            type="text"
            value={ssSearchTerm}
            onChange={(e) => setSsSearchTerm(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSsSearch()}
            placeholder="Search all papers..."
            className="search-input"
          />
          <button
            onClick={handleSsSearch}
            disabled={ssSearching}
            className="search-btn"
          >
            {ssSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
        {ssSearchResults.length > 0 && (
          <div className="search-results">
            {ssSearchResults.map(paper => {
              const uniqueId = `ss-${paper.paperId}`
              return (
                <div key={paper.paperId} className="search-result-item">
                  <div className="paper-info">
                    <div className="paper-title">{paper.title}</div>
                    <div className="paper-meta">
                      {paper.authors.map(a => a.name).join(', ')}
                      {paper.year && ` (${paper.year})`}
                      {paper.citationCount > 0 && ` • ${paper.citationCount} citations`}
                    </div>
                    {paper.abstract && (
                      <div className="paper-abstract">
                        {paper.abstract.substring(0, 200)}...
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleAddSsPaper(paper)}
                    disabled={selectedPapers.some(p => p.id === uniqueId)}
                    className="add-btn"
                  >
                    {selectedPapers.some(p => p.id === uniqueId) ? '✓ Added' : '+ Add'}
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Generate Button */}
      <div className="generate-section">
        <button
          onClick={handleGenerateEpisode}
          disabled={generating || !theme.trim() || selectedPapers.length === 0}
          className="generate-btn"
        >
          {generating ? 'Generating...' : `Generate Episode (${selectedPapers.length} papers)`}
        </button>
        {generationStatus && (
          <div className="generation-status">{generationStatus}</div>
        )}
      </div>

      <style>{`
        .custom-episode-creator {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin-bottom: 30px;
        }

        .theme-section, .selected-papers, .search-section, .generate-section {
          margin-bottom: 30px;
        }

        .theme-section h3, .selected-papers h3, .search-section h3 {
          margin-bottom: 10px;
          color: #333;
        }

        .theme-input {
          width: 100%;
          padding: 12px;
          border: 2px solid #e0e0e0;
          border-radius: 6px;
          font-size: 16px;
        }

        .theme-input:focus {
          outline: none;
          border-color: #007bff;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .fetch-citations-btn {
          padding: 8px 16px;
          background: #6f42c1;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
        }

        .fetch-citations-btn:hover {
          background: #5a32a3;
        }

        .fetch-citations-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .empty-state {
          color: #666;
          font-style: italic;
          padding: 20px;
          text-align: center;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .selected-papers-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .selected-paper-item, .search-result-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px;
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          background: #f8f9fa;
        }

        .selected-paper-item {
          background: #e7f3ff;
          border-color: #0066cc;
        }

        .paper-info {
          flex: 1;
        }

        .paper-title {
          font-weight: 600;
          color: #333;
          margin-bottom: 4px;
        }

        .paper-meta {
          font-size: 14px;
          color: #666;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .paper-abstract {
          font-size: 13px;
          color: #666;
          margin-top: 6px;
          line-height: 1.4;
        }

        .source-badge {
          display: inline-block;
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .source-badge.database {
          background: #28a745;
          color: white;
        }

        .source-badge.semantic-scholar {
          background: #007bff;
          color: white;
        }

        .remove-btn {
          padding: 4px 12px;
          background: #dc3545;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 20px;
          line-height: 1;
        }

        .remove-btn:hover {
          background: #c82333;
        }

        .search-box {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
        }

        .search-input {
          flex: 1;
          padding: 10px;
          border: 2px solid #e0e0e0;
          border-radius: 6px;
          font-size: 14px;
        }

        .search-input:focus {
          outline: none;
          border-color: #007bff;
        }

        .search-btn {
          padding: 10px 20px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          white-space: nowrap;
        }

        .search-btn:hover {
          background: #0056b3;
        }

        .search-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .search-results {
          max-height: 400px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .add-btn {
          padding: 8px 16px;
          background: #28a745;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          white-space: nowrap;
        }

        .add-btn:hover {
          background: #218838;
        }

        .add-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .generate-section {
          text-align: center;
          padding-top: 20px;
          border-top: 2px solid #e0e0e0;
        }

        .generate-btn {
          padding: 15px 40px;
          background: #ff6b35;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-size: 16px;
          font-weight: 600;
        }

        .generate-btn:hover {
          background: #e55a2b;
        }

        .generate-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .generation-status {
          margin-top: 15px;
          padding: 10px;
          background: #f8f9fa;
          border-radius: 6px;
          color: #333;
        }
      `}</style>
    </div>
  )
}

export default CustomEpisodeCreator
