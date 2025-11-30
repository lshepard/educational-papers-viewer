import React, { useState, useEffect, useCallback } from 'react'
import { GenaiPaper } from '../supabase'
import { PapersService } from '../services/papersService'
import { PodcastService } from '../services/podcastService'
import { useAuth } from '../contexts/AuthContext'
import config from '../config'

interface PapersListProps {
  onSelectPaper: (paper: GenaiPaper) => void
}

interface PodcastEpisode {
  id: string
  paper_id: string
  audio_url: string | null
  generation_status: 'pending' | 'processing' | 'completed' | 'failed'
}

const PapersList: React.FC<PapersListProps> = ({ onSelectPaper }) => {
  const { user } = useAuth()
  const [papers, setPapers] = useState<GenaiPaper[]>([])
  const [filteredPapers, setFilteredPapers] = useState<GenaiPaper[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [fileKindFilter, setFileKindFilter] = useState<string>('')
  const [yearFilter, setYearFilter] = useState<string>('')
  const [generatingPodcast, setGeneratingPodcast] = useState<Record<string, boolean>>({})
  const [podcastStatus, setPodcastStatus] = useState<Record<string, string>>({})
  const [podcastEpisodes, setPodcastEpisodes] = useState<Record<string, PodcastEpisode>>({})

  useEffect(() => {
    fetchPapers()
    fetchPodcastEpisodes()
  }, [])

  const fetchPapers = async () => {
    try {
      const data = await PapersService.getAllPapers()
      setPapers(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch papers')
    } finally {
      setLoading(false)
    }
  }

  const fetchPodcastEpisodes = async () => {
    try {
      const response = await fetch(`${config.backendUrl}/podcast/episodes`)
      const data = await response.json()

      if (data.success && data.episodes) {
        // Create a map of paper_id -> episode for quick lookup
        const episodesMap: Record<string, PodcastEpisode> = {}
        data.episodes.forEach((episode: PodcastEpisode) => {
          episodesMap[episode.paper_id] = episode
        })
        setPodcastEpisodes(episodesMap)
      }
    } catch (err) {
      console.error('Failed to fetch podcast episodes:', err)
    }
  }

  const filterPapers = useCallback(() => {
    let filtered = papers

    if (searchTerm) {
      filtered = filtered.filter(paper =>
        (paper.title && paper.title.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (paper.authors && paper.authors.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (paper.venue && paper.venue.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (paper.application && paper.application.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (paper.source_url && paper.source_url.toLowerCase().includes(searchTerm.toLowerCase()))
      )
    }

    if (fileKindFilter) {
      filtered = filtered.filter(paper => paper.file_kind === fileKindFilter)
    }

    if (yearFilter) {
      filtered = filtered.filter(paper =>
        paper.year && paper.year.toString() === yearFilter
      )
    }

    setFilteredPapers(filtered)
  }, [papers, searchTerm, fileKindFilter, yearFilter])

  useEffect(() => {
    filterPapers()
  }, [filterPapers])

  const getUniqueYears = () => {
    const years = papers
      .filter(paper => paper.year)
      .map(paper => paper.year!)
      .filter((year, index, arr) => arr.indexOf(year) === index)
      .sort((a, b) => b - a)
    return years
  }

  const handleGeneratePodcast = async (paperId: string) => {
    try {
      setGeneratingPodcast(prev => ({ ...prev, [paperId]: true }))
      setPodcastStatus(prev => ({ ...prev, [paperId]: 'Generating... (2-5 min)' }))

      // Generate podcast (synchronous, takes 2-5 minutes)
      const response = await PodcastService.generatePodcast(paperId)

      // Success!
      setPodcastStatus(prev => ({ ...prev, [paperId]: 'Completed!' }))

      // Refresh podcast episodes list
      await fetchPodcastEpisodes()

      // Open audio in new tab
      if (response.audio_url) {
        window.open(response.audio_url, '_blank')
      }

      // Reset status after 3 seconds
      setTimeout(() => {
        setPodcastStatus(prev => ({ ...prev, [paperId]: '' }))
      }, 3000)
    } catch (error) {
      console.error('Failed to generate podcast:', error)
      setPodcastStatus(prev => ({
        ...prev,
        [paperId]: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }))
    } finally {
      setGeneratingPodcast(prev => ({ ...prev, [paperId]: false }))
    }
  }

  if (loading) return <div className="loading">Loading papers...</div>
  if (error) return <div className="error">Error: {error}</div>

  return (
    <div className="papers-list">
      <div className="filters">
        <input
          type="text"
          placeholder="Search papers..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        
        <select
          value={fileKindFilter}
          onChange={(e) => setFileKindFilter(e.target.value)}
          className="filter-select"
        >
          <option value="">All file types</option>
          <option value="pdf">PDF</option>
          <option value="html">HTML</option>
          <option value="markdown">Markdown</option>
          <option value="other">Other</option>
        </select>

        <select
          value={yearFilter}
          onChange={(e) => setYearFilter(e.target.value)}
          className="filter-select"
        >
          <option value="">All years</option>
          {getUniqueYears().map(year => (
            <option key={year} value={year.toString()}>{year}</option>
          ))}
        </select>
      </div>

      <div className="papers-table-container">
        <table className="papers-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Authors</th>
              <th>Year</th>
              <th>Venue</th>
              <th>Application</th>
              <th>Type</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPapers.map(paper => (
              <tr key={paper.id} className="paper-row">
                <td className="paper-title">
                  <strong>{paper.title || 'Untitled'}</strong>
                </td>
                <td className="paper-authors">
                  {paper.authors || 'Unknown'}
                </td>
                <td className="paper-year">
                  {paper.year || '-'}
                </td>
                <td className="paper-venue">
                  {paper.venue || '-'}
                </td>
                <td className="paper-application">
                  {paper.application || '-'}
                </td>
                <td className="paper-type">
                  <span className={`file-kind ${paper.file_kind || 'other'}`}>
                    {(paper.file_kind || 'other').toUpperCase()}
                  </span>
                </td>
                <td className="paper-actions">
                  <button
                    onClick={() => onSelectPaper(paper)}
                    className="view-btn"
                    disabled={!paper.file_kind || (paper.file_kind !== 'pdf' && paper.file_kind !== 'markdown')}
                  >
                    {paper.file_kind === 'pdf' ? 'View PDF' : paper.file_kind === 'markdown' ? 'View Markdown' : 'View'}
                  </button>

                  {/* Show podcast link if episode exists and is completed */}
                  {podcastEpisodes[paper.id]?.generation_status === 'completed' && podcastEpisodes[paper.id]?.audio_url && (
                    <a
                      href={podcastEpisodes[paper.id].audio_url!}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="podcast-link"
                      title="Listen to podcast episode"
                    >
                      🎧 Podcast
                    </a>
                  )}

                  {/* Show processing status if generating */}
                  {podcastEpisodes[paper.id]?.generation_status === 'processing' && (
                    <span className="podcast-status processing">
                      ⏳ Generating...
                    </span>
                  )}

                  {/* Show failed status if failed */}
                  {podcastEpisodes[paper.id]?.generation_status === 'failed' && user && (
                    <span className="podcast-status failed" title="Podcast generation failed">
                      ❌ Failed
                    </span>
                  )}

                  {/* Show "Add to Podcast" button only if: logged in as admin, no podcast exists, and is PDF */}
                  {user && !podcastEpisodes[paper.id] && paper.file_kind === 'pdf' && (
                    <button
                      onClick={() => handleGeneratePodcast(paper.id)}
                      className="podcast-btn"
                      disabled={generatingPodcast[paper.id]}
                      title="Generate podcast episode"
                    >
                      {generatingPodcast[paper.id]
                        ? podcastStatus[paper.id] || 'Generating...'
                        : '🎙️ Add to Podcast'}
                    </button>
                  )}

                  <a
                    href={paper.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    Source
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredPapers.length === 0 && (
        <div className="no-results">No papers found matching your criteria.</div>
      )}
    </div>
  )
}

export default PapersList