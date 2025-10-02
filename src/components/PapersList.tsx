import React, { useState, useEffect } from 'react'
import { CombinedPaper } from '../supabase'
import { PapersService } from '../services/papersService'

interface PapersListProps {
  onSelectPaper: (paper: CombinedPaper) => void
}

const PapersList: React.FC<PapersListProps> = ({ onSelectPaper }) => {
  const [papers, setPapers] = useState<CombinedPaper[]>([])
  const [filteredPapers, setFilteredPapers] = useState<CombinedPaper[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [fileKindFilter, setFileKindFilter] = useState<string>('')
  const [yearFilter, setYearFilter] = useState<string>('')

  useEffect(() => {
    fetchPapers()
  }, [])

  useEffect(() => {
    filterPapers()
  }, [papers, searchTerm, fileKindFilter, yearFilter])

  const fetchPapers = async () => {
    try {
      const data = await PapersService.getCombinedPapers()
      setPapers(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch papers')
    } finally {
      setLoading(false)
    }
  }

  const filterPapers = () => {
    let filtered = papers

    if (searchTerm) {
      filtered = filtered.filter(paper =>
        paper.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        paper.authors.toLowerCase().includes(searchTerm.toLowerCase()) ||
        paper.venue.toLowerCase().includes(searchTerm.toLowerCase()) ||
        paper.application.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (paper.url && paper.url.toLowerCase().includes(searchTerm.toLowerCase()))
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
  }

  const getUniqueYears = () => {
    const years = papers
      .filter(paper => paper.year)
      .map(paper => paper.year!)
      .filter((year, index, arr) => arr.indexOf(year) === index)
      .sort((a, b) => b - a)
    return years
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
                    disabled={!paper.file_kind || paper.file_kind !== 'pdf'}
                  >
                    {paper.file_kind === 'pdf' ? 'View PDF' : 'View'}
                  </button>
                  <a 
                    href={paper.url} 
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