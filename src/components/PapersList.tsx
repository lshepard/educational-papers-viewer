import React, { useState, useEffect, useCallback } from 'react'
import { GenaiPaper } from '../supabase'
import { PapersService } from '../services/papersService'

interface PapersListProps {
  onSelectPaper: (paper: GenaiPaper) => void
}

const PapersList: React.FC<PapersListProps> = ({ onSelectPaper }) => {
  const [papers, setPapers] = useState<GenaiPaper[]>([])
  const [filteredPapers, setFilteredPapers] = useState<GenaiPaper[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [fileKindFilter, setFileKindFilter] = useState<string>('')
  const [yearFilter, setYearFilter] = useState<string>('')

  useEffect(() => {
    fetchPapers()
  }, [])

  const fetchPapers = async () => {
    try {
      const data = await PapersService.getAllPapers()
      setPapers(data)
    } catch (err) {
      console.error('Failed to fetch papers:', err)
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
            </tr>
          </thead>
          <tbody>
            {filteredPapers.map(paper => (
              <tr
                key={paper.id}
                className="paper-row clickable"
                onClick={() => onSelectPaper(paper)}
                style={{ cursor: 'pointer' }}
              >
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