import React, { useState, useEffect, useCallback, useRef } from 'react'
import { GenaiPaper, PaperNote } from '../supabase'
import { PapersService } from '../services/papersService'

interface PapersListProps {
  onSelectPaper: (paper: GenaiPaper) => void
}

type PaperWithNote = GenaiPaper & { note?: PaperNote }

type RatingOption = 'highlight' | 'ok' | 'ignore' | 'unrated'

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// Default: show all except ignored
const DEFAULT_RATING_FILTERS: Set<RatingOption> = new Set<RatingOption>(['highlight', 'ok', 'unrated'])

// localStorage keys
const STORAGE_KEYS = {
  searchTerm: 'papers-filter-search',
  fileKind: 'papers-filter-fileKind',
  year: 'papers-filter-year',
  month: 'papers-filter-month',
  ratings: 'papers-filter-ratings',
}

const loadRatingFilters = (): Set<RatingOption> => {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.ratings)
    if (stored) {
      const arr = JSON.parse(stored) as RatingOption[]
      return new Set<RatingOption>(arr)
    }
  } catch (e) {
    console.error('Failed to load rating filters:', e)
  }
  return DEFAULT_RATING_FILTERS
}

const PapersList: React.FC<PapersListProps> = ({ onSelectPaper }) => {
  const [papers, setPapers] = useState<PaperWithNote[]>([])
  const [filteredPapers, setFilteredPapers] = useState<PaperWithNote[]>([])
  const [searchTerm, setSearchTerm] = useState(() => localStorage.getItem(STORAGE_KEYS.searchTerm) || '')
  const [fileKindFilter, setFileKindFilter] = useState<string>(() => localStorage.getItem(STORAGE_KEYS.fileKind) || '')
  const [yearFilter, setYearFilter] = useState<string>(() => localStorage.getItem(STORAGE_KEYS.year) || '')
  const [monthFilter, setMonthFilter] = useState<string>(() => localStorage.getItem(STORAGE_KEYS.month) || '')
  const [ratingFilters, setRatingFilters] = useState<Set<RatingOption>>(loadRatingFilters)
  const [showRatingDropdown, setShowRatingDropdown] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [ftsMatchingIds, setFtsMatchingIds] = useState<Set<string> | null>(null)
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const ratingDropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchPapers()
  }, [])

  // Persist filters to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.searchTerm, searchTerm)
  }, [searchTerm])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.fileKind, fileKindFilter)
  }, [fileKindFilter])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.year, yearFilter)
  }, [yearFilter])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.month, monthFilter)
  }, [monthFilter])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.ratings, JSON.stringify(Array.from(ratingFilters)))
  }, [ratingFilters])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ratingDropdownRef.current && !ratingDropdownRef.current.contains(event.target as Node)) {
        setShowRatingDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const fetchPapers = async () => {
    try {
      const data = await PapersService.getAllPapersWithNotes()
      setPapers(data)
    } catch (err) {
      console.error('Failed to fetch papers:', err)
    }
  }

  // Debounced full-text search
  const performFTSSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setFtsMatchingIds(null)
      setIsSearching(false)
      return
    }

    setIsSearching(true)
    try {
      const matchingIds = await PapersService.searchPaperIds(query)
      setFtsMatchingIds(new Set(matchingIds))
    } catch (err) {
      console.error('Full-text search failed:', err)
      setFtsMatchingIds(null)
    } finally {
      setIsSearching(false)
    }
  }, [])

  // Handle search term changes with debounce
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }

    if (searchTerm.trim()) {
      setIsSearching(true)
      searchTimeoutRef.current = setTimeout(() => {
        performFTSSearch(searchTerm)
      }, 300)
    } else {
      setFtsMatchingIds(null)
      setIsSearching(false)
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current)
      }
    }
  }, [searchTerm, performFTSSearch])

  const filterPapers = useCallback(() => {
    let filtered = papers

    // Full-text search filter
    if (searchTerm && ftsMatchingIds !== null) {
      // Filter by FTS results
      filtered = filtered.filter(paper => ftsMatchingIds.has(paper.id))
    } else if (searchTerm && ftsMatchingIds === null && !isSearching) {
      // Fallback to metadata search if FTS hasn't run yet or returned nothing
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(paper =>
        (paper.title && paper.title.toLowerCase().includes(term)) ||
        (paper.authors && paper.authors.toLowerCase().includes(term)) ||
        (paper.venue && paper.venue.toLowerCase().includes(term)) ||
        (paper.application && paper.application.toLowerCase().includes(term)) ||
        (paper.source_url && paper.source_url.toLowerCase().includes(term))
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

    if (monthFilter) {
      filtered = filtered.filter(paper =>
        paper.month && paper.month.toString() === monthFilter
      )
    }

    // Rating filter (multi-select)
    if (ratingFilters.size < 4) { // If not all selected, apply filter
      filtered = filtered.filter(paper => {
        const paperRating = paper.note?.rating
        if (!paperRating) {
          return ratingFilters.has('unrated')
        }
        return ratingFilters.has(paperRating as RatingOption)
      })
    }

    // Sort by import date (created_at) descending - most recently imported first
    filtered = [...filtered].sort((a, b) => {
      const dateA = new Date(a.created_at).getTime()
      const dateB = new Date(b.created_at).getTime()
      return dateB - dateA
    })

    setFilteredPapers(filtered)
  }, [papers, searchTerm, fileKindFilter, yearFilter, monthFilter, ratingFilters, ftsMatchingIds, isSearching])

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

  const getUniqueMonths = () => {
    const months = papers
      .filter(paper => paper.month)
      .map(paper => paper.month!)
      .filter((month, index, arr) => arr.indexOf(month) === index)
      .sort((a, b) => a - b)
    return months
  }

  const formatDate = (paper: PaperWithNote): string => {
    // Prefer published_at if available
    if (paper.published_at) {
      const date = new Date(paper.published_at)
      const month = date.getUTCMonth()
      const year = date.getUTCFullYear()
      return `${MONTH_NAMES[month]} ${year}`
    }
    // Fall back to month/year fields
    if (!paper.year) return '-'
    if (paper.month && paper.month >= 1 && paper.month <= 12) {
      return `${MONTH_NAMES[paper.month - 1]} ${paper.year}`
    }
    return paper.year.toString()
  }

  const getRatingBadge = (rating: string | null | undefined) => {
    if (!rating) return null

    switch (rating) {
      case 'highlight':
        return <span className="rating-badge highlight" title="Highlight">★</span>
      case 'ok':
        return <span className="rating-badge ok" title="Ok">●</span>
      case 'ignore':
        return <span className="rating-badge ignore" title="Ignore">✕</span>
      default:
        return null
    }
  }

  const toggleRatingFilter = (rating: RatingOption) => {
    setRatingFilters(prev => {
      const next = new Set(prev)
      if (next.has(rating)) {
        next.delete(rating)
      } else {
        next.add(rating)
      }
      return next
    })
  }

  const getRatingFilterLabel = () => {
    if (ratingFilters.size === 4) return 'All ratings'
    if (ratingFilters.size === 0) return 'No ratings'
    const labels: string[] = []
    if (ratingFilters.has('highlight')) labels.push('Highlight')
    if (ratingFilters.has('ok')) labels.push('Ok')
    if (ratingFilters.has('ignore')) labels.push('Ignore')
    if (ratingFilters.has('unrated')) labels.push('Unrated')
    return labels.join(', ')
  }

  return (
    <div className="papers-list">
      <div className="filters">
        <div className="search-wrapper">
          <input
            type="text"
            placeholder="Search paper content..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          {isSearching && <span className="search-indicator">Searching...</span>}
        </div>

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

        <select
          value={monthFilter}
          onChange={(e) => setMonthFilter(e.target.value)}
          className="filter-select"
        >
          <option value="">All months</option>
          {getUniqueMonths().map(month => (
            <option key={month} value={month.toString()}>{MONTH_NAMES[month - 1]}</option>
          ))}
        </select>

        <div className="rating-filter-dropdown" ref={ratingDropdownRef}>
          <button
            className="rating-filter-button"
            onClick={() => setShowRatingDropdown(!showRatingDropdown)}
          >
            {getRatingFilterLabel()}
            <span className="dropdown-arrow">▼</span>
          </button>
          {showRatingDropdown && (
            <div className="rating-filter-menu">
              <label className="rating-filter-option">
                <input
                  type="checkbox"
                  checked={ratingFilters.has('highlight')}
                  onChange={() => toggleRatingFilter('highlight')}
                />
                <span className="rating-label highlight">★ Highlight</span>
              </label>
              <label className="rating-filter-option">
                <input
                  type="checkbox"
                  checked={ratingFilters.has('ok')}
                  onChange={() => toggleRatingFilter('ok')}
                />
                <span className="rating-label ok">● Ok</span>
              </label>
              <label className="rating-filter-option">
                <input
                  type="checkbox"
                  checked={ratingFilters.has('ignore')}
                  onChange={() => toggleRatingFilter('ignore')}
                />
                <span className="rating-label ignore">✕ Ignore</span>
              </label>
              <label className="rating-filter-option">
                <input
                  type="checkbox"
                  checked={ratingFilters.has('unrated')}
                  onChange={() => toggleRatingFilter('unrated')}
                />
                <span className="rating-label unrated">○ Unrated</span>
              </label>
            </div>
          )}
        </div>
      </div>

      <div className="papers-table-container">
        <table className="papers-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Authors</th>
              <th>Date</th>
              <th>Venue</th>
              <th>Application</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            {filteredPapers.map(paper => (
              <tr
                key={paper.id}
                className={`paper-row clickable ${paper.note?.rating || ''}`}
                onClick={() => onSelectPaper(paper)}
                style={{ cursor: 'pointer' }}
              >
                <td className="paper-title">
                  {getRatingBadge(paper.note?.rating)}
                  <strong>{paper.title || 'Untitled'}</strong>
                </td>
                <td className="paper-authors">
                  {paper.authors || 'Unknown'}
                </td>
                <td className="paper-date">
                  {formatDate(paper)}
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
