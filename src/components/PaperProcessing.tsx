import React, { useState, useEffect } from 'react'
import { supabase, Paper } from '../supabase'
import { useAuth } from '../contexts/AuthContext'
import config from '../config'

const PaperProcessing: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)
  const [processingPaper, setProcessingPaper] = useState<string | null>(null)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const { user } = useAuth()

  useEffect(() => {
    fetchPapers()
  }, [])

  const fetchPapers = async () => {
    setLoading(true)
    try {
      const { data, error } = await supabase
        .from('papers')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50)

      if (error) throw error
      setPapers(data || [])
    } catch (error) {
      console.error('Failed to fetch papers:', error)
      setMessage({ type: 'error', text: 'Failed to load papers' })
    } finally {
      setLoading(false)
    }
  }

  const extractPaperContent = async (paperId: string) => {
    setProcessingPaper(paperId)
    setMessage(null)

    try {
      // Call local Python backend instead of Edge Function
      const response = await fetch(`${config.backendUrl}/extract`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ paper_id: paperId }),
      })

      const result = await response.json()

      if (!response.ok || !result.success) {
        throw new Error(result.error || result.detail || 'Extraction failed')
      }

      setMessage({
        type: 'success',
        text: `Successfully extracted ${result.sections_count} sections and ${result.images_count} images`,
      })

      // Refresh papers list
      await fetchPapers()
    } catch (error) {
      console.error('Extraction error:', error)
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Extraction failed',
      })
    } finally {
      setProcessingPaper(null)
    }
  }

  if (!user) {
    return (
      <div className="paper-processing">
        <p>Please log in as admin to process papers</p>
      </div>
    )
  }

  if (loading) {
    return <div className="paper-processing">Loading papers...</div>
  }

  return (
    <div className="paper-processing">
      <div className="processing-header">
        <h2>Paper Processing</h2>
        <button onClick={fetchPapers} className="refresh-btn">
          Refresh
        </button>
      </div>

      {message && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="papers-table-container">
        <table className="papers-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Source Type</th>
              <th>Status</th>
              <th>Processed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {papers.map((paper) => (
              <tr key={paper.id}>
                <td className="paper-title-cell">
                  {paper.title || 'Untitled'}
                </td>
                <td>{paper.source_type || 'unknown'}</td>
                <td>
                  <span className={`status-badge ${paper.processing_status || 'pending'}`}>
                    {paper.processing_status || 'pending'}
                  </span>
                </td>
                <td>
                  {paper.processed_at
                    ? new Date(paper.processed_at).toLocaleDateString()
                    : '-'}
                </td>
                <td>
                  <button
                    onClick={() => extractPaperContent(paper.id)}
                    disabled={
                      processingPaper === paper.id ||
                      paper.processing_status === 'processing'
                    }
                    className="extract-btn"
                  >
                    {processingPaper === paper.id
                      ? 'Processing...'
                      : paper.processing_status === 'completed'
                      ? 'Re-extract'
                      : 'Extract'}
                  </button>
                  {paper.processing_error && (
                    <div className="error-tooltip" title={paper.processing_error}>
                      ⚠️
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {papers.length === 0 && (
        <p className="no-papers">No papers found</p>
      )}
    </div>
  )
}

export default PaperProcessing
