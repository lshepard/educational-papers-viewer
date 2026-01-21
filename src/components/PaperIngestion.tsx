import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'
import config from '../config'

interface IngestionResult {
  success: boolean
  status: string
  pages_scanned: number
  papers_found: number
  papers_imported: number
  papers_skipped: number
  errors: string[]
}

const PaperIngestion: React.FC = () => {
  const [source, setSource] = useState('stanford_scale')
  const [maxPages, setMaxPages] = useState<number | ''>('')
  const [autoExtract, setAutoExtract] = useState(true)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<IngestionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { user } = useAuth()
  const navigate = useNavigate()

  const runIngestion = async () => {
    if (!window.confirm(
      'This will scrape the Stanford SCALE repository for new papers and import them. ' +
      'This may take several minutes. Continue?'
    )) {
      return
    }

    setIsRunning(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(`${config.backendUrl}/admin/run-ingestion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source,
          max_pages: maxPages === '' ? null : maxPages,
          auto_extract: autoExtract,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Ingestion failed')
      }

      setResult(data)
    } catch (err) {
      console.error('Ingestion error:', err)
      setError(err instanceof Error ? err.message : 'Ingestion failed')
    } finally {
      setIsRunning(false)
    }
  }

  if (!user) {
    return (
      <div className="paper-ingestion">
        <p>Please log in as admin to run paper ingestion</p>
      </div>
    )
  }

  return (
    <div className="paper-ingestion">
      <div className="ingestion-header">
        <button onClick={() => navigate('/admin')} className="back-btn">
          &larr; Back to Admin
        </button>
        <h2>Paper Ingestion</h2>
      </div>

      <div className="ingestion-description">
        <p>
          Automatically scrape papers from configured sources and import them into the database.
          The ingestion will stop early when it encounters papers that already exist.
        </p>
      </div>

      <div className="ingestion-controls">
        <div className="control-group">
          <label htmlFor="source">Source:</label>
          <select
            id="source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            disabled={isRunning}
          >
            <option value="stanford_scale">Stanford SCALE Repository</option>
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="maxPages">Max Pages (optional):</label>
          <input
            id="maxPages"
            type="number"
            min="1"
            value={maxPages}
            onChange={(e) => setMaxPages(e.target.value === '' ? '' : parseInt(e.target.value, 10))}
            placeholder="No limit"
            disabled={isRunning}
          />
          <span className="help-text">Limit pages to scan (for testing)</span>
        </div>

        <div className="control-group checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={autoExtract}
              onChange={(e) => setAutoExtract(e.target.checked)}
              disabled={isRunning}
            />
            Auto-extract content after import
          </label>
          <span className="help-text">Extract sections and images using Gemini (uses API credits)</span>
        </div>

        <button
          onClick={runIngestion}
          disabled={isRunning}
          className="run-ingestion-btn"
        >
          {isRunning ? 'Running Ingestion...' : 'Run Ingestion'}
        </button>
      </div>

      {isRunning && (
        <div className="ingestion-status running">
          <div className="spinner"></div>
          <p>Ingestion in progress... This may take several minutes.</p>
          <p className="status-hint">
            The process will stop automatically when it reaches papers that already exist.
          </p>
        </div>
      )}

      {error && (
        <div className="ingestion-status error">
          <h3>Ingestion Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className={`ingestion-status ${result.success ? 'success' : 'warning'}`}>
          <h3>Ingestion {result.status === 'completed' ? 'Complete' : 'Finished with Issues'}</h3>

          <div className="result-stats">
            <div className="stat">
              <span className="stat-value">{result.pages_scanned}</span>
              <span className="stat-label">Pages Scanned</span>
            </div>
            <div className="stat">
              <span className="stat-value">{result.papers_found}</span>
              <span className="stat-label">Papers Found</span>
            </div>
            <div className="stat highlight">
              <span className="stat-value">{result.papers_imported}</span>
              <span className="stat-label">Papers Imported</span>
            </div>
            <div className="stat">
              <span className="stat-value">{result.papers_skipped}</span>
              <span className="stat-label">Papers Skipped</span>
            </div>
          </div>

          {result.errors.length > 0 && (
            <div className="result-errors">
              <h4>Errors ({result.errors.length}):</h4>
              <ul>
                {result.errors.map((err, idx) => (
                  <li key={idx}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {result.papers_imported > 0 && (
            <div className="result-actions">
              <button onClick={() => navigate('/admin/processing')} className="view-papers-btn">
                View & Process Papers
              </button>
            </div>
          )}
        </div>
      )}

      <style>{`
        .paper-ingestion {
          padding: 20px;
          max-width: 800px;
          margin: 0 auto;
        }

        .ingestion-header {
          display: flex;
          align-items: center;
          gap: 20px;
          margin-bottom: 20px;
        }

        .ingestion-header h2 {
          margin: 0;
        }

        .back-btn {
          padding: 8px 16px;
          background: #f0f0f0;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
        }

        .back-btn:hover {
          background: #e0e0e0;
        }

        .ingestion-description {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .ingestion-description p {
          margin: 0;
          color: #666;
        }

        .ingestion-controls {
          background: white;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #ddd;
          margin-bottom: 20px;
        }

        .control-group {
          margin-bottom: 15px;
        }

        .control-group label {
          display: block;
          margin-bottom: 5px;
          font-weight: 500;
        }

        .control-group select,
        .control-group input[type="number"] {
          padding: 8px 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
          width: 100%;
          max-width: 300px;
        }

        .checkbox-group label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }

        .checkbox-group input[type="checkbox"] {
          width: 18px;
          height: 18px;
        }

        .help-text {
          display: block;
          font-size: 12px;
          color: #888;
          margin-top: 4px;
        }

        .run-ingestion-btn {
          margin-top: 10px;
          padding: 12px 24px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 16px;
          cursor: pointer;
        }

        .run-ingestion-btn:hover:not(:disabled) {
          background: #0056b3;
        }

        .run-ingestion-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .ingestion-status {
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .ingestion-status.running {
          background: #e3f2fd;
          border: 1px solid #90caf9;
          text-align: center;
        }

        .ingestion-status.success {
          background: #e8f5e9;
          border: 1px solid #a5d6a7;
        }

        .ingestion-status.warning {
          background: #fff3e0;
          border: 1px solid #ffcc80;
        }

        .ingestion-status.error {
          background: #ffebee;
          border: 1px solid #ef9a9a;
        }

        .ingestion-status h3 {
          margin-top: 0;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid #90caf9;
          border-top-color: #1976d2;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 15px;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .status-hint {
          font-size: 13px;
          color: #666;
        }

        .result-stats {
          display: flex;
          gap: 20px;
          margin: 20px 0;
          flex-wrap: wrap;
        }

        .stat {
          background: white;
          padding: 15px 20px;
          border-radius: 8px;
          text-align: center;
          min-width: 100px;
          border: 1px solid #ddd;
        }

        .stat.highlight {
          background: #e3f2fd;
          border-color: #90caf9;
        }

        .stat-value {
          display: block;
          font-size: 28px;
          font-weight: bold;
          color: #333;
        }

        .stat-label {
          display: block;
          font-size: 12px;
          color: #666;
          margin-top: 4px;
        }

        .result-errors {
          background: white;
          padding: 15px;
          border-radius: 8px;
          border: 1px solid #ef9a9a;
          margin-top: 15px;
        }

        .result-errors h4 {
          margin: 0 0 10px;
          color: #c62828;
        }

        .result-errors ul {
          margin: 0;
          padding-left: 20px;
        }

        .result-errors li {
          color: #666;
          font-size: 13px;
          margin-bottom: 5px;
        }

        .result-actions {
          margin-top: 20px;
        }

        .view-papers-btn {
          padding: 10px 20px;
          background: #4caf50;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .view-papers-btn:hover {
          background: #388e3c;
        }
      `}</style>
    </div>
  )
}

export default PaperIngestion
