import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import config from '../config';

const PaperImport: React.FC = () => {
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();
  const [url, setUrl] = useState('');
  const [autoExtract, setAutoExtract] = useState(true);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    if (!authLoading && !user) {
      navigate('/admin/login');
    }
  }, [user, authLoading, navigate]);

  const handleImport = async () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    setImporting(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${config.backendUrl}/papers/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url.trim(),
          auto_extract: autoExtract
        })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        setUrl('');
      } else {
        setError(data.detail || 'Import failed');
      }
    } catch (err: any) {
      console.error('Import failed:', err);
      setError(err.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  const handleExtract = async (paperId: string) => {
    try {
      const response = await fetch(`${config.backendUrl}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paperId })
      });

      const data = await response.json();

      if (data.success) {
        alert('Content extraction started! This may take a few minutes.');
      } else {
        alert('Failed to start extraction');
      }
    } catch (err) {
      console.error('Extraction failed:', err);
      alert('Failed to start extraction');
    }
  };

  if (authLoading || !user) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="paper-import">
      <div className="import-header">
        <h1>Import Paper</h1>
        <button onClick={() => navigate('/admin')} className="back-btn">
          ← Back to Dashboard
        </button>
      </div>

      <div className="import-form">
        <h2>Add Paper from URL</h2>
        <p className="description">
          Import papers from arXiv or direct PDF links. The system will automatically
          extract metadata and upload to storage.
        </p>

        <div className="form-section">
          <label>
            Paper URL *
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://arxiv.org/abs/2510.12915 or direct PDF URL"
              disabled={importing}
            />
          </label>

          <div className="checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={autoExtract}
                onChange={(e) => setAutoExtract(e.target.checked)}
                disabled={importing}
              />
              <span>Automatically extract content after import</span>
            </label>
            <small>
              Extract sections and images from the PDF using AI. Takes 1-2 minutes.
            </small>
          </div>

          <button
            onClick={handleImport}
            disabled={importing || !url.trim()}
            className="import-btn"
          >
            {importing ? 'Importing...' : 'Import Paper'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="success-message">
            <h3>✅ Paper Imported Successfully!</h3>
            <div className="result-details">
              <p><strong>Title:</strong> {result.title}</p>
              {result.authors && <p><strong>Authors:</strong> {result.authors}</p>}
              <p><strong>Paper ID:</strong> {result.paper_id}</p>
              <p>
                <a href={result.paper_url} target="_blank" rel="noopener noreferrer">
                  View Paper
                </a>
              </p>
            </div>

            <div className="result-actions">
              {!autoExtract && (
                <button
                  onClick={() => handleExtract(result.paper_id)}
                  className="extract-btn"
                >
                  Extract Content Now
                </button>
              )}
              <button onClick={() => navigate('/admin/processing')} className="view-btn">
                View in Processing Queue
              </button>
              <button onClick={() => setResult(null)} className="another-btn">
                Import Another Paper
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="examples-section">
        <h3>Supported URL Formats</h3>
        <ul>
          <li>
            <strong>arXiv Abstract:</strong>{' '}
            <code>https://arxiv.org/abs/2510.12915</code>
          </li>
          <li>
            <strong>arXiv PDF:</strong>{' '}
            <code>https://arxiv.org/pdf/2510.12915.pdf</code>
          </li>
          <li>
            <strong>Direct PDF:</strong> Any direct link to a PDF file
          </li>
        </ul>

        <div className="tip">
          <strong>💡 Tip:</strong> For arXiv papers, metadata (title, authors, abstract)
          is automatically fetched from the arXiv API. For other PDFs, metadata is
          extracted from the PDF file itself.
        </div>
      </div>

      <style>{`
        .paper-import {
          padding: 20px;
          max-width: 900px;
          margin: 0 auto;
        }

        .import-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
        }

        .import-header h1 {
          margin: 0;
        }

        .back-btn {
          background: #6c757d;
          color: white;
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .back-btn:hover {
          background: #5a6268;
        }

        .import-form {
          background: white;
          padding: 30px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          margin-bottom: 30px;
        }

        .import-form h2 {
          margin-top: 0;
        }

        .description {
          color: #666;
          margin-bottom: 25px;
        }

        .form-section {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .form-section label {
          display: flex;
          flex-direction: column;
          gap: 8px;
          font-weight: 500;
        }

        .form-section input[type="text"] {
          padding: 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
        }

        .form-section input[type="text"]:focus {
          outline: none;
          border-color: #4299e1;
          box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
        }

        .checkbox-group {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .checkbox-group label {
          flex-direction: row;
          align-items: center;
          gap: 10px;
          font-weight: normal;
        }

        .checkbox-group input[type="checkbox"] {
          width: auto;
        }

        .checkbox-group small {
          color: #666;
          margin-left: 28px;
        }

        .import-btn {
          background: #4299e1;
          color: white;
          padding: 12px 24px;
          border: none;
          border-radius: 6px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .import-btn:hover:not(:disabled) {
          background: #3182ce;
        }

        .import-btn:disabled {
          background: #cbd5e0;
          cursor: not-allowed;
        }

        .error-message {
          background: #fee;
          border: 1px solid #fcc;
          padding: 15px;
          border-radius: 6px;
          color: #c33;
          margin-top: 20px;
        }

        .success-message {
          background: #f0fff4;
          border: 2px solid #9ae6b4;
          padding: 25px;
          border-radius: 8px;
          margin-top: 20px;
        }

        .success-message h3 {
          color: #22543d;
          margin-top: 0;
        }

        .result-details {
          margin: 15px 0;
          padding: 15px;
          background: white;
          border-radius: 6px;
        }

        .result-details p {
          margin: 8px 0;
        }

        .result-details a {
          color: #4299e1;
          text-decoration: none;
        }

        .result-details a:hover {
          text-decoration: underline;
        }

        .result-actions {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 15px;
        }

        .result-actions button {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
        }

        .extract-btn {
          background: #48bb78;
          color: white;
        }

        .extract-btn:hover {
          background: #38a169;
        }

        .view-btn {
          background: #4299e1;
          color: white;
        }

        .view-btn:hover {
          background: #3182ce;
        }

        .another-btn {
          background: #e2e8f0;
          color: #2d3748;
        }

        .another-btn:hover {
          background: #cbd5e0;
        }

        .examples-section {
          background: #f7fafc;
          padding: 25px;
          border-radius: 8px;
        }

        .examples-section h3 {
          margin-top: 0;
        }

        .examples-section ul {
          list-style: none;
          padding: 0;
        }

        .examples-section li {
          margin-bottom: 12px;
        }

        .examples-section code {
          background: white;
          padding: 4px 8px;
          border-radius: 3px;
          font-size: 13px;
        }

        .tip {
          background: #bee3f8;
          padding: 15px;
          border-radius: 6px;
          margin-top: 20px;
        }

        .loading {
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
        }
      `}</style>
    </div>
  );
};

export default PaperImport;
