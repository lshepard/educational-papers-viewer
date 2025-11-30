import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'

interface PodcastEpisode {
  id: string
  paper_id: string
  title: string
  description: string | null
  script: string | null
  audio_url: string | null
  generation_status: 'pending' | 'processing' | 'completed' | 'failed'
  generation_error: string | null
  published_at: string
  created_at: string
}

const PodcastManager: React.FC = () => {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [episodes, setEpisodes] = useState<PodcastEpisode[]>([])
  const [loading, setLoading] = useState(true)
  const [editingEpisode, setEditingEpisode] = useState<PodcastEpisode | null>(null)
  const [editedScript, setEditedScript] = useState('')
  const [regenerating, setRegenerating] = useState<string | null>(null)

  useEffect(() => {
    if (!user) {
      navigate('/admin/login')
    } else {
      fetchEpisodes()
    }
  }, [user, navigate])

  const fetchEpisodes = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/podcast/episodes`)
      const data = await response.json()
      setEpisodes(data.episodes || [])
    } catch (error) {
      console.error('Failed to fetch episodes:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (episode: PodcastEpisode) => {
    setEditingEpisode(episode)
    setEditedScript(episode.script || '')
  }

  const handleSaveScript = async () => {
    if (!editingEpisode) return

    try {
      const response = await fetch(`${BACKEND_URL}/podcast/episodes/${editingEpisode.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: editedScript })
      })

      if (response.ok) {
        alert('Script saved! Click "Regenerate Audio" to create new audio from the edited script.')
        setEditingEpisode(null)
        fetchEpisodes()
      }
    } catch (error) {
      console.error('Failed to save script:', error)
      alert('Failed to save script')
    }
  }

  const handleRegenerateAudio = async (episodeId: string) => {
    if (!confirm('Regenerate audio from the current script? This will take 2-3 minutes.')) return

    setRegenerating(episodeId)
    try {
      const response = await fetch(`${BACKEND_URL}/podcast/episodes/${episodeId}/regenerate`, {
        method: 'POST'
      })

      const data = await response.json()
      if (data.success) {
        alert('Audio regenerated successfully!')
        fetchEpisodes()
        if (data.audio_url) {
          window.open(data.audio_url, '_blank')
        }
      }
    } catch (error) {
      console.error('Failed to regenerate audio:', error)
      alert('Failed to regenerate audio')
    } finally {
      setRegenerating(null)
    }
  }

  const handleDelete = async (episodeId: string) => {
    if (!confirm('Delete this episode? This cannot be undone.')) return

    try {
      const response = await fetch(`${BACKEND_URL}/podcast/episodes/${episodeId}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        fetchEpisodes()
      }
    } catch (error) {
      console.error('Failed to delete episode:', error)
      alert('Failed to delete episode')
    }
  }

  if (!user) return null
  if (loading) return <div className="loading">Loading episodes...</div>

  return (
    <div className="podcast-manager">
      <div className="podcast-manager-header">
        <h1>Podcast Manager</h1>
        <button onClick={() => navigate('/admin')} className="back-btn">
          ← Back to Dashboard
        </button>
      </div>

      <div className="podcast-feed-link">
        <h3>RSS Feed URL:</h3>
        <code>{BACKEND_URL}/podcast/feed.xml</code>
      </div>

      {editingEpisode ? (
        <div className="episode-editor">
          <h2>Edit Script: {editingEpisode.title}</h2>
          <textarea
            value={editedScript}
            onChange={(e) => setEditedScript(e.target.value)}
            rows={20}
            style={{ width: '100%', fontFamily: 'monospace', padding: '10px' }}
          />
          <div className="editor-actions">
            <button onClick={handleSaveScript} className="save-btn">
              Save Script
            </button>
            <button onClick={() => setEditingEpisode(null)} className="cancel-btn">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="episodes-list">
          <h2>Episodes ({episodes.length})</h2>
          {episodes.length === 0 ? (
            <p>No episodes yet. Generate one from the papers list!</p>
          ) : (
            <table className="episodes-table">
              <thead>
                <tr>
                  <th>Title</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {episodes.map(episode => (
                  <tr key={episode.id}>
                    <td>
                      <strong>{episode.title}</strong>
                      <br />
                      <small>{episode.description}</small>
                    </td>
                    <td>
                      <span className={`status-badge status-${episode.generation_status}`}>
                        {episode.generation_status}
                      </span>
                      {episode.generation_error && (
                        <div className="error-text">{episode.generation_error}</div>
                      )}
                    </td>
                    <td>{new Date(episode.created_at).toLocaleDateString()}</td>
                    <td>
                      <div className="action-buttons">
                        {episode.audio_url && (
                          <a
                            href={episode.audio_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="play-btn"
                          >
                            ▶ Play
                          </a>
                        )}
                        {episode.script && (
                          <button
                            onClick={() => handleEdit(episode)}
                            className="edit-btn"
                          >
                            Edit Script
                          </button>
                        )}
                        {episode.script && episode.generation_status === 'completed' && (
                          <button
                            onClick={() => handleRegenerateAudio(episode.id)}
                            disabled={regenerating === episode.id}
                            className="regenerate-btn"
                          >
                            {regenerating === episode.id ? 'Regenerating...' : '🔄 Regenerate Audio'}
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(episode.id)}
                          className="delete-btn"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <style>{`
        .podcast-manager {
          padding: 20px;
          max-width: 1400px;
          margin: 0 auto;
        }

        .podcast-manager-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
        }

        .podcast-feed-link {
          background: #f5f5f5;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 30px;
        }

        .podcast-feed-link code {
          background: white;
          padding: 8px;
          border-radius: 4px;
          display: inline-block;
          margin-top: 10px;
        }

        .episode-editor {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .editor-actions {
          margin-top: 15px;
          display: flex;
          gap: 10px;
        }

        .episodes-table {
          width: 100%;
          background: white;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .episodes-table th {
          background: #f5f5f5;
          padding: 12px;
          text-align: left;
          font-weight: 600;
        }

        .episodes-table td {
          padding: 12px;
          border-top: 1px solid #eee;
        }

        .status-badge {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .status-completed { background: #d4edda; color: #155724; }
        .status-processing { background: #fff3cd; color: #856404; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-pending { background: #e2e3e5; color: #383d41; }

        .action-buttons {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .action-buttons button, .action-buttons a {
          padding: 6px 12px;
          border-radius: 4px;
          font-size: 14px;
          cursor: pointer;
          text-decoration: none;
          border: none;
        }

        .play-btn {
          background: #28a745;
          color: white;
        }

        .edit-btn {
          background: #007bff;
          color: white;
        }

        .regenerate-btn {
          background: #17a2b8;
          color: white;
        }

        .regenerate-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .delete-btn {
          background: #dc3545;
          color: white;
        }

        .save-btn {
          background: #28a745;
          color: white;
          padding: 10px 20px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .cancel-btn {
          background: #6c757d;
          color: white;
          padding: 10px 20px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .back-btn {
          background: #6c757d;
          color: white;
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .error-text {
          color: #721c24;
          font-size: 12px;
          margin-top: 4px;
        }
      `}</style>
    </div>
  )
}

export default PodcastManager
