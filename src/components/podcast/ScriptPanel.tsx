import React from 'react'
import { usePodcastCreation } from '../../contexts/PodcastCreationContext'
import './ScriptPanel.css'

const ScriptPanel: React.FC = () => {
  const {
    selectedSources,
    config,
    updateConfig,
    generatePodcast,
    isGeneratingScript,
    isGeneratingAudio,
    audioUrl,
    episodeId,
  } = usePodcastCreation()

  const isGenerating = isGeneratingScript || isGeneratingAudio

  return (
    <div className="script-panel">
      {/* Episode Configuration */}
      <div className="config-section">
        <h2>Episode Configuration</h2>

        <div className="config-form">
          <div className="form-group">
            <label>Theme / Topic *</label>
            <input
              type="text"
              value={config.theme}
              onChange={(e) => updateConfig({ theme: e.target.value })}
              placeholder="e.g., 'The evolution of transformer architectures'"
              className="config-input"
            />
            <span className="helper-text">
              What's the main theme or angle for this episode?
            </span>
          </div>

          <div className="form-group">
            <label>Custom Title (Optional)</label>
            <input
              type="text"
              value={config.title}
              onChange={(e) => updateConfig({ title: e.target.value })}
              placeholder="Leave empty to auto-generate"
              className="config-input"
            />
            <span className="helper-text">
              AI will generate an engaging title if left blank
            </span>
          </div>

          <div className="form-group">
            <label>Custom Description (Optional)</label>
            <textarea
              value={config.description}
              onChange={(e) => updateConfig({ description: e.target.value })}
              placeholder="Leave empty to auto-generate"
              className="config-textarea"
              rows={3}
            />
            <span className="helper-text">
              AI will generate a compelling description if left blank
            </span>
          </div>
        </div>
      </div>

      {/* Sources Summary */}
      <div className="sources-summary-section">
        <h3>Selected Sources ({selectedSources.length})</h3>
        {selectedSources.length === 0 ? (
          <p className="empty-state-small">No sources selected yet</p>
        ) : (
          <div className="sources-summary">
            {selectedSources.map((source) => (
              <div key={source.id} className="source-summary-item">
                <span className="source-icon">📄</span>
                <div className="source-summary-content">
                  <div className="source-summary-title">{source.title}</div>
                  <div className="source-summary-meta">
                    {source.authors} {source.year && `(${source.year})`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Generation */}
      <div className="generation-section">
        <button
          onClick={generatePodcast}
          disabled={isGenerating || selectedSources.length === 0 || !config.theme.trim()}
          className="generate-btn"
        >
          {isGenerating ? (
            <>
              <span className="spinner">⏳</span>
              Generating... (5-10 min)
            </>
          ) : (
            <>
              🎙️ Generate Podcast Episode
            </>
          )}
        </button>

        {selectedSources.length === 0 && (
          <p className="warning-text">Select at least one source to generate</p>
        )}

        {!config.theme.trim() && selectedSources.length > 0 && (
          <p className="warning-text">Enter a theme for the episode</p>
        )}

        {isGenerating && (
          <div className="progress-info">
            <p>Generating your podcast episode...</p>
            <ul>
              <li>✓ Analyzing selected papers</li>
              <li>✓ Generating engaging script</li>
              <li>⏳ Creating audio with AI voice...</li>
            </ul>
          </div>
        )}
      </div>

      {/* Audio Player */}
      {audioUrl && episodeId && (
        <div className="audio-section">
          <h3>✨ Episode Generated!</h3>
          <audio controls className="audio-player" src={audioUrl}>
            Your browser does not support the audio element.
          </audio>
          <div className="audio-actions">
            <a
              href={audioUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="download-btn"
            >
              📥 Download MP3
            </a>
            <button
              onClick={() => window.location.href = '/admin/podcast-manager'}
              className="view-episodes-btn"
            >
              📚 View All Episodes
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ScriptPanel
