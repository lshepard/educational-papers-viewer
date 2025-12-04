import React from 'react'
import { useNavigate } from 'react-router-dom'
import { PodcastCreationProvider } from '../../contexts/PodcastCreationContext'
import SourcesPanel from './SourcesPanel'
import ScriptPanel from './ScriptPanel'
import './PodcastCreator.css'

const PodcastCreator: React.FC = () => {
  const navigate = useNavigate()

  return (
    <PodcastCreationProvider>
      <div className="podcast-creator">
        <div className="podcast-creator-header">
          <h1>Create Podcast Episode</h1>
          <button onClick={() => navigate('/admin/podcast-manager')} className="back-btn">
            ← Back to Manager
          </button>
        </div>

        <div className="podcast-creator-content">
          <div className="sources-pane">
            <SourcesPanel />
          </div>

          <div className="script-pane">
            <ScriptPanel />
          </div>
        </div>
      </div>
    </PodcastCreationProvider>
  )
}

export default PodcastCreator
