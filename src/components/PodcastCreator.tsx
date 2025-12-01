import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './PodcastCreator.css';

interface PodcastSession {
  id: string;
  theme: string;
  description: string;
  resource_links: string[];
  status: string;
  current_step: number;
  show_notes: any;
}

interface ResearchMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
  bias_analysis?: any[];
}

interface YouTubeVideo {
  video_id: string;
  title: string;
  channel: string;
  duration_seconds: number;
  url: string;
  thumbnail: string;
  description: string;
}

interface Clip {
  id: string;
  video_id: string;
  video_title: string;
  channel_name: string;
  start_time_seconds: number;
  end_time_seconds: number;
  duration_seconds: number;
  clip_purpose: string;
  quote_text: string;
  status: string;
  audio_url?: string;
}

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const PodcastCreator: React.FC = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [session, setSession] = useState<PodcastSession | null>(null);
  const [messages, setMessages] = useState<ResearchMessage[]>([]);
  const [clips, setClips] = useState<Clip[]>([]);

  // Step 1: Theme and Resources
  const [theme, setTheme] = useState('');
  const [description, setDescription] = useState('');
  const [resourceLinks, setResourceLinks] = useState<string[]>(['']);

  // Step 2: Research Chat
  const [userMessage, setUserMessage] = useState('');
  const [isResearching, setIsResearching] = useState(false);

  // Step 3: YouTube Search
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<YouTubeVideo[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<YouTubeVideo | null>(null);
  const [clipStart, setClipStart] = useState(0);
  const [clipEnd, setClipEnd] = useState(10);
  const [clipPurpose, setClipPurpose] = useState('');
  const [clipQuote, setClipQuote] = useState('');

  // Step 4: Show Notes
  const [showNotes, setShowNotes] = useState<any>(null);
  const [isGeneratingNotes, setIsGeneratingNotes] = useState(false);

  const handleAddResourceLink = () => {
    setResourceLinks([...resourceLinks, '']);
  };

  const handleUpdateResourceLink = (index: number, value: string) => {
    const newLinks = [...resourceLinks];
    newLinks[index] = value;
    setResourceLinks(newLinks);
  };

  const handleRemoveResourceLink = (index: number) => {
    setResourceLinks(resourceLinks.filter((_, i) => i !== index));
  };

  const handleCreateSession = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/podcast-creator/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          theme,
          description,
          resource_links: resourceLinks.filter(link => link.trim() !== '')
        })
      });

      const data = await response.json();
      if (data.success) {
        setSession(data.session);
        setCurrentStep(2);
      }
    } catch (error) {
      console.error('Failed to create session:', error);
      alert('Failed to create session');
    }
  };

  const handleSendMessage = async () => {
    if (!session || !userMessage.trim()) return;

    setIsResearching(true);
    try {
      const response = await fetch(
        `${BACKEND_URL}/podcast-creator/sessions/${session.id}/research`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage })
        }
      );

      const data = await response.json();
      if (data.success) {
        // Reload messages
        await loadSession();
        setUserMessage('');
      }
    } catch (error) {
      console.error('Research failed:', error);
      alert('Research failed');
    } finally {
      setIsResearching(false);
    }
  };

  const handleSearchYouTube = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const response = await fetch(
        `${BACKEND_URL}/podcast-creator/sessions/${session!.id}/youtube/search`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: searchQuery, max_results: 10 })
        }
      );

      const data = await response.json();
      if (data.success) {
        setSearchResults(data.videos);
      }
    } catch (error) {
      console.error('YouTube search failed:', error);
      alert('YouTube search failed');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveClip = async () => {
    if (!session || !selectedVideo) return;

    try {
      const response = await fetch(
        `${BACKEND_URL}/podcast-creator/sessions/${session.id}/clips`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_id: selectedVideo.video_id,
            video_title: selectedVideo.title,
            channel_name: selectedVideo.channel,
            start_time: clipStart,
            end_time: clipEnd,
            clip_purpose: clipPurpose,
            quote_text: clipQuote
          })
        }
      );

      const data = await response.json();
      if (data.success) {
        await loadSession();
        setSelectedVideo(null);
        setClipPurpose('');
        setClipQuote('');
        alert('Clip saved! Audio extraction in progress...');
      }
    } catch (error) {
      console.error('Failed to save clip:', error);
      alert('Failed to save clip');
    }
  };

  const handleGenerateShowNotes = async () => {
    if (!session) return;

    setIsGeneratingNotes(true);
    try {
      const response = await fetch(
        `${BACKEND_URL}/podcast-creator/sessions/${session.id}/show-notes/generate`,
        {
          method: 'POST'
        }
      );

      const data = await response.json();
      if (data.success) {
        setShowNotes(data.show_notes);
        await loadSession();
      }
    } catch (error) {
      console.error('Failed to generate show notes:', error);
      alert('Failed to generate show notes');
    } finally {
      setIsGeneratingNotes(false);
    }
  };

  const loadSession = async () => {
    if (!session) return;

    try {
      const response = await fetch(
        `${BACKEND_URL}/podcast-creator/sessions/${session.id}`
      );
      const data = await response.json();

      if (data.success) {
        setSession(data.session);
        setMessages(data.messages);
        setClips(data.clips);
        if (data.session.show_notes) {
          setShowNotes(data.session.show_notes);
        }
      }
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  const renderStep1 = () => (
    <div className="step-content">
      <h2>Step 1: Theme & Resources</h2>
      <p className="step-description">
        Describe your podcast theme and add any resource links you want to reference.
      </p>

      <div className="form-group">
        <label>Podcast Theme *</label>
        <input
          type="text"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          placeholder="e.g., 'The future of AI in healthcare'"
          className="theme-input"
        />
      </div>

      <div className="form-group">
        <label>Description (optional)</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Provide more context about what you want to explore..."
          rows={4}
          className="description-textarea"
        />
      </div>

      <div className="form-group">
        <label>Resource Links (optional)</label>
        {resourceLinks.map((link, index) => (
          <div key={index} className="resource-link-row">
            <input
              type="url"
              value={link}
              onChange={(e) => handleUpdateResourceLink(index, e.target.value)}
              placeholder="https://..."
              className="resource-link-input"
            />
            {resourceLinks.length > 1 && (
              <button
                onClick={() => handleRemoveResourceLink(index)}
                className="remove-link-btn"
              >
                ✕
              </button>
            )}
          </div>
        ))}
        <button onClick={handleAddResourceLink} className="add-link-btn">
          + Add Another Link
        </button>
      </div>

      <button
        onClick={handleCreateSession}
        disabled={!theme.trim()}
        className="next-btn"
      >
        Start Research →
      </button>
    </div>
  );

  const renderStep2 = () => (
    <div className="step-content">
      <h2>Step 2: Research & Discussion</h2>
      <p className="step-description">
        Chat with AI to research your topic. The AI will use Perplexity to find sources
        and analyze them for bias.
      </p>

      <div className="research-chat">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Start by asking a research question...</p>
              <p className="hint">
                e.g., "What are the latest developments in {session?.theme}?"
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`message message-${msg.role}`}>
              <div className="message-content">{msg.content}</div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.map((source: any, idx: number) => (
                      <li key={idx}>
                        <a href={source.url} target="_blank" rel="noopener noreferrer">
                          {source.title || source.url}
                        </a>
                        {msg.bias_analysis?.[idx] && (
                          <span className={`bias-badge bias-${msg.bias_analysis[idx].bias_type}`}>
                            {msg.bias_analysis[idx].bias_type === 'corporate_self_promotion' && '⚠️ Promotional'}
                            {msg.bias_analysis[idx].bias_type === 'academic' && '✓ Academic'}
                            {msg.bias_analysis[idx].bias_type === 'neutral' && 'Neutral'}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}

          {isResearching && (
            <div className="message message-assistant">
              <div className="message-content loading">Researching...</div>
            </div>
          )}
        </div>

        <div className="chat-input-container">
          <textarea
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Ask a research question..."
            rows={3}
            className="chat-input"
            disabled={isResearching}
          />
          <button
            onClick={handleSendMessage}
            disabled={!userMessage.trim() || isResearching}
            className="send-btn"
          >
            {isResearching ? 'Researching...' : 'Send'}
          </button>
        </div>
      </div>

      <div className="step-actions">
        <button onClick={() => setCurrentStep(3)} className="next-btn">
          Continue to Clip Selection →
        </button>
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="step-content">
      <h2>Step 3: Find & Select Clips</h2>
      <p className="step-description">
        Search YouTube for videos and select clips to include in your podcast.
      </p>

      <div className="youtube-search">
        <div className="search-bar">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearchYouTube()}
            placeholder="Search YouTube videos..."
            className="search-input"
          />
          <button
            onClick={handleSearchYouTube}
            disabled={isSearching}
            className="search-btn"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        <div className="search-results">
          {searchResults.map((video) => (
            <div key={video.video_id} className="video-card">
              <img src={video.thumbnail} alt={video.title} className="video-thumbnail" />
              <div className="video-info">
                <h4>{video.title}</h4>
                <p className="video-channel">{video.channel}</p>
                <p className="video-description">{video.description}</p>
                <button
                  onClick={() => setSelectedVideo(video)}
                  className="select-video-btn"
                >
                  Select Clip from This Video
                </button>
              </div>
            </div>
          ))}
        </div>

        {selectedVideo && (
          <div className="clip-selector-modal">
            <div className="modal-content">
              <h3>Select Clip from: {selectedVideo.title}</h3>

              <div className="clip-timing">
                <label>
                  Start Time (seconds):
                  <input
                    type="number"
                    value={clipStart}
                    onChange={(e) => setClipStart(Number(e.target.value))}
                    min={0}
                    max={selectedVideo.duration_seconds}
                  />
                </label>

                <label>
                  End Time (seconds):
                  <input
                    type="number"
                    value={clipEnd}
                    onChange={(e) => setClipEnd(Number(e.target.value))}
                    min={clipStart}
                    max={selectedVideo.duration_seconds}
                  />
                </label>

                <p>Duration: {clipEnd - clipStart} seconds</p>
              </div>

              <div className="clip-context">
                <label>
                  What does this clip illustrate? *
                  <textarea
                    value={clipPurpose}
                    onChange={(e) => setClipPurpose(e.target.value)}
                    placeholder="e.g., 'Shows how the product works in practice'"
                    rows={2}
                  />
                </label>

                <label>
                  Quote/Content from clip (optional):
                  <textarea
                    value={clipQuote}
                    onChange={(e) => setClipQuote(e.target.value)}
                    placeholder="Transcribe key quote if applicable"
                    rows={2}
                  />
                </label>
              </div>

              <div className="modal-actions">
                <button onClick={() => setSelectedVideo(null)} className="cancel-btn">
                  Cancel
                </button>
                <button
                  onClick={handleSaveClip}
                  disabled={!clipPurpose.trim()}
                  className="save-clip-btn"
                >
                  Save Clip
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="selected-clips">
          <h3>Selected Clips ({clips.length})</h3>
          {clips.map((clip) => (
            <div key={clip.id} className="clip-item">
              <div className="clip-info">
                <strong>{clip.video_title}</strong>
                <p>{clip.clip_purpose}</p>
                <span className="clip-duration">
                  {clip.start_time_seconds}s - {clip.end_time_seconds}s
                  ({clip.duration_seconds}s)
                </span>
                <span className={`clip-status status-${clip.status}`}>
                  {clip.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="step-actions">
        <button onClick={() => setCurrentStep(2)} className="back-btn">
          ← Back to Research
        </button>
        <button onClick={() => setCurrentStep(4)} className="next-btn">
          Continue to Show Notes →
        </button>
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="step-content">
      <h2>Step 4: Review Show Notes</h2>
      <p className="step-description">
        Generate structured show notes with quotes and clip markers.
      </p>

      {!showNotes ? (
        <div className="generate-notes-section">
          <p>Ready to generate your show notes? This will create a structured outline based on your research and clips.</p>
          <button
            onClick={handleGenerateShowNotes}
            disabled={isGeneratingNotes}
            className="generate-btn"
          >
            {isGeneratingNotes ? 'Generating...' : 'Generate Show Notes'}
          </button>
        </div>
      ) : (
        <div className="show-notes-display">
          <h3>{showNotes.title}</h3>
          <p className="estimated-duration">
            Estimated Duration: {showNotes.estimated_duration_minutes} minutes
          </p>

          {showNotes.segments?.map((segment: any, idx: number) => (
            <div key={idx} className="segment">
              <h4>{segment.title}</h4>
              <p className="segment-duration">~{segment.duration_minutes} min</p>

              {segment.talking_points && (
                <div className="talking-points">
                  <strong>Talking Points:</strong>
                  <ul>
                    {segment.talking_points.map((point: string, i: number) => (
                      <li key={i}>{point}</li>
                    ))}
                  </ul>
                </div>
              )}

              {segment.quotes && segment.quotes.length > 0 && (
                <div className="quotes">
                  <strong>Quotes:</strong>
                  {segment.quotes.map((quote: any, i: number) => (
                    <blockquote key={i}>
                      "{quote.text}"
                      <cite>— {quote.source}</cite>
                      {quote.bias_note && (
                        <span className="bias-note">⚠️ {quote.bias_note}</span>
                      )}
                    </blockquote>
                  ))}
                </div>
              )}

              {segment.clips && segment.clips.length > 0 && (
                <div className="segment-clips">
                  <strong>Clips:</strong>
                  {segment.clips.map((clip: any, i: number) => (
                    <div key={i} className="clip-marker">
                      🎬 {clip.purpose} ({clip.play_at})
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="step-actions">
        <button onClick={() => setCurrentStep(3)} className="back-btn">
          ← Back to Clips
        </button>
        {showNotes && (
          <button className="next-btn" disabled>
            Production Mode (Coming Soon)
          </button>
        )}
      </div>
    </div>
  );

  return (
    <div className="podcast-creator">
      <header className="creator-header">
        <button onClick={() => navigate('/admin/podcast-manager')} className="back-link">
          ← Back to Podcast Manager
        </button>
        <h1>Podcast Creator Studio</h1>
        {session && <span className="session-id">Session: {session.id}</span>}
      </header>

      <div className="progress-indicator">
        <div className={`step-badge ${currentStep >= 1 ? 'active' : ''}`}>
          1. Theme
        </div>
        <div className={`step-badge ${currentStep >= 2 ? 'active' : ''}`}>
          2. Research
        </div>
        <div className={`step-badge ${currentStep >= 3 ? 'active' : ''}`}>
          3. Clips
        </div>
        <div className={`step-badge ${currentStep >= 4 ? 'active' : ''}`}>
          4. Show Notes
        </div>
      </div>

      <div className="creator-main">
        {currentStep === 1 && renderStep1()}
        {currentStep === 2 && renderStep2()}
        {currentStep === 3 && renderStep3()}
        {currentStep === 4 && renderStep4()}
      </div>
    </div>
  );
};

export default PodcastCreator;
