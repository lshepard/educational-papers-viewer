import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import PapersList from './components/PapersList';
import SearchView from './components/SearchView';
import PDFViewer from './components/PDFViewer';
import AdminLogin from './components/AdminLogin';
import PaperProcessing from './components/PaperProcessing';
import { GenaiPaper } from './supabase';
import { AuthProvider, useAuth } from './contexts/AuthContext';

type ViewMode = 'browse' | 'search';

function MainApp() {
  const [selectedPaper, setSelectedPaper] = useState<GenaiPaper | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('browse');
  const { user, signOut } = useAuth();

  const handleSelectPaper = (paper: GenaiPaper) => {
    setSelectedPaper(paper);
  };

  const handleClosePaper = () => {
    setSelectedPaper(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>GenAI Papers Viewer</h1>
        <div className="header-actions">
          {user ? (
            <div className="admin-controls">
              <span className="admin-badge">Admin</span>
              <Link to="/admin/processing" className="admin-link">
                Processing
              </Link>
              <button onClick={signOut} className="sign-out-btn">
                Sign Out
              </button>
            </div>
          ) : (
            <Link to="/admin" className="admin-link">
              Admin
            </Link>
          )}
        </div>
      </header>

      {!selectedPaper && (
        <nav className="view-tabs">
          <button
            className={`tab-button ${viewMode === 'browse' ? 'active' : ''}`}
            onClick={() => setViewMode('browse')}
          >
            Browse Papers
          </button>
          <button
            className={`tab-button ${viewMode === 'search' ? 'active' : ''}`}
            onClick={() => setViewMode('search')}
          >
            Search Content
          </button>
        </nav>
      )}

      <main className="App-main">
        {selectedPaper ? (
          <PDFViewer
            paper={selectedPaper}
            onClose={handleClosePaper}
          />
        ) : viewMode === 'browse' ? (
          <PapersList onSelectPaper={handleSelectPaper} />
        ) : (
          <SearchView onSelectPaper={handleSelectPaper} />
        )}
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<MainApp />} />
          <Route path="/admin" element={<AdminLogin />} />
          <Route path="/admin/processing" element={<PaperProcessing />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
