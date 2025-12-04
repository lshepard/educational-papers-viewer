import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import PapersList from './components/PapersList';
import PDFViewer from './components/PDFViewer';
import AdminLogin from './components/AdminLogin';
import AdminDashboard from './components/AdminDashboard';
import PaperProcessing from './components/PaperProcessing';
import PodcastManager from './components/PodcastManager';
import PodcastCreator from './components/podcast/PodcastCreator';
import PaperImport from './components/PaperImport';
import { GenaiPaper } from './supabase';
import { AuthProvider, useAuth } from './contexts/AuthContext';

function MainApp() {
  const [selectedPaper, setSelectedPaper] = useState<GenaiPaper | null>(null);
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
          {user && (
            <div className="admin-controls">
              <Link to="/admin" className="admin-badge">
                Admin
              </Link>
              <button onClick={signOut} className="sign-out-btn">
                Sign Out
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="App-main">
        {selectedPaper ? (
          <PDFViewer
            paper={selectedPaper}
            onClose={handleClosePaper}
          />
        ) : (
          <PapersList onSelectPaper={handleSelectPaper} />
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
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route path="/admin/processing" element={<PaperProcessing />} />
          <Route path="/admin/podcast-manager" element={<PodcastManager />} />
          <Route path="/admin/podcast-creator" element={<PodcastCreator />} />
          <Route path="/admin/import" element={<PaperImport />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
