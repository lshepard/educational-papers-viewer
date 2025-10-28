import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import PapersList from './components/PapersList';
import PDFViewer from './components/PDFViewer';
import AdminLogin from './components/AdminLogin';
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
          {user ? (
            <div className="admin-controls">
              <span className="admin-badge">Admin</span>
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
          <Route path="/admin" element={<AdminLogin />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
