import React, { useState } from 'react';
import './App.css';
import PapersList from './components/PapersList';
import PDFViewer from './components/PDFViewer';
import { GenaiPaper } from './supabase';

function App() {
  const [selectedPaper, setSelectedPaper] = useState<GenaiPaper | null>(null);

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

export default App;
