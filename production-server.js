const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const csv = require('csv-parser');
const { Readable } = require('stream');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Cache
let cachedData = null;
let cacheTimestamp = null;
const CACHE_DURATION = 5 * 60 * 1000;

const SHEETS_CSV_URL = 'https://docs.google.com/spreadsheets/d/1x4_HTxZSEcLkGqStB7Ee8etjqtiEdKCksfa1YJyf2zo/export?format=csv&gid=0';

function parseCSVData(csvText) {
  return new Promise((resolve, reject) => {
    const results = [];
    const stream = Readable.from([csvText]);
    
    stream
      .pipe(csv())
      .on('data', (data) => {
        results.push({
          id: data.id || '',
          title: data.title || '',
          url: data.url || '',
          authors: data.authors || '',
          month: parseInt(data.month) || 0,
          year: parseInt(data.year) || 0,
          venue: data.venue || '',
          application: data.application || '',
          users: data.users || '',
          ages: data.ages || '',
          why: data.why || '',
          study_design: data.study_design || '',
          page: data.page || '',
          scraped_at: data.scraped_at || '',
          request_id: data.request_id || '',
          markdown: data.markdown || '',
          error: data.error || '',
          result: data.result || ''
        });
      })
      .on('end', () => resolve(results))
      .on('error', reject);
  });
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.get('/api/sheets', async (req, res) => {
  try {
    if (cachedData && cacheTimestamp && Date.now() - cacheTimestamp < CACHE_DURATION) {
      console.log('Returning cached data');
      return res.json(cachedData);
    }

    console.log('Fetching fresh data from Google Sheets...');
    const response = await fetch(SHEETS_CSV_URL, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; PapersApp/1.0)'
      }
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const csvText = await response.text();
    console.log('CSV data fetched, length:', csvText.length);
    
    const parsedData = await parseCSVData(csvText);
    console.log('Parsed', parsedData.length, 'records');

    cachedData = parsedData;
    cacheTimestamp = Date.now();

    res.json(parsedData);
  } catch (error) {
    console.error('Error fetching sheets data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch sheets data',
      message: error.message 
    });
  }
});

app.post('/api/cache/clear', (req, res) => {
  cachedData = null;
  cacheTimestamp = null;
  res.json({ message: 'Cache cleared' });
});

// Serve React static files
app.use(express.static(path.join(__dirname, 'build')));

// Handle React routing - serve index.html for all non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📄 Sheets data: http://localhost:${PORT}/api/sheets`);
});