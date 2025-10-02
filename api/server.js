const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const csv = require('csv-parser');
const { Readable } = require('stream');

const app = express();
const PORT = process.env.PORT || 3001;

// Enable CORS for all routes
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true
}));

app.use(express.json());

// Cache for storing sheets data
let cachedData = null;
let cacheTimestamp = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Google Sheets CSV URL
const SHEETS_CSV_URL = 'https://docs.google.com/spreadsheets/d/1x4_HTxZSEcLkGqStB7Ee8etjqtiEdKCksfa1YJyf2zo/export?format=csv&gid=0';

// Function to parse CSV data
function parseCSVData(csvText) {
  return new Promise((resolve, reject) => {
    const results = [];
    const stream = Readable.from([csvText]);
    
    stream
      .pipe(csv())
      .on('data', (data) => {
        // Clean up the data and ensure consistent property names
        const cleanedData = {
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
        };
        results.push(cleanedData);
      })
      .on('end', () => {
        resolve(results);
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// Endpoint to fetch sheets data
app.get('/api/sheets', async (req, res) => {
  try {
    // Check if we have cached data that's still fresh
    if (cachedData && cacheTimestamp && Date.now() - cacheTimestamp < CACHE_DURATION) {
      console.log('Returning cached data');
      return res.json(cachedData);
    }

    console.log('Fetching fresh data from Google Sheets...');
    
    // Fetch the CSV data
    const response = await fetch(SHEETS_CSV_URL, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const csvText = await response.text();
    console.log('CSV data fetched, length:', csvText.length);

    // Parse the CSV data
    const parsedData = await parseCSVData(csvText);
    console.log('Parsed', parsedData.length, 'records');

    // Cache the data
    cachedData = parsedData;
    cacheTimestamp = Date.now();

    res.json(parsedData);
  } catch (error) {
    console.error('Error fetching sheets data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch sheets data', 
      message: error.message,
      // Return mock data as fallback
      data: [{
        id: "/ai/repository/ends-tests-possibilities-transformative-assessment-and-learning-generative-ai",
        title: "The Ends Of Tests: Possibilities For Transformative Assessment And Learning With Generative AI",
        url: "https://scale.stanford.edu/ai/repository/ends-tests-possibilities-transformative-assessment-and-learning-generative-ai",
        authors: "Bill Cope, Mary Kalantzis, Akash Kumar Saini",
        month: 9,
        year: 2025,
        venue: "Unesco",
        application: "Assessment and Testing",
        users: "Student; Educator",
        ages: "Elementary (PK5); Middle School (6-8); High School (9-12); Post-Secondary; Adult",
        why: "Assessment transformation",
        study_design: "Theoretical Framework",
        page: "",
        scraped_at: "",
        request_id: "",
        markdown: "",
        error: "",
        result: ""
      }]
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Clear cache endpoint (useful for development)
app.post('/api/cache/clear', (req, res) => {
  cachedData = null;
  cacheTimestamp = null;
  res.json({ message: 'Cache cleared' });
});

// Start the server
app.listen(PORT, () => {
  console.log(`API server running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
  console.log(`Sheets data: http://localhost:${PORT}/api/sheets`);
});