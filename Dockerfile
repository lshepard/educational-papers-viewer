# Use Node.js LTS version
FROM node:18-alpine AS base

# Set working directory
WORKDIR /app

# Copy package files for both frontend and API
COPY package*.json ./
COPY api/package*.json ./api/

# Install dependencies for both frontend and API
RUN npm ci --only=production
RUN cd api && npm ci --only=production

# Build stage for React app
FROM base AS build

# Copy source code
COPY . .

# Build the React app with production environment
RUN REACT_APP_API_URL="" npm run build

# Production stage
FROM node:18-alpine AS production

# Set working directory
WORKDIR /app

# Copy package files and install production dependencies
COPY api/package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy API server code
COPY api/server.js ./server.js

# Copy built React app from build stage
COPY --from=build /app/build ./public

# Create a combined server that serves both API and React app
RUN cat > combined-server.js << 'EOF'
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const csv = require('csv-parser');
const { Readable } = require('stream');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Enable CORS
app.use(cors({
  origin: true,
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

// API Routes
app.get('/api/sheets', async (req, res) => {
  try {
    if (cachedData && cacheTimestamp && Date.now() - cacheTimestamp < CACHE_DURATION) {
      console.log('Returning cached data');
      return res.json(cachedData);
    }

    console.log('Fetching fresh data from Google Sheets...');
    
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

    const parsedData = await parseCSVData(csvText);
    console.log('Parsed', parsedData.length, 'records');

    cachedData = parsedData;
    cacheTimestamp = Date.now();

    res.json(parsedData);
  } catch (error) {
    console.error('Error fetching sheets data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch sheets data', 
      message: error.message,
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
        page: "", scraped_at: "", request_id: "", markdown: "", error: "", result: ""
      }]
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.post('/api/cache/clear', (req, res) => {
  cachedData = null;
  cacheTimestamp = null;
  res.json({ message: 'Cache cleared' });
});

// Serve static files from React build
app.use(express.static(path.join(__dirname, 'public')));

// Handle React routing - serve index.html for all non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`API Health check: http://0.0.0.0:${PORT}/api/health`);
  console.log(`Sheets data: http://0.0.0.0:${PORT}/api/sheets`);
});
EOF

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3001/api/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) }).on('error', () => { process.exit(1) })"

# Start the combined server
CMD ["node", "combined-server.js"]