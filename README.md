# Papers Viewer

A React application for viewing research papers with metadata from Google Sheets and PDF storage via Supabase.

## Features

- 📊 **Table-based UI**: Compact table view showing paper metadata
- 🔍 **Advanced Filtering**: Search and filter by title, authors, venue, year, and file type
- 📄 **PDF Viewer**: Built-in PDF viewer with navigation controls
- 🔗 **Google Sheets Integration**: Fetches paper metadata from Google Sheets via API proxy
- ☁️ **Supabase Storage**: Serves PDF files from Supabase storage
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Architecture

### Frontend (React + TypeScript)
- React application with TypeScript
- Table-based UI with filtering and search
- PDF viewer using react-pdf
- Responsive design with CSS

### Backend API (Node.js + Express)
- Express server that proxies Google Sheets CSV data
- Handles CORS issues with Google Sheets
- Includes caching for better performance
- Fallback data for when API is unavailable

### Data Sources
- **Google Sheets**: Source of truth for paper metadata
- **Supabase**: PDF file storage and additional paper data

## Quick Start

### Run both frontend and backend together (Recommended)
```bash
npm run start:dev
```
This will start:
- Backend API server on http://localhost:3001
- React frontend on http://localhost:3000

### Environment Setup
Your `.env` file should contain:
```
REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your_key_here
REACT_APP_API_URL=http://localhost:3001
```

## Available Scripts

### `npm run start:dev`
Runs both the backend API and React frontend simultaneously.

### `npm start`
Runs only the React app in development mode.

### `npm run start:api`
Runs only the backend API server.

### `npm run build`
Builds the app for production.

## API Endpoints

### Backend API (Port 3001)
- `GET /api/health` - Health check
- `GET /api/sheets` - Fetch Google Sheets data (with caching)
- `POST /api/cache/clear` - Clear cache (development)

## Data Flow

1. **Frontend** requests papers data from **Backend API**
2. **Backend API** fetches CSV data from **Google Sheets**
3. **Backend API** caches and returns parsed data
4. **Frontend** combines sheets data with **Supabase** storage info
5. **PDF Viewer** loads PDFs directly from **Supabase Storage**

## Troubleshooting

### API Server Issues
- Make sure backend is running on port 3001
- Check console for CORS errors
- Verify Google Sheets URL is accessible

### PDF Loading Issues
- Ensure Supabase storage bucket is publicly accessible
- Check that PDF files exist at the specified paths
- Verify PDF.js worker is loading correctly
