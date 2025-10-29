# Papers Viewer Backend

Python backend service for extracting content from research papers using Gemini AI.

## Features

- Extract paper sections (Introduction, Methods, Results, etc.)
- Extract and describe images, figures, charts, and tables
- Store extracted content in Supabase
- FastAPI REST API

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Install dependencies using uv:
```bash
cd backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Required environment variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key (not anon key!)
- `GEMINI_API_KEY` - Google Gemini API key

### Running the Server

```bash
./start.sh
```

Or directly:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /extract

Extract content from a paper.

**Request:**
```json
{
  "paper_id": "uuid-of-paper"
}
```

**Response:**
```json
{
  "success": true,
  "paper_id": "uuid",
  "sections_count": 7,
  "images_count": 12
}
```

### GET /health

Health check endpoint.

### GET /

API information and available endpoints.

## Development

The backend is built with:
- FastAPI for the web framework
- Google Generative AI (Gemini) for content extraction
- Supabase Python client for database access
- httpx for async HTTP requests

## Architecture

1. Client requests extraction via `/extract` endpoint
2. Backend fetches paper from Supabase
3. Downloads PDF from storage
4. Uploads to Gemini Files API
5. Runs two extraction passes:
   - Pass 1: Extract text sections
   - Pass 2: Extract and describe images
6. Stores results in `paper_sections` and `paper_images` tables
7. Updates paper processing status

## Notes

- The backend uses the Supabase service role key to bypass RLS policies
- Temporary files are cleaned up after processing
- Gemini files are deleted after extraction
- Processing status is tracked in the `papers` table
