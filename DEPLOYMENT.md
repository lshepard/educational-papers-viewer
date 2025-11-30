# Deployment Configuration

## Backend URL Configuration

The frontend uses the `REACT_APP_BACKEND_URL` environment variable to configure where the backend API is located.

### Configuration Options

#### 1. Same Domain Deployment (Recommended)
Deploy frontend and backend on the same server/domain:

```bash
# .env.production
REACT_APP_BACKEND_URL=""
```

With this configuration:
- Frontend: `https://papers.yourdomain.com`
- Backend: `https://papers.yourdomain.com/podcast/generate` (relative paths)

#### 2. Separate API Domain
Deploy backend on a different domain:

```bash
# .env.production
REACT_APP_BACKEND_URL="https://api.yourdomain.com"
```

With this configuration:
- Frontend: `https://papers.yourdomain.com`
- Backend: `https://api.yourdomain.com/podcast/generate`

#### 3. Local Development
```bash
# .env.local (default)
REACT_APP_BACKEND_URL="http://localhost:8000"
```

## Quick Start for Same-Domain Deployment

1. **Build frontend with empty backend URL:**
   ```bash
   echo 'REACT_APP_BACKEND_URL=""' > .env.production
   npm run build
   ```

2. **Configure reverse proxy to route API calls:**
   - `/podcast/*` → backend:8000
   - `/extract`, `/search`, `/research/*` → backend:8000
   - `/*` → frontend static files

3. **Start backend:**
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

That's it! Both frontend and backend run from the same domain.

## Backend Deployment Options

### Option 1: Docker (Recommended for Production)

#### Build the Docker Image
```bash
cd backend
docker build -t papers-backend .
```

#### Run with Environment Variables

**Method 1: Using .env file (easiest)**
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your actual values

# Run container
docker run -p 8000:8000 --env-file .env papers-backend
```

**Method 2: Direct environment variables**
```bash
docker run -p 8000:8000 \
  -e SUPABASE_URL="https://your-project.supabase.co" \
  -e SUPABASE_SERVICE_KEY="your-service-role-key" \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  -e FRONTEND_URL="https://papers.yourdomain.com" \
  papers-backend
```

#### Required Environment Variables
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Supabase service role key (not anon key!)
- `GEMINI_API_KEY`: Google Gemini API key for AI features
- `FRONTEND_URL`: Frontend URL for CORS (optional, defaults to localhost:3000)

See `backend/.env.example` for additional optional variables like Google Cloud credentials.

### Option 2: Direct Python Execution

```bash
cd backend
# Create virtual environment and install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your actual values

# Run with uvicorn
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```
