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
