# Railway Deployment Guide

This guide explains how to deploy both the frontend and backend services to Railway.

## Overview

You'll need to deploy **two separate services** on Railway:

1. **Frontend** - React app with Express server (root directory)
2. **Backend** - Python FastAPI service (in `backend/` directory)

## Prerequisites

- Railway account
- Both services in the same Railway project
- Supabase database already set up

---

## Step 1: Deploy Backend Service

### Create New Service

1. In your Railway project, click **+ New**
2. Select **GitHub Repo** (or use existing repo)
3. If using the same repo:
   - Set **Root Directory** to `backend`
   - Railway will auto-detect the `backend/Dockerfile`

### Configure Backend Environment Variables

In the Backend service **Variables** tab, add:

```env
SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key-from-supabase
GEMINI_API_KEY=your-gemini-api-key
FRONTEND_URL=https://your-frontend.up.railway.app
PORT=8000
```

**Important Notes:**
- `SUPABASE_SERVICE_KEY` is the **service role key** (not anon key)
  - Get it from Supabase Dashboard → Settings → API → service_role key
- `FRONTEND_URL` should be your frontend's Railway URL (add this after deploying frontend)
- `PORT=8000` is the default, but Railway may override it

### Generate Domain

1. Go to Backend service → **Settings → Networking**
2. Click **Generate Domain**
3. Copy the URL (e.g., `https://backend-production-xxxx.up.railway.app`)
4. **Save this URL** - you'll need it for frontend configuration

---

## Step 2: Deploy/Update Frontend Service

### Add/Update Frontend Environment Variables

In your **Frontend service** → **Variables** tab, add:

```env
REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your-anon-key
REACT_APP_BACKEND_URL=https://backend-production-xxxx.up.railway.app
```

**Important Notes:**
- Use the backend URL from Step 1
- **No trailing slash** on `REACT_APP_BACKEND_URL`
- Use the **anon key** (not service role key)

### Redeploy Frontend

After adding `REACT_APP_BACKEND_URL`:

1. Click **Deployments** tab
2. Click the three dots on latest deployment → **Redeploy**

Or push a new commit to trigger rebuild.

---

## Step 3: Update Backend with Frontend URL

Now that your frontend is deployed:

1. Copy your frontend Railway URL from **Settings → Domains**
2. Go to Backend service → **Variables**
3. Set `FRONTEND_URL=https://your-frontend.up.railway.app`
4. Redeploy backend service

This enables CORS so the frontend can call the backend API.

---

## Verification

### 1. Test Backend Health

Visit: `https://your-backend.up.railway.app/health`

Expected response:
```json
{
  "success": true,
  "status": "healthy",
  "services": {
    "supabase": "connected",
    "gemini": "configured"
  }
}
```

### 2. Test Frontend

1. Visit your frontend Railway URL
2. Browse papers (should work without backend)
3. Sign in as admin
4. Open a paper
5. Click **Extract Content** button
6. Should successfully extract content and show images

---

## Troubleshooting

### CORS Errors

**Symptom:** Browser console shows CORS policy errors

**Fix:**
- Verify `FRONTEND_URL` in backend matches your frontend Railway URL exactly
- Ensure no trailing slash in URL
- Redeploy backend after changing `FRONTEND_URL`

### "supabaseUrl is required" Error

**Symptom:** Frontend shows "supabaseUrl is required" error

**Fix:**
- Verify all `REACT_APP_*` variables are set in frontend service
- Redeploy frontend (environment variables are embedded at build time)
- Check Railway build logs for errors

### Backend Health Check Fails

**Symptom:** `/health` endpoint returns 500 or unhealthy status

**Fix:**
- Verify `SUPABASE_URL` is correct
- Verify `SUPABASE_SERVICE_KEY` is the **service role key** (not anon)
- Check Railway logs: **Deployments → [latest] → View Logs**
- Test Supabase connection from Supabase dashboard

### Extract Content Button Does Nothing

**Symptom:** Button clicks but nothing happens

**Fix:**
- Open browser console and check for errors
- Verify `REACT_APP_BACKEND_URL` is set in frontend
- Verify backend service is running (check health endpoint)
- Check backend logs for errors
- Verify CORS is configured correctly

### Backend Build Fails

**Symptom:** Backend deployment fails during build

**Fix:**
- Check that `backend/Dockerfile` exists
- Verify root directory is set to `backend` in Railway settings
- Check Railway build logs for specific errors
- Verify `pyproject.toml` and `uv.lock` are committed

---

## Environment Variables Summary

### Frontend Service (Root Directory)
```env
REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
REACT_APP_SUPABASE_ANON_KEY=eyJhb... (anon key)
REACT_APP_BACKEND_URL=https://backend-production-xxxx.up.railway.app
```

### Backend Service (backend/ Directory)
```env
SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
SUPABASE_SERVICE_KEY=eyJhb... (service_role key)
GEMINI_API_KEY=AIza... (your Gemini API key)
FRONTEND_URL=https://frontend-production-xxxx.up.railway.app
PORT=8000
```

---

## Architecture

```
Railway Project
├── Frontend Service (root/)
│   ├── React App (build/ folder)
│   ├── Express Server (production-server.js)
│   ├── /api/sheets → Google Sheets proxy
│   ├── /api/health → Health check
│   └── Serves on PORT=3001
│
└── Backend Service (backend/)
    ├── FastAPI App (main.py)
    ├── /extract → PDF extraction with Gemini
    ├── /search → Full-text search
    ├── /health → Health check
    └── Serves on PORT=8000
```

---

## Local Development

### Frontend
```bash
npm start
# Runs on http://localhost:3000
# Uses REACT_APP_BACKEND_URL=http://localhost:8000
```

### Backend
```bash
cd backend
uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python main.py
# Runs on http://localhost:8000
```

### Environment Files

**Root `.env`:**
```env
REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your-anon-key
REACT_APP_BACKEND_URL=http://localhost:8000
```

**Backend `.env`:**
```env
SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-api-key
FRONTEND_URL=http://localhost:3000
```

---

## Notes

- Both services should be in the same Railway **project**
- Each service needs its own set of environment variables
- Backend uses **service role key** for admin access
- Frontend uses **anon key** for user access
- CORS is configured to allow cross-origin requests between services
- Changes to environment variables require a redeploy to take effect
- Backend extracts content using Gemini AI (requires API key)
- Backend processes PDF files and stores images in Supabase storage
