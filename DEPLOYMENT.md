# Railway Deployment Guide

This guide explains how to deploy the Papers Viewer application to Railway.

## 🚀 Quick Deploy

### Option 1: Deploy from GitHub

1. **Push to GitHub**: Commit and push your code to a GitHub repository
2. **Connect to Railway**: Go to [railway.app](https://railway.app) and connect your GitHub repo
3. **Set Environment Variables**: In Railway dashboard, add these variables:
   ```
   REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
   REACT_APP_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh3bW11anN1YmNja3liemtndmlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkxNDU1NTIsImV4cCI6MjA3NDcyMTU1Mn0.tk-BD7MledKOkFOrKJydu2RNUyy9zk8oGhIHr4rHKvM
   ```
4. **Deploy**: Railway will automatically build and deploy using the Dockerfile

### Option 2: Railway CLI

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Initialize**:
   ```bash
   railway login
   railway init
   ```

3. **Set Environment Variables**:
   ```bash
   railway variables set REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co
   railway variables set REACT_APP_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh3bW11anN1YmNja3liemtndmlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkxNDU1NTIsImV4cCI6MjA3NDcyMTU1Mn0.tk-BD7MledKOkFOrKJydu2RNUyy9zk8oGhIHr4rHKvM
   ```

4. **Deploy**:
   ```bash
   railway up
   ```

## 🏗️ How It Works

The Docker setup:

1. **Multi-stage Build**: 
   - Builds the React app in production mode
   - Creates a combined Node.js server

2. **Combined Server**:
   - Serves the React app as static files
   - Provides API endpoints at `/api/*`
   - Handles React Router with catch-all route

3. **Production Optimizations**:
   - Lightweight Alpine Linux base
   - Production dependencies only
   - Health checks included
   - Proper error handling

## 🔧 Architecture

```
Railway Container
├── React App (served as static files)
├── Express API Server
│   ├── /api/sheets (Google Sheets proxy)
│   ├── /api/health (health check)
│   └── /api/cache/clear (cache management)
└── Single process serving both
```

## 🌐 URLs

After deployment, your app will be available at:
- **Main App**: `https://your-app.railway.app`
- **API Health**: `https://your-app.railway.app/api/health`  
- **Sheets Data**: `https://your-app.railway.app/api/sheets`

## 📋 Environment Variables

Set these in Railway dashboard:

| Variable | Value | Description |
|----------|--------|-------------|
| `REACT_APP_SUPABASE_URL` | Your Supabase URL | Supabase project URL |
| `REACT_APP_SUPABASE_ANON_KEY` | Your Supabase key | Supabase anonymous key |
| `PORT` | (auto-set by Railway) | Server port |

## ✅ Verification

After deployment, verify everything works:

1. **Health Check**: Visit `/api/health` - should return JSON status
2. **Sheets Data**: Visit `/api/sheets` - should return paper data  
3. **Main App**: Visit root URL - should load React app
4. **PDF Viewing**: Test PDF viewer with papers that have storage

## 🐛 Troubleshooting

### Build Issues
- Check Railway build logs for dependency errors
- Verify all files are committed to git

### Runtime Issues  
- Check Railway deployment logs
- Verify environment variables are set
- Test API endpoints directly

### CORS Issues
- The combined server handles CORS automatically
- API and frontend are served from same domain

## 🚧 Local Testing

Test the production build locally:

```bash
# Build the Docker image
docker build -t papers-viewer .

# Run the container
docker run -p 3001:3001 \
  -e REACT_APP_SUPABASE_URL=https://hwmmujsubcckybzkgvil.supabase.co \
  -e REACT_APP_SUPABASE_ANON_KEY=your_key_here \
  papers-viewer

# Visit http://localhost:3001
```

## 📝 Notes

- The app combines both frontend and backend in a single container
- Google Sheets data is cached for 5 minutes to improve performance
- Railway automatically handles HTTPS and custom domains
- The build process optimizes for production deployment