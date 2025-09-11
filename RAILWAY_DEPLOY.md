# Railway Deployment Guide for DMA Bot

## Pre-deployment Checklist ✅
- [x] Fixed hardcoded paths to use dynamic BASE_DIR
- [x] Added local embedding model support
- [x] Added railway.json configuration
- [x] CORS enabled for frontend API calls
- [x] Health check endpoint available at `/health`

## Required Environment Variables
Set these in Railway dashboard after creating your project:

```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

Optional (if you want to force use of a specific embedding model path):
```
EMBEDDING_MODEL_PATH=/app/models/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
```

## Railway Deployment Steps

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub (recommended)

2. **Deploy from GitHub**
   - Push your code to GitHub repository
   - In Railway dashboard: "New Project" → "Deploy from GitHub repo"
   - Select your DMA_BOT repository
   - Railway will auto-detect Python and use railway.json config

3. **Set Environment Variables**
   - In project dashboard → "Variables" tab
   - Add GEMINI_API_KEY and PINECONE_API_KEY
   - Click "Deploy" to redeploy with new variables

4. **Test Deployment**
   - Wait for build to complete (~3-5 minutes)
   - Click "View Logs" to monitor startup
   - Open the provided railway.app URL
   - Test the chat interface

## Expected Build Time
- First deployment: 3-5 minutes (installing dependencies)
- Subsequent deployments: 1-2 minutes

## Troubleshooting
- **Build fails**: Check logs for missing dependencies
- **App crashes on startup**: Verify environment variables are set
- **Embedding model errors**: Model will download on first run (may take extra time)
- **503 errors**: Service is likely still starting up (Railway apps sleep when idle)

## Resource Usage (Free Tier)
- Memory: ~1GB (torch + sentence-transformers)
- Build time: ~3-5 minutes
- Cold start: ~10-15 seconds when sleeping

## Local Testing Command
```bash
# Test locally before deploying
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```
