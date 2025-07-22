# 🚀 Deployment Guide - Stage 7

This guide covers deploying the Crypto Price Prediction App to production using Vercel (frontend) and Railway/Render (backend).

## 📋 Prerequisites

- [ ] GitHub repository with all code
- [ ] Supabase project with database set up
- [ ] API keys (optional but recommended)
- [ ] Vercel account
- [ ] Railway or Render account

---

## 🎯 1. Backend Deployment (Railway/Render)

### Option A: Railway Deployment (Recommended)

1. **Connect GitHub Repository**
   - Go to [Railway](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Build Settings**
   - Railway will automatically detect the `railway.json` configuration
   - Docker build will use `backend/Dockerfile`

3. **Set Environment Variables**
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   ENVIRONMENT=production
   DEBUG=False
   LOG_LEVEL=INFO
   REDIS_ENABLED=false
   
   # Optional API Keys
   COINGECKO_API_KEY=your_coingecko_api_key
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   ```

4. **Deploy**
   - Railway will automatically build and deploy
   - Note the generated domain (e.g., `your-app.railway.app`)

### Option B: Render Deployment

1. **Create New Web Service**
   - Go to [Render](https://render.com)
   - Create new "Web Service"
   - Connect GitHub repository

2. **Configure Service**
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: Leave empty (or `backend` if needed)

3. **Set Environment Variables** (same as Railway above)

4. **Deploy**
   - Render will build and deploy automatically
   - Note the generated domain (e.g., `your-app.onrender.com`)

---

## 🌐 2. Frontend Deployment (Vercel)

1. **Connect GitHub Repository**
   - Go to [Vercel](https://vercel.com)
   - Sign in with GitHub
   - Click "New Project"
   - Import your repository

2. **Configure Build Settings**
   - **Framework**: Next.js (auto-detected)
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

3. **Set Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-domain.railway.app
   ```
   (Replace with your actual backend domain from step 1)

4. **Deploy**
   - Vercel will automatically build and deploy
   - Note your frontend domain (e.g., `your-app.vercel.app`)

---

## 🔧 3. Post-Deployment Configuration

### Update CORS Settings

1. **Update backend CORS configuration**
   - The backend is already configured to handle production domains
   - CORS will automatically allow your Vercel domain

### Update Frontend API URL

1. **Update environment variable in Vercel**
   - Go to your Vercel project settings
   - Update `NEXT_PUBLIC_API_URL` with your actual backend domain
   - Redeploy if necessary

---

## 🛡️ 4. Security Configuration

### Supabase Security

1. **Configure RLS (Row Level Security)**
   - Enable RLS on your Supabase tables
   - Set up appropriate policies for your data access patterns

2. **API Key Security**
   - Use anon key for client-side operations
   - Use service role key only for server-side operations
   - Never expose service role key in frontend

### Environment Variables Security

1. **Backend Environment Variables**
   - All sensitive keys should be set in Railway/Render dashboard
   - Never commit actual keys to repository

2. **Frontend Environment Variables**
   - Only `NEXT_PUBLIC_*` variables are exposed to browser
   - Keep API URLs public (they're visible anyway)

---

## 📊 5. Monitoring Setup

### Health Checks

1. **Backend Health Check**
   - Endpoint: `https://your-backend-domain.railway.app/health`
   - Should return status information

2. **Frontend Health Check**
   - Endpoint: `https://your-frontend-domain.vercel.app`
   - Should load the application

### Automation Monitoring

1. **GitHub Actions Status**
   - Check GitHub Actions tab for automation status
   - Daily automation should run at 6 AM UTC

2. **API Monitoring Endpoints**
   - `/automation/status` - Real-time automation health
   - `/automation/history` - Performance metrics

### Logging

1. **Backend Logs**
   - Railway: View logs in Railway dashboard
   - Render: View logs in Render dashboard

2. **Frontend Logs**
   - Vercel: View function logs in Vercel dashboard

---

## 🧪 6. Testing Production Deployment

### Backend Testing

```bash
# Test health endpoint
curl https://your-backend-domain.railway.app/health

# Test data status
curl https://your-backend-domain.railway.app/data_status

# Test predictions (may take a few seconds)
curl -X POST https://your-backend-domain.railway.app/predict/BTC
```

### Frontend Testing

1. **Manual Testing**
   - Visit your Vercel domain
   - Test all pages and functionality
   - Check browser console for errors

2. **API Integration Testing**
   - Verify frontend can connect to backend
   - Test data fetching and predictions
   - Check network tab for API calls

---

## 🚨 7. Troubleshooting

### Common Issues

1. **CORS Errors**
   - Verify backend CORS configuration includes your frontend domain
   - Check environment variables are set correctly

2. **Environment Variable Issues**
   - Verify all required environment variables are set
   - Check variable names match exactly (case-sensitive)

3. **Build Failures**
   - Check build logs for specific error messages
   - Verify all dependencies are listed in requirements.txt/package.json

4. **Database Connection Issues**
   - Verify Supabase URL and keys are correct
   - Check Supabase project is active and accessible

### Getting Help

1. **Logs**
   - Check platform-specific logs (Railway/Render/Vercel)
   - Look for error messages and stack traces

2. **Health Endpoints**
   - Use `/health` and `/data_status` endpoints to diagnose issues

---

## ✅ 8. Deployment Checklist

### Pre-Deployment
- [ ] All code committed and pushed to GitHub
- [ ] Environment variables documented
- [ ] Database schema up to date
- [ ] Tests passing

### Backend Deployment
- [ ] Railway/Render service created
- [ ] Environment variables configured
- [ ] Service deployed successfully
- [ ] Health check endpoint responding
- [ ] Domain/URL noted for frontend configuration

### Frontend Deployment
- [ ] Vercel project created
- [ ] Backend API URL configured
- [ ] Project deployed successfully
- [ ] Frontend loads without errors
- [ ] API integration working

### Post-Deployment
- [ ] CORS configuration verified
- [ ] All features tested in production
- [ ] Automation running successfully
- [ ] Monitoring endpoints accessible
- [ ] Documentation updated with production URLs

---

## 🎉 Success!

Once deployed, your crypto prediction app will be live at:
- **Frontend**: `https://your-app.vercel.app`
- **Backend API**: `https://your-app.railway.app`
- **API Documentation**: `https://your-app.railway.app/docs`

The automation system will continue running daily via GitHub Actions, fetching data and generating predictions automatically. 