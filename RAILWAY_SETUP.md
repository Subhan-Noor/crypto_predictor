# Railway Deployment Guide

## Overview
This guide walks you through deploying the Crypto Prediction System backend to Railway.

## Prerequisites
- [Railway account](https://railway.app/)
- [GitHub account](https://github.com/)
- [Supabase project](https://supabase.com/)

## Step 1: Prepare Your Repository

### 1.1 Fork/Clone the Repository
```bash
git clone <your-repo-url>
cd capstone
```

### 1.2 Verify Required Files
Ensure these files exist in your repository:
- `railway.json` (Railway configuration)
- `backend/Dockerfile` (Docker configuration)
- `backend/requirements.txt` (Python dependencies)
- `backend/app/enhanced_main.py` (Main FastAPI app)

## Step 2: Set Up Supabase

### 2.1 Create Supabase Project
1. Go to [supabase.com](https://supabase.com/)
2. Create a new project
3. Note your project URL and API keys

### 2.2 Database Schema
The application will automatically create required tables:
- `crypto_prices` - Price data
- `crypto_sentiment` - Sentiment data  
- `predictions` - Prediction data

## Step 3: Deploy to Railway

### 3.1 Connect Repository
1. Go to [railway.app](https://railway.app/)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will detect the `railway.json` configuration

### 3.2 Configure Environment Variables
In Railway dashboard, add these environment variables:

**Required:**
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

**Optional:**
```
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
REDIS_URL=your_redis_url
REDIS_ENABLED=false
```

### 3.3 Deploy
1. Railway will automatically build and deploy
2. Monitor the build logs for any errors
3. Once deployed, note your Railway URL

## Step 4: Verify Deployment

### 4.1 Health Check
Visit: `https://your-railway-url.railway.app/health`

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX...",
  "version": "2.0.0"
}
```

### 4.2 API Documentation
Visit: `https://your-railway-url.railway.app/docs`

This shows the interactive Swagger UI for all API endpoints.

## Step 5: Train Models (First Time)

### 5.1 Access Railway Shell
1. Go to your Railway project
2. Click on your service
3. Go to "Settings" → "Shell"
4. Open the shell

### 5.2 Run Model Training
```bash
cd backend
python scripts/clean_train_models.py
```

This will:
- Download historical data
- Train ML models for BTC and ETH
- Save models to the `models/` directory

### 5.3 Generate Initial Predictions
```bash
python scripts/clean_generate_predictions.py --current
```

## Step 6: Set Up Automation (Optional)

### 6.1 GitHub Actions
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Add these secrets:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`

### 6.2 Verify Workflows
The following workflows will run automatically:
- **Daily Predictions:** 6 AM UTC
- **Daily Validation:** 8 AM UTC
- **Data Ingestion:** 2 AM UTC

## Step 7: Connect Frontend

### 7.1 Update Frontend Configuration
In your frontend deployment (Vercel, etc.), set:
```
NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app
```

### 7.2 Test Integration
1. Deploy your frontend
2. Test that it can fetch data from your Railway backend
3. Verify predictions are working

## Troubleshooting

### Common Issues

**1. Build Failures**
- Check Railway build logs
- Verify all dependencies in `requirements.txt`
- Ensure Dockerfile is correct

**2. Database Connection Errors**
- Verify Supabase credentials
- Check if Supabase project is active
- Ensure database tables exist

**3. Model Training Failures**
- Check if you have sufficient data
- Verify Python dependencies
- Monitor Railway logs

**4. API Errors**
- Check environment variables
- Verify CORS settings
- Test endpoints individually

### Getting Help

**Logs:**
- Railway dashboard → Your service → "Logs"
- Check for error messages and stack traces

**Health Check:**
```bash
curl https://your-railway-url.railway.app/health
```

**Database Test:**
```bash
curl https://your-railway-url.railway.app/predictions/BTC/history?days=1
```

## Cost Optimization

### Railway Pricing
- **Free Tier:** $5 credit monthly
- **Pro:** Pay-as-you-go
- **Team:** $20/month per user

### Optimization Tips
1. **Sleep Application:** Enable in Railway settings
2. **Reduce Logs:** Set `LOG_LEVEL=WARNING` in production
3. **Cache Data:** Use Redis for caching (optional)
4. **Monitor Usage:** Check Railway dashboard regularly

## Security Best Practices

### Environment Variables
- Never commit `.env` files
- Use Railway's environment variable system
- Rotate API keys regularly

### API Security
- The API is public (no authentication required)
- Rate limiting is handled by Railway
- CORS is configured for frontend domains

### Data Privacy
- No personal data is collected
- All data is stored in your Supabase instance
- You control your own data

## Next Steps

After successful deployment:

1. **Monitor Performance:** Check Railway metrics
2. **Set Up Alerts:** Configure monitoring
3. **Scale if Needed:** Upgrade Railway plan
4. **Customize:** Modify models or add features

## Support

- **Railway Docs:** [docs.railway.app](https://docs.railway.app/)
- **Supabase Docs:** [supabase.com/docs](https://supabase.com/docs)
- **Project Issues:** Create GitHub issue

---

**Deployment Complete! 🚀**

Your Crypto Prediction System is now live and ready to make predictions! 