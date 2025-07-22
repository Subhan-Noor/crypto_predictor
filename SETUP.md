# Setup Guide - Crypto Price Prediction App

This guide will help you set up the crypto price prediction application locally.

## Prerequisites

- **Python 3.10+** installed
- **Node.js 18+** and npm installed
- **Git** installed
- A **Supabase** account (free tier available)

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd capstone
```

## 2. Database Setup (Supabase)

1. **Create a Supabase Project:**
   - Go to [supabase.com](https://supabase.com)
   - Sign up and create a new project
   - Wait for the project to be initialized

2. **Set up Database Schema:**
   - In your Supabase dashboard, go to "SQL Editor"
   - Copy and paste the contents of `database/schema.sql`
   - Execute the SQL to create tables

3. **Get Supabase Credentials:**
   - Go to Settings → API in your Supabase dashboard
   - Copy your Project URL and anon public key

## 3. Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   
   Create a `.env` file in the `backend/` directory:
   ```bash
   # Database Configuration
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   
   # API Keys (Optional for testing)
   COINGECKO_API_KEY=your_coingecko_api_key
   FEAR_GREED_API_URL=https://api.alternative.me/fng/
   
   # Social Media APIs (Optional)
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=CryptoPredictorBot/1.0
   
   # Application Settings
   ENVIRONMENT=development
   DEBUG=True
   LOG_LEVEL=INFO
   ```

5. **Test the backend:**
   ```bash
   python run.py
   ```
   
   The API should be available at `http://localhost:8000`
   Visit `http://localhost:8000/docs` for the interactive API documentation.

## 4. Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   
   The frontend should be available at `http://localhost:3000`

## 5. Initial Data Setup

With both backend and frontend running, you can populate the database with initial data:

1. **Run initial data ingestion:**
   ```bash
   cd backend
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   python scripts/data_ingestion.py initial 30
   ```
   
   This will fetch the last 30 days of price data and current sentiment data.

## 6. API Keys Setup (Optional but Recommended)

### CoinGecko API (Free)
1. Visit [CoinGecko API](https://www.coingecko.com/api)
2. Sign up for a free account
3. Get your API key and add it to `.env`

### Twitter API (Free Basic Tier)
1. Apply for a Twitter Developer account
2. Create a new app and get Bearer Token
3. Add to `.env` file

### Reddit API (Free)
1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create a new application
3. Get client ID and secret
4. Add to `.env` file

## 7. Verify Setup

1. **Check API Health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check Database Status:**
   ```bash
   curl http://localhost:8000/data_status
   ```

3. **Test Frontend:**
   - Visit `http://localhost:3000`
   - Verify the dashboard loads

## 8. Automated Data Ingestion

To set up daily automated data ingestion:

1. **Manual Run:**
   ```bash
   cd backend
   python scripts/data_ingestion.py daily
   ```

2. **Automated via GitHub Actions:**
   - The project includes a GitHub Actions workflow
   - Add your API keys as repository secrets
   - The workflow runs daily at 6 AM UTC

## Common Issues

### Database Connection Issues
- Verify Supabase URL and key are correct
- Check if your IP is allowed in Supabase (should be allowed by default)
- Ensure tables are created using the schema.sql

### API Rate Limits
- CoinGecko: Free tier has rate limits, upgrade if needed
- Twitter: Basic tier has monthly limits
- Reddit: Generous rate limits for personal use

### Missing Dependencies
```bash
# If you get import errors, try:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Frontend Build Issues
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Next Steps

After setup is complete:

1. **Stage 3**: Implement ML models for price prediction
2. **Stage 4**: Enhance API endpoints
3. **Stage 5**: Build comprehensive frontend dashboard
4. **Stage 6**: Set up production deployment

## Getting Help

- Check the GitHub Issues for common problems
- Review the API documentation at `http://localhost:8000/docs`
- Verify your environment variables are set correctly 