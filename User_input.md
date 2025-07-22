# User Input Checklist Before Stage 3

To ensure the project is ready for ML model development, please complete the following steps:

---

## 1. Supabase Setup

1. **Create a Supabase Project:**
   - Go to [https://supabase.com](https://supabase.com) and sign up/log in.
   - Click "New Project" and follow the prompts.
   - Wait for the project to initialize.

2. **Set Up Database Schema:**
   - In the Supabase dashboard, go to "SQL Editor".
   - Copy and paste the contents of `database/schema.sql` from this repo.
   - Click "Run" to execute and create the tables.

3. **Get Supabase Credentials:**
   - In the dashboard, go to **Settings → API**.
   - Copy the following:
     - **Project URL** (e.g., `https://xyzcompany.supabase.co`)
     - **anon public key** (for SUPABASE_KEY)
     - **service_role key** (for SUPABASE_SERVICE_ROLE_KEY, optional for now)

---

## 2. .env File Setup (backend/.env)

Create a file named `.env` in the `backend/` directory with the following content:

```
# Database Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key  # Optional, not required for basic usage

# API Keys (Optional for testing, but recommended for full functionality)
COINGECKO_API_KEY=your_coingecko_api_key
FEAR_GREED_API_URL=https://api.alternative.me/fng/

# Social Media APIs (Optional, but recommended for sentiment analysis)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key  # Not required for basic usage
TWITTER_API_SECRET=your_twitter_api_secret  # Not required for basic usage
TWITTER_ACCESS_TOKEN=your_twitter_access_token  # Not required for basic usage
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret  # Not required for basic usage

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=CryptoPredictorBot/1.0

# Redis (for caching, optional)
REDIS_URL=redis://localhost:6379

# Application Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
```

**Replace all `your_...` values with your actual credentials.**

---

## 3. API Key Instructions

### CoinGecko API (Free)
- Go to [https://www.coingecko.com/api](https://www.coingecko.com/api)
- Sign up for a free account
- Get your API key and add it to `.env` as `COINGECKO_API_KEY`

### Twitter API (Free Basic Tier)
- Go to [https://developer.twitter.com/en](https://developer.twitter.com/en)
- Apply for a developer account
- Create a new app and get the Bearer Token
- Add it to `.env` as `TWITTER_BEARER_TOKEN`

### Reddit API (Free)
- Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Create a new application (script type)
- Get the client ID and secret
- Add them to `.env` as `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`

---

## 4. Initial Data Ingestion

After completing the above steps:

1. **Activate your Python virtual environment:**
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run initial data ingestion:**
   ```bash
   python scripts/data_ingestion.py initial 30
   ```
   This will fetch the last 30 days of price and sentiment data for BTC and ETH.

---

## 5. Verify Setup

- Visit `http://localhost:8000/health` to check API health.
- Visit `http://localhost:8000/data_status` to check data in the database.
- If you see errors about missing credentials or failed connections, double-check your `.env` file and Supabase setup.

---

## 6. (Optional) Enable Daily Automation

- Add your API keys and Supabase credentials as GitHub repository secrets if you want to use the automated daily ingestion workflow.
- The workflow will run daily at 6 AM UTC.

---

**Once all steps are complete and data is flowing, you are ready for Stage 3 (ML model development)!** 