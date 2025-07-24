# 🎭 Sentiment Analysis Setup Guide

This guide will help you set up and configure the sentiment analysis system for the crypto price prediction app.

## 📋 Overview

The sentiment analysis system collects and analyzes social media sentiment from:
- **Twitter**: Using Twitter API v2 with Tweepy
- **Reddit**: Using Reddit API with PRAW (fallback to Pushshift)
- **Analysis**: TextBlob and VADER sentiment analysis engines

## 🔧 Prerequisites

### Required Dependencies
The following are already included in `requirements.txt`:
```
textblob==0.17.1
vaderSentiment==3.3.2
tweepy==4.14.0
praw==7.7.1
```

### API Accounts Needed
1. **Twitter Developer Account** (Optional but recommended)
2. **Reddit Developer Application** (Optional but recommended)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt

# Download required NLTK data for TextBlob
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

### 2. Test Basic Functionality (No API Keys Required)
```bash
cd backend
python scripts/sentiment_data_collection.py --mode test
```

This will test the sentiment analyzer with sample text even without API keys.

### 3. Configure API Keys (Optional but Recommended)

Add these to your environment variables or `.env` file:

#### Twitter API Configuration
```bash
# Option 1: Bearer Token (Recommended - easier setup)
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Option 2: Full OAuth (More features)
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret  
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
```

#### Reddit API Configuration
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT="CryptoPredictorBot/1.0"
```

### 4. Run Daily Collection
```bash
cd backend
python scripts/sentiment_data_collection.py --mode daily --verbose
```

## 🔑 API Setup Instructions

### Twitter API Setup

#### Option 1: Bearer Token (Easier)
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new app or use existing
3. Go to "Keys and Tokens" → "Bearer Token"
4. Copy the Bearer Token
5. Set `TWITTER_BEARER_TOKEN` environment variable

#### Option 2: Full OAuth (More Features)
1. In Twitter Developer Portal
2. Go to "Keys and Tokens"
3. Generate all tokens:
   - API Key → `TWITTER_API_KEY`
   - API Secret → `TWITTER_API_SECRET`
   - Access Token → `TWITTER_ACCESS_TOKEN`
   - Access Token Secret → `TWITTER_ACCESS_TOKEN_SECRET`

### Reddit API Setup

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create Application"
3. Choose "script" type
4. Note down:
   - Client ID (under app name) → `REDDIT_CLIENT_ID`
   - Client Secret → `REDDIT_CLIENT_SECRET`
5. Set a descriptive User Agent → `REDDIT_USER_AGENT`

## 📊 Usage Examples

### Daily Collection
```bash
# Collect today's sentiment data
python scripts/sentiment_data_collection.py --mode daily

# With verbose logging
python scripts/sentiment_data_collection.py --mode daily --verbose
```

### Test Services
```bash
# Test all sentiment services
python scripts/sentiment_data_collection.py --mode test --verbose
```

### Historical Collection (Limited)
```bash
# Try to collect last 7 days (limited by API availability)
python scripts/sentiment_data_collection.py --mode historical --days 7
```

## 🔍 Understanding the Output

### Sentiment Scores
- **Range**: -1.0 (very negative) to +1.0 (very positive)
- **0.0**: Neutral sentiment
- **> 0.1**: Generally positive
- **< -0.1**: Generally negative

### Example Output
```
📈 Processing BTC...
  ✅ Twitter: 0.2341 (confidence: 0.7823)
  ✅ Reddit: 0.1567 (confidence: 0.6432)
  💾 Successfully saved BTC sentiment to database

📈 Processing ETH...
  ✅ Twitter: -0.0892 (confidence: 0.5431)
  ✅ Reddit: 0.0234 (confidence: 0.4321)
  💾 Successfully saved ETH sentiment to database
```

## 🛠️ Configuration Options

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | ✅ | Database connection |
| `SUPABASE_KEY` | ✅ | Database authentication |
| `TWITTER_BEARER_TOKEN` | ⚠️ | Twitter API access (recommended) |
| `REDDIT_CLIENT_ID` | ⚠️ | Reddit API access (recommended) |
| `REDDIT_CLIENT_SECRET` | ⚠️ | Reddit API authentication |
| `LOG_LEVEL` | ❌ | Logging level (INFO, DEBUG) |

### Fallback Behavior

The system is designed to work even without API keys:

1. **No Twitter API**: Uses fallback neutral sentiment
2. **No Reddit API**: Attempts Pushshift API fallback
3. **No APIs**: Still provides basic sentiment analysis for any text

## 🤖 Automated Collection

### GitHub Actions (Production)

The system includes automated daily collection via GitHub Actions:

1. **File**: `.github/workflows/daily-sentiment-collection.yml`
2. **Schedule**: Daily at 9:00 AM UTC
3. **Manual Trigger**: Available in GitHub Actions tab

#### Setup GitHub Secrets:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
TWITTER_BEARER_TOKEN=your_twitter_token (optional)
REDDIT_CLIENT_ID=your_reddit_id (optional)
REDDIT_CLIENT_SECRET=your_reddit_secret (optional)
```

### Local Scheduling (Development)

#### Using Cron (Linux/Mac)
```bash
# Add to crontab: Run daily at 9 AM
0 9 * * * cd /path/to/project/backend && python scripts/sentiment_data_collection.py --mode daily
```

#### Using Task Scheduler (Windows)
1. Open Task Scheduler
2. Create Basic Task
3. Set daily trigger
4. Action: Start program
5. Program: `python`
6. Arguments: `scripts/sentiment_data_collection.py --mode daily`
7. Start in: `C:\path\to\project\backend`

## 🔧 Troubleshooting

### Common Issues

#### 1. "Twitter API not available"
- **Cause**: Missing or invalid Twitter credentials
- **Solution**: Check `TWITTER_BEARER_TOKEN` or OAuth credentials
- **Impact**: System continues with Reddit data only

#### 2. "Reddit API not available"  
- **Cause**: Missing Reddit credentials
- **Solution**: Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
- **Impact**: Falls back to Pushshift API (limited)

#### 3. "No sentiment data found"
- **Cause**: APIs returning no relevant posts/tweets
- **Solution**: Normal behavior; data availability varies
- **Impact**: Neutral sentiment (0.0) stored

#### 4. Rate Limiting
- **Cause**: Too many API requests
- **Solution**: Built-in rate limiting; wait and retry
- **Impact**: Temporary delays in collection

### Debug Mode
```bash
# Enable verbose logging
python scripts/sentiment_data_collection.py --mode test --verbose

# Check specific service
python -c "
from app.services.twitter_service import twitter_service
print(twitter_service.test_connection())
"
```

### Database Issues
```bash
# Test database connection
cd backend
python test_connection.py

# Check sentiment data
python -c "
from app.database import db_manager
data = db_manager.get_latest_sentiment('BTC', 5)
print(data)
"
```

## 📈 Monitoring & Analytics

### Collection Statistics

The system provides detailed statistics:
- Total records processed
- Success/error rates
- Confidence scores
- Data source information

### Database Schema

Sentiment data is stored in the `crypto_sentiment` table:
```sql
CREATE TABLE crypto_sentiment (
    id UUID PRIMARY KEY,
    currency TEXT NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    twitter_sentiment DECIMAL(5, 4),
    reddit_sentiment DECIMAL(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### API Endpoints

Access sentiment data via REST API:
```bash
# Get latest sentiment
GET /sentiment/BTC?days=30

# Get detailed sentiment analysis
GET /sentiment/BTC/detailed
```

## 🚀 Next Steps

1. **Configure API Keys**: Set up Twitter and Reddit API access
2. **Test Collection**: Run test mode to verify setup
3. **Schedule Daily Runs**: Set up automated collection
4. **Monitor Performance**: Check logs and database for data quality
5. **Integrate with ML Models**: Sentiment data automatically feeds into prediction models

## 📚 Additional Resources

- [Twitter API Documentation](https://developer.twitter.com/en/docs/twitter-api)
- [Reddit API Documentation](https://www.reddit.com/dev/api/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [VADER Sentiment Documentation](https://github.com/cjhutto/vaderSentiment)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs with `--verbose` flag
3. Test individual components with `--mode test`
4. Verify API credentials and database connection

The sentiment analysis system is designed to be robust and will continue operating even with partial API availability. 