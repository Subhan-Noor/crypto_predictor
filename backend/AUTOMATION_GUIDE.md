# 🤖 Automation Guide - Stage 6

This guide covers all automation components implemented in Stage 6: Integrations & Automation.

## 📋 Overview

The automation system provides multiple ways to schedule and run daily operations:

1. **GitHub Actions** - Cloud-based automation (already running)
2. **Python Scripts** - Flexible automation tools
3. **Cron Scripts** - Traditional Unix/Linux scheduling
4. **API Endpoints** - Manual and programmatic triggering
5. **Monitoring** - Health checks and status tracking

---

## 🚀 Automation Components

### 1. GitHub Actions (Recommended for Cloud)

**File**: `.github/workflows/ci.yml`

**Features**:
- Runs daily at 6:00 AM UTC
- Includes data ingestion and predictions
- Comprehensive error handling
- Automatic status reporting

**Current Schedule**:
```yaml
schedule:
  - cron: '0 6 * * *'  # Daily at 6 AM UTC
```

**Manual Trigger**:
- Go to GitHub Actions tab in your repository
- Select "CI" workflow
- Click "Run workflow"

### 2. Python Automation Scripts

#### Main Automation Script
**File**: `backend/scripts/daily_automation.py`

**Usage**:
```bash
# Complete automation pipeline
python daily_automation.py --full

# Individual components
python daily_automation.py --data-ingestion
python daily_automation.py --predictions
python daily_automation.py --health-check

# With result saving
python daily_automation.py --full --save-results

# Check status
python daily_automation.py --status
```

#### Prediction Generation Script
**File**: `backend/scripts/generate_predictions.py`

**Usage**:
```bash
# Generate predictions for all currencies
python generate_predictions.py --daily

# Generate for specific currency
python generate_predictions.py --currency BTC
python generate_predictions.py --currency ETH

# Test mode (no database save)
python generate_predictions.py --test --all
```

### 3. Cron Automation (Unix/Linux)

**File**: `backend/scripts/cron_automation.sh`

**Features**:
- Shell script for traditional cron scheduling
- Automatic logging and cleanup
- Environment validation
- Error notification support

**Setup**:
```bash
# Make executable
chmod +x backend/scripts/cron_automation.sh

# Test the script
./backend/scripts/cron_automation.sh health

# Add to crontab
crontab -e
```

**Cron Examples**:
```bash
# Daily automation at 6:00 AM
0 6 * * * /path/to/capstone/backend/scripts/cron_automation.sh full

# Data ingestion every 6 hours
0 */6 * * * /path/to/capstone/backend/scripts/cron_automation.sh data

# Predictions every 12 hours
0 */12 * * * /path/to/capstone/backend/scripts/cron_automation.sh predictions

# Health check every hour
0 * * * * /path/to/capstone/backend/scripts/cron_automation.sh health
```

### 4. API Endpoints

Base URL: `http://localhost:8000` (development) or your production URL

#### Automation Status
```http
GET /automation/status
```
Returns current automation pipeline health.

#### Manual Triggering
```http
POST /automation/trigger?task=full
POST /automation/trigger?task=data-ingestion
POST /automation/trigger?task=predictions
POST /automation/trigger?task=health-check
```

#### Automation History
```http
GET /automation/history?days=7
```
Returns automation performance metrics.

#### Daily Predictions
```http
POST /predictions/daily
```
Generates predictions for both BTC and ETH.

---

## 🔧 Configuration

### Environment Variables

Required variables for automation:

```bash
# Database (Required)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Data Sources (Optional but recommended)
COINGECKO_API_KEY=your_coingecko_key
TWITTER_BEARER_TOKEN=your_twitter_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Application Settings
ENVIRONMENT=production
REDIS_ENABLED=false  # Set to false for simple deployments
DEBUG=false
LOG_LEVEL=INFO
```

### GitHub Secrets

For GitHub Actions, add these secrets to your repository:
- `SUPABASE_URL`
- `SUPABASE_KEY` 
- `COINGECKO_API_KEY`
- `TWITTER_BEARER_TOKEN`
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`

---

## 📊 Monitoring and Logging

### Log Files

**Location**: `backend/logs/`

**Types**:
- `automation_YYYYMMDD_HHMMSS.log` - Main automation logs
- `automation_errors_YYYYMMDD_HHMMSS.log` - Error logs only

**Cleanup**: Logs older than 30 days are automatically cleaned up.

### Health Checks

The automation system includes comprehensive health checks:

1. **Database Connectivity** - Verifies Supabase connection
2. **Data Availability** - Checks for recent price data
3. **Model Availability** - Verifies ML models are present
4. **Recent Predictions** - Confirms predictions are being generated

### Status Monitoring

**Health Scores**:
- `healthy` (75%+ components working)
- `degraded` (50-74% components working)  
- `unhealthy` (<50% components working)

**API Monitoring**:
```bash
# Check automation status
curl http://localhost:8000/automation/status

# Check automation history
curl http://localhost:8000/automation/history?days=7
```

---

## 🛠️ Setup Instructions

### 1. Development Environment

```bash
# Navigate to backend
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (if not already done)
pip install -r requirements.txt

# Test automation
python scripts/daily_automation.py --health-check
```

### 2. Production Deployment

#### Option A: GitHub Actions (Recommended)
1. Push code to GitHub repository
2. Add required secrets to repository settings
3. Automation runs automatically daily at 6 AM UTC

#### Option B: Server with Cron
```bash
# Clone repository on server
git clone your-repo-url
cd capstone/backend

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file with production values
cp .env.example .env
# Edit .env with your production values

# Test automation
./scripts/cron_automation.sh health

# Add to cron
crontab -e
# Add: 0 6 * * * /path/to/capstone/backend/scripts/cron_automation.sh full
```

#### Option C: Container Deployment
```bash
# Build container (if Dockerfile exists)
docker build -t crypto-automation .

# Run automation
docker run --env-file .env crypto-automation python scripts/daily_automation.py --full
```

---

## 🔄 Automation Workflow

The complete daily automation workflow:

1. **Data Ingestion**
   - Fetch latest cryptocurrency prices from Binance
   - Collect sentiment data from Twitter and Reddit
   - Store data in Supabase database
   - Validate data quality

2. **Feature Engineering**
   - Process historical price data
   - Calculate technical indicators
   - Merge price and sentiment data
   - Prepare features for ML models

3. **Prediction Generation**
   - Load trained ML models
   - Generate 7-day price predictions for BTC and ETH
   - Calculate confidence scores
   - Store predictions in database

4. **Health Monitoring**
   - Verify system components
   - Check data freshness
   - Validate prediction accuracy
   - Generate health reports

5. **Cleanup & Reporting**
   - Clean up old logs and temporary files
   - Save automation results
   - Send notifications (if configured)

---

## 🚨 Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check environment variables
python -c "from config import settings; print(f'DB URL: {settings.supabase_url[:20]}...')"

# Test database connection
python scripts/daily_automation.py --health-check
```

**2. Missing ML Models**
```bash
# Check models directory
ls -la backend/models/

# Train models if missing
python scripts/train_models.py --all
```

**3. API Rate Limits**
- Check API key limits for external services
- Increase delays between API calls if needed
- Use caching to reduce API calls

**4. Cron Job Not Running**
```bash
# Check cron service
sudo service cron status

# Check cron logs
grep CRON /var/log/syslog

# Verify script permissions
ls -la backend/scripts/cron_automation.sh
```

### Error Recovery

The automation system includes automatic error recovery:
- **Partial Failures**: Continue with remaining tasks
- **Data Errors**: Skip problematic records, continue processing
- **Model Errors**: Fall back to simpler models
- **API Errors**: Retry with exponential backoff

### Notification Setup

To enable notifications, uncomment and configure in `cron_automation.sh`:

**Email Notifications**:
```bash
# Install mail command
sudo apt-get install mailutils

# Uncomment email section in script
```

**Webhook Notifications**:
```bash
# Set webhook URL environment variable
export WEBHOOK_URL="https://your-webhook-url.com/notifications"

# Uncomment webhook section in script
```

---

## 📈 Performance Optimization

### Recommended Schedules

**High Frequency** (for active trading):
- Data ingestion: Every 4 hours
- Predictions: Every 6 hours
- Health checks: Every hour

**Standard** (for general monitoring):
- Data ingestion: Daily at 6 AM
- Predictions: Daily at 6:30 AM  
- Health checks: Every 6 hours

**Low Frequency** (for research):
- Data ingestion: Daily
- Predictions: Weekly
- Health checks: Daily

### Scaling Considerations

- **Database**: Monitor Supabase usage and upgrade plan if needed
- **API Limits**: Track external API usage and implement caching
- **Storage**: Regular cleanup of logs and temporary files
- **Compute**: Consider upgrading server resources for larger datasets

---

## ✅ Stage 6 Deliverables Complete

✅ **Schedule prediction runs (daily) and store results in database**
- GitHub Actions workflow with daily predictions
- Python scripts for flexible scheduling
- Cron-compatible shell scripts

✅ **Set up cron job or GitHub Actions workflow for automation**
- Enhanced GitHub Actions with error handling
- Cron automation script with logging
- API endpoints for manual triggering

✅ **Comprehensive monitoring and error handling**
- Health checks and status monitoring
- Automated logging and cleanup
- Error recovery and notification system

✅ **Multiple deployment options**
- Cloud automation via GitHub Actions
- Server automation via cron
- Manual/programmatic via API

**Stage 6 is now complete and ready for Stage 7 deployment!** 