# Crypto Price Prediction System

## Overview
AI-powered crypto price direction prediction for BTC/ETH using multiple machine learning models (Random Forest & Logistic Regression). Features automated daily predictions, validation against actual price movements, real-time accuracy tracking, and a production-ready web dashboard with enhanced API endpoints.

## Project Description

This is a comprehensive cryptocurrency price prediction system that combines machine learning, real-time data processing, and web technologies to forecast BTC and ETH price movements. The system analyzes historical price data, social sentiment from Twitter and Reddit, and technical indicators to generate daily predictions with confidence scores.

### Key Features
- **Multi-Model ML Pipeline**: Uses both Random Forest and Logistic Regression models with automatic best-model selection
- **Real-Time Data Integration**: Fetches live price data from Binance API and social sentiment data
- **Automated Workflow**: Daily predictions, validation, and model retraining via GitHub Actions
- **Production-Ready API**: FastAPI backend with Redis caching, rate limiting, and WebSocket support
- **Interactive Dashboard**: Next.js frontend with real-time updates and comprehensive analytics
- **Performance Monitoring**: Automated accuracy tracking and model performance evaluation
- **Overfitting Prevention**: Conservative confidence calibration and regularization techniques

### Use Cases
- **Crypto Traders**: Get daily price direction predictions with confidence scores
- **Investors**: Monitor market sentiment and technical indicators
- **Researchers**: Analyze crypto market patterns and ML model performance
- **Developers**: Learn about ML pipelines, API development, and automated workflows

### Project Goals
- **Accuracy**: Achieve reliable price direction predictions with transparent confidence scores
- **Automation**: Minimize manual intervention through automated workflows and monitoring
- **Scalability**: Build a production-ready system that can handle real-world usage
- **Transparency**: Provide clear performance metrics and model explanations
- **Education**: Demonstrate modern ML and web development best practices

### Target Audience
- **Crypto Enthusiasts**: Individuals interested in cryptocurrency trading and analysis
- **Data Scientists**: Professionals looking to understand ML model deployment
- **Full-Stack Developers**: Developers learning about ML integration in web applications
- **Students**: Computer science students studying ML and web development
- **Researchers**: Academic researchers studying cryptocurrency market behavior

---

## Technologies Used

### Backend Technologies
- **Python 3.11**: Core programming language
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for FastAPI applications
- **Redis**: In-memory data store for caching and session management
- **WebSockets**: Real-time bidirectional communication
- **Asyncio**: Asynchronous programming for concurrent operations

### Machine Learning & Data Science
- **Scikit-learn**: Machine learning library (Random Forest, Logistic Regression)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization and plotting
- **Feature Engineering**: Custom technical indicators and sentiment analysis
- **Model Validation**: Cross-validation and performance metrics

### Database & Storage
- **Supabase**: PostgreSQL database with real-time capabilities
- **PostgreSQL**: Primary database for storing predictions, prices, and sentiment data
- **Database Migrations**: Automated schema management

### Frontend Technologies
- **Next.js 14**: React framework with App Router
- **React 18**: Component-based UI library
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js/Recharts**: Data visualization and interactive charts
- **Axios**: HTTP client for API communication

### DevOps & Deployment
- **GitHub Actions**: CI/CD and automated workflows
- **Railway**: Backend deployment and hosting
- **Vercel**: Frontend deployment and hosting
- **Docker**: Containerization for consistent deployments
- **Environment Management**: Secure configuration with environment variables

### External APIs & Services
- **Binance API**: Real-time cryptocurrency price data
- **Twitter API**: Social sentiment analysis
- **Reddit API**: Community sentiment data
- **Rate Limiting**: API protection and quota management

### Development Tools
- **Git**: Version control
- **ESLint**: Code linting and quality assurance
- **Prettier**: Code formatting
- **Jupyter Notebooks**: Data analysis and experimentation
- **Logging**: Comprehensive application logging

### Monitoring & Analytics
- **Performance Metrics**: Response time and accuracy tracking
- **Health Checks**: Service status monitoring
- **Error Tracking**: Exception handling and reporting
- **Cache Analytics**: Redis performance monitoring

---

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   External      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   APIs          │
│                 │    │                 │    │                 │
│ • React 18      │    │ • Python 3.11   │    │ • Binance API   │
│ • TypeScript    │    │ • FastAPI       │    │ • Twitter API   │
│ • Tailwind CSS  │    │ • Redis Cache   │    │ • Reddit API    │
│ • Real-time UI  │    │ • WebSockets    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Deployment    │    │   Database      │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • Vercel        │    │ • Supabase      │    │ • Scikit-learn  │
│ • Railway       │    │ • PostgreSQL    │    │ • Random Forest │
│ • Docker        │    │ • Real-time     │    │ • Log. Regress. │
│ • GitHub Actions│    │ • Migrations    │    │ • Feature Eng.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Data Ingestion**: External APIs → Backend → Database
2. **ML Training**: Historical data → Feature engineering → Model training
3. **Prediction**: Real-time data → ML pipeline → Predictions → Database
4. **Validation**: Actual prices → Accuracy calculation → Performance metrics
5. **Frontend**: API calls → Real-time updates → Interactive dashboard

### Key Components
- **API Gateway**: FastAPI with rate limiting and authentication
- **ML Engine**: Automated training and prediction pipeline
- **Cache Layer**: Redis for improved performance
- **Real-time Updates**: WebSocket connections for live data
- **Monitoring**: Health checks and performance analytics

---

## Quick Start

### 1. Setup
```bash
# Clone and install
git clone <repo-url>
cd capstone
pip install -r backend/requirements.txt

# Set environment variables (see backend/env_template.txt)
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"
```

### 2. Train Models
```bash
python backend/scripts/clean_train_models.py
```

### 3. Generate Predictions
```bash
# Today's predictions
python backend/scripts/clean_generate_predictions.py --current

# Historical predictions (for testing)
python backend/scripts/clean_generate_predictions.py --historic 30
```

### 4. Validate Predictions
```bash
python backend/scripts/auto_validate_predictions.py
```

---

## Automation

### GitHub Actions (Recommended)
- **Data ingestion & automation:** Every 4 hours + daily at 2 AM UTC (`.github/workflows/enhanced_daily_data_ingestion.yml`)
- **Daily predictions:** 6 AM UTC (`.github/workflows/daily_predictions.yml`)
- **Daily validation:** 8 AM UTC (`.github/workflows/daily_validation.yml`)
- **Simple model retraining:** Daily at 3 AM UTC (`.github/workflows/simple_retraining_no_git.yml`)
- **Model monitoring:** Daily at 2 AM UTC (`.github/workflows/simple_monitoring.yml`)
- **Weekly model retraining:** Every Sunday at 2 AM UTC (`.github/workflows/weekly_model_retraining.yml`)

**Setup:** Add repository secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `REDIS_URL`, `REDIS_ENABLED`

### Model Retraining System

The system includes simplified, reliable retraining approaches:

#### 1. Simple Daily Retraining (Recommended)
- **Schedule:** Daily at 3 AM UTC
- **Logic:** Always retrains models with fresh data for maximum reliability
- **Benefits:** Consistent, reliable execution without complex monitoring logic
- **File:** `.github/workflows/simple_retraining_no_git.yml`
- **Output:** Models uploaded as GitHub artifacts for download

#### 2. Weekly Retraining
- **Schedule:** Every Sunday at 2 AM UTC
- **Logic:** Always retrains models with fresh data
- **Benefits:** Ensures models stay current with market changes
- **File:** `.github/workflows/weekly_model_retraining.yml`

#### 3. Model Monitoring
- **Schedule:** Daily at 2 AM UTC (before retraining)
- **Purpose:** Monitor model performance and generate reports
- **File:** `.github/workflows/simple_monitoring.yml`
- **Output:** Performance reports uploaded as artifacts

#### Manual Retraining
```bash
# Monitor performance
python backend/scripts/monitor_model_performance.py

# Train models manually
python backend/scripts/clean_train_models.py

# Generate predictions
python backend/scripts/clean_generate_predictions.py --current
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/predictions/{currency}/history` | Get prediction history with validation |
| `/predictions/{currency}/best` | Get best predictions (prioritizes Random Forest) |
| `/predictions/accuracy/{currency}` | Get accuracy stats and metrics |
| `/predict/{currency}` | Generate new prediction (manual) |
| `/current_prices` | Get real-time BTC/ETH prices with 24h changes |
| `/prices/{currency}` | Get historical price data |
| `/sentiment/{currency}` | Get social sentiment data |
| `/analytics/correlation` | Get BTC/ETH correlation analytics |
| `/analytics/cache` | Get cache performance metrics |
| `/health` | Enhanced health check with service status |
| `/tasks/retrain_models` | Start model retraining (background task) |
| `/tasks/{task_id}` | Get retraining task status |

**Enhanced Features:**
- **Multi-model support:** Both Random Forest and Logistic Regression predictions
- **Best predictions:** API prioritizes Random Forest for better accuracy
- **Real-time caching:** Redis-powered caching for improved performance
- **Rate limiting:** Built-in API protection
- **WebSocket support:** Real-time updates (optional)

See [API_DOCS.md](API_DOCS.md) for complete documentation.

---

## Deployment

### Backend (FastAPI)
```bash
# Production deployment (recommended)
uvicorn backend.app.enhanced_main:app --host 0.0.0.0 --port 8000

# Alternative: Standard deployment
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**Enhanced Features:**
- **Redis caching** for improved performance
- **Rate limiting** for API protection
- **WebSocket support** for real-time updates
- **Comprehensive health checks** with service status

**Railway:** See [RAILWAY_SETUP.md](RAILWAY_SETUP.md) for detailed Railway deployment steps.

### Frontend (Next.js)
```bash
# Vercel or any Next.js host
cd frontend && npm run build
```

**Environment Variables:**
- Set `NEXT_PUBLIC_API_URL` to your backend URL
- Configure for production deployment

### Docker Support
```bash
# Build and run with Docker
docker build -t crypto-predictor .
docker run -p 8000:8000 crypto-predictor
```

---

## Core Files

### Scripts (backend/scripts/)
- `clean_train_models.py` - Train ML models (Random Forest & Logistic Regression)
- `clean_generate_predictions.py` - Generate predictions for all models
- `auto_validate_predictions.py` - Validate predictions against actual prices
- `daily_automation.py` - Complete automation (data ingestion, predictions, validation)
- `cleanup_models.py` - Remove old model files
- `monitor_model_performance.py` - Monitor model performance and generate reports
- `robust_monitor.py` - Enhanced monitoring with detailed metrics

### ML Components (backend/ml/)
- `clean_model_trainer.py` - Enhanced ML training pipeline with overfitting prevention
- `clean_prediction_pipeline.py` - Multi-model prediction pipeline
- `data_preprocessor.py` - Data preprocessing and cleaning
- `feature_engineering.py` - Feature engineering and selection

### API Components (backend/app/)
- `enhanced_main.py` - Production-ready FastAPI app with caching and rate limiting
- `main.py` - Standard FastAPI app
- `database.py` - Database management with multi-model support
- `models/api_models.py` - API request/response models

### Frontend Components (frontend/)
- `types/prediction.ts` - TypeScript type definitions
- `components/PredictionRow.tsx` - React prediction component
- `components/EnhancedDashboard.tsx` - Enhanced dashboard with multi-model support
- `utils/api.ts` - API service with best predictions support

---

## Current Performance

### Multi-Model System
- **Random Forest (Primary):** 60-65% accuracy (prioritized in dashboard)
- **Logistic Regression (Secondary):** 30-35% accuracy (available for comparison)
- **Confidence Range:** Realistic 45-85% (calibrated based on model performance)
- **Model Selection:** System automatically prioritizes Random Forest for better accuracy

### Recent Validation Results
- **BTC:** ~60-65% accuracy (Random Forest predictions)
- **ETH:** ~60-65% accuracy (Random Forest predictions)
- **Validation:** Daily automated validation against actual price movements
- **Overfitting Prevention:** Conservative confidence calibration and regularization

---

## Recent Upgrades & Improvements

### Multi-Model System (v2.0)
- **Dual Model Support:** Both Random Forest and Logistic Regression models
- **Smart Prioritization:** Random Forest prioritized for better accuracy (60-65% vs 30-35%)
- **Best Predictions API:** New endpoint that automatically selects the best model
- **Model Comparison:** Dashboard shows performance of both models

### Enhanced API (v2.0)
- **Production-Ready:** Enhanced FastAPI app with Redis caching and rate limiting
- **Real-Time Features:** WebSocket support for live updates
- **Advanced Analytics:** Correlation analysis and cache performance metrics
- **Comprehensive Health Checks:** Detailed service status monitoring

### Simplified Automation
- **Reliable Workflows:** Simplified GitHub Actions without complex monitoring logic
- **Artifact Management:** Models uploaded as GitHub artifacts for easy download
- **Separate Monitoring:** Independent monitoring workflow for better reliability
- **No Git Issues:** Option for artifact-based model management

### Overfitting Prevention
- **Conservative Confidence:** Realistic confidence ranges (45-85%) based on model performance
- **Regularization:** Enhanced model training with proper regularization techniques
- **Performance Calibration:** Confidence scores calibrated based on actual model performance

See [MULTI_MODEL_UPGRADE.md](MULTI_MODEL_UPGRADE.md) and [WORKFLOW_REBUILD.md](WORKFLOW_REBUILD.md) for detailed upgrade information.

---

## Security & Privacy

### Data Sources
- **Price Data:** Binance Public REST API (no authentication required)
- **Sentiment Data:** Twitter/Reddit public data (no personal data collected)
- **Storage:** Supabase PostgreSQL (your own database)

### Rate Limits
- **Binance API:** 1200 requests per minute
- **Twitter/Reddit:** Respects platform rate limits
- **Supabase:** Depends on your plan

### Environment Variables
Never commit `.env` files. Use environment variables for:
- Database credentials
- API keys (optional)
- Application settings

---

## Troubleshooting
- Check logs in `backend/logs/`
- Test API health: `GET /health`
- Verify database: `python -c "from app.database import db_manager; print(db_manager.is_connected())"`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Crypto Prediction Platform
