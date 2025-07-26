# 📁 Project Structure

## Root Level
```
capstone/
├── README.md                    # Main project documentation
├── API_DOCS.md                  # API endpoint documentation  
├── RAILWAY_SETUP.md             # Railway deployment guide
├── PROJECT_STRUCTURE.md         # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── .gitattributes               # Git attributes
├── .dockerignore                # Docker ignore rules
├── railway.json                 # Railway deployment config
├── .github/                     # GitHub Actions workflows
│   └── workflows/
│       ├── daily_predictions.yml    # Daily prediction automation
│       ├── daily_validation.yml     # Daily validation automation
│       ├── daily_data_ingestion.yml # Daily data ingestion
│       ├── daily-sentiment-collection.yml # Sentiment collection
│       ├── enhanced_daily_data_ingestion.yml # Enhanced data ingestion
│       ├── initial_data_population.yml # Initial data setup
│       └── ci.yml                   # Continuous integration
├── backend/                     # Python backend application
└── frontend/                    # Next.js frontend application
```

## Backend Structure
```
backend/
├── app/                         # FastAPI application
├── ml/                          # Machine learning components
│   ├── clean_model_trainer.py       # Training pipeline
│   ├── clean_prediction_pipeline.py # Prediction pipeline
│   ├── data_preprocessor.py         # Data preprocessing
│   └── feature_engineering.py       # Feature engineering
├── scripts/                     # Operational scripts
│   ├── clean_train_models.py        # Train ML models
│   ├── clean_generate_predictions.py # Generate predictions
│   ├── auto_validate_predictions.py # Validate predictions
│   └── data_ingestion.py             # Data ingestion
├── models/                      # Trained ML models
│   ├── BTC_random_forest_*.pkl
│   ├── BTC_logistic_regression_*.pkl
│   ├── ETH_random_forest_*.pkl
│   └── ETH_logistic_regression_*.pkl
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── requirements-railway.txt     # Railway-specific deps
├── Dockerfile                   # Docker container config
├── run.py                       # Application runner
└── env_template.txt             # Environment variables template
```

## Frontend Structure
```
frontend/
├── types/
│   └── prediction.ts            # TypeScript type definitions
├── components/
│   └── PredictionRow.tsx        # React prediction component
└── ... (Next.js application)
```

## Key Files by Purpose

### 🚀 **Production Ready**
- `backend/app/enhanced_main.py` - Main FastAPI application
- `backend/scripts/clean_*.py` - Production scripts
- `backend/ml/clean_*.py` - Production ML pipeline

### 🤖 **Automation**
- `.github/workflows/*.yml` - GitHub Actions automation
- `RAILWAY_SETUP.md` - Deployment instructions

### 📚 **Documentation**
- `README.md` - Quick start guide
- `API_DOCS.md` - Complete API documentation
- `PROJECT_STRUCTURE.md` - This file

### ⚙️ **Configuration**
- `backend/config.py` - Application configuration
- `backend/requirements*.txt` - Dependencies
- `backend/Dockerfile` - Container configuration

## Cleaned Up Files ✨

The following obsolete files were removed during cleanup:
- `User_input.md` - Development notes
- `SENTIMENT_SETUP.md` - Obsolete setup guide
- `PROJECT_STATUS.md` - Development tracking
- `CONTRIBUTING.md` - Over-engineered guide
- `MONITORING_SETUP.md` - Over-engineered monitoring
- `DEPLOYMENT_GUIDE.md` - Redundant with README
- `SETUP.md` - Redundant with README  
- `Guide.md` - Development planning
- `backend/HANDOFF_PROGRESS.md` - Development handoff
- `backend/scripts/retrain_models_fixed.py` - Replaced by clean scripts
- `backend/scripts/train_models.py` - Replaced by clean scripts
- `backend/scripts/daily_automation.py` - Replaced by GitHub Actions
- `docs/` directory - Over-engineered documentation
- `database/` directory - Schema managed in Supabase
- `scripts/` directory - Moved to backend/scripts
- All `__pycache__/` directories - Compiled Python cache

## Result 🎉

**Before Cleanup:** 40+ files with redundant documentation, obsolete scripts, and development artifacts

**After Cleanup:** 15 essential files with clear purpose, streamlined documentation, and production-ready code

The project is now clean, focused, and ready for deployment! 🚀 