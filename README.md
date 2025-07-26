# Crypto Price Prediction System

## Overview
AI-powered crypto price direction prediction for BTC/ETH using machine learning models. Features automated daily predictions, validation against actual price movements, and real-time accuracy tracking.

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
- **Daily predictions:** 6 AM UTC (`.github/workflows/daily_predictions.yml`)
- **Daily validation:** 8 AM UTC (`.github/workflows/daily_validation.yml`)

**Setup:** Add repository secrets: `SUPABASE_URL`, `SUPABASE_KEY`

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/predictions/{currency}/history` | Get prediction history with validation |
| `/predictions/accuracy/{currency}` | Get accuracy stats and metrics |
| `/predict/{currency}` | Generate new prediction (manual) |

See [API_DOCS.md](API_DOCS.md) for complete documentation.

---

## Deployment

### Backend (FastAPI)
```bash
# Railway, Render, or any Python host
uvicorn backend.app.enhanced_main:app --host 0.0.0.0 --port 8000
```

**Railway:** See [RAILWAY_SETUP.md](RAILWAY_SETUP.md) for detailed Railway deployment steps.

### Frontend (Next.js)
```bash
# Vercel or any Next.js host
cd frontend && npm run build
```

Set `NEXT_PUBLIC_API_URL` to your backend URL.

---

## Core Files

### Scripts (backend/scripts/)
- `clean_train_models.py` - Train ML models
- `clean_generate_predictions.py` - Generate predictions  
- `auto_validate_predictions.py` - Validate predictions
- `data_ingestion.py` - Fetch price/sentiment data

### Components
- `backend/ml/clean_model_trainer.py` - ML training pipeline
- `backend/ml/clean_prediction_pipeline.py` - Prediction pipeline
- `frontend/types/prediction.ts` - TypeScript types
- `frontend/components/PredictionRow.tsx` - React component

---

## Current Performance
- **BTC:** ~69.6% accuracy (validated predictions)
- **ETH:** ~67.4% accuracy (validated predictions)
- **Confidence:** Realistic 45-85% range (no more 100% overconfidence)

---

## Troubleshooting
- Check logs in `backend/logs/`
- Test API health: `GET /health`
- Verify database: `python -c "from app.database import db_manager; print(db_manager.is_connected())"`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Crypto Prediction Platform
