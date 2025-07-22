from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

from .database import db_manager
from .services.twitter_service import TwitterScraper
from .services.reddit_service import RedditScraper
from .services.binance_service import BinancePriceFetcher
from .models.crypto_models import PredictionRequest, PredictionResponse, DataStatusResponse, HealthResponse

# Import ML components for Stage 3
from ..ml.prediction_pipeline import CryptoPredictionPipeline

app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for cryptocurrency price prediction using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
twitter_scraper = TwitterScraper()
reddit_scraper = RedditScraper()
binance_fetcher = BinancePriceFetcher()

# Initialize ML prediction pipeline
prediction_pipeline = CryptoPredictionPipeline()


@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "message": "Crypto Price Prediction API is up and running!",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_manager.is_connected()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        db_status = db_manager.is_connected()
        
        # Get basic stats
        btc_count = len(await db_manager.get_records('crypto_prices', {'currency': 'BTC'}))
        eth_count = len(await db_manager.get_records('crypto_prices', {'currency': 'ETH'}))
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "connected": db_status,
                "btc_records": btc_count,
                "eth_records": eth_count
            },
            "services": {
                "binance_api": "available",
                "twitter_scraper": "available",
                "reddit_scraper": "available"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/prices/{currency}")
async def get_prices(currency: str, days: int = 30):
    """Get historical price data for a currency"""
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get records from database
        records = await db_manager.get_records('crypto_prices', {
            'currency': currency.upper()
        })
        
        # Filter by date range and sort
        filtered_records = []
        for record in records:
            record_date = record['date']
            if isinstance(record_date, str):
                record_date = datetime.strptime(record_date, '%Y-%m-%d %H:%M:%S%z')
            if start_date <= record_date <= end_date:
                filtered_records.append({
                    'date': record['date'],
                    'open': float(record['open']),
                    'high': float(record['high']),
                    'low': float(record['low']),
                    'close': float(record['close']),
                    'volume': float(record['volume'])
                })
        
        # Sort by date
        filtered_records.sort(key=lambda x: x['date'])
        
        return {
            "currency": currency.upper(),
            "data": filtered_records,
            "count": len(filtered_records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")


@app.get("/sentiment/{currency}")
async def get_sentiment(currency: str, days: int = 30):
    """Get historical sentiment data for a currency"""
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get records from database
        records = await db_manager.get_records('crypto_sentiment', {
            'currency': currency.upper()
        })
        
        # Filter by date range
        filtered_records = []
        for record in records:
            record_date = record['date']
            if isinstance(record_date, str):
                record_date = datetime.strptime(record_date, '%Y-%m-%d %H:%M:%S%z')
            if start_date <= record_date <= end_date:
                filtered_records.append({
                    'date': record['date'],
                    'twitter_sentiment': float(record['twitter_sentiment']) if record['twitter_sentiment'] else None,
                    'reddit_sentiment': float(record['reddit_sentiment']) if record['reddit_sentiment'] else None
                })
        
        # Sort by date
        filtered_records.sort(key=lambda x: x['date'])
        
        return {
            "currency": currency.upper(),
            "data": filtered_records,
            "count": len(filtered_records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment: {str(e)}")


@app.get("/current_prices")
async def get_current_prices():
    """Get current prices for BTC and ETH"""
    try:
        # Fetch current prices
        btc_price = await binance_fetcher.get_current_price('BTCUSDT')
        eth_price = await binance_fetcher.get_current_price('ETHUSDT')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "BTC": btc_price,
            "ETH": eth_price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching current prices: {str(e)}")


@app.get("/data_status")
async def get_data_status():
    """Get status of available data in the database"""
    try:
        status = {}
        
        for currency in ['BTC', 'ETH']:
            # Get price data stats
            price_records = await db_manager.get_records('crypto_prices', {'currency': currency})
            sentiment_records = await db_manager.get_records('crypto_sentiment', {'currency': currency})
            
            latest_price_date = None
            latest_sentiment_date = None
            
            if price_records:
                dates = [r['date'] for r in price_records]
                latest_price_date = max(dates)
            
            if sentiment_records:
                dates = [r['date'] for r in sentiment_records]
                latest_sentiment_date = max(dates)
            
            status[currency] = {
                "price_records": len(price_records),
                "sentiment_records": len(sentiment_records),
                "latest_price_date": latest_price_date,
                "latest_sentiment_date": latest_sentiment_date
            }
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data status: {str(e)}")


# ===== STAGE 3: ML PREDICTION ENDPOINTS =====

@app.post("/predict/{currency}", response_model=PredictionResponse)
async def make_prediction(currency: str, request: PredictionRequest):
    """
    Make a price prediction for a cryptocurrency
    
    This endpoint uses trained ML models to predict if the price will go UP or DOWN
    over the specified prediction horizon (default: 7 days).
    """
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        # Use the prediction horizon from request, default to 7 days
        if hasattr(request, 'prediction_horizon'):
            prediction_horizon = request.prediction_horizon
        else:
            prediction_horizon = 7
        
        # Make prediction using the best available model
        prediction_result = await prediction_pipeline.make_prediction(
            currency=currency.upper(),
            model_type="best"
        )
        
        # Save prediction to database
        prediction_id = await prediction_pipeline.save_prediction(prediction_result)
        prediction_result['id'] = prediction_id
        
        # Format response
        response = PredictionResponse(
            currency=currency.upper(),
            prediction_date=datetime.now().isoformat(),
            prediction_horizon=prediction_horizon,
            predicted_direction=prediction_result['predicted_direction'],
            confidence_score=prediction_result['confidence_score'],
            model_version=prediction_result['model_version']
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predictions/{currency}")
async def get_predictions(currency: str, days: int = 30):
    """
    Get recent predictions for a cryptocurrency
    
    Returns historical predictions made by the ML models, useful for
    tracking prediction accuracy over time.
    """
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        # Get recent predictions
        predictions = await prediction_pipeline.get_recent_predictions(
            currency=currency.upper(),
            days=days
        )
        
        return {
            "currency": currency.upper(),
            "predictions": predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/prediction_accuracy/{currency}")
async def get_prediction_accuracy(currency: str, days: int = 30):
    """
    Evaluate the accuracy of recent predictions
    
    Compares predictions with actual price movements to calculate
    accuracy metrics and model performance statistics.
    """
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        # Evaluate prediction accuracy
        accuracy_results = await prediction_pipeline.evaluate_prediction_accuracy(
            currency=currency.upper(),
            days=days
        )
        
        return {
            "currency": currency.upper(),
            "accuracy_metrics": accuracy_results,
            "model_performance": {} # Placeholder, actual model performance will be added later
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating prediction accuracy: {str(e)}")


@app.post("/predictions/daily")
async def make_daily_predictions():
    """
    Generate daily predictions for both BTC and ETH
    
    This endpoint can be called daily (e.g., via cron job) to generate
    fresh predictions using the latest available data.
    """
    try:
        results = await prediction_pipeline.make_daily_predictions()
        
        return {
            "message": "Daily predictions completed",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making daily predictions: {str(e)}")


@app.get("/models/status")
async def get_model_status():
    """
    Get status of available trained models
    
    Returns information about which models are available for each currency,
    their training dates, and performance metrics.
    """
    try:
        import glob
        import os
        from ..ml.model_trainer import CryptoModelTrainer
        
        model_trainer = CryptoModelTrainer()
        models_dir = "models"
        
        if not os.path.exists(models_dir):
            return {"message": "No models directory found"}
        
        status = {}
        
        for currency in ['BTC', 'ETH']:
            status[currency] = {
                "available_models": [],
                "latest_models": {}
            }
            
            # Find model files for this currency
            pattern = os.path.join(models_dir, f"{currency}_*.pkl")
            model_files = glob.glob(pattern)
            
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace(f"{currency}_", "").replace(".pkl", "")
                status[currency]["available_models"].append(model_name)
                
                # Get model info
                try:
                    model_info = model_trainer.load_model_info(model_file)
                    status[currency]["latest_models"][model_name] = model_info
                except:
                    status[currency]["latest_models"][model_name] = {"status": "loaded"}
        
        return {
            "models_directory": models_dir,
            "currencies": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)