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

app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for cryptocurrency price prediction using ML and sentiment analysis",
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
price_fetcher = BinancePriceFetcher()
twitter_scraper = TwitterScraper()
reddit_scraper = RedditScraper()


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
    return {
        "status": "healthy",
        "database": "connected" if db_manager.is_connected() else "disconnected",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment
    }


@app.get("/prices/{currency}")
async def get_crypto_prices(
    currency: str,
    limit: int = 30,
    days: Optional[int] = None
):
    """Get historical price data for a cryptocurrency"""
    if currency.upper() not in ["BTC", "ETH"]:
        raise HTTPException(status_code=400, detail="Supported currencies: BTC, ETH")
    
    try:
        symbol = f"{currency.upper()}USDT"
        historical_data = price_fetcher.get_historical_prices(symbol, limit=limit)
        
        return {
            "currency": currency.upper(),
            "data": historical_data,
            "count": len(historical_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")


@app.get("/sentiment/{currency}")
async def get_crypto_sentiment(
    currency: str,
    limit: int = 30
):
    """Get historical sentiment data for a cryptocurrency"""
    if currency.upper() not in ["BTC", "ETH"]:
        raise HTTPException(status_code=400, detail="Supported currencies: BTC, ETH")
    
    try:
        twitter_sentiment = twitter_scraper.get_crypto_sentiment(currency)
        reddit_sentiment = reddit_scraper.get_crypto_sentiment(currency)
        
        return {
            "currency": currency.upper(),
            "twitter_sentiment": twitter_sentiment,
            "reddit_sentiment": reddit_sentiment
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment data: {str(e)}")


@app.get("/current_prices")
async def get_current_prices():
    """Get current prices for all supported currencies"""
    try:
        prices = {}
        
        for currency in ["BTC", "ETH"]:
            symbol = f"{currency.upper()}USDT"
            current_price = price_fetcher.get_current_price(symbol)
            if current_price:
                prices[currency] = current_price
        
        return prices
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching current prices: {str(e)}")


@app.post("/predict/{currency}")
async def predict_price_movement(currency: str):
    """Predict price movement for a cryptocurrency (placeholder - will be implemented in Stage 3)"""
    if currency.upper() not in ["BTC", "ETH"]:
        raise HTTPException(status_code=400, detail="Supported currencies: BTC, ETH")
    
    # Placeholder response - will be implemented with ML model in Stage 3
    return {
        "currency": currency.upper(),
        "prediction": "up",  # placeholder
        "confidence": 0.75,  # placeholder
        "target_date": (datetime.now() + timedelta(days=7)).isoformat(),
        "message": "This is a placeholder. ML model will be implemented in Stage 3."
    }


@app.get("/data_status")
async def get_data_status():
    """Get status of available data in the database"""
    db_client = db_manager.get_client()
    if not db_client:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    try:
        status = {}
        
        for currency in ["BTC", "ETH"]:
            # Count price records
            price_count = db_client.table("crypto_prices").select("id", count="exact").eq("currency", currency).execute()
            
            # Count sentiment records
            sentiment_count = db_client.table("crypto_sentiment").select("id", count="exact").eq("currency", currency).execute()
            
            # Get latest dates
            latest_price = db_client.table("crypto_prices").select("date").eq("currency", currency).order("date", desc=True).limit(1).execute()
            latest_sentiment = db_client.table("crypto_sentiment").select("date").eq("currency", currency).order("date", desc=True).limit(1).execute()
            
            status[currency] = {
                "price_records": price_count.count,
                "sentiment_records": sentiment_count.count,
                "latest_price_date": latest_price.data[0]["date"] if latest_price.data else None,
                "latest_sentiment_date": latest_sentiment.data[0]["date"] if latest_sentiment.data else None
            }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)