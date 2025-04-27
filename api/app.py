"""
FastAPI/Flask backend for model serving
Endpoints for predictions, data retrieval, and model info
"""
from fastapi import FastAPI, HTTPException, Query, WebSocket, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
from collections import defaultdict
import random
import numpy as np
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_fetchers.crypto_data_fetcher import CryptoDataFetcher
from data_fetchers.sentiment_fetcher import SentimentFetcher
from models.price_predictor import PricePredictor
from models.sentiment_model import SentimentModel

app = FastAPI(
    title="Crypto Predictor API",
    description="API for cryptocurrency price predictions and analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections and tasks
active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
update_tasks: Dict[str, asyncio.Task] = {}

# Mock price data
MOCK_PRICES = {
    'BTC': 45000,
    'ETH': 2800,
    'XRP': 0.58,
    'ADA': 1.20,
    'SOL': 98,
    'DOT': 15
}

def generate_market_update(symbol: str) -> Dict:
    """Generate a mock market update for a given symbol"""
    base_price = MOCK_PRICES.get(symbol, 1000)
    current_price = base_price * (1 + random.uniform(-0.02, 0.02))
    
    return {
        "type": "update",
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "price_data": {
            "open": base_price,
            "high": current_price * 1.02,
            "low": current_price * 0.98,
            "close": current_price,
            "volume": random.uniform(1000000, 5000000)
        },
        "prediction": {
            "price": current_price * (1 + random.uniform(-0.05, 0.15)),
            "confidence_interval": [
                current_price * 0.95,
                current_price * 1.15
            ]
        },
        "sentiment": random.uniform(0.3, 0.8)
    }

async def send_initial_update(websocket: WebSocket, symbol: str):
    """Send initial update immediately after connection"""
    try:
        update = generate_market_update(symbol)
        await websocket.send_json(update)
    except Exception as e:
        print(f"Error sending initial update: {e}")

# Background task for fetching real-time updates
async def fetch_and_broadcast_updates(symbol: str):
    """Background task to fetch and broadcast updates for a symbol"""
    while True:
        try:
            if not active_connections[symbol]:
                print(f"No active connections for {symbol}, stopping updates")
                break
                
            # Generate mock update
            update = generate_market_update(symbol)
            
            # Broadcast to all connected clients for this symbol
            dead_connections = []
            for connection in active_connections[symbol]:
                try:
                    await connection.send_json(update)
                except Exception as e:
                    print(f"Error sending to client: {e}")
                    dead_connections.append(connection)
            
            # Remove dead connections
            for dead_conn in dead_connections:
                if dead_conn in active_connections[symbol]:
                    active_connections[symbol].remove(dead_conn)
            
        except Exception as e:
            print(f"Error in update loop for {symbol}: {e}")
        
        # Wait before next update (5 seconds)
        await asyncio.sleep(5)
    
    # Clean up when the loop ends
    if symbol in update_tasks:
        del update_tasks[symbol]

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    print(f"WebSocket connection opened for {symbol}")
    
    # Add connection to active connections
    symbol = symbol.upper()
    active_connections[symbol].append(websocket)
    
    # Send initial update immediately
    await send_initial_update(websocket, symbol)
    
    # Start background task if not already running
    if symbol not in update_tasks or update_tasks[symbol].done():
        update_tasks[symbol] = asyncio.create_task(fetch_and_broadcast_updates(symbol))
    
    try:
        while True:
            # Handle incoming messages
            message = await websocket.receive_json()
            message_type = message.get('type')
            
            if message_type == 'subscribe':
                # Send immediate update
                await websocket.send_json({
                    "type": "subscribed",
                    "symbol": symbol,
                    "message": f"Successfully subscribed to {symbol} updates"
                })
            elif message_type == 'heartbeat':
                # Respond to heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print(f"WebSocket connection closed for {symbol}")
        if websocket in active_connections[symbol]:
            active_connections[symbol].remove(websocket)
            
        # If this was the last connection, cancel the update task
        if not active_connections[symbol] and symbol in update_tasks:
            update_tasks[symbol].cancel()
            del update_tasks[symbol]

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "24h"  # Options: 24h, 7d, 30d
    include_sentiment: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    confidence_interval: tuple[float, float]
    prediction_time: str
    sentiment_score: Optional[float] = None

class HistoricalDataRequest(BaseModel):
    symbol: str
    days: int = 30
    include_indicators: bool = False

@app.get("/")
async def root():
    """API health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """Get price prediction for a specific cryptocurrency"""
    try:
        # Get current price from the current-price endpoint
        fetcher = CryptoDataFetcher()
        
        # Fetch the latest data point
        data = fetcher.fetch_historical_data(request.symbol, days=1)
        
        if data is None or data.empty:
            current_price = MOCK_PRICES.get(request.symbol.upper(), 1000)
        else:
            # Get the latest price
            current_price = float(data.iloc[-1]['close'])
        
        # Initialize price predictor
        price_predictor = PricePredictor()
        
        # Get prediction and include sentiment if requested
        prediction = price_predictor.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            current_price=current_price,
            include_sentiment=request.include_sentiment
        )
        
        # Format response
        response = PredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            predicted_price=prediction['predicted_price'],
            confidence_interval=prediction['confidence_interval'],
            prediction_time=datetime.now().isoformat(),
            sentiment_score=prediction.get('sentiment_score')
        )
        
        return response
    except Exception as e:
        print(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(default=30, le=365),
    include_indicators: bool = False
):
    """Get historical price data for a cryptocurrency"""
    try:
        fetcher = CryptoDataFetcher()
        data = fetcher.fetch_historical_data(symbol, days)
        
        if include_indicators:
            # Add technical indicators
            data = fetcher.add_technical_indicators(data)
        
        return {
            "symbol": symbol,
            "data": data.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current-price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a cryptocurrency"""
    try:
        fetcher = CryptoDataFetcher()
        # Fetch the latest data point
        data = fetcher.fetch_historical_data(symbol, days=1)
        
        if data is None or data.empty:
            # Update mock prices with realistic current values
            mock_prices = {
                'BTC': 94500,
                'ETH': 3100,
                'XRP': 0.53,
                'ADA': 0.45,
                'SOL': 132,
                'DOT': 7.5
            }
            return {
                "symbol": symbol,
                "price": mock_prices.get(symbol.upper(), 1000)
            }
        
        # Return the latest price
        latest_price = float(data.iloc[-1]['close'])
        print(f"Current price for {symbol}: {latest_price}")
        return {
            "symbol": symbol,
            "price": latest_price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}")
async def get_sentiment_data(
    symbol: str,
    days: int = Query(default=7, le=30)
):
    """Get sentiment analysis data for a cryptocurrency"""
    try:
        # Check if we have a cached version first (used by the SentimentModel)
        cache_dir = 'data/sentiment'
        cache_file = os.path.join(cache_dir, f"{symbol.lower()}_sentiment.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is recent (within 5 minutes)
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                if datetime.now() - timedelta(minutes=5) < cache_time:
                    print(f"Using cached sentiment data for {symbol}")
                    # Return the cached data if the requested days is the same or less
                    if 'data' in cache_data and len(cache_data['data']) >= days:
                        return {
                            "symbol": symbol,
                            "sentiment_data": cache_data['data'][-days:],
                            "last_updated": cache_data['timestamp'],
                            "source": "cache"
                        }
            except Exception as e:
                print(f"Error reading sentiment cache: {e}")
        
        # If not in cache or cache expired, fetch fresh data
        sentiment_fetcher = SentimentFetcher()
        sentiment_data = sentiment_fetcher.get_historical_sentiment(symbol, days)
        
        # Process the data to ensure consistent format for frontend
        processed_data = []
        
        for item in sentiment_data:
            # Ensure each item has all the expected fields
            entry = {
                'date': item['date'],
                'sentiment': item['sentiment'],
                'source': item.get('source', 'API'),
                # Generate realistic mock data for fields that might be missing
                'volume': item.get('volume', 1000000 + (5000000 * abs(item['sentiment']))),
                'positive_mentions': item.get('positive_mentions', 10 * max(0.3, item['sentiment'])),
                'negative_mentions': item.get('negative_mentions', 10 * max(0.3, (1 - item['sentiment']))),
                'neutral_mentions': item.get('neutral_mentions', 5 + abs(item['sentiment'] * 3))
            }
            processed_data.append(entry)
        
        # Cache the processed data for future consistency
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'data': processed_data
        }
        
        # Ensure cache directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error writing sentiment cache: {e}")
        
        return {
            "symbol": symbol,
            "sentiment_data": processed_data,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching sentiment data for {symbol}: {e}")
        # Generate mock sentiment data
        mock_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate daily sentiment data with slight randomness
        base_sentiment = 0.2  # Slightly positive base sentiment
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            # Random sentiment value between -1 and 1, mostly positive
            sentiment = base_sentiment + np.random.normal(0, 0.3)
            sentiment = max(-1, min(1, sentiment))  # Clamp between -1 and 1
            
            # Add realistic mock data
            mock_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'sentiment': float(sentiment),
                'source': 'mock',
                'volume': 1000000 + (5000000 * abs(sentiment)),
                'positive_mentions': 10 * max(0.3, sentiment), 
                'negative_mentions': 10 * max(0.3, (1 - sentiment)),
                'neutral_mentions': 5 + abs(sentiment * 3)
            })
            
            # Slightly change the base sentiment for the next day
            base_sentiment += np.random.normal(0, 0.1)
            base_sentiment = max(-0.5, min(0.5, base_sentiment))
        
        return {
            "symbol": symbol,
            "sentiment_data": mock_data,
            "last_updated": datetime.now().isoformat(),
            "note": "Using mock data due to error fetching real sentiment data"
        }

@app.get("/models/info")
async def get_model_info():
    """Get information about available prediction models and their performance"""
    try:
        predictor = PricePredictor()
        return {
            "available_models": predictor.list_models(),
            "performance_metrics": predictor.get_performance_metrics(),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)