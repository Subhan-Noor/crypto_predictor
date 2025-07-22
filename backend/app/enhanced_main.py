"""
Enhanced Crypto Price Prediction API with Stage 4 Features

This module includes:
- Redis caching for performance optimization
- Rate limiting for API protection
- WebSocket support for real-time updates
- Advanced filtering and pagination
- Enhanced error handling and monitoring
- Background task processing
"""

import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, WebSocket, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Import original components
from .database import db_manager
from .services.twitter_service import TwitterScraper
from .services.reddit_service import RedditScraper
from .services.binance_service import BinancePriceFetcher
from .models.crypto_models import PredictionRequest, PredictionResponse

# Import Stage 4 enhancements
from .services.cache_service import cache_service
from .middleware.rate_limiter import rate_limit_middleware, rate_limiter
from .services.websocket_service import websocket_service
from .models.api_models import (
    PaginationParams, DateRangeFilter, PriceFilter, SentimentFilter,
    EnhancedPriceResponse, EnhancedSentimentResponse, EnhancedPredictionResponse,
    APIHealthStatus, EnhancedErrorResponse, TaskStatus, BackgroundTask,
    PredictionRequest as EnhancedPredictionRequest
)

# Import ML components
from ..ml.prediction_pipeline import CryptoPredictionPipeline

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Enhanced Crypto Price Prediction API",
    description="""
    Advanced cryptocurrency price prediction API with ML models.
    
    Features:
    - Real-time price predictions using ML models
    - Historical data analysis with advanced filtering
    - Real-time WebSocket updates
    - Redis caching for optimal performance
    - Rate limiting for API protection
    - Background task processing
    - Comprehensive monitoring and analytics
    """,
    version="2.0.0",
    contact={
        "name": "Crypto Prediction API",
        "email": "admin@cryptoprediction.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.vercel.app",  # For production frontend
        "https://*.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Initialize services
twitter_scraper = TwitterScraper()
reddit_scraper = RedditScraper()
binance_fetcher = BinancePriceFetcher()
prediction_pipeline = CryptoPredictionPipeline()

# Background tasks storage
background_tasks: Dict[str, BackgroundTask] = {}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await websocket_service.start_service()
    print("🚀 Enhanced Crypto Prediction API started successfully!")
    print(f"📊 Cache Status: {'✅ Connected' if cache_service.is_available() else '❌ Disconnected'}")
    print(f"⚡ Rate Limiter: {'✅ Active' if rate_limiter.is_available() else '❌ Inactive'}")
    print(f"🔗 WebSocket Service: ✅ Running")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await websocket_service.stop_service()
    print("👋 Enhanced Crypto Prediction API shutdown complete")

# Enhanced error handler
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced error handling with detailed responses"""
    error_id = str(uuid.uuid4())[:8]
    
    error_response = EnhancedErrorResponse(
        error="HTTP Exception",
        error_code=f"HTTP_{exc.status_code}",
        message=exc.detail,
        timestamp=datetime.now(),
        path=request.url.path,
        method=request.method,
        request_id=error_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers={"X-Request-ID": error_id}
    )

# Utility functions
def create_pagination_info(page: int, limit: int, total_items: int) -> Dict[str, Any]:
    """Create pagination metadata"""
    total_pages = (total_items + limit - 1) // limit
    return {
        "current_page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

def apply_date_filter(records: List[Dict], date_filter: DateRangeFilter) -> List[Dict]:
    """Apply date filtering to records"""
    if not date_filter.start_date and not date_filter.end_date and not date_filter.days:
        return records
    
    # Calculate date range
    if date_filter.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_filter.days)
    else:
        start_date = date_filter.start_date
        end_date = date_filter.end_date
    
    filtered_records = []
    for record in records:
        record_date = record['date']
        if isinstance(record_date, str):
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d %H:%M:%S%z')
            except ValueError:
                try:
                    record_date = datetime.strptime(record_date, '%Y-%m-%d')
                except ValueError:
                    continue
        
        if start_date and record_date < start_date:
            continue
        if end_date and record_date > end_date:
            continue
        
        filtered_records.append(record)
    
    return filtered_records

def paginate_data(data: List[Dict], page: int, limit: int, sort_by: str = "date", sort_order: str = "desc") -> tuple:
    """Paginate and sort data"""
    # Sort data
    reverse = sort_order.lower() == "desc"
    try:
        sorted_data = sorted(data, key=lambda x: x.get(sort_by, ""), reverse=reverse)
    except:
        sorted_data = data  # Fallback if sorting fails
    
    # Paginate
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = sorted_data[start_idx:end_idx]
    
    return paginated_data, len(sorted_data)

# ===== ENHANCED HEALTH AND STATUS ENDPOINTS =====

@app.get("/", response_model=APIHealthStatus)
async def enhanced_root():
    """Enhanced health check endpoint with comprehensive status"""
    cache_stats = cache_service.get_stats() if cache_service.is_available() else None
    
    # Get database stats
    try:
        btc_count = len(await db_manager.get_records('crypto_prices', {'currency': 'BTC'}))
        eth_count = len(await db_manager.get_records('crypto_prices', {'currency': 'ETH'}))
        db_status = "healthy"
    except Exception as e:
        btc_count = 0
        eth_count = 0
        db_status = f"error: {str(e)}"
    
    return APIHealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="2.0.0",
        environment=settings.environment,
        services={
            "database": {
                "status": db_status,
                "btc_records": btc_count,
                "eth_records": eth_count
            },
            "cache": {
                "status": "connected" if cache_service.is_available() else "disconnected",
                "redis_url": settings.redis_url if cache_service.is_available() else None
            },
            "rate_limiter": {
                "status": "active" if rate_limiter.is_available() else "inactive"
            },
            "websocket": websocket_service.get_stats(),
            "ml_pipeline": {
                "status": "available",
                "models_loaded": "ready"
            }
        },
        performance_metrics={
            "uptime": "calculated_dynamically",
            "memory_usage": "available_via_cache"
        },
        cache_stats=cache_stats
    )

@app.get("/health")
async def enhanced_health_check():
    """Detailed health check with performance metrics"""
    start_time = time.time()
    
    # Test database
    try:
        db_test = await db_manager.get_records('crypto_prices', {'currency': 'BTC'})
        db_latency = (time.time() - start_time) * 1000
        db_status = "healthy"
    except Exception as e:
        db_latency = None
        db_status = f"error: {str(e)}"
    
    # Test cache
    cache_test_start = time.time()
    cache_available = cache_service.is_available()
    cache_latency = (time.time() - cache_test_start) * 1000 if cache_available else None
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "database": {
                "status": db_status,
                "latency_ms": db_latency
            },
            "cache": {
                "status": "connected" if cache_available else "disconnected",
                "latency_ms": cache_latency
            },
            "rate_limiter": {
                "status": "active" if rate_limiter.is_available() else "inactive"
            },
            "websocket": {
                "status": "running" if websocket_service.is_running else "stopped",
                "connections": len(websocket_service.connection_manager.active_connections)
            }
        },
        "cache_stats": cache_service.get_stats(),
        "background_tasks": {
            "total": len(background_tasks),
            "running": len([t for t in background_tasks.values() if t.status == TaskStatus.RUNNING])
        }
    }

# ===== ENHANCED DATA ENDPOINTS WITH CACHING AND PAGINATION =====

@app.get("/prices/{currency}", response_model=EnhancedPriceResponse)
async def get_enhanced_prices(
    currency: str,
    pagination: PaginationParams = Depends(),
    date_filter: DateRangeFilter = Depends(),
    price_filter: PriceFilter = Depends()
):
    """Get historical price data with enhanced filtering and pagination"""
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        currency = currency.upper()
        
        # Check cache first
        cache_key = f"enhanced_prices_{currency}_{pagination.page}_{pagination.limit}_{date_filter.days or 30}"
        cached_data = cache_service.get_prices(currency, date_filter.days or 30, pagination.page, pagination.limit)
        
        if cached_data:
            return EnhancedPriceResponse(**cached_data)
        
        # Get records from database
        records = await db_manager.get_records('crypto_prices', {'currency': currency})
        
        # Apply date filtering
        filtered_records = apply_date_filter(records, date_filter)
        
        # Apply price filtering
        if price_filter.min_price or price_filter.max_price or price_filter.min_volume or price_filter.max_volume:
            price_filtered = []
            for record in filtered_records:
                if price_filter.min_price and float(record['close']) < price_filter.min_price:
                    continue
                if price_filter.max_price and float(record['close']) > price_filter.max_price:
                    continue
                if price_filter.min_volume and float(record['volume']) < price_filter.min_volume:
                    continue
                if price_filter.max_volume and float(record['volume']) > price_filter.max_volume:
                    continue
                price_filtered.append(record)
            filtered_records = price_filtered
        
        # Paginate data
        paginated_data, total_items = paginate_data(
            filtered_records, pagination.page, pagination.limit, 
            pagination.sort_by, pagination.sort_order
        )
        
        # Format data
        formatted_data = []
        for record in paginated_data:
            formatted_data.append({
                'date': record['date'],
                'open': float(record['open']),
                'high': float(record['high']),
                'low': float(record['low']),
                'close': float(record['close']),
                'volume': float(record['volume'])
            })
        
        # Calculate price summary
        if formatted_data:
            prices = [item['close'] for item in formatted_data]
            price_summary = {
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': sum(prices) / len(prices),
                'price_change': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0
            }
        else:
            price_summary = {}
        
        # Create response
        pagination_info = create_pagination_info(pagination.page, pagination.limit, total_items)
        
        response_data = {
            "currency": currency,
            "data": formatted_data,
            "pagination": pagination_info,
            "total_items": total_items,
            "date_range": {
                "start_date": formatted_data[0]['date'] if formatted_data else None,
                "end_date": formatted_data[-1]['date'] if formatted_data else None
            },
            "price_summary": price_summary,
            "count": len(formatted_data)
        }
        
        # Cache the response
        cache_service.set_prices(currency, date_filter.days or 30, pagination.page, pagination.limit, response_data)
        
        return EnhancedPriceResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")

@app.get("/sentiment/{currency}", response_model=EnhancedSentimentResponse)
async def get_enhanced_sentiment(
    currency: str,
    pagination: PaginationParams = Depends(),
    date_filter: DateRangeFilter = Depends(),
    sentiment_filter: SentimentFilter = Depends()
):
    """Get historical sentiment data with enhanced filtering"""
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        currency = currency.upper()
        
        # Check cache first
        cached_data = cache_service.get_sentiment(currency, date_filter.days or 30)
        if cached_data and not sentiment_filter.min_twitter_sentiment:  # Only use cache for simple queries
            return EnhancedSentimentResponse(**cached_data)
        
        # Get records from database
        records = await db_manager.get_records('crypto_sentiment', {'currency': currency})
        
        # Apply date filtering
        filtered_records = apply_date_filter(records, date_filter)
        
        # Apply sentiment filtering
        if (sentiment_filter.min_twitter_sentiment is not None or 
            sentiment_filter.max_twitter_sentiment is not None or
            sentiment_filter.min_reddit_sentiment is not None or
            sentiment_filter.max_reddit_sentiment is not None):
            
            sentiment_filtered = []
            for record in filtered_records:
                twitter_sentiment = float(record['twitter_sentiment']) if record['twitter_sentiment'] else None
                reddit_sentiment = float(record['reddit_sentiment']) if record['reddit_sentiment'] else None
                
                if (sentiment_filter.min_twitter_sentiment is not None and 
                    (twitter_sentiment is None or twitter_sentiment < sentiment_filter.min_twitter_sentiment)):
                    continue
                if (sentiment_filter.max_twitter_sentiment is not None and 
                    (twitter_sentiment is None or twitter_sentiment > sentiment_filter.max_twitter_sentiment)):
                    continue
                if (sentiment_filter.min_reddit_sentiment is not None and 
                    (reddit_sentiment is None or reddit_sentiment < sentiment_filter.min_reddit_sentiment)):
                    continue
                if (sentiment_filter.max_reddit_sentiment is not None and 
                    (reddit_sentiment is None or reddit_sentiment > sentiment_filter.max_reddit_sentiment)):
                    continue
                
                sentiment_filtered.append(record)
            filtered_records = sentiment_filtered
        
        # Paginate data
        paginated_data, total_items = paginate_data(
            filtered_records, pagination.page, pagination.limit,
            pagination.sort_by, pagination.sort_order
        )
        
        # Format data
        formatted_data = []
        for record in paginated_data:
            formatted_data.append({
                'date': record['date'],
                'twitter_sentiment': float(record['twitter_sentiment']) if record['twitter_sentiment'] else None,
                'reddit_sentiment': float(record['reddit_sentiment']) if record['reddit_sentiment'] else None
            })
        
        # Calculate sentiment summary
        twitter_sentiments = [item['twitter_sentiment'] for item in formatted_data if item['twitter_sentiment'] is not None]
        reddit_sentiments = [item['reddit_sentiment'] for item in formatted_data if item['reddit_sentiment'] is not None]
        
        sentiment_summary = {
            'avg_twitter_sentiment': sum(twitter_sentiments) / len(twitter_sentiments) if twitter_sentiments else None,
            'avg_reddit_sentiment': sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else None,
            'twitter_data_points': len(twitter_sentiments),
            'reddit_data_points': len(reddit_sentiments)
        }
        
        # Create response
        pagination_info = create_pagination_info(pagination.page, pagination.limit, total_items)
        
        response_data = {
            "currency": currency,
            "data": formatted_data,
            "pagination": pagination_info,
            "total_items": total_items,
            "sentiment_summary": sentiment_summary,
            "count": len(formatted_data)
        }
        
        # Cache the response (only simple queries)
        if not sentiment_filter.min_twitter_sentiment:
            cache_service.set_sentiment(currency, date_filter.days or 30, response_data)
        
        return EnhancedSentimentResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment: {str(e)}")

@app.get("/current_prices")
async def get_enhanced_current_prices():
    """Get current prices with caching"""
    try:
        # Check cache first
        cached_prices = cache_service.get_current_prices()
        if cached_prices:
            return cached_prices
        
        # Fetch current prices
        btc_price = await binance_fetcher.get_current_price('BTCUSDT')
        eth_price = await binance_fetcher.get_current_price('ETHUSDT')
        
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "BTC": btc_price,
            "ETH": eth_price
        }
        
        # Cache for 1 minute
        cache_service.set_current_prices(response_data, ttl=60)
        
        # Broadcast to WebSocket clients
        await websocket_service.broadcast_price_update("BTC", btc_price)
        await websocket_service.broadcast_price_update("ETH", eth_price)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching current prices: {str(e)}")

# ===== ENHANCED PREDICTION ENDPOINTS =====

@app.post("/predict/{currency}", response_model=EnhancedPredictionResponse)
async def make_enhanced_prediction(
    currency: str, 
    request: EnhancedPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make enhanced price prediction with caching and real-time updates"""
    try:
        if currency.upper() not in ['BTC', 'ETH']:
            raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
        
        currency = currency.upper()
        
        # Check cache first
        cached_prediction = cache_service.get_prediction(currency, request.model_type)
        if cached_prediction:
            # Still broadcast the cached prediction
            background_tasks.add_task(
                websocket_service.broadcast_prediction_update,
                currency, cached_prediction
            )
            return EnhancedPredictionResponse(**cached_prediction)
        
        # Make prediction using ML pipeline
        prediction_result = await prediction_pipeline.make_prediction(
            currency=currency,
            model_type=request.model_type if request.model_type != "best" else "best"
        )
        
        # Save prediction to database
        prediction_id = await prediction_pipeline.save_prediction(prediction_result)
        prediction_result['id'] = prediction_id
        
        # Create enhanced response
        response_data = {
            "currency": currency,
            "prediction_date": datetime.now().isoformat(),
            "prediction_horizon": request.prediction_horizon,
            "predicted_direction": prediction_result['predicted_direction'],
            "confidence_score": prediction_result['confidence_score'],
            "model_version": prediction_result['model_version'],
            "model_type": request.model_type,
            "features_importance": prediction_result.get('features_importance') if request.include_features else None,
            "confidence_interval": prediction_result.get('confidence_interval') if request.include_confidence else None,
            "market_context": {
                "current_timestamp": datetime.now().isoformat(),
                "model_used": prediction_result.get('model_name', request.model_type)
            }
        }
        
        # Cache the prediction
        cache_service.set_prediction(currency, request.model_type, response_data, ttl=1800)  # 30 minutes
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(
            websocket_service.broadcast_prediction_update,
            currency, response_data
        )
        
        return EnhancedPredictionResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# ===== WEBSOCKET ENDPOINT =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_service.handle_client_connection(websocket)

# ===== ANALYTICS AND MONITORING ENDPOINTS =====

@app.get("/analytics/cache")
async def get_cache_analytics():
    """Get cache performance analytics"""
    if not cache_service.is_available():
        return {"status": "cache_unavailable"}
    
    return cache_service.get_stats()

@app.get("/analytics/rate_limits")
async def get_rate_limit_analytics(request: Request):
    """Get rate limit statistics for current client"""
    return rate_limiter.get_client_stats(request)

@app.get("/analytics/websocket")
async def get_websocket_analytics():
    """Get WebSocket connection analytics"""
    return websocket_service.get_stats()

# ===== BACKGROUND TASK ENDPOINTS =====

@app.post("/tasks/retrain_models")
async def start_model_retraining(background_tasks: BackgroundTasks):
    """Start background model retraining task"""
    task_id = str(uuid.uuid4())
    
    task = BackgroundTask(
        task_id=task_id,
        task_type="model_retraining",
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )
    
    background_tasks[task_id] = task
    
    # Add the actual background task
    background_tasks.add_task(run_model_retraining_task, task_id)
    
    return {"task_id": task_id, "status": "started", "message": "Model retraining task started"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status"""
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks[task_id]

@app.get("/tasks")
async def list_background_tasks():
    """List all background tasks"""
    return {
        "tasks": list(background_tasks.values()),
        "total": len(background_tasks)
    }

# Background task functions
async def run_model_retraining_task(task_id: str):
    """Background task for model retraining"""
    if task_id not in background_tasks:
        return
    
    task = background_tasks[task_id]
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now()
    
    try:
        # Simulate model retraining (replace with actual retraining logic)
        await asyncio.sleep(10)  # Simulated work
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = {"message": "Models retrained successfully"}
        
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.completed_at = datetime.now()

# ===== CACHE MANAGEMENT ENDPOINTS =====

@app.post("/cache/invalidate/{currency}")
async def invalidate_currency_cache(currency: str):
    """Invalidate cache for a specific currency"""
    if currency.upper() not in ['BTC', 'ETH']:
        raise HTTPException(status_code=400, detail="Currency must be BTC or ETH")
    
    deleted_count = cache_service.invalidate_currency_cache(currency.upper())
    
    return {
        "message": f"Cache invalidated for {currency.upper()}",
        "deleted_entries": deleted_count
    }

@app.post("/cache/clear")
async def clear_all_cache():
    """Clear all cache (admin endpoint)"""
    # This should be protected with authentication in production
    if cache_service.is_available():
        # Implementation depends on your cache clearing strategy
        return {"message": "Cache clear initiated"}
    else:
        return {"message": "Cache not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 