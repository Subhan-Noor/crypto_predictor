# Enhanced FastAPI application with production-ready features
"""
Enhanced Crypto Price Prediction API

This module provides:
- Redis caching for improved performance
- Rate limiting for API protection
- WebSocket real-time updates
- Enhanced error handling and validation
- Background task processing
- Comprehensive monitoring and analytics
- Real sentiment analysis for crypto markets (credential-free)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
import uuid
import pytz
from dateutil import parser as date_parser

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import uvicorn

# Import services with graceful fallbacks for credential-free operation
from .services.twitter_service import twitter_service, TwitterSentimentService
from .services.reddit_service import reddit_service, RedditSentimentService
from .services.sentiment_analyzer import sentiment_analyzer

# Database and ML imports
from .database import db_manager
from .logger import logger
from ml.prediction_pipeline import CryptoPredictionPipeline
from ml.crypto_data_fetcher import CryptoDataFetcher

# Import enhanced models
from .models.api_models import (
    PaginationParams, DateRangeFilter, PriceFilter, SentimentFilter,
    EnhancedPriceResponse, EnhancedSentimentResponse, EnhancedPredictionResponse,
    PredictionRequest, APIHealthStatus, EnhancedErrorResponse
)

# Import configuration
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize services
binance_service = CryptoDataFetcher() # Changed from BinancePriceFetcher
prediction_pipeline = CryptoPredictionPipeline()
background_task_service = BackgroundTaskService()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Crypto Price Prediction API",
    description="Production-ready API with caching, rate limiting, real-time updates, and sentiment analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware_wrapper(request: Request, call_next):
    return await rate_limit_middleware(request, call_next)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Start WebSocket service
        await websocket_service.start_service()
        
        # Test Redis connection
        if cache_service.is_available():
            logger.info("✅ Redis cache connected")
        else:
            logger.warning("⚠️ Redis cache unavailable - using fallback mode")
        
        # Test rate limiter
        if rate_limiter.is_available():
            logger.info("✅ Rate limiter active")
        else:
            logger.warning("⚠️ Rate limiter unavailable - using fallback mode")
        
        logger.info("🚀 Enhanced Crypto Prediction API started successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await websocket_service.stop_service()
        logger.info("Enhanced API shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Enhanced error handlers
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
        content=jsonable_encoder(error_response),
        headers={"X-Request-ID": error_id}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Enhanced validation error handler"""
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "invalid_value": error.get("input")
        })
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(EnhancedErrorResponse(
            error="Validation Error",
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            validation_errors=validation_errors,
            timestamp=datetime.now(),
            path=request.url.path,
            method=request.method
        ))
    )

# Utility functions
def create_pagination_info(page: int, limit: int, total_items: int) -> Dict[str, Any]:
    """Create pagination metadata"""
    total_pages = (total_items + limit - 1) // limit
    return {
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

# PATCHED: Robust date handling for Supabase data

def apply_date_filter(records: List[Dict], date_filter: DateRangeFilter) -> List[Dict]:
    """Apply date filtering to records (robust for bad/missing dates)"""
    if not date_filter.start_date and not date_filter.end_date and not date_filter.days:
        return records
    
    filtered_records = []
    skipped = 0
    current_time = datetime.now(pytz.UTC)
    
    def to_utc(dt):
        if dt is None:
            return None
        if isinstance(dt, str):
            try:
                dt = date_parser.parse(dt)
            except Exception:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        else:
            dt = dt.astimezone(pytz.UTC)
        return dt
    
    start = to_utc(date_filter.start_date)
    end = to_utc(date_filter.end_date)
    
    # Calculate date range based on days parameter
    if date_filter.days and not start and not end:
        end = current_time
        start = current_time - timedelta(days=date_filter.days)
    
    for record in records:
        record_date = record.get("date")
        record_date_utc = to_utc(record_date)
        if not record_date_utc:
            skipped += 1
            continue
        
        include_record = True
        
        # Check start date
        if start and record_date_utc < start:
            include_record = False
        
        # Check end date
        if end and record_date_utc > end:
            include_record = False
        
        if include_record:
            filtered_records.append(record)
    
    if skipped > 0:
        logger.warning(f"apply_date_filter: Skipped {skipped} records with bad/missing dates out of {len(records)} total.")
    
    # Sort by date (newest first)
    filtered_records.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    return filtered_records

def paginate_data(data: List[Dict], page: int, limit: int, sort_by: str = "date", sort_order: str = "desc") -> tuple:
    """Paginate and sort data"""
    # Sort data
    reverse = sort_order.lower() == "desc"
    data.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)
    
    # Calculate pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = data[start_idx:end_idx]
    
    return paginated_data, len(data)

# Enhanced endpoints
@app.get("/", response_model=APIHealthStatus)
async def enhanced_root():
    """Enhanced root endpoint with comprehensive health status"""
    start_time = time.time()
    
    # Check database connection
    db_status = "healthy"
    try:
        await db_manager.test_connection()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check cache status
    cache_stats = cache_service.get_stats() if cache_service.is_available() else {"status": "unavailable"}
    
    # Check rate limiter status
    rate_limit_status = "active" if rate_limiter.is_available() else "unavailable"
    
    # Check WebSocket status
    websocket_stats = websocket_service.get_stats()
    
    response_time = (time.time() - start_time) * 1000
    
    return APIHealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="2.0.0",
        environment=settings.environment,
        services={
            "database": {"status": db_status},
            "cache": cache_stats,
            "rate_limiter": {"status": rate_limit_status},
            "websocket": websocket_stats
        },
        performance_metrics={
            "response_time_ms": round(response_time, 2)
        }
    )

@app.get("/health")
async def enhanced_health_check():
    """Detailed health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": settings.environment,
        "services": {}
    }
    
    # Database health
    try:
        await db_manager.test_connection()
        health_data["services"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_data["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Cache health
    if cache_service.is_available():
        health_data["services"]["cache"] = cache_service.get_stats()
    else:
        health_data["services"]["cache"] = {"status": "unavailable"}
    
    # Rate limiter health
    health_data["services"]["rate_limiter"] = {
        "status": "active" if rate_limiter.is_available() else "unavailable"
    }
    
    # WebSocket health
    health_data["services"]["websocket"] = websocket_service.get_stats()
    
    return health_data

# PATCH: In get_enhanced_prices, return 404 if no valid records after filtering
@app.get("/prices/{currency}", response_model=EnhancedPriceResponse)
async def get_enhanced_prices(
    currency: str,
    days: Optional[int] = 30,  # Direct days parameter for frontend compatibility
    start_date: Optional[str] = None,  # Optional start_date as string
    end_date: Optional[str] = None,    # Optional end_date as string
    pagination: PaginationParams = Depends(),
    price_filter: PriceFilter = Depends()
):
    """Enhanced price endpoint with caching, filtering, and pagination"""
    start_time = time.time()
    
    # Create DateRangeFilter from query parameters
    try:
        date_filter = DateRangeFilter(
            days=days,
            start_date=datetime.fromisoformat(start_date) if start_date else None,
            end_date=datetime.fromisoformat(end_date) if end_date else None
        )
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        date_filter = DateRangeFilter(days=days or 30)
    
    # Check cache first
    cache_key = f"prices:{currency}:{pagination.page}:{pagination.limit}:{days}"
    if cache_service.is_available():
        cached_data = cache_service.get_prices(
            currency, 
            date_filter.days or 30, 
            pagination.page, 
            pagination.limit
        )
        if cached_data:
            logger.info(f"Cache hit for {currency} prices (days: {days})")
            return EnhancedPriceResponse(**cached_data)
    # Fetch from database
    try:
        prices_data = await db_manager.get_crypto_prices(currency, limit=1000)
        if not prices_data:
            raise HTTPException(status_code=404, detail=f"No price data found for {currency}")
        filtered_data = apply_date_filter(prices_data, date_filter)
        if not filtered_data:
            logger.warning(f"No valid price records for {currency} after date filtering.")
            raise HTTPException(status_code=404, detail=f"No valid price data found for {currency} (bad/missing dates?)")
        
        # Apply price filtering
        if price_filter.min_price or price_filter.max_price or price_filter.min_volume or price_filter.max_volume:
            price_filtered = []
            for record in filtered_data:
                include = True
                
                if price_filter.min_price and record.get("close", 0) < float(price_filter.min_price):
                    include = False
                if price_filter.max_price and record.get("close", 0) > float(price_filter.max_price):
                    include = False
                if price_filter.min_volume and record.get("volume", 0) < float(price_filter.min_volume):
                    include = False
                if price_filter.max_volume and record.get("volume", 0) > float(price_filter.max_volume):
                    include = False
                
                if include:
                    price_filtered.append(record)
            filtered_data = price_filtered
        
        # Paginate data
        paginated_data, total_items = paginate_data(
            filtered_data, 
            pagination.page, 
            pagination.limit, 
            pagination.sort_by, 
            pagination.sort_order
        )
        
        # Create pagination info
        pagination_info = create_pagination_info(pagination.page, pagination.limit, total_items)
        
        # Calculate price summary
        if paginated_data:
            prices = [float(record.get("close", 0)) for record in paginated_data]
            volumes = [float(record.get("volume", 0)) for record in paginated_data]
            
            price_summary = {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "min_volume": min(volumes),
                "max_volume": max(volumes),
                "avg_volume": sum(volumes) / len(volumes)
            }
        else:
            price_summary = {}
        
        # Create response
        response = EnhancedPriceResponse(
            currency=currency,
            data=paginated_data,
            pagination=pagination_info,
            total_items=total_items,
            date_range={
                "start_date": paginated_data[-1]["date"] if paginated_data else None,
                "end_date": paginated_data[0]["date"] if paginated_data else None
            },
            price_summary=price_summary,
            count=len(paginated_data)
        )
        
        # Cache the response
        if cache_service.is_available():
            cache_service.set_prices(
                currency, 
                date_filter.days or 30, 
                pagination.page, 
                pagination.limit, 
                response.dict(),
                ttl=300  # 5 minutes
            )
        
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Enhanced prices for {currency}: {len(paginated_data)} records in {response_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching enhanced prices for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sentiment/{currency}", response_model=EnhancedSentimentResponse)
async def get_enhanced_sentiment(
    currency: str,
    days: Optional[int] = 30,  # Direct days parameter for frontend compatibility
    start_date: Optional[str] = None,  # Optional start_date as string
    end_date: Optional[str] = None,    # Optional end_date as string
    pagination: PaginationParams = Depends(),
    sentiment_filter: SentimentFilter = Depends()
):
    """Enhanced sentiment endpoint with caching, filtering, and pagination"""
    start_time = time.time()
    
    # Create DateRangeFilter from query parameters
    try:
        date_filter = DateRangeFilter(
            days=days,
            start_date=datetime.fromisoformat(start_date) if start_date else None,
            end_date=datetime.fromisoformat(end_date) if end_date else None
        )
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        date_filter = DateRangeFilter(days=days or 30)
    
    # Check cache first
    if cache_service.is_available():
        cached_data = cache_service.get_sentiment(currency, date_filter.days or 30)
        if cached_data:
            logger.info(f"Cache hit for {currency} sentiment (days: {days})")
            return EnhancedSentimentResponse(**cached_data)
    # Fetch from database
    try:
        sentiment_data = await db_manager.get_crypto_sentiment(currency, limit=1000)
        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {currency}")
        # Filter out records with invalid float values
        def safe_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        filtered_data = []
        for record in sentiment_data:
            record['twitter_sentiment'] = safe_float(record.get('twitter_sentiment'))
            record['reddit_sentiment'] = safe_float(record.get('reddit_sentiment'))
            filtered_data.append(record)
        # Apply date filtering
        filtered_data = apply_date_filter(filtered_data, date_filter)
        # Apply sentiment filtering
        if (sentiment_filter.min_twitter_sentiment or sentiment_filter.max_twitter_sentiment or 
            sentiment_filter.min_reddit_sentiment or sentiment_filter.max_reddit_sentiment):
            sentiment_filtered = []
            for record in filtered_data:
                include = True
                twitter_sentiment = record.get("twitter_sentiment")
                reddit_sentiment = record.get("reddit_sentiment")
                if sentiment_filter.min_twitter_sentiment and (twitter_sentiment is None or twitter_sentiment < sentiment_filter.min_twitter_sentiment):
                    include = False
                if sentiment_filter.max_twitter_sentiment and (twitter_sentiment is None or twitter_sentiment > sentiment_filter.max_twitter_sentiment):
                    include = False
                if sentiment_filter.min_reddit_sentiment and (reddit_sentiment is None or reddit_sentiment < sentiment_filter.min_reddit_sentiment):
                    include = False
                if sentiment_filter.max_reddit_sentiment and (reddit_sentiment is None or reddit_sentiment > sentiment_filter.max_reddit_sentiment):
                    include = False
                if include:
                    sentiment_filtered.append(record)
            filtered_data = sentiment_filtered
        # Paginate data
        paginated_data, total_items = paginate_data(
            filtered_data, 
            pagination.page, 
            pagination.limit, 
            pagination.sort_by, 
            pagination.sort_order
        )
        # Create pagination info
        pagination_info = create_pagination_info(pagination.page, pagination.limit, total_items)
        # Calculate sentiment summary
        if paginated_data:
            twitter_sentiments = [r["twitter_sentiment"] for r in paginated_data if r["twitter_sentiment"] is not None]
            reddit_sentiments = [r["reddit_sentiment"] for r in paginated_data if r["reddit_sentiment"] is not None]
            sentiment_summary = {
                "avg_twitter_sentiment": sum(twitter_sentiments) / len(twitter_sentiments) if twitter_sentiments else None,
                "avg_reddit_sentiment": sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else None,
                "twitter_sentiment_range": {
                    "min": min(twitter_sentiments) if twitter_sentiments else None,
                    "max": max(twitter_sentiments) if twitter_sentiments else None
                },
                "reddit_sentiment_range": {
                    "min": min(reddit_sentiments) if reddit_sentiments else None,
                    "max": max(reddit_sentiments) if reddit_sentiments else None
                }
            }
        else:
            sentiment_summary = {}
        # Create response
        response = EnhancedSentimentResponse(
            currency=currency,
            data=paginated_data,
            pagination=pagination_info,
            total_items=total_items,
            sentiment_summary=sentiment_summary,
            count=len(paginated_data)
        )
        # Cache the response
        if cache_service.is_available():
            cache_service.set_sentiment(currency, date_filter.days or 30, response.dict(), ttl=600)
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Enhanced sentiment for {currency}: {len(paginated_data)} records in {response_time:.2f}ms")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching enhanced sentiment for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- PATCH: Fix /current_prices to call binance_service.get_current_price synchronously ---
@app.get("/current_prices")
async def get_enhanced_current_prices():
    """Enhanced current prices endpoint with caching"""
    # Check cache first
    if cache_service.is_available():
        cached_data = cache_service.get_current_prices()
        if cached_data:
            return cached_data
    # Fetch current prices
    try:
        btc_price = binance_service.get_current_price("BTCUSDT")
        eth_price = binance_service.get_current_price("ETHUSDT")
        # Binance returns dicts with 'price' as string
        btc_price_val = float(btc_price["price"]) if isinstance(btc_price, dict) and "price" in btc_price else None
        eth_price_val = float(eth_price["price"]) if isinstance(eth_price, dict) and "price" in eth_price else None
        current_prices = {
            "timestamp": datetime.now().isoformat(),
            "prices": {
                "BTC": {
                    "price": btc_price_val,
                    "currency": "USD",
                    "symbol": "BTCUSDT"
                },
                "ETH": {
                    "price": eth_price_val,
                    "currency": "USD",
                    "symbol": "ETHUSDT"
                }
            }
        }
        # Cache the response
        if cache_service.is_available():
            cache_service.set_current_prices(current_prices, ttl=60)  # 1 minute TTL
        return current_prices
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        raise HTTPException(status_code=500, detail="Error fetching current prices")

# --- PATCH: Fix /predict/{currency} to use make_prediction ---
@app.post("/predict/{currency}", response_model=EnhancedPredictionResponse)
async def make_enhanced_prediction(
    currency: str, 
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Enhanced prediction endpoint with confidence intervals and feature importance"""
    start_time = time.time()
    # Check cache first
    if cache_service.is_available():
        cached_data = cache_service.get_prediction(currency, request.model_type)
        if cached_data:
            logger.info(f"Cache hit for {currency} prediction")
            return EnhancedPredictionResponse(**cached_data)
    try:
        # Make prediction and save to database
        prediction_result = await prediction_pipeline.make_and_save_prediction(
            currency=currency,
            model_type=request.model_type
        )
        if not prediction_result:
            raise HTTPException(status_code=500, detail="Failed to generate prediction")
        # Create enhanced response
        response = EnhancedPredictionResponse(
            currency=currency,
            prediction_date=datetime.now().isoformat(),
            prediction_horizon=7,
            predicted_direction=prediction_result.get("predicted_direction", "UNKNOWN"),
            confidence_score=prediction_result.get("confidence_score", 0.0),
            model_version=prediction_result.get("model_version", "unknown"),
            model_type=request.model_type,
            features_importance=None,
            confidence_interval=None,
            market_context={}
        )
        # Cache the response
        if cache_service.is_available():
            cache_service.set_prediction(currency, request.model_type, response.dict(), ttl=1800)
        # Broadcast prediction update via WebSocket
        background_tasks.add_task(
            websocket_service.broadcast_prediction_update,
            currency,
            response.dict()
        )
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Enhanced prediction for {currency}: {response.predicted_direction} in {response_time:.2f}ms")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making enhanced prediction for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Error generating prediction")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_service.handle_client_connection(websocket)

# Analytics endpoints
@app.get("/analytics/cache")
async def get_cache_analytics():
    """Get cache performance analytics"""
    return cache_service.get_stats()

@app.get("/analytics/rate_limits")
async def get_rate_limit_analytics(request: Request):
    """Get rate limit statistics for current client"""
    return rate_limiter.get_client_stats(request)

@app.get("/analytics/websocket")
async def get_websocket_analytics():
    """Get WebSocket connection analytics"""
    return websocket_service.get_stats()

# Background task endpoints
@app.post("/tasks/retrain_models")
async def start_model_retraining(background_tasks: BackgroundTasks):
    """Start model retraining in background"""
    try:
        task_id = background_task_service.create_task("model_retraining")
        background_tasks.add_task(background_task_service.start_task, task_id)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Model retraining started in background"
        }
    except Exception as e:
        logger.error(f"Error starting model retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to start model retraining")

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status"""
    task = background_task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/tasks")
async def list_background_tasks():
    """List all background tasks"""
    return {
        "tasks": background_task_service.list_tasks(),
        "total": len(background_task_service.list_tasks())
    }

# Background task implementation
async def run_model_retraining_task(task_id: str):
    """Background task for model retraining"""
    try:
        # Update task status to running
        background_task_service.update_task_status(task_id, "running")
        
        # Import training script
        from scripts.train_models import train_models_for_currency
        
        # Train models for both currencies
        results = {}
        for currency in ["BTC", "ETH"]:
            try:
                result = await train_models_for_currency(currency)
                results[currency] = result
            except Exception as e:
                results[currency] = {"error": str(e)}
        
        # Update task status to completed
        background_task_service.update_task_status(
            task_id, 
            "completed", 
            result={"results": results}
        )
        
        # Invalidate cache for predictions
        if cache_service.is_available():
            cache_service.invalidate_currency_cache("BTC")
            cache_service.invalidate_currency_cache("ETH")
        
        logger.info(f"Model retraining completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Model retraining failed for task {task_id}: {e}")
        background_task_service.update_task_status(task_id, "failed", error=str(e))

# Cache management endpoints
@app.post("/cache/invalidate/{currency}")
async def invalidate_currency_cache(currency: str):
    """Invalidate cache for specific currency"""
    if not cache_service.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")
    
    deleted_count = cache_service.invalidate_currency_cache(currency)
    return {
        "message": f"Cache invalidated for {currency}",
        "deleted_entries": deleted_count
    }

@app.post("/cache/clear")
async def clear_all_cache():
    """Clear all cache"""
    if not cache_service.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")
    
    # This would need to be implemented in cache_service
    return {"message": "Cache cleared successfully"}

# Keep existing endpoints for backward compatibility
@app.get("/prices/{currency}/basic")
async def get_basic_prices(currency: str, days: int = 30):
    """Basic price endpoint (backward compatibility)"""
    return await get_enhanced_prices(
        currency=currency,
        pagination=PaginationParams(page=1, limit=100),
        date_filter=DateRangeFilter(days=days),
        price_filter=PriceFilter()
    )

@app.get("/sentiment/{currency}/basic")
async def get_basic_sentiment(currency: str, days: int = 30):
    """Basic sentiment endpoint (backward compatibility)"""
    return await get_enhanced_sentiment(
        currency=currency,
        pagination=PaginationParams(page=1, limit=100),
        date_filter=DateRangeFilter(days=days),
        sentiment_filter=SentimentFilter()
    )

@app.post("/predict/{currency}/basic")
async def make_basic_prediction(currency: str):
    """Basic prediction endpoint (backward compatibility)"""
    return await make_enhanced_prediction(
        currency=currency,
        request=PredictionRequest(),
        background_tasks=BackgroundTasks()
    ) 

@app.get("/predictions/{currency}/history")
async def get_prediction_history(
    currency: str,
    days: int = 30,
    limit: int = 100
):
    """Get historical predictions for a currency"""
    try:
        # Get predictions from database
        predictions = await db_manager.get_predictions(currency, days, limit)
        
        return {
            "currency": currency,
            "predictions": predictions,
            "count": len(predictions),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error fetching prediction history for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching prediction history")

@app.get("/predictions/accuracy/{currency}")
async def get_prediction_accuracy(currency: str, days: int = 30):
    """Get prediction accuracy metrics for a currency"""
    try:
        # Get predictions with validation data
        predictions = await db_manager.get_predictions(currency, days, 1000)
        
        if not predictions:
            return {
                "currency": currency,
                "accuracy": 0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "validated_predictions": 0
            }
        
        # Calculate real accuracy from validation data
        validated_predictions = [p for p in predictions if p.get("is_correct") is not None]
        correct_predictions = [p for p in validated_predictions if p.get("is_correct") == True]
        
        accuracy = (len(correct_predictions) / len(validated_predictions) * 100) if validated_predictions else 0
        
        return {
            "currency": currency,
            "accuracy": accuracy,
            "total_predictions": len(predictions),
            "validated_predictions": len(validated_predictions),
            "correct_predictions": len(correct_predictions),
            "recent_predictions": predictions[:10],
            "prediction_distribution": {
                "up": len([p for p in predictions if p.get("predicted_direction") == "UP"]),
                "down": len([p for p in predictions if p.get("predicted_direction") == "DOWN"])
            }
        }
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Error calculating prediction accuracy") 

@app.get("/predictions/auto-validate")
async def auto_validate_predictions():
    """Automatically validate predictions that are ready for validation"""
    try:
        from scripts.auto_validate_predictions import AutoPredictionValidator
        
        validator = AutoPredictionValidator()
        summary = await validator.auto_validate_predictions()
        
        return {
            "status": "success",
            "message": f"Auto-validation complete. {summary['total_validated']} predictions validated.",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Auto-validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-validation failed: {str(e)}") 