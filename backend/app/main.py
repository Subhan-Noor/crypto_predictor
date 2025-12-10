# Enhanced FastAPI application with improved date filtering
"""
Crypto Price Prediction API with enhanced date filtering

This module provides:
- Improved date filtering for time ranges
- Better error handling and validation
- Enhanced response formatting
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

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.encoders import jsonable_encoder

# Import existing services
from .database import db_manager
from .services.binance_service import BinancePriceFetcher

# Import models
from .models.api_models import (
    PaginationParams, DateRangeFilter, PriceFilter, SentimentFilter,
    EnhancedPriceResponse, EnhancedSentimentResponse, EnhancedPredictionResponse,
    PredictionRequest, APIHealthStatus, EnhancedErrorResponse
)

# Import ML components
from ml.prediction_pipeline import CryptoPredictionPipeline

# Import configuration
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize services
binance_service = BinancePriceFetcher()
prediction_pipeline = CryptoPredictionPipeline()

# Create FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for cryptocurrency price predictions with enhanced date filtering",
    version="1.0.0",
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

# Enhanced date filtering function
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

def create_pagination_info(page: int, limit: int, total_items: int) -> Dict[str, Any]:
    """Create pagination information"""
    total_pages = (total_items + limit - 1) // limit
    return {
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

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
    
    error_id = str(uuid.uuid4())[:8]
    error_response = EnhancedErrorResponse(
        error="Validation Error",
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        timestamp=datetime.now(),
        path=request.url.path,
        method=request.method,
        request_id=error_id,
        details=validation_errors
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(error_response),
        headers={"X-Request-ID": error_id}
    )

# Enhanced endpoints
@app.get("/", response_model=APIHealthStatus)
async def root():
    """Root endpoint with health status"""
    start_time = time.time()
    
    # Check database connection
    db_status = "healthy"
    try:
        await db_manager.test_connection()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    response_time = (time.time() - start_time) * 1000
    
    return APIHealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        environment=settings.environment,
        services={
            "database": {"status": db_status},
        },
        performance_metrics={
            "response_time_ms": round(response_time, 2)
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "connected": db_manager.is_connected(),
            "btc_records": 0,
            "eth_records": 0
        },
        "services": {
            "binance_api": "available",
            "twitter_scraper": "unavailable",
            "reddit_scraper": "unavailable"
        }
    }
    
    # Get record counts if connected
    if db_manager.is_connected():
        try:
            btc_data = await db_manager.get_crypto_prices("BTC", limit=1)
            eth_data = await db_manager.get_crypto_prices("ETH", limit=1)
            health_data["database"]["btc_records"] = len(btc_data) if btc_data else 0
            health_data["database"]["eth_records"] = len(eth_data) if eth_data else 0
        except Exception as e:
            logger.error(f"Error getting record counts: {e}")
    
    return health_data

@app.get("/prices/{currency}", response_model=EnhancedPriceResponse)
async def get_enhanced_prices(
    currency: str,
    days: Optional[int] = 30,  # Direct days parameter for frontend compatibility
    start_date: Optional[str] = None,  # Optional start_date as string
    end_date: Optional[str] = None,    # Optional end_date as string
    pagination: PaginationParams = Depends(),
    price_filter: PriceFilter = Depends()
):
    """Enhanced price endpoint with improved date filtering"""
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
        
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Enhanced prices for {currency}: {len(paginated_data)} records in {response_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching enhanced prices for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/current_prices")
async def get_current_prices():
    """Get current prices for all supported currencies"""
    try:
        currencies = ["BTC", "ETH"]
        current_prices = {}
        
        for currency in currencies:
            try:
                # Get latest price data
                price_data = await db_manager.get_crypto_prices(currency, limit=1)
                if price_data and len(price_data) > 0:
                    latest = price_data[0]
                    current_prices[currency] = {
                        "currency": currency,
                        "price": float(latest.get("close", 0)),
                        "change_24h": 0.0,  # Placeholder
                        "change_percentage_24h": 0.0,  # Placeholder
                        "volume": float(latest.get("volume", 0)),
                        "high_24h": float(latest.get("high", 0)),
                        "low_24h": float(latest.get("low", 0)),
                        "last_updated": latest.get("date")
                    }
                else:
                    current_prices[currency] = {
                        "currency": currency,
                        "price": 0.0,
                        "change_24h": 0.0,
                        "change_percentage_24h": 0.0,
                        "volume": 0.0,
                        "high_24h": 0.0,
                        "low_24h": 0.0,
                        "last_updated": None
                    }
            except Exception as e:
                logger.error(f"Error fetching current price for {currency}: {e}")
                current_prices[currency] = {
                    "currency": currency,
                    "price": 0.0,
                    "change_24h": 0.0,
                    "change_percentage_24h": 0.0,
                    "volume": 0.0,
                    "high_24h": 0.0,
                    "low_24h": 0.0,
                    "last_updated": None
                }
        
        return current_prices
        
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/{currency}", response_model=EnhancedPredictionResponse)
async def make_prediction(currency: str, request: PredictionRequest):
    """Make prediction for a currency"""
    try:
        # Make prediction using the pipeline
        prediction_result = await prediction_pipeline.make_prediction(
            currency=currency,
            model_type=request.model_type
        )
        
        # Save prediction to database
        await prediction_pipeline.save_prediction(prediction_result)
        
        return EnhancedPredictionResponse(
            currency=currency,
            predicted_direction=prediction_result["prediction"],
            confidence_score=prediction_result["confidence"],
            prediction_date=prediction_result["target_date"],
            features_importance=prediction_result.get("features", {}),
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Error making prediction for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sentiment/{currency}", response_model=EnhancedSentimentResponse)
async def get_sentiment(currency: str, days: int = 30):
    """Get sentiment data for a currency"""
    try:
        sentiment_data = await db_manager.get_crypto_sentiment(currency, limit=1000)
        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {currency}")
        
        # Create date filter
        date_filter = DateRangeFilter(days=days)
        filtered_data = apply_date_filter(sentiment_data, date_filter)
        
        return EnhancedSentimentResponse(
            currency=currency,
            data=filtered_data,
            count=len(filtered_data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {currency}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")