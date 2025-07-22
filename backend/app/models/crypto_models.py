from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

class PriceData(BaseModel):
    """Model for cryptocurrency price data"""
    id: Optional[str] = None
    currency: str = Field(..., pattern="^(BTC|ETH)$")
    date: datetime
    open: Decimal = Field(..., ge=0)
    high: Decimal = Field(..., ge=0)
    low: Decimal = Field(..., ge=0)
    close: Decimal = Field(..., ge=0)
    volume: Decimal = Field(..., ge=0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SentimentData(BaseModel):
    """Model for cryptocurrency sentiment data"""
    id: Optional[str] = None
    currency: str = Field(..., pattern="^(BTC|ETH)$")
    date: datetime
    twitter_sentiment: Optional[Decimal] = Field(None, ge=-1, le=1)
    reddit_sentiment: Optional[Decimal] = Field(None, ge=-1, le=1)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PredictionData(BaseModel):
    """Model for ML prediction data"""
    id: Optional[str] = None
    currency: str = Field(..., pattern="^(BTC|ETH)$")
    prediction_date: datetime
    prediction_horizon: int = Field(..., ge=1, le=30)  # Days ahead
    predicted_direction: str = Field(..., pattern="^(UP|DOWN)$")
    confidence_score: Decimal = Field(..., ge=0, le=1)
    model_version: Optional[str] = None
    features_used: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

class PriceResponse(BaseModel):
    """Response model for price data endpoints"""
    currency: str
    data: List[Dict[str, Any]]
    count: int

class SentimentResponse(BaseModel):
    """Response model for sentiment data endpoints"""
    currency: str
    twitter_sentiment: float
    reddit_sentiment: float

class PredictionRequest(BaseModel):
    """Request model for prediction endpoints"""
    currency: str = Field(..., pattern="^(BTC|ETH)$")
    prediction_horizon: int = Field(7, ge=1, le=30)

class PredictionResponse(BaseModel):
    """Response model for prediction endpoints"""
    currency: str
    prediction_date: datetime
    prediction_horizon: int
    predicted_direction: str
    confidence_score: float
    model_version: Optional[str] = None

class DataStatusResponse(BaseModel):
    """Response model for data status endpoint"""
    currency: str
    price_records: int
    sentiment_records: int
    latest_price_date: Optional[datetime] = None
    latest_sentiment_date: Optional[datetime] = None

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    message: str
    timestamp: datetime
    database_connected: bool

class ErrorResponse(BaseModel):
    """Standard error response model"""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now) 