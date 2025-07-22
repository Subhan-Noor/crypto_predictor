"""
Enhanced API Models for Crypto Price Prediction API

This module provides:
- Pagination models
- Filtering models
- Enhanced response models
- WebSocket message models
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum


class SortOrder(str, Enum):
    """Sort order enumeration"""
    ASC = "asc"
    DESC = "desc"


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (starts from 1)")
    limit: int = Field(100, ge=1, le=1000, description="Number of items per page")
    sort_by: Optional[str] = Field("date", description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order")


class DateRangeFilter(BaseModel):
    """Date range filtering"""
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    days: Optional[int] = Field(None, ge=1, le=365, description="Number of days from today")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v <= values['start_date']:
                raise ValueError('End date must be after start date')
        return v


class PriceFilter(BaseModel):
    """Price data filtering"""
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price filter")
    min_volume: Optional[Decimal] = Field(None, ge=0, description="Minimum volume filter")
    max_volume: Optional[Decimal] = Field(None, ge=0, description="Maximum volume filter")


class SentimentFilter(BaseModel):
    """Sentiment data filtering"""
    min_twitter_sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Minimum Twitter sentiment")
    max_twitter_sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Maximum Twitter sentiment")
    min_reddit_sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Minimum Reddit sentiment")
    max_reddit_sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Maximum Reddit sentiment")


class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    total_items: int
    total_pages: int
    current_page: int
    has_next: bool
    has_previous: bool


class EnhancedPriceResponse(BaseModel):
    """Enhanced price data response with metadata"""
    currency: str
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    total_items: int
    date_range: Dict[str, Any]
    price_summary: Dict[str, Any]
    count: int


class EnhancedSentimentResponse(BaseModel):
    """Enhanced sentiment data response with analytics"""
    currency: str
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    total_items: int
    sentiment_summary: Dict[str, Any]
    count: int


class PredictionRequest(BaseModel):
    """Enhanced prediction request"""
    prediction_horizon: int = Field(7, ge=1, le=30, description="Days ahead to predict")
    model_type: str = Field("best", description="Model type to use")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    include_features: bool = Field(False, description="Include feature importance")


class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction response with additional metadata"""
    currency: str
    prediction_date: str
    prediction_horizon: int
    predicted_direction: str
    confidence_score: float = Field(..., ge=0, le=1)
    model_version: str
    model_type: str
    features_importance: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Dict[str, float]] = None
    market_context: Optional[Dict[str, Any]] = None


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    auc_roc: Optional[float] = Field(None, ge=0, le=1)
    training_date: datetime
    data_points: int


class ModelStatus(BaseModel):
    """Enhanced model status"""
    model_name: str
    model_type: str
    currency: str
    status: str
    performance: ModelPerformanceMetrics
    last_prediction: Optional[datetime] = None
    file_size: Optional[str] = None
    created_at: datetime


class APIHealthStatus(BaseModel):
    """Enhanced API health status"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    cache_stats: Optional[Dict[str, Any]] = None
    rate_limit_stats: Optional[Dict[str, Any]] = None


class ErrorDetail(BaseModel):
    """Detailed error response"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    path: str
    method: str


class ValidationErrorDetail(BaseModel):
    """Validation error details"""
    field: str
    message: str
    invalid_value: Any


class EnhancedErrorResponse(BaseModel):
    """Enhanced error response with debugging info"""
    error: str
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[ValidationErrorDetail]] = None
    timestamp: datetime
    path: str
    method: str
    request_id: Optional[str] = None


# WebSocket Message Models

class WebSocketMessageType(str, Enum):
    """WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PRICE_UPDATE = "price_update"
    PREDICTION_UPDATE = "prediction_update"
    SENTIMENT_UPDATE = "sentiment_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: WebSocketMessageType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""
    type: WebSocketMessageType = WebSocketMessageType.SUBSCRIBE
    channels: List[str] = Field(..., description="Channels to subscribe to")
    currencies: Optional[List[str]] = Field(None, description="Currencies to filter")


class PriceUpdateMessage(BaseModel):
    """Real-time price update message"""
    type: WebSocketMessageType = WebSocketMessageType.PRICE_UPDATE
    currency: str
    price_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionUpdateMessage(BaseModel):
    """Real-time prediction update message"""
    type: WebSocketMessageType = WebSocketMessageType.PREDICTION_UPDATE
    currency: str
    prediction: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class SentimentUpdateMessage(BaseModel):
    """Real-time sentiment update message"""
    type: WebSocketMessageType = WebSocketMessageType.SENTIMENT_UPDATE
    currency: str
    sentiment_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Background Task Models

class TaskStatus(str, Enum):
    """Background task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundTask(BaseModel):
    """Background task model"""
    task_id: str
    task_type: str
    status: TaskStatus
    progress: Optional[float] = Field(None, ge=0, le=100)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # seconds


class TaskResult(BaseModel):
    """Background task result"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: Optional[float] = None  # seconds
    completed_at: Optional[datetime] = None


# Analytics Models

class APIUsageStats(BaseModel):
    """API usage statistics"""
    total_requests: int
    successful_requests: int
    error_requests: int
    average_response_time: float
    requests_by_endpoint: Dict[str, int]
    requests_by_hour: Dict[str, int]
    unique_clients: int
    cache_hit_rate: Optional[float] = None
    period_start: datetime
    period_end: datetime


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    cache_hit_rate: Optional[float] = None
    database_query_time: Optional[float] = None
    memory_usage: Optional[str] = None 