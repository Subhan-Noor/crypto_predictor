from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uuid


class CryptoPriceBase(BaseModel):
    currency: str  # BTC or ETH
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CryptoPriceCreate(CryptoPriceBase):
    pass


class CryptoPrice(CryptoPriceBase):
    id: str
    
    class Config:
        from_attributes = True


class CryptoSentimentBase(BaseModel):
    currency: str  # BTC or ETH
    date: datetime
    fear_greed_index: Optional[int] = None
    twitter_sentiment: Optional[float] = None
    reddit_sentiment: Optional[float] = None


class CryptoSentimentCreate(CryptoSentimentBase):
    pass


class CryptoSentiment(CryptoSentimentBase):
    id: str
    
    class Config:
        from_attributes = True


class PredictionCreate(BaseModel):
    currency: str
    prediction_date: datetime
    target_date: datetime  # Date for which prediction is made (7 days ahead)
    prediction: int  # 1 for price increase, 0 for decrease/no change
    confidence: float  # Confidence score between 0 and 1
    features_used: dict  # JSON object containing the features used for prediction


class Prediction(PredictionCreate):
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True 