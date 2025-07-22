import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database Configuration
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")
    supabase_service_role_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # API Keys
    coingecko_api_key: Optional[str] = os.getenv("COINGECKO_API_KEY")
    fear_greed_api_url: str = os.getenv("FEAR_GREED_API_URL", "https://api.alternative.me/fng/")
    
    # Social Media APIs
    twitter_bearer_token: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = os.getenv("TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = os.getenv("TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = os.getenv("TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: Optional[str] = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    
    reddit_client_id: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "CryptoPredictorBot/1.0")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application Settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"


settings = Settings() 