"""
Redis Caching Service for Crypto Price Prediction API

This module provides caching functionality to improve API performance by:
- Caching frequently accessed data (prices, predictions, sentiment)
- Reducing database load
- Implementing cache invalidation strategies
- Graceful fallback when Redis is unavailable
"""

import redis
import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service with graceful fallback"""
    
    def __init__(self):
        """Initialize Redis connection"""
        self.redis_client = None
        self.default_ttl = 300  # 5 minutes default
        self.enabled = settings.redis_enabled
        self.connect()
    
    def connect(self):
        """Connect to Redis"""
        if not self.enabled:
            logger.info("Redis caching disabled by configuration")
            return
            
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            logger.info("API will run without caching (performance may be reduced)")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.enabled:
            return False
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort parameters for consistent keys
        sorted_params = sorted(params.items())
        params_str = json.dumps(sorted_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{prefix}:{params_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_available():
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.is_available():
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.is_available():
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern error: {e}")
            return 0
    
    # Specific caching methods for crypto API
    
    def get_prices(self, currency: str, days: int, page: int = 1, limit: int = 100) -> Optional[Dict]:
        """Get cached price data"""
        key = self._generate_key("prices", {
            "currency": currency,
            "days": days,
            "page": page,
            "limit": limit
        })
        return self.get(key)
    
    def set_prices(self, currency: str, days: int, page: int, limit: int, data: Dict, ttl: int = 300) -> bool:
        """Cache price data"""
        key = self._generate_key("prices", {
            "currency": currency,
            "days": days,
            "page": page,
            "limit": limit
        })
        return self.set(key, data, ttl)
    
    def get_sentiment(self, currency: str, days: int) -> Optional[Dict]:
        """Get cached sentiment data"""
        key = self._generate_key("sentiment", {
            "currency": currency,
            "days": days
        })
        return self.get(key)
    
    def set_sentiment(self, currency: str, days: int, data: Dict, ttl: int = 600) -> bool:
        """Cache sentiment data (longer TTL as it changes less frequently)"""
        key = self._generate_key("sentiment", {
            "currency": currency,
            "days": days
        })
        return self.set(key, data, ttl)
    
    def get_current_prices(self) -> Optional[Dict]:
        """Get cached current prices"""
        return self.get("current_prices")
    
    def set_current_prices(self, data: Dict, ttl: int = 60) -> bool:
        """Cache current prices (short TTL for real-time data)"""
        return self.set("current_prices", data, ttl)
    
    def get_prediction(self, currency: str, model_type: str) -> Optional[Dict]:
        """Get cached prediction"""
        key = self._generate_key("prediction", {
            "currency": currency,
            "model_type": model_type
        })
        return self.get(key)
    
    def set_prediction(self, currency: str, model_type: str, data: Dict, ttl: int = 1800) -> bool:
        """Cache prediction (30 minutes TTL)"""
        key = self._generate_key("prediction", {
            "currency": currency,
            "model_type": model_type
        })
        return self.set(key, data, ttl)
    
    def invalidate_currency_cache(self, currency: str):
        """Invalidate all cache entries for a currency"""
        patterns = [
            f"prices:*{currency}*",
            f"sentiment:*{currency}*",
            f"prediction:*{currency}*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = self.delete_pattern(pattern)
            total_deleted += deleted
        
        logger.info(f"Invalidated {total_deleted} cache entries for {currency}")
        return total_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {"status": "disabled"}
        if not self.is_available():
            return {"status": "disconnected"}
        
        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "memory_used": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "connected_clients": info.get("connected_clients"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": round(
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100, 2
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}


# Global cache service instance
cache_service = CacheService() 