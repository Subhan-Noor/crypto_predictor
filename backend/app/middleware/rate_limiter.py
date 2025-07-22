"""
Rate Limiting Middleware for Crypto Price Prediction API

This module provides rate limiting functionality to:
- Prevent API abuse
- Ensure fair usage
- Protect against DDoS attacks
- Provide different limits for different endpoints
"""

import time
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import redis
import hashlib
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter with different limits for different endpoints"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.redis_client = None
        self.connect()
        
        # Rate limit configurations
        self.limits = {
            "default": {"requests": 100, "window": 3600},  # 100 requests per hour
            "predictions": {"requests": 50, "window": 3600},  # 50 predictions per hour
            "current_prices": {"requests": 200, "window": 3600},  # 200 price checks per hour
            "health": {"requests": 1000, "window": 3600},  # 1000 health checks per hour
            "models": {"requests": 20, "window": 3600},  # 20 model operations per hour
        }
    
    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Rate limiter could not connect to Redis: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        # Include user agent for better uniqueness
        user_agent = request.headers.get("User-Agent", "")
        user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        
        return f"{client_ip}:{user_agent_hash}"
    
    def _get_endpoint_category(self, path: str) -> str:
        """Categorize endpoint for rate limiting"""
        if path.startswith("/predict"):
            return "predictions"
        elif path.startswith("/current_prices"):
            return "current_prices"
        elif path in ["/", "/health"]:
            return "health"
        elif path.startswith("/models"):
            return "models"
        else:
            return "default"
    
    def _get_rate_limit_key(self, client_id: str, endpoint_category: str, window_start: int) -> str:
        """Generate rate limit key"""
        return f"rate_limit:{endpoint_category}:{client_id}:{window_start}"
    
    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits
        
        Returns:
            (is_allowed, limit_info)
        """
        if not self.is_available():
            # If Redis is not available, allow the request but log warning
            logger.warning("Rate limiter unavailable, allowing request")
            return True, {"status": "rate_limiter_unavailable"}
        
        client_id = self._get_client_id(request)
        endpoint_category = self._get_endpoint_category(request.url.path)
        
        # Get rate limit configuration
        limit_config = self.limits.get(endpoint_category, self.limits["default"])
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]
        
        # Calculate current window
        current_time = int(time.time())
        window_start = current_time - (current_time % window_seconds)
        
        # Generate Redis key
        key = self._get_rate_limit_key(client_id, endpoint_category, window_start)
        
        try:
            # Get current request count
            current_count = self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            # Check if limit exceeded
            if current_count >= max_requests:
                reset_time = window_start + window_seconds
                return False, {
                    "status": "rate_limited",
                    "limit": max_requests,
                    "remaining": 0,
                    "reset_time": reset_time,
                    "retry_after": reset_time - current_time
                }
            
            # Increment counter
            new_count = current_count + 1
            self.redis_client.setex(key, window_seconds, new_count)
            
            # Calculate remaining requests
            remaining = max_requests - new_count
            reset_time = window_start + window_seconds
            
            return True, {
                "status": "allowed",
                "limit": max_requests,
                "remaining": remaining,
                "reset_time": reset_time,
                "used": new_count
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # On error, allow the request
            return True, {"status": "error", "error": str(e)}
    
    def get_client_stats(self, request: Request) -> Dict[str, any]:
        """Get rate limit statistics for a client"""
        if not self.is_available():
            return {"status": "unavailable"}
        
        client_id = self._get_client_id(request)
        current_time = int(time.time())
        stats = {}
        
        for endpoint_category, limit_config in self.limits.items():
            window_seconds = limit_config["window"]
            window_start = current_time - (current_time % window_seconds)
            key = self._get_rate_limit_key(client_id, endpoint_category, window_start)
            
            try:
                current_count = self.redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                
                stats[endpoint_category] = {
                    "limit": limit_config["requests"],
                    "used": current_count,
                    "remaining": limit_config["requests"] - current_count,
                    "window_seconds": window_seconds,
                    "reset_time": window_start + window_seconds
                }
            except Exception as e:
                stats[endpoint_category] = {"error": str(e)}
        
        return stats


# Rate limiter middleware function
rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Check rate limit
    is_allowed, limit_info = rate_limiter.check_rate_limit(request)
    
    if not is_allowed:
        # Return rate limit error
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": f"Too many requests. Limit: {limit_info['limit']} per hour",
                "limit": limit_info["limit"],
                "remaining": limit_info["remaining"],
                "reset_time": limit_info["reset_time"],
                "retry_after": limit_info["retry_after"]
            },
            headers={
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Remaining": str(limit_info["remaining"]),
                "X-RateLimit-Reset": str(limit_info["reset_time"]),
                "Retry-After": str(limit_info["retry_after"])
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    if "status" in limit_info and limit_info["status"] == "allowed":
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
    
    return response 