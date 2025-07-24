#!/usr/bin/env python3
"""
Startup Debug Script for Railway Deployment

This script helps identify issues during Railway deployment startup.
"""

import os
import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports for Railway deployment...")
    
    try:
        # Test basic Python imports
        print("  Testing basic imports...")
        import asyncio
        import logging
        import time
        from datetime import datetime, timedelta
        from typing import Dict, List, Any, Optional
        from decimal import Decimal
        import uuid
        import pytz
        from dateutil import parser as date_parser
        print("  ✅ Basic imports successful")
        
        # Test FastAPI imports
        print("  Testing FastAPI...")
        from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.responses import JSONResponse
        from fastapi.exceptions import RequestValidationError
        from pydantic import ValidationError
        from fastapi.encoders import jsonable_encoder
        from contextlib import asynccontextmanager
        import uvicorn
        print("  ✅ FastAPI imports successful")
        
        # Test service imports
        print("  Testing services...")
        from app.services.twitter_service import twitter_service, TwitterSentimentService
        from app.services.reddit_service import reddit_service, RedditSentimentService
        from app.services.sentiment_analyzer import sentiment_analyzer
        from app.services.background_tasks import BackgroundTaskService
        from app.services.websocket_service import websocket_service
        from app.services.cache_service import cache_service
        from app.middleware.rate_limiter import rate_limit_middleware, rate_limiter
        print("  ✅ Service imports successful")
        
        # Test database and ML imports
        print("  Testing database and ML...")
        from app.database import db_manager
        from app.logger import logger
        from ml.prediction_pipeline import CryptoPredictionPipeline
        from app.services.binance_service import BinancePriceFetcher
        print("  ✅ Database and ML imports successful")
        
        # Test enhanced models
        print("  Testing enhanced models...")
        from app.models.api_models import (
            PaginationParams, DateRangeFilter, PriceFilter, SentimentFilter,
            EnhancedPriceResponse, EnhancedSentimentResponse, EnhancedPredictionResponse,
            PredictionRequest, APIHealthStatus, EnhancedErrorResponse
        )
        print("  ✅ Enhanced models imports successful")
        
        # Test configuration
        print("  Testing configuration...")
        from config import settings
        print("  ✅ Configuration import successful")
        
        # Test main app import
        print("  Testing main app...")
        from app.enhanced_main import app
        print("  ✅ Main app import successful")
        
        print("\n🎉 All imports successful! Railway deployment should work.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   File: {e.__traceback__.tb_frame.f_code.co_filename}")
        print(f"   Line: {e.__traceback__.tb_lineno}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment variables"""
    print("\n🔧 Testing environment variables...")
    
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'SUPABASE_SERVICE_ROLE_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {value[:20]}...")
        else:
            print(f"  ⚠️ {var}: Not set (will be needed at runtime)")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"  📝 Note: {len(missing_vars)} environment variables missing")
        print(f"     These are required for the app to function properly")
        print(f"     Set them in Railway dashboard or use Railway CLI")
    
    print(f"  📍 Current working directory: {os.getcwd()}")
    print(f"  🐍 Python version: {sys.version}")
    
    return len(missing_vars) == 0

if __name__ == "__main__":
    print("🚀 Railway Startup Debug Script")
    print("=" * 50)
    
    env_ok = test_environment()
    import_ok = test_imports()
    
    if import_ok:
        if env_ok:
            print("\n✅ All tests passed! Ready for deployment.")
        else:
            print("\n⚠️ Imports successful but environment variables missing.")
            print("   Set environment variables in Railway dashboard before deploying.")
        sys.exit(0)
    else:
        print("\n❌ Import tests failed! Check the errors above.")
        sys.exit(1) 