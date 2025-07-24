#!/usr/bin/env python3
"""
Test Deployment Imports

This script tests that all required imports work correctly for deployment.
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing deployment imports...")
    
    try:
        # Test FastAPI imports
        print("  Testing FastAPI...")
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        print("  ✅ FastAPI imports successful")
        
        # Test database imports
        print("  Testing database...")
        from app.database import db_manager
        print("  ✅ Database imports successful")
        
        # Test ML imports
        print("  Testing ML modules...")
        from ml.prediction_pipeline import CryptoPredictionPipeline
        from ml.data_preprocessor import CryptoDataPreprocessor
        from app.services.binance_service import BinancePriceFetcher
        print("  ✅ ML imports successful")
        
        # Test service imports
        print("  Testing services...")
        from app.services.twitter_service import TwitterSentimentService
        from app.services.reddit_service import RedditSentimentService
        from app.services.sentiment_analyzer import sentiment_analyzer
        from app.services.background_tasks import BackgroundTaskService
        from app.services.websocket_service import websocket_service
        from app.services.cache_service import cache_service
        print("  ✅ Service imports successful")
        
        # Test middleware imports
        print("  Testing middleware...")
        from app.middleware.rate_limiter import rate_limit_middleware
        print("  ✅ Middleware imports successful")
        
        # Test main app import
        print("  Testing main app...")
        from app.enhanced_main import app
        print("  ✅ Main app import successful")
        
        print("\n🎉 All imports successful! Deployment should work.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 