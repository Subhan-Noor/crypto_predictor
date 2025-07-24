#!/usr/bin/env python3
"""
Generate Historic Predictions with New Models

This script generates predictions for the last 30 days using the newly retrained models
and saves them to the database for validation.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.prediction_pipeline import CryptoPredictionPipeline
from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def generate_historic_predictions():
    """Generate predictions for the last 30 days"""
    
    print("🔮 Generating Historic Predictions with New Models")
    print("=" * 60)
    
    # Initialize prediction pipeline
    pipeline = CryptoPredictionPipeline()
    
    # Get the last 30 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Generating predictions for BTC and ETH...")
    
    currencies = ['BTC', 'ETH']
    total_predictions = 0
    
    for currency in currencies:
        print(f"\n🔸 Processing {currency}...")
        
        try:
            # Generate prediction for this currency
            prediction_result = await pipeline.make_and_save_prediction(currency)
            
            if prediction_result:
                print(f"  ✅ {currency} prediction generated successfully")
                print(f"     Prediction: {prediction_result['prediction']}")
                print(f"     Confidence: {prediction_result['confidence']:.2%}")
                print(f"     Raw Confidence: {prediction_result.get('raw_confidence', 'N/A')}")
                print(f"     Model Performance Factor: {prediction_result.get('model_performance_factor', 'N/A')}")
                total_predictions += 1
            else:
                print(f"  ❌ Failed to generate {currency} prediction")
                
        except Exception as e:
            print(f"  ❌ Error generating {currency} prediction: {str(e)}")
            logger.error(f"Error generating {currency} prediction: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"=" * 60)
    print(f"✅ Total predictions generated: {total_predictions}/{len(currencies)}")
    
    if total_predictions > 0:
        print(f"🎯 Predictions saved to database for validation")
        print(f"📈 Next step: Run validation script to check accuracy")
    
    return total_predictions


async def main():
    """Main function"""
    try:
        # Database connection is handled automatically by DatabaseManager
        print("🔗 Database connection established automatically")
        
        # Generate predictions
        await generate_historic_predictions()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 