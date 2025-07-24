#!/usr/bin/env python3
"""
Clean Prediction Generation Script

This script generates predictions using the clean pipeline that fixes
the feature mismatch issues. It properly loads models and generates
predictions with realistic confidence scores.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.clean_prediction_pipeline import CleanPredictionPipeline
from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def generate_prediction_for_currency(currency: str) -> Dict[str, Any]:
    """
    Generate a prediction for a specific currency
    
    Args:
        currency: Currency to generate prediction for
        
    Returns:
        Prediction result or error information
    """
    logger.info(f"🔮 Generating prediction for {currency}")
    
    try:
        # Initialize prediction pipeline
        pipeline = CleanPredictionPipeline()
        
        # Make and save prediction
        prediction = await pipeline.make_and_save_prediction(currency)
        
        logger.info(f"✅ {currency} prediction complete!")
        logger.info(f"   Direction: {prediction['predicted_direction']}")
        logger.info(f"   Confidence: {prediction['confidence_score']:.2%}")
        logger.info(f"   Model: {prediction['model_metadata']['model_type']}")
        
        return {
            'currency': currency,
            'success': True,
            'prediction': prediction
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating prediction for {currency}: {str(e)}")
        return {
            'currency': currency,
            'success': False,
            'error': str(e)
        }


async def generate_historic_predictions(currency: str, days: int = 90) -> Dict[str, Any]:
    """
    Generate predictions for historic dates
    
    Args:
        currency: Currency to generate predictions for
        days: Number of historic days to generate predictions for
        
    Returns:
        Results summary
    """
    logger.info(f"📅 Generating historic predictions for {currency} (last {days} days)")
    
    pipeline = CleanPredictionPipeline()
    
    # Get date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    predictions_generated = 0
    errors = 0
    
    current_date = start_date
    while current_date <= end_date:
        try:
            # Convert date to datetime for prediction
            prediction_datetime = datetime.combine(current_date, datetime.min.time())
            
            # Make prediction for this date
            prediction = await pipeline.make_prediction(currency, prediction_datetime)
            
            # Save to database
            await pipeline.save_prediction(prediction)
            
            predictions_generated += 1
            
            if predictions_generated % 10 == 0:
                logger.info(f"   Generated {predictions_generated} predictions so far...")
                
        except Exception as e:
            logger.warning(f"   Failed to generate prediction for {current_date}: {str(e)}")
            errors += 1
        
        # Move to next day
        current_date += timedelta(days=1)
    
    logger.info(f"✅ Historic predictions complete for {currency}")
    logger.info(f"   Generated: {predictions_generated} predictions")
    logger.info(f"   Errors: {errors}")
    
    return {
        'currency': currency,
        'predictions_generated': predictions_generated,
        'errors': errors,
        'success': predictions_generated > 0
    }


async def generate_current_predictions():
    """Generate predictions for current date"""
    print("🔮 Clean Prediction Generation Script")
    print("=" * 60)
    print("This script generates predictions using the clean pipeline")
    print("")
    
    # Check database connection
    if not db_manager.is_connected():
        print("❌ Database not connected. Please check your Supabase credentials.")
        return
    
    logger.info("Database connection verified ✅")
    
    # Currencies to generate predictions for
    currencies = ['BTC', 'ETH']
    results = {}
    
    # Generate predictions for each currency
    for currency in currencies:
        result = await generate_prediction_for_currency(currency)
        results[currency] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PREDICTION GENERATION SUMMARY")
    print("=" * 60)
    
    successful = 0
    
    for currency, result in results.items():
        if result['success']:
            successful += 1
            prediction = result['prediction']
            print(f"✅ {currency}: {prediction['predicted_direction']} "
                  f"({prediction['confidence_score']:.1%} confidence)")
        else:
            print(f"❌ {currency}: Failed - {result['error']}")
    
    print(f"\n🎉 Prediction generation complete!")
    print(f"   Successful: {successful}/{len(currencies)} currencies")
    
    if successful > 0:
        print(f"\n📊 Predictions saved to database and ready for validation.")
    
    return results


async def generate_all_historic_predictions(days: int = 90):
    """Generate historic predictions for all currencies"""
    print("📅 Historic Prediction Generation")
    print("=" * 60)
    print(f"Generating predictions for the last {days} days")
    print("")
    
    # Check database connection
    if not db_manager.is_connected():
        print("❌ Database not connected. Please check your Supabase credentials.")
        return
    
    logger.info("Database connection verified ✅")
    
    # Currencies to generate predictions for
    currencies = ['BTC', 'ETH']
    results = {}
    
    # Generate historic predictions for each currency
    for currency in currencies:
        result = await generate_historic_predictions(currency, days)
        results[currency] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 HISTORIC PREDICTION SUMMARY")
    print("=" * 60)
    
    total_predictions = 0
    total_errors = 0
    
    for currency, result in results.items():
        if result['success']:
            predictions = result['predictions_generated']
            errors = result['errors']
            total_predictions += predictions
            total_errors += errors
            print(f"✅ {currency}: {predictions} predictions generated ({errors} errors)")
        else:
            print(f"❌ {currency}: Failed")
    
    print(f"\n🎉 Historic prediction generation complete!")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Total errors: {total_errors}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate crypto predictions')
    parser.add_argument('--historic', type=int, metavar='DAYS', 
                       help='Generate historic predictions for N days')
    parser.add_argument('--current', action='store_true',
                       help='Generate predictions for current date (default)')
    
    args = parser.parse_args()
    
    if args.historic:
        asyncio.run(generate_all_historic_predictions(args.historic))
    else:
        asyncio.run(generate_current_predictions()) 