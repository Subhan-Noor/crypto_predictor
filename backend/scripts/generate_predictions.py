"""
Generate Real ML Predictions Script

This script generates actual machine learning predictions using trained models
and saves them to the database to replace any mock predictions.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.prediction_pipeline import CryptoPredictionPipeline
from app.database import db_manager
from app.logger import logger


async def generate_real_predictions():
    """Generate real predictions for both BTC and ETH"""
    
    print("🔮 Generating Real ML Predictions...")
    print("=" * 50)
    
    pipeline = CryptoPredictionPipeline()
    results = {}
    
    for currency in ['BTC', 'ETH']:
        print(f"\n📈 Processing {currency}...")
        
        try:
            # Generate prediction using real ML model
            prediction = await pipeline.make_and_save_prediction(currency, model_type="best")
            
            print(f"  ✅ Prediction Generated:")
            print(f"     Direction: {prediction.get('predicted_direction', 'N/A')}")
            print(f"     Confidence: {prediction.get('confidence_score', 0):.4f}")
            print(f"     Model: {prediction.get('model_version', 'N/A')}")
            print(f"     Features Used: {len(prediction.get('features_used', []))} features")
            print(f"     Database ID: {prediction.get('id', 'N/A')}")
            
            results[currency] = {
                'success': True,
                'prediction': prediction
            }
            
        except Exception as e:
            print(f"  ❌ Error generating prediction: {str(e)}")
            results[currency] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 PREDICTION GENERATION SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    
    print(f"Total Predictions: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    if successful > 0:
        print(f"\n✅ Successfully generated {successful} real predictions!")
        print("These predictions are now available in your dashboard.")
    else:
        print(f"\n❌ No predictions generated. Check the errors above.")
    
    return results


async def check_predictions_in_db():
    """Check recent predictions in the database"""
    
    print("\n🔍 Checking Recent Predictions in Database...")
    print("=" * 50)
    
    for currency in ['BTC', 'ETH']:
        predictions = await db_manager.get_predictions(currency, days=7, limit=5)
        
        print(f"\n{currency} Recent Predictions:")
        if predictions:
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. Date: {pred.get('prediction_date', 'N/A')}")
                print(f"     Direction: {pred.get('predicted_direction', 'N/A')}")
                print(f"     Confidence: {pred.get('confidence_score', 0)}")
                print(f"     Model: {pred.get('model_version', 'N/A')}")
        else:
            print("  No predictions found")


async def main():
    """Main function"""
    
    print("🚀 Real ML Prediction Generator")
    print("This script will generate actual ML predictions using trained models")
    print("and replace any mock predictions in your dashboard.\n")
    
    try:
        # Generate new real predictions
        results = await generate_real_predictions()
        
        # Check what's now in the database
        await check_predictions_in_db()
        
        print("\n🎉 Prediction generation complete!")
        print("Refresh your dashboard to see the real ML predictions.")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 