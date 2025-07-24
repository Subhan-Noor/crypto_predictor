#!/usr/bin/env python3
"""
Generate Predictions for Validation Testing

This script generates predictions for the last 7 days and saves them to the database
so we can validate them against actual price movements.
"""

import os
import sys
import asyncio
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_preprocessor import CryptoDataPreprocessor
from ml.feature_engineering import CryptoFeatureEngineer
from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_latest_model(currency: str, model_type: str = "random_forest"):
    """Load the latest model for a currency"""
    models_dir = "models"
    pattern = os.path.join(models_dir, f"{currency}_{model_type}_*.pkl")
    
    import glob
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise ValueError(f"No {model_type} models found for {currency}")
    
    # Get the latest model file
    latest_model_file = max(model_files, key=os.path.getctime)
    print(f"Loading model: {os.path.basename(latest_model_file)}")
    
    # Load the saved data (contains model and metadata)
    saved_data = joblib.load(latest_model_file)
    
    # Extract the actual model from the saved data
    if isinstance(saved_data, dict) and 'model' in saved_data:
        model = saved_data['model']
        metadata = saved_data['metadata']
        return model, metadata
    else:
        # Direct model (old format)
        return saved_data, {}


def calibrate_confidence(raw_confidence: float, model_performance_factor: float = 0.55) -> float:
    """Calibrate confidence score based on model performance"""
    # Calculate calibrated confidence
    calibrated_confidence = raw_confidence * model_performance_factor
    
    # Apply realistic bounds for crypto predictions
    max_realistic_confidence = 0.85  # 85% max for crypto
    min_realistic_confidence = 0.45  # 45% min for crypto
    
    # Clamp to realistic bounds
    calibrated_confidence = max(min_realistic_confidence, 
                               min(max_realistic_confidence, calibrated_confidence))
    
    return calibrated_confidence


async def generate_prediction_for_date(currency: str, model, prediction_date: datetime):
    """Generate a prediction for a specific date"""
    try:
        # Prepare data
        preprocessor = CryptoDataPreprocessor()
        feature_engineer = CryptoFeatureEngineer()
        
        # Get dataset
        dataset = await preprocessor.prepare_ml_dataset(currency)
        if dataset is None:
            return None
        
        # Use the full dataset for feature engineering
        full_dataset = dataset['full']
        
        # Create features
        features_df = feature_engineer.create_features(full_dataset)
        
        # Get the data point closest to the prediction date
        features_df['date'] = pd.to_datetime(features_df['date']).dt.date  # Convert to date only
        features_df = features_df.sort_values('date')
        
        # Find the closest date to our prediction date
        target_date = prediction_date.date()  # Use date only
        closest_idx = (features_df['date'] - target_date).abs().idxmin()
        prediction_features = features_df.loc[closest_idx:closest_idx].copy()
        
        if len(prediction_features) == 0:
            return None
        
        # Remove non-feature columns
        feature_cols = [col for col in prediction_features.columns 
                       if col not in ['date', 'target', 'future_close', 'price_change_pct']]
        
        X_pred = prediction_features[feature_cols].values
        
        # Handle NaN values
        if np.isnan(X_pred).any():
            X_pred = pd.DataFrame(X_pred, columns=feature_cols).fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Make prediction
        raw_proba = model.predict_proba(X_pred)[0]
        raw_confidence = max(raw_proba)  # Higher probability class
        
        # Determine prediction
        prediction = "UP" if raw_proba[1] > raw_proba[0] else "DOWN"
        
        # Calibrate confidence
        calibrated_confidence = calibrate_confidence(raw_confidence, model_performance_factor=0.55)
        
        result = {
            'currency': currency,
            'prediction': prediction,
            'confidence': calibrated_confidence,
            'raw_confidence': raw_confidence,
            'model_performance_factor': 0.55,
            'prediction_date': prediction_date.isoformat(),
            'raw_probabilities': raw_proba.tolist()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating prediction for {currency} on {prediction_date.date()}: {str(e)}")
        return None


async def save_prediction_to_db(prediction_data: dict) -> bool:
    """Save prediction to database"""
    try:
        # Prepare data for database - use correct column names
        db_data = {
            'currency': prediction_data['currency'],
            'predicted_direction': prediction_data['prediction'],  # Correct column name
            'prediction_date': prediction_data['prediction_date'],
            'confidence_score': prediction_data['confidence'],  # Correct column name
            'model_version': 'random_forest_20250723_230655',  # Correct column name
            'prediction_horizon': 7  # Required field for 7-day predictions
        }
        
        # Save to database
        prediction_id = await db_manager.insert_prediction(db_data)
        
        if prediction_id:
            print(f"  ✅ Saved to DB: {prediction_id}")
            print(f"     Confidence: {prediction_data['confidence']:.1%}")
            return True
        else:
            print(f"  ❌ Failed to save to DB")
            return False
            
    except Exception as e:
        logger.error(f"Error saving prediction to DB: {str(e)}")
        return False


async def generate_historic_predictions():
    """Generate predictions for the last 7 days"""
    
    print("🔮 Generating Historic Predictions for Validation")
    print("=" * 60)
    
    # Generate predictions for the last 7 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Generating predictions for BTC and ETH...")
    
    currencies = ['BTC', 'ETH']
    total_predictions = 0
    
    for currency in currencies:
        print(f"\n🔸 Processing {currency}...")
        
        try:
            # Load model
            model, metadata = load_latest_model(currency, "random_forest")
            print(f"  ✅ Model loaded (F1: {metadata.get('results', {}).get('test_f1', 'unknown'):.4f})")
            
            # Generate predictions for each day
            current_date = start_date
            while current_date <= end_date:
                prediction_date = datetime.combine(current_date, datetime.min.time())
                
                print(f"  📅 {current_date}: ", end="")
                
                # Generate prediction
                prediction_result = await generate_prediction_for_date(currency, model, prediction_date)
                
                if prediction_result:
                    print(f"{prediction_result['prediction']} ({prediction_result['confidence']:.1%})")
                    
                    # Save to database
                    saved = await save_prediction_to_db(prediction_result)
                    if saved:
                        total_predictions += 1
                else:
                    print("Failed")
                
                current_date += timedelta(days=1)
                
        except Exception as e:
            print(f"  ❌ Error processing {currency}: {str(e)}")
            logger.error(f"Error processing {currency}: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"=" * 60)
    print(f"✅ Total predictions generated and saved: {total_predictions}")
    
    if total_predictions > 0:
        print(f"🎯 Predictions ready for validation!")
        print(f"📈 Run validation script in 7 days to check accuracy")
    
    return total_predictions


async def main():
    """Main function"""
    try:
        print("🔗 Database connection established automatically")
        
        # Generate predictions
        await generate_historic_predictions()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 