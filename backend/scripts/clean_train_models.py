#!/usr/bin/env python3
"""
Clean Model Training Script

This script retrains ML models using the clean infrastructure that fixes
the feature mismatch issues. It properly handles feature names as lists
and provides robust model training and saving.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.clean_model_trainer import CleanModelTrainer
from ml.data_preprocessor import CryptoDataPreprocessor
from ml.feature_engineering import CryptoFeatureEngineer
from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def load_and_prepare_data(currency: str, days: int = 365) -> pd.DataFrame:
    """
    Load and prepare data for model training
    
    Args:
        currency: Currency to load data for (BTC/ETH)
        days: Number of days of historical data to load
        
    Returns:
        DataFrame with features and target ready for training
    """
    logger.info(f"Loading data for {currency} (last {days} days)...")
    
    # Initialize components
    preprocessor = CryptoDataPreprocessor()
    feature_engineer = CryptoFeatureEngineer()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Load data using the complete preprocessing pipeline
    data = await preprocessor.load_data(currency, start_date, end_date)
    
    if data['prices'].empty:
        raise ValueError(f"No price data available for {currency}")
    
    logger.info(f"Loaded {len(data['prices'])} price records and {len(data['sentiment'])} sentiment records")
    
    # Merge price and sentiment data
    merged_df = preprocessor.merge_data(data['prices'], data['sentiment'])
    
    if merged_df.empty:
        raise ValueError(f"No merged data available for {currency}")
    
    logger.info(f"Merged data: {len(merged_df)} records")
    
    # Create target labels (this adds the 'target' column)
    labeled_df = preprocessor.create_labels(merged_df)
    
    if labeled_df.empty:
        raise ValueError("No labeled data available")
    
    logger.info(f"Created labels: {len(labeled_df)} records with targets")
    
    # Create features
    features_df = feature_engineer.create_features(labeled_df)
    
    if features_df.empty:
        raise ValueError("No features could be generated")
    
    logger.info(f"Generated features: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
    
    # Check for target column
    if 'target' not in features_df.columns:
        raise ValueError("Target column not found in features")
    
    # Remove rows with NaN targets
    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=['target'])
    final_rows = len(features_df)
    
    if final_rows < initial_rows:
        logger.warning(f"Dropped {initial_rows - final_rows} rows with NaN targets")
    
    if len(features_df) < 100:
        raise ValueError(f"Insufficient data after cleaning: {len(features_df)} rows")
    
    logger.info(f"Final dataset: {len(features_df)} rows ready for training")
    
    return features_df


async def train_currency_models(currency: str) -> Dict[str, Any]:
    """
    Train all models for a specific currency
    
    Args:
        currency: Currency to train models for
        
    Returns:
        Training results summary
    """
    logger.info(f"🚀 Training models for {currency}")
    logger.info("=" * 50)
    
    try:
        # Load and prepare data
        features_df = await load_and_prepare_data(currency)
        
        # Initialize trainer
        trainer = CleanModelTrainer()
        
        # Train all models
        results = trainer.train_all_models(features_df, currency)
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_type, result in results.items():
            if 'results' in result:
                f1_score = result['results'].get('test_f1', 0)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_type
        
        summary = {
            'currency': currency,
            'success': True,
            'models_trained': len([r for r in results.values() if 'results' in r]),
            'best_model': best_model,
            'best_f1': best_f1,
            'results': results
        }
        
        logger.info(f"✅ {currency} training complete!")
        logger.info(f"   Best model: {best_model} (F1: {best_f1:.4f})")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error training models for {currency}: {str(e)}")
        return {
            'currency': currency,
            'success': False,
            'error': str(e)
        }


async def train_all_models():
    """Train models for all currencies"""
    print("🤖 Clean Model Training Script")
    print("=" * 60)
    print("This script trains clean ML models that fix the feature mismatch issues")
    print("")
    
    # Check database connection
    if not db_manager.is_connected():
        print("❌ Database not connected. Please check your Supabase credentials.")
        return
    
    logger.info("Database connection verified ✅")
    
    # Currencies to train
    currencies = ['BTC', 'ETH']
    results = {}
    
    # Train models for each currency
    for currency in currencies:
        result = await train_currency_models(currency)
        results[currency] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    successful = 0
    total_models = 0
    
    for currency, result in results.items():
        if result['success']:
            successful += 1
            models_trained = result['models_trained']
            total_models += models_trained
            print(f"✅ {currency}: {result['best_model']} (F1: {result['best_f1']:.4f}) - {models_trained} models")
        else:
            print(f"❌ {currency}: Failed - {result['error']}")
    
    print(f"\n🎉 Training complete!")
    print(f"   Currencies: {successful}/{len(currencies)} successful")
    print(f"   Models: {total_models} total models trained")
    
    if successful > 0:
        print(f"\n🔮 Ready to make predictions! Models saved in 'models/' directory.")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(train_all_models())
        print("\n" + "="*60)
        print("✅ Training script completed successfully")
        print("="*60)
        
        # Check if any training was successful
        successful_training = any(result.get('success', False) for result in results.values())
        if not successful_training:
            print("⚠️  Warning: No models were successfully trained")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("="*60)
        print("Training failed - check logs for details")
        print("="*60)
        sys.exit(1) 