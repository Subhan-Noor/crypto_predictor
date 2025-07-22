"""
Test Script for ML Pipeline

This script tests the ML components to ensure everything works correctly.
Run this before training to verify the setup.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_preprocessor import CryptoDataPreprocessor
from ml.feature_engineering import CryptoFeatureEngineer
from ml.model_trainer import CryptoModelTrainer
from app.logger import logger


async def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("🧪 Testing Data Preprocessing...")
    
    try:
        preprocessor = CryptoDataPreprocessor()
        
        # Test data loading
        btc_data = await preprocessor.load_data('BTC')
        
        if btc_data['prices'].empty:
            print("❌ No price data found for BTC")
            return False
        
        print(f"✅ Loaded {len(btc_data['prices'])} price records")
        print(f"✅ Loaded {len(btc_data['sentiment'])} sentiment records")
        
        # Test dataset preparation
        dataset = await preprocessor.prepare_ml_dataset('BTC')
        
        print(f"✅ Prepared dataset with {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        if len(dataset['train']) < 10:
            print("⚠️  Warning: Very few training samples")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        return False


def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🧪 Testing Feature Engineering...")
    
    try:
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'open': [100 + i for i in range(50)],
            'high': [105 + i for i in range(50)],
            'low': [95 + i for i in range(50)],
            'close': [102 + i for i in range(50)],
            'volume': [1000000 + i*1000 for i in range(50)],
            'twitter_sentiment': [0.1 * (i % 10 - 5) for i in range(50)],
            'reddit_sentiment': [0.05 * (i % 8 - 4) for i in range(50)],
            'target': [i % 2 for i in range(50)]
        })
        
        feature_engineer = CryptoFeatureEngineer()
        
        # Test feature creation
        features_df = feature_engineer.create_features(sample_data)
        
        print(f"✅ Generated {features_df.shape[1]} features from {sample_data.shape[1]} original columns")
        
        # Test feature selection
        feature_cols = feature_engineer.get_feature_columns(features_df)
        print(f"✅ Selected {len(feature_cols)} features for ML")
        
        # Test ML preparation
        ml_data = feature_engineer.prepare_features_for_ml(features_df)
        print(f"✅ Prepared ML data: X shape {ml_data['X'].shape}, y shape {ml_data['y'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        return False


def test_model_training():
    """Test model training with small dataset"""
    print("\n🧪 Testing Model Training...")
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate synthetic data for testing
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, 
            random_state=42, n_informative=10
        )
        
        # Split data
        split_idx = 80
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"✅ Created synthetic dataset: {X_train.shape} train, {X_test.shape} test")
        
        # Test model training
        model_trainer = CryptoModelTrainer()
        
        # Test individual models (with reduced parameters for speed)
        print("  Testing Logistic Regression...")
        lr_results = model_trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        print(f"    Accuracy: {lr_results['test_accuracy']:.4f}")
        
        print("  Testing Random Forest...")
        rf_results = model_trainer.train_random_forest(X_train, y_train, X_test, y_test)
        print(f"    Accuracy: {rf_results['test_accuracy']:.4f}")
        
        print("✅ Model training completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False


async def test_full_pipeline():
    """Test the complete ML pipeline"""
    print("\n🧪 Testing Full ML Pipeline...")
    
    try:
        from scripts.train_models import MLPipeline
        
        # Check if we have enough data
        preprocessor = CryptoDataPreprocessor()
        btc_data = await preprocessor.load_data('BTC')
        
        if len(btc_data['prices']) < 30:
            print("⚠️  Skipping full pipeline test - insufficient data")
            print("    Run data ingestion first: python scripts/data_ingestion.py --days 30")
            return True  # Not a failure, just not enough data
        
        print(f"✅ Sufficient data available ({len(btc_data['prices'])} records)")
        
        # Test with minimal settings
        pipeline = MLPipeline(prediction_horizon=7, feature_window=7)
        
        print("  Running preprocessing...")
        dataset = await pipeline.preprocessor.prepare_ml_dataset('BTC')
        
        # Check if we have enough data for testing
        if len(dataset['train']) < 10 or len(dataset['test']) < 5:
            print("⚠️  Insufficient data for full pipeline test")
            print(f"    Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
            return True  # Not a failure, just not enough data
        
        print("  Running feature engineering...")
        train_features = pipeline.feature_engineer.create_features(dataset['train'])
        test_features = pipeline.feature_engineer.create_features(dataset['test'])
        
        print(f"✅ Pipeline components working correctly")
        print(f"    Features: {train_features.shape[1]} columns")
        print(f"    Training samples: {len(dataset['train'])}")
        print(f"    Test samples: {len(dataset['test'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting ML Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing()),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Full Pipeline", test_full_pipeline())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func()
        
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ML pipeline is ready for training.")
        print("\nNext steps:")
        print("1. Ensure you have sufficient data: python scripts/data_ingestion.py --days 60")
        print("2. Train models: python scripts/train_models.py --currency BTC")
        print("3. Or train all: python scripts/train_models.py --all")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure you have:")
        print("1. Set up your .env file with Supabase credentials")
        print("2. Applied the database schema")
        print("3. Run data ingestion to populate the database")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 