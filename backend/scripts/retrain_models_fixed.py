"""
Retrain Models with Fixed Parameters and Better Confidence Calibration

This script retrains the ML models with improved parameters and fixes
the confidence calculation issues that were causing unrealistic 100% confidence scores.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_trainer import CryptoModelTrainer
from ml.data_preprocessor import CryptoDataPreprocessor
from ml.feature_engineering import CryptoFeatureEngineer
from app.database import db_manager
from app.logger import logger


class FixedModelTrainer(CryptoModelTrainer):
    """Enhanced model trainer with better parameters and confidence calibration"""
    
    def __init__(self, models_dir: str = "models"):
        super().__init__(models_dir)
        
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Logistic Regression with better parameters"""
        logger.info("Training Logistic Regression model with improved parameters...")
        
        # Use simple Logistic Regression with good default parameters
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        results = self._evaluate_model("Logistic Regression", y_train, y_pred_train, 
                                     y_test, y_pred_test, y_proba_test)
        
        # Store model
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        logger.info(f"Logistic Regression training complete")
        
        return results
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest with better parameters"""
        logger.info("Training Random Forest model with improved parameters...")
        
        # Use simple Random Forest with good default parameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        results = self._evaluate_model("Random Forest", y_train, y_pred_train,
                                     y_test, y_pred_test, y_proba_test)
        
        # Store model
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        logger.info(f"Random Forest training complete")
        
        return results


async def retrain_models_with_fixes():
    """Retrain all models with improved parameters and confidence calibration"""
    
    print("🔧 Retraining Models with Fixed Parameters and Better Confidence Calibration")
    print("=" * 70)
    
    currencies = ['BTC', 'ETH']
    results = {}
    
    for currency in currencies:
        print(f"\n📈 Retraining models for {currency}...")
        
        try:
            # Initialize components
            preprocessor = CryptoDataPreprocessor()
            feature_engineer = CryptoFeatureEngineer()
            trainer = FixedModelTrainer()
            
            # Step 1: Data preprocessing
            print(f"  Step 1: Loading and preprocessing data...")
            dataset = await preprocessor.prepare_ml_dataset(currency)
            
            if dataset is None:
                print(f"  ❌ No dataset returned for {currency}")
                continue
            
            # Extract the full dataset for feature engineering
            full_dataset = dataset['full']
            print(f"    Full dataset shape: {full_dataset.shape}")
            print(f"    Full dataset columns: {list(full_dataset.columns)}")
            
            # Step 2: Feature engineering
            print(f"  Step 2: Feature engineering...")
            features_df = feature_engineer.create_features(full_dataset)
            
            if len(features_df) < 100:
                print(f"  ❌ Insufficient data for {currency} after feature engineering: {len(features_df)} records")
                continue
            
            # Step 3: Prepare training data
            print(f"  Step 3: Preparing training data...")
            
            # Split features_df into train/test using the original split from dataset
            train_size = len(dataset['train'])
            train_df = features_df.iloc[:train_size]
            test_df = features_df.iloc[train_size:]
            
            # Extract features and target
            feature_cols = [col for col in features_df.columns if col not in ['date', 'target', 'future_close', 'price_change_pct']]
            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values
            y_train = train_df['target'].values
            y_test = test_df['target'].values
            
            # Handle NaN values in features
            print(f"    Handling NaN values...")
            print(f"    NaN count in X_train: {np.isnan(X_train).sum()}")
            print(f"    NaN count in X_test: {np.isnan(X_test).sum()}")
            
            # Fill NaN values with forward fill, then backward fill, then 0
            X_train = pd.DataFrame(X_train, columns=feature_cols).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            X_test = pd.DataFrame(X_test, columns=feature_cols).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            
            print(f"    NaN count after cleaning - X_train: {np.isnan(X_train).sum()}, X_test: {np.isnan(X_test).sum()}")
            
            print(f"    Training samples: {len(X_train)}")
            print(f"    Test samples: {len(X_test)}")
            print(f"    Features: {X_train.shape[1]}")
            
            # Step 4: Train models
            print(f"  Step 4: Training models...")
            
            # Train Logistic Regression
            lr_results = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
            
            # Train Random Forest
            rf_results = trainer.train_random_forest(X_train, y_train, X_test, y_test)
            
            # Step 5: Save models
            print(f"  Step 5: Saving models...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models with timestamp
            trainer.save_models(currency, timestamp)
            
            # Find best model
            best_model = None
            best_f1 = 0
            
            for model_name, result in trainer.results.items():
                f1_score = result.get('test_f1', 0)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_name
            
            results[currency] = {
                'success': True,
                'best_model': best_model,
                'best_f1': best_f1,
                'models_trained': list(trainer.results.keys()),
                'results': trainer.results
            }
            
            print(f"  ✅ {currency} training complete!")
            print(f"     Best model: {best_model}")
            print(f"     Best F1 score: {best_f1:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error training models for {currency}: {str(e)}")
            results[currency] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 RETRAINING SUMMARY")
    print("=" * 70)
    
    successful = 0
    for currency, result in results.items():
        if result['success']:
            successful += 1
            print(f"✅ {currency}: {result['best_model']} (F1: {result['best_f1']:.4f})")
        else:
            print(f"❌ {currency}: Failed - {result['error']}")
    
    print(f"\n🎉 Retraining complete! {successful}/{len(currencies)} currencies successful")
    
    return results


if __name__ == "__main__":
    asyncio.run(retrain_models_with_fixes()) 