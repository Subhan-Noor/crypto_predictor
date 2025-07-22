"""
ML Model Training Script for Crypto Price Prediction

This script runs the complete ML pipeline:
1. Data preprocessing
2. Feature engineering
3. Model training (baseline and advanced)
4. Model evaluation and comparison
5. Model saving

Usage:
    python scripts/train_models.py --currency BTC --prediction-horizon 7
    python scripts/train_models.py --currency ETH --days 60
    python scripts/train_models.py --all  # Train for both BTC and ETH
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import json
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_preprocessor import CryptoDataPreprocessor
from ml.feature_engineering import CryptoFeatureEngineer
from ml.model_trainer import CryptoModelTrainer
from app.logger import logger


class MLPipeline:
    """Complete ML pipeline for crypto price prediction"""
    
    def __init__(self, prediction_horizon: int = 7, feature_window: int = 7):
        """
        Initialize ML pipeline
        
        Args:
            prediction_horizon: Days ahead to predict (default: 7)
            feature_window: Days of historical data for features (default: 7)
        """
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        
        # Initialize components
        self.preprocessor = CryptoDataPreprocessor(prediction_horizon)
        self.feature_engineer = CryptoFeatureEngineer(feature_window)
        self.model_trainer = CryptoModelTrainer()
        
    async def run_full_pipeline(self, currency: str) -> dict:
        """
        Run the complete ML pipeline for a currency
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Dictionary with training results and model information
        """
        logger.info(f"Starting ML pipeline for {currency}")
        
        try:
            # Step 1: Data preprocessing
            logger.info("Step 1: Data preprocessing")
            dataset = await self.preprocessor.prepare_ml_dataset(currency)
            
            train_df = dataset['train']
            test_df = dataset['test']
            full_df = dataset['full']
            
            logger.info(f"Dataset prepared: {len(train_df)} train, {len(test_df)} test samples")
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering")
            train_features = self.feature_engineer.create_features(train_df)
            test_features = self.feature_engineer.create_features(test_df)
            
            # Get feature columns from training data to ensure consistency
            feature_names = self.feature_engineer.get_feature_columns(train_features)
            
            # Prepare features for ML using the same feature columns
            train_ml_data = self.feature_engineer.prepare_features_for_ml(train_features)
            test_ml_data = self.feature_engineer.prepare_features_for_ml(test_features)
            
            # Ensure both datasets have the same features
            X_train = train_ml_data['X']
            y_train = train_ml_data['y']
            X_test = test_ml_data['X']
            y_test = test_ml_data['y']
            
            # If test data has fewer features, pad with zeros
            if X_train.shape[1] != X_test.shape[1]:
                logger.warning(f"Feature mismatch: train has {X_train.shape[1]} features, test has {X_test.shape[1]} features")
                if X_train.shape[1] > X_test.shape[1]:
                    # Pad test data with zeros
                    padding = np.zeros((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))
                    X_test = np.hstack([X_test, padding])
                    logger.info(f"Padded test data to match training features: {X_test.shape}")
                else:
                    # Pad training data with zeros
                    padding = np.zeros((X_train.shape[0], X_test.shape[1] - X_train.shape[1]))
                    X_train = np.hstack([X_train, padding])
                    logger.info(f"Padded training data to match test features: {X_train.shape}")
            
            feature_names = train_ml_data['feature_names']
            
            logger.info(f"Features prepared: {len(feature_names)} features")
            
            # Step 3: Model training
            logger.info("Step 3: Model training")
            training_results = self.model_trainer.train_all_models(
                X_train, X_test, y_train, y_test
            )
            
            # Step 4: Save models
            logger.info("Step 4: Saving models")
            self.model_trainer.save_models(currency, feature_names)
            
            # Create summary
            summary = {
                'currency': currency,
                'prediction_horizon': self.prediction_horizon,
                'feature_window': self.feature_window,
                'dataset_info': {
                    'total_samples': len(full_df),
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'features_count': len(feature_names),
                    'date_range': {
                        'start': str(full_df['date'].min()),
                        'end': str(full_df['date'].max())
                    }
                },
                'model_results': training_results['results'],
                'best_model': training_results['best_model'],
                'summary': training_results['summary'],
                'feature_names': feature_names,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"ML pipeline completed for {currency}")
            logger.info(f"Best model: {training_results['best_model']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in ML pipeline for {currency}: {str(e)}")
            raise


async def train_currency(currency: str, args: argparse.Namespace) -> dict:
    """
    Train models for a specific currency
    
    Args:
        currency: Currency to train models for
        args: Command line arguments
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Starting model training for {currency}")
    
    # Initialize pipeline
    pipeline = MLPipeline(
        prediction_horizon=args.prediction_horizon,
        feature_window=args.feature_window
    )
    
    # Run training
    results = await pipeline.run_full_pipeline(currency)
    
    # Save results to file
    results_file = f"training_results_{currency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path = os.path.join("models", results_file)
    
    os.makedirs("models", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training results saved to {results_path}")
    
    return results


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train ML models for crypto price prediction')
    
    parser.add_argument('--currency', choices=['BTC', 'ETH'], 
                       help='Currency to train models for')
    parser.add_argument('--all', action='store_true',
                       help='Train models for all currencies (BTC and ETH)')
    parser.add_argument('--prediction-horizon', type=int, default=7,
                       help='Days ahead to predict (default: 7)')
    parser.add_argument('--feature-window', type=int, default=7,
                       help='Days of historical data for features (default: 7)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.currency and not args.all:
        parser.error("Must specify either --currency or --all")
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine currencies to train
    if args.all:
        currencies = ['BTC', 'ETH']
    else:
        currencies = [args.currency]
    
    logger.info(f"Training models for: {', '.join(currencies)}")
    logger.info(f"Prediction horizon: {args.prediction_horizon} days")
    logger.info(f"Feature window: {args.feature_window} days")
    
    # Train models for each currency
    all_results = {}
    
    for currency in currencies:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING MODELS FOR {currency}")
            logger.info(f"{'='*50}")
            
            results = await train_currency(currency, args)
            all_results[currency] = results
            
            # Print summary
            best_model = results['best_model']
            best_accuracy = results['model_results'][best_model]['test_accuracy']
            best_f1 = results['model_results'][best_model]['test_f1']
            
            print(f"\n✅ {currency} Training Complete!")
            print(f"   Best Model: {best_model}")
            print(f"   Test Accuracy: {best_accuracy:.4f}")
            print(f"   Test F1 Score: {best_f1:.4f}")
            print(f"   Features Used: {len(results['feature_names'])}")
            
        except Exception as e:
            logger.error(f"Failed to train models for {currency}: {str(e)}")
            all_results[currency] = {'error': str(e)}
            print(f"\n❌ {currency} Training Failed: {str(e)}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    successful_currencies = [c for c, r in all_results.items() if 'error' not in r]
    failed_currencies = [c for c, r in all_results.items() if 'error' in r]
    
    print(f"✅ Successful: {len(successful_currencies)}/{len(currencies)}")
    if successful_currencies:
        print(f"   Currencies: {', '.join(successful_currencies)}")
    
    if failed_currencies:
        print(f"❌ Failed: {len(failed_currencies)}/{len(currencies)}")
        print(f"   Currencies: {', '.join(failed_currencies)}")
    
    # Model comparison across currencies
    if len(successful_currencies) > 1:
        print(f"\n{'='*30}")
        print("MODEL COMPARISON")
        print(f"{'='*30}")
        
        for currency in successful_currencies:
            results = all_results[currency]
            best_model = results['best_model']
            accuracy = results['model_results'][best_model]['test_accuracy']
            f1 = results['model_results'][best_model]['test_f1']
            
            print(f"{currency}: {best_model} (Acc: {accuracy:.4f}, F1: {f1:.4f})")
    
    print(f"\n🎉 Training pipeline completed!")
    
    # Save combined results
    combined_results_file = f"combined_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    combined_results_path = os.path.join("models", combined_results_file)
    
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Combined results saved to: {combined_results_path}")


if __name__ == "__main__":
    asyncio.run(main()) 