"""
Clean Prediction Pipeline for Crypto Price Prediction

This module provides a robust prediction pipeline that properly handles:
- Loading models with correct feature names
- Feature preparation and alignment
- Realistic confidence calibration
- Prediction saving to database
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database import db_manager
from app.logger import logger
from .data_preprocessor import CryptoDataPreprocessor
from .feature_engineering import CryptoFeatureEngineer
from .clean_model_trainer import CleanModelTrainer


class CleanPredictionPipeline:
    """Clean, robust prediction pipeline that fixes feature mismatch issues"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the prediction pipeline"""
        self.models_dir = models_dir
        self.preprocessor = CryptoDataPreprocessor()
        self.feature_engineer = CryptoFeatureEngineer()
        self.model_trainer = CleanModelTrainer(models_dir)
        
        logger.info("CleanPredictionPipeline initialized")
    
    def find_best_model(self, currency: str) -> Optional[str]:
        """
        Find the best model for a currency based on F1 score
        
        Args:
            currency: Currency (BTC/ETH)
            
        Returns:
            Path to best model file or None if not found
        """
        model_files = []
        
        # Find all model files for this currency
        for filename in os.listdir(self.models_dir):
            if filename.startswith(f"{currency}_") and filename.endswith('.pkl'):
                filepath = os.path.join(self.models_dir, filename)
                model_files.append(filepath)
        
        if not model_files:
            logger.warning(f"No models found for {currency}")
            return None
        
        # Load each model and compare F1 scores
        best_model = None
        best_f1 = 0
        
        for filepath in model_files:
            try:
                _, metadata = self.model_trainer.load_model(filepath)
                f1_score = metadata.get('results', {}).get('test_f1', 0)
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = filepath
                    
            except Exception as e:
                logger.warning(f"Failed to load model {filepath}: {str(e)}")
        
        if best_model:
            logger.info(f"Best model for {currency}: {os.path.basename(best_model)} (F1: {best_f1:.4f})")
        
        return best_model
    
    async def prepare_prediction_features(self, currency: str, 
                                        prediction_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Prepare features for making a prediction
        
        Args:
            currency: Currency (BTC/ETH)
            prediction_date: Date to make prediction for
            
        Returns:
            DataFrame with features ready for prediction
        """
        if prediction_date is None:
            prediction_date = datetime.now(timezone.utc)
        
        # Ensure prediction_date is timezone-aware
        if prediction_date.tzinfo is None:
            prediction_date = prediction_date.replace(tzinfo=timezone.utc)
        
        # Load recent data (need historical data for feature engineering)
        start_date = prediction_date - timedelta(days=60)
        end_date = prediction_date
        
        logger.info(f"Loading data for {currency} from {start_date.date()} to {end_date.date()}")
        
        # Load data using preprocessor
        data = await self.preprocessor.load_data(currency, start_date, end_date)
        
        if data['prices'].empty:
            raise ValueError(f"No price data available for {currency}")
        
        # Merge price and sentiment data
        merged_df = self.preprocessor.merge_data(data['prices'], data['sentiment'])
        
        if merged_df.empty:
            raise ValueError(f"No merged data available for {currency}")
        
        # Create features
        features_df = self.feature_engineer.create_features(merged_df)
        
        if features_df.empty:
            raise ValueError("No features could be generated from the data")
        
        logger.info(f"Generated features: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
        
        # Return the latest row for prediction
        return features_df.iloc[-1:].copy()
    
    def align_features(self, features_df: pd.DataFrame, 
                      expected_features: List[str]) -> np.ndarray:
        """
        Align features with what the model expects
        
        Args:
            features_df: DataFrame with generated features
            expected_features: List of feature names the model expects
            
        Returns:
            Numpy array with features in correct order
        """
        # Find available and missing features
        available_features = [col for col in expected_features if col in features_df.columns]
        missing_features = [col for col in expected_features if col not in features_df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
            
            # Add missing features with zero values
            for feature in missing_features:
                features_df[feature] = 0.0
        
        # Select features in the exact order expected by the model
        aligned_features = features_df[expected_features].fillna(0).values
        
        logger.info(f"Aligned features: {aligned_features.shape[1]} features ready for prediction")
        
        return aligned_features
    
    def calibrate_confidence(self, raw_confidence: float, model_metadata: Dict[str, Any]) -> float:
        """
        Calibrate raw model confidence to realistic levels
        
        Args:
            raw_confidence: Raw confidence from model
            model_metadata: Model metadata with performance metrics
            
        Returns:
            Calibrated confidence score
        """
        # Get model performance metrics
        results = model_metadata.get('results', {})
        test_accuracy = results.get('test_accuracy', 0.5)
        test_f1 = results.get('test_f1', 0.5)
        model_type = model_metadata.get('model_type', 'unknown')
        
        # Calculate performance factor
        performance_factor = (test_accuracy + test_f1) / 2
        
        # Calibrate confidence based on model performance
        calibrated_confidence = raw_confidence * performance_factor
        
        # Apply model-specific adjustments
        if model_type == 'logistic_regression':
            calibrated_confidence *= 0.9  # More conservative
        elif model_type == 'random_forest':
            calibrated_confidence *= 0.95  # Moderately confident
        
        # Realistic bounds for crypto predictions
        min_confidence = 0.45  # 45% minimum
        max_confidence = 0.85  # 85% maximum
        
        # Ensure confidence is within realistic bounds
        final_confidence = max(min_confidence, min(max_confidence, calibrated_confidence))
        
        logger.debug(f"Confidence calibration: raw={raw_confidence:.4f}, "
                    f"performance_factor={performance_factor:.4f}, "
                    f"calibrated={calibrated_confidence:.4f}, "
                    f"final={final_confidence:.4f}")
        
        return final_confidence
    
    async def make_prediction(self, currency: str, 
                            prediction_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Make a price prediction for a currency
        
        Args:
            currency: Currency (BTC/ETH)
            prediction_date: Date to make prediction for
            
        Returns:
            Prediction result dictionary
        """
        try:
            logger.info(f"Making prediction for {currency}...")
            
            # Find best model
            model_path = self.find_best_model(currency)
            if not model_path:
                raise ValueError(f"No trained models found for {currency}")
            
            # Load model
            model, metadata = self.model_trainer.load_model(model_path)
            
            # Verify feature_names is a list
            feature_names = metadata.get('feature_names')
            if not isinstance(feature_names, list):
                raise TypeError(f"feature_names is not a list: {type(feature_names)}")
            
            logger.info(f"Loaded model: {metadata.get('model_type')} with {len(feature_names)} features")
            
            # Prepare features
            features_df = await self.prepare_prediction_features(currency, prediction_date)
            
            # Align features with model expectations
            X = self.align_features(features_df, feature_names)
            
            # Make prediction
            predicted_class = model.predict(X)[0]
            predicted_probabilities = model.predict_proba(X)[0]
            raw_confidence = max(predicted_probabilities)
            
            # Calibrate confidence
            calibrated_confidence = self.calibrate_confidence(raw_confidence, metadata)
            
            # Convert to direction
            predicted_direction = "UP" if predicted_class == 1 else "DOWN"
            
            # Create prediction result
            prediction_result = {
                'currency': currency,
                'prediction_date': (prediction_date or datetime.now(timezone.utc)),
                'prediction_horizon': 7,  # 7 days ahead
                'predicted_direction': predicted_direction,
                'confidence_score': float(calibrated_confidence),
                'raw_confidence': float(raw_confidence),
                'model_version': f"{metadata.get('model_type')}_{metadata.get('timestamp')}",
                'model_metadata': {
                    'model_type': metadata.get('model_type'),
                    'timestamp': metadata.get('timestamp'),
                    'test_accuracy': metadata.get('results', {}).get('test_accuracy'),
                    'test_f1': metadata.get('results', {}).get('test_f1'),
                    'feature_count': len(feature_names)
                }
            }
            
            logger.info(f"Prediction complete: {currency} -> {predicted_direction} "
                       f"(confidence: {calibrated_confidence:.2%})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction for {currency}: {str(e)}")
            raise
    
    async def save_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Save prediction to database
        
        Args:
            prediction: Prediction result dictionary
            
        Returns:
            Database record ID
        """
        try:
            # Prepare data for database
            prediction_date = prediction['prediction_date']
            if hasattr(prediction_date, 'isoformat'):
                prediction_date = prediction_date.isoformat()
            elif hasattr(prediction_date, 'strftime'):
                prediction_date = prediction_date.strftime('%Y-%m-%d')
            else:
                prediction_date = str(prediction_date)
            
            prediction_data = {
                'currency': prediction['currency'],
                'prediction_date': prediction_date,
                'prediction_horizon': prediction['prediction_horizon'],
                'predicted_direction': prediction['predicted_direction'],
                'confidence_score': float(prediction['confidence_score']),
                'model_version': str(prediction['model_version'])
            }
            
            # Insert into database
            record_id = await db_manager.insert_prediction(prediction_data)
            
            if record_id:
                logger.info(f"Saved prediction to database with ID: {record_id}")
            else:
                logger.warning("Failed to save prediction to database")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise
    
    async def make_and_save_prediction(self, currency: str) -> Dict[str, Any]:
        """
        Make a prediction and save it to database
        
        Args:
            currency: Currency (BTC/ETH)
            
        Returns:
            Prediction result with database ID
        """
        # Make prediction
        prediction = await self.make_prediction(currency)
        
        # Save to database
        prediction_id = await self.save_prediction(prediction)
        prediction['id'] = prediction_id
        
        return prediction 