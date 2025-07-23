"""
Prediction Pipeline for Crypto Price Prediction

This module handles:
- Loading trained models
- Real-time feature engineering
- Making predictions with confidence scores
- Storing predictions in database
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database import db_manager
from app.logger import logger
from .data_preprocessor import CryptoDataPreprocessor
from .feature_engineering import CryptoFeatureEngineer
from .model_trainer import CryptoModelTrainer


class CryptoPredictionPipeline:
    """Handles real-time crypto price predictions"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize prediction pipeline
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.loaded_models = {}
        self.preprocessor = CryptoDataPreprocessor()
        self.feature_engineer = CryptoFeatureEngineer()
        self.model_trainer = CryptoModelTrainer(models_dir)
        
    def load_latest_model(self, currency: str, model_type: str = "best") -> Tuple[Any, Dict[str, Any]]:
        """
        Load the latest trained model for a currency
        
        Args:
            currency: 'BTC' or 'ETH'
            model_type: 'best', 'logistic_regression', 'random_forest', or 'lstm'
            
        Returns:
            Tuple of (model, metadata)
        """
        model_key = f"{currency}_{model_type}"
        
        # Return cached model if already loaded
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Find latest model files for this currency
        if model_type == "best":
            # Load all models and find the best one
            all_models = {}
            for mt in ['logistic_regression', 'random_forest', 'lstm']:
                try:
                    model, metadata = self.load_latest_model(currency, mt)
                    all_models[mt] = (model, metadata)
                except Exception as e:
                    logger.warning(f"Could not load {mt} model for {currency}: {e}")
            
            if not all_models:
                raise ValueError(f"No trained models found for {currency}")
            
            # Find best model based on test F1 score
            best_model_type = None
            best_score = 0
            for mt, (model, metadata) in all_models.items():
                f1_score = metadata.get('results', {}).get('test_f1', 0)
                if f1_score > best_score:
                    best_score = f1_score
                    best_model_type = mt
            
            if best_model_type:
                model, metadata = all_models[best_model_type]
                self.loaded_models[model_key] = (model, metadata)
                logger.info(f"Loaded best model for {currency}: {best_model_type} (F1: {best_score:.4f})")
                return model, metadata
            else:
                raise ValueError(f"Could not determine best model for {currency}")
        
        else:
            # Load specific model type
            pattern = os.path.join(self.models_dir, f"{currency}_{model_type}_*.pkl")
            if model_type == 'lstm':
                pattern = os.path.join(self.models_dir, f"{currency}_{model_type}_*.pth")
            
            model_files = glob.glob(pattern)
            
            if not model_files:
                raise ValueError(f"No {model_type} model files found for {currency}")
            
            # Get the latest model file (by timestamp)
            latest_file = max(model_files, key=os.path.getctime)
            
            # Load the model
            model, metadata = self.model_trainer.load_model(latest_file)
            
            # Cache the loaded model
            self.loaded_models[model_key] = (model, metadata)
            
            logger.info(f"Loaded {model_type} model for {currency} from {latest_file}")
            return model, metadata
    
    async def prepare_prediction_data(self, currency: str, prediction_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Prepare data for making a prediction
        
        Args:
            currency: 'BTC' or 'ETH'
            prediction_date: Date to make prediction for (default: today)
            
        Returns:
            DataFrame with features ready for prediction
        """
        if prediction_date is None:
            prediction_date = datetime.now(timezone.utc).date()
        
        # Load recent data (we need historical data for feature engineering)
        start_date = (prediction_date - timedelta(days=60))
        end_date = prediction_date
        
        # Ensure start_date and end_date are timezone-aware
        if isinstance(start_date, datetime):
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
        else:
            start_date = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(end_date, datetime):
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

        # Load data
        data = await self.preprocessor.load_data(currency, start_date, end_date)
        
        if data['prices'].empty:
            raise ValueError(f"No price data available for {currency}")
        
        # Merge price and sentiment data
        merged_df = self.preprocessor.merge_data(data['prices'], data['sentiment'])
        
        # Create features
        features_df = self.feature_engineer.create_features(merged_df)
        
        # Return the latest row (for prediction)
        if features_df.empty:
            raise ValueError("No features could be generated from the data")
        
        return features_df.iloc[-1:].copy()  # Return last row as DataFrame
    
    async def make_prediction(self, currency: str, model_type: str = "best", 
                            prediction_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Make a price prediction for a currency
        
        Args:
            currency: 'BTC' or 'ETH'
            model_type: Type of model to use for prediction
            prediction_date: Date to make prediction for
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model
            model, metadata = self.load_latest_model(currency, model_type)
            
            # Prepare data
            features_df = await self.prepare_prediction_data(currency, prediction_date)
            
            # Get feature columns used during training
            feature_names = metadata['feature_names']
            
            # Select and order features as they were during training
            available_features = [col for col in feature_names if col in features_df.columns]
            missing_features = [col for col in feature_names if col not in features_df.columns]
            
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                # Fill missing features with zeros
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Prepare features in the correct order
            X = features_df[feature_names].fillna(0).values
            
            # Scale features using the scaler from training
            scaler = metadata.get('scaler')
            if scaler:
                X = scaler.transform(X)
            
            # Make prediction based on model type
            actual_model_type = metadata['model_name']
            
            if actual_model_type == 'lstm':
                # LSTM prediction
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = model(X_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            else:
                # Sklearn prediction
                predicted_class = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                confidence = probabilities[predicted_class]
            
            # Convert to prediction direction
            predicted_direction = "UP" if predicted_class == 1 else "DOWN"
            
            # Create prediction result
            prediction_result = {
                'currency': currency,
                'prediction_date': (prediction_date or datetime.now(timezone.utc).date()),
                'prediction_horizon': 7,  # 7 days ahead
                'predicted_direction': predicted_direction,
                'confidence_score': float(confidence),
                'model_version': f"{actual_model_type}_{metadata['timestamp']}",
                'features_used': feature_names,
                'model_metadata': {
                    'model_type': actual_model_type,
                    'training_date': metadata['timestamp'],
                    'test_accuracy': metadata.get('results', {}).get('test_accuracy'),
                    'test_f1': metadata.get('results', {}).get('test_f1')
                }
            }
            
            logger.info(f"Prediction for {currency}: {predicted_direction} (confidence: {confidence:.4f})")
            
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
            ID of saved prediction
        """
        try:
            # Prepare data for database
            prediction_data = {
                'currency': prediction['currency'],
                'prediction_date': prediction['prediction_date'],
                'prediction_horizon': prediction['prediction_horizon'],
                'predicted_direction': prediction['predicted_direction'],
                'confidence_score': prediction['confidence_score'],
                'model_version': prediction['model_version'],
                'features_used': prediction['features_used']
            }
            
            # Insert into database
            record_id = await db_manager.insert_prediction(prediction_data)
            
            logger.info(f"Saved prediction to database with ID: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise
    
    async def make_and_save_prediction(self, currency: str, model_type: str = "best") -> Dict[str, Any]:
        """
        Make a prediction and save it to database
        
        Args:
            currency: 'BTC' or 'ETH'
            model_type: Type of model to use
            
        Returns:
            Prediction result with database ID
        """
        # Make prediction
        prediction = await self.make_prediction(currency, model_type)
        
        # Save to database
        prediction_id = await self.save_prediction(prediction)
        prediction['id'] = prediction_id
        
        return prediction


# Utility functions for quick predictions
async def make_prediction(currency: str, model_type: str = "best") -> Dict[str, Any]:
    """
    Quick utility function to make a prediction
    
    Args:
        currency: 'BTC' or 'ETH'
        model_type: Type of model to use
        
    Returns:
        Prediction result
    """
    pipeline = CryptoPredictionPipeline()
    return await pipeline.make_prediction(currency, model_type)


async def make_daily_predictions() -> Dict[str, Any]:
    """
    Make daily predictions for both BTC and ETH
    
    Returns:
        Dictionary with predictions for both currencies
    """
    pipeline = CryptoPredictionPipeline()
    results = {}
    
    for currency in ['BTC', 'ETH']:
        try:
            prediction = await pipeline.make_and_save_prediction(currency)
            results[currency] = prediction
            logger.info(f"Daily prediction completed for {currency}")
        except Exception as e:
            logger.error(f"Failed to make daily prediction for {currency}: {str(e)}")
            results[currency] = {'error': str(e)}
    
    return results 