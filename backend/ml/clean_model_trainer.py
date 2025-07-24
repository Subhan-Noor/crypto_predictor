"""
Clean Model Training System for Crypto Price Prediction

This module provides a robust, clean implementation of model training 
that fixes the feature mismatch issues by properly handling feature names
and metadata.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logger import logger


class CleanModelTrainer:
    """Clean, robust model trainer that properly handles feature names"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the clean model trainer"""
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []  # Properly store feature names as list
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"CleanModelTrainer initialized with models directory: {models_dir}")
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target from dataframe
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define target column and exclude non-feature columns
        target_col = 'target'
        exclude_cols = ['date', 'target', 'future_close', 'price_change_pct', 'currency']
        
        # Get feature columns (all columns except excluded ones)
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found in dataframe")
        
        # Extract features and target
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
        
        return X, y, feature_cols
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray, 
                   feature_names: List[str], model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train a single model with the given data
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data  
            feature_names: List of feature column names
            model_type: Type of model to train
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_type} model...")
        
        # Store feature names for this training session
        self.feature_names = feature_names.copy()  # Ensure it's a list copy
        
        # Initialize and train model based on type
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Store results
        results = {
            'model_type': model_type,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'features_count': len(feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Store model and results
        self.models[model_type] = model
        self.results[model_type] = results
        
        logger.info(f"{model_type} training complete:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test F1 Score: {test_f1:.4f}")
        
        return results
    
    def save_model(self, currency: str, model_type: str) -> str:
        """
        Save a trained model to disk with proper metadata
        
        Args:
            currency: Currency (BTC/ETH)
            model_type: Type of model to save
            
        Returns:
            Path to saved model file
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{currency}_{model_type}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Prepare metadata with proper feature names
        metadata = {
            'currency': currency,
            'model_type': model_type,
            'timestamp': timestamp,
            'feature_names': self.feature_names,  # This is guaranteed to be a list
            'feature_count': len(self.feature_names),
            'results': self.results.get(model_type, {}),
            'scaler': self.scalers.get('standard_scaler', None)
        }
        
        # Verify feature_names is a list
        if not isinstance(metadata['feature_names'], list):
            raise TypeError(f"feature_names must be a list, got {type(metadata['feature_names'])}")
        
        # Save model and metadata
        model_data = {
            'model': self.models[model_type],
            'metadata': metadata
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info(f"Saved {model_type} model to: {filepath}")
        logger.info(f"Model metadata: currency={currency}, features={len(self.feature_names)}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            Tuple of (model, metadata)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        model = model_data['model']
        metadata = model_data['metadata']
        
        # Verify feature_names is a list
        if not isinstance(metadata.get('feature_names'), list):
            raise TypeError(f"Loaded feature_names is not a list: {type(metadata.get('feature_names'))}")
        
        logger.info(f"Loaded model from: {filepath}")
        logger.info(f"Model type: {metadata.get('model_type')}, features: {len(metadata.get('feature_names', []))}")
        
        return model, metadata
    
    def find_latest_model(self, currency: str, model_type: str = None) -> Optional[str]:
        """
        Find the latest model file for a currency
        
        Args:
            currency: Currency (BTC/ETH)
            model_type: Specific model type (optional)
            
        Returns:
            Path to latest model file or None if not found
        """
        pattern = f"{currency}_*" if model_type is None else f"{currency}_{model_type}_*"
        model_files = []
        
        for filename in os.listdir(self.models_dir):
            if filename.startswith(pattern.replace('*', '')) and filename.endswith('.pkl'):
                filepath = os.path.join(self.models_dir, filename)
                model_files.append(filepath)
        
        if not model_files:
            return None
        
        # Sort by modification time and return the latest
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]
    
    def train_all_models(self, features_df: pd.DataFrame, currency: str) -> Dict[str, Any]:
        """
        Train all model types for a currency
        
        Args:
            features_df: DataFrame with features and target
            currency: Currency (BTC/ETH)
            
        Returns:
            Training summary
        """
        logger.info(f"Training all models for {currency}...")
        
        # Prepare data
        X, y, feature_names = self.prepare_data(features_df)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store feature names for saving
        self.feature_names = feature_names
        
        # Train models
        model_types = ["random_forest", "logistic_regression"]
        trained_models = {}
        
        for model_type in model_types:
            try:
                results = self.train_model(X_train, y_train, X_test, y_test, feature_names, model_type)
                model_path = self.save_model(currency, model_type)
                trained_models[model_type] = {
                    'results': results,
                    'model_path': model_path
                }
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {currency}: {str(e)}")
                trained_models[model_type] = {
                    'error': str(e)
                }
        
        return trained_models 