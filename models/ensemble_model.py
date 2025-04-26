"""
Ensemble model implementation that combines predictions from multiple base models.
Supports both weighted averaging and stacking approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import logging
from .price_predictor import BaseMLModel
from .model_factory import create_model

logger = logging.getLogger(__name__)

class WeightedEnsemble(BaseMLModel):
    """Ensemble model that combines predictions using weighted averaging."""
    
    def __init__(self,
                 feature_columns: List[str],
                 target_column: str,
                 base_models_config: List[Dict[str, Any]],
                 weights: Optional[List[float]] = None):
        """
        Initialize weighted ensemble.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            base_models_config: List of dictionaries containing base model configurations
                              Each dict should have 'type' and 'params' keys
            weights: List of weights for each base model (optional)
                    If None, equal weights will be used
        """
        super().__init__(feature_columns, target_column)
        self.base_models = []
        self.base_models_config = base_models_config
        
        # Initialize weights
        if weights is None:
            self.weights = [1.0 / len(base_models_config)] * len(base_models_config)
        else:
            if len(weights) != len(base_models_config):
                raise ValueError("Number of weights must match number of base models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
            
        # Initialize base models
        for config in base_models_config:
            model = create_model(
                config['type'],
                config.get('params', {}),
                feature_columns=feature_columns,
                target_column=target_column
            )
            self.base_models.append(model)
            
    def train(self, df: pd.DataFrame) -> None:
        """Train all base models."""
        logger.info("Training ensemble base models...")
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training model {i+1}/{len(self.base_models)}: {self.base_models_config[i]['type']}")
            model.train(df)
            
        logger.info("Ensemble training completed")
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        predictions = []
        
        for model in self.base_models:
            pred = model.predict(df)
            predictions.append(pred)
            
        # Stack predictions and compute weighted average
        stacked_predictions = np.stack(predictions, axis=-1)
        weighted_predictions = np.average(stacked_predictions, axis=-1, weights=self.weights)
        
        return weighted_predictions
        
    def optimize_weights(self, df: pd.DataFrame, validation_size: float = 0.2) -> None:
        """Optimize ensemble weights using validation data performance."""
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions from all models on validation set
        val_predictions = []
        for model in self.base_models:
            pred = model.predict(df)
            val_predictions.append(pred)
            
        # Stack predictions and prepare target
        X = np.stack(val_predictions, axis=-1)
        y = df[self.target_column].values
        
        # Use linear regression to find optimal weights
        lr = LinearRegression(fit_intercept=False, positive=True)
        lr.fit(X, y)
        
        # Normalize weights to sum to 1
        weights = lr.coef_
        weights = weights / np.sum(weights)
        
        self.weights = weights.tolist()
        logger.info(f"Optimized weights: {dict(zip([m['type'] for m in self.base_models_config], self.weights))}")


class StackingEnsemble(BaseMLModel):
    """Ensemble model that uses stacking to combine predictions."""
    
    def __init__(self,
                 feature_columns: List[str],
                 target_column: str,
                 base_models_config: List[Dict[str, Any]],
                 meta_model_config: Dict[str, Any]):
        """
        Initialize stacking ensemble.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            base_models_config: List of dictionaries containing base model configurations
            meta_model_config: Dictionary containing meta-model configuration
        """
        super().__init__(feature_columns, target_column)
        self.base_models = []
        self.base_models_config = base_models_config
        
        # Initialize base models
        for config in base_models_config:
            model = create_model(
                config['type'],
                config.get('params', {}),
                feature_columns=feature_columns,
                target_column=target_column
            )
            self.base_models.append(model)
            
        # Initialize meta-model
        meta_features = [f'model_{i}_pred' for i in range(len(base_models_config))]
        self.meta_model = create_model(
            meta_model_config['type'],
            meta_model_config.get('params', {}),
            feature_columns=meta_features,
            target_column=target_column
        )
        
    def train(self, df: pd.DataFrame) -> None:
        """Train both base models and meta-model."""
        logger.info("Training stacking ensemble...")
        
        # Split data for meta-model training
        train_df, meta_train_df = train_test_split(df, test_size=0.3, shuffle=False)
        
        # Train base models on training data
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {self.base_models_config[i]['type']}")
            model.train(train_df)
        
        # Generate meta-features
        meta_features = []
        for model in self.base_models:
            pred = model.predict(meta_train_df)
            meta_features.append(pred)
        
        # Prepare meta-model training data
        X_meta = np.column_stack(meta_features)
        y_meta = meta_train_df[self.target_column].values
        
        # Create meta-features DataFrame
        meta_df = pd.DataFrame(
            X_meta,
            columns=[f'model_{i}_pred' for i in range(len(self.base_models))],
            index=meta_train_df.index
        )
        meta_df[self.target_column] = y_meta
        
        # Train meta-model
        logger.info("Training meta-model...")
        self.meta_model.train(meta_df)
        
        logger.info("Stacking ensemble training completed")
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(df)
            base_predictions.append(pred)
            
        # Prepare meta-features
        X_meta = np.column_stack(base_predictions)
        
        # Create meta-features DataFrame
        meta_df = pd.DataFrame(
            X_meta,
            columns=[f'model_{i}_pred' for i in range(len(self.base_models))],
            index=df.index
        )
        
        # Make final predictions using meta-model
        return self.meta_model.predict(meta_df)


def create_ensemble_model(ensemble_type: str, config: Dict[str, Any]) -> BaseMLModel:
    """
    Factory function to create ensemble models.
    
    Args:
        ensemble_type: Type of ensemble ('weighted' or 'stacking')
        config: Configuration dictionary for the ensemble
        
    Returns:
        Configured ensemble model instance
    """
    ensembles = {
        'weighted': WeightedEnsemble,
        'stacking': StackingEnsemble
    }
    
    if ensemble_type.lower() not in ensembles:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}. Available types: {list(ensembles.keys())}")
        
    return ensembles[ensemble_type.lower()](**config) 