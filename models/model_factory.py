"""
Factory module for creating cryptocurrency price prediction models.
Supports various model types including Random Forest, XGBoost, and LSTM.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(
    model_type: str,
    model_params: Dict,
    feature_columns: List[str],
    target_column: str
) -> object:
    """
    Create and return a model instance based on the specified type and parameters.
    
    Args:
        model_type: Type of model to create ('random_forest', 'xgboost', 'lstm')
        model_params: Dictionary of model parameters
        feature_columns: List of feature column names
        target_column: Target column name
        
    Returns:
        Initialized model instance
    """
    logger.info(f"Creating {model_type} model with parameters: {model_params}")
    
    if model_type == 'random_forest':
        return RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")