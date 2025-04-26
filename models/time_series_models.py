"""
Time series models for cryptocurrency price prediction.
Implements ARIMA and Prophet models with proper preprocessing and evaluation capabilities.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Union, List
import logging
from .price_predictor import BaseMLModel

logger = logging.getLogger(__name__)

class ARIMAModel(BaseMLModel):
    """ARIMA model implementation."""
    
    def __init__(self, feature_columns: List[str], target_column: str, order: Tuple[int, int, int] = (5,1,2)):
        """
        Initialize ARIMA model.
        
        Args:
            feature_columns: List of feature column names (only target will be used)
            target_column: Target column name
            order: ARIMA order (p,d,q)
        """
        super().__init__(feature_columns, target_column)
        self.order = order
        self.model = None
        
    def train(self, df: pd.DataFrame) -> None:
        """Train ARIMA model."""
        logger.info(f"Training ARIMA model with order {self.order}...")
        
        # Use only the target column for ARIMA
        data = df[self.target_column].astype(float)
        
        try:
            self.model = ARIMA(data, order=self.order)
            self.model = self.model.fit()
            logger.info("ARIMA model training completed")
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using ARIMA model."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # For ARIMA, we need to predict one step at a time
            predictions = []
            for i in range(len(df)):
                forecast = self.model.forecast(steps=1)
                predictions.append(forecast[0])
                
                # Update the model with actual value if available
                if i < len(df) - 1:
                    actual = df[self.target_column].iloc[i]
                    self.model = self.model.append([actual])
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {str(e)}")
            raise

class ProphetModel(BaseMLModel):
    """Facebook Prophet model implementation."""
    
    def __init__(self, 
                 feature_columns: List[str],
                 target_column: str,
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        """
        Initialize Prophet model.
        
        Args:
            feature_columns: List of feature column names (only target will be used)
            target_column: Target column name
            seasonality_mode: Type of seasonality ('multiplicative' or 'additive')
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Flexibility of the seasonality
        """
        super().__init__(feature_columns, target_column)
        self.params = {
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale
        }
        self.model = None
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet."""
        # Prophet requires columns named 'ds' (date) and 'y' (target)
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df[self.target_column].values
        return prophet_df
        
    def train(self, df: pd.DataFrame) -> None:
        """Train Prophet model."""
        logger.info("Training Prophet model...")
        
        try:
            data = self._prepare_data(df)
            self.model = Prophet(**self.params)
            self.model.fit(data)
            logger.info("Prophet model training completed")
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using Prophet model."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # Create future dataframe
            future = pd.DataFrame()
            future['ds'] = df.index
            
            # Make predictions
            forecast = self.model.predict(future)
            return forecast['yhat'].values
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {str(e)}")
            raise

def create_time_series_model(model_type: str, **kwargs) -> BaseMLModel:
    """
    Factory function to create time series models.
    
    Args:
        model_type: Type of model ('arima' or 'prophet')
        **kwargs: Model parameters
    
    Returns:
        Configured time series model instance
    """
    models = {
        'arima': ARIMAModel,
        'prophet': ProphetModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(models.keys())}")
        
    return models[model_type.lower()](**kwargs) 