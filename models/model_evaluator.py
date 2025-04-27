"""
Model evaluation and backtesting functionality
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates model performance through backtesting."""
    
    def __init__(self, price_predictor, data_fetcher):
        self.price_predictor = price_predictor
        self.data_fetcher = data_fetcher
        
    def calculate_metrics(self, symbol: str, timeframe: str = "24h", 
                         evaluation_days: int = 30) -> Dict[str, float]:
        """
        Calculate performance metrics by backtesting on recent historical data.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Prediction timeframe (24h, 7d, 30d)
            evaluation_days: Number of days to use for evaluation
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Fetch historical data
            historical_data = self.data_fetcher.fetch_historical_data(
                symbol, 
                days=evaluation_days + 30  # Extra data for longer timeframes
            )
            
            if historical_data is None or historical_data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            predictions = []
            actuals = []
            
            # Calculate prediction interval based on timeframe
            interval_days = 1 if timeframe == "24h" else 7 if timeframe == "7d" else 30
            
            # Generate predictions and collect actual values
            for i in range(evaluation_days):
                current_date = historical_data.index[-evaluation_days + i]
                
                # Get data up to current date
                data_up_to_date = historical_data[:current_date]
                if len(data_up_to_date) < 2:  # Need some history for prediction
                    continue
                
                # Get current price
                current_price = float(data_up_to_date.iloc[-1]['close'])
                
                # Make prediction
                prediction = self.price_predictor.predict(
                    symbol=symbol,
                    timeframe=timeframe,
                    current_price=current_price
                )
                
                predicted_price = prediction['predicted_price']
                predictions.append(predicted_price)
                
                # Get actual future price
                future_date_idx = min(
                    len(historical_data) - 1,
                    historical_data.index.get_loc(current_date) + interval_days
                )
                actual_price = float(historical_data.iloc[future_date_idx]['close'])
                actuals.append(actual_price)
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            metrics = {}
            
            # Percentage errors
            percent_errors = np.abs((predictions - actuals) / actuals) * 100
            metrics['mae'] = float(np.mean(percent_errors))
            metrics['rmse'] = float(np.sqrt(np.mean(np.square(percent_errors))))
            
            # Direction accuracy
            price_changes_predicted = np.diff(np.concatenate(([actuals[0]], predictions)))
            price_changes_actual = np.diff(actuals)
            correct_directions = np.sum(np.sign(price_changes_predicted) == np.sign(price_changes_actual))
            metrics['direction_accuracy'] = float(correct_directions / (len(actuals) - 1))
            
            # R-squared
            mean_actual = np.mean(actuals)
            ss_tot = np.sum((actuals - mean_actual) ** 2)
            ss_res = np.sum((actuals - predictions) ** 2)
            metrics['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
            
            # Success rate (predictions within 5% of actual)
            within_threshold = np.sum(percent_errors <= 5.0)
            metrics['success_rate'] = float(within_threshold / len(percent_errors))
            
            logger.info(f"Performance metrics for {symbol} ({timeframe}): {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'direction_accuracy': 0.0,
                'r2': 0.0,
                'success_rate': 0.0
            }
    
    def evaluate_all_timeframes(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance across all timeframes.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary containing metrics for each timeframe
        """
        timeframes = ["24h", "7d", "30d"]
        return {
            timeframe: self.calculate_metrics(symbol, timeframe)
            for timeframe in timeframes
        } 