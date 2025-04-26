"""
Test script for the evaluation framework components.
Tests backtesting, visualization, and optimization functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from evaluation.backtester import Backtester, SimpleThresholdStrategy, BacktestResult
from evaluation.visualizer import PerformanceVisualizer
from evaluation.model_optimizer import ModelOptimizer
from models.model_factory import create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate sample cryptocurrency data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=n_days)
    np.random.seed(42)
    
    # Generate synthetic price data with trend and seasonality
    t = np.arange(n_days)
    trend = 0.001 * t
    seasonality = 0.1 * np.sin(2 * np.pi * t / 30)  # 30-day cycle
    noise = 0.02 * np.random.randn(n_days)
    
    # Combine components
    log_price = 4.6 + trend + seasonality + noise.cumsum()
    close = np.exp(log_price)
    volume = np.exp(10 + 0.1 * np.random.randn(n_days))
    
    # Create base DataFrame
    df = pd.DataFrame(index=dates)
    df['close'] = close
    df['volume'] = volume
    
    # Add OHLC data first
    df['open'] = df['close'].shift(1)
    df.loc[df.index[0], 'open'] = df['close'].iloc[0] * (1 + np.random.randn() * 0.001)  # Set first open price
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.randn(n_days) * 0.001))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.randn(n_days) * 0.001))
    
    # Calculate technical indicators
    df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
    df['ma21'] = df['close'].rolling(window=21, min_periods=1).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Fill initial NaN with neutral RSI
    
    return df

def test_backtester(data: pd.DataFrame) -> None:
    """Test the backtesting functionality"""
    logger.info("Testing backtester...")
    logger.info(f"Data shape: {data.shape}")
    
    # Initialize backtester
    feature_columns = ['open', 'high', 'low', 'volume', 'ma7', 'ma21', 'rsi']
    target_column = 'close'
    backtester = Backtester(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    # Create a simple model for testing
    model = create_model(
        model_type='random_forest',
        model_params={'n_estimators': 100, 'max_depth': 5},
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    # Create strategy
    strategy = SimpleThresholdStrategy({'threshold': 0.01})
    
    # Run walk-forward optimization with smaller window
    results = backtester.walk_forward_optimization(
        model=model,
        strategy=strategy,
        window_size=30,  # 1 month
        step_size=10     # ~2 weeks
    )
    
    logger.info(f"Generated {len(results)} backtest results")
    for i, result in enumerate(results):
        logger.info(f"Window {i+1} Metrics: {result.metrics}")

def test_visualizer(data: pd.DataFrame) -> None:
    """Test the visualization functionality"""
    logger.info("Testing visualizer...")
    
    # Create some sample predictions
    y_true = data['close'].values
    y_pred = y_true * (1 + np.random.randn(len(y_true)) * 0.1)  # Add some noise
    dates = data.index.values
    
    # Initialize visualizer
    visualizer = PerformanceVisualizer()
    
    # Create output directory
    output_dir = Path("evaluation/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate various plots
    visualizer.plot_prediction_vs_actual(
        dates=dates,
        y_true=y_true,
        y_pred=y_pred,
        coin_name="Test Coin",
        save_path=str(output_dir / "prediction_vs_actual.html")
    )
    
    visualizer.plot_error_distribution(
        y_true=y_true,
        y_pred=y_pred,
        save_path=str(output_dir / "error_distribution.html")
    )
    
    # Test feature importance plot
    feature_names = ['open', 'high', 'low', 'volume', 'ma7', 'ma21', 'rsi']
    importance_scores = np.random.rand(len(feature_names))
    visualizer.plot_feature_importance(
        feature_names=feature_names,
        importance_scores=importance_scores,
        save_path=str(output_dir / "feature_importance.html")
    )

def test_optimizer():
    """Test model optimization functionality"""
    # Create test data
    data = generate_sample_data()
    
    # Initialize backtester
    backtester = Backtester(
        data=data,
        feature_columns=['open', 'high', 'low', 'volume'],
        target_column='close',
        initial_capital=10000.0,
        transaction_costs=0.001
    )
    
    # Create strategy instance
    strategy = SimpleThresholdStrategy(parameters={'threshold': 0.01})
    
    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5]
    }
    
    # Initialize optimizer
    optimizer = ModelOptimizer(
        backtester=backtester,
        model_type='random_forest',
        feature_columns=['open', 'high', 'low', 'volume'],
        target_column='close',
        optimization_metric='sharpe_ratio'
    )
    
    # Run grid search optimization
    result = optimizer.grid_search(
        param_grid=param_grid,
        strategy=strategy,  # Pass strategy instance instead of params
        window_size=60,
        step_size=20
    )
    
    assert result is not None
    assert isinstance(result.best_parameters, dict)
    assert isinstance(result.best_metrics, dict)
    assert len(result.parameter_history) > 0
    assert len(result.metric_history) > 0

def main():
    """Run all tests"""
    logger.info("Starting evaluation framework tests...")
    
    # Generate sample data
    data = generate_sample_data()
    logger.info(f"Generated data shape: {data.shape}")
    
    # Run tests
    test_backtester(data)
    test_visualizer(data)
    test_optimizer()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 