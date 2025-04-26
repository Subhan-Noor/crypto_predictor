"""
Backtesting framework for cryptocurrency prediction models.
Supports:
- Multiple models and cryptocurrencies
- Walk-forward optimization
- Various trading strategies
- Comprehensive performance metrics
- Detailed reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    model_name: str
    strategy_name: str
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.Series
    parameters: Dict
    test_period: Tuple[datetime, datetime]

class BacktestStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, parameters: Dict = None):
        self.parameters = parameters or {}
        
    def generate_signals(self, predictions: pd.Series, actual: pd.Series) -> pd.Series:
        """Generate trading signals based on predictions"""
        raise NotImplementedError("Subclasses must implement generate_signals")

class SimpleThresholdStrategy(BacktestStrategy):
    """Simple strategy based on prediction threshold"""
    
    def generate_signals(self, predictions: pd.Series, actual: pd.Series) -> pd.Series:
        threshold = self.parameters.get('threshold', 0.01)
        signals = pd.Series(0, index=predictions.index)
        
        # Calculate predicted returns
        pred_returns = predictions.pct_change()
        
        # Generate signals based on predicted returns
        signals[pred_returns > threshold] = 1  # Buy signal
        signals[pred_returns < -threshold] = -1  # Sell signal
        
        return signals

class Backtester:
    """Main backtesting engine"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        initial_capital: float = 10000.0,
        transaction_costs: float = 0.001
    ):
        """
        Initialize backtester with data and parameters
        
        Args:
            data: DataFrame with OHLCV and feature data
            feature_columns: List of feature column names
            target_column: Name of the target column (usually 'close')
            initial_capital: Starting capital for backtesting
            transaction_costs: Transaction cost as a fraction
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        
    def prepare_window_data(self, window_data: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for backtesting with train/test split"""
        split_idx = int(len(window_data) * train_size)
        train_data = window_data.iloc[:split_idx]
        test_data = window_data.iloc[split_idx:]
        return train_data, test_data
    
    def walk_forward_optimization(
        self,
        model: object,
        strategy: BacktestStrategy,
        window_size: int = 60,
        step_size: int = 20
    ) -> List[BacktestResult]:
        """
        Perform walk-forward optimization
        
        Args:
            model: Model object with fit/predict methods
            strategy: Trading strategy instance
            window_size: Size of rolling window in days
            step_size: Number of days to move forward
        """
        results = []
        data_length = len(self.data)
        
        # Need at least window_size days of data
        if data_length < window_size:
            logger.warning(f"Not enough data for window size {window_size}")
            return results
        
        # Walk forward through the data
        for start_idx in range(0, data_length - window_size, step_size):
            end_idx = start_idx + window_size
            if end_idx > data_length:
                break
                
            # Get window data
            window_data = self.data.iloc[start_idx:end_idx].copy()
            
            # Split into train/test
            train_data, test_data = self.prepare_window_data(window_data)
            
            try:
                # Train model
                model.fit(train_data[self.feature_columns], train_data[self.target_column])
                
                # Generate predictions
                predictions = model.predict(test_data[self.feature_columns])
                predictions = pd.Series(predictions, index=test_data.index)
                
                # Generate trading signals
                signals = strategy.generate_signals(predictions, test_data[self.target_column])
                
                # Calculate returns and metrics
                result = self._calculate_backtest_metrics(
                    predictions=predictions,
                    actual_data=test_data,
                    signals=signals,
                    model_name=model.__class__.__name__,
                    strategy_name=strategy.__class__.__name__,
                    parameters=strategy.parameters,
                    test_period=(test_data.index[0], test_data.index[-1])
                )
                
                results.append(result)
                logger.info(f"Completed backtest for window {start_idx}-{end_idx}")
                
            except Exception as e:
                logger.error(f"Error in walk-forward optimization: {str(e)}")
                continue
        
        return results
    
    def _calculate_backtest_metrics(
        self,
        predictions: pd.Series,
        actual_data: pd.DataFrame,
        signals: pd.Series,
        model_name: str,
        strategy_name: str,
        parameters: Dict,
        test_period: Tuple[datetime, datetime]
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        
        # Calculate prediction accuracy metrics
        actual_returns = actual_data[self.target_column].pct_change()
        strategy_returns = signals.shift(1) * actual_returns
        strategy_returns = strategy_returns.fillna(0)
        
        # Apply transaction costs
        trades = signals.diff().fillna(0)
        transaction_costs = abs(trades) * self.transaction_costs
        strategy_returns -= transaction_costs
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        
        # Record trades
        trades_df = pd.DataFrame({
            'signal': signals,
            'price': actual_data[self.target_column],
            'returns': strategy_returns
        })
        
        # Calculate performance metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0,
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'win_rate': np.mean(strategy_returns > 0),
            'rmse': np.sqrt(mean_squared_error(actual_data[self.target_column], predictions)),
            'mae': mean_absolute_error(actual_data[self.target_column], predictions)
        }
        
        return BacktestResult(
            model_name=model_name,
            strategy_name=strategy_name,
            metrics=metrics,
            trades=trades_df,
            equity_curve=equity_curve,
            parameters=parameters,
            test_period=test_period
        )
    
    def calculate_metrics(
        self,
        predictions: pd.Series,
        actual: pd.Series,
        signals: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Prediction accuracy metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        # Direction accuracy
        actual_direction = np.sign(actual.pct_change())
        pred_direction = np.sign(predictions.pct_change())
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Trading metrics
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': (equity_curve[-1] / equity_curve[0]) - 1,
            'win_rate': np.mean(returns > 0)
        }
    
    def run_backtest(
        self,
        predictions: pd.Series,
        actual_data: pd.DataFrame,
        model_name: str,
        strategy: BacktestStrategy
    ) -> BacktestResult:
        """
        Run backtest for a single model and strategy
        
        Args:
            predictions: Series of price predictions
            actual_data: DataFrame with actual price data
            model_name: Name of the model
            strategy: Trading strategy instance
        """
        # Generate trading signals
        signals = strategy.generate_signals(predictions, actual_data[self.target_column])
        
        # Calculate positions and equity curve
        position_sizes = self.initial_capital * signals
        price_changes = actual_data[self.target_column].pct_change()
        
        # Apply transaction costs
        trades = signals.diff().fillna(0)
        transaction_costs = abs(trades) * self.transaction_costs
        
        # Calculate returns and equity curve
        returns = (position_sizes.shift(1) * price_changes) - (self.initial_capital * transaction_costs)
        equity_curve = self.initial_capital + returns.cumsum()
        
        # Record trades
        trades_df = pd.DataFrame({
            'signal': signals,
            'price': actual_data[self.target_column],
            'position_size': position_sizes,
            'returns': returns
        })
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            predictions=predictions,
            actual=actual_data[self.target_column],
            signals=signals,
            equity_curve=equity_curve
        )
        
        return BacktestResult(
            model_name=model_name,
            strategy_name=strategy.__class__.__name__,
            metrics=metrics,
            trades=trades_df,
            equity_curve=equity_curve,
            parameters=strategy.parameters,
            test_period=(actual_data.index[0], actual_data.index[-1])
        )
    
    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Generate comprehensive performance visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Equity curve
        result.equity_curve.plot(ax=axes[0])
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True)
        
        # Plot 2: Drawdown
        drawdown = (result.equity_curve / result.equity_curve.cummax() - 1)
        drawdown.plot(ax=axes[1], color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True)
        
        # Plot 3: Daily returns distribution
        daily_returns = result.equity_curve.pct_change().dropna()
        sns.histplot(daily_returns, ax=axes[2], kde=True)
        axes[2].set_title('Daily Returns Distribution')
        axes[2].set_xlabel('Return')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_results(self, result: BacktestResult, save_dir: str):
        """Save backtest results to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(result.metrics, f, indent=4)
        
        # Save trades
        result.trades.to_csv(save_dir / 'trades.csv')
        
        # Save equity curve
        result.equity_curve.to_csv(save_dir / 'equity_curve.csv')
        
        # Save plots
        self.plot_results(result, str(save_dir / 'performance_plots.png'))
        
        logger.info(f"Results saved to {save_dir}")