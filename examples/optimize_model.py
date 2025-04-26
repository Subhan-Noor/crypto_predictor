"""
Example script demonstrating model optimization using backtesting results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.backtester import Backtester, SimpleThresholdStrategy
from evaluation.model_optimizer import ModelOptimizer
from data_fetchers.crypto_data_fetcher import fetch_historical_data

def main():
    # Load and prepare data
    data = fetch_historical_data(
        symbol='BTC/USD',
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Define features and target
    feature_columns = [
        'open', 'high', 'low', 'volume',
        'ma7', 'ma21', 'rsi'
    ]
    target_column = 'close'
    
    # Initialize backtester
    backtester = Backtester(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        initial_capital=10000.0,
        transaction_costs=0.001
    )
    
    # Initialize model optimizer
    optimizer = ModelOptimizer(
        backtester=backtester,
        model_type='rf',  # Random Forest
        feature_columns=feature_columns,
        target_column=target_column,
        optimization_metric='sharpe_ratio'
    )
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Define trading strategy parameters
    strategy_params = {
        'threshold': 0.01  # 1% price movement threshold
    }
    
    print("Starting grid search optimization...")
    grid_result = optimizer.grid_search(param_grid, strategy_params)
    
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_result.best_parameters}")
    print(f"Best metrics: {grid_result.best_metrics}")
    
    # Bayesian optimization with continuous parameters
    param_bounds = {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    
    print("\nStarting Bayesian optimization...")
    bayesian_result = optimizer.bayesian_optimization(
        param_bounds,
        n_iterations=30,
        strategy_params=strategy_params
    )
    
    print("\nBayesian Optimization Results:")
    print(f"Best parameters: {bayesian_result.best_parameters}")
    print(f"Best metrics: {bayesian_result.best_metrics}")
    
    # Compare results
    print("\nOptimization Comparison:")
    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
    
    print("\nMetric      | Grid Search | Bayesian Opt")
    print("-" * 45)
    for metric in metrics:
        grid_value = grid_result.best_metrics[metric]
        bayesian_value = bayesian_result.best_metrics[metric]
        print(f"{metric:11} | {grid_value:10.4f} | {bayesian_value:10.4f}")
    
    # Save best model parameters
    best_result = (
        grid_result if grid_result.best_metrics['sharpe_ratio'] > 
        bayesian_result.best_metrics['sharpe_ratio'] else bayesian_result
    )
    
    output_dir = Path('optimization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save parameters and metrics
    pd.DataFrame([best_result.best_parameters]).to_csv(
        output_dir / 'best_parameters.csv',
        index=False
    )
    pd.DataFrame([best_result.best_metrics]).to_csv(
        output_dir / 'best_metrics.csv',
        index=False
    )
    
    print(f"\nBest parameters and metrics saved to {output_dir}")

if __name__ == '__main__':
    main() 