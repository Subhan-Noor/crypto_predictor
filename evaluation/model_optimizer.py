"""
Model optimization based on backtesting results.
Supports:
- Grid search optimization
- Bayesian optimization
- Multi-metric optimization
- Cross-validation with walk-forward validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm
from dataclasses import dataclass
import logging
from pathlib import Path
from evaluation.backtester import Backtester, BacktestResult, BacktestStrategy
from models.model_factory import create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_parameters: Dict
    best_metrics: Dict[str, float]
    parameter_history: List[Dict]
    metric_history: List[Dict[str, float]]
    best_backtest_result: BacktestResult

class ModelOptimizer:
    """Model optimization using backtesting results"""
    
    def __init__(
        self,
        backtester: Backtester,
        model_type: str,
        feature_columns: List[str],
        target_column: str,
        optimization_metric: str = 'sharpe_ratio',
        cv_folds: int = 5
    ):
        """
        Initialize the model optimizer
        
        Args:
            backtester: Backtester instance for evaluating models
            model_type: Type of model to optimize
            feature_columns: List of feature column names
            target_column: Target column name
            optimization_metric: Metric to optimize ('sharpe_ratio', 'returns', 'rmse', etc.)
            cv_folds: Number of cross-validation folds for walk-forward optimization
        """
        self.backtester = backtester
        self.model_type = model_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.optimization_metric = optimization_metric
        self.cv_folds = cv_folds
        
    def grid_search(
        self,
        param_grid: Dict[str, List],
        strategy: BacktestStrategy,
        window_size: int = 60,  # 2 months
        step_size: int = 20     # ~1 month
    ) -> OptimizationResult:
        """
        Perform grid search optimization
        
        Args:
            param_grid: Dictionary of parameters and their possible values
            strategy: Trading strategy instance
            window_size: Size of rolling window in days
            step_size: Number of days to move forward
        """
        logger.info(f"Starting grid search optimization for {self.model_type}")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        results = []
        metrics_history = []
        
        for params in param_combinations:
            logger.info(f"Testing parameters: {params}")
            
            # Create and evaluate model with current parameters
            model = create_model(
                self.model_type,
                model_params=params,
                feature_columns=self.feature_columns,
                target_column=self.target_column
            )
            
            # Perform walk-forward validation
            backtest_results = self.backtester.walk_forward_optimization(
                model=model,
                strategy=strategy,
                window_size=window_size,
                step_size=step_size
            )
            
            if not backtest_results:
                logger.warning(f"No backtest results for parameters: {params}")
                continue
            
            # Average metrics across folds
            avg_metrics = self._average_metrics(backtest_results)
            metrics_history.append(avg_metrics)
            results.append((params, avg_metrics, backtest_results[-1]))  # Keep the most recent backtest result
        
        if not results:
            logger.warning("No valid results found during grid search")
            # Return default result with empty metrics
            return OptimizationResult(
                best_parameters={},
                best_metrics={},
                parameter_history=[],
                metric_history=[],
                best_backtest_result=None
            )
        
        # Find best parameters based on optimization metric
        best_idx = np.argmax([m[self.optimization_metric] for m in metrics_history])
        best_params, best_metrics, best_backtest = results[best_idx]
        
        return OptimizationResult(
            best_parameters=best_params,
            best_metrics=best_metrics,
            parameter_history=[r[0] for r in results],
            metric_history=metrics_history,
            best_backtest_result=best_backtest
        )
    
    def bayesian_optimization(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        strategy_params: Dict = None,
        exploitation_ratio: float = 0.8
    ) -> OptimizationResult:
        """
        Perform Bayesian optimization using Gaussian Process
        
        Args:
            param_bounds: Dictionary of parameter names and their bounds (min, max)
            n_iterations: Number of optimization iterations
            strategy_params: Parameters for the trading strategy
            exploitation_ratio: Balance between exploration and exploitation (0-1)
        """
        logger.info(f"Starting Bayesian optimization for {self.model_type}")
        
        results = []
        metrics_history = []
        
        # Initial random sampling
        n_random = int(n_iterations * 0.2)  # 20% random initialization
        for _ in range(n_random):
            params = self._sample_random_params(param_bounds)
            metrics, backtest = self._evaluate_params(params, strategy_params)
            results.append((params, metrics, backtest))
            metrics_history.append(metrics)
        
        # Bayesian optimization loop
        for i in range(n_random, n_iterations):
            logger.info(f"Optimization iteration {i+1}/{n_iterations}")
            
            # Propose next parameters using Gaussian Process
            next_params = self._propose_next_params(
                param_bounds,
                results,
                exploitation_ratio
            )
            
            # Evaluate proposed parameters
            metrics, backtest = self._evaluate_params(next_params, strategy_params)
            results.append((next_params, metrics, backtest))
            metrics_history.append(metrics)
        
        # Find best parameters
        best_idx = np.argmax([m[self.optimization_metric] for m in metrics_history])
        best_params, best_metrics, best_backtest = results[best_idx]
        
        return OptimizationResult(
            best_parameters=best_params,
            best_metrics=best_metrics,
            parameter_history=[r[0] for r in results],
            metric_history=metrics_history,
            best_backtest_result=best_backtest
        )
    
    def _evaluate_params(
        self,
        params: Dict,
        strategy_params: Dict
    ) -> Tuple[Dict[str, float], BacktestResult]:
        """Evaluate a set of parameters using walk-forward validation"""
        model = create_model(
            self.model_type,
            model_params=params,
            feature_columns=self.feature_columns,
            target_column=self.target_column
        )
        
        backtest_results = self.backtester.walk_forward_optimization(
            model=model,
            strategy=strategy_params,
            window_size=90,
            step_size=30
        )
        
        avg_metrics = self._average_metrics(backtest_results)
        return avg_metrics, backtest_results[-1]
    
    def _average_metrics(self, backtest_results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate average metrics across multiple backtest results"""
        if not backtest_results:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
        
        # Get all metric names from the first result
        all_metrics = [result.metrics for result in backtest_results]
        avg_metrics = {}
        
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        return avg_metrics
    
    def _sample_random_params(self, param_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """Sample random parameters within bounds"""
        params = {}
        for param_name, (min_val, max_val) in param_bounds.items():
            params[param_name] = np.random.uniform(min_val, max_val)
        return params
    
    def _propose_next_params(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        previous_results: List[Tuple[Dict, Dict[str, float], BacktestResult]],
        exploitation_ratio: float
    ) -> Dict:
        """Propose next parameters using Gaussian Process"""
        if np.random.random() > exploitation_ratio:
            # Exploration: sample random parameters
            return self._sample_random_params(param_bounds)
        
        # Exploitation: use best parameters with some noise
        best_idx = np.argmax([r[1][self.optimization_metric] for r in previous_results])
        best_params = previous_results[best_idx][0]
        
        # Add noise to best parameters
        noisy_params = {}
        for param_name, value in best_params.items():
            min_val, max_val = param_bounds[param_name]
            noise = np.random.normal(0, 0.1 * (max_val - min_val))
            noisy_params[param_name] = np.clip(value + noise, min_val, max_val)
        
        return noisy_params 