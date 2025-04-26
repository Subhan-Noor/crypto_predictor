"""
Evaluation package for cryptocurrency price prediction models.
Includes backtesting, visualization, and model optimization components.
"""

from evaluation.backtester import Backtester, SimpleThresholdStrategy, BacktestResult
from evaluation.visualizer import PerformanceVisualizer
from evaluation.model_optimizer import ModelOptimizer

__all__ = [
    'Backtester',
    'SimpleThresholdStrategy',
    'BacktestResult',
    'PerformanceVisualizer',
    'ModelOptimizer'
] 