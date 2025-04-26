"""
Performance visualization and metrics calculation for cryptocurrency price predictions.
Includes functions for:
- Performance metrics calculation
- Prediction vs actual charts
- Model comparison visualizations
- Feature importance plots
- Error distribution analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Union, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class PerformanceVisualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings"""
        plt.style.use('default')  # Use default style instead of seaborn
        self.colors = px.colors.qualitative.Set3
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics for the predictions.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary containing various performance metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def plot_prediction_vs_actual(self, 
                                dates: np.ndarray,
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                coin_name: str = "Cryptocurrency",
                                save_path: Optional[str] = None) -> None:
        """
        Create an interactive plot comparing predicted vs actual prices.
        
        Args:
            dates: Array of dates
            y_true: Array of actual values
            y_pred: Array of predicted values
            coin_name: Name of the cryptocurrency
            save_path: Optional path to save the plot
        """
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_true,
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_pred,
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{coin_name} Price Prediction vs Actual',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_error_distribution(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of prediction errors.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            save_path: Optional path to save the plot
        """
        errors = y_true - y_pred
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name='Error Distribution',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Prediction Error Distribution',
            xaxis_title='Error',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_feature_importance(self,
                              feature_names: List[str],
                              importance_scores: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Create a bar plot of feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            save_path: Optional path to save the plot
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        fig = go.Figure()
        
        # Add bar plot
        fig.add_trace(go.Bar(
            x=importance_scores[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation='h'
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_model_comparison(self,
                            model_metrics: Dict[str, Dict[str, float]],
                            metric_name: str = 'rmse',
                            save_path: Optional[str] = None) -> None:
        """
        Create a bar plot comparing different models' performance.
        
        Args:
            model_metrics: Dictionary of model names and their metrics
            metric_name: Name of the metric to compare
            save_path: Optional path to save the plot
        """
        models = list(model_metrics.keys())
        metric_values = [metrics[metric_name] for metrics in model_metrics.values()]
        
        fig = go.Figure()
        
        # Add bar plot
        fig.add_trace(go.Bar(
            x=models,
            y=metric_values,
            text=np.round(metric_values, 4),
            textposition='auto',
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Model Comparison - {metric_name.upper()}',
            xaxis_title='Model',
            yaxis_title=metric_name.upper(),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_performance_dashboard(self,
                                   dates: np.ndarray,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   feature_names: List[str],
                                   importance_scores: np.ndarray,
                                   coin_name: str = "Cryptocurrency",
                                   save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive performance dashboard with multiple plots.
        
        Args:
            dates: Array of dates
            y_true: Array of actual values
            y_pred: Array of predicted values
            feature_names: List of feature names
            importance_scores: Array of importance scores
            coin_name: Name of the cryptocurrency
            save_path: Optional path to save the dashboard
        """
        # Create individual plots
        pred_vs_actual = self.plot_prediction_vs_actual(dates, y_true, y_pred, coin_name)
        error_dist = self.plot_error_distribution(y_true, y_pred)
        feature_imp = self.plot_feature_importance(feature_names, importance_scores)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create metrics table
        metrics_table = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[list(metrics.keys()), list(metrics.values())],
                      fill_color='lavender',
                      align='left'))
        ])
        
        # Save individual plots if save_path is provided
        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            pred_vs_actual.write_html(f"{base_path}_predictions.html")
            error_dist.write_html(f"{base_path}_errors.html")
            feature_imp.write_html(f"{base_path}_importance.html")
            metrics_table.write_html(f"{base_path}_metrics.html")