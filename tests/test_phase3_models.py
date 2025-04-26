"""
Comprehensive test script for Phase 3 model implementations.
Tests individual models and ensemble approaches.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model
from data_fetchers.crypto_data_fetcher import fetch_crypto_data
from processors.data_processor import prepare_data_for_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate and return evaluation metrics."""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    logger.info(f"\nMetrics for {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    return metrics

def plot_predictions(test_df: pd.DataFrame, predictions_dict: dict, title: str):
    """Plot actual vs predicted values for multiple models."""
    plt.figure(figsize=(15, 7))
    
    # Plot actual values
    plt.plot(test_df.index, test_df['close'], label='Actual', color='black', alpha=0.7)
    
    # Plot predictions for each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        plt.plot(test_df.index, predictions, label=f'{model_name} Predictions', 
                color=color, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join('tests', 'results', f'{title.lower().replace(" ", "_")}.png'))
    plt.close()

def main():
    """Main test function."""
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join('tests', 'results'), exist_ok=True)
    
    # 1. Fetch and prepare data
    logger.info("Fetching and preparing data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use 1 year of data
    
    df = fetch_crypto_data('BTC', start_date, end_date)
    df = prepare_data_for_training(df)
    
    # Split data into train, validation, and test sets
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Define feature columns
    feature_columns = ['open', 'high', 'low', 'volume', 'ma7', 'ma21', 'rsi']
    target_column = 'close'
    
    # 2. Test individual models
    logger.info("\nTesting individual models...")
    
    # Configure individual models
    model_configs = {
        'ARIMA': {
            'type': 'arima',
            'params': {
                'feature_columns': feature_columns,
                'target_column': target_column,
                'order': (5,1,2)
            }
        },
        'Prophet': {
            'type': 'prophet',
            'params': {
                'feature_columns': feature_columns,
                'target_column': target_column,
                'seasonality_mode': 'multiplicative'
            }
        },
        'LSTM': {
            'type': 'lstm',
            'params': {
                'feature_columns': feature_columns,
                'target_column': target_column,
                'sequence_length': 60,
                'lstm_units': [128, 64],
                'dropout_rate': 0.2
            }
        }
    }
    
    # Train and evaluate individual models
    predictions_dict = {}
    for model_name, config in model_configs.items():
        logger.info(f"\nTraining {model_name} model...")
        model = create_model(config['type'], config['params'])
        model.train(train_df)
        predictions = model.predict(test_df)
        predictions_dict[model_name] = predictions
        evaluate_model(test_df[target_column].values, predictions, model_name)
    
    # 3. Test ensemble approaches
    logger.info("\nTesting ensemble approaches...")
    
    # Configure weighted ensemble
    weighted_config = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'base_models_config': [
            {
                'type': 'arima',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'order': (5,1,2)
                }
            },
            {
                'type': 'prophet',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'seasonality_mode': 'multiplicative'
                }
            },
            {
                'type': 'lstm',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'sequence_length': 60,
                    'lstm_units': [128, 64],
                    'dropout_rate': 0.2
                }
            }
        ]
    }
    
    # Train and evaluate weighted ensemble
    weighted_ensemble = create_model('weighted_ensemble', weighted_config)
    weighted_ensemble.train(train_df)
    weighted_ensemble.optimize_weights(val_df)
    weighted_predictions = weighted_ensemble.predict(test_df)
    predictions_dict['Weighted Ensemble'] = weighted_predictions
    evaluate_model(test_df[target_column].values, weighted_predictions, 'Weighted Ensemble')
    
    # Configure stacking ensemble
    stacking_config = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'base_models_config': [
            {
                'type': 'arima',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'order': (5,1,2)
                }
            },
            {
                'type': 'prophet',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'seasonality_mode': 'multiplicative'
                }
            },
            {
                'type': 'lstm',
                'params': {
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'sequence_length': 60,
                    'lstm_units': [128, 64],
                    'dropout_rate': 0.2
                }
            }
        ],
        'meta_model_config': {
            'type': 'xgboost',
            'params': {
                'feature_columns': ['model_1_pred', 'model_2_pred', 'model_3_pred'],
                'target_column': target_column,
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1
            }
        }
    }
    
    # Train and evaluate stacking ensemble
    stacking_ensemble = create_model('stacking_ensemble', stacking_config)
    stacking_ensemble.train(train_df)
    stacking_predictions = stacking_ensemble.predict(test_df)
    predictions_dict['Stacking Ensemble'] = stacking_predictions
    evaluate_model(test_df[target_column].values, stacking_predictions, 'Stacking Ensemble')
    
    # 4. Visualize results
    logger.info("\nGenerating visualization...")
    plot_predictions(test_df, predictions_dict, 'Cryptocurrency Price Predictions Comparison')
    
    logger.info("\nTest completed successfully! Check the 'tests/results' directory for visualization.")

if __name__ == "__main__":
    main() 