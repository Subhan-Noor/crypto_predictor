"""
Deep learning models for cryptocurrency price prediction.
Currently implements LSTM networks for time series prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict
import logging
from .price_predictor import BaseMLModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel(BaseMLModel):
    """LSTM implementation for price prediction."""
    
    def __init__(self,
                 feature_columns: List[str],
                 target_column: str,
                 sequence_length: int = 60,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Number of time steps to look back
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        super().__init__(feature_columns, target_column)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self._build_model()
        
    def _build_model(self):
        """Build the LSTM network architecture."""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, len(self.feature_columns))
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.lstm_units)):
            self.model.add(LSTM(
                units=self.lstm_units[i],
                return_sequences=i < len(self.lstm_units) - 1
            ))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # Assuming target is first column
        return np.array(X), np.array(y)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for LSTM."""
        # Scale features
        X = df[self.feature_columns].values
        X = self.feature_scaler.fit_transform(X)
        
        # Scale target
        if self.target_column in df.columns:
            y = df[[self.target_column]].values
            y = self.target_scaler.fit_transform(y)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X)
            return X_seq, y_seq
        else:
            # For prediction, just return sequences without target
            X_seq, _ = self._create_sequences(X)
            return X_seq, None
    
    def train(self, df: pd.DataFrame) -> None:
        """Train the LSTM model."""
        logger.info("Training LSTM model...")
        X, y = self.prepare_features(df)
        
        # Split into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        return history
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using LSTM."""
        X, _ = self.prepare_features(df)
        predictions = self.model.predict(X)
        # Inverse transform predictions to original scale
        return self.target_scaler.inverse_transform(predictions)

def create_lstm_model(params: Dict) -> LSTMModel:
    """
    Create an LSTM model instance with the specified parameters.
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Configured LSTM model instance
    """
    return LSTMModel(**params) 