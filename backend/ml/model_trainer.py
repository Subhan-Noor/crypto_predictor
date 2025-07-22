"""
Model Training for Crypto Price Prediction

This module handles:
- Baseline models (Logistic Regression, Random Forest)
- Advanced models (LSTM Neural Network)
- Model evaluation and comparison
- Model serialization and storage
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logger import logger


class CryptoDataset(Dataset):
    """PyTorch Dataset for crypto prediction data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM Neural Network for crypto price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 2)  # Binary classification
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for LSTM (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class CryptoModelTrainer:
    """Trains and evaluates crypto prediction models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def prepare_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training
        
        Args:
            X_train: Training features
            X_test: Test features  
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Scaled features and targets
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scalers['feature_scaler'] = scaler
        
        logger.info(f"Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with model and evaluation results
        """
        logger.info("Training Logistic Regression model...")
        
        # Initialize and train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        results = self._evaluate_model("Logistic Regression", y_train, y_pred_train, 
                                     y_test, y_pred_test, y_proba_test)
        
        # Store model
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return results
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with model and evaluation results
        """
        logger.info("Training Random Forest model...")
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        results = self._evaluate_model("Random Forest", y_train, y_pred_train,
                                     y_test, y_pred_test, y_proba_test)
        
        # Store model
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return results
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM Neural Network model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with model and evaluation results
        """
        logger.info("Training LSTM Neural Network model...")
        
        # Create datasets and data loaders
        train_dataset = CryptoDataset(X_train, y_train)
        test_dataset = CryptoDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = LSTMModel(input_size)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training predictions
            train_outputs = []
            for batch_X, batch_y in DataLoader(train_dataset, batch_size=batch_size):
                outputs = model(batch_X)
                train_outputs.append(outputs)
            train_outputs = torch.cat(train_outputs, dim=0)
            y_pred_train = torch.argmax(train_outputs, dim=1).numpy()
            
            # Test predictions
            test_outputs = []
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                test_outputs.append(outputs)
            test_outputs = torch.cat(test_outputs, dim=0)
            y_pred_test = torch.argmax(test_outputs, dim=1).numpy()
            y_proba_test = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
        
        # Evaluate model
        results = self._evaluate_model("LSTM", y_train, y_pred_train,
                                     y_test, y_pred_test, y_proba_test)
        
        # Store model
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        return results
    
    def _evaluate_model(self, model_name: str, y_train: np.ndarray, y_pred_train: np.ndarray,
                       y_test: np.ndarray, y_pred_test: np.ndarray, 
                       y_proba_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            model_name: Name of the model
            y_train: True training labels
            y_pred_train: Predicted training labels
            y_test: True test labels
            y_pred_test: Predicted test labels
            y_proba_test: Prediction probabilities for test set
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'y_pred_test': y_pred_test.tolist(),
            'y_proba_test': y_proba_test.tolist()
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test F1 Score: {test_f1:.4f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train all models and compare results
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Starting training of all models...")
        
        # Prepare data
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(
            X_train, X_test, y_train, y_test
        )
        
        # Train baseline models
        self.train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
        self.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Train advanced model
        self.train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Find best model
        best_model = self._find_best_model()
        
        logger.info(f"Training complete. Best model: {best_model}")
        
        return {
            'results': self.results,
            'best_model': best_model,
            'summary': self._create_summary()
        }
    
    def _find_best_model(self) -> str:
        """Find the best performing model based on test F1 score"""
        best_score = 0
        best_model = None
        
        for model_name, results in self.results.items():
            f1_score = results['test_f1']
            if f1_score > best_score:
                best_score = f1_score
                best_model = model_name
        
        return best_model
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary of all model performances"""
        summary = {
            'models_trained': len(self.results),
            'best_model': self._find_best_model(),
            'comparison': {}
        }
        
        for model_name, results in self.results.items():
            summary['comparison'][model_name] = {
                'test_accuracy': results['test_accuracy'],
                'test_f1': results['test_f1'],
                'test_precision': results['test_precision'],
                'test_recall': results['test_recall']
            }
        
        return summary
    
    def save_models(self, currency: str, feature_names: List[str]):
        """
        Save trained models and scalers to disk
        
        Args:
            currency: Currency code (BTC/ETH)
            feature_names: List of feature column names
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            # Create model metadata
            metadata = {
                'currency': currency,
                'model_name': model_name,
                'timestamp': timestamp,
                'feature_names': feature_names,
                'results': self.results.get(model_name, {}),
                'scaler': self.scalers.get('feature_scaler')
            }
            
            # Save model file
            model_file = os.path.join(self.models_dir, f"{currency}_{model_name}_{timestamp}.pkl")
            
            if model_name == 'lstm':
                # Save PyTorch model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata
                }, model_file.replace('.pkl', '.pth'))
            else:
                # Save sklearn model
                joblib.dump({
                    'model': model,
                    'metadata': metadata
                }, model_file)
            
            logger.info(f"Saved {model_name} model to {model_file}")
    
    def load_model(self, model_file: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model from disk
        
        Args:
            model_file: Path to model file
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_file.endswith('.pth'):
            # Load PyTorch model
            checkpoint = torch.load(model_file)
            metadata = checkpoint['metadata']
            
            # Recreate model architecture
            input_size = len(metadata['feature_names'])
            model = LSTMModel(input_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
        else:
            # Load sklearn model
            saved_data = joblib.load(model_file)
            model = saved_data['model']
            metadata = saved_data['metadata']
        
        return model, metadata


# Utility function for quick model training
def train_models(X_train: np.ndarray, X_test: np.ndarray, 
                y_train: np.ndarray, y_test: np.ndarray,
                currency: str, feature_names: List[str]) -> Dict[str, Any]:
    """
    Quick utility function to train all models
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        currency: Currency code
        feature_names: List of feature names
        
    Returns:
        Training results dictionary
    """
    trainer = CryptoModelTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.save_models(currency, feature_names)
    return results 