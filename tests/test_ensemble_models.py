"""
Test suite for ensemble model implementations.
"""

import unittest
import pandas as pd
import numpy as np
from models.ensemble_model import WeightedEnsemble, StackingEnsemble, create_ensemble_model
from models.model_factory import create_model

class TestEnsembleModels(unittest.TestCase):
    """Test cases for ensemble model implementations."""
    
    def setUp(self):
        """Set up test data and model configurations."""
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        self.df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'target': 2 * np.random.randn(n_samples) + 1
        })
        
        self.feature_columns = ['feature1', 'feature2']
        self.target_column = 'target'
        
        # Base model configurations
        self.base_models_config = [
            {
                'type': 'linear',
                'params': {'fit_intercept': True}
            },
            {
                'type': 'random_forest',
                'params': {'n_estimators': 100}
            }
        ]
        
        # Meta-model configuration
        self.meta_model_config = {
            'type': 'linear',
            'params': {'fit_intercept': True}
        }
        
    def test_weighted_ensemble_initialization(self):
        """Test weighted ensemble initialization with different weight configurations."""
        # Test with default weights
        ensemble = WeightedEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config
        )
        self.assertEqual(len(ensemble.weights), len(self.base_models_config))
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)
        
        # Test with custom weights
        custom_weights = [0.7, 0.3]
        ensemble = WeightedEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config,
            weights=custom_weights
        )
        self.assertEqual(ensemble.weights, custom_weights)
        
        # Test invalid weights
        with self.assertRaises(ValueError):
            WeightedEnsemble(
                self.feature_columns,
                self.target_column,
                self.base_models_config,
                weights=[0.7, 0.7]  # Sum > 1
            )
            
    def test_weighted_ensemble_training_and_prediction(self):
        """Test weighted ensemble training and prediction workflow."""
        ensemble = WeightedEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config
        )
        
        # Train the ensemble
        ensemble.train(self.df)
        
        # Make predictions
        predictions = ensemble.predict(self.df)
        
        self.assertEqual(len(predictions), len(self.df))
        self.assertTrue(isinstance(predictions, np.ndarray))
        
    def test_weighted_ensemble_weight_optimization(self):
        """Test weight optimization in weighted ensemble."""
        ensemble = WeightedEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config
        )
        
        # Train and optimize weights
        ensemble.train(self.df)
        original_weights = ensemble.weights.copy()
        
        ensemble.optimize_weights(self.df)
        
        # Check if weights changed and still sum to 1
        self.assertNotEqual(ensemble.weights, original_weights)
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)
        
    def test_stacking_ensemble_initialization(self):
        """Test stacking ensemble initialization."""
        ensemble = StackingEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config,
            self.meta_model_config
        )
        
        self.assertEqual(len(ensemble.base_models), len(self.base_models_config))
        self.assertIsNotNone(ensemble.meta_model)
        
    def test_stacking_ensemble_training_and_prediction(self):
        """Test stacking ensemble training and prediction workflow."""
        ensemble = StackingEnsemble(
            self.feature_columns,
            self.target_column,
            self.base_models_config,
            self.meta_model_config
        )
        
        # Train the ensemble
        ensemble.train(self.df)
        
        # Make predictions
        predictions = ensemble.predict(self.df)
        
        self.assertEqual(len(predictions), len(self.df))
        self.assertTrue(isinstance(predictions, np.ndarray))
        
    def test_ensemble_factory(self):
        """Test ensemble model factory function."""
        # Test weighted ensemble creation
        weighted_config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'base_models_config': self.base_models_config
        }
        
        weighted = create_ensemble_model('weighted', weighted_config)
        self.assertIsInstance(weighted, WeightedEnsemble)
        
        # Test stacking ensemble creation
        stacking_config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'base_models_config': self.base_models_config,
            'meta_model_config': self.meta_model_config
        }
        
        stacking = create_ensemble_model('stacking', stacking_config)
        self.assertIsInstance(stacking, StackingEnsemble)
        
        # Test invalid ensemble type
        with self.assertRaises(ValueError):
            create_ensemble_model('invalid_type', {})
            
if __name__ == '__main__':
    unittest.main() 