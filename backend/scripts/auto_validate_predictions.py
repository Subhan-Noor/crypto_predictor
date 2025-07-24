"""
Automatic Prediction Validation Script

This script can be run automatically (e.g., daily via cron job) to validate
predictions that are older than their prediction horizon (7 days).

It validates predictions against actual price movements and updates the database
with the results (actual direction, correctness, accuracy).
"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger


class AutoPredictionValidator:
    """Automatically validates predictions that are ready for validation"""
    
    def __init__(self):
        """Initialize the auto validator"""
        self.currencies = ['BTC', 'ETH']
        self.prediction_horizon = 7  # days
        self.price_threshold = 0.5  # 0.5% threshold for UP/DOWN classification
        
        logger.info("Auto Prediction Validator initialized")
    
    async def get_unvalidated_predictions(self, currency: str) -> List[Dict[str, Any]]:
        """
        Get predictions that haven't been validated yet
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            List of predictions ready for validation
        """
        try:
            # Get all predictions for this currency
            all_predictions = await db_manager.get_predictions(currency, days=365, limit=1000)
            
            unvalidated_predictions = []
            current_date = datetime.now(timezone.utc)
            
            for prediction in all_predictions:
                # Parse prediction date
                pred_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00'))
                
                # Calculate target date (prediction_date + prediction_horizon)
                target_date = pred_date + timedelta(days=self.prediction_horizon)
                
                # Check if enough time has passed for validation
                if current_date > target_date:
                    # Check if this prediction hasn't been validated yet
                    # (look for actual_direction field - if missing, it's unvalidated)
                    if 'actual_direction' not in prediction or prediction['actual_direction'] is None:
                        prediction['target_date'] = target_date
                        unvalidated_predictions.append(prediction)
            
            logger.info(f"Found {len(unvalidated_predictions)} unvalidated predictions for {currency}")
            return unvalidated_predictions
            
        except Exception as e:
            logger.error(f"Error getting unvalidated predictions for {currency}: {e}")
            return []
    
    async def get_price_at_date(self, currency: str, target_date: datetime, 
                              window_days: int = 2) -> Optional[float]:
        """
        Get the closing price at or near a specific date
        
        Args:
            currency: 'BTC' or 'ETH'
            target_date: Date to get price for
            window_days: Days around target date to search
            
        Returns:
            Closing price or None if not found
        """
        try:
            # Get price data around the target date
            price_data = await db_manager.get_crypto_prices(currency, limit=100)
            
            if not price_data:
                return None
            
            # Convert to DataFrame for easier manipulation
            price_df = pd.DataFrame(price_data)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Find prices within the window
            start_date = target_date - timedelta(days=window_days)
            end_date = target_date + timedelta(days=window_days)
            
            relevant_prices = price_df[
                (price_df['date'] >= start_date) & 
                (price_df['date'] <= end_date)
            ].sort_values('date')
            
            if relevant_prices.empty:
                logger.warning(f"No price data found near {target_date.date()} for {currency}")
                return None
            
            # Find the closest date to our target
            relevant_prices['date_diff'] = abs((relevant_prices['date'] - target_date).dt.total_seconds())
            closest_price_row = relevant_prices.loc[relevant_prices['date_diff'].idxmin()]
            
            return float(closest_price_row['close'])
            
        except Exception as e:
            logger.error(f"Error getting price for {currency} at {target_date.date()}: {e}")
            return None
    
    async def validate_single_prediction(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate a single prediction against actual price movement
        
        Args:
            prediction: Prediction to validate
            
        Returns:
            Validation result with updated fields
        """
        try:
            currency = prediction['currency']
            pred_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00'))
            target_date = prediction['target_date']
            predicted_direction = prediction['predicted_direction']
            
            # Get prices at prediction date and target date
            pred_price = await self.get_price_at_date(currency, pred_date)
            target_price = await self.get_price_at_date(currency, target_date)
            
            if pred_price is None or target_price is None:
                logger.warning(f"Could not get prices for validation of prediction {prediction.get('id', 'unknown')}")
                return None
            
            # Calculate actual price movement
            price_change_pct = (target_price - pred_price) / pred_price * 100
            actual_direction = 'UP' if price_change_pct > self.price_threshold else 'DOWN'
            
            # Check if prediction was correct
            is_correct = (predicted_direction == actual_direction)
            
            validation_result = {
                'id': prediction['id'],
                'currency': currency,
                'prediction_date': pred_date.isoformat(),
                'target_date': target_date.isoformat(),
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'is_correct': is_correct,
                'price_change_pct': price_change_pct,
                'pred_price': pred_price,
                'target_price': target_price,
                'confidence_score': prediction.get('confidence_score', 0.0)
            }
            
            logger.info(f"Auto-validated {currency} {pred_date.date()}: "
                       f"Predicted={predicted_direction}, Actual={actual_direction}, "
                       f"Correct={'✅' if is_correct else '❌'} "
                       f"({price_change_pct:+.2f}%)")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating prediction {prediction.get('id', 'unknown')}: {e}")
            return None
    
    async def update_prediction_in_database(self, validation_result: Dict[str, Any]) -> bool:
        """
        Update a prediction in the database with validation results
        
        Args:
            validation_result: Validation result to save
            
        Returns:
            True if update was successful
        """
        try:
            prediction_id = validation_result['id']
            
            # Prepare update data
            update_data = {
                'actual_direction': validation_result['actual_direction'],
                'is_correct': validation_result['is_correct'],
                'price_change_pct': validation_result['price_change_pct'],
                'validated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Update the prediction in database
            success = await db_manager.update_prediction(prediction_id, update_data)
            
            if success:
                logger.info(f"Auto-updated prediction {prediction_id} in database")
                return True
            else:
                logger.error(f"Failed to auto-update prediction {prediction_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error auto-updating prediction in database: {e}")
            return False
    
    async def auto_validate_predictions(self) -> Dict[str, Any]:
        """
        Automatically validate all ready predictions
        
        Returns:
            Validation summary
        """
        summary = {
            'total_predictions': 0,
            'total_validated': 0,
            'total_correct': 0,
            'overall_accuracy': 0.0,
            'currencies': {}
        }
        
        for currency in self.currencies:
            logger.info(f"\n🤖 Auto-validating predictions for {currency}...")
            
            # Get unvalidated predictions
            unvalidated_predictions = await self.get_unvalidated_predictions(currency)
            
            currency_results = {
                'currency': currency,
                'total_predictions': len(unvalidated_predictions),
                'validated_predictions': 0,
                'correct_predictions': 0,
                'failed_validations': 0,
                'accuracy_percentage': 0.0
            }
            
            if not unvalidated_predictions:
                logger.info(f"No unvalidated predictions found for {currency}")
                summary['currencies'][currency] = currency_results
                continue
            
            # Validate each prediction
            for prediction in unvalidated_predictions:
                try:
                    validation_result = await self.validate_single_prediction(prediction)
                    
                    if validation_result:
                        # Update in database
                        update_success = await self.update_prediction_in_database(validation_result)
                        
                        if update_success:
                            currency_results['validated_predictions'] += 1
                            summary['total_validated'] += 1
                            
                            if validation_result['is_correct']:
                                currency_results['correct_predictions'] += 1
                                summary['total_correct'] += 1
                        else:
                            currency_results['failed_validations'] += 1
                    else:
                        currency_results['failed_validations'] += 1
                        
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error auto-validating prediction {prediction.get('id', 'unknown')}: {e}")
                    currency_results['failed_validations'] += 1
            
            # Calculate accuracy for this currency
            if currency_results['validated_predictions'] > 0:
                currency_results['accuracy_percentage'] = (currency_results['correct_predictions'] / currency_results['validated_predictions']) * 100
            
            summary['total_predictions'] += currency_results['total_predictions']
            summary['currencies'][currency] = currency_results
            
            logger.info(f"Auto-validation complete for {currency}:")
            logger.info(f"  Total predictions: {currency_results['total_predictions']}")
            logger.info(f"  Successfully validated: {currency_results['validated_predictions']}")
            logger.info(f"  Correct predictions: {currency_results['correct_predictions']}")
            logger.info(f"  Accuracy: {currency_results['accuracy_percentage']:.1f}%")
        
        # Calculate overall accuracy
        if summary['total_validated'] > 0:
            summary['overall_accuracy'] = (summary['total_correct'] / summary['total_validated']) * 100
        
        return summary


async def main():
    """Main function for automatic prediction validation"""
    
    print("🤖 Automatic Prediction Validation")
    print("=" * 50)
    
    validator = AutoPredictionValidator()
    
    try:
        print("\n📊 Auto-validating predictions that are ready...")
        
        # Auto-validate all predictions
        summary = await validator.auto_validate_predictions()
        
        print("\n📈 Auto-Validation Results:")
        print(f"  Total predictions found: {summary['total_predictions']}")
        print(f"  Successfully validated: {summary['total_validated']}")
        print(f"  Correct predictions: {summary['total_correct']}")
        
        if summary['total_validated'] > 0:
            print(f"  Overall accuracy: {summary['overall_accuracy']:.1f}%")
        
        # Show results by currency
        for currency, results in summary['currencies'].items():
            print(f"\n  {currency} Results:")
            print(f"    Predictions validated: {results['validated_predictions']}")
            print(f"    Correct predictions: {results['correct_predictions']}")
            print(f"    Accuracy: {results['accuracy_percentage']:.1f}%")
            print(f"    Failed validations: {results['failed_validations']}")
        
        if summary['total_validated'] > 0:
            print(f"\n🎉 Auto-validation complete! {summary['total_validated']} predictions validated.")
        else:
            print(f"\nℹ️ No predictions ready for validation at this time.")
        
    except KeyboardInterrupt:
        print("\n🛑 Auto-validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Auto-validation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 