"""
Historic Prediction Validation Script

This script validates historic predictions against actual price movements
and updates the database with the results (actual direction, correctness).

For predictions that are older than their prediction horizon (7 days),
we can now check if they were correct by comparing predicted vs actual price movement.
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


class PredictionValidator:
    """Validates historic predictions against actual price movements"""
    
    def __init__(self):
        """Initialize the prediction validator"""
        self.currencies = ['BTC', 'ETH']
        self.prediction_horizon = 7  # days
        self.price_threshold = 0.5  # 0.5% threshold for UP/DOWN classification
        
        logger.info("Prediction Validator initialized")
    
    async def get_validatable_predictions(self, currency: str) -> List[Dict[str, Any]]:
        """
        Get predictions that can be validated (older than prediction horizon)
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            List of predictions that can be validated
        """
        try:
            # Get all predictions for this currency
            all_predictions = await db_manager.get_predictions(currency, days=365, limit=1000)
            
            validatable_predictions = []
            current_date = datetime.now(timezone.utc)
            
            for prediction in all_predictions:
                # Parse prediction date
                pred_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00'))
                
                # Calculate target date (prediction_date + prediction_horizon)
                target_date = pred_date + timedelta(days=self.prediction_horizon)
                
                # Check if enough time has passed for validation
                if current_date > target_date:
                    # Check if this prediction hasn't been validated yet
                    # (we can re-validate, but prioritize unvalidated ones)
                    prediction['target_date'] = target_date
                    validatable_predictions.append(prediction)
            
            logger.info(f"Found {len(validatable_predictions)} predictions ready for validation for {currency}")
            return validatable_predictions
            
        except Exception as e:
            logger.error(f"Error getting validatable predictions for {currency}: {e}")
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
            
            logger.info(f"Validated {currency} {pred_date.date()}: "
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
                logger.info(f"Updated prediction {prediction_id} in database")
                return True
            else:
                logger.error(f"Failed to update prediction {prediction_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating prediction in database: {e}")
            return False
    
    async def validate_predictions_for_currency(self, currency: str) -> Dict[str, Any]:
        """
        Validate all ready predictions for a currency
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Validation summary for the currency
        """
        results = {
            'currency': currency,
            'total_predictions': 0,
            'validated_predictions': 0,
            'correct_predictions': 0,
            'failed_validations': 0,
            'accuracy_percentage': 0.0,
            'validation_details': []
        }
        
        try:
            # Get predictions that can be validated
            validatable_predictions = await self.get_validatable_predictions(currency)
            results['total_predictions'] = len(validatable_predictions)
            
            if not validatable_predictions:
                logger.info(f"No predictions ready for validation for {currency}")
                return results
            
            # Validate each prediction
            for prediction in validatable_predictions:
                try:
                    validation_result = await self.validate_single_prediction(prediction)
                    
                    if validation_result:
                        # Update in database
                        update_success = await self.update_prediction_in_database(validation_result)
                        
                        if update_success:
                            results['validated_predictions'] += 1
                            
                            if validation_result['is_correct']:
                                results['correct_predictions'] += 1
                            
                            results['validation_details'].append(validation_result)
                        else:
                            results['failed_validations'] += 1
                    else:
                        results['failed_validations'] += 1
                        
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error validating prediction {prediction.get('id', 'unknown')}: {e}")
                    results['failed_validations'] += 1
            
            # Calculate accuracy
            if results['validated_predictions'] > 0:
                results['accuracy_percentage'] = (results['correct_predictions'] / results['validated_predictions']) * 100
            
            logger.info(f"Validation complete for {currency}:")
            logger.info(f"  Total predictions: {results['total_predictions']}")
            logger.info(f"  Successfully validated: {results['validated_predictions']}")
            logger.info(f"  Correct predictions: {results['correct_predictions']}")
            logger.info(f"  Accuracy: {results['accuracy_percentage']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error validating predictions for {currency}: {e}")
            results['error'] = str(e)
        
        return results
    
    async def validate_all_predictions(self) -> Dict[str, Any]:
        """
        Validate predictions for all currencies
        
        Returns:
            Overall validation summary
        """
        summary = {
            'total_predictions': 0,
            'total_validated': 0,
            'total_correct': 0,
            'overall_accuracy': 0.0,
            'currencies': {}
        }
        
        for currency in self.currencies:
            logger.info(f"\n📈 Validating predictions for {currency}...")
            
            currency_results = await self.validate_predictions_for_currency(currency)
            summary['currencies'][currency] = currency_results
            
            # Add to totals
            summary['total_predictions'] += currency_results['total_predictions']
            summary['total_validated'] += currency_results['validated_predictions']
            summary['total_correct'] += currency_results['correct_predictions']
        
        # Calculate overall accuracy
        if summary['total_validated'] > 0:
            summary['overall_accuracy'] = (summary['total_correct'] / summary['total_validated']) * 100
        
        return summary


async def main():
    """Main function to validate historic predictions"""
    
    print("🔍 Historic Prediction Validation")
    print("=" * 50)
    
    validator = PredictionValidator()
    
    try:
        print("\n📊 Validating historic predictions against actual price movements...")
        
        # Validate all predictions
        summary = await validator.validate_all_predictions()
        
        print("\n📈 Validation Results:")
        print(f"  Total predictions found: {summary['total_predictions']}")
        print(f"  Successfully validated: {summary['total_validated']}")
        print(f"  Correct predictions: {summary['total_correct']}")
        
        if summary['total_validated'] > 0:
            print(f"  Overall accuracy: {summary['overall_accuracy']:.1f}%")
        
        # Show results by currency
        for currency, results in summary['currencies'].items():
            if 'error' not in results:
                print(f"\n  {currency} Results:")
                print(f"    Predictions validated: {results['validated_predictions']}")
                print(f"    Correct predictions: {results['correct_predictions']}")
                print(f"    Accuracy: {results['accuracy_percentage']:.1f}%")
                print(f"    Failed validations: {results['failed_validations']}")
            else:
                print(f"\n  {currency}: Error - {results['error']}")
        
        print(f"\n🎉 Validation complete!")
        print("Refresh your predictions dashboard to see the updated results!")
        
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 