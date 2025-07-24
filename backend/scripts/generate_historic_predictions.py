"""
Historic Prediction Generation Script

This script generates ML predictions for historical dates to build up a prediction database.
This allows us to:
- Build prediction history for analysis
- Test prediction accuracy against actual price movements
- Analyze model performance over time

The script generates predictions for past dates and validates them against actual outcomes.
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
from ml.prediction_pipeline import CryptoPredictionPipeline
from ml.data_preprocessor import CryptoDataPreprocessor  
from ml.feature_engineering import CryptoFeatureEngineer


class HistoricPredictionGenerator:
    """Generates historic ML predictions and validates them against actual outcomes"""
    
    def __init__(self):
        """Initialize the historic prediction generator"""
        self.pipeline = CryptoPredictionPipeline()
        self.preprocessor = CryptoDataPreprocessor()
        self.feature_engineer = CryptoFeatureEngineer()
        self.currencies = ['BTC', 'ETH']
        self.prediction_horizon = 7  # days
        
        logger.info("Historic Prediction Generator initialized")
    
    async def get_available_price_data_range(self, currency: str) -> Dict[str, Any]:
        """
        Get the range of available price data
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Dictionary with data range info
        """
        try:
            price_data = await db_manager.get_crypto_prices(currency, limit=1000)
            
            if not price_data:
                return {'available': False}
            
            # Convert to DataFrame for analysis
            price_df = pd.DataFrame(price_data)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            return {
                'available': True,
                'earliest_date': price_df['date'].min(),
                'latest_date': price_df['date'].max(),
                'total_records': len(price_df),
                'date_range_days': (price_df['date'].max() - price_df['date'].min()).days
            }
            
        except Exception as e:
            logger.error(f"Error checking price data range for {currency}: {e}")
            return {'available': False, 'error': str(e)}
    
    async def check_prediction_exists(self, currency: str, prediction_date: datetime) -> bool:
        """
        Check if a prediction already exists for a given date
        
        Args:
            currency: 'BTC' or 'ETH'
            prediction_date: Date to check
            
        Returns:
            True if prediction exists
        """
        try:
            existing_predictions = await db_manager.get_predictions(currency, days=1)
            
            for pred in existing_predictions:
                pred_date = datetime.fromisoformat(pred['prediction_date'].replace('Z', '+00:00')).date()
                if pred_date == prediction_date.date():
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking existing predictions: {e}")
            return False
    
    async def generate_prediction_for_date(self, currency: str, prediction_date: datetime, 
                                         model_type: str = "logistic_regression") -> Optional[Dict[str, Any]]:
        """
        Generate a prediction for a specific historical date
        
        Args:
            currency: 'BTC' or 'ETH'
            prediction_date: Historical date to generate prediction for
            model_type: Type of model to use
            
        Returns:
            Prediction result or None if failed
        """
        try:
            logger.info(f"🔮 Generating historic prediction for {currency} on {prediction_date.date()}")
            
            # Check if prediction already exists
            if await self.check_prediction_exists(currency, prediction_date):
                logger.info(f"Prediction already exists for {currency} on {prediction_date.date()}, skipping")
                return None
            
            # Generate prediction using the ML pipeline with specific date
            prediction = await self.pipeline.make_prediction(currency, model_type, prediction_date)
            
            if prediction:
                # Save to database
                prediction_id = await self.pipeline.save_prediction(prediction)
                prediction['id'] = prediction_id
                
                logger.info(f"✅ Historic prediction generated for {currency} on {prediction_date.date()}")
                logger.info(f"   Direction: {prediction.get('predicted_direction', 'N/A')}")
                logger.info(f"   Confidence: {prediction.get('confidence_score', 0):.4f}")
                logger.info(f"   Database ID: {prediction_id}")
                
                return prediction
            else:
                logger.warning(f"❌ Failed to generate historic prediction for {currency} on {prediction_date.date()}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating historic prediction for {currency} on {prediction_date.date()}: {e}")
            return None
    
    async def validate_prediction_accuracy(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate a historic prediction against actual price movement
        
        Args:
            prediction: Prediction to validate
            
        Returns:
            Validation result with accuracy info
        """
        try:
            currency = prediction['currency']
            pred_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00')) if isinstance(prediction['prediction_date'], str) else prediction['prediction_date']
            
            # Calculate target date (prediction_horizon days after prediction)
            target_date = pred_date + timedelta(days=self.prediction_horizon)
            
            # Get price data around prediction and target dates
            start_date = pred_date - timedelta(days=1)
            end_date = target_date + timedelta(days=1)
            
            price_data = await db_manager.get_crypto_prices(currency, limit=100)
            price_df = pd.DataFrame(price_data)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Filter to relevant date range
            relevant_prices = price_df[
                (price_df['date'] >= start_date) & 
                (price_df['date'] <= end_date)
            ].sort_values('date')
            
            if len(relevant_prices) < 2:
                logger.warning(f"Insufficient price data for validation of {prediction.get('id', 'unknown')}")
                return None
            
            # Find closest prices to prediction and target dates
            pred_price_row = relevant_prices.iloc[0]  # Earliest available price
            target_price_row = relevant_prices.iloc[-1]  # Latest available price
            
            pred_price = float(pred_price_row['close'])
            target_price = float(target_price_row['close'])
            
            # Calculate actual direction
            price_change_pct = (target_price - pred_price) / pred_price * 100
            actual_direction = 'UP' if price_change_pct > 0.5 else 'DOWN'  # 0.5% threshold
            
            # Check if prediction was correct
            predicted_direction = prediction['predicted_direction']
            is_correct = (predicted_direction == actual_direction)
            
            validation_result = {
                'prediction_id': prediction.get('id'),
                'currency': currency,
                'prediction_date': pred_date.isoformat(),
                'target_date': target_date.isoformat(),
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'is_correct': is_correct,
                'price_change_pct': price_change_pct,
                'confidence_score': prediction.get('confidence_score', 0),
                'pred_price': pred_price,
                'target_price': target_price
            }
            
            logger.info(f"Validation: {currency} {pred_date.date()} - "
                       f"Predicted: {predicted_direction}, Actual: {actual_direction}, "
                       f"Correct: {'✅' if is_correct else '❌'} "
                       f"({price_change_pct:+.2f}%)")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return None
    
    async def generate_historic_predictions_for_currency(self, currency: str, 
                                                       start_date: datetime, 
                                                       end_date: datetime,
                                                       model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Generate historic predictions for a currency over a date range
        
        Args:
            currency: 'BTC' or 'ETH'
            start_date: Start date for predictions
            end_date: End date for predictions  
            model_type: Model type to use
            
        Returns:
            Summary of generation results
        """
        results = {
            'currency': currency,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_dates': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'skipped_predictions': 0,
            'predictions': []
        }
        
        # Generate predictions every 2-3 days to avoid too many predictions
        current_date = start_date
        prediction_interval = 2  # days between predictions
        
        while current_date <= end_date:
            try:
                results['total_dates'] += 1
                
                prediction = await self.generate_prediction_for_date(currency, current_date, model_type)
                
                if prediction:
                    results['successful_predictions'] += 1
                    results['predictions'].append(prediction)
                    
                    # Add a small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                elif prediction is None:
                    results['skipped_predictions'] += 1
                else:
                    results['failed_predictions'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {currency} on {current_date.date()}: {e}")
                results['failed_predictions'] += 1
            
            # Move to next prediction date
            current_date += timedelta(days=prediction_interval)
        
        logger.info(f"Historic prediction generation complete for {currency}")
        logger.info(f"  Successful: {results['successful_predictions']}")
        logger.info(f"  Failed: {results['failed_predictions']}")
        logger.info(f"  Skipped: {results['skipped_predictions']}")
        
        return results
    
    async def validate_all_historic_predictions(self) -> Dict[str, Any]:
        """
        Validate all existing historic predictions against actual outcomes
        
        Returns:
            Validation summary
        """
        validation_summary = {
            'total_predictions': 0,
            'validated_predictions': 0,
            'correct_predictions': 0,
            'currencies': {}
        }
        
        for currency in self.currencies:
            try:
                # Get all predictions for this currency
                predictions = await db_manager.get_predictions(currency, days=365, limit=500)  # Last year
                
                currency_results = {
                    'total_predictions': len(predictions),
                    'validated_predictions': 0,
                    'correct_predictions': 0,
                    'validation_details': []
                }
                
                for prediction in predictions:
                    validation_result = await self.validate_prediction_accuracy(prediction)
                    
                    if validation_result:
                        currency_results['validated_predictions'] += 1
                        validation_summary['validated_predictions'] += 1
                        
                        if validation_result['is_correct']:
                            currency_results['correct_predictions'] += 1
                            validation_summary['correct_predictions'] += 1
                        
                        currency_results['validation_details'].append(validation_result)
                
                validation_summary['total_predictions'] += currency_results['total_predictions']
                validation_summary['currencies'][currency] = currency_results
                
                # Calculate accuracy for this currency
                if currency_results['validated_predictions'] > 0:
                    accuracy = (currency_results['correct_predictions'] / currency_results['validated_predictions']) * 100
                    logger.info(f"{currency} accuracy: {accuracy:.1f}% ({currency_results['correct_predictions']}/{currency_results['validated_predictions']})")
                
            except Exception as e:
                logger.error(f"Error validating predictions for {currency}: {e}")
                validation_summary['currencies'][currency] = {'error': str(e)}
        
        # Calculate overall accuracy
        if validation_summary['validated_predictions'] > 0:
            overall_accuracy = (validation_summary['correct_predictions'] / validation_summary['validated_predictions']) * 100
            validation_summary['overall_accuracy'] = overall_accuracy
            logger.info(f"Overall accuracy: {overall_accuracy:.1f}% ({validation_summary['correct_predictions']}/{validation_summary['validated_predictions']})")
        
        return validation_summary


async def main():
    """Main function to generate historic predictions"""
    
    print("🕰️ Historic ML Prediction Generator")
    print("=" * 50)
    
    generator = HistoricPredictionGenerator()
    
    try:
        # Check available data
        print("\n🔍 Checking Available Data...")
        for currency in ['BTC', 'ETH']:
            data_info = await generator.get_available_price_data_range(currency)
            if data_info['available']:
                print(f"  {currency}: {data_info['total_records']} records")
                print(f"      Date range: {data_info['earliest_date'].date()} to {data_info['latest_date'].date()}")
                print(f"      Span: {data_info['date_range_days']} days")
            else:
                print(f"  {currency}: No data available")
        
        # Define date range for historic predictions (last 3 months)
        end_date = datetime.now(timezone.utc) - timedelta(days=7)  # Stop 7 days ago
        start_date = end_date - timedelta(days=90)  # Go back 90 days
        
        print(f"\n🔮 Generating Historic Predictions...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Generate historic predictions for each currency
        all_results = {}
        for currency in ['BTC', 'ETH']:
            print(f"\n📈 Processing {currency}...")
            
            results = await generator.generate_historic_predictions_for_currency(
                currency, start_date, end_date, model_type="logistic_regression"
            )
            
            all_results[currency] = results
            
            print(f"  ✅ {results['successful_predictions']} predictions generated")
            print(f"  ⏭️ {results['skipped_predictions']} predictions skipped (already exist)")
            print(f"  ❌ {results['failed_predictions']} predictions failed")
        
        # Validate all predictions
        print(f"\n🔍 Validating Prediction Accuracy...")
        validation_summary = await generator.validate_all_historic_predictions()
        
        print(f"\n📊 Validation Results:")
        print(f"  Total predictions: {validation_summary['total_predictions']}")
        print(f"  Validated: {validation_summary['validated_predictions']}")
        print(f"  Correct: {validation_summary['correct_predictions']}")
        
        if 'overall_accuracy' in validation_summary:
            print(f"  Overall accuracy: {validation_summary['overall_accuracy']:.1f}%")
        
        for currency, stats in validation_summary['currencies'].items():
            if 'error' not in stats and stats['validated_predictions'] > 0:
                accuracy = (stats['correct_predictions'] / stats['validated_predictions']) * 100
                print(f"  {currency} accuracy: {accuracy:.1f}%")
        
        print(f"\n🎉 Historic prediction generation complete!")
        print("Refresh your predictions dashboard to see the historic data!")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 