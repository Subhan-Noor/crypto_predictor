"""
Real ML Prediction Generation Script

This script generates actual machine learning predictions using:
- Latest historical price data from Supabase
- Current sentiment data (Reddit/Twitter)  
- Trained ML models (Random Forest, Logistic Regression, LSTM)
- Real feature engineering pipeline

Predictions are saved to the database and can be validated against actual price movements.
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
from ml.model_trainer import CryptoModelTrainer


class RealPredictionGenerator:
    """Generates real ML predictions using latest data and trained models"""
    
    def __init__(self):
        """Initialize the prediction generator"""
        self.pipeline = CryptoPredictionPipeline()
        self.preprocessor = CryptoDataPreprocessor()
        self.feature_engineer = CryptoFeatureEngineer()
        self.currencies = ['BTC', 'ETH']
        self.model_types = ['random_forest', 'logistic_regression', 'lstm']
        
        logger.info("Real Prediction Generator initialized")
    
    async def check_data_availability(self, currency: str) -> Dict[str, Any]:
        """
        Check if we have sufficient data for making predictions
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Dictionary with data availability status
        """
        try:
            # Check price data (need at least 60 days for features)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=90)  # Extra buffer
            
            price_data = await db_manager.get_crypto_prices(currency, limit=100)
            sentiment_data = await db_manager.get_crypto_sentiment(currency, limit=100)
            
            # Convert to DataFrames for analysis
            price_df = pd.DataFrame(price_data) if price_data else pd.DataFrame()
            sentiment_df = pd.DataFrame(sentiment_data) if sentiment_data else pd.DataFrame()
            
            # Check data quality
            data_status = {
                'currency': currency,
                'price_records': len(price_df),
                'sentiment_records': len(sentiment_df),
                'sufficient_data': len(price_df) >= 30,  # Only require price data, sentiment is optional
                'latest_price_date': price_df['date'].max() if not price_df.empty else None,
                'latest_sentiment_date': sentiment_df['date'].max() if not sentiment_df.empty else None,
                'data_quality': 'good' if len(price_df) >= 60 else 'limited',
                'has_sentiment': len(sentiment_df) > 0
            }
            
            logger.info(f"Data availability for {currency}: {data_status}")
            return data_status
            
        except Exception as e:
            logger.error(f"Error checking data availability for {currency}: {e}")
            return {
                'currency': currency,
                'error': str(e),
                'sufficient_data': False
            }
    
    async def generate_prediction_for_currency(self, currency: str, model_type: str = "best") -> Optional[Dict[str, Any]]:
        """
        Generate a real ML prediction for a specific currency
        
        Args:
            currency: 'BTC' or 'ETH'
            model_type: Type of model to use ('best', 'random_forest', 'logistic_regression', 'lstm')
            
        Returns:
            Prediction result dictionary or None if failed
        """
        try:
            logger.info(f"🔮 Generating {model_type} prediction for {currency}...")
            
            # Check data availability first
            data_status = await self.check_data_availability(currency)
            if not data_status.get('sufficient_data', False):
                logger.warning(f"Insufficient data for {currency}: {data_status}")
                return None
            
            # Generate prediction using the ML pipeline
            prediction = await self.pipeline.make_and_save_prediction(currency, model_type)
            
            # If best model fails for ETH due to sklearn issues, try logistic regression
            if prediction is None and currency == 'ETH' and model_type == 'best':
                logger.warning(f"Best model failed for {currency}, trying logistic_regression...")
                prediction = await self.pipeline.make_and_save_prediction(currency, 'logistic_regression')
            
            if prediction:
                # Add target date (7 days from now)
                prediction_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00')) if isinstance(prediction['prediction_date'], str) else prediction['prediction_date']
                target_date = prediction_date + timedelta(days=7)
                prediction['target_date'] = target_date.isoformat()
                
                # Add data quality info
                prediction['data_quality'] = data_status['data_quality']
                prediction['features_count'] = len(prediction.get('features_used', []))
                
                logger.info(f"✅ Successfully generated {model_type} prediction for {currency}")
                logger.info(f"   Direction: {prediction.get('predicted_direction', 'N/A')}")
                logger.info(f"   Confidence: {prediction.get('confidence_score', 0):.4f}")
                logger.info(f"   Model: {prediction.get('model_version', 'N/A')}")
                logger.info(f"   Target Date: {prediction.get('target_date', 'N/A')}")
                
                return prediction
            else:
                logger.error(f"❌ Failed to generate prediction for {currency}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating prediction for {currency} with {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate_all_predictions(self, model_type: str = "best") -> Dict[str, Any]:
        """
        Generate predictions for all currencies
        
        Args:
            model_type: Type of model to use
            
        Returns:
            Dictionary with results for each currency
        """
        results = {}
        
        logger.info(f"🚀 Starting prediction generation for all currencies with {model_type} model")
        
        for currency in self.currencies:
            try:
                prediction = await self.generate_prediction_for_currency(currency, model_type)
                
                if prediction:
                    results[currency] = {
                        'success': True,
                        'prediction': prediction
                    }
                else:
                    results[currency] = {
                        'success': False,
                        'error': 'Failed to generate prediction'
                    }
                    
            except Exception as e:
                logger.error(f"Error processing {currency}: {e}")
                results[currency] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def validate_old_predictions(self, days_old: int = 7) -> Dict[str, Any]:
        """
        Validate predictions that are now due for validation
        
        Args:
            days_old: How many days old predictions to validate
            
        Returns:
            Validation results
        """
        logger.info(f"🔍 Validating predictions from {days_old} days ago...")
        
        validation_results = {}
        
        for currency in self.currencies:
            try:
                # Get predictions from N days ago
                target_date = datetime.now(timezone.utc) - timedelta(days=days_old)
                predictions = await db_manager.get_predictions(currency, days=days_old+2, limit=50)
                
                # Filter to predictions that need validation
                predictions_to_validate = []
                for pred in predictions:
                    pred_date = datetime.fromisoformat(pred['prediction_date'].replace('Z', '+00:00'))
                    if abs((pred_date - target_date).days) <= 1:  # Within 1 day
                        predictions_to_validate.append(pred)
                
                if not predictions_to_validate:
                    logger.info(f"No predictions to validate for {currency}")
                    validation_results[currency] = {'predictions_validated': 0}
                    continue
                
                # Get actual price data for validation
                current_price_data = await db_manager.get_crypto_prices(currency, limit=10)
                
                if not current_price_data:
                    logger.warning(f"No current price data for validation of {currency}")
                    continue
                
                # Validate each prediction
                validated_count = 0
                for pred in predictions_to_validate:
                    try:
                        validation_result = await self._validate_single_prediction(pred, current_price_data)
                        if validation_result:
                            validated_count += 1
                    except Exception as e:
                        logger.error(f"Error validating prediction {pred.get('id', 'unknown')}: {e}")
                
                validation_results[currency] = {
                    'predictions_found': len(predictions_to_validate),
                    'predictions_validated': validated_count
                }
                
                logger.info(f"✅ Validated {validated_count}/{len(predictions_to_validate)} predictions for {currency}")
                
            except Exception as e:
                logger.error(f"Error validating predictions for {currency}: {e}")
                validation_results[currency] = {'error': str(e)}
        
        return validation_results
    
    async def _validate_single_prediction(self, prediction: Dict[str, Any], price_data: List[Dict[str, Any]]) -> bool:
        """
        Validate a single prediction against actual price data
        
        Args:
            prediction: Prediction to validate
            price_data: Recent price data
            
        Returns:
            True if validation was successful
        """
        try:
            # Get prediction details
            pred_date = datetime.fromisoformat(prediction['prediction_date'].replace('Z', '+00:00'))
            predicted_direction = prediction['predicted_direction']
            
            # Find price at prediction time and current price
            pred_price = None
            current_price = None
            
            # Convert price data to DataFrame for easier manipulation
            price_df = pd.DataFrame(price_data)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Find price closest to prediction date
            price_df['date_diff'] = abs((price_df['date'] - pred_date).dt.total_seconds())
            pred_price_row = price_df.loc[price_df['date_diff'].idxmin()]
            pred_price = float(pred_price_row['close'])
            
            # Get most recent price
            latest_price_row = price_df.loc[price_df['date'].idxmax()]
            current_price = float(latest_price_row['close'])
            
            # Determine actual direction
            if current_price > pred_price:
                actual_direction = 'UP'
            else:
                actual_direction = 'DOWN'
            
            # Check if prediction was correct
            is_correct = (predicted_direction == actual_direction)
            
            # Update prediction in database (would need to implement this method)
            # For now, just log the result
            logger.info(f"Prediction {prediction.get('id', 'unknown')}: "
                       f"Predicted {predicted_direction}, Actual {actual_direction}, "
                       f"Correct: {is_correct}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating single prediction: {e}")
            return False
    
    async def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent predictions and their status
        
        Returns:
            Summary of prediction status
        """
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'currencies': {}
        }
        
        for currency in self.currencies:
            try:
                # Get recent predictions
                recent_predictions = await db_manager.get_predictions(currency, days=30, limit=20)
                
                if recent_predictions:
                    # Calculate basic stats
                    total_predictions = len(recent_predictions)
                    avg_confidence = np.mean([p.get('confidence_score', 0) for p in recent_predictions])
                    
                    # Count by direction
                    up_predictions = len([p for p in recent_predictions if p.get('predicted_direction') == 'UP'])
                    down_predictions = len([p for p in recent_predictions if p.get('predicted_direction') == 'DOWN'])
                    
                    # Get latest prediction
                    latest_prediction = recent_predictions[0] if recent_predictions else None
                    
                    summary['currencies'][currency] = {
                        'total_predictions': total_predictions,
                        'average_confidence': float(avg_confidence),
                        'up_predictions': up_predictions,
                        'down_predictions': down_predictions,
                        'latest_prediction': {
                            'date': latest_prediction.get('prediction_date') if latest_prediction else None,
                            'direction': latest_prediction.get('predicted_direction') if latest_prediction else None,
                            'confidence': latest_prediction.get('confidence_score') if latest_prediction else None
                        }
                    }
                else:
                    summary['currencies'][currency] = {
                        'total_predictions': 0,
                        'error': 'No predictions found'
                    }
                    
            except Exception as e:
                logger.error(f"Error getting summary for {currency}: {e}")
                summary['currencies'][currency] = {'error': str(e)}
        
        return summary


async def main():
    """Main function to run prediction generation"""
    
    print("🤖 Real ML Prediction Generator")
    print("=" * 50)
    
    generator = RealPredictionGenerator()
    
    try:
        # Check data availability
        print("\n🔍 Checking Data Availability...")
        for currency in ['BTC', 'ETH']:
            data_status = await generator.check_data_availability(currency)
            print(f"  {currency}: {data_status['price_records']} price records, "
                  f"{data_status['sentiment_records']} sentiment records")
            print(f"      Sufficient data: {'✅' if data_status['sufficient_data'] else '❌'}")
        
        # Generate new predictions
        print("\n🔮 Generating New Predictions...")
        results = await generator.generate_all_predictions(model_type="best")
        
        # Display results
        print("\n📊 Prediction Generation Results:")
        successful = 0
        for currency, result in results.items():
            if result['success']:
                prediction = result['prediction']
                print(f"  ✅ {currency}:")
                print(f"     Direction: {prediction.get('predicted_direction', 'N/A')}")
                print(f"     Confidence: {prediction.get('confidence_score', 0):.4f}")
                print(f"     Model: {prediction.get('model_version', 'N/A')}")
                print(f"     Database ID: {prediction.get('id', 'N/A')}")
                successful += 1
            else:
                print(f"  ❌ {currency}: {result.get('error', 'Unknown error')}")
        
        # Get prediction summary
        print("\n📈 Prediction Summary:")
        summary = await generator.get_prediction_summary()
        for currency, stats in summary['currencies'].items():
            if 'error' not in stats:
                print(f"  {currency}: {stats['total_predictions']} total predictions, "
                      f"avg confidence: {stats['average_confidence']:.3f}")
            else:
                print(f"  {currency}: {stats['error']}")
        
        print(f"\n🎉 Generation complete! {successful}/{len(results)} currencies successful")
        print("\nRefresh your predictions dashboard to see the real ML predictions!")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 