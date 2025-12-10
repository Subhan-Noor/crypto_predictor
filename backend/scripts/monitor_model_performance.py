#!/usr/bin/env python3
"""
Model Performance Monitor

This script monitors model performance and can trigger retraining if needed.
It checks prediction accuracy over the last 30 days and compares it to historical performance.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.55  # 55% minimum accuracy
PERFORMANCE_CHECK_DAYS = 30    # Check last 30 days of predictions
RETRAIN_TRIGGER_DAYS = 7       # Trigger retraining if accuracy is low for 7+ days


async def get_recent_predictions(currency: str, days: int = 30) -> pd.DataFrame:
    """Get recent predictions for performance analysis"""
    try:
        # Use the new database method for monitoring
        predictions = await db_manager.get_recent_predictions_for_monitoring(currency, days)
        
        if not predictions:
            logger.warning(f"No predictions found for {currency} in the last {days} days")
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions, columns=[
            'date', 'predicted_direction', 'actual_direction', 
            'confidence', 'model_version', 'is_correct'
        ])
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching predictions for {currency}: {e}")
        return pd.DataFrame()


async def calculate_performance_metrics(currency: str) -> Dict[str, Any]:
    """Calculate performance metrics for a currency"""
    logger.info(f"Calculating performance metrics for {currency}")
    
    # Get recent predictions
    df = await get_recent_predictions(currency, PERFORMANCE_CHECK_DAYS)
    
    if df.empty:
        return {
            'currency': currency,
            'has_data': False,
            'message': f"No prediction data available for {currency}"
        }
    
    # Calculate metrics
    total_predictions = len(df)
    correct_predictions = df['is_correct'].sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate confidence-weighted accuracy
    if 'confidence' in df.columns and not df['confidence'].isna().all():
        confidence_weighted_accuracy = (
            (df['is_correct'] * df['confidence']).sum() / df['confidence'].sum()
            if df['confidence'].sum() > 0 else 0
        )
    else:
        confidence_weighted_accuracy = accuracy
    
    # Get model version distribution
    model_distribution = df['model_version'].value_counts().to_dict()
    
    # Calculate daily accuracy trend
    daily_accuracy = df.groupby('date')['is_correct'].mean().reset_index()
    recent_accuracy = daily_accuracy.tail(7)['is_correct'].mean() if len(daily_accuracy) >= 7 else accuracy
    
    metrics = {
        'currency': currency,
        'has_data': True,
        'total_predictions': total_predictions,
        'accuracy': round(accuracy, 4),
        'confidence_weighted_accuracy': round(confidence_weighted_accuracy, 4),
        'recent_accuracy': round(recent_accuracy, 4),
        'model_distribution': model_distribution,
        'date_range': {
            'start': df['date'].min().isoformat() if not df.empty else None,
            'end': df['date'].max().isoformat() if not df.empty else None
        }
    }
    
    logger.info(f"{currency} - Accuracy: {accuracy:.2%}, Recent: {recent_accuracy:.2%}")
    
    return metrics


async def check_retraining_needed(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if retraining is needed based on performance metrics"""
    currency = metrics['currency']
    
    if not metrics['has_data']:
        return {
            'currency': currency,
            'retraining_needed': False,
            'reason': 'No prediction data available'
        }
    
    accuracy = metrics['accuracy']
    recent_accuracy = metrics['recent_accuracy']
    
    # Check if accuracy is below threshold
    if accuracy < MIN_ACCURACY_THRESHOLD:
        return {
            'currency': currency,
            'retraining_needed': True,
            'reason': f'Low accuracy: {accuracy:.2%} < {MIN_ACCURACY_THRESHOLD:.2%}',
            'current_accuracy': accuracy,
            'threshold': MIN_ACCURACY_THRESHOLD
        }
    
    # Check if recent accuracy is significantly lower than overall
    if recent_accuracy < accuracy * 0.8:  # 20% drop in recent performance
        return {
            'currency': currency,
            'retraining_needed': True,
            'reason': f'Performance degradation: recent {recent_accuracy:.2%} vs overall {accuracy:.2%}',
            'recent_accuracy': recent_accuracy,
            'overall_accuracy': accuracy
        }
    
    return {
        'currency': currency,
        'retraining_needed': False,
        'reason': f'Performance acceptable: {accuracy:.2%}',
        'current_accuracy': accuracy
    }


async def monitor_all_models():
    """Monitor performance for all currencies"""
    try:
        print("🔍 Model Performance Monitor")
        print("=" * 50)
        
        # Check database connection
        if not db_manager.is_connected():
            print("❌ Database not connected. Please check your Supabase credentials.")
            return {
                'metrics': {},
                'retraining_recommendations': []
            }
        
        logger.info("Database connection verified ✅")
        
        currencies = ['BTC', 'ETH']
        all_metrics = {}
        retraining_recommendations = []
        
        # Calculate metrics for each currency
        for currency in currencies:
            try:
                metrics = await calculate_performance_metrics(currency)
                all_metrics[currency] = metrics
                
                # Check if retraining is needed
                retraining_check = await check_retraining_needed(metrics)
                if retraining_check['retraining_needed']:
                    retraining_recommendations.append(retraining_check)
            except Exception as e:
                logger.error(f"Error processing {currency}: {str(e)}")
                all_metrics[currency] = {
                    'currency': currency,
                    'has_data': False,
                    'message': f"Error processing: {str(e)}"
                }
        
        # Print summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        for currency, metrics in all_metrics.items():
            if metrics['has_data']:
                print(f"✅ {currency}:")
                print(f"   Accuracy: {metrics['accuracy']:.2%}")
                print(f"   Recent: {metrics['recent_accuracy']:.2%}")
                print(f"   Predictions: {metrics['total_predictions']}")
                print(f"   Models: {', '.join(metrics['model_distribution'].keys())}")
            else:
                print(f"❌ {currency}: {metrics['message']}")
        
        # Print retraining recommendations
        if retraining_recommendations:
            print(f"\n⚠️  RETRAINING RECOMMENDATIONS")
            print("=" * 50)
            for rec in retraining_recommendations:
                print(f"🔴 {rec['currency']}: {rec['reason']}")
            
            print(f"\n💡 To trigger retraining, run:")
            print(f"   curl -X POST https://your-api-url/tasks/retrain_models")
        else:
            print(f"\n✅ All models performing well - no retraining needed")
        
        return {
            'metrics': all_metrics,
            'retraining_recommendations': retraining_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error in monitor_all_models: {str(e)}")
        print(f"\n❌ Error during monitoring: {str(e)}")
        return {
            'metrics': {},
            'retraining_recommendations': []
        }


if __name__ == "__main__":
    try:
        result = asyncio.run(monitor_all_models())
        if result is None:
            print("\n" + "="*50)
            print("❌ Monitoring failed - no result returned")
            print("="*50)
            sys.exit(1)
        else:
            print("\n" + "="*50)
            print("✅ Monitoring completed successfully")
            print("="*50)
    except Exception as e:
        print(f"\n❌ Error during monitoring: {str(e)}")
        print("="*50)
        print("RETRAINING RECOMMENDATIONS")
        print("="*50)
        print("🔴 Error occurred - manual investigation required")
        print(f"Error details: {str(e)}")
        sys.exit(1) 