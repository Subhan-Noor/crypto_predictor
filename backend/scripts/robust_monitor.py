#!/usr/bin/env python3
"""
Robust Model Performance Monitor

This script adapts to the actual table structure and provides clear feedback.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def robust_monitor_models():
    """Robust monitoring that adapts to actual table structure"""
    try:
        print("🔍 Robust Model Performance Monitor")
        print("=" * 50)
        
        # Check database connection
        if not db_manager.is_connected():
            print("❌ Database not connected. Please check your Supabase credentials.")
            return {
                'metrics': {},
                'retraining_recommendations': []
            }
        
        logger.info("Database connection verified ✅")
        
        # Get table structure first
        try:
            sample_result = db_manager.client.table("predictions").select("*").limit(1).execute()
            
            if not sample_result.data:
                print("⚠️  No prediction data found - models may need initial training")
                return {
                    'metrics': {'BTC': {'has_data': False, 'message': 'No predictions found'}},
                    'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': 'No prediction data available'}]
                }
            
            sample = sample_result.data[0]
            available_columns = list(sample.keys())
            print(f"📋 Available columns: {available_columns}")
            
            # Check for required columns
            date_column = None
            if 'prediction_date' in available_columns:
                date_column = 'prediction_date'
            elif 'date' in available_columns:
                date_column = 'date'
            elif 'created_at' in available_columns:
                date_column = 'created_at'
            
            if not date_column:
                print("❌ No date column found for monitoring")
                return {
                    'metrics': {'BTC': {'has_data': False, 'message': 'No date column found'}},
                    'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': 'Table structure incomplete - no date column'}]
                }
            
            # Check for performance monitoring columns
            has_actual_direction = 'actual_direction' in available_columns
            has_is_correct = 'is_correct' in available_columns
            has_confidence = 'confidence' in available_columns
            
            print(f"✅ Date column: {date_column}")
            print(f"✅ Actual direction: {has_actual_direction}")
            print(f"✅ Is correct: {has_is_correct}")
            print(f"✅ Confidence: {has_confidence}")
            
            # Get total prediction count
            count_result = db_manager.client.table("predictions").select("id").execute()
            total_predictions = len(count_result.data) if count_result.data else 0
            print(f"📊 Total predictions: {total_predictions}")
            
            if total_predictions == 0:
                print("⚠️  No predictions found - models need training")
                return {
                    'metrics': {'BTC': {'has_data': False, 'message': 'No predictions found'}},
                    'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': 'No prediction data available'}]
                }
            
            # Check if we can do performance monitoring
            if has_actual_direction and has_is_correct:
                print("✅ Can perform detailed performance monitoring")
                
                # Get recent predictions with actual direction
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                recent_result = db_manager.client.table("predictions")\
                    .select("id")\
                    .gte(date_column, start_date.date().isoformat())\
                    .lte(date_column, end_date.date().isoformat())\
                    .not_.is_("actual_direction", "null")\
                    .execute()
                
                recent_count = len(recent_result.data) if recent_result.data else 0
                print(f"📊 Recent predictions with actual direction: {recent_count}")
                
                if recent_count == 0:
                    print("⚠️  No recent predictions with actual direction - models may need validation")
                    return {
                        'metrics': {'BTC': {'has_data': False, 'message': 'No recent predictions with actual direction'}},
                        'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': 'No recent predictions with actual direction'}]
                    }
                else:
                    print("✅ Recent predictions available for performance analysis")
                    return {
                        'metrics': {'BTC': {'has_data': True, 'message': f'Found {recent_count} recent predictions with actual direction'}},
                        'retraining_recommendations': []
                    }
            else:
                print("⚠️  Cannot perform detailed performance monitoring - missing required columns")
                return {
                    'metrics': {'BTC': {'has_data': False, 'message': f'Missing columns: actual_direction={has_actual_direction}, is_correct={has_is_correct}'}},
                    'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': f'Table structure incomplete - missing performance columns'}]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing table structure: {str(e)}")
            print(f"❌ Error analyzing table structure: {str(e)}")
            return {
                'metrics': {},
                'retraining_recommendations': []
            }
        
    except Exception as e:
        logger.error(f"Error in robust_monitor_models: {str(e)}")
        print(f"\n❌ Error during monitoring: {str(e)}")
        return {
            'metrics': {},
            'retraining_recommendations': []
        }


if __name__ == "__main__":
    try:
        result = asyncio.run(robust_monitor_models())
        if result is None:
            print("\n" + "="*50)
            print("❌ Robust monitoring failed - no result returned")
            print("="*50)
            sys.exit(1)
        else:
            print("\n" + "="*50)
            print("✅ Robust monitoring completed successfully")
            print("="*50)
            
            # Check if retraining is recommended
            if result.get('retraining_recommendations'):
                print("\n⚠️  RETRAINING RECOMMENDATIONS")
                print("=" * 50)
                for rec in result['retraining_recommendations']:
                    print(f"🔴 {rec.get('currency', 'Unknown')}: {rec.get('reason', 'Unknown reason')}")
            else:
                print("\n✅ No retraining needed")
                
    except Exception as e:
        print(f"\n❌ Error during robust monitoring: {str(e)}")
        print("="*50)
        print("RETRAINING RECOMMENDATIONS")
        print("="*50)
        print("🔴 Error occurred - manual investigation required")
        print(f"Error details: {str(e)}")
        sys.exit(1)
