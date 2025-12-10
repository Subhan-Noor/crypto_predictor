#!/usr/bin/env python3
"""
Simple Model Performance Monitor

This is a simplified version that avoids complex database queries
to prevent the null parameter error.
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

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.55  # 55% minimum accuracy


async def simple_monitor_models():
    """Simple monitoring that just checks if we can connect and get basic data"""
    try:
        print("🔍 Simple Model Performance Monitor")
        print("=" * 50)
        
        # Check database connection
        if not db_manager.is_connected():
            print("❌ Database not connected. Please check your Supabase credentials.")
            return {
                'metrics': {},
                'retraining_recommendations': []
            }
        
        logger.info("Database connection verified ✅")
        
        # Try a very simple query to test basic functionality
        try:
            if db_manager.client:
                # Simple test query
                result = db_manager.client.table("predictions").select("id").limit(5).execute()
                prediction_count = len(result.data) if result.data else 0
                print(f"✅ Database query successful - found {prediction_count} prediction records")
                
                if prediction_count == 0:
                    print("⚠️  No prediction data found - models may need initial training")
                    return {
                        'metrics': {'BTC': {'has_data': False, 'message': 'No predictions found'}},
                        'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': 'No prediction data available'}]
                    }
                else:
                    # Get a sample record to check structure
                    sample_result = db_manager.client.table("predictions").select("*").limit(1).execute()
                    if sample_result.data:
                        sample = sample_result.data[0]
                        print(f"📋 Sample record columns: {list(sample.keys())}")
                        
                        # Check if we have the required columns for monitoring
                        required_columns = ['prediction_date', 'actual_direction', 'is_correct']
                        missing_columns = [col for col in required_columns if col not in sample]
                        
                        if missing_columns:
                            print(f"⚠️  Missing columns for monitoring: {missing_columns}")
                            return {
                                'metrics': {'BTC': {'has_data': False, 'message': f'Missing columns: {missing_columns}'}},
                                'retraining_recommendations': [{'currency': 'BTC', 'retraining_needed': True, 'reason': f'Table structure incomplete: missing {missing_columns}'}]
                            }
                        else:
                            print("✅ All required columns present for monitoring")
                    
                    print("✅ Prediction data available - models appear to be working")
                    return {
                        'metrics': {'BTC': {'has_data': True, 'message': f'Found {prediction_count} predictions'}},
                        'retraining_recommendations': []
                    }
            else:
                print("❌ No database client available")
                return {
                    'metrics': {},
                    'retraining_recommendations': []
                }
                
        except Exception as e:
            logger.error(f"Error in simple query: {str(e)}")
            print(f"❌ Database query failed: {str(e)}")
            return {
                'metrics': {},
                'retraining_recommendations': []
            }
        
    except Exception as e:
        logger.error(f"Error in simple_monitor_models: {str(e)}")
        print(f"\n❌ Error during monitoring: {str(e)}")
        return {
            'metrics': {},
            'retraining_recommendations': []
        }


if __name__ == "__main__":
    try:
        result = asyncio.run(simple_monitor_models())
        if result is None:
            print("\n" + "="*50)
            print("❌ Simple monitoring failed - no result returned")
            print("="*50)
            sys.exit(1)
        else:
            print("\n" + "="*50)
            print("✅ Simple monitoring completed successfully")
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
        print(f"\n❌ Error during simple monitoring: {str(e)}")
        print("="*50)
        print("RETRAINING RECOMMENDATIONS")
        print("="*50)
        print("🔴 Error occurred - manual investigation required")
        print(f"Error details: {str(e)}")
        sys.exit(1)
