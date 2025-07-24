#!/usr/bin/env python3
"""
Check Prediction Schema

This script checks the actual schema of the predictions table in the database.
"""

import os
import sys
import asyncio

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger


async def check_prediction_schema():
    """Check the schema of the predictions table"""
    
    print("🔍 Checking Predictions Table Schema")
    print("=" * 50)
    
    try:
        # Get a sample prediction to see the structure
        sample_predictions = await db_manager.get_predictions("BTC", days=1, limit=1)
        
        if sample_predictions:
            print("✅ Found existing predictions. Schema:")
            sample = sample_predictions[0]
            for key, value in sample.items():
                print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print("❌ No existing predictions found")
            
            # Try to get table info from Supabase
            if db_manager.client:
                try:
                    # Try a simple insert to see what columns are expected
                    test_data = {
                        "currency": "BTC",
                        "prediction": "UP",
                        "prediction_date": "2025-07-23T00:00:00Z",
                        "model_type": "random_forest"
                    }
                    
                    print(f"\n🔧 Testing insert with minimal data:")
                    for key, value in test_data.items():
                        print(f"  {key}: {value}")
                    
                    result = db_manager.client.table("predictions").insert(test_data).execute()
                    print(f"✅ Test insert successful: {result.data}")
                    
                except Exception as e:
                    print(f"❌ Test insert failed: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error checking schema: {str(e)}")


async def main():
    """Main function"""
    try:
        print("🔗 Database connection established automatically")
        
        # Check schema
        await check_prediction_schema()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 