#!/usr/bin/env python3
"""
Simple test script to verify Supabase connection
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db_manager

def test_connection():
    """Test database connection"""
    print("🔍 Testing Supabase connection...")
    
    try:
        # Test connection
        if db_manager.is_connected():
            print("✅ Database connection successful!")
            
            # Test a simple query
            result = db_manager.client.table("crypto_prices").select("id").limit(1).execute()
            print(f"✅ Query test successful! Found {len(result.data)} records")
            
            return True
        else:
            print("❌ Database connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Ready to run historical data population!")
    else:
        print("\n⚠️ Please check your .env file and Supabase credentials") 