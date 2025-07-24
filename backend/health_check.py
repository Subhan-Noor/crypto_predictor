#!/usr/bin/env python3
"""
Simple Health Check for Railway Deployment

This script provides a basic health check that Railway can use to verify the deployment.
"""

import os
import sys

def main():
    """Simple health check"""
    try:
        # Check if we can import the main app
        from app.enhanced_main import app
        print("✅ App import successful")
        
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_ROLE_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {missing_vars}")
            return 1
        else:
            print("✅ Environment variables configured")
        
        print("✅ Health check passed")
        return 0
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 