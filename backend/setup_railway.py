#!/usr/bin/env python3
"""
Railway Setup Helper Script

This script helps you set up Railway environment variables quickly.
"""

import os
import sys

def main():
    print("🚀 Railway Setup Helper")
    print("=" * 40)
    
    print("\n📋 Required Environment Variables:")
    print("1. SUPABASE_URL")
    print("2. SUPABASE_KEY") 
    print("3. SUPABASE_SERVICE_ROLE_KEY")
    print("4. ENVIRONMENT (optional, set to 'production')")
    
    print("\n🔧 How to set them in Railway:")
    print("1. Go to your Railway project dashboard")
    print("2. Click on your backend service")
    print("3. Go to the 'Variables' tab")
    print("4. Add each variable with its value")
    
    print("\n📝 Example variables to add:")
    print("SUPABASE_URL=https://your-project.supabase.co")
    print("SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    print("SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    print("ENVIRONMENT=production")
    
    print("\n🔍 To get your Supabase credentials:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Go to Settings → API")
    print("4. Copy the Project URL and keys")
    
    print("\n✅ After setting variables:")
    print("1. Redeploy your Railway service")
    print("2. Check the logs for successful startup")
    print("3. Test the health endpoint")
    
    print("\n📚 For more help, see RAILWAY_SETUP.md")

if __name__ == "__main__":
    main() 