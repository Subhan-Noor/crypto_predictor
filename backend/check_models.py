#!/usr/bin/env python3
"""
Check if new models were created successfully
"""

import os
import glob
from datetime import datetime

def check_models():
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("❌ Models directory does not exist")
        return
    
    print("🔍 Checking for trained models...")
    print("=" * 50)
    
    # Find all model files
    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    model_files.extend(glob.glob(os.path.join(models_dir, "*.pkl")))
    
    if not model_files:
        print("❌ No model files found")
        return
    
    print(f"✅ Found {len(model_files)} model files:")
    
    for model_file in sorted(model_files):
        filename = os.path.basename(model_file)
        size = os.path.getsize(model_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(model_file))
        
        print(f"  📁 {filename}")
        print(f"     Size: {size:,} bytes")
        print(f"     Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Check for specific currency models
    currencies = ['BTC', 'ETH']
    for currency in currencies:
        currency_models = [f for f in model_files if currency in f]
        if currency_models:
            print(f"✅ {currency} models: {len(currency_models)} found")
        else:
            print(f"❌ {currency} models: None found")

if __name__ == "__main__":
    check_models() 