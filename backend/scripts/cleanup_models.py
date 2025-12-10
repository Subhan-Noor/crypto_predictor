#!/usr/bin/env python3
"""
Clean up old model files to reduce repository size.

This script keeps only the most recent model files for each currency and model type,
removing older versions to reduce repository bloat.
"""

import os
import glob
import re
from datetime import datetime
from pathlib import Path

def parse_model_filename(filename: str) -> tuple:
    """Parse model filename to extract currency, model type, and timestamp."""
    # Pattern: {currency}_{model_type}_{timestamp}.pkl
    pattern = r'(\w+)_(\w+)_(\d{8}_\d{6})\.pkl'
    match = re.match(pattern, filename)
    
    if match:
        currency = match.group(1)
        model_type = match.group(2)
        timestamp = match.group(3)
        return currency, model_type, timestamp
    return None

def get_latest_models(models_dir: str = "models") -> dict:
    """Get the latest model file for each currency and model type combination."""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory {models_dir} does not exist")
        return {}
    
    # Group models by currency and type
    model_groups = {}
    
    for model_file in models_path.glob("*.pkl"):
        parsed = parse_model_filename(model_file.name)
        if parsed:
            currency, model_type, timestamp = parsed
            key = f"{currency}_{model_type}"
            
            if key not in model_groups:
                model_groups[key] = []
            
            model_groups[key].append((timestamp, model_file))
    
    # Keep only the latest model for each group
    latest_models = {}
    for key, models in model_groups.items():
        if models:
            # Sort by timestamp (newest first)
            models.sort(key=lambda x: x[0], reverse=True)
            latest_models[key] = models[0][1]  # Keep the newest
    
    return latest_models

def cleanup_old_models(models_dir: str = "models", dry_run: bool = True) -> None:
    """Remove old model files, keeping only the latest ones."""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory {models_dir} does not exist")
        return
    
    latest_models = get_latest_models(models_dir)
    all_model_files = list(models_path.glob("*.pkl"))
    
    files_to_remove = []
    files_to_keep = []
    
    for model_file in all_model_files:
        parsed = parse_model_filename(model_file.name)
        if parsed:
            currency, model_type, timestamp = parsed
            key = f"{currency}_{model_type}"
            
            if key in latest_models and model_file == latest_models[key]:
                files_to_keep.append(model_file)
            else:
                files_to_remove.append(model_file)
        else:
            print(f"Warning: Could not parse filename {model_file.name}")
    
    print(f"Found {len(all_model_files)} total model files")
    print(f"Keeping {len(files_to_keep)} latest model files")
    print(f"Would remove {len(files_to_remove)} old model files")
    
    if dry_run:
        print("\n=== DRY RUN - No files will be deleted ===")
        print("Files to keep:")
        for file in files_to_keep:
            print(f"  ✓ {file.name}")
        
        print("\nFiles to remove:")
        for file in files_to_remove:
            print(f"  ✗ {file.name}")
    else:
        print("\n=== REMOVING OLD MODEL FILES ===")
        for file in files_to_remove:
            try:
                file.unlink()
                print(f"  ✗ Removed: {file.name}")
            except Exception as e:
                print(f"  ✗ Error removing {file.name}: {e}")
        
        print(f"\nCleanup complete! Kept {len(files_to_keep)} files, removed {len(files_to_remove)} files")

def main():
    """Main function to run the cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old model files")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry run)")
    
    args = parser.parse_args()
    
    print("🔧 Model Cleanup Script")
    print("=" * 50)
    
    cleanup_old_models(args.models_dir, dry_run=not args.execute)
    
    if not args.execute:
        print("\n💡 To actually delete files, run with --execute flag")

if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*50)
        print("✅ Cleanup script completed successfully")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Error during cleanup: {str(e)}")
        print("="*50)
        print("Cleanup failed - check logs for details")
        print("="*50)
        import sys
        sys.exit(1) 