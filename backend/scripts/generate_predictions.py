#!/usr/bin/env python3
"""
Standalone Prediction Generation Script

This script generates predictions for cryptocurrencies and can be used:
- As a standalone automation tool
- For manual prediction generation
- For testing the prediction pipeline
- For integration with cron or other schedulers

Usage:
    python generate_predictions.py --currency BTC        # Predict for BTC only
    python generate_predictions.py --currency ETH        # Predict for ETH only
    python generate_predictions.py --all                 # Predict for both currencies
    python generate_predictions.py --test                # Test mode (no database save)
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger
from ml.prediction_pipeline import CryptoPredictionPipeline, make_daily_predictions
from config import settings


class PredictionGenerator:
    """Handles prediction generation for crypto currencies"""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.pipeline = CryptoPredictionPipeline()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_mode": test_mode,
            "predictions": {},
            "summary": {},
            "errors": []
        }
    
    async def generate_prediction(self, currency: str, model_type: str = "best") -> Dict[str, Any]:
        """Generate a single prediction for a currency"""
        logger.info(f"🤖 Generating prediction for {currency}...")
        
        try:
            if self.test_mode:
                # Test mode: make prediction but don't save
                prediction = await self.pipeline.make_prediction(
                    currency=currency,
                    model_type=model_type
                )
                logger.info(f"✅ Test prediction for {currency}: {prediction['predicted_direction']} (confidence: {prediction['confidence_score']:.3f})")
            else:
                # Production mode: make and save prediction
                prediction = await self.pipeline.make_and_save_prediction(
                    currency=currency,
                    model_type=model_type
                )
                logger.info(f"✅ Prediction saved for {currency}: {prediction['predicted_direction']} (confidence: {prediction['confidence_score']:.3f})")
            
            self.results["predictions"][currency] = {
                "status": "success",
                "prediction": prediction["predicted_direction"],
                "confidence_score": prediction["confidence_score"],
                "model_version": prediction.get("model_version", "unknown"),
                "saved_to_db": not self.test_mode,
                "prediction_id": prediction.get("id")
            }
            
            return {"status": "success", "prediction": prediction}
            
        except Exception as e:
            error_msg = f"Failed to generate prediction for {currency}: {str(e)}"
            logger.error(error_msg)
            
            self.results["predictions"][currency] = {
                "status": "failed",
                "error": error_msg
            }
            self.results["errors"].append(error_msg)
            
            return {"status": "failed", "error": error_msg}
    
    async def generate_all_predictions(self, currencies: List[str] = None, model_type: str = "best") -> Dict[str, Any]:
        """Generate predictions for multiple currencies"""
        if currencies is None:
            currencies = ["BTC", "ETH"]
        
        logger.info(f"🚀 Generating predictions for {len(currencies)} currencies...")
        
        successful = 0
        failed = 0
        
        for currency in currencies:
            result = await self.generate_prediction(currency, model_type)
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
        
        # Generate summary
        self.results["summary"] = {
            "total_currencies": len(currencies),
            "successful_predictions": successful,
            "failed_predictions": failed,
            "success_rate": successful / len(currencies) if currencies else 0,
            "overall_status": "success" if failed == 0 else "partial" if successful > 0 else "failed"
        }
        
        # Log summary
        logger.info(f"📊 Prediction summary: {successful} successful, {failed} failed")
        
        return self.results
    
    async def use_daily_pipeline(self) -> Dict[str, Any]:
        """Use the existing daily predictions pipeline"""
        logger.info("🔄 Using daily predictions pipeline...")
        
        try:
            if self.test_mode:
                logger.warning("⚠️ Test mode enabled - predictions will not be saved to database")
            
            # Use the existing make_daily_predictions function
            prediction_results = await make_daily_predictions()
            
            successful = 0
            failed = 0
            
            for currency, result in prediction_results.items():
                if "error" in result:
                    failed += 1
                    self.results["predictions"][currency] = {
                        "status": "failed",
                        "error": result["error"]
                    }
                    self.results["errors"].append(f"{currency}: {result['error']}")
                else:
                    successful += 1
                    self.results["predictions"][currency] = {
                        "status": "success",
                        "prediction": result["predicted_direction"],
                        "confidence_score": result["confidence_score"],
                        "model_version": result.get("model_version", "unknown"),
                        "saved_to_db": True,  # Daily pipeline always saves
                        "prediction_id": result.get("id")
                    }
            
            self.results["summary"] = {
                "total_currencies": len(prediction_results),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "success_rate": successful / len(prediction_results) if prediction_results else 0,
                "overall_status": "success" if failed == 0 else "partial" if successful > 0 else "failed"
            }
            
            logger.info(f"📊 Daily pipeline summary: {successful} successful, {failed} failed")
            return self.results
            
        except Exception as e:
            error_msg = f"Daily pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            self.results["summary"] = {
                "overall_status": "failed",
                "error": error_msg
            }
            return self.results
    
    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*50)
        print("🤖 PREDICTION GENERATION RESULTS")
        print("="*50)
        
        print(f"📅 Timestamp: {self.results['timestamp']}")
        print(f"🧪 Test Mode: {'Yes' if self.test_mode else 'No'}")
        
        if self.results["predictions"]:
            print(f"\n📊 PREDICTIONS:")
            for currency, result in self.results["predictions"].items():
                if result["status"] == "success":
                    confidence = result.get("confidence_score", 0)
                    direction = result.get("prediction", "Unknown")
                    saved = "💾" if result.get("saved_to_db", False) else "🧪"
                    print(f"  {saved} {currency}: {direction} (confidence: {confidence:.3f})")
                else:
                    print(f"  ❌ {currency}: {result.get('error', 'Unknown error')}")
        
        if self.results["summary"]:
            summary = self.results["summary"]
            print(f"\n📈 SUMMARY:")
            print(f"  Status: {summary.get('overall_status', 'unknown')}")
            if "successful_predictions" in summary:
                print(f"  Success Rate: {summary['successful_predictions']}/{summary['total_currencies']} ({summary['success_rate']:.1%})")
        
        if self.results["errors"]:
            print(f"\n❌ ERRORS:")
            for error in self.results["errors"]:
                print(f"  • {error}")
        
        print("="*50)
    
    def save_results(self, output_file: str = None) -> str:
        """Save results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"prediction_results_{timestamp}.json"
        
        try:
            # Ensure results directory exists
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
            os.makedirs(results_dir, exist_ok=True)
            
            output_path = os.path.join(results_dir, output_file)
            
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"📄 Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate crypto price predictions")
    
    # Prediction target
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--currency", choices=["BTC", "ETH"], help="Generate prediction for specific currency")
    target_group.add_argument("--all", action="store_true", help="Generate predictions for all currencies")
    target_group.add_argument("--daily", action="store_true", help="Use daily predictions pipeline")
    
    # Options
    parser.add_argument("--model-type", default="best", help="Model type to use for predictions")
    parser.add_argument("--test", action="store_true", help="Test mode (don't save to database)")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Initialize prediction generator
    generator = PredictionGenerator(test_mode=args.test)
    
    try:
        # Determine what to predict
        if args.currency:
            currencies = [args.currency]
            results = await generator.generate_all_predictions(currencies, args.model_type)
        elif args.all:
            currencies = ["BTC", "ETH"]
            results = await generator.generate_all_predictions(currencies, args.model_type)
        elif args.daily:
            results = await generator.use_daily_pipeline()
        else:
            # Default: use daily pipeline
            results = await generator.use_daily_pipeline()
        
        # Save results if requested
        if args.save_results:
            generator.save_results(args.output)
        
        # Print results unless in quiet mode
        if not args.quiet:
            generator.print_results()
        
        # Exit with appropriate code
        summary = results.get("summary", {})
        overall_status = summary.get("overall_status", "failed")
        
        if overall_status == "success":
            sys.exit(0)
        elif overall_status == "partial":
            sys.exit(0)  # Partial success is still acceptable
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 