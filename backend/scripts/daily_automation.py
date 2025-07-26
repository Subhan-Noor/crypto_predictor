#!/usr/bin/env python3
"""
Daily Automation Script for Crypto Price Prediction App

This script handles:
1. Daily data ingestion (prices and sentiment)
2. Daily prediction generation
3. Prediction validation
4. Health checks and monitoring

Usage:
    python daily_automation.py --data-ingestion    # Run data ingestion only
    python daily_automation.py --predictions       # Run prediction generation only
    python daily_automation.py --validation        # Run prediction validation only
    python daily_automation.py --health-check      # Run health checks only
    python daily_automation.py --full              # Run all automation tasks
"""

import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.services.binance_service import BinancePriceFetcher
from app.logger import logger

class DailyAutomationManager:
    """Manages daily automation tasks for the crypto prediction app"""
    
    def __init__(self):
        self.price_fetcher = BinancePriceFetcher()
        self.currencies = ["BTC", "ETH"]
        self.start_time = datetime.now()
        
    async def run_data_ingestion(self) -> Dict[str, Any]:
        """Run daily data ingestion for prices and sentiment"""
        logger.info("🚀 Starting daily data ingestion...")
        
        results = {
            "prices": {},
            "sentiment": {},
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "errors": []
        }
        
        try:
            # Fetch and store price data
            for currency in self.currencies:
                try:
                    logger.info(f"📊 Fetching price data for {currency}")
                    
                    # Get latest prices from Binance
                    symbol = f"{currency}USDT"
                    historical_data = await self.price_fetcher.get_historical_prices(
                        symbol, interval="1d", limit=7  # Get last 7 days
                    )
                    
                    # Store in database
                    stored_count = 0
                    for entry in historical_data:
                        date_str = datetime.fromtimestamp(entry["open_time"] / 1000).strftime("%Y-%m-%d")
                        
                        success = db_manager.insert_price_data(
                            currency=currency,
                            date=date_str,
                            open_price=float(entry["open"]),
                            high=float(entry["high"]),
                            low=float(entry["low"]),
                            close=float(entry["close"]),
                            volume=float(entry["volume"])
                        )
                        
                        if success:
                            stored_count += 1
                    
                    results["prices"][currency] = stored_count
                    logger.info(f"✅ Stored {stored_count} price records for {currency}")
                    
                except Exception as e:
                    error_msg = f"Error fetching price data for {currency}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)
                    results["prices"][currency] = 0
            
            # Note: Sentiment data collection is handled by separate workflow
            # to avoid API rate limits and complexity
            results["sentiment"] = {"status": "handled_by_separate_workflow"}
            
        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["success"] = False
        
        logger.info(f"✅ Data ingestion completed in {datetime.now() - self.start_time}")
        return results
    
    async def run_prediction_generation(self) -> Dict[str, Any]:
        """Run daily prediction generation"""
        logger.info("🎯 Starting daily prediction generation...")
        
        results = {
            "predictions": {},
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "errors": []
        }
        
        try:
            # Import prediction pipeline
            from ml.clean_prediction_pipeline import CleanPredictionPipeline
            
            prediction_pipeline = CleanPredictionPipeline()
            
            for currency in self.currencies:
                try:
                    logger.info(f"🔮 Generating prediction for {currency}")
                    
                    # Generate and save prediction
                    prediction_result = await prediction_pipeline.make_and_save_prediction(currency)
                    
                    if prediction_result:
                        results["predictions"][currency] = {
                            "predicted_direction": prediction_result.get("predicted_direction"),
                            "confidence_score": prediction_result.get("confidence_score"),
                            "model_version": prediction_result.get("model_version")
                        }
                        logger.info(f"✅ Generated prediction for {currency}: {prediction_result.get('predicted_direction')}")
                    else:
                        error_msg = f"Failed to generate prediction for {currency}"
                        logger.error(f"❌ {error_msg}")
                        results["errors"].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error generating prediction for {currency}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)
            
        except Exception as e:
            error_msg = f"Prediction generation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["success"] = False
        
        logger.info(f"✅ Prediction generation completed in {datetime.now() - self.start_time}")
        return results
    
    async def run_prediction_validation(self) -> Dict[str, Any]:
        """Run prediction validation for completed predictions"""
        logger.info("✅ Starting prediction validation...")
        
        results = {
            "validated": 0,
            "correct": 0,
            "accuracy": 0.0,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "errors": []
        }
        
        try:
            # Import validation script
            from scripts.auto_validate_predictions import AutoPredictionValidator
            
            validator = AutoPredictionValidator()
            validation_summary = await validator.auto_validate_predictions()
            
            results["validated"] = validation_summary.get("total_validated", 0)
            results["correct"] = validation_summary.get("total_correct", 0)
            results["accuracy"] = validation_summary.get("overall_accuracy", 0.0)
            
            logger.info(f"✅ Validation completed: {results['validated']} predictions validated")
            
        except Exception as e:
            error_msg = f"Prediction validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["success"] = False
        
        logger.info(f"✅ Prediction validation completed in {datetime.now() - self.start_time}")
        return results
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run system health checks"""
        logger.info("🏥 Starting health checks...")
        
        results = {
            "database": {"status": "unknown"},
            "api_services": {"status": "unknown"},
            "models": {"status": "unknown"},
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown"
        }
        
        try:
            # Check database connection
            try:
                db_connected = await db_manager.test_connection()
                results["database"]["status"] = "healthy" if db_connected else "unhealthy"
                results["database"]["connected"] = db_connected
            except Exception as e:
                results["database"]["status"] = "error"
                results["database"]["error"] = str(e)
            
            # Check API services
            try:
                # Test Binance API
                btc_price = await self.price_fetcher.get_current_price("BTCUSDT")
                results["api_services"]["binance"] = "healthy" if btc_price else "unhealthy"
            except Exception as e:
                results["api_services"]["binance"] = "error"
                results["api_services"]["binance_error"] = str(e)
            
            # Check ML models
            try:
                import os
                model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')] if os.path.exists(model_dir) else []
                
                results["models"]["status"] = "healthy" if len(model_files) > 0 else "no_models"
                results["models"]["count"] = len(model_files)
                results["models"]["files"] = model_files
            except Exception as e:
                results["models"]["status"] = "error"
                results["models"]["error"] = str(e)
            
            # Determine overall status
            healthy_components = 0
            total_components = 3
            
            if results["database"]["status"] == "healthy":
                healthy_components += 1
            if results["api_services"].get("binance") == "healthy":
                healthy_components += 1
            if results["models"]["status"] == "healthy":
                healthy_components += 1
            
            if healthy_components == total_components:
                results["overall_status"] = "healthy"
            elif healthy_components > 0:
                results["overall_status"] = "degraded"
            else:
                results["overall_status"] = "unhealthy"
            
            logger.info(f"✅ Health check completed: {results['overall_status']}")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            results["overall_status"] = "error"
            results["error"] = str(e)
        
        return results
    
    async def run_full_automation(self) -> Dict[str, Any]:
        """Run all automation tasks in sequence"""
        logger.info("🚀 Starting full daily automation...")
        
        full_results = {
            "data_ingestion": {},
            "prediction_generation": {},
            "prediction_validation": {},
            "health_check": {},
            "timestamp": datetime.now().isoformat(),
            "total_duration": "",
            "overall_success": True
        }
        
        try:
            # Run all tasks
            full_results["data_ingestion"] = await self.run_data_ingestion()
            full_results["prediction_generation"] = await self.run_prediction_generation()
            full_results["prediction_validation"] = await self.run_prediction_validation()
            full_results["health_check"] = await self.run_health_check()
            
            # Check overall success
            all_successful = (
                full_results["data_ingestion"].get("success", False) and
                full_results["prediction_generation"].get("success", False) and
                full_results["prediction_validation"].get("success", False)
            )
            
            full_results["overall_success"] = all_successful
            
        except Exception as e:
            logger.error(f"❌ Full automation failed: {str(e)}")
            full_results["overall_success"] = False
            full_results["error"] = str(e)
        
        total_duration = datetime.now() - self.start_time
        full_results["total_duration"] = str(total_duration)
        
        logger.info(f"✅ Full automation completed in {total_duration}")
        return full_results

async def main():
    """Main function to handle command line arguments and run automation"""
    parser = argparse.ArgumentParser(description="Daily Automation for Crypto Prediction App")
    parser.add_argument("--data-ingestion", action="store_true", help="Run data ingestion only")
    parser.add_argument("--predictions", action="store_true", help="Run prediction generation only")
    parser.add_argument("--validation", action="store_true", help="Run prediction validation only")
    parser.add_argument("--health-check", action="store_true", help="Run health checks only")
    parser.add_argument("--full", action="store_true", help="Run all automation tasks")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    automation_manager = DailyAutomationManager()
    
    try:
        if args.data_ingestion:
            results = await automation_manager.run_data_ingestion()
            print(f"Data Ingestion Results: {results}")
            
        elif args.predictions:
            results = await automation_manager.run_prediction_generation()
            print(f"Prediction Generation Results: {results}")
            
        elif args.validation:
            results = await automation_manager.run_prediction_validation()
            print(f"Prediction Validation Results: {results}")
            
        elif args.health_check:
            results = await automation_manager.run_health_check()
            print(f"Health Check Results: {results}")
            
        elif args.full:
            results = await automation_manager.run_full_automation()
            print(f"Full Automation Results: {results}")
            
        else:
            print("Please specify a task to run. Use --help for options.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Automation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 