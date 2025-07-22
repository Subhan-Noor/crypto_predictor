#!/usr/bin/env python3
"""
Daily Automation Script for Crypto Price Prediction App

This script handles the complete daily automation pipeline:
- Data ingestion (prices and sentiment)
- Daily predictions generation
- Health checks and monitoring
- Error handling and recovery

Usage:
    python daily_automation.py --full                    # Run complete pipeline
    python daily_automation.py --data-ingestion         # Data ingestion only
    python daily_automation.py --predictions           # Predictions only
    python daily_automation.py --health-check          # Health check only
    python daily_automation.py --status                # Check automation status
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.logger import logger
from scripts.data_ingestion import DataIngestionManager
from ml.prediction_pipeline import make_daily_predictions
from config import settings


class AutomationManager:
    """Manages daily automation tasks for the crypto prediction app"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": settings.environment,
            "tasks": {},
            "summary": {},
            "errors": []
        }
        
    async def run_data_ingestion(self) -> Dict[str, Any]:
        """Run data ingestion pipeline"""
        logger.info("🔄 Starting data ingestion...")
        
        try:
            ingestion_manager = DataIngestionManager()
            
            # Run data ingestion for latest day
            ingestion_results = ingestion_manager.run_full_ingestion(days_back=1)
            
            # Check if ingestion was successful
            if ingestion_results.get("summary", {}).get("status") == "completed":
                total_records = ingestion_results["summary"].get("total_price_records", 0)
                sentiment_records = ingestion_results["summary"].get("successful_sentiment_records", 0)
                
                logger.info(f"✅ Data ingestion completed: {total_records} price records, {sentiment_records} sentiment records")
                
                self.results["tasks"]["data_ingestion"] = {
                    "status": "success",
                    "records": {
                        "price_records": total_records,
                        "sentiment_records": sentiment_records
                    },
                    "details": ingestion_results
                }
                
                return {"status": "success", "results": ingestion_results}
            else:
                error_msg = ingestion_results.get("error", "Unknown error during ingestion")
                logger.error(f"❌ Data ingestion failed: {error_msg}")
                
                self.results["tasks"]["data_ingestion"] = {
                    "status": "failed",
                    "error": error_msg,
                    "details": ingestion_results
                }
                self.results["errors"].append(f"Data ingestion: {error_msg}")
                
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Data ingestion exception: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results["tasks"]["data_ingestion"] = {
                "status": "failed",
                "error": error_msg
            }
            self.results["errors"].append(error_msg)
            
            return {"status": "failed", "error": error_msg}
    
    async def run_predictions(self) -> Dict[str, Any]:
        """Run daily predictions pipeline"""
        logger.info("🤖 Starting daily predictions...")
        
        try:
            # Generate predictions for both BTC and ETH
            prediction_results = await make_daily_predictions()
            
            successful_predictions = 0
            failed_predictions = 0
            prediction_details = {}
            
            for currency, result in prediction_results.items():
                if "error" in result:
                    failed_predictions += 1
                    prediction_details[currency] = {
                        "status": "failed",
                        "error": result["error"]
                    }
                    logger.error(f"❌ Prediction failed for {currency}: {result['error']}")
                else:
                    successful_predictions += 1
                    prediction_details[currency] = {
                        "status": "success",
                        "prediction": result["predicted_direction"],
                        "confidence": result["confidence_score"],
                        "model_version": result.get("model_version", "unknown")
                    }
                    logger.info(f"✅ Prediction completed for {currency}: {result['predicted_direction']} (confidence: {result['confidence_score']:.3f})")
            
            self.results["tasks"]["predictions"] = {
                "status": "success" if successful_predictions > 0 else "failed",
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "details": prediction_details
            }
            
            if failed_predictions > 0:
                error_msg = f"Some predictions failed: {failed_predictions}/{len(prediction_results)}"
                self.results["errors"].append(error_msg)
            
            logger.info(f"📊 Predictions summary: {successful_predictions} successful, {failed_predictions} failed")
            
            return {
                "status": "success" if successful_predictions > 0 else "failed",
                "successful": successful_predictions,
                "failed": failed_predictions,
                "results": prediction_results
            }
            
        except Exception as e:
            error_msg = f"Predictions exception: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results["tasks"]["predictions"] = {
                "status": "failed",
                "error": error_msg
            }
            self.results["errors"].append(error_msg)
            
            return {"status": "failed", "error": error_msg}
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run system health checks"""
        logger.info("🔍 Running health checks...")
        
        health_status = {
            "database": False,
            "data_availability": False,
            "models_available": False,
            "recent_predictions": False
        }
        
        try:
            # Import db_manager at the top of the try block
            from app.database import db_manager
            
            # Check database connection
            if db_manager.is_connected() and db_manager.test_connection():
                health_status["database"] = True
                logger.info("✅ Database connection healthy")
            else:
                logger.error("❌ Database connection failed")
            
            # Check data availability (recent price data)
            if health_status["database"]:
                try:
                    recent_data = db_manager.get_latest_prices("BTC", limit=7)
                    if recent_data and len(recent_data) >= 3:  # At least 3 days of recent data
                        health_status["data_availability"] = True
                        logger.info(f"✅ Data availability healthy: {len(recent_data)} recent records")
                    else:
                        logger.warning("⚠️ Limited recent data available")
                except Exception as e:
                    logger.error(f"❌ Data availability check failed: {e}")
            
            # Check if models are available
            try:
                models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
                if os.path.exists(models_dir) and os.listdir(models_dir):
                    health_status["models_available"] = True
                    logger.info("✅ ML models available")
                else:
                    logger.warning("⚠️ No ML models found")
            except Exception as e:
                logger.error(f"❌ Model availability check failed: {e}")
            
            # Check for recent predictions
            if health_status["database"]:
                try:
                    recent_btc_data = db_manager.client.table("predictions").select("*").eq("currency", "BTC").order("prediction_date", desc=True).limit(3).execute()
                    if recent_btc_data.data and len(recent_btc_data.data) > 0:
                        health_status["recent_predictions"] = True
                        logger.info(f"✅ Recent predictions available: {len(recent_btc_data.data)} records")
                    else:
                        logger.warning("⚠️ No recent predictions found")
                except Exception as e:
                    logger.error(f"❌ Recent predictions check failed: {e}")
            
            # Overall health score
            healthy_components = sum(health_status.values())
            total_components = len(health_status)
            health_score = healthy_components / total_components
            
            overall_status = "healthy" if health_score >= 0.75 else "degraded" if health_score >= 0.5 else "unhealthy"
            
            self.results["tasks"]["health_check"] = {
                "status": "success",
                "overall_status": overall_status,
                "health_score": health_score,
                "components": health_status
            }
            
            logger.info(f"🏥 Health check completed: {overall_status} ({health_score:.2f})")
            
            return {
                "status": "success",
                "overall_status": overall_status,
                "health_score": health_score,
                "components": health_status
            }
            
        except Exception as e:
            error_msg = f"Health check exception: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results["tasks"]["health_check"] = {
                "status": "failed",
                "error": error_msg
            }
            self.results["errors"].append(error_msg)
            
            return {"status": "failed", "error": error_msg}
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete daily automation pipeline"""
        logger.info("🚀 Starting full daily automation pipeline...")
        
        # Step 1: Data ingestion
        ingestion_result = await self.run_data_ingestion()
        
        # Step 2: Generate predictions (run even if ingestion partially failed)
        prediction_result = await self.run_predictions()
        
        # Step 3: Health check
        health_result = await self.run_health_check()
        
        # Generate summary
        successful_tasks = sum(1 for task in self.results["tasks"].values() if task.get("status") == "success")
        total_tasks = len(self.results["tasks"])
        
        self.results["summary"] = {
            "successful_tasks": successful_tasks,
            "total_tasks": total_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "overall_status": "success" if successful_tasks == total_tasks else "partial" if successful_tasks > 0 else "failed",
            "duration": (datetime.now() - datetime.fromisoformat(self.results["timestamp"])).total_seconds()
        }
        
        # Log final summary
        status = self.results["summary"]["overall_status"]
        duration = self.results["summary"]["duration"]
        
        if status == "success":
            logger.info(f"🎉 Daily automation completed successfully in {duration:.1f}s")
        elif status == "partial":
            logger.warning(f"⚠️ Daily automation partially completed in {duration:.1f}s")
        else:
            logger.error(f"❌ Daily automation failed in {duration:.1f}s")
        
        return self.results
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save automation results to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"automation_results_{timestamp}.json"
        
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
    """Main function for daily automation"""
    parser = argparse.ArgumentParser(description="Crypto Prediction Daily Automation")
    
    # Automation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--full", action="store_true", help="Run complete automation pipeline")
    mode_group.add_argument("--data-ingestion", action="store_true", help="Run data ingestion only")
    mode_group.add_argument("--predictions", action="store_true", help="Run predictions only")
    mode_group.add_argument("--health-check", action="store_true", help="Run health check only")
    mode_group.add_argument("--status", action="store_true", help="Check automation status")
    
    # Options
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Initialize automation manager
    automation = AutomationManager()
    
    try:
        # Run the specified automation task
        if args.full:
            results = await automation.run_full_pipeline()
        elif args.data_ingestion:
            results = await automation.run_data_ingestion()
        elif args.predictions:
            results = await automation.run_predictions()
        elif args.health_check:
            results = await automation.run_health_check()
        elif args.status:
            results = await automation.run_health_check()
            print("\n=== Automation Status ===")
            if "health_check" in automation.results["tasks"]:
                health = automation.results["tasks"]["health_check"]
                print(f"Overall Status: {health.get('overall_status', 'unknown')}")
                print(f"Health Score: {health.get('health_score', 0):.2f}")
                print("\nComponents:")
                for component, status in health.get("components", {}).items():
                    print(f"  {component}: {'✅' if status else '❌'}")
            return
        
        # Save results if requested
        if args.save_results:
            automation.save_results(args.output)
        
        # Print summary
        print("\n=== Automation Summary ===")
        if "summary" in automation.results:
            summary = automation.results["summary"]
            print(f"Status: {summary.get('overall_status', 'unknown')}")
            print(f"Tasks: {summary.get('successful_tasks', 0)}/{summary.get('total_tasks', 0)} successful")
            print(f"Duration: {summary.get('duration', 0):.1f}s")
        else:
            # Handle case where only individual task was run
            if "health_check" in automation.results.get("tasks", {}):
                health_task = automation.results["tasks"]["health_check"]
                print(f"Status: {health_task.get('overall_status', 'unknown')}")
                print(f"Health Score: {health_task.get('health_score', 0):.2f}")

        
        if automation.results["errors"]:
            print("\nErrors:")
            for error in automation.results["errors"]:
                print(f"  ❌ {error}")
        
        # Exit with appropriate code
        if "summary" in automation.results and "overall_status" in automation.results["summary"]:
            exit_code = 0 if automation.results["summary"]["overall_status"] in ["success", "partial"] else 1
        elif "health_check" in automation.results.get("tasks", {}):
            # Handle health check only
            health_result = automation.results["tasks"]["health_check"]
            # For health checks, consider "degraded" as acceptable
            if health_result.get("status") == "success" and health_result.get("overall_status") in ["healthy", "degraded"]:
                exit_code = 0
            else:
                exit_code = 1
        else:
            # Handle other individual tasks
            exit_code = 0 if results.get("status") == "success" else 1
        
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error in automation: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 