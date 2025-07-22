"""
Background Tasks Service for Crypto Price Prediction API

This module provides:
- Model retraining tasks
- Data ingestion tasks
- Cache warming tasks
- Analytics computation tasks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.models.api_models import BackgroundTask, TaskStatus
from app.services.cache_service import cache_service
from app.services.websocket_service import websocket_service

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """Manages background tasks for the API"""
    
    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    def create_task(self, task_type: str, **kwargs) -> str:
        """Create a new background task"""
        task_id = str(uuid.uuid4())
        
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            **kwargs
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created background task {task_id} of type {task_type}")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, task_type: Optional[str] = None, status: Optional[TaskStatus] = None) -> list:
        """List tasks with optional filtering"""
        tasks = list(self.tasks.values())
        
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    async def start_task(self, task_id: str) -> bool:
        """Start executing a background task"""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not in PENDING status")
            return False
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Start the appropriate task function
        if task.task_type == "model_retraining":
            coroutine = self._run_model_retraining(task_id)
        elif task.task_type == "data_ingestion":
            coroutine = self._run_data_ingestion(task_id)
        elif task.task_type == "cache_warming":
            coroutine = self._run_cache_warming(task_id)
        elif task.task_type == "analytics_computation":
            coroutine = self._run_analytics_computation(task_id)
        else:
            logger.error(f"Unknown task type: {task.task_type}")
            task.status = TaskStatus.FAILED
            task.error = f"Unknown task type: {task.task_type}"
            return False
        
        # Create and store the asyncio task
        async_task = asyncio.create_task(coroutine)
        self.running_tasks[task_id] = async_task
        
        logger.info(f"Started background task {task_id}")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            async_task = self.running_tasks[task_id]
            async_task.cancel()
            
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED
                self.tasks[task_id].completed_at = datetime.now()
            
            del self.running_tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        return len(tasks_to_remove)
    
    # Task implementations
    
    async def _run_model_retraining(self, task_id: str):
        """Run model retraining task"""
        task = self.tasks[task_id]
        
        try:
            task.progress = 0
            logger.info(f"Starting model retraining for task {task_id}")
            
            # Import ML components
            from ...ml.model_trainer import CryptoModelTrainer
            from ...ml.data_preprocessor import CryptoDataPreprocessor
            from ...ml.feature_engineering import CryptoFeatureEngineer
            
            # Initialize components
            preprocessor = CryptoDataPreprocessor()
            feature_engineer = CryptoFeatureEngineer()
            trainer = CryptoModelTrainer()
            
            currencies = ["BTC", "ETH"]
            total_currencies = len(currencies)
            
            results = {}
            
            for i, currency in enumerate(currencies):
                logger.info(f"Training models for {currency}")
                task.progress = (i / total_currencies) * 100
                
                try:
                    # Load and preprocess data
                    data = await preprocessor.load_data(currency)
                    if not data['prices'].empty and not data['sentiment'].empty:
                        
                        # Prepare ML dataset
                        processed_data = preprocessor.prepare_ml_dataset(
                            data['prices'], data['sentiment']
                        )
                        
                        if processed_data is not None:
                            # Engineer features
                            features_df = feature_engineer.engineer_features(
                                processed_data['prices'], processed_data['sentiment']
                            )
                            
                            # Prepare train/test data
                            X, y = feature_engineer.prepare_features_and_labels(features_df)
                            X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
                            
                            # Train models
                            training_results = trainer.train_all_models(X_train, X_test, y_train, y_test)
                            results[currency] = training_results
                            
                            logger.info(f"Successfully trained models for {currency}")
                        else:
                            results[currency] = {"error": "No processed data available"}
                    else:
                        results[currency] = {"error": "No raw data available"}
                        
                except Exception as e:
                    logger.error(f"Error training models for {currency}: {e}")
                    results[currency] = {"error": str(e)}
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100
            task.result = {
                "message": "Model retraining completed",
                "currencies_processed": len(currencies),
                "results": results,
                "training_timestamp": datetime.now().isoformat()
            }
            
            # Invalidate prediction cache since models are updated
            for currency in currencies:
                cache_service.invalidate_currency_cache(currency)
            
            logger.info(f"Model retraining task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Model retraining task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
        
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _run_data_ingestion(self, task_id: str):
        """Run data ingestion task"""
        task = self.tasks[task_id]
        
        try:
            task.progress = 0
            logger.info(f"Starting data ingestion for task {task_id}")
            
            # Import data ingestion components
            from ...scripts.data_ingestion import DataIngestionManager
            
            ingestion_manager = DataIngestionManager()
            
            # Fetch price data
            task.progress = 25
            price_results = ingestion_manager.fetch_and_store_prices(days=1)  # Fetch latest day
            
            # Fetch sentiment data
            task.progress = 75
            sentiment_results = ingestion_manager.fetch_and_store_sentiment()
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100
            task.result = {
                "message": "Data ingestion completed",
                "price_results": price_results,
                "sentiment_results": sentiment_results,
                "ingestion_timestamp": datetime.now().isoformat()
            }
            
            # Invalidate relevant cache
            cache_service.delete_pattern("prices:*")
            cache_service.delete_pattern("sentiment:*")
            cache_service.delete("current_prices")
            
            logger.info(f"Data ingestion task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Data ingestion task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
        
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _run_cache_warming(self, task_id: str):
        """Run cache warming task"""
        task = self.tasks[task_id]
        
        try:
            task.progress = 0
            logger.info(f"Starting cache warming for task {task_id}")
            
            from app.database import db_manager
            
            currencies = ["BTC", "ETH"]
            time_periods = [7, 30, 60]
            
            total_operations = len(currencies) * len(time_periods) * 2  # prices + sentiment
            completed_operations = 0
            
            for currency in currencies:
                for days in time_periods:
                    # Warm price cache
                    try:
                        records = await db_manager.get_records('crypto_prices', {'currency': currency})
                        # Process and cache the data (simplified)
                        cache_key = f"prices_{currency}_{days}"
                        cache_service.set(cache_key, {"cached_at": datetime.now().isoformat()}, ttl=3600)
                        completed_operations += 1
                        task.progress = (completed_operations / total_operations) * 100
                    except Exception as e:
                        logger.warning(f"Failed to warm price cache for {currency}, {days} days: {e}")
                    
                    # Warm sentiment cache
                    try:
                        records = await db_manager.get_records('crypto_sentiment', {'currency': currency})
                        cache_key = f"sentiment_{currency}_{days}"
                        cache_service.set(cache_key, {"cached_at": datetime.now().isoformat()}, ttl=3600)
                        completed_operations += 1
                        task.progress = (completed_operations / total_operations) * 100
                    except Exception as e:
                        logger.warning(f"Failed to warm sentiment cache for {currency}, {days} days: {e}")
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100
            task.result = {
                "message": "Cache warming completed",
                "operations_completed": completed_operations,
                "warming_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Cache warming task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Cache warming task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
        
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _run_analytics_computation(self, task_id: str):
        """Run analytics computation task"""
        task = self.tasks[task_id]
        
        try:
            task.progress = 0
            logger.info(f"Starting analytics computation for task {task_id}")
            
            from app.database import db_manager
            
            # Compute various analytics
            analytics_results = {}
            
            # Database statistics
            task.progress = 20
            try:
                btc_price_count = len(await db_manager.get_records('crypto_prices', {'currency': 'BTC'}))
                eth_price_count = len(await db_manager.get_records('crypto_prices', {'currency': 'ETH'}))
                btc_sentiment_count = len(await db_manager.get_records('crypto_sentiment', {'currency': 'BTC'}))
                eth_sentiment_count = len(await db_manager.get_records('crypto_sentiment', {'currency': 'ETH'}))
                
                analytics_results['database_stats'] = {
                    'btc_price_records': btc_price_count,
                    'eth_price_records': eth_price_count,
                    'btc_sentiment_records': btc_sentiment_count,
                    'eth_sentiment_records': eth_sentiment_count,
                    'total_records': btc_price_count + eth_price_count + btc_sentiment_count + eth_sentiment_count
                }
            except Exception as e:
                analytics_results['database_stats'] = {'error': str(e)}
            
            # Cache statistics
            task.progress = 50
            try:
                cache_stats = cache_service.get_stats()
                analytics_results['cache_stats'] = cache_stats
            except Exception as e:
                analytics_results['cache_stats'] = {'error': str(e)}
            
            # WebSocket statistics
            task.progress = 75
            try:
                websocket_stats = websocket_service.get_stats()
                analytics_results['websocket_stats'] = websocket_stats
            except Exception as e:
                analytics_results['websocket_stats'] = {'error': str(e)}
            
            # System health summary
            task.progress = 90
            analytics_results['system_health'] = {
                'database_connected': True,  # We got here, so it's connected
                'cache_available': cache_service.is_available(),
                'websocket_running': websocket_service.is_running,
                'analytics_computed_at': datetime.now().isoformat()
            }
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100
            task.result = {
                "message": "Analytics computation completed",
                "analytics": analytics_results,
                "computation_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Analytics computation task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Analytics computation task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
        
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]


# Global background task service instance
background_task_service = BackgroundTaskService() 