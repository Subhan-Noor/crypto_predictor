"""
WebSocket Service for Real-time Crypto Price Updates

This module provides:
- Real-time price updates
- Live prediction broadcasts
- Sentiment data streaming
- Client connection management
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.models.api_models import (
    WebSocketMessage, WebSocketMessageType, SubscriptionRequest,
    PriceUpdateMessage, PredictionUpdateMessage, SentimentUpdateMessage
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Store active connections with metadata
        self.active_connections: Dict[str, Dict] = {}
        # Track subscriptions by channel
        self.subscriptions: Dict[str, Set[str]] = {
            "prices": set(),
            "predictions": set(),
            "sentiment": set(),
            "all": set()
        }
        # Track currency-specific subscriptions
        self.currency_subscriptions: Dict[str, Dict[str, Set[str]]] = {
            "BTC": {"prices": set(), "predictions": set(), "sentiment": set()},
            "ETH": {"prices": set(), "predictions": set(), "sentiment": set()}
        }
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection info
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "connected_at": datetime.now(),
            "subscriptions": set(),
            "currencies": set(),
            "last_heartbeat": datetime.now()
        }
        
        logger.info(f"WebSocket client {connection_id} connected")
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type=WebSocketMessageType.HEARTBEAT,
            data={
                "message": "Connected to Crypto Prediction WebSocket",
                "connection_id": connection_id,
                "available_channels": ["prices", "predictions", "sentiment"],
                "available_currencies": ["BTC", "ETH"]
            }
        )
        await self._send_to_connection(connection_id, welcome_msg.dict())
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove connection and clean up subscriptions"""
        if connection_id in self.active_connections:
            # Remove from all subscriptions
            connection = self.active_connections[connection_id]
            
            for channel in connection.get("subscriptions", set()):
                self.subscriptions.get(channel, set()).discard(connection_id)
            
            for currency in connection.get("currencies", set()):
                for channel in ["prices", "predictions", "sentiment"]:
                    self.currency_subscriptions.get(currency, {}).get(channel, set()).discard(connection_id)
            
            # Remove connection
            del self.active_connections[connection_id]
            
            logger.info(f"WebSocket client {connection_id} disconnected")
    
    async def _send_to_connection(self, connection_id: str, message: Dict):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]["websocket"]
                await websocket.send_text(json.dumps(message, default=str))
                return True
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
                return False
        return False
    
    async def broadcast_to_channel(self, channel: str, message: Dict, currency: Optional[str] = None):
        """Broadcast message to all subscribers of a channel"""
        # Get subscribers for this channel
        subscribers = set()
        
        if currency:
            # Add currency-specific subscribers
            subscribers.update(
                self.currency_subscriptions.get(currency, {}).get(channel, set())
            )
        else:
            # Add general channel subscribers
            subscribers.update(self.subscriptions.get(channel, set()))
        
        # Add subscribers to "all" channel
        subscribers.update(self.subscriptions.get("all", set()))
        
        # Send to all subscribers
        failed_connections = []
        success_count = 0
        
        for connection_id in subscribers:
            success = await self._send_to_connection(connection_id, message)
            if success:
                success_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            self.disconnect(connection_id)
        
        logger.debug(f"Broadcasted to {success_count} clients on channel {channel}" + 
                    (f" for {currency}" if currency else ""))
        
        return success_count
    
    async def handle_subscription(self, connection_id: str, subscription: SubscriptionRequest):
        """Handle subscription request from client"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        for channel in subscription.channels:
            if channel in self.subscriptions:
                # Add to general channel subscription
                self.subscriptions[channel].add(connection_id)
                connection["subscriptions"].add(channel)
                
                # Add to currency-specific subscriptions if specified
                if subscription.currencies:
                    for currency in subscription.currencies:
                        if currency in ["BTC", "ETH"]:
                            self.currency_subscriptions[currency][channel].add(connection_id)
                            connection["currencies"].add(currency)
        
        # Send confirmation
        confirmation = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={
                "status": "subscribed",
                "channels": subscription.channels,
                "currencies": subscription.currencies or []
            }
        )
        await self._send_to_connection(connection_id, confirmation.dict())
        
        logger.info(f"Client {connection_id} subscribed to {subscription.channels}")
    
    async def handle_unsubscription(self, connection_id: str, channels: List[str], currencies: Optional[List[str]] = None):
        """Handle unsubscription request"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        for channel in channels:
            # Remove from general subscriptions
            self.subscriptions.get(channel, set()).discard(connection_id)
            connection["subscriptions"].discard(channel)
            
            # Remove from currency-specific subscriptions
            if currencies:
                for currency in currencies:
                    if currency in ["BTC", "ETH"]:
                        self.currency_subscriptions[currency][channel].discard(connection_id)
                        # Remove currency from connection if no more subscriptions
                        has_currency_subscriptions = any(
                            connection_id in self.currency_subscriptions[currency][ch]
                            for ch in ["prices", "predictions", "sentiment"]
                        )
                        if not has_currency_subscriptions:
                            connection["currencies"].discard(currency)
        
        # Send confirmation
        confirmation = WebSocketMessage(
            type=WebSocketMessageType.UNSUBSCRIBE,
            data={
                "status": "unsubscribed",
                "channels": channels,
                "currencies": currencies or []
            }
        )
        await self._send_to_connection(connection_id, confirmation.dict())
    
    async def send_heartbeat(self):
        """Send heartbeat to all connections"""
        heartbeat_msg = WebSocketMessage(
            type=WebSocketMessageType.HEARTBEAT,
            data={"timestamp": datetime.now().isoformat()}
        )
        
        failed_connections = []
        for connection_id in list(self.active_connections.keys()):
            success = await self._send_to_connection(connection_id, heartbeat_msg.dict())
            if success:
                self.active_connections[connection_id]["last_heartbeat"] = datetime.now()
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            self.disconnect(connection_id)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        total_connections = len(self.active_connections)
        subscription_stats = {
            channel: len(subscribers) 
            for channel, subscribers in self.subscriptions.items()
        }
        
        currency_stats = {}
        for currency, channels in self.currency_subscriptions.items():
            currency_stats[currency] = {
                channel: len(subscribers)
                for channel, subscribers in channels.items()
            }
        
        return {
            "total_connections": total_connections,
            "subscription_stats": subscription_stats,
            "currency_stats": currency_stats,
            "uptime": (datetime.now() - min(
                (conn["connected_at"] for conn in self.active_connections.values()),
                default=datetime.now()
            )).total_seconds() if total_connections > 0 else 0
        }


class WebSocketService:
    """High-level WebSocket service for crypto data streaming"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.is_running = False
        self.heartbeat_task = None
    
    async def start_service(self):
        """Start the WebSocket service"""
        if not self.is_running:
            self.is_running = True
            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocket service started")
    
    async def stop_service(self):
        """Stop the WebSocket service"""
        self.is_running = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        logger.info("WebSocket service stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                await self.connection_manager.send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def handle_client_connection(self, websocket: WebSocket):
        """Handle new client connection"""
        connection_id = await self.connection_manager.connect(websocket)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    subscription = SubscriptionRequest(**message)
                    await self.connection_manager.handle_subscription(connection_id, subscription)
                
                elif message.get("type") == "unsubscribe":
                    channels = message.get("channels", [])
                    currencies = message.get("currencies")
                    await self.connection_manager.handle_unsubscription(connection_id, channels, currencies)
                
                elif message.get("type") == "heartbeat":
                    # Update last heartbeat
                    if connection_id in self.connection_manager.active_connections:
                        self.connection_manager.active_connections[connection_id]["last_heartbeat"] = datetime.now()
                
        except WebSocketDisconnect:
            self.connection_manager.disconnect(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
            self.connection_manager.disconnect(connection_id)
    
    async def broadcast_price_update(self, currency: str, price_data: Dict):
        """Broadcast real-time price update"""
        message = PriceUpdateMessage(
            currency=currency,
            price_data=price_data
        )
        
        await self.connection_manager.broadcast_to_channel(
            "prices", 
            message.dict(), 
            currency
        )
    
    async def broadcast_prediction_update(self, currency: str, prediction: Dict):
        """Broadcast new prediction"""
        message = PredictionUpdateMessage(
            currency=currency,
            prediction=prediction
        )
        
        await self.connection_manager.broadcast_to_channel(
            "predictions", 
            message.dict(), 
            currency
        )
    
    async def broadcast_sentiment_update(self, currency: str, sentiment_data: Dict):
        """Broadcast sentiment data update"""
        message = SentimentUpdateMessage(
            currency=currency,
            sentiment_data=sentiment_data
        )
        
        await self.connection_manager.broadcast_to_channel(
            "sentiment", 
            message.dict(), 
            currency
        )
    
    def get_stats(self) -> Dict:
        """Get WebSocket service statistics"""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "connections": self.connection_manager.get_connection_stats()
        }


# Global WebSocket service instance
websocket_service = WebSocketService() 