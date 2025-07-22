import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.main import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test the root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "timestamp" in data
        assert "database_connected" in data
    
    def test_health_endpoint(self):
        """Test the detailed health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

class TestPriceEndpoints:
    """Test price data endpoints"""
    
    @patch('app.main.price_fetcher.get_historical_prices')
    def test_get_crypto_prices_btc(self, mock_get_prices):
        """Test getting BTC price data"""
        # Mock the price fetcher response
        mock_data = [
            {
                "open_time": 1640995200000,
                "open": "50000.0",
                "high": "51000.0",
                "low": "49000.0",
                "close": "50500.0",
                "volume": "1000.0",
                "close_time": 1641081600000
            }
        ]
        mock_get_prices.return_value = mock_data
        
        response = client.get("/prices/BTC")
        assert response.status_code == 200
        data = response.json()
        assert data["currency"] == "BTC"
        assert "data" in data
        assert "count" in data
    
    @patch('app.main.price_fetcher.get_historical_prices')
    def test_get_crypto_prices_eth(self, mock_get_prices):
        """Test getting ETH price data"""
        mock_data = [
            {
                "open_time": 1640995200000,
                "open": "4000.0",
                "high": "4100.0",
                "low": "3900.0",
                "close": "4050.0",
                "volume": "500.0",
                "close_time": 1641081600000
            }
        ]
        mock_get_prices.return_value = mock_data
        
        response = client.get("/prices/ETH")
        assert response.status_code == 200
        data = response.json()
        assert data["currency"] == "ETH"
        assert "data" in data
        assert "count" in data
    
    def test_get_crypto_prices_invalid_currency(self):
        """Test getting price data for invalid currency"""
        response = client.get("/prices/INVALID")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

class TestSentimentEndpoints:
    """Test sentiment data endpoints"""
    
    @patch('app.main.twitter_scraper.get_crypto_sentiment')
    @patch('app.main.reddit_scraper.get_crypto_sentiment')
    def test_get_crypto_sentiment_btc(self, mock_reddit_sentiment, mock_twitter_sentiment):
        """Test getting BTC sentiment data"""
        mock_twitter_sentiment.return_value = 0.5
        mock_reddit_sentiment.return_value = 0.3
        
        response = client.get("/sentiment/BTC")
        assert response.status_code == 200
        data = response.json()
        assert data["currency"] == "BTC"
        assert "twitter_sentiment" in data
        assert "reddit_sentiment" in data
    
    @patch('app.main.twitter_scraper.get_crypto_sentiment')
    @patch('app.main.reddit_scraper.get_crypto_sentiment')
    def test_get_crypto_sentiment_eth(self, mock_reddit_sentiment, mock_twitter_sentiment):
        """Test getting ETH sentiment data"""
        mock_twitter_sentiment.return_value = 0.2
        mock_reddit_sentiment.return_value = 0.4
        
        response = client.get("/sentiment/ETH")
        assert response.status_code == 200
        data = response.json()
        assert data["currency"] == "ETH"
        assert "twitter_sentiment" in data
        assert "reddit_sentiment" in data
    
    def test_get_crypto_sentiment_invalid_currency(self):
        """Test getting sentiment data for invalid currency"""
        response = client.get("/sentiment/INVALID")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

class TestCurrentPricesEndpoint:
    """Test current prices endpoint"""
    
    @patch('app.main.price_fetcher.get_current_price')
    def test_get_current_prices(self, mock_get_current_price):
        """Test getting current prices for all currencies"""
        mock_get_current_price.side_effect = [
            {"symbol": "BTCUSDT", "price": "50000.0"},
            {"symbol": "ETHUSDT", "price": "4000.0"}
        ]
        
        response = client.get("/current_prices")
        assert response.status_code == 200
        data = response.json()
        assert "BTC" in data
        assert "ETH" in data
        assert "timestamp" in data

class TestDataStatusEndpoint:
    """Test data status endpoint"""
    
    @patch('app.main.db_manager.get_client')
    def test_get_data_status(self, mock_get_client):
        """Test getting data status"""
        # Mock the database client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Mock the database responses
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.count = 100
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [{"date": "2024-01-01"}]
        
        response = client.get("/data_status")
        assert response.status_code == 200
        data = response.json()
        assert "BTC" in data
        assert "ETH" in data

class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_endpoint_placeholder(self):
        """Test the prediction endpoint (placeholder for Stage 3)"""
        response = client.post("/predict/BTC", json={"prediction_horizon": 7})
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "stage" in data

if __name__ == "__main__":
    pytest.main([__file__]) 