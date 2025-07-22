import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to sys.path to import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.main import app

client = TestClient(app)


def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "timestamp" in data
    assert "database_connected" in data


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data
    assert "timestamp" in data


def test_invalid_currency():
    """Test API with invalid currency"""
    response = client.get("/prices/INVALID")
    assert response.status_code == 400
    assert "Supported currencies" in response.json()["detail"]


def test_predict_endpoint():
    """Test the prediction endpoint (placeholder)"""
    response = client.post("/predict/BTC")
    assert response.status_code == 200
    data = response.json()
    assert data["currency"] == "BTC"
    assert "prediction" in data
    assert "confidence" in data 