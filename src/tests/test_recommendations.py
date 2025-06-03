import pytest
from fastapi.testclient import TestClient
from src.serving.serve import app, RecommendationRequest

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_recommendations_collaborative():
    """Test collaborative filtering recommendations."""
    request_data = {
        "user_id": 1,
        "num_recommendations": 5,
        "use_content_based": False
    }
    response = client.post("/recommend", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "model_used" in data
    assert data["model_used"] == "collaborative_filtering"
    assert len(data["recommendations"]) == 5

def test_recommendations_content_based():
    """Test content-based recommendations."""
    request_data = {
        "user_id": 1,
        "num_recommendations": 5,
        "use_content_based": True
    }
    response = client.post("/recommend", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "model_used" in data
    assert data["model_used"] == "content_based"
    assert len(data["recommendations"]) == 5

def test_invalid_user_id():
    """Test invalid user ID handling."""
    request_data = {
        "user_id": -1,
        "num_recommendations": 5,
        "use_content_based": False
    }
    response = client.post("/recommend", json=request_data)
    assert response.status_code == 500 