"""
SmartFault AI — API Tests
Run: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock the predictor so tests don't need trained models
MOCK_PREDICTION = {
    "machine_id": "MACHINE_001",
    "failure_probability": 0.78,
    "risk_level": "HIGH",
    "predicted_failure_in_hours": 5.3,
    "recommended_action": "Schedule maintenance within 6-12 hours.",
    "models_used": ["xgboost", "rf"],
    "confidence": 0.83,
    "timestamp": "2024-04-28T14:30:00",
}


@pytest.fixture
def client():
    with patch("api.main.get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.models = {"xgboost": MagicMock(), "rf": MagicMock()}
        mock_predictor.predict_single.return_value = MOCK_PREDICTION
        mock_get.return_value = mock_predictor
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "SmartFault AI Predictive Maintenance"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data


def test_predict_valid(client):
    payload = {
        "machine_id": "MACHINE_001",
        "temperature": 87.4,
        "vibration": 0.83,
        "pressure": 112.5,
        "rpm": 1450.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "failure_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert 0 <= data["failure_probability"] <= 1


def test_predict_invalid_temperature(client):
    payload = {
        "machine_id": "M1",
        "temperature": 999,  # out of range
        "vibration": 0.3,
        "pressure": 110,
        "rpm": 1440,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Validation error


def test_predict_batch(client):
    payload = {
        "readings": [
            {"machine_id": f"MACHINE_{i:03d}", "temperature": 70 + i, "vibration": 0.3, "pressure": 108, "rpm": 1440}
            for i in range(1, 4)
        ]
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "predictions" in data
    assert "high_risk_count" in data


def test_history_not_found(client):
    resp = client.get("/history/UNKNOWN_MACHINE")
    assert resp.status_code == 404


def test_machines_list(client):
    resp = client.get("/machines")
    assert resp.status_code == 200
    assert "machines" in resp.json()
