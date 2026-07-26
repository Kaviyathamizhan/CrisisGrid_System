"""
tests/test_api.py
Automated Pytest test suite for CrisisGrid REST API and WebSocket streaming endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data


def test_seeds_endpoint():
    response = client.get("/api/seeds")
    assert response.status_code == 200
    data = response.json()
    assert "seeds" in data
    assert isinstance(data["seeds"], list)
    assert 123 in data["seeds"]


def test_replay_endpoint():
    response = client.get("/api/replay?seed=123")
    assert response.status_code == 200
    data = response.json()
    assert data["seed"] == 123
    assert len(data["steps"]) == 51
    assert "metrics" in data
    assert "final_survival" in data["metrics"]


def test_comparison_endpoint():
    response = client.get("/api/comparison?seed=123")
    assert response.status_code == 200
    data = response.json()
    assert "trained" in data
    assert "random" in data
    assert "comparison" in data
    
    comp = data["comparison"]
    assert "survival_delta" in comp
    assert "population_saved_delta" in comp
    assert "policies_match" in comp
    assert "decision_similarity" in comp
    assert comp["policies_match"] is False


def test_simulate_endpoint_replay():
    payload = {"seed": 123, "mode": "replay"}
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["seed"] == 123
    assert len(data["steps"]) > 0


def test_websocket_replay_stream():
    with client.websocket_connect("/api/ws/simulate?seed=123&mode=replay") as websocket:
        # 1. Receive init frame
        init_frame = websocket.receive_json()
        assert init_frame["type"] == "init"
        assert init_frame["seed"] == 123

        # 2. Receive step frames
        step_count = 0
        while True:
            frame = websocket.receive_json()
            ftype = frame.get("type")
            if ftype == "step":
                step_count += 1
            elif ftype == "complete":
                assert "metrics" in frame
                break
            elif ftype == "error":
                pytest.fail(f"WebSocket received error: {frame.get('message')}")

        assert step_count == 51
