from fastapi.testclient import TestClient
from task2_fastapi.app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_ticket():
    response = client.post("/predict", json={"text": "I was charged twice"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Billing"
    assert "priority" in data
    assert "confidence" in data

def test_empty_text_is_rejected():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_missing_text_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_batch_bonus():
    response = client.post("/predict-batch", json={"tickets": [{"text": "My package is delayed"}]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1
