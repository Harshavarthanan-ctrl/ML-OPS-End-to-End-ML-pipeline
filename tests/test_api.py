from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction():
    sample = [0.1] * 30
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
