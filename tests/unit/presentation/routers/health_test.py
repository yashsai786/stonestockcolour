from fastapi.testclient import TestClient
from src.presentation.api.main import app

def test_health_check_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["pipeline"] == "active"
