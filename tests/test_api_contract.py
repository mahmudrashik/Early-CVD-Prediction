from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_page_renders():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert "Early CVD Prediction" in response.text
