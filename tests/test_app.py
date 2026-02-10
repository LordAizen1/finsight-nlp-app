import json


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_analyze_valid_text(client):
    response = client.post(
        "/analyze",
        data=json.dumps({"text": "AAPL stock rose 5% today."}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "html" in data
    assert "legend" in data
    assert "sentiment_score" in data
    assert "sentiment_label" in data


def test_analyze_missing_text_returns_400(client):
    response = client.post(
        "/analyze",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_analyze_empty_text_returns_400(client):
    response = client.post(
        "/analyze",
        data=json.dumps({"text": "   "}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_analyze_no_json_body_returns_400(client):
    response = client.post("/analyze", content_type="application/json")
    assert response.status_code == 400
