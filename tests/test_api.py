"""Exercise HTTP lifecycle, privacy and failure boundaries without live lookups."""

import json
import logging
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from phishlens.features.email_text import clean_text
from phishlens.predict import Prediction


class FakePredictor:
    def __init__(self):
        self.offline = {"metadata": {"version": "offline-test"}}
        self.online = {"metadata": {"version": "online-pilot", "experimental": True}}
        self.email = {"metadata": {"version": "email-test"}}

    def predict_url(self, url, mode):
        return Prediction(
            80,
            "phishing",
            0.8,
            mode,
            "online-pilot" if mode == "online" else "offline-test",
            [
                {
                    "name": "url_length",
                    "value": len(url),
                    "contribution": 0.2,
                    "direction": "raises_risk",
                    "description": "URL length",
                }
            ],
        )

    def predict_email(self, text):
        cleaned = clean_text(text)
        return Prediction(
            20,
            "legitimate",
            0.2,
            "offline",
            "email-test",
            [],
            [
                {
                    "term": cleaned.split()[0],
                    "start": 0,
                    "end": len(cleaned.split()[0]),
                    "direction": "lowers_risk",
                    "contribution": -0.3,
                }
            ],
            cleaned,
        )


@pytest.fixture
def model():
    return FakePredictor()


@pytest.fixture
def client(model):
    with TestClient(create_app(predictor_factory=lambda _: model)) as client:
        yield client


def test_startup_once_and_health(model):
    factory = Mock(return_value=model)
    with TestClient(create_app(predictor_factory=factory)) as client:
        for _ in range(2):
            response = client.get("/health")
            assert response.json()["status"] == "ok"
            assert response.json()["models"]["url_online"]["experimental"]
            assert response.json()["models"]["email"]["version"] == "email-test"
            assert (
                client.post("/predict/url", json={"url": "https://python.org"}).status_code == 200
            )
    factory.assert_called_once()


def test_url_email_contract(client):
    result = client.post("/predict/url", json={"url": "https://python.org"})
    assert result.status_code == 200
    assert result.json()["risk_score"] == 80
    assert result.json()["top_features"][0]["direction"] == "raises_risk"
    assert result.headers["cache-control"] == "no-store"
    assert len(result.headers["x-request-id"]) == 32
    email = client.post("/predict/email", json={"text": "<b>Hello</b> meeting tomorrow"}).json()
    span = email["highlights"][0]
    assert email["cleaned_text"][span["start"] : span["end"]] == span["term"]


@pytest.mark.parametrize(
    "path,body",
    [
        ("url", {}),
        ("url", {"url": 42}),
        ("url", {"url": "bad url"}),
        ("url", {"url": "https://python.org", "mode": "other"}),
        ("url", {"url": "https://python.org", "extra": "secret"}),
        ("url", {"url": "https://python.org/" + "x" * 8192}),
        ("email", {"text": " "}),
        ("email", {"text": "<script>secret</script>"}),
        ("email", {"text": "x" * 100001}),
        ("email", {"text": ["secret"]}),
    ],
)
def test_validation(client, path, body):
    assert client.post(f"/predict/{path}", json=body).status_code == 422


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://[::1]", "http://example.com"])
def test_unsafe_online_rejected_before_model(client, model, url):
    model.predict_url = Mock(side_effect=AssertionError("must not run"))
    assert client.post("/predict/url", json={"url": url, "mode": "online"}).status_code == 422
    model.predict_url.assert_not_called()


def test_online_and_fallback(client, model):
    assert (
        client.post("/predict/url", json={"url": "python.org", "mode": "online"}).json()[
            "model_version"
        ]
        == "online-pilot"
    )
    offline = model.predict_url("python.org", "offline")
    model.predict_url = Mock(
        return_value=replace(offline, mode="offline_fallback", fallback_reason="lookup_timeout")
    )
    result = client.post("/predict/url", json={"url": "python.org", "mode": "online"}).json()
    assert result["mode"] == "offline_fallback"
    assert result["fallback_reason"] == "lookup_timeout"
    assert result["model_version"] == "offline-test"


def test_unavailable_and_degraded(model):
    with TestClient(create_app(predictor_factory=Mock(side_effect=FileNotFoundError))) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["status"] == "unavailable"
        assert client.post("/predict/url", json={"url": "python.org"}).status_code == 503
    model.email = None
    model.online = None
    with TestClient(create_app(predictor_factory=lambda _: model)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict/email", json={"text": "hello there"}).status_code == 503
        assert client.post("/predict/url", json={"url": "python.org"}).status_code == 200


def test_rate_limit_expiry_and_health(model):
    now = [0.0]
    app = create_app(predictor_factory=lambda _: model, rate_limit=1, clock=lambda: now[0])
    with TestClient(app) as client:
        assert client.post("/predict/email", json={"text": "hello there"}).status_code == 200
        response = client.post("/predict/email", json={"text": "hello there"})
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"
        assert client.get("/health").status_code == 200
        now[0] = 60
        assert client.post("/predict/email", json={"text": "hello there"}).status_code == 200


def test_size_limit_chunked_and_malformed_json(client):
    response = client.post("/predict/email", content=iter([b"x" * 600000, b"x" * 500000]))
    assert response.status_code == 413
    assert client.post("/predict/email", content=b"{bad").status_code == 422


def test_privacy_and_internal_error(client, model, caplog):
    secret = "PRIVATE_EMAIL_CONTENT_987"
    with caplog.at_level(logging.INFO, logger="phishlens.api"):
        assert client.post("/predict/email", json={"text": secret}).status_code == 200
        invalid = client.post("/predict/email", json={"text": [secret]})
        assert secret not in invalid.text
        model.predict_email = Mock(side_effect=RuntimeError(secret))
        failure = client.post("/predict/email", json={"text": secret})
        assert failure.status_code == 500
        assert secret not in failure.text
    assert secret not in caplog.text
    records = [
        json.loads(record.message) for record in caplog.records if record.name == "phishlens.api"
    ]
    assert any(record.get("status") == 500 for record in records)
